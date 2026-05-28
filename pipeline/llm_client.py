"""Client gọi teacher model qua LM Studio (API OpenAI-compatible).

Mọi call — kể cả lỗi — được ghi đầy đủ (input + output) vào logs/trace.jsonl
qua module trace, và in một dòng tóm tắt ra console. Nhờ đó luôn soi được
model có chạy đúng không.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from config import (
    LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, TEACHER_MODEL,
    LLM_CONCURRENCY, LLM_MAX_RETRIES,
)
from pipeline import trace

# Timeout: Qwen3 thinking can run indefinitely if not capped at the HTTP layer.
# max_tokens limits output but NOT thinking tokens in llama.cpp/LM Studio.
# 120s is ~10x a normal judge call; raises httpx.TimeoutError -> retry logic fires.
_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY,
                 timeout=120.0)

# Qwen3 có chế độ "thinking" — phòng khi output kèm <think>...</think> inline.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text) -> str:
    return _THINK_RE.sub("", text or "").strip()


def chat(messages, temperature=0.9, max_tokens=2048, tag="?") -> str:
    """Gọi teacher một lần. Trả text đã làm sạch. Tự retry khi lỗi.

    tag: nhãn loại call ("expand"/"generate"/"judge") — để lọc logs/trace.jsonl.
    """
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        t0 = time.time()
        try:
            resp = _client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            usage = resp.usage.model_dump() if resp.usage else None
            cleaned = _strip_thinking(content)
            dt = time.time() - t0

            trace.log_call(tag, messages, cleaned, reasoning,
                           choice.finish_reason, usage, dt)
            print(f"    [{tag}] {dt:.1f}s  finish={choice.finish_reason}  "
                  f"reasoning={len(reasoning)}ch  output={len(cleaned)}ch",
                  flush=True)
            if not cleaned.strip():
                print(f"    !! [{tag}] OUTPUT RỖNG — soi logs/trace.jsonl "
                      f"(finish={choice.finish_reason})", flush=True)
            return cleaned
        except Exception as e:                       # pilot: log & retry
            last_err = e
            trace.log_call(tag, messages, "", "", "error", None,
                           time.time() - t0, error=str(e))
            print(f"    !! [{tag}] lỗi LM Studio (thử {attempt + 1}/"
                  f"{LLM_MAX_RETRIES}): {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LM Studio thất bại sau {LLM_MAX_RETRIES} lần: {last_err}")


def chat_many(list_of_messages, temperature=0.9, max_tokens=2048, tag="?"):
    """Gọi song song nhiều prompt. Kết quả giữ đúng thứ tự đầu vào."""
    n = len(list_of_messages)
    print(f"    [chat_many] START n={n} workers={LLM_CONCURRENCY}", flush=True)

    def _call(args):
        idx, m = args
        print(f"    [chat_many] thread-{idx} START", flush=True)
        result = chat(m, temperature, max_tokens, tag)
        print(f"    [chat_many] thread-{idx} END", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        print(f"    [chat_many] pool created, submitting {n} tasks...", flush=True)
        results = list(pool.map(_call, enumerate(list_of_messages)))
        print(f"    [chat_many] map() done — {len(results)} results, waiting for pool shutdown...", flush=True)

    print(f"    [chat_many] pool shutdown complete", flush=True)
    return results


def extract_json(text: str):
    """Rút object JSON đầu tiên ra khỏi text teacher trả về.

    Chịu được các lỗi hay gặp của model:
    - bọc trong ```json ... ```
    - dấu phẩy thừa trước } hoặc ]  (trailing comma)
    - có chữ/object thừa phía sau JSON  (raw_decode bỏ qua phần thừa)
    Ném ValueError kèm trích đoạn nếu vẫn không parse được.
    """
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # bỏ dấu phẩy thừa trước dấu đóng ngoặc
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Không tìm thấy JSON trong: {text[:300]!r}")
    try:
        # raw_decode parse đúng MỘT object, bỏ qua mọi dữ liệu thừa phía sau
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON lỗi ({e}): {text[start:start + 300]!r}")

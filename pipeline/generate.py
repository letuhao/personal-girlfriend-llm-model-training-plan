"""C2 (Cách A) — scenario -> hội thoại multi-turn.

Cách A: một call sinh TRỌN hội thoại (teacher viết cả lượt User lẫn Linh).
Đơn giản, nhanh — đủ cho pilot. Sinh REJECTION_K bản/scenario; việc chọn bản
tốt nhất để ở bước judge (D3).

Nâng cấp về sau (Cách B): tách "user simulator" và "Linh" thành 2 vai luân
phiên để lượt User thật hơn — xem README.
"""
import random

from config import (
    SCENARIOS_FILE, RAW_CONV_FILE, PROMPT_DIR, CATEGORY_CONFIG,
    REJECTION_K, GEN_TEMPERATURE, GEN_MAX_TOKENS,
)
from pipeline import jsonl
from pipeline.llm_client import chat_many, extract_json


def _normalize_turns(turns):
    """Chuẩn hoá list lượt -> trả list sạch, hoặc None nếu không cứu được.

    - Gộp các lượt cùng role liên tiếp (model hay tách 1 lời thành nhiều
      bong bóng) thành một lượt, nối bằng xuống dòng.
    - Bỏ lượt rỗng / role lạ.
    - Hội thoại phải bắt đầu bằng user và kết thúc bằng assistant (lượt user
      cuối không có lời đáp -> vô dụng để train).
    """
    if not isinstance(turns, list):
        return None
    merged = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = t.get("role")
        content = str(t.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    if merged and merged[0]["role"] == "assistant":
        merged = merged[1:]                       # phải bắt đầu bằng user
    if merged and merged[-1]["role"] == "user":
        merged = merged[:-1]                      # phải kết thúc bằng assistant
    return merged if len(merged) >= 2 else None


def _build_prompt(scenario, character_tpl, conv_tpl):
    """Dựng prompt Cách A: sample mood/tone theo category rồi điền vào template."""
    cfg = CATEGORY_CONFIG[scenario["category"]]
    mood = random.choice(cfg["moods"])
    tone = random.choice(cfg["tones"])
    lo, hi = cfg["turns"]
    num_turns = random.choice(range(lo, hi + 1, 2))      # số lượt chẵn

    extra = "- " + cfg["extra"]
    if scenario.get("note"):                             # seed đặc biệt
        extra += f"\n- LƯU Ý RIÊNG: {scenario['note']}"

    character_block = character_tpl.replace("{char_mood}", mood)
    prompt = (conv_tpl
              .replace("{character_block}", character_block)
              .replace("{scenario}", scenario["situation"])
              .replace("{char_mood}", mood)
              .replace("{user_tone}", tone)
              .replace("{num_turns}", str(num_turns))
              .replace("{extra}", extra))
    return prompt, mood, tone


def generate():
    scenarios = jsonl.load(SCENARIOS_FILE)
    character_tpl = (PROMPT_DIR / "linh_character.txt").read_text(encoding="utf-8")
    conv_tpl = (PROMPT_DIR / "conversation_genA.txt").read_text(encoding="utf-8")

    # RESUME: bỏ qua scenario đã có trong file output (nếu lần trước bị ngắt)
    done = jsonl.done_keys(RAW_CONV_FILE, "scenario_id")
    todo = [sc for sc in scenarios if sc["id"] not in done]
    print(f"Resume: {len(done)} đã xong, còn {len(todo)}/{len(scenarios)} "
          f"scenario cần làm", flush=True)

    # ghi từng record NGAY khi sinh xong (append + flush) -> chịu được ngắt
    for sc in todo:
        prompt, mood, tone = _build_prompt(sc, character_tpl, conv_tpl)
        # k bản độc lập cho cùng prompt -> rejection sampling ở bước judge
        batch = [[{"role": "user", "content": prompt}] for _ in range(REJECTION_K)]
        outs = chat_many(batch, temperature=GEN_TEMPERATURE,
                         max_tokens=GEN_MAX_TOKENS, tag="generate")

        candidates = []
        for raw in outs:
            try:
                turns = extract_json(raw).get("turns", [])
            except ValueError:
                continue                          # bản hỏng JSON -> bỏ
            norm = _normalize_turns(turns)
            if norm:
                candidates.append(norm)

        if not candidates:
            print(f"  [bỏ] {sc['id']}: không bản nào hợp lệ", flush=True)
            continue

        jsonl.append(RAW_CONV_FILE, {
            "scenario_id": sc["id"],
            "category": sc["category"],
            "situation": sc["situation"],
            "char_mood": mood,
            "user_tone": tone,
            "candidates": candidates,
        })
        print(f"  [{sc['id']}] {len(candidates)}/{REJECTION_K} bản hợp lệ",
              flush=True)

    total = len(jsonl.load(RAW_CONV_FILE))
    print(f"Tổng {total} record trong {RAW_CONV_FILE}", flush=True)


if __name__ == "__main__":
    generate()

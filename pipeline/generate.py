"""C2 (Cách B) — sinh hội thoại turn-by-turn.

User simulator + Linh generator luân phiên:
- User sim: chỉ thấy situation + tone + history (KHÔNG thấy character Linh)
  → user turns tự nhiên hơn, ít "feed-y" hơn Cách A
- Linh: system = full character card + category extra → gần inference thật

Sinh REJECTION_K conversation song song (mỗi conversation sequential nội bộ);
judge.py sau đó chọn bản tốt nhất (rejection sampling).

    python -m pipeline.generate
"""
import random
from concurrent.futures import ThreadPoolExecutor

from config import (
    SCENARIOS_FILE, RAW_CONV_FILE, PROMPT_DIR, CATEGORY_CONFIG,
    REJECTION_K, GEN_TEMPERATURE, GEN_MAX_TOKENS,
    PILOT, MAX_SCENARIOS, USER_SIM_MAX_TOKENS,
)
from pipeline import jsonl
from pipeline.llm_client import chat

# Mô tả tone inject vào user simulator — giữ ngắn để không leak character Linh
_TONE_DESC: dict[str, str] = {
    "quan tâm":         "Bạn quan tâm đến Linh, hay hỏi thăm nhưng không quá lộ liễu.",
    "cộc lốc vô tâm":  "Bạn đang bận hoặc mải việc — trả lời cộc, đôi khi lạc đề.",
    "nhây trêu":        "Bạn hay trêu chọc, nói hớ vô ý hoặc cố tình để Linh gắt.",
    "tình cảm":         "Bạn muốn gần Linh, nói những điều tình cảm kiểu nam thật.",
    "nói hớ":           "Bạn vô tình nói sai, hớ, hoặc sexist nhẹ — không cố ý.",
    "nhõng nhẽo":       "Bạn đang nhõng nhẽo, muốn được chú ý hoặc chiều chuộng.",
    "xấc khiêu khích":  "Bạn cố tình khiêu khích Linh để xem cô ấy phản ứng thế nào.",
    "rụt rè":           "Bạn rụt rè, không dám nói thẳng, hỏi vòng vo.",
    "hỏi việc":         "Bạn đang nhờ Linh giúp một việc cụ thể — đặt câu hỏi rõ ràng.",
}


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(bắt đầu hội thoại — bạn nhắn trước)"
    lines = []
    for t in history:
        who = "Bạn" if t["role"] == "user" else "Linh"
        lines.append(f"{who}: {t['content']}")
    return "\n".join(lines)


def _clean_output(text: str) -> str:
    """Bỏ prefix role thừa model hay thêm vào đầu output."""
    text = text.strip()
    for prefix in ("Bạn:", "User:", "user:", "bạn:", "Linh:", "linh:",
                   "Tôi:", "tôi:", "Anh:", "anh:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    return text


def _generate_one_conversation(
    scenario: dict,
    linh_system: str,
    user_sim_tpl: str,
    tone: str,
    num_turns: int,
) -> list[dict] | None:
    """Sinh MỘT hội thoại turn-by-turn. Trả list lượt hoặc None nếu thất bại."""
    tone_desc = _TONE_DESC.get(tone, tone)
    history: list[dict] = []

    for _ in range(num_turns // 2):
        # ── Lượt User (user simulator — không thấy character Linh) ───────
        user_prompt = (
            user_sim_tpl
            .replace("{situation}", scenario["situation"])
            .replace("{tone_description}", tone_desc)
            .replace("{history}", _format_history(history))
        )
        # Retry một lần nếu output rỗng — Qwen3 reasoning đôi khi dùng hết
        # token cho thinking mà không còn chỗ output.
        user_content = ""
        for _retry in range(2):
            try:
                user_raw = chat(
                    [{"role": "user", "content": user_prompt}],
                    temperature=GEN_TEMPERATURE,
                    max_tokens=USER_SIM_MAX_TOKENS,
                    tag="usersim",
                )
                user_content = _clean_output(user_raw)
                if user_content:
                    break
            except RuntimeError:
                break
        if not user_content:
            break
        history.append({"role": "user", "content": user_content})

        # ── Lượt Linh (inference-style: system = full character) ──────────
        try:
            linh_raw = chat(
                [{"role": "system", "content": linh_system}] + history,
                temperature=GEN_TEMPERATURE,
                max_tokens=GEN_MAX_TOKENS,
                tag="linh",
            )
        except RuntimeError:
            break
        linh_content = _clean_output(linh_raw)
        if not linh_content:
            break
        history.append({"role": "assistant", "content": linh_content})

    # Phải có ít nhất 1 cặp user/assistant và kết thúc bằng assistant
    if len(history) < 2 or history[-1]["role"] != "assistant":
        return None
    return history


def _sample_pilot(scenarios: list[dict]) -> list[dict]:
    """Pilot mode: lấy MAX_SCENARIOS scenario đầu, phân bổ đều theo category."""
    by_cat: dict[str, list] = {}
    for sc in scenarios:
        by_cat.setdefault(sc["category"], []).append(sc)
    per_cat = max(1, MAX_SCENARIOS // len(by_cat))
    result = []
    for items in by_cat.values():
        result.extend(items[:per_cat])
    return result


def generate() -> None:
    scenarios = jsonl.load(SCENARIOS_FILE)
    if PILOT:
        scenarios = _sample_pilot(scenarios)
        n_cats = len({s["category"] for s in scenarios})
        print(
            f"[PILOT] {len(scenarios)} scenario "
            f"({MAX_SCENARIOS // n_cats} / category)",
            flush=True,
        )

    linh_base = (PROMPT_DIR / "linh_character.txt").read_text(encoding="utf-8")
    user_sim_tpl = (PROMPT_DIR / "user_simulator.txt").read_text(encoding="utf-8")

    # RESUME: bỏ qua scenario đã có trong output
    done = jsonl.done_keys(RAW_CONV_FILE, "scenario_id")
    todo = [sc for sc in scenarios if sc["id"] not in done]
    print(
        f"Resume: {len(done)} đã xong, còn {len(todo)}/{len(scenarios)}",
        flush=True,
    )

    for sc in todo:
        cfg = CATEGORY_CONFIG[sc["category"]]
        mood = random.choice(cfg["moods"])
        lo, hi = cfg["turns"]
        num_turns = random.choice(range(lo, hi + 1, 2))

        # System prompt Linh = base character + category extra + seed note
        linh_system = linh_base.replace("{char_mood}", mood)
        cat_extra_path = PROMPT_DIR / f"linh_{sc['category']}.txt"
        if cat_extra_path.exists():
            linh_system += "\n\n" + cat_extra_path.read_text(encoding="utf-8")
        if sc.get("note"):
            linh_system += f"\n\n## LƯU Ý RIÊNG CHO TÌNH HUỐNG NÀY\n{sc['note']}"

        tones = cfg["tones"]

        # Chạy REJECTION_K conversation song song, mỗi conversation sequential
        sampled_tones = [random.choice(tones) for _ in range(REJECTION_K)]

        def _worker(tone: str) -> list[dict] | None:
            return _generate_one_conversation(
                sc, linh_system, user_sim_tpl, tone, num_turns
            )

        with ThreadPoolExecutor(max_workers=REJECTION_K) as pool:
            results = list(pool.map(_worker, sampled_tones))

        candidates = [h for h in results if h is not None]

        if not candidates:
            print(f"  [bỏ] {sc['id']}: không conversation nào hợp lệ", flush=True)
            continue

        jsonl.append(RAW_CONV_FILE, {
            "scenario_id": sc["id"],
            "category":    sc["category"],
            "situation":   sc["situation"],
            "char_mood":   mood,
            "user_tone":   sampled_tones[0],
            "candidates":  candidates,
        })
        print(
            f"  [{sc['id']}] {len(candidates)}/{REJECTION_K} conversation hợp lệ",
            flush=True,
        )

    total = len(jsonl.load(RAW_CONV_FILE))
    print(f"Tổng {total} record trong {RAW_CONV_FILE}", flush=True)


if __name__ == "__main__":
    generate()

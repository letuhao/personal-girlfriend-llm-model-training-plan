"""Chẩn đoán teacher TRƯỚC khi chạy cả pipeline.

Gọi teacher đúng MỘT lần cho mỗi loại prompt (expand / generate / judge),
in rõ input + output + kết quả parse JSON ra console. Mọi call cũng được
ghi đầy đủ vào logs/trace.jsonl.

    python -u diagnose.py

Dùng để xác nhận: model có chạy không, output có parse được JSON không,
chất lượng tiếng Việt / nhập vai có ổn không — trước khi tốn thời gian
chạy toàn bộ pipeline.
"""
import json
import time

import yaml

from config import (
    SEEDS_FILE, PROMPT_DIR,
    EXPAND_TEMPERATURE, EXPAND_MAX_TOKENS,
    GEN_TEMPERATURE, GEN_MAX_TOKENS,
    JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS,
)
from pipeline.generate import _build_prompt
from pipeline.llm_client import chat, extract_json

SEP = "=" * 74


def run(title, prompt, tag, temperature, max_tokens):
    """Gọi teacher một lần, in input + output + parse, trả object JSON (hoặc None)."""
    print(f"\n{SEP}\n  {title}\n{SEP}")
    print("------ INPUT (prompt gửi cho teacher) ------")
    print(prompt[:1800] + ("\n...[cắt bớt]" if len(prompt) > 1800 else ""))

    print(f"\n>>> đang gọi teacher (tag={tag})...")
    t0 = time.time()
    raw = chat([{"role": "user", "content": prompt}],
               temperature=temperature, max_tokens=max_tokens, tag=tag)
    print(f"<<< xong sau {time.time() - t0:.1f}s")

    print("\n------ OUTPUT (raw teacher trả về) ------")
    print(raw[:3000] + ("\n...[cắt bớt]" if len(raw) > 3000 else "")
          if raw else "(RỖNG)")

    print("\n------ PARSE JSON ------")
    try:
        obj = extract_json(raw)
        print("OK. Keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))
        return obj
    except Exception as e:
        print(f"THẤT BẠI: {e}")
        return None


def main():
    print("CHẨN ĐOÁN TEACHER — mỗi loại prompt gọi 1 lần.\n"
          "Chi tiết đầy đủ ghi ở logs/trace.jsonl")

    # ── TEST 1/3 — EXPAND ──────────────────────────────────────────────
    seeds = yaml.safe_load(open(SEEDS_FILE, encoding="utf-8"))
    daily = [s for s in seeds if s["category"] == "daily"]
    examples = "\n".join(f"- {s['situation']}" for s in daily)
    expand_tpl = (PROMPT_DIR / "expand.txt").read_text(encoding="utf-8")
    p1 = (expand_tpl
          .replace("{category}", "daily")
          .replace("{examples}", examples)
          .replace("{n}", "3"))
    obj1 = run("TEST 1/3 — EXPAND (sinh scenario)", p1, "expand",
               EXPAND_TEMPERATURE, EXPAND_MAX_TOKENS)
    if obj1 and obj1.get("scenarios"):
        print("Scenario sinh được:")
        for s in obj1["scenarios"]:
            print("  -", s)

    # ── TEST 2/3 — GENERATE ────────────────────────────────────────────
    character_tpl = (PROMPT_DIR / "linh_character.txt").read_text(encoding="utf-8")
    conv_tpl = (PROMPT_DIR / "conversation_genA.txt").read_text(encoding="utf-8")
    scenario = {
        "category": "daily", "note": "",
        "situation": "User nhắn lúc 2h sáng hỏi Linh còn thức không",
    }
    p2, mood, tone = _build_prompt(scenario, character_tpl, conv_tpl)
    print(f"\n(generate: char_mood={mood}, user_tone={tone})")
    obj2 = run("TEST 2/3 — GENERATE (sinh hội thoại)", p2, "generate",
               GEN_TEMPERATURE, GEN_MAX_TOKENS)
    if obj2 and obj2.get("turns"):
        print("Hội thoại sinh được:")
        for t in obj2["turns"]:
            who = "User" if t.get("role") == "user" else "Linh"
            print(f"  {who}: {t.get('content')}")

    # ── TEST 3/3 — JUDGE ───────────────────────────────────────────────
    turns = (obj2 or {}).get("turns") or [
        {"role": "user", "content": "còn thức k"},
        {"role": "assistant", "content": "k. đang nói chuyện với mày bằng niềm tin à 🙂"},
    ]
    convo_text = "\n".join(
        ("User" if t.get("role") == "user" else "Linh") + ": " + str(t.get("content"))
        for t in turns
    )
    judge_tpl = (PROMPT_DIR / "judge.txt").read_text(encoding="utf-8")
    p3 = (judge_tpl
          .replace("{category}", "daily")
          .replace("{conversation}", convo_text))
    obj3 = run("TEST 3/3 — JUDGE (chấm điểm)", p3, "judge",
               JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
    if obj3:
        print("Điểm judge:", json.dumps(obj3, ensure_ascii=False))

    # ── Kết luận ───────────────────────────────────────────────────────
    print(f"\n{SEP}\n  KẾT QUẢ")
    print(f"  EXPAND   : {'OK' if obj1 else 'LỖI'}")
    print(f"  GENERATE : {'OK' if obj2 else 'LỖI'}")
    print(f"  JUDGE    : {'OK' if obj3 else 'LỖI'}")
    print(f"  Trace đầy đủ: logs/trace.jsonl\n{SEP}")


if __name__ == "__main__":
    main()

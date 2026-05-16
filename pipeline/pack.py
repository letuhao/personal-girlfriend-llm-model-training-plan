"""E — Đóng gói: dataset.jsonl -> ChatML messages, split train/val.

Mỗi hội thoại thành một training example dạng OpenAI messages:
  [system = full operational prompt][user][assistant][user][assistant]...
System prompt giữ NGUYÊN bản đầy đủ (quyết định 'Full' ở phần thiết kế),
điền đúng tâm trạng đã dùng lúc sinh hội thoại.

Split 95/5 stratified theo category (val phủ đủ cả 5 loại).
"""
import json
import random

from config import (
    DATASET_FILE, TRAIN_FILE, VAL_FILE, PROMPT_DIR, VAL_RATIO, SPLIT_SEED,
)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pack():
    records = _load_jsonl(DATASET_FILE)
    character_tpl = (PROMPT_DIR / "linh_character.txt").read_text(encoding="utf-8")

    examples = []
    for r in records:
        # full system prompt, điền đúng mood đã dùng khi sinh hội thoại
        system = character_tpl.replace("{char_mood}", r["char_mood"])
        messages = [{"role": "system", "content": system}] + r["conversation"]
        examples.append({
            "messages": messages,
            "meta": {
                "category": r["category"],
                "char_mood": r["char_mood"],
                "user_tone": r["user_tone"],
                "scenario_id": r["scenario_id"],
            },
        })

    # split 95/5 stratified theo category
    rng = random.Random(SPLIT_SEED)
    by_cat = {}
    for ex in examples:
        by_cat.setdefault(ex["meta"]["category"], []).append(ex)

    train, val = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_val = round(len(items) * VAL_RATIO) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    _write_jsonl(TRAIN_FILE, train)
    _write_jsonl(VAL_FILE, val)
    print(f"train={len(train)}  val={len(val)}")
    print(f"  -> {TRAIN_FILE}")
    print(f"  -> {VAL_FILE}")


if __name__ == "__main__":
    pack()

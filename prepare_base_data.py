"""
Tải và convert 3 Vietnamese instruction datasets sang ChatML format.
Output: data/base_instructions.jsonl — dùng để mix với character data khi train.

Usage:
    python prepare_base_data.py
    python prepare_base_data.py --dry-run   # chỉ preview 5 examples mỗi dataset

Datasets:
    bkai-foundation-models/vi-alpaca           → 1500 samples (native VI, GPT-4)
    5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca   →  500 samples (multi-turn)
    nguyenphuttien/vietnamese_ultrachat_200k   →  500 samples (topic diversity)
"""
import argparse
import json
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
OUT_FILE = Path(__file__).parent / "data" / "base_instructions.jsonl"

NEUTRAL_SYSTEM = "Bạn là một trợ lý hữu ích, trả lời bằng tiếng Việt."

PLAN = [
    {
        "hf_path": "bkai-foundation-models/vi-alpaca",
        "split": "train",
        "n": 1500,
        "format": "alpaca",     # instruction / input / output
    },
    {
        "hf_path": "5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca",
        "split": "train",
        "n": 700,
        "format": "sharegpt",   # conversations: [{from: human/gpt, value: ...}]
    },
    # UltraChat fallbacks — thử lần lượt cho đến khi load được
    # Uncomment nếu tìm được path đúng:
    # {"hf_path": "nguyenphuthien/vietnamese_ultrachat_200k", "split": "train_sft", "n": 300, "format": "messages"},
]

# Phrases AI identity trong base data → lọc bỏ conversation đó.
# Tránh dạy model "tôi là AI" qua data phụ.
_AI_LEAK_PATTERNS = [
    "là một mô hình ngôn ngữ ai",
    "với tư cách là một mô hình",
    "tôi là một ai",
    "tôi là một trợ lý ai",
    "tôi là chatbot",
    "tôi không có cảm xúc",
    "tôi không có khả năng cảm nhận",
    "tôi không thể cảm xúc",
    "tôi không có ý thức",
    "as an ai language model",
    "as an ai assistant",
]


# ── Converters ───────────────────────────────────────────────────────────────

def alpaca_to_chatml(row: dict) -> dict | None:
    """bkai vi-alpaca: instruction + input + output → ChatML."""
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()

    if not instruction or not output:
        return None

    user_text = instruction
    if inp:
        user_text = f"{instruction}\n\n{inp}"

    # Lọc output rõ ràng là garbage hoặc quá ngắn
    if len(output) < 10:
        return None

    return {
        "messages": [
            {"role": "system",    "content": NEUTRAL_SYSTEM},
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": output},
        ]
    }


def sharegpt_to_chatml(row: dict) -> dict | None:
    """5CD-AI multi-turn: conversations list."""
    convs = row.get("conversations") or []
    if not convs:
        return None

    messages = [{"role": "system", "content": NEUTRAL_SYSTEM}]
    role_map = {"human": "user", "gpt": "assistant"}

    for turn in convs:
        role = role_map.get(turn.get("from", ""), "")
        value = (turn.get("value") or "").strip()
        if not role or not value:
            return None
        messages.append({"role": role, "content": value})

    # Phải có ít nhất 1 user + 1 assistant
    roles = [m["role"] for m in messages if m["role"] != "system"]
    if "user" not in roles or "assistant" not in roles:
        return None

    return {"messages": messages}


def messages_to_chatml(row: dict) -> dict | None:
    """UltraChat: messages list với role: user/assistant."""
    msgs = row.get("messages") or []
    if not msgs:
        return None

    messages = [{"role": "system", "content": NEUTRAL_SYSTEM}]
    for m in msgs:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})

    roles = [m["role"] for m in messages if m["role"] != "system"]
    if "user" not in roles or "assistant" not in roles:
        return None

    return {"messages": messages}


CONVERTERS = {
    "alpaca":   alpaca_to_chatml,
    "sharegpt": sharegpt_to_chatml,
    "messages": messages_to_chatml,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def _has_ai_leak(example: dict) -> bool:
    """True nếu có bất kỳ turn nào chứa AI identity phrase."""
    for msg in example.get("messages", []):
        content_lower = msg.get("content", "").lower()
        for pat in _AI_LEAK_PATTERNS:
            if pat in content_lower:
                return True
    return False


def load_and_sample(cfg: dict, rng: random.Random, dry_run: bool) -> list[dict]:
    from datasets import load_dataset

    print(f"\n  Đang tải {cfg['hf_path']} (split={cfg['split']}) ...")
    try:
        ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    except Exception as e:
        print(f"  [WARN] Không tải được: {e}")
        return []

    print(f"  Raw size: {len(ds):,} rows")

    convert = CONVERTERS[cfg["format"]]
    converted = []
    filtered_ai = 0
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for i in indices:
        row = ds[i]
        result = convert(row)
        if not result:
            continue
        if _has_ai_leak(result):
            filtered_ai += 1
            continue
        converted.append(result)
        if len(converted) >= cfg["n"]:
            break

    print(f"  Converted: {len(converted):,} (target {cfg['n']}, AI-leak filtered: {filtered_ai})")

    if dry_run:
        print("\n  --- Preview (3 examples) ---")
        for ex in converted[:3]:
            for m in ex["messages"]:
                role = m["role"].upper()
                snippet = m["content"][:120].replace("\n", " ")
                print(f"    [{role}] {snippet}...")
            print()

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview 5 examples mỗi dataset, không ghi file")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    all_examples: list[dict] = []

    for cfg in PLAN:
        examples = load_and_sample(cfg, rng, args.dry_run)
        all_examples.extend(examples)

    print(f"\nTổng: {len(all_examples):,} examples")

    if args.dry_run:
        print("Dry-run mode — không ghi file.")
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(all_examples)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Đã ghi: {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.0f} KB)")
    print("\nBước tiếp theo:")
    print("  python pack_mixed.py   # mix base + character data → train/val")


if __name__ == "__main__":
    main()

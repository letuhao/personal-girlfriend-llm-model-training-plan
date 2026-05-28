"""Đóng gói DPO pairs -> format training cho TRL DPOTrainer.

Đọc từ: data/dpo_pairs.jsonl  (sinh bởi judge.py)
Ghi ra: data/dpo_train.jsonl, data/dpo_val.jsonl

Format mỗi example:
  {
    "prompt":   [{"role":"system",...}, {"role":"user",...}, ...],  # lịch sử đến trước lượt cuối
    "chosen":   "<nội dung lượt assistant tốt nhất>",
    "rejected": "<nội dung lượt assistant kém nhất>",
  }

Chạy:
    python -m pipeline.pack_dpo
    python -m pipeline.pack_dpo --stats   # chỉ in thống kê
"""
import argparse
import json
import random
from pathlib import Path

from config import DATA_DIR, PROMPT_DIR, SPLIT_SEED

DPO_FILE  = DATA_DIR / "dpo_pairs.jsonl"
DPO_TRAIN = DATA_DIR / "dpo_train.jsonl"
DPO_VAL   = DATA_DIR / "dpo_val.jsonl"

VAL_RATIO = 0.1


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_prompt_messages(turns: list[dict], system: str) -> list[dict]:
    """Tất cả lượt trừ lượt assistant cuối → đây là 'prompt' cho DPO."""
    msgs = [{"role": "system", "content": system}]
    # Bỏ lượt assistant cuối
    prefix = turns[:-1] if turns and turns[-1]["role"] == "assistant" else turns
    msgs.extend(prefix)
    return msgs


def pack_dpo(stats_only: bool = False) -> None:
    linh_base = (PROMPT_DIR / "linh_character.txt").read_text(encoding="utf-8")
    pairs = _load_jsonl(DPO_FILE)

    if not pairs:
        print(f"Không tìm thấy {DPO_FILE} hoặc file rỗng.")
        print("Chạy pipeline.generate + pipeline.judge trước.")
        return

    print(f"DPO pairs raw: {len(pairs)}")

    examples: list[dict] = []
    skipped = 0
    for p in pairs:
        chosen_conv  = p.get("chosen", [])
        rejected_conv = p.get("rejected", [])

        # Cần ít nhất kết thúc bằng assistant ở cả hai
        if (not chosen_conv  or chosen_conv[-1]["role"]  != "assistant" or
                not rejected_conv or rejected_conv[-1]["role"] != "assistant"):
            skipped += 1
            continue

        chosen_resp   = chosen_conv[-1]["content"]
        rejected_resp = rejected_conv[-1]["content"]

        if not chosen_resp or not rejected_resp:
            skipped += 1
            continue

        system = linh_base.replace("{char_mood}", p.get("char_mood", "vui vẻ"))
        prompt_msgs = _build_prompt_messages(chosen_conv, system)

        examples.append({
            "prompt":   prompt_msgs,
            "chosen":   chosen_resp,
            "rejected": rejected_resp,
            "meta": {
                "scenario_id":    p["scenario_id"],
                "category":       p["category"],
                "chosen_score":   p.get("chosen_score"),
                "rejected_score": p.get("rejected_score"),
                "gap":            p.get("gap"),
            },
        })

    print(f"DPO examples hợp lệ: {len(examples)}  (skipped: {skipped})")

    # Thống kê gap phân bổ
    gaps = [e["meta"]["gap"] for e in examples if e["meta"]["gap"] is not None]
    if gaps:
        print(f"  gap avg={sum(gaps)/len(gaps):.1f}  "
              f"min={min(gaps)}  max={max(gaps)}")

    if stats_only:
        print("Stats-only mode — không ghi file.")
        return

    # Split 90/10 stratified theo category
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(examples)
    n_val = max(1, round(len(examples) * VAL_RATIO))

    _write_jsonl(DPO_TRAIN, examples[n_val:])
    _write_jsonl(DPO_VAL,   examples[:n_val])

    print(f"\nĐã ghi:")
    print(f"  {DPO_TRAIN}  ({len(examples) - n_val} examples)")
    print(f"  {DPO_VAL}    ({n_val} examples)")
    print("\nBước tiếp theo — DPO training:")
    print("  python -m train.train_dpo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    pack_dpo(stats_only=args.stats)

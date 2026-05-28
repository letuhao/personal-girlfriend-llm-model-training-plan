"""
Mix base_instructions.jsonl + dataset.jsonl (character) → train_mixed.jsonl + val_mixed.jsonl.

Chiến lược mix:
- Val set: chỉ lấy từ character data (để eval thuần character quality)
- Train set: character data (trừ val) + base instruction data (trộn đều)

Usage:
    python pack_mixed.py
    python pack_mixed.py --base-ratio 0.5   # điều chỉnh tỉ lệ base (default 0.5)
    python pack_mixed.py --stats            # chỉ in thống kê, không ghi file

Đọc từ:
    data/dataset.jsonl          (character data đã judge/filter)
    data/base_instructions.jsonl (từ prepare_base_data.py)
Ghi ra:
    data/train_mixed.jsonl
    data/val_mixed.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from config import DATA_DIR, VAL_RATIO, SPLIT_SEED, TRAIN_FILE, VAL_FILE

BASE_FILE    = DATA_DIR / "base_instructions.jsonl"
# Dùng train.jsonl / val.jsonl đã pack (có messages + system prompt),
# KHÔNG đọc dataset.jsonl (field là "conversation", chưa có system prompt).
TRAIN_MIXED  = DATA_DIR / "train_mixed.jsonl"
VAL_MIXED    = DATA_DIR / "val_mixed.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tag(rows: list[dict], source: str) -> list[dict]:
    """Thêm metadata source — dùng để debug, không ảnh hưởng training."""
    return [{**r, "_source": source} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ratio", type=float, default=0.5,
                        help="Tỉ lệ base data trong train set (default: 0.5)")
    parser.add_argument("--stats", action="store_true",
                        help="Chỉ in thống kê, không ghi file")
    args = parser.parse_args()

    rng = random.Random(SPLIT_SEED)

    # ── Load ────────────────────────────────────────────────────────────────
    # train.jsonl / val.jsonl đã có messages + system prompt (từ pack.py)
    char_train = load_jsonl(TRAIN_FILE)
    val        = load_jsonl(VAL_FILE)
    base_all   = load_jsonl(BASE_FILE) if BASE_FILE.exists() else []

    print(f"Character train    : {len(char_train):>6,}")
    print(f"Character val      : {len(val):>6,}")
    print(f"Base examples      : {len(base_all):>6,}")

    # ── Train mix ────────────────────────────────────────────────────────────
    # Tính số base cần lấy để đạt base_ratio
    #   base_ratio = n_base / (n_char_train + n_base)
    #   n_base = n_char_train * base_ratio / (1 - base_ratio)
    n_base_target = int(len(char_train) * args.base_ratio / (1 - args.base_ratio))
    n_base_actual = min(n_base_target, len(base_all))

    rng.shuffle(base_all)
    base_sample = base_all[:n_base_actual]

    train = tag(char_train, "character") + tag(base_sample, "base")
    rng.shuffle(train)

    # Xóa tag trước khi ghi (không muốn lọt vào training)
    train_clean = [{k: v for k, v in r.items() if k != "_source"} for r in train]
    val_clean   = val  # val không tag

    # ── Stats ────────────────────────────────────────────────────────────────
    base_pct = n_base_actual / max(len(train), 1) * 100
    print(f"\nTrain set breakdown:")
    print(f"  character : {len(char_train):>6,}  ({100 - base_pct:.1f}%)")
    print(f"  base      : {n_base_actual:>6,}  ({base_pct:.1f}%)")
    print(f"  TOTAL     : {len(train_clean):>6,}")
    print(f"\nVal set (character only): {len(val_clean):,} — dùng val.jsonl từ pack.py")

    if args.base_ratio != 0.5:
        print(f"\n[INFO] base-ratio={args.base_ratio} — "
              f"khuyến nghị 0.4–0.5 để không mất character flavor")

    if args.stats:
        print("\nStats-only mode — không ghi file.")
        return

    # ── Save ─────────────────────────────────────────────────────────────────
    save_jsonl(TRAIN_MIXED, train_clean)
    save_jsonl(VAL_MIXED,   val_clean)

    print(f"\nĐã ghi:")
    print(f"  {TRAIN_MIXED}  ({TRAIN_MIXED.stat().st_size / 1024:.0f} KB)")
    print(f"  {VAL_MIXED}    ({VAL_MIXED.stat().st_size / 1024:.0f} KB)")
    print("\nBước tiếp theo — train với file mới:")
    print("  Sửa train/train_qlora.py: TRAIN_FILE → data/train_mixed.jsonl")
    print("  (hoặc pass --train-file argument)")


if __name__ == "__main__":
    main()

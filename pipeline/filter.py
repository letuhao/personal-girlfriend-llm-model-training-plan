"""D — Lọc 3 lớp: D1 luật + D2 dedup + cổng chất lượng cuối.

Chạy rẻ -> đắt: luật trước (loại lỗi hiển nhiên), rồi cổng điểm judge, cuối
cùng dedup cấp hội thoại. Bản thắng rejection sampling vẫn có thể bị loại ở
đây nếu nó không vượt ngưỡng tuyệt đối.
"""
import json

from config import (
    JUDGED_FILE, DATASET_FILE, MIN_LINH_CHARS, MAX_LINH_CHARS,
    JUDGE_MIN_SCORE, SLOP_PHRASES, REFUSAL_PATTERNS, DEDUP_THRESHOLD,
)
from pipeline.dedup import dedup_texts

CORE_AXES = ["nhap_vai", "giong_nguoi", "mach_lac", "luot_user"]
OPTIONAL_AXES = ["dung", "nhip"]


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _linh_turns(conv):
    return [t["content"] for t in conv if t["role"] == "assistant"]


def _passes_rules(rec):
    """D1 — luật rẻ tiền. Trả (ok, lý_do_loại)."""
    linh = _linh_turns(rec["conversation"])
    if not linh:
        return False, "không có lượt Linh"
    for c in linh:
        low = c.lower()
        if len(c.strip()) < MIN_LINH_CHARS:
            return False, "lượt Linh rỗng"
        if len(c) > MAX_LINH_CHARS:
            return False, "lượt Linh quá dài (nghi slop)"
        if any(p in low for p in SLOP_PHRASES):
            return False, "chứa cụm slop"
        if any(p in low for p in REFUSAL_PATTERNS):
            return False, "chứa dấu hiệu từ chối / phá vai"
        if "```" in c:
            return False, "có khối code/markdown trong tin nhắn"
    return True, ""


def _passes_gate(rec):
    """Cổng chất lượng cuối — dựa trên điểm judge. Trả (ok, lý_do_loại)."""
    sc = rec["score"]
    if sc.get("loi_nghiem_trong"):
        return False, "judge gắn cờ lỗi nghiêm trọng"
    for a in CORE_AXES:
        if int(sc.get(a) or 0) < JUDGE_MIN_SCORE:
            return False, f"trục {a} dưới ngưỡng"
    for a in OPTIONAL_AXES:
        if sc.get(a) is not None and int(sc[a]) < JUDGE_MIN_SCORE:
            return False, f"trục {a} dưới ngưỡng"
    return True, ""


def run_filter():
    records = _load_jsonl(JUDGED_FILE)
    kept, dropped = [], []

    for rec in records:
        ok, why = _passes_rules(rec)            # D1
        if not ok:
            dropped.append((rec["scenario_id"], "D1: " + why))
            continue
        ok, why = _passes_gate(rec)             # cổng cuối
        if not ok:
            dropped.append((rec["scenario_id"], "gate: " + why))
            continue
        kept.append(rec)

    # D2 — dedup cấp hội thoại (khoá = toàn bộ text các lượt ghép lại)
    before = len(kept)
    kept = dedup_texts(
        kept,
        key=lambda r: " ".join(t["content"] for t in r["conversation"]),
        threshold=DEDUP_THRESHOLD,
    )
    dedup_removed = before - len(kept)

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nGiữ {len(kept)} / {len(records)} hội thoại -> {DATASET_FILE}")
    print(f"  loại bởi luật/cổng: {len(dropped)}   loại bởi dedup: {dedup_removed}")
    for sid, why in dropped[:40]:
        print(f"  - {sid}: {why}")


if __name__ == "__main__":
    run_filter()

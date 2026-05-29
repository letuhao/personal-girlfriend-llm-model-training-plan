"""D — Lọc 3 lớp: D1 luật + D2 dedup + cổng chất lượng cuối.

Chạy rẻ -> đắt: luật trước (loại lỗi hiển nhiên), rồi cổng điểm judge, cuối
cùng dedup cấp hội thoại. Bản thắng rejection sampling vẫn có thể bị loại ở
đây nếu nó không vượt ngưỡng tuyệt đối.
"""
import json
import re
from collections import Counter

from config import (
    JUDGED_FILE, DATASET_FILE, MIN_LINH_CHARS, MAX_LINH_CHARS,
    JUDGE_MIN_SCORE, MIN_AVG_SCORE, USEFUL_MAX_AVG_CHARS,
    SLOP_PHRASES, REFUSAL_PATTERNS, DEDUP_THRESHOLD, ALLOWED_EMOJI,
)
from pipeline.dedup import dedup_texts

_TAO_MAY_RE = re.compile(r"\b(tao|mày)\b", re.IGNORECASE)
_EM_ANH_RE  = re.compile(r"\b(em|anh)\b",  re.IGNORECASE)
# Emoji bất kỳ — dùng để đếm tần suất
_EMOJI_RE   = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F]"
)

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
        v = sc.get(a)
        if v is None or int(v) < JUDGE_MIN_SCORE:
            return False, f"trục {a} dưới ngưỡng"
    for a in OPTIONAL_AXES:
        v = sc.get(a)
        if v is not None and int(v) < JUDGE_MIN_SCORE:
            return False, f"trục {a} dưới ngưỡng"
    # Ngưỡng avg tổng — loại example trung bình
    all_vals = [v for v in sc.values() if isinstance(v, (int, float))]
    if all_vals:
        avg = sum(all_vals) / len(all_vals)
        if avg < MIN_AVG_SCORE:
            return False, f"avg score {avg:.2f} < {MIN_AVG_SCORE}"
    return True, ""


def _passes_xungho(rec) -> tuple[bool, str]:
    """Detect tao/mày + em/anh trong cùng 1 lượt Linh ở category không phải conflict."""
    if rec["category"] in ("conflict", "edge", "persona"):
        return True, ""
    for t in rec["conversation"]:
        if t["role"] != "assistant":
            continue
        c = t["content"]
        if _TAO_MAY_RE.search(c) and _EM_ANH_RE.search(c):
            return False, "xưng hô lộn xộn trong lượt Linh"
    return True, ""


def _passes_repetition(rec) -> tuple[bool, str]:
    """Detect mode collapse: Linh lặp lại nguyên văn lượt của chính mình."""
    linh_turns = [t["content"].strip() for t in rec["conversation"]
                  if t["role"] == "assistant"]
    _FP = 30
    seen_fps: list[str] = []
    for turn in linh_turns:
        fp = turn[:_FP].lower()
        if len(fp) >= 20 and fp in seen_fps:
            return False, "Linh lặp nguyên văn (mode collapse)"
        seen_fps.append(fp)
    return True, ""


def _passes_emoji_spam(rec) -> tuple[bool, str]:
    """Detect khi cùng emoji xuất hiện ở ≥ 3 lượt Linh — trở thành tic."""
    linh_turns = [t["content"] for t in rec["conversation"]
                  if t["role"] == "assistant"]
    if len(linh_turns) < 3:
        return True, ""
    # Đếm số lượt chứa từng emoji
    per_emoji: Counter = Counter()
    for turn in linh_turns:
        unique_in_turn = set(_EMOJI_RE.findall(turn))
        per_emoji.update(unique_in_turn)
    for emoji, cnt in per_emoji.items():
        if cnt >= 3:
            return False, f"emoji tic: {emoji!r} trong {cnt}/{len(linh_turns)} lượt"
    return True, ""


def _passes_forbidden_emoji(rec) -> tuple[bool, str]:
    """Detect lượt Linh dùng emoji ngoài ALLOWED_EMOJI (🙂 💀 🤡)."""
    for t in rec["conversation"]:
        if t["role"] != "assistant":
            continue
        for emoji in _EMOJI_RE.findall(t["content"]):
            if emoji not in ALLOWED_EMOJI:
                return False, f"emoji bị cấm: {emoji!r}"
    return True, ""


def _passes_useful_brevity(rec) -> tuple[bool, str]:
    """Useful: avg độ dài lượt Linh > USEFUL_MAX_AVG_CHARS -> over-explain."""
    if rec["category"] != "useful":
        return True, ""
    linh = [t["content"] for t in rec["conversation"] if t["role"] == "assistant"]
    if not linh:
        return True, ""
    avg = sum(len(t) for t in linh) / len(linh)
    if avg > USEFUL_MAX_AVG_CHARS:
        return False, f"useful Linh quá dài (avg {avg:.0f} chars)"
    return True, ""


def run_filter():
    records = _load_jsonl(JUDGED_FILE)
    kept, dropped = [], []

    for rec in records:
        ok, why = _passes_rules(rec)            # D1 luật rẻ tiền
        if not ok:
            dropped.append((rec["scenario_id"], "D1: " + why))
            continue
        ok, why = _passes_repetition(rec)       # mode collapse
        if not ok:
            dropped.append((rec["scenario_id"], "repeat: " + why))
            continue
        ok, why = _passes_emoji_spam(rec)       # emoji tic
        if not ok:
            dropped.append((rec["scenario_id"], "emoji: " + why))
            continue
        ok, why = _passes_forbidden_emoji(rec)  # emoji bị cấm theo LUẬT CỨNG
        if not ok:
            dropped.append((rec["scenario_id"], "emoji: " + why))
            continue
        ok, why = _passes_xungho(rec)           # xưng hô lộn xộn
        if not ok:
            dropped.append((rec["scenario_id"], "xungho: " + why))
            continue
        ok, why = _passes_useful_brevity(rec)   # useful over-explain
        if not ok:
            dropped.append((rec["scenario_id"], "brevity: " + why))
            continue
        ok, why = _passes_gate(rec)             # cổng điểm judge
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

"""Tiện ích đọc/ghi JSONL + hỗ trợ resume cho pipeline chạy dài.

Cơ chế resume: mỗi stage dài (generate, judge) ghi từng record xuống file
output NGAY khi xong (append + flush). Khi chạy lại, đọc file output để biết
record nào đã xong rồi bỏ qua. Nhờ đó ngắt giữa chừng -> chạy lại là tiếp tục.
"""
import json


def load(path):
    """Đọc file JSONL -> list dict. File không tồn tại -> list rỗng.

    Bỏ qua dòng hỏng (vd dòng cuối bị cắt dở khi tiến trình bị ngắt) — nhờ vậy
    resume vẫn an toàn dù file output bị đứt giữa chừng.
    """
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def done_keys(path, key):
    """Tập giá trị `key` đã có trong file output — để resume bỏ qua."""
    return {r[key] for r in load(path) if isinstance(r, dict) and key in r}


def append(path, record):
    """Ghi (append) một record xuống JSONL, flush ngay để an toàn khi bị ngắt."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_all(path, records):
    """Ghi đè toàn bộ file JSONL (dùng cho stage ngắn, không cần resume)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

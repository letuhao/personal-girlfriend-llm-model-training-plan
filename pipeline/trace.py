"""Ghi log mọi call tới teacher vào logs/trace.jsonl — để soi model có chạy đúng.

Mỗi dòng JSON là một lần gọi: thời điểm, loại call (tag), thời gian chạy,
toàn bộ input messages, output đã làm sạch, độ dài reasoning, usage, và lỗi
(nếu có). Mở file này khi cần điều tra vì sao một sample hỏng.
"""
import json
import threading
from datetime import datetime

from config import ROOT

LOG_DIR = ROOT / "logs"
TRACE_FILE = LOG_DIR / "trace.jsonl"

# chat_many gọi song song nhiều thread -> cần khoá khi ghi file.
_lock = threading.Lock()


def log_call(tag, messages, output, reasoning, finish_reason, usage,
             duration_s, error=None):
    """Ghi một lần gọi teacher xuống logs/trace.jsonl (append, flush ngay)."""
    LOG_DIR.mkdir(exist_ok=True)
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "duration_s": round(duration_s, 1),
        "finish_reason": finish_reason,
        "error": error,
        "output_len": len(output or ""),
        "reasoning_len": len(reasoning or ""),
        "usage": usage,
        "input": messages,
        "output": output,
    }
    line = json.dumps(rec, ensure_ascii=False)
    with _lock:
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

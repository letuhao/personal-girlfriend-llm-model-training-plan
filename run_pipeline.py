"""Chạy full pipeline: generate -> judge -> filter -> pack -> pack_dpo.

Mỗi bước chạy xong mới chạy bước tiếp — resume-safe ở generate và judge.
Log tổng kết mỗi bước ra console.

    python run_pipeline.py
"""
import subprocess
import sys
import time
from datetime import datetime

STEPS = [
    ("generate", [sys.executable, "-m", "pipeline.generate"]),
    ("judge",    [sys.executable, "-m", "pipeline.judge"]),
    ("filter",   [sys.executable, "-m", "pipeline.filter"]),
    ("pack",     [sys.executable, "-m", "pipeline.pack"]),
    ("pack_dpo", [sys.executable, "-m", "pipeline.pack_dpo"]),
]


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(name, cmd):
    print(f"\n{'='*60}", flush=True)
    print(f"[{ts()}] BẮT ĐẦU: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    dt = time.time() - t0
    h, m = divmod(int(dt), 3600)
    m, s = divmod(m, 60)
    status = "OK" if result.returncode == 0 else f"LỖI (code={result.returncode})"
    print(f"\n[{ts()}] {name}: {status}  ({h:02d}h{m:02d}m{s:02d}s)", flush=True)
    return result.returncode == 0


if __name__ == "__main__":
    print(f"[{ts()}] Pipeline full bắt đầu — {len(STEPS)} bước", flush=True)
    for name, cmd in STEPS:
        ok = run_step(name, cmd)
        if not ok:
            print(f"\n!! Dừng tại bước '{name}' — kiểm tra lỗi trên.", flush=True)
            sys.exit(1)
    print(f"\n[{ts()}] XONG toàn bộ pipeline.", flush=True)

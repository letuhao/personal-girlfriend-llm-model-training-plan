"""Chạy toàn bộ pipeline thu thập dữ liệu theo thứ tự.

    python run_pilot.py

Đọc công tắc PILOT trong config.py. Pilot (mặc định) chạy nhỏ để kiểm tra
pipeline thông suốt — KHÔNG phải để tạo dataset train thật.

Yêu cầu: LM Studio đang chạy server với teacher model đã nạp.
"""
from config import PILOT
from pipeline.expand import expand
from pipeline.generate import generate
from pipeline.judge import judge
from pipeline.filter import run_filter
from pipeline.pack import pack


def main():
    print(f"=== PIPELINE THU THẬP DỮ LIỆU ({'PILOT' if PILOT else 'FULL'}) ===\n")
    print("[1/5] C1 — Expansion (seed -> scenario)")
    expand()
    print("\n[2/5] C2 — Generation Cách A (scenario -> hội thoại)")
    generate()
    print("\n[3/5] D3 — Judge + rejection sampling")
    judge()
    print("\n[4/5] D1+D2 — Filter (luật + dedup + cổng cuối)")
    run_filter()
    print("\n[5/5] E — Pack (ChatML + split train/val)")
    pack()
    print("\n=== XONG. Soi data ở các file data/*.jsonl trước khi train. ===")


if __name__ == "__main__":
    main()

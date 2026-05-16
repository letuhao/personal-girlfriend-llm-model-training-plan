"""C1 — Expansion: 79 seed -> nhiều scenario cụ thể (Self-Instruct + dedup).

Mỗi category: lặp gọi teacher nhiều batch nhỏ (EXPAND_BATCH scenario/call) tới
khi đủ SCENARIOS_PER_CATEGORY scenario mới — vì một call không thể sinh hàng
trăm scenario khác biệt. Dedup ngữ nghĩa để chặn mode collapse từ gốc.

Resume: ghi TỪNG BATCH xuống scenarios.jsonl ngay khi dedup xong (append +
flush). Chạy lại thì đếm scenario đã có mỗi category rồi sinh tiếp phần thiếu
— mất điện chỉ mất tối đa một batch đang dở.
"""
import random

import yaml

from config import (
    SEEDS_FILE, SCENARIOS_FILE, PROMPT_DIR, SCENARIOS_PER_CATEGORY,
    EXPAND_BATCH, EXPAND_TEMPERATURE, EXPAND_MAX_TOKENS, DEDUP_THRESHOLD,
    MAX_SCENARIOS,
)
from pipeline import jsonl
from pipeline.dedup import Deduper
from pipeline.llm_client import chat, extract_json


def expand():
    seeds = yaml.safe_load(open(SEEDS_FILE, encoding="utf-8"))
    template = (PROMPT_DIR / "expand.txt").read_text(encoding="utf-8")

    by_cat = {}
    for s in seeds:
        by_cat.setdefault(s["category"], []).append(s)

    # RESUME: đếm scenario đã ghi mỗi category
    have_sits = {}
    for sc in jsonl.load(SCENARIOS_FILE):
        have_sits.setdefault(sc["category"], []).append(sc["situation"])
    if have_sits:
        print("Resume expand: " + ", ".join(f"{c}={len(v)}"
              for c, v in have_sits.items()), flush=True)

    # pilot: cap tổng scenario -> mỗi category lấy MAX_SCENARIOS / số category
    per_cat_cap = (MAX_SCENARIOS // len(by_cat)) if MAX_SCENARIOS else None

    for category, cat_seeds in by_cat.items():
        seed_sits = [s["situation"] for s in cat_seeds]
        target = per_cat_cap or (len(seed_sits) + SCENARIOS_PER_CATEGORY)

        cat_sits = list(have_sits.get(category, []))   # situation đã ghi
        idx = len(cat_sits)
        deduper = Deduper(DEDUP_THRESHOLD)

        def _write(sit, note=""):
            """Ghi NGAY một scenario xuống file; cập nhật trạng thái."""
            nonlocal idx
            idx += 1
            jsonl.append(SCENARIOS_FILE, {
                "id": f"{category}-{idx:04d}",
                "category": category,
                "situation": sit,
                "note": note,
            })
            cat_sits.append(sit)

        if idx == 0:
            # category mới: ghi seed gốc trước
            for sit in seed_sits:
                if idx >= target:
                    break
                note = next((s.get("note", "") for s in cat_seeds
                             if s["situation"] == sit), "")
                _write(sit, note)
            deduper.prime(cat_sits)
        else:
            # resume: nạp lại trạng thái dedup từ scenario đã ghi
            deduper.prime(cat_sits)

        if idx >= target:
            print(f"  [{category}] đã đủ {idx} scenario -> bỏ qua", flush=True)
            continue

        max_calls = ((target - idx) // EXPAND_BATCH + 1) * 4
        for call_i in range(max_calls):
            if idx >= target:
                break
            avoid = random.sample(cat_sits, min(len(cat_sits), 45))
            examples = "\n".join(f"- {s}" for s in avoid)
            prompt = (template
                      .replace("{category}", category)
                      .replace("{examples}", examples)
                      .replace("{n}", str(EXPAND_BATCH)))
            try:
                raw = chat([{"role": "user", "content": prompt}],
                           temperature=EXPAND_TEMPERATURE,
                           max_tokens=EXPAND_MAX_TOKENS, tag="expand")
                batch = [str(x).strip() for x in extract_json(raw).get("scenarios", [])]
            except (ValueError, RuntimeError) as e:
                print(f"  [{category}] call {call_i + 1} lỗi: {e}", flush=True)
                continue

            keep = deduper.filter_new(batch)       # dedup ngữ nghĩa tăng dần
            for sit in keep:
                if idx >= target:
                    break
                _write(sit)                        # ghi NGAY từng scenario
            print(f"  [{category}] call {call_i + 1}: +{len(keep)} mới "
                  f"-> {idx}/{target}", flush=True)

    total = len(jsonl.load(SCENARIOS_FILE))
    print(f"Tổng {total} scenario -> {SCENARIOS_FILE}", flush=True)


if __name__ == "__main__":
    expand()

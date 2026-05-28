"""D3 — LLM-judge: chấm điểm từng bản + rejection sampling.

Với mỗi scenario có k bản hội thoại: chấm từng bản theo rubric đa trục, rồi
CHỌN bản tổng điểm cao nhất. Điểm của bản thắng được giữ lại để bước filter
(D) dùng làm cổng chất lượng cuối.
"""
from datetime import datetime

from config import (
    RAW_CONV_FILE, JUDGED_FILE, DPO_FILE, PROMPT_DIR,
    JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS, DPO_MIN_GAP,
)
from pipeline import jsonl
from pipeline.llm_client import chat_many, extract_json

# 4 trục core luôn áp dụng. 'dung' và 'nhip' chỉ áp dụng cho 1 số category.
CORE_AXES = ["nhap_vai", "giong_nguoi", "mach_lac", "luot_user"]


OPTIONAL_AXES = ["dung", "nhip"]


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _format_conversation(turns) -> str:
    """Đổi list lượt thành text dễ đọc cho judge."""
    out = []
    for t in turns:
        who = "User" if t["role"] == "user" else "Linh"
        out.append(f"{who}: {t['content']}")
    return "\n".join(out)


def _judge_prompt(turns, category, judge_tpl):
    """Dựng prompt judge cho một bản hội thoại."""
    return (judge_tpl
            .replace("{category}", category)
            .replace("{conversation}", _format_conversation(turns))
            + "\n/no_think")  # tắt thinking — judge chỉ cần scoring, không cần lý luận dài


def _total(score) -> int:
    """Tổng điểm để xếp hạng giữa các bản (rejection sampling)."""
    total = sum(int(score.get(a) or 0) for a in CORE_AXES)
    for a in OPTIONAL_AXES:
        if score.get(a) is not None:
            total += int(score[a])
    return total


def judge():
    records = jsonl.load(RAW_CONV_FILE)
    judge_tpl = (PROMPT_DIR / "judge.txt").read_text(encoding="utf-8")

    # RESUME: bỏ qua scenario đã chấm xong (nếu lần trước bị ngắt)
    done = jsonl.done_keys(JUDGED_FILE, "scenario_id")
    todo = [r for r in records if r["scenario_id"] not in done]
    print(f"Resume: {len(done)} đã xong, còn {len(todo)}/{len(records)} "
          f"cần chấm", flush=True)

    for i, r in enumerate(todo):
        print(f"\n[{_ts()}] --- scenario {i+1}/{len(todo)}: {r['scenario_id']} "
              f"({len(r['candidates'])} candidates) ---", flush=True)

        # chấm k bản SONG SONG — mỗi bản một call judge độc lập
        prompts = [[{"role": "user",
                     "content": _judge_prompt(t, r["category"], judge_tpl)}]
                   for t in r["candidates"]]
        print(f"[{_ts()}] calling chat_many...", flush=True)
        try:
            raws = chat_many(prompts, temperature=JUDGE_TEMPERATURE,
                             max_tokens=JUDGE_MAX_TOKENS, tag="judge")
        except RuntimeError as e:
            print(f"  [bỏ] {r['scenario_id']}: judge lỗi ({e})", flush=True)
            continue
        print(f"[{_ts()}] chat_many returned, processing {len(raws)} raws...",
              flush=True)

        scored = []
        for j, (turns, raw) in enumerate(zip(r["candidates"], raws)):
            print(f"[{_ts()}] extract_json candidate {j+1}/{len(raws)}...", flush=True)
            try:
                scored.append((turns, extract_json(raw)))
                print(f"[{_ts()}] extract_json {j+1} OK", flush=True)
            except ValueError as e:
                print(f"[{_ts()}] extract_json {j+1} FAIL: {e}", flush=True)
                continue

        if not scored:
            print(f"  [bỏ] {r['scenario_id']}: judge không chấm được bản nào",
                  flush=True)
            continue

        # rejection sampling: chọn bản tổng điểm cao nhất
        print(f"[{_ts()}] sorting scored...", flush=True)
        scored_sorted = sorted(scored, key=lambda x: _total(x[1]), reverse=True)
        best_turns, best_score = scored_sorted[0]

        print(f"[{_ts()}] jsonl.append JUDGED_FILE...", flush=True)
        jsonl.append(JUDGED_FILE, {
            "scenario_id": r["scenario_id"],
            "category": r["category"],
            "situation": r["situation"],
            "char_mood": r["char_mood"],
            "user_tone": r["user_tone"],
            "conversation": best_turns,
            "score": best_score,
            "num_candidates": len(scored),
        })
        print(f"[{_ts()}] jsonl.append JUDGED_FILE done", flush=True)
        print(f"  [{r['scenario_id']}] chọn 1/{len(scored)}  "
              f"tổng={_total(best_score)}  ly_do={best_score.get('ly_do', '')}",
              flush=True)

        # DPO pair: lưu (best, worst) nếu chênh lệch đủ lớn
        if len(scored_sorted) >= 2:
            worst_turns, worst_score = scored_sorted[-1]
            gap = _total(best_score) - _total(worst_score)
            if gap >= DPO_MIN_GAP:
                print(f"[{_ts()}] jsonl.append DPO_FILE (gap={gap})...", flush=True)
                jsonl.append(DPO_FILE, {
                    "scenario_id":    r["scenario_id"],
                    "category":       r["category"],
                    "situation":      r["situation"],
                    "char_mood":      r["char_mood"],
                    "chosen":         best_turns,
                    "chosen_score":   _total(best_score),
                    "rejected":       worst_turns,
                    "rejected_score": _total(worst_score),
                    "gap":            gap,
                })
                print(f"[{_ts()}] jsonl.append DPO_FILE done", flush=True)

    total = len(jsonl.load(JUDGED_FILE))
    print(f"Tổng {total} record trong {JUDGED_FILE}", flush=True)


if __name__ == "__main__":
    judge()

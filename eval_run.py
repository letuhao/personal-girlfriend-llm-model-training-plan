"""
Chạy golden eval deck tự động với linh-7b-qlora@f16.
Output: từng response kèm rubric để chấm tay.

Usage:
    python eval_run.py                   # f16 (default)
    python eval_run.py --model q4_k_m   # q4_k_m
    python eval_run.py --ids N-D01 P-R02  # chỉ một số ID
"""
import argparse
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL   = "http://localhost:1234/v1"
MODEL_F16  = "linh-7b-qlora@f16"
MODEL_Q4   = "linh-7b-qlora@q4_k_m"
TEMP             = 0.7
MAX_TOKENS       = 180   # giữ ngắn — tin nhắn không phải bài văn
FREQ_PENALTY     = 1.3   # chống repetition loop
MOOD             = "bình thường, hơi mệt"
SLEEP_SEC        = 0.5   # giữa mỗi call để LM Studio thở

# Stop tokens để model dừng đúng chỗ — ChatML format của Qwen2.5
STOP = ["<|im_end|>", "<|endoftext|>", "\nuser\n", "\nUser\n", "\nanh\n"]

# Regex lọc garbage unicode (SMP range U+10000+) và Latin CamelCase artifacts
_GARBAGE_RE = re.compile(r"[\U00010000-\U0010FFFF]|use[A-Z]\w{2,}|[A-Z][a-z]+[A-Z]\w+")

ROOT       = Path(__file__).parent
GOLDEN     = ROOT / "data" / "eval_golden.yaml"
SYSPROMPT  = ROOT / "prompts" / "linh_character.txt"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_system_prompt(mood: str) -> str:
    txt = SYSPROMPT.read_text(encoding="utf-8")
    return txt.replace("{char_mood}", mood)


def _clean(text: str) -> str:
    """Strip garbage tokens (Chinese ext-B, Thai noise, bare init chars)."""
    text = _GARBAGE_RE.sub("", text)
    # Cắt bỏ mọi thứ sau dấu hiệu model simulate turn kế tiếp
    for marker in ("<|im_end|>", "<|endoftext|>", "\nuser\n", "\nUser\n",
                   "\nanh\n", "\n\nuser:", "精彩", "初始化"):
        if marker in text:
            text = text[:text.index(marker)]
    return text.strip()


def call(client: OpenAI, model: str, messages: list[dict]) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMP,
            max_tokens=MAX_TOKENS,
            frequency_penalty=FREQ_PENALTY,
            stop=STOP,
        )
        return _clean(resp.choices[0].message.content)
    except Exception as exc:
        # LM Studio đôi khi throw 400 nhưng nhét response vào error message.
        # Extract nếu có thể.
        err_str = str(exc)
        m = re.search(r"'error':\s*'Failed to parse input at pos \d+:\s*(.*?)'}", err_str, re.DOTALL)
        if m:
            raw = m.group(1).replace("\\n", "\n")
            return "[PARTIAL] " + _clean(raw)
        raise


def fmt_list(items: list[str], prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}• {x}" for x in items)


def run_prompt(client: OpenAI, model: str, system: str, prompt: dict) -> str:
    """
    Xây dựng conversation history rồi call model.
    Với setup_turns: generate response cho mỗi lượt trước.
    """
    messages = [{"role": "system", "content": system}]

    for turn in prompt.get("setup_turns", []):
        user_msg = turn.get("user", "")
        messages.append({"role": "user", "content": user_msg})

        asst_hint = turn.get("assistant", "")
        if asst_hint.startswith("[") and asst_hint.endswith("]"):
            # placeholder → generate real response
            asst_text = call(client, model, messages)
        else:
            asst_text = asst_hint
        messages.append({"role": "assistant", "content": asst_text})
        time.sleep(SLEEP_SEC)

    messages.append({"role": "user", "content": prompt["user"]})
    return call(client, model, messages)


def print_result(prompt: dict, response: str, idx: int, total: int) -> None:
    pid = prompt["id"]
    cat = prompt["category"]
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"[{idx}/{total}] {pid}  ({cat})")

    if probe_type := prompt.get("probe_type"):
        print(f"  probe_type : {probe_type}")

    if setup := prompt.get("setup_turns"):
        print("  setup_turns:")
        for t in setup:
            print(f"    U: {t.get('user','')}")
            print(f"    A: {t.get('assistant','')}")

    print(f"  user       : {prompt['user']}")
    print()

    wrapped = textwrap.fill(response, width=68, initial_indent="  ", subsequent_indent="  ")
    print(f"  RESPONSE:\n{wrapped}")
    print()
    print(f"  look_for:")
    print(fmt_list(prompt.get("look_for", [])))
    print(f"  red_flags:")
    print(fmt_list(prompt.get("red_flags", [])))

    note = prompt.get("note")
    if note:
        print(f"  note: {note}")

    print(f"\n  SCORE (1-5): ___   PROBE (P/F): ___   Notes: _______________")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["f16", "q4_k_m"], default="f16")
    parser.add_argument("--ids", nargs="*", help="Chỉ chạy các ID này")
    args = parser.parse_args()

    model = MODEL_F16 if args.model == "f16" else MODEL_Q4
    client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
    system = load_system_prompt(MOOD)

    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

    all_prompts: list[dict] = []
    for section in ("normal", "probe"):
        for p in data.get(section, []):
            all_prompts.append(p)

    if args.ids:
        all_prompts = [p for p in all_prompts if p["id"] in args.ids]

    total = len(all_prompts)
    print(f"Model  : {model}")
    print(f"Prompts: {total}")
    print(f"Mood   : {MOOD}")
    print(f"Temp   : {TEMP}  MaxTok: {MAX_TOKENS}")

    for idx, prompt in enumerate(all_prompts, 1):
        try:
            response = run_prompt(client, model, system, prompt)
        except Exception as e:
            response = f"[ERROR] {e}"
        print_result(prompt, response, idx, total)
        time.sleep(SLEEP_SEC)

    print("\n" + "═" * 70)
    print("DONE. Chấm điểm và copy scoreboard vào eval_scoring.md.")


if __name__ == "__main__":
    main()

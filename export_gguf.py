"""
Export DPO model → GGUF q4_k_m.

Usage:
    HF_HUB_OFFLINE=1 python export_gguf.py
"""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import bitsandbytes as bnb
import unsloth  # must be first — applies all patches
from unsloth import FastLanguageModel
from peft.tuners.lora.bnb import Linear4bit as PeftBnbLinear4bit
from peft.tuners.lora.layer import Linear as PeftLinearStd

# Transformers 5.3.0 bug: revert_weight_conversion raises NotImplementedError for Qwen2.5.
# modeling_utils imports the function by name, so patch the local binding there.
import transformers.modeling_utils as _mu
_mu.revert_weight_conversion = lambda model, state_dict: state_dict

# ── Paths ────────────────────────────────────────────────────────────────────
DPO_LORA      = "outputs/linh-dpo/lora"
BASE_MODEL    = "/mnt/c/Users/NeneScarlet/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B-Instruct-bnb-4bit/snapshots/bdd404162d94997f390efbfa660eb3f21cbbc81d"
MERGED_DIR    = "outputs/linh-dpo/merged_hf_16bit"
GGUF_DIR      = "outputs/linh-dpo/gguf"
GGUF_F16      = f"{GGUF_DIR}/linh-dpo-f16.gguf"
GGUF_Q4       = f"{GGUF_DIR}/linh-dpo-q4_k_m.gguf"
LLAMA_CONVERT = "/home/nenecarlet/.unsloth/llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANT   = "/home/nenecarlet/.unsloth/llama.cpp/llama-quantize"

os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(GGUF_DIR, exist_ok=True)

# ── Step 1: Load via Unsloth ─────────────────────────────────────────────────
# Unsloth handles the nested SFT→DPO adapter chain correctly.
# Raw PEFT stacking produces wrong key paths and silently drops DPO weights.
print("\n[1/5] Loading DPO model via Unsloth...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DPO_LORA,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# ── Step 2: Dequantize 4-bit → bf16 ─────────────────────────────────────────
print("[2/5] Dequantizing 4-bit → bf16...", flush=True)
model.dequantize()

# After dequantize(), weights are bf16 but module classes are still
# bnb.nn.Linear4bit / peft.tuners.lora.bnb.Linear4bit.
# PEFT's bnb merge path then tries to read .quant_state → AttributeError.
# Fix: switch module classes so PEFT uses the standard (non-bnb) merge path.
print("    Switching module classes to standard Linear...", flush=True)
for module in model.modules():
    if type(module) is PeftBnbLinear4bit:
        module.__class__ = PeftLinearStd
    elif isinstance(module, bnb.nn.Linear4bit):
        module.__class__ = torch.nn.Linear

# ── Step 3: Merge LoRA into bf16 base ────────────────────────────────────────
print("[3/5] Merging LoRA adapters...", flush=True)
model = model.merge_and_unload()
model = model.to(torch.bfloat16)
gc.collect()
torch.cuda.empty_cache()

# ── Step 4: Save as standard bf16 HF model ───────────────────────────────────
print("[4/5] Saving bf16 HF model...", flush=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

cfg_path = Path(MERGED_DIR) / "config.json"
cfg = json.loads(cfg_path.read_text())
cfg.pop("quantization_config", None)
cfg_path.write_text(json.dumps(cfg, indent=2))
print("    config.json: quantization_config removed", flush=True)

del model
gc.collect()
torch.cuda.empty_cache()

# ── Step 5: HF → GGUF f16 → q4_k_m ─────────────────────────────────────────
print("[5a/5] Converting HF → GGUF f16...", flush=True)
subprocess.run([sys.executable, LLAMA_CONVERT, MERGED_DIR,
                "--outfile", GGUF_F16, "--outtype", "f16"], check=True)

print("[5b/5] Quantizing f16 → q4_k_m...", flush=True)
subprocess.run([LLAMA_QUANT, GGUF_F16, GGUF_Q4, "Q4_K_M"], check=True)

Path(GGUF_F16).unlink(missing_ok=True)
print(f"\nDone. Load in LM Studio: {GGUF_Q4}", flush=True)

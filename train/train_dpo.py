"""DPO preference tuning tiếp trên SFT adapter (outputs/linh-qlora/lora).

Load adapter SFT, train thêm một epoch DPO để model phân biệt
chosen vs rejected từ rejection sampling.

Dùng TRL 1.5.0 DPOTrainer stable với ref_model=None:
- Khi ref_model=None + PEFT model: TRL chỉ copy LoRA weights làm "ref" adapter
  (~vài chục MB), KHÔNG duplicate base model 4-bit (~4GB) → không RAM explosion
- Khác với Unsloth PatchDPOTrainer (incompatible với TRL mới, gây memory leak)

Data: data/dpo_train.jsonl, data/dpo_val.jsonl
Format mỗi record:
  prompt   : list[dict]  — system + conversation đến trước lượt cuối
  chosen   : str         — lượt assistant tốt nhất
  rejected : str         — lượt assistant kém nhất

    python -m train.train_dpo
"""
import gc
import os

import torch
import unsloth  # phải import trước trl/transformers/peft
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainerCallback
from trl import DPOConfig, DPOTrainer

import config as C


class _MemoryCallback(TrainerCallback):
    """Force-free Unsloth's activation offload buffers sau mỗi gradient step.

    Unsloth's "smartly offload gradients" đẩy activations lên CPU RAM trong
    gradient_accumulation_steps và chỉ free sau khi .train() xong — gây RAM
    tăng liên tục. Callback này reset buffer thủ công sau mỗi step.
    """
    def on_step_end(self, args, state, control, **kwargs):
        try:
            from unsloth_zoo.gradient_checkpointing import (
                reset_unsloth_gradient_checkpointing_buffers,
            )
            reset_unsloth_gradient_checkpointing_buffers()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

# ── Config ──────────────────────────────────────────────────────────────────
SFT_LORA_DIR  = "outputs/linh-qlora/lora"
OUTPUT_DIR    = "outputs/linh-dpo"
MAX_SEQ_LEN   = 2048
EPOCHS        = 1
LEARNING_RATE = 5e-6    # post-SFT DPO trên 7B — per philschmid 2025 guide

DPO_TRAIN = C.DATA_DIR / "dpo_train.jsonl"
DPO_VAL   = C.DATA_DIR / "dpo_val.jsonl"


def _find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    path = os.path.join(output_dir, checkpoints[-1])
    print(f"  [resume] Tìm thấy checkpoint: {path}", flush=True)
    return path


def main():
    # 1) Load SFT adapter — base 4-bit + LoRA weights từ SFT
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_LORA_DIR,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )

    # 2) Data
    train_ds = load_dataset("json", data_files=str(DPO_TRAIN), split="train")
    val_ds   = load_dataset("json", data_files=str(DPO_VAL),   split="train")

    # Bỏ field meta — DPOTrainer chỉ cần prompt/chosen/rejected
    keep = {"prompt", "chosen", "rejected"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    val_ds   = val_ds.remove_columns([c for c in val_ds.column_names   if c not in keep])

    # Convert prompt (list[dict]) -> text với add_generation_prompt=True
    def apply_prompt_template(batch):
        return {
            "prompt": [
                tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
                for msgs in batch["prompt"]
            ]
        }

    train_ds = train_ds.map(apply_prompt_template, batched=True)
    val_ds   = val_ds.map(apply_prompt_template,   batched=True)

    print(f"DPO train: {len(train_ds)}  val: {len(val_ds)}", flush=True)

    resume_ckpt = _find_latest_checkpoint(OUTPUT_DIR)

    # 3) DPOTrainer với ref_model=None — TRL tạo "ref" adapter copy LoRA weights,
    #    base model 4-bit được share → RAM an toàn
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[_MemoryCallback()],
        args=DPOConfig(
            max_length=MAX_SEQ_LEN,
            precompute_ref_log_probs=True,  # tính ref logprobs 1 lần trước train, không giữ ref model trong RAM
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # standard PyTorch checkpointing, không offload lên CPU RAM
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=37,  # 10% of ~367 steps (2940 / 1 / 8)
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=OUTPUT_DIR,
            seed=C.SPLIT_SEED,
            dataset_num_proc=1,
        ),
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 4) Lưu adapter + GGUF
    os.makedirs(f"{OUTPUT_DIR}/lora", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/gguf", exist_ok=True)
    model.save_pretrained(f"{OUTPUT_DIR}/lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")
    model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer,
                               quantization_method="q4_k_m")
    print(f"Xong. DPO adapter + GGUF nằm trong {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

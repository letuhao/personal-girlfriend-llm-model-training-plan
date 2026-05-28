"""Train QLoRA cho nhân vật Linh bằng Unsloth.

Mặc định dùng data/train_mixed.jsonl (character + base instruction 50/50)
để tránh catastrophic forgetting. Fallback về train.jsonl nếu chưa có mixed.

    python -m train.train_qlora              # dùng train_mixed.jsonl
    python -m train.train_qlora --no-mixed   # dùng train.jsonl thuần character

LƯU Ý: API của Unsloth/trl thay đổi khá nhanh. Nếu lỗi import hoặc lỗi tham
số, đối chiếu với notebook Qwen2.5 mới nhất trên repo Unsloth — phần logic
(siêu tham số, loss masking) vẫn giữ nguyên ý nghĩa.
"""
import argparse
import os

import unsloth  # phải import trước trl/transformers/peft để optimization có hiệu lực
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import config as C

# ── Siêu tham số (khớp phần thảo luận thiết kế) ────────────────────────
BASE_MODEL    = "/mnt/c/Users/NeneScarlet/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B-Instruct-bnb-4bit/snapshots/bdd404162d94997f390efbfa660eb3f21cbbc81d"
MAX_SEQ_LEN   = 4096        # full system prompt (~1500 tok) + hội thoại
LORA_R        = 64
LORA_ALPHA    = 64
EPOCHS        = 3           # tăng lên 3 — mixed data nhiều hơn, cần thêm epoch
LEARNING_RATE = 2e-4
OUTPUT_DIR    = "outputs/linh-qlora"

TRAIN_MIXED = C.DATA_DIR / "train_mixed.jsonl"
VAL_MIXED   = C.DATA_DIR / "val_mixed.jsonl"


def _find_latest_checkpoint(output_dir):
    """Trả path checkpoint mới nhất, hoặc None nếu chưa có."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # sắp theo step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    path = os.path.join(output_dir, checkpoints[-1])
    print(f"  [resume] Tìm thấy checkpoint: {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mixed", action="store_true",
                        help="Dùng train.jsonl thuần character thay vì train_mixed.jsonl")
    args = parser.parse_args()

    if args.no_mixed:
        train_file = str(C.TRAIN_FILE)
        val_file   = str(C.VAL_FILE)
        print("[DATA] Chế độ character-only (train.jsonl)")
    elif TRAIN_MIXED.exists():
        train_file = str(TRAIN_MIXED)
        val_file   = str(VAL_MIXED) if VAL_MIXED.exists() else str(C.VAL_FILE)
        print(f"[DATA] Chế độ mixed (train_mixed.jsonl) — tránh catastrophic forgetting")
    else:
        train_file = str(C.TRAIN_FILE)
        val_file   = str(C.VAL_FILE)
        print("[DATA] Không tìm thấy train_mixed.jsonl, fallback về train.jsonl")
        print("       Chạy: python prepare_base_data.py && python pack_mixed.py")

    # 1) Nạp base model ở 4-bit (QLoRA: base đóng băng, chỉ train adapter)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,                         # tự chọn bf16 trên 4090
    )

    # 2) Base model không có chat template -> gắn ChatML của Qwen2.5
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # 3) Gắn adapter LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,                    # ổn định ở rank cao (r=64)
        use_gradient_checkpointing="unsloth",
        random_state=C.SPLIT_SEED,
    )

    # 4) Dữ liệu — áp chat template ChatML lên field "messages"
    def to_text(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=False)
            for m in batch["messages"]
        ]}

    train_ds = load_dataset("json", data_files=train_file, split="train")
    val_ds   = load_dataset("json", data_files=val_file,   split="train")

    def _valid(x):
        msgs = x.get("messages")
        return (isinstance(msgs, list) and len(msgs) >= 2
                and all(isinstance(m, dict) and m.get("role") and m.get("content")
                        for m in msgs))

    n_train_before = len(train_ds)
    n_val_before   = len(val_ds)
    train_ds = train_ds.filter(_valid)
    val_ds   = val_ds.filter(_valid)
    if len(train_ds) < n_train_before:
        print(f"[WARN] loại {n_train_before - len(train_ds)} record lỗi khỏi train", flush=True)
    if len(val_ds) < n_val_before:
        print(f"[WARN] loại {n_val_before - len(val_ds)} record lỗi khỏi val", flush=True)

    train_ds = train_ds.map(to_text, batched=True)
    val_ds   = val_ds.map(to_text, batched=True)

    # 5) Trainer
    resume_ckpt = _find_latest_checkpoint(OUTPUT_DIR)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,        # -> batch hiệu dụng 16
            warmup_ratio=0.05,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch",                # log val loss mỗi epoch
            save_strategy="epoch",                # checkpoint mỗi epoch
            output_dir=OUTPUT_DIR,
            seed=C.SPLIT_SEED,
        ),
    )

    # 6) Loss masking — CHỈ tính loss trên lượt của Linh (assistant),
    #    bỏ qua system prompt và lượt User.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 7) Train (resume tự động nếu có checkpoint)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 8) Lưu adapter LoRA
    model.save_pretrained(f"{OUTPUT_DIR}/lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

    # 9) Xuất GGUF (q4_k_m) để nạp thẳng vào LM Studio chat với Linh
    model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer,
                               quantization_method="q4_k_m")
    print(f"Xong. Adapter LoRA + GGUF nằm trong {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

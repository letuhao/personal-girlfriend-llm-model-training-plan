"""Train QLoRA cho nhân vật Linh bằng Unsloth.

Chạy SAU khi đã có data/train.jsonl và data/val.jsonl.
Yêu cầu GPU NVIDIA (đủ cho RTX 4090 24GB). Cài: xem README mục
"Cài cho training".

    python -m train.train_qlora

LƯU Ý: API của Unsloth/trl thay đổi khá nhanh. Nếu lỗi import hoặc lỗi tham
số, đối chiếu với notebook Qwen2.5 mới nhất trên repo Unsloth — phần logic
(siêu tham số, loss masking) vẫn giữ nguyên ý nghĩa.
"""
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import config as C

# ── Siêu tham số (khớp phần thảo luận thiết kế) ────────────────────────
BASE_MODEL    = "unsloth/Qwen2.5-7B-bnb-4bit"   # BASE 4-bit, KHÔNG phải Instruct
MAX_SEQ_LEN   = 4096        # full system prompt (~1500 tok) + hội thoại
LORA_R        = 64
LORA_ALPHA    = 64
EPOCHS        = 2           # personality fine-tune overfit nhanh — bắt đầu 2
LEARNING_RATE = 2e-4
OUTPUT_DIR    = "outputs/linh-qlora"


def main():
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

    train_ds = load_dataset("json", data_files=str(C.TRAIN_FILE), split="train")
    val_ds   = load_dataset("json", data_files=str(C.VAL_FILE), split="train")
    train_ds = train_ds.map(to_text, batched=True)
    val_ds   = val_ds.map(to_text, batched=True)

    # 5) Trainer
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
            save_strategy="epoch",                # checkpoint mỗi epoch để eval
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

    # 7) Train
    trainer.train()

    # 8) Lưu adapter LoRA
    model.save_pretrained(f"{OUTPUT_DIR}/lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

    # 9) Xuất GGUF (q4_k_m) để nạp thẳng vào LM Studio chat với Linh
    model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer,
                               quantization_method="q4_k_m")
    print(f"Xong. Adapter LoRA + GGUF nằm trong {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

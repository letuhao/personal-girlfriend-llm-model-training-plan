# fine-tune-girlfriend

Distill một nhân vật chatbot tiếng Việt — **Linh**, cô gái Hà Nội gắt gỏng,
nữ quyền sắc bén — từ một teacher model lớn chạy local sang một student model
nhỏ (Qwen2.5-7B) chạy được trên RTX 4090.

Phương pháp: **response distillation** — teacher sinh dữ liệu hội thoại, student
học bắt chước qua QLoRA fine-tuning.

> Dự án học fine-tune từ zero. Code ưu tiên rõ ràng, comment giải thích "tại sao".

## Cấu trúc repo

```
config.py              cấu hình trung tâm (PILOT, teacher model, hyperparams)
diagnose.py            test teacher 1 call/loại prompt trước khi chạy full
run_pilot.py           chạy toàn bộ pipeline thu thập dữ liệu
prompts/               linh_character · judge · expand · conversation_genA
data/seeds.yaml        79 seed viết tay
pipeline/
  llm_client.py        client LM Studio + trace + parse JSON chịu lỗi
  trace.py             ghi log mọi call teacher -> logs/trace.jsonl
  jsonl.py             đọc/ghi JSONL + tiện ích resume
  dedup.py             dedup ngữ nghĩa bằng embedding bge-m3
  expand · generate · judge · filter · pack   — 5 stage của pipeline
train/train_qlora.py   train QLoRA bằng Unsloth
```

## Pipeline

```
data/seeds.yaml          79 seed viết tay
   │  C1  expand.py      Self-Instruct theo batch + dedup
   ▼
scenarios.jsonl
   │  C2  generate.py    sinh hội thoại multi-turn (Cách A), k bản/scenario
   ▼
raw_conversations.jsonl
   │  D3  judge.py       LLM-judge chấm điểm + rejection sampling (chọn 1/k)
   ▼
judged.jsonl
   │  D1+D2  filter.py   luật + dedup + cổng chất lượng
   ▼
dataset.jsonl
   │  E   pack.py        -> ChatML, split stratified 95/5
   ▼
train.jsonl / val.jsonl  →  train/train_qlora.py
```

## Chuẩn bị

1. **Teacher** — mở **LM Studio**, nạp model teacher rồi bật server
   OpenAI-compatible ở tab Developer. Trong settings server, tăng số request
   song song lên 4+.
   Teacher nên là model **non-reasoning + MoE (A3B) + vừa hẳn VRAM** để nhanh
   (đang dùng `huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated`).
2. **Kiểm tra `config.py`** — `TEACHER_MODEL` phải khớp định danh model trong
   LM Studio (xem `GET http://localhost:1234/v1/models`).
3. **Cài phụ thuộc pipeline:**
   ```
   pip install -r requirements.txt
   ```

## Chẩn đoán teacher (chạy trước)

Trước khi chạy cả pipeline, kiểm tra teacher hoạt động + đủ nhanh:

```
python -u diagnose.py
```

Nó gọi teacher 1 lần cho mỗi loại prompt (expand / generate / judge), in rõ
input + output + kết quả parse JSON. Mọi call ghi vào `logs/trace.jsonl`.

## Chạy pipeline thu thập dữ liệu

`config.py` có công tắc `PILOT`. `PILOT = True` chạy nhỏ để kiểm tra pipeline
thông suốt:

```
python run_pilot.py
```

Hoặc chạy từng bước để soi dữ liệu giữa chừng:

```
python -m pipeline.expand
python -m pipeline.generate
python -m pipeline.judge
python -m pipeline.filter
python -m pipeline.pack
```

Sau khi pilot chạy ổn và đã **soi kỹ data từng chặng** (đọc file `.jsonl`),
đổi `PILOT = False` trong `config.py` rồi chạy lại để tạo dataset thật.

## Ngắt & resume

Full run rất dài — pipeline **chịu được ngắt giữa chừng**. Mọi stage dài ghi
từng record/batch xuống file ngay khi xong (append + flush, giống `logs/`).
Khi chạy lại:

- `expand`   — resume theo category: category đã đủ scenario thì bỏ qua,
  category đang dở thì sinh tiếp phần thiếu (ghi từng batch ~30 scenario).
- `generate` — bỏ qua scenario đã có trong `raw_conversations.jsonl`.
- `judge`    — bỏ qua scenario đã có trong `judged.jsonl`.
- `filter` / `pack` — nhanh, luôn chạy lại từ đầu.

Cứ chạy lại `python run_pilot.py` là pipeline tiếp tục từ chỗ dở. Mất điện chỉ
mất tối đa một batch/record đang dở. Muốn chạy lại một stage TỪ ĐẦU: xoá file
output của stage đó rồi chạy lại.

## Cài cho training (cần GPU)

Unsloth + các thư viện train, cài riêng vì nặng và phụ thuộc CUDA:

```
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
```

> API của Unsloth/trl thay đổi khá nhanh. Nếu `train_qlora.py` lỗi, đối chiếu
> với notebook Qwen2.5 mới nhất ở repo Unsloth.

## Train

```
python -m train.train_qlora
```

Kết quả nằm trong `outputs/linh-qlora/`: adapter LoRA + bản GGUF. Nạp file
GGUF vào LM Studio để chat thử với Linh.

## Lưu ý quan trọng

- **Eval bằng cách chat thật**, không tin val loss.
- Soi data ở mỗi file `.jsonl` trung gian — pipeline cố tình ghi ra từng chặng
  để debug và rerun một bước không phải chạy lại từ đầu.
- Triết lý: chạy bản đơn giản trước (pilot, Cách A), *nhìn thấy lỗi trên data
  thật*, rồi mới nâng cấp (Cách B role-simulation cho các category quan trọng).

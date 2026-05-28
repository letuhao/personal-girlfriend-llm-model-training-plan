# Troubleshooting & Lessons Learned

Toàn bộ vấn đề gặp trong dự án fine-tune nhân vật Linh, theo thứ tự thời gian.
Mục đích: không lặp lại sai lầm ở vòng sau, và có tài liệu tra cứu khi gặp lỗi tương tự.

---

## Giai đoạn 1 — Chọn teacher model

### Vấn đề: Teacher model quá chậm (~120s/call)
**Setup ban đầu:** Thử dùng model reasoning dày đặc (non-MoE) + dense offload một phần sang CPU.

**Triệu chứng:** Mỗi call mất ~120 giây. Với pipeline cần ~15.000 call (3000 scenario × k=5 + judge), tổng thời gian > 500 giờ — không khả thi.

**Root cause:** Model reasoning (có thinking tokens) + dense model không vừa VRAM → phải offload sang CPU → I/O bottleneck.

**Fix:** Chuyển sang `huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated`:
- A3B MoE: active params nhỏ hơn nhiều so với total params → vừa VRAM
- Abliterated: không có RLHF refusal → cần thiết cho NSFW content
- Distilled từ Claude 4.7 Opus: prose human-like hơn các base model khác
- **Kết quả:** ~3–5s/call

**Lesson:** Với bulk data generation (>10k calls), inference speed quan trọng hơn model quality một mức độ nhất định. 10x faster → 10x more data trong cùng thời gian.

---

### Vấn đề: Teacher viết quá bóng bẩy, literary
**Triệu chứng:** Nhân vật Linh (nên "cộc, gắt, viết thường") ra câu đủ ngữ pháp, từ sách vở, viết hoa đầu câu, văn phong báo chí.

**Root cause:** Model distilled từ Claude → kế thừa văn phong cẩn thận. Mô tả character bằng tính từ ("cộc", "gắt") không đủ — model không biết "cộc" trông như thế nào trên màn hình chat.

**Fix:** Thêm **few-shot examples** trực tiếp trong prompt generation. Model bắt chước ví dụ tốt hơn nhiều so với làm theo instruction thuần văn bản.
- Chia ví dụ theo register: lượt em–anh vs tao–mày
- Ví dụ cho thấy: viết thường, câu ngắn, không dấu hỏi lịch sự cuối

**Lesson:** Với character voice, few-shot examples > nhân vật description. Cho model xem 3 ví dụ đúng > mô tả 3 đoạn văn.

---

## Giai đoạn 2 — Tạo dataset (pipeline)

### Vấn đề: Judge step bị treo vĩnh viễn (Qwen3 runaway thinking)
**Triệu chứng:** LM Studio hiển thị "Reasoned for 113645.57 seconds" — model đang trong vòng lặp thinking vô tận. Thread-2 trong `chat_many` không bao giờ return. Cả pipeline bị block.

**Root cause:** Qwen3 có chế độ "thinking" (extended reasoning). Trong llama.cpp / LM Studio, `max_tokens` giới hạn output tokens nhưng **KHÔNG giới hạn thinking tokens**. Model có thể think vô tận trước khi bắt đầu output.

**Fix:**
1. `timeout=120.0` trên OpenAI client (`httpx.TimeoutError` → retry logic) — **fix thực sự**
2. Thêm `\n/no_think` vào cuối judge prompt — model thường bỏ qua nhưng đôi khi có tác dụng
3. Thêm per-thread logging vào `chat_many` để biết thread nào bị kẹt

**Lesson:** Với reasoning models, luôn set HTTP timeout ở client level. `max_tokens` chỉ giới hạn output, không giới hạn thinking.

---

### Vấn đề: extract_json fail vì output format của teacher
**Triệu chứng:** JSON parse error ở bước judge — teacher trả về JSON bọc trong ` ```json ... ``` `, hoặc có trailing comma trước `}`.

**Fix trong `extract_json()`:**
- Strip ` ```json ``` ` fence
- `re.sub(r",(\s*[}\]])", r"\1", text)` — bỏ trailing comma
- `json.JSONDecoder().raw_decode()` — parse đúng một object, bỏ qua dữ liệu thừa sau

**Lesson:** LLM output JSON cần defensive parsing. Không dùng `json.loads()` trực tiếp.

---

## Giai đoạn 3 — Training v1 (student model đầu tiên)

> **Lưu ý:** Phần này tái dựng từ artifacts trong code — cần bạn confirm chi tiết.

### Vấn đề: Model sau training "ngu ngẳn" (catastrophic forgetting)
**Triệu chứng:** Model mất khả năng trả lời câu hỏi hữu ích sau khi fine-tune. Chỉ biết roleplay, hỏi thứ gì liên quan đến thực tế thì trả lời sai hoặc không trả lời.

**Root cause:** Dataset thuần character (daily/conflict/intimate/edge/persona) — không có task hữu ích nào giữ lại general knowledge. Fine-tune overwrite kiến thức nền.

**Fix:** Thêm `base_instructions.jsonl` — tập data instruction-following thông thường, mix 50/50 với character data. Xem `pack_mixed.py`.

**Lesson:** Character fine-tune **luôn cần data mix** với general instruction data để chống catastrophic forgetting. Tỉ lệ 40–50% base data là khuyến nghị.

---

### Vấn đề: Model generate token không bao giờ dừng (infinite generation)
**Triệu chứng:** Model không dừng ở `<|im_end|>`. Tiếp tục generate vô tận, đôi khi simulate cả lượt user tiếp theo, hoặc xuất ra ký tự Chinese (`精彩`, `初始化`), Unicode garbage (SMP range), CamelCase artifacts (`useStateHook`).

**Root cause (suy đoán):** Dùng **base model** (không phải Instruct) — base model chưa được RLHF để biết dừng theo chat format. Hoặc/và: `pad_token = eos_token` gây vòng lặp vô tận khi model sample `<|endoftext|>`.

**Fix:**
1. Chuyển sang **Instruct variant** (`Qwen2.5-7B-Instruct`) — đã được train để follow ChatML và dừng đúng
2. Trong `eval_run.py`: thêm `stop=["<|im_end|>", "<|endoftext|>", "\nuser\n", ...]` và `frequency_penalty=1.3` như guardrail
3. Thêm `_clean()` post-processing để cắt garbage nếu model vẫn rò một ít

**Lesson:** Với character fine-tune, dùng **Instruct model** làm base, không phải base model. Instruct model đã có chat format baked in — ít rủi ro hơn nhiều.

---

## Giai đoạn 4 — Tạo dataset v2 + data mix

### Vấn đề: WSL2 không thấy GPU (torch CPU-only)
**Triệu chứng:** `torch.cuda.is_available()` trả về `False` mặc dù `nvidia-smi` thấy RTX 4090 bình thường.

**Root cause:** PyPI default `torch` package là CPU-only. Unsloth cài lại torch và chọn CPU version.

**Fix:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Và thêm `pip.conf` để ngăn Unsloth overwrite:
```ini
# .venv-train-wsl/pip.conf
[global]
index-url = https://download.pytorch.org/whl/cu128
extra-index-url = https://pypi.org/simple
```

**Lesson:** Trên WSL2 phải chỉ định CUDA wheel index URL. Không để pip tự chọn — nó luôn chọn CPU version vì compatible rộng hơn.

---

### Vấn đề: HuggingFace download lỗi từ WSL (connection reset)
**Triệu chứng:** `huggingface-cli download` hoặc `AutoModel.from_pretrained()` bị `ConnectionResetError` từ WSL2.

**Root cause:** Network stack khác nhau giữa WSL2 và Windows host. Một số HF CDN endpoint không ổn định qua WSL NAT.

**Fix:**
```python
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
```
Và trỏ model path thẳng đến Windows cache (accessible từ WSL qua `/mnt/c/`):
```python
BASE_MODEL = "/mnt/c/Users/<user>/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B-Instruct-bnb-4bit/snapshots/<hash>"
```

---

### Vấn đề: pack_mixed.py đọc sai file nguồn → Jinja2 crash
**Triệu chứng:** `train_qlora.py` crash với `UndefinedError: None has no element 0` trong `apply_chat_template`. Filter báo loại toàn bộ 2340 records.

**Root cause:** `pack_mixed.py` đọc từ `data/dataset.jsonl` (field là `"conversation"`, không có system prompt). Nhưng `train_qlora.py` kỳ vọng `"messages"` field đã có system prompt — chỉ có trong `train.jsonl` (output của `pack.py`).

**Fix:** Sửa `pack_mixed.py` để đọc từ `TRAIN_FILE`/`VAL_FILE` (đã packed, có `messages` + system prompt) thay vì `dataset.jsonl`.

**Lesson:** Khi có nhiều file trung gian trong pipeline, luôn document rõ field schema ở mỗi bước. `dataset.jsonl` ≠ `train.jsonl` dù tên nghe giống.

---

### Vấn đề: Checkpoint cũ incompatible với PyTorch mới
**Triệu chứng:** Resume từ checkpoint cũ lỗi `torch.load(rng_file, weights_only=True)` vì numpy globals.

**Fix:** Xóa `outputs/linh-qlora/` — checkpoint cũ đã train trên data sai (trước khi fix pack_mixed.py) nên không cần giữ.

---

## Giai đoạn 5 — SFT Training v2

### Vấn đề: TRL import error (TRANSFORMERS_CACHE deprecated)
**Triệu chứng:** `ImportError: llm_blender.pair_ranker` kéo theo `TRANSFORMERS_CACHE` removed trong transformers 5.3.0.

**Fix:** `pip install --upgrade trl`

---

### Vấn đề: `warmup_ratio` deprecated warning
**Triệu chứng:** `warmup_ratio is deprecated and will be removed in v5.2. Use warmup_steps instead.`

**Fix:** Thay `warmup_ratio=0.10` bằng `warmup_steps=N` tính thủ công. Công thức: `floor(total_steps * ratio)` với `total_steps = len(train_ds) / batch_size / grad_accum`. Ví dụ DPO: 2940 / 1 / 8 ≈ 367 steps → 10% = 37 steps.

---

## Giai đoạn 6 — Preference Training (DPO/ORPO/SimPO)

### Vấn đề: DPO RAM explosion v1 — PatchDPOTrainer incompatible
**Triệu chứng:** RAM tăng từ 20GB → 70GB+ trong 14/368 steps, tiếp tục tăng không dừng.

**Root cause:** Unsloth's `PatchDPOTrainer` incompatible với TRL mới → tạo full ref model copy trong CPU RAM như memory leak. Thêm vào đó: `DPOConfig.max_prompt_length` đã bị remove khỏi TRL 1.5.0.

**Fix ban đầu:** Chuyển sang `ORPOTrainer` (không cần ref model).

**Vấn đề với fix này:** ORPO được thiết kế để thay thế pipeline SFT+DPO trong một bước — train từ base model. Dùng ORPO *sau* SFT là sai design intent (NLL component redundant). Thêm nữa, TRL 1.5.0 đã chuyển ORPO vào `trl.experimental`.

---

### Vấn đề: TRL 1.5.0 — CPOTrainer/ORPOTrainer bị remove khỏi stable API
**Triệu chứng:** `ImportError: cannot import name 'CPOConfig' from 'trl'`

**Root cause:** TRL v1.0 (March 2026) tái cấu trúc: chỉ giữ SFT/DPO/GRPO/KTO trong stable. CPO và ORPO chuyển vào `trl.experimental` hoặc bị loại.

**Fix:** Dùng `DPOTrainer` (TRL stable) với `ref_model=None`. Khi `ref_model=None` + PEFT model, TRL chỉ copy LoRA adapter (~50–100MB) làm ref — không duplicate base model 4-bit (~4GB).

---

### Vấn đề: DPO RAM explosion v2 — Unsloth activation offloading
**Triệu chứng:** RAM tăng mỗi step (~vài trăm MB/step). Xuất hiện message `"Unsloth: Will smartly offload gradients to save VRAM!"`.

**Root cause:** Unsloth tự replace `DPOTrainer` bằng `UnslothDPOTrainer` (lưu ở `unsloth_compiled_cache/`). Feature "smart gradient offloading" của Unsloth đẩy activation buffers lên CPU RAM trong mỗi micro-step của gradient accumulation. Với `gradient_accumulation_steps=8`, activations của 8 forward pass tích lũy trong RAM và **chỉ được free sau khi `.train()` kết thúc hoàn toàn** — không giữa các steps.

**Fix:**
1. `gradient_checkpointing=True` + `gradient_checkpointing_kwargs={"use_reentrant": False}` trong `DPOConfig` → dùng standard PyTorch checkpointing thay vì Unsloth's offloaded version (recompute trên GPU thay vì offload sang CPU)
2. `precompute_ref_log_probs=True` → tính ref logprobs một lần trước training, lưu vào dataset columns, không cần forward pass trong mỗi step
3. `_MemoryCallback` → force reset Unsloth's activation buffers sau mỗi gradient step thủ công

**Lesson:** Unsloth patches nhiều TRL trainers tự động tại import time. Các "optimization" của nó đôi khi trade VRAM cho RAM — không phải lúc nào cũng là deal tốt nếu RAM của máy hạn chế.

---

## Tóm tắt nhanh — Lookup table

| Triệu chứng | Root cause | Fix |
|---|---|---|
| Teacher 120s/call | Dense model + CPU offload | Chuyển MoE A3B model |
| Teacher viết bóng bẩy | Instruction text không đủ | Few-shot examples trong prompt |
| Judge treo vĩnh viễn | Qwen3 infinite thinking | `timeout=120s` trên OpenAI client |
| JSON parse fail | Model bọc ```json, trailing comma | Defensive `extract_json()` |
| Model ngu sau SFT | Catastrophic forgetting | Mix 40–50% base instruction data |
| Model gen không dừng | Base model / pad=eos | Dùng Instruct model, thêm stop tokens |
| torch.cuda = False | PyPI chọn CPU torch | `--index-url cu128` + pip.conf |
| HF download fail (WSL) | WSL2 network stack | `snapshot_download()` + Windows cache path |
| Jinja2 UndefinedError | pack_mixed.py đọc sai file | Đọc `train.jsonl` không phải `dataset.jsonl` |
| Checkpoint incompatible | Old numpy globals | `rm -rf outputs/linh-qlora` |
| TRL import error | `TRANSFORMERS_CACHE` removed | `pip install --upgrade trl` |
| DPO RAM 70GB+ (v1) | PatchDPOTrainer leak | Dùng standard `DPOTrainer` |
| DPO RAM tăng/step (v2) | Unsloth activation offload | standard GC kwargs + `precompute_ref_log_probs` + MemoryCallback |

---

## Thứ tự chạy pipeline (reference)

```
# 1. Tạo data
python run_pipeline.py          # expand → generate → judge → filter

# 2. Chuẩn bị base data (chống forgetting)
python prepare_base_data.py

# 3. Pack + mix
python -m pipeline.pack         # → train.jsonl, val.jsonl
python pack_mixed.py            # → train_mixed.jsonl, val_mixed.jsonl
python -m pipeline.pack_dpo     # → dpo_train.jsonl, dpo_val.jsonl

# 4. SFT
python -m train.train_qlora     # → outputs/linh-qlora/lora + gguf

# 5. DPO preference tuning
python -m train.train_dpo       # → outputs/linh-dpo/lora + gguf

# 6. Eval
python eval_run.py              # load gguf vào LM Studio trước
```

---

### Vấn đề: Unsloth `from_pretrained` crash với `LocalEntryNotFoundError`
**Triệu chứng:** `huggingface_hub.errors.LocalEntryNotFoundError: Got: ConnectError: [Errno 104] Connection reset by peer` ngay khi gọi `FastLanguageModel.from_pretrained()` — không liên quan đến model download.

**Root cause:** Unsloth gọi `get_statistics()` (telemetry ping) về HuggingFace Hub trong mỗi lần `from_pretrained`. Khi WSL2 network không ổn định, connection bị reset → Unsloth abort toàn bộ load.

**Fix:** Set `HF_HUB_OFFLINE=1` trước khi chạy:
```bash
HF_HUB_OFFLINE=1 python your_script.py
# hoặc trong script:
os.environ["HF_HUB_OFFLINE"] = "1"
```

---

### Vấn đề: Flash Attention 2 không cài được trên WSL
**Triệu chứng:** Build từ source fail:
`RuntimeError: The current installed version of g++ (13.3.0) is greater than the maximum required version by CUDA 12.0. Please make sure to use g++ (>=6.0.0, <13.0).`

**Root cause:** 2 mismatch xếp chồng:
1. Pre-built wheel không tồn tại cho `cu128+torch2.11` (404 từ GitHub releases)
2. System `nvcc` là 12.0 (từ `apt install nvidia-cuda-toolkit`), trong khi PyTorch dùng CUDA 12.8 (pip package). nvcc 12.0 có constraint `gcc < 13.0`, nhưng Ubuntu 24.04 mặc định gcc 13.3.

**Fix nếu cần:**
```bash
sudo apt install gcc-12 g++-12
CXX=g++-12 CC=gcc-12 pip install flash-attn --no-build-isolation
```
Build ~15–20 phút.

**Workaround hiện tại:** Dùng Xformers (đã cài sẵn, cùng numerics). FA2 chỉ nhanh hơn ~10–15% — không cần thiết cho run ngắn.

---

## Câu hỏi còn mở / cần bạn confirm

1. **Lần đầu train với Qwen2.5-7B base**: có thêm triệu chứng nào khác ngoài infinite gen và "ngu ngẳn" không?
2. **Garbage tokens** (`精彩`, `初始化`): có xuất hiện ở base model run hay cả Instruct run?
3. **DPO memory leak hiện tại** với fix mới nhất (`gradient_checkpointing_kwargs` + `precompute_ref_log_probs` + `MemoryCallback`): đã ổn chưa?

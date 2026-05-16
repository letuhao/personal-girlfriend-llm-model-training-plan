# fine-tune-girlfriend

Distill a Vietnamese character chatbot — **Linh**, an abrasive, sharply
feminist girl from Hanoi — from a large local teacher model into a small
student model (Qwen2.5-7B) that runs on an RTX 4090.

Method: **response distillation** — the teacher generates conversation data,
the student learns to imitate it via QLoRA fine-tuning.

> A learn-fine-tuning-from-zero project. Code favors clarity, with comments
> that explain the "why".

## Repository layout

```
config.py              central config (PILOT switch, teacher model, hyperparams)
diagnose.py            test the teacher with 1 call per prompt type before a full run
run_pilot.py           run the whole data-collection pipeline
prompts/               linh_character · judge · expand · conversation_genA
data/seeds.yaml        79 hand-written seeds
pipeline/
  llm_client.py        LM Studio client + tracing + lenient JSON parsing
  trace.py             logs every teacher call -> logs/trace.jsonl
  jsonl.py             JSONL read/write + resume helpers
  dedup.py             semantic dedup with the bge-m3 embedding model
  expand · generate · judge · filter · pack   — the 5 pipeline stages
train/train_qlora.py   QLoRA training with Unsloth
```

## Pipeline

```
data/seeds.yaml          79 hand-written seeds
   │  C1  expand.py      batched Self-Instruct + dedup
   ▼
scenarios.jsonl
   │  C2  generate.py    generate multi-turn conversations (Method A), k per scenario
   ▼
raw_conversations.jsonl
   │  D3  judge.py       LLM-judge scoring + rejection sampling (pick 1 of k)
   ▼
judged.jsonl
   │  D1+D2  filter.py   rules + dedup + final quality gate
   ▼
dataset.jsonl
   │  E   pack.py        -> ChatML, stratified 95/5 split
   ▼
train.jsonl / val.jsonl  →  train/train_qlora.py
```

## Setup

1. **Teacher** — open **LM Studio**, load the teacher model, and start the
   OpenAI-compatible server in the Developer tab. In the server settings,
   raise the parallel-request count to 4+.
   The teacher should be a **non-reasoning + MoE (A3B) model that fits fully
   in VRAM** for speed (currently using
   `huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated`).
2. **Check `config.py`** — `TEACHER_MODEL` must match the model id in LM
   Studio (see `GET http://localhost:1234/v1/models`).
3. **Install pipeline dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Diagnose the teacher (run first)

Before running the whole pipeline, verify the teacher works and is fast
enough:

```
python -u diagnose.py
```

It calls the teacher once for each prompt type (expand / generate / judge)
and prints the input + output + JSON parse result. Every call is logged to
`logs/trace.jsonl`.

## Run the data-collection pipeline

`config.py` has a `PILOT` switch. `PILOT = True` runs a small batch to verify
the pipeline end to end:

```
python run_pilot.py
```

Or run each stage separately to inspect the data in between:

```
python -m pipeline.expand
python -m pipeline.generate
python -m pipeline.judge
python -m pipeline.filter
python -m pipeline.pack
```

Once the pilot runs cleanly and you have **inspected the data at each stage**
(read the `.jsonl` files), set `PILOT = False` in `config.py` and re-run to
build the real dataset.

## Interruption & resume

A full run is very long — the pipeline **survives interruption**. Every long
stage writes each record/batch to disk as soon as it is done (append + flush,
just like `logs/`). On restart:

- `expand`   — resumes per category: skips categories that already have enough
  scenarios, continues partial ones (writes each ~30-scenario batch).
- `generate` — skips scenarios already in `raw_conversations.jsonl`.
- `judge`    — skips scenarios already in `judged.jsonl`.
- `filter` / `pack` — fast, always re-run from scratch.

Just re-run `python run_pilot.py` and the pipeline continues where it left
off. A power loss costs at most one in-flight batch/record. To re-run a stage
from scratch, delete its output file first.

## Install for training (needs a GPU)

Unsloth and the training libraries are installed separately because they are
heavy and CUDA-dependent:

```
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
```

> The Unsloth/trl APIs change fairly quickly. If `train_qlora.py` breaks,
> cross-check with the latest Qwen2.5 notebook in the Unsloth repo.

## Train

```
python -m train.train_qlora
```

Output lands in `outputs/linh-qlora/`: the LoRA adapter plus a GGUF build.
Load the GGUF into LM Studio to chat with Linh.

## Notes

- **Evaluate by actually chatting** with the model, not by trusting val loss.
- Inspect the data in each intermediate `.jsonl` file — the pipeline writes
  every stage on purpose, so you can debug and re-run one step without
  redoing everything.
- Philosophy: run the simple version first (pilot, Method A), *see the
  failures on real data*, then upgrade (Method B role-simulation for the
  important categories).
```

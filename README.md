# FDM_GEMMA4

Porting **Frequency-Division Multiplexed (FDM) memory** to **Gemma 4 31B Dense**, and
sweeping the channel count *K* to find where Gemma's sliding-window attention breaks
carrier retrieval.

FDM encodes structured facts as orthogonal sinusoidal carriers written into a model's
**attention weights** via two-block fine-tuning (attention weights trainable;
embeddings and MLPs frozen). It has been validated on dense uniform-attention models
(Qwen3-0.6B, LFM2.5-1.2B, Hermes3-3B, GPT-2 Medium) at 40 channels with ~99%
per-channel recall. This repo tests whether it ports to Gemma 4, which is **not**
uniform-attention.

---

## Why Gemma 4 is the interesting case

Gemma 4 31B interleaves **local sliding-window attention (window = 1024 tokens)** with
**global full-context attention** in a 5:1 ratio (final layer always global). FDM
encodes facts across the whole token block, so:

- When the encoding block **W ≤ 1024**, every layer can read the whole block — FDM
  should behave as it does on uniform-attention models.
- When **W > 1024**, the local layers (5 of every 6) can no longer see the full block;
  only the ~10 global layers (of 60) can integrate carriers across it. Whether
  global-only retrieval holds is the open question.

The sweep walks *K* (and therefore W) across the 1024 boundary and stops at the first K
whose per-channel recall drops below the bar.

---

## The sweep

K-ladder, each rung a two-block fine-tune gated at 99% per-channel recall
(invariant: Δf = f_s/N = 0.390625 Hz):

| Stage | K | f_s (Hz) | W (tokens) | W / window | note |
|------:|--:|---------:|-----------:|-----------:|------|
| K40  | 40  | 100 | 512  | 0.5× | baseline, inside window |
| K80  | 80  | 200 | 1024 | 1.0× | window edge |
| K99  | 99  | 200 | 1024 | 1.0× | edge, thin (~1 Hz) Nyquist margin |
| K120 | 120 | 400 | 2048 | 2.0× | first global-only test |
| K160 | 160 | 400 | 2048 | 2.0× | global-only |
| K320 | 320 | 800 | 4096 | 4.0× | deep global-only |

Demo channel count = the highest K that holds ≥99% recall. Most likely break is where
W crosses 1024 (K120–160).

---

## Files

| File | Purpose |
|------|---------|
| `fdm_gemma4_ksweep.py` | Sweep harness. Dual launch: `python3` (single-process / naive model-parallel) or `accelerate launch` (FSDP). Reuses the encoder, dataset, and data-gen from the eidetic script. |
| `eidetic_qwen3_fdm_e2e.py` | Validated FDM encoder, dataset, reasoning logic, and 40 base channel schemas. Imported by the sweep. |
| `fsdp.yaml` | Accelerate FSDP config (FULL_SHARD, `use_orig_params: true`, `mixed_precision: "no"`). |
| `recall_fix.py` / `apply_recall_fix.py` | Corrected per-channel recall metric + splicer (see Fixes). |
| `diag_eval.py` | Eval diagnostic that surfaced the metric bug. |
| `instructions.md` | Step-by-step run instructions. |
| `startup_gemma4_sweep.sh` | Pod environment setup (deps, HF login, gated-model check, weight prefetch). |

> All files for a run must live in the **same folder** — the sweep imports
> `eidetic_qwen3_fdm_e2e` from its own directory.

---

## Setup on a fresh pod

Two steps are **not** in git and must be redone each new pod:

**1. Pin torch to a CUDA build that matches the driver.** The blanket `pip install -U
torch` pulls a cu130 build that can't see the GPUs (silent CPU fallback). Force cu124:

```bash
pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio          # mismatched; not needed here
pip install -q bitsandbytes                       # 8-bit Adam (FSDP memory)
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# expect: 2.6.0+cu124 True 2
```

**2. Gemma 4 31B weights (~62 GB) at `/workspace/gemma-4-31B`.** Persist these on a
network volume to avoid re-downloading. Requires accepting the license at
`https://huggingface.co/google/gemma-4-31B`.

Then set env in every shell:

```bash
export WORKDIR=/workspace/FDM_GEMMA4 GEMMA_DIR=/workspace/gemma-4-31B FDM_REPO=$WORKDIR
cd $WORKDIR
```

---

## Run

Preflight (no GPU):

```bash
python3 -c "import eidetic_qwen3_fdm_e2e; print('import OK')"
python3 fdm_gemma4_ksweep.py --print
```

Single stage under FSDP (run in tmux; clear GPUs first if relaunching):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py --only K40
```

Full gated ladder:

```bash
accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py
```

Results land in `ksweep_results.json`; checkpoints in `ckpt_K<n>/`.

---

## Fixes discovered (Gemma-4-specific)

Gemma 4 is multimodal, which broke several assumptions from the uniform-LLM pipeline:

- **Nested config.** `vocab_size` lives at `config.text_config.vocab_size` (262144),
  not the root. `assert_vocab` reads the nested value.
- **Vision tower.** The model has a `vision_tower` whose attention projections also
  match `q/k/v/o_proj`. Two-block freezing must exclude `vision` so only the language
  model's attention (8.48B params) trains, not vision attention (143M).
- **FSDP wrap class.** The decoder block is `Gemma4TextDecoderLayer` at
  `model.language_model.layers` (not `Gemma4DecoderLayer`).
- **FSDP OOM.** `mixed_precision: bf16` upcasts params to fp32 for the optimizer;
  combined with full AdamW's two fp32 moments over 8.48B params this OOMs even sharded.
  Fix: `mixed_precision: "no"` (keep bf16) **+ 8-bit Adam** (`bnb.optim.AdamW8bit`) **+**
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Recall metric.** Per-channel recall must match the validated eval: substring-score
  the **recall channels (ch8+)** against the model's Context dump, and score the
  **reasoning channels (ch0–7)** via action/rule/meta/fact presence — *not* raw
  substring over all 40 (which scores ch0–7 as failures because their values drive
  behavior rather than being echoed). Scoring all 40 by substring undercounts recall
  badly (it reported ~44% on a model that was in fact recalling correctly).

---

## Results

Method: two-block (attention weights trained; embeddings + MLPs frozen). Recall =
per-channel substring match over recall channels (ch8 … K-1), matching the validated
eidetic eval.

| K | W | W/window | mean recall (ch8+) | passed (≥0.99) |
|--:|--:|---------:|-------------------:|:--------------:|
| 40  | 512  | 0.5× | _pending_ | _pending_ |
| 80  | 1024 | 1.0× | — | — |
| 99  | 1024 | 1.0× | — | — |
| 120 | 2048 | 2.0× | — | — |
| 160 | 2048 | 2.0× | — | — |
| 320 | 4096 | 4.0× | — | — |

> K40 is the baseline rung — its job is to confirm FDM ports to Gemma 4 at all. Fill in
> as each stage completes; the highest passing K is the demo target.

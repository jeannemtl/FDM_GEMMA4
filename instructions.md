# FDM Gemma 4 K-sweep — Run Instructions

Target hardware: 2× H200 SXM (141 GB each). Method: two-block. Repo: `FDM_GEMMA4`.

---

## 0. Prerequisites (one time, before touching the pod)

**Push the scripts to the (empty) repo** from your laptop:

```bash
git clone git@github.com:jeannemtl/FDM_GEMMA4.git
cd FDM_GEMMA4
cp /path/to/eidetic_qwen3_fdm_e2e.py .
cp /path/to/fdm_gemma4_ksweep.py .
cp /path/to/fsdp.yaml .
git add . && git commit -m "sweep scripts" && git push
```

**Accept the Gemma 4 license** (the startup script stops with this URL if not done):

```
https://huggingface.co/google/gemma-4-31B
```

---

## 1. Environment setup (on the pod)

```bash
bash startup_gemma4_sweep.sh
```

Installs deps, generates an SSH key (paste it into https://github.com/settings/keys when
prompted), logs into HuggingFace, verifies Gemma 4 access, clones `FDM_GEMMA4`, and
pre-downloads the 31B weights (~62 GB).

---

## 2. Verify the FSDP layer-class name (once)

A wrong class name silently disables FSDP sharding and causes OOM. Confirm it:

```bash
cd /workspace/FDM_GEMMA4
export GEMMA_DIR=/workspace/gemma-4-31B FDM_REPO=/workspace/FDM_GEMMA4
python3 -c "from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('$GEMMA_DIR'); print(type(m.model.layers[0]).__name__)"
```

If the output is **not** `Gemma4DecoderLayer`, edit `fsdp_transformer_layer_cls_to_wrap`
in `fsdp.yaml` to match. Also set `num_processes` in `fsdp.yaml` to your GPU count (2).

---

## 3. (Optional) Inspect the stage table — no GPU

```bash
python3 fdm_gemma4_ksweep.py --print
```

---

## 4. Calibrate — one stage, single-process, timed

```bash
tmux new -s sweep
export GEMMA_DIR=/workspace/gemma-4-31B FDM_REPO=/workspace/FDM_GEMMA4
time python3 fdm_gemma4_ksweep.py --only K40
```

Expect ~99% per-channel recall at K40 (confirms the port + vocab). The wall-clock here
sets the scale for every later stage.

---

## 5. Run the full ladder under FSDP (fast path, 2 GPUs)

```bash
accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py
```

Trains K40 → 320 in sequence, gates each at 99% recall, stops at the first failure, and
prints the demo-target K.

**Fallback (no FSDP):** runs the whole ladder in naive model-parallel across both cards —
works, reaches K320, just slower:

```bash
python3 fdm_gemma4_ksweep.py
```

**Single stage under FSDP** (e.g. just the expensive top stage):

```bash
accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py --only K320
```

---

## 6. Detach / reattach / results

```bash
# detach:    Ctrl+b then d
# reattach:  tmux attach -t sweep

cat /workspace/FDM_GEMMA4/ksweep_results.json
```

---

## Command quick-reference

| Step | Command |
|---|---|
| Setup | `bash startup_gemma4_sweep.sh` |
| Verify layer class | `python3 -c "from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('$GEMMA_DIR'); print(type(m.model.layers[0]).__name__)"` |
| Stage table (no GPU) | `python3 fdm_gemma4_ksweep.py --print` |
| Calibrate K40 | `time python3 fdm_gemma4_ksweep.py --only K40` |
| Full ladder (FSDP) | `accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py` |
| Full ladder (naive MP) | `python3 fdm_gemma4_ksweep.py` |
| One stage (FSDP) | `accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py --only K320` |
| Results | `cat /workspace/FDM_GEMMA4/ksweep_results.json` |

Always export the env vars in each new shell/tmux pane:

```bash
export GEMMA_DIR=/workspace/gemma-4-31B FDM_REPO=/workspace/FDM_GEMMA4
```

---

## What the sweep is doing

K-ladder 40 → 80 → 99 → 120 → 160 → 320, each a two-block training run gated at 99%
per-channel recall. The sweep stops at the first K that fails. Gemma 4's local
sliding-window attention is 1024 tokens, so the likely break is where W crosses 1024
(around K120–160). Your demo channel count is the highest K that passes.
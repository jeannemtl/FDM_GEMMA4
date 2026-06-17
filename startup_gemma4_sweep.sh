#!/bin/bash
set -e

# =============================================================
# startup_gemma4_sweep.sh
# RunPod environment setup for the Gemma 4 31B K-sweep.
# Method: TWO-BLOCK (attention weights trained directly). No LoRA.
# Repo: github.com/jeannemtl/FDM_GEMMA4
# =============================================================

REPO_SSH="git@github.com:jeannemtl/FDM_GEMMA4.git"
WORK=/workspace/FDM_GEMMA4
GEMMA_ID="google/gemma-4-31B"
GEMMA_DIR=/workspace/gemma-4-31B

# === 0. VERIFY GPU ===
echo "=== Checking GPU ==="
nvidia-smi || { echo "ERROR: No GPU!"; exit 1; }
echo ""
echo "NOTE: two-block on 31B trains the attention weights directly, so it needs"
echo "      a multi-GPU H200 pod — especially at K160/K320 (W=2048/4096)."
echo "      Plain python3 uses device_map=auto (naive model-parallel: works,"
echo "      heavier). For efficiency at high K, launch under accelerate+FSDP."
echo ""

# === 1. INSTALL PACKAGES ===
echo "=== Installing packages ==="
# transformers must know the gemma4 arch (shipped Apr 2026).
# accelerate enables the FSDP launch path for high-K two-block training.
# No peft / bitsandbytes — LoRA/QLoRA cannot install FDM's carrier structure.
pip install -q -U \
    "torch" \
    "transformers>=4.55" \
    "accelerate" \
    "numpy" "tqdm" "scipy" "huggingface_hub" "safetensors" "sentencepiece"

python3 - << 'PYEOF'
import transformers
from transformers.models.auto import configuration_auto as cfg
known = "gemma4" in str(cfg.CONFIG_MAPPING_NAMES)
print(f"transformers {transformers.__version__}; gemma4 arch known: {known}")
if not known:
    print("  WARNING: if model load throws KeyError 'gemma4', run:")
    print("    pip install -U git+https://github.com/huggingface/transformers")
PYEOF

# === 2. SSH KEY FOR GITHUB ===
echo "=== Setting up SSH key ==="
mkdir -p ~/.ssh
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -q
fi
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
echo ""
echo "=========================================="
echo "ADD THIS SSH KEY TO GITHUB:"
echo "https://github.com/settings/keys"
echo "=========================================="
cat ~/.ssh/id_ed25519.pub
echo "=========================================="
echo ""
read -p "Press ENTER after adding key to GitHub..."

# === 3. HUGGINGFACE LOGIN ===
echo "=== Huggingface setup ==="
read -p "Enter your HF token: " HF_TOKEN
export HF_TOKEN
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# === 4. VERIFY GEMMA 4 GATED ACCESS ===
echo "=== Verifying Gemma 4 access ==="
python3 - << PYEOF
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
try:
    hf_hub_download(repo_id="${GEMMA_ID}", filename="config.json", token="${HF_TOKEN}")
    print("  OK: ${GEMMA_ID} accessible.")
except GatedRepoError:
    print("  ERROR: ${GEMMA_ID} is GATED and not yet accepted on your account.")
    print("  Accept here, then re-run: https://huggingface.co/${GEMMA_ID}")
    raise SystemExit(1)
except RepositoryNotFoundError:
    print("  ERROR: repo id ${GEMMA_ID} not found — check the exact name.")
    raise SystemExit(1)
PYEOF

# === 5. GIT CONFIG ===
git config --global user.email "shih.jeanne@gmail.com"
git config --global user.name "jeannemtl"

# === 6. CLONE FDM_GEMMA4 ===
echo "=== Cloning FDM_GEMMA4 ==="
mkdir -p /workspace
cd /workspace
ssh -o StrictHostKeyChecking=no git@github.com || true
if [ ! -d "$WORK" ]; then
    git clone "$REPO_SSH" "$WORK" || {
        echo "  clone failed — if FDM_GEMMA4 is empty/new, create it on GitHub first,"
        echo "  then push eidetic_qwen3_fdm_e2e.py and fdm_gemma4_ksweep.py to it."
    }
fi
mkdir -p "$WORK"
cd "$WORK"

# === 7. CHECK THE TWO SCRIPTS THE SWEEP NEEDS ===
echo "=== Checking sweep scripts ==="
MISSING=0
for f in eidetic_qwen3_fdm_e2e.py fdm_gemma4_ksweep.py; do
    if [ -f "$WORK/$f" ]; then
        echo "  found: $f"
    else
        echo "  MISSING: $f  — push it to FDM_GEMMA4 or scp it to $WORK/"
        MISSING=1
    fi
done
if [ "$MISSING" = "1" ]; then
    echo ""
    echo "  e.g.:  scp -i ~/.ssh/id_ed25519 fdm_gemma4_ksweep.py root@<POD_IP>:$WORK/"
fi

# === 8. PRE-FETCH GEMMA 4 31B WEIGHTS (~62GB) ===
echo "=== Pre-fetching ${GEMMA_ID} (~62GB) ==="
echo "Disk before:"; df -h /workspace | tail -1
python3 - << PYEOF
from huggingface_hub import snapshot_download
import os
os.makedirs("${GEMMA_DIR}", exist_ok=True)
snapshot_download(repo_id="${GEMMA_ID}", local_dir="${GEMMA_DIR}",
                  local_dir_use_symlinks=False, token="${HF_TOKEN}",
                  ignore_patterns=["*.gguf", "*.pth", "original/*"])
print("  OK: weights at ${GEMMA_DIR}")
PYEOF
echo "Disk after:"; df -h /workspace | tail -1

# === 9. VERIFY ===
echo ""
echo "=== Verification ==="
cd "$WORK"
echo "Sweep scripts:"; ls eidetic_qwen3_fdm_e2e.py fdm_gemma4_ksweep.py 2>&1 | head
echo "Gemma weights:"; ls "${GEMMA_DIR}"/*.safetensors 2>&1 | head -3
echo "Stage table (physics check, no GPU):"
FDM_REPO="$WORK" GEMMA_DIR="$GEMMA_DIR" python3 fdm_gemma4_ksweep.py --print 2>&1 | tail -8 || true

# === 10. DONE ===
echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "In tmux:"
echo ""
echo "1. Calibrate the K40 baseline (two-block) and time it:"
echo "   cd $WORK"
echo "   tmux new -s gemma_sweep"
echo "   export GEMMA_DIR=$GEMMA_DIR FDM_REPO=$WORK"
echo "   time python3 fdm_gemma4_ksweep.py --only K40"
echo ""
echo "2. If K40 trains + evals cleanly, run the full ladder:"
echo "   python3 fdm_gemma4_ksweep.py            # 40 -> 320, gated, two-block"
echo ""
echo "   For efficient high-K runs, use FSDP:"
echo "   accelerate launch fdm_gemma4_ksweep.py --only K320"
echo ""
echo "3. Detach: Ctrl+b then d   |   Reattach: tmux attach -t gemma_sweep"
echo ""
echo "The sweep stops at the first K where per-channel recall < 0.99 —"
echo "watch whether that drop lands right as W crosses 1024 (the local window),"
echo "and whether the weakest channels cluster at the high-frequency end."

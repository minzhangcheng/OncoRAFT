#!/bin/bash
# OncoRAFT inference pipeline — full end-to-end (5-fold ensemble).
#
#   Step 1: merge each fold's LoRA into a per-fold full model
#   Step 2: MSK held-out reasoning + score generation (per-fold, no leakage)
#   Step 3: TCGA reasoning + score generation per fold,
#           then aggregate to ensemble (mean score across folds + per-fold
#           rationales preserved; fold-0 rationale used as canonical text).
#
# Required env vars:
#   ONCORAFT_CHECKPOINT_DIR   — root directory containing fold_{0..4} subdirs
#   ONCORAFT_TRAINING_DATA    — same JSONL used for training (for held-out split)
#   ONCORAFT_TEXT_GEN_OUTPUT  — output directory for external text generation
#
# Optional:
#   ONCORAFT_INFERENCE_CLEAN=1   wipe prior text-gen outputs before running
set -e

: "${ONCORAFT_CHECKPOINT_DIR:?must be set}"
: "${ONCORAFT_TRAINING_DATA:?must be set}"
: "${ONCORAFT_TEXT_GEN_OUTPUT:?must be set}"

LOG_FILE="${ONCORAFT_LOG_DIR:-$(pwd)}/inference.log"
TRAINING_DIR="${ONCORAFT_TRAINING_DIR:-$(dirname "$0")}"
CKPT_DIR="$ONCORAFT_CHECKPOINT_DIR"

exec >> "$LOG_FILE" 2>&1

source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pytorch}"

if [ "${ONCORAFT_INFERENCE_CLEAN:-0}" = "1" ]; then
    rm -rf "$CKPT_DIR/msk_text_generation"
    rm -rf "$ONCORAFT_TEXT_GEN_OUTPUT"
fi

echo ""
echo "============================================================"
echo "$(date): OncoRAFT inference pipeline (5-fold ensemble)"
echo "============================================================"
echo "  Checkpoint:   $CKPT_DIR"
echo "  External out: $ONCORAFT_TEXT_GEN_OUTPUT"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# ============================================================
# Step 1: Merge LoRA per fold
# ============================================================
echo ""
echo "============================================================"
echo "$(date): Step 1 - Merge LoRA per fold"
echo "============================================================"
cd "$TRAINING_DIR"
python merge_lora.py 2>&1
echo "$(date): Step 1 complete."

# ============================================================
# Step 2: MSK held-out reasoning + score generation (per fold)
# ============================================================
echo ""
echo "============================================================"
echo "$(date): Step 2 - MSK held-out text generation (per fold)"
echo "============================================================"
cd "$TRAINING_DIR"
accelerate launch --multi_gpu --num_processes 8 \
    msk_text_generation.py --batch_size 4 \
    2>&1
echo "$(date): Step 2 complete."

# ============================================================
# Step 3: External TCGA per-fold + ensemble
# ============================================================
echo ""
echo "============================================================"
echo "$(date): Step 3 - TCGA text generation (per fold + ensemble)"
echo "============================================================"
cd "$TRAINING_DIR"
accelerate launch --multi_gpu --num_processes 8 \
    text_generation_inference.py --dataset tcga --batch_size 4 \
    2>&1
echo "$(date): Step 3 complete."

echo ""
echo "============================================================"
echo "$(date): Inference pipeline finished."
echo "============================================================"
echo "  Merged folds:        $CKPT_DIR/fold_*_merged"
echo "  MSK held-out:        $CKPT_DIR/msk_text_generation"
echo "  External (per fold): $ONCORAFT_TEXT_GEN_OUTPUT/tcga_generated_F{0..4}.jsonl"
echo "  External (ensemble): $ONCORAFT_TEXT_GEN_OUTPUT/tcga_predictions_ensemble.csv"

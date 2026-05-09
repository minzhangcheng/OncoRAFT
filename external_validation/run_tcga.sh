#!/bin/bash
# OncoRAFT TCGA External Validation Pipeline
#
# Prerequisites:
#   - ONCORAFT_VALIDATION_DIR set, containing data/ subtree (see external_validation/config.py)
#   - Trained model checkpoints merged (run training/merge_lora.py)
#
# Steps:
#   1. Build prompts (MSK-compatible format)
#   2. Run inference on GPU server

set -e

echo "=== TCGA External Validation Pipeline ==="

# Step 1: Build prompts
echo "[Step 1] Building TCGA prompts..."
python build_prompts.py

# Step 2: Inference is handled by the main training pipeline
echo "[Step 2] Run inference using the training pipeline:"
echo "    cd ../training"
echo "    bash launch_inference.sh"
echo "  This produces score + reasoning per sample (5-fold ensemble)."

echo ""
echo "=== Pipeline complete ==="

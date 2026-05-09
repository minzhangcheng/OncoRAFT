"""
Configuration for Canonical Fine-Tuning (CFT).

CFT uses the standard causal LM objective with LoRA on the training data,
with a 5-fold patient-level CV split.

All paths are configured via environment variables.
"""
import os
from pathlib import Path

# ============================================================
# Base model
# ============================================================
BASE_MODEL_PATH = os.environ.get("CFT_BASE_MODEL", "")

# ============================================================
# Training data
# ============================================================
REASONING_DATA = os.environ.get("CFT_REASONING_DATA", "")

# ============================================================
# Output directories
# ============================================================
CHECKPOINT_DIR = Path(os.environ.get(
    "CFT_CHECKPOINT_DIR",
    str(Path(__file__).resolve().parent.parent / "checkpoints"),
))

INFERENCE_OUTPUT = Path(os.environ.get(
    "CFT_INFERENCE_OUTPUT",
    str(Path(__file__).resolve().parent.parent / "results"),
))

# ============================================================
# Inference prompt files
# ============================================================
TCGA_PROMPTS = os.environ.get("CFT_TCGA_PROMPTS", "")

# ============================================================
# Training hyperparameters
# ============================================================
TRAIN = {
    "max_length": 2048,
    "output_max_length": 1024,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "num_epochs": 1,
    "n_folds": 5,
    "seed": 42,
    "grad_clip": 1.0,
}

# ============================================================
# LoRA config
# ============================================================
LORA = {
    "r": 16,
    "alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "dropout": 0.05,
}

# ============================================================
# Inference config
# ============================================================
INFERENCE = {
    "batch_size": 4,              # text generation needs more VRAM
    "max_prompt_length": 1536,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.2,
}

FOLD_INDICES = [0, 1, 2, 3, 4]

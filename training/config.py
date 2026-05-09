"""
Centralized configuration for OncoRAFT training and inference.

All paths, model settings, and training hyperparameters are defined here.
Update paths as needed for your environment.
"""
import os
from pathlib import Path

# ============================================================
# Base model
# ============================================================
BASE_MODEL_PATH = os.environ.get("ONCORAFT_BASE_MODEL", "")

# ============================================================
# Training data
# ============================================================
TRAINING_DATA = os.environ.get("ONCORAFT_TRAINING_DATA", "")

# ============================================================
# Output directories
# ============================================================
CHECKPOINT_DIR = Path(os.environ.get("ONCORAFT_CHECKPOINT_DIR", ""))

SCORE_INFERENCE_OUTPUT = Path(os.environ.get("ONCORAFT_SCORE_OUTPUT", ""))

TEXT_GENERATION_OUTPUT = Path(os.environ.get("ONCORAFT_TEXT_GEN_OUTPUT", ""))

# ============================================================
# Inference datasets (prompt files)
# ============================================================
TCGA_PROMPTS = os.environ.get("ONCORAFT_TCGA_PROMPTS", "")

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
        "gate_proj", "up_proj", "down_proj"
    ],
    "dropout": 0.05,
}

# ============================================================
# Score head config
# ============================================================
SCORE_HEAD = {
    "hidden_size": 4096,    # base LLM hidden dim
    "intermediate": 512,
    "dropout": 0.1,
}

# ============================================================
# Inference config
# ============================================================
INFERENCE = {
    "text_gen_batch_size": 4,       # text generation batch size (per GPU)
    "max_prompt_length": 1536,
    "max_new_tokens": 1024,         # max reasoning tokens to generate
    "repetition_penalty": 1.2,
}

FOLD_INDICES = [0, 1, 2, 3, 4]

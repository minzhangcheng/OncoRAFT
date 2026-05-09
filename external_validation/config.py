"""
Centralized configuration for OncoRAFT external validation.
Covers TCGA prompt generation and inference.
Update paths via environment variables.
"""
import os
from pathlib import Path

# ============================================================
# Base directories
# ============================================================
VALIDATION_BASE = Path(os.environ.get("ONCORAFT_VALIDATION_DIR", ""))

# ============================================================
# Shared resources
# ============================================================
DRUGBANK_FILE = Path(os.environ.get("ONCORAFT_DRUGBANK", ""))
MODEL_CHECKPOINT_DIR = Path(os.environ.get("ONCORAFT_CHECKPOINT_DIR", ""))

# Drug normalization module (optional, falls back to uppercase)
DRUG_UTILS_PATH = str(Path(__file__).resolve().parent.parent / "ml_baseline")

# ============================================================
# TCGA Configuration
# ============================================================
TCGA_DATA_DIR = VALIDATION_BASE / "data"
TCGA_CLINICAL_DIR = TCGA_DATA_DIR / "clinical"
TCGA_MUTATION_DIR = Path(os.environ.get("ONCORAFT_TCGA_MUTATION_DIR", ""))
TCGA_TREATMENT_PLAN_DIR = TCGA_DATA_DIR / "treatment_plans"
TCGA_SURVIVAL_FILE = TCGA_DATA_DIR / "survival" / "TCGA-CDR.xlsx"
TCGA_CBIOPORTAL_DIR = TCGA_DATA_DIR / "cbioportal_data"
TCGA_PROMPT_OUTPUT_DIR = VALIDATION_BASE / "outputs" / "prompts"

# ============================================================
# Inference Configuration
# ============================================================
INFERENCE = {
    "batch_size": 8,
    "seed": 42,
    "world_size": 4,     # DDP GPUs
    "port": 12375,       # base port (incremented per fold)
}

# ============================================================
# Instruction template used by the TCGA prompt builder
# ============================================================
INSTRUCTION = """You are an expert AI assistant for precision oncology.

Drug Response Inference Task
Based on the patient's genetic profile, clinical and diagnosis Information, and the drug's mechanism of action, please provide:

1. Output a score representing the likelihood of positive treatment response (a number between 0 and 1).
 - Scores closer to 1 indicate higher likelihood of complete response(positive)
 - Scores closer to 0 indicate higher likelihood of disease progression(negative)
2. Key genetic factors influencing this prediction and why they matter for this treatment plan
3. Critical clinical determinants that significantly impact the efficacy of this treatment"""

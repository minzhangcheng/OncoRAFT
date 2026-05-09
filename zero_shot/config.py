"""
Zero-shot Inference Configuration
==================================
Model registry, prompt variants, and shared settings.

Usage:
    python inference.py --model llama-8b --prompt summary
    python inference.py --model qwen3-235b --prompt simple --thinking
    python inference.py --model medgemma-4b --prompt summary
"""
import os
from pathlib import Path

LLM_DIR = os.environ.get("LLM_DIR", "")

def _llm_path(rel: str) -> str:
    """Resolve a relative model path under $LLM_DIR. Empty if LLM_DIR not set."""
    return os.path.join(LLM_DIR, rel) if LLM_DIR else ""


# ============================================================
# Model Registry
# ============================================================
# Each entry: (model_path, tensor_parallel_size, model_family)
# model_family: "standard" | "qwen3" (has thinking mode) | "gemma"
MODELS = {
    # --- Llama ---
    "llama-8b": {
        "path": _llm_path("llama3.1_8B_instruct"),
        "tp": 4,
        "family": "standard",
    },
    "llama-70b": {
        "path": _llm_path("llama3.1_70B_instruct"),
        "tp": 8,
        "family": "standard",
    },

    # --- Qwen 2.5 ---
    "qwen-7b": {
        "path": _llm_path("Qwen/Qwen2.5-7B-Instruct"),
        "tp": 4,
        "family": "standard",
    },
    "qwen-72b": {
        "path": _llm_path("Qwen/Qwen2.5-72B-Instruct"),
        "tp": 8,
        "family": "standard",
    },

    # --- Qwen 3 (supports thinking mode) ---
    "qwen3-8b": {
        "path": _llm_path("Qwen/Qwen3-8B"),
        "tp": 4,
        "family": "qwen3",
    },
    "qwen3-235b": {
        "path": _llm_path("Qwen/Qwen3-235B-A22B"),
        "tp": 8,
        "family": "qwen3",
    },

    # --- DeepSeek Distill ---
    "distill-llama-8b": {
        "path": _llm_path("DeepSeek-R1-Distill-Llama-8B"),
        "tp": 4,
        "family": "standard",
    },
    "distill-llama-70b": {
        "path": _llm_path("DeepSeek-R1-Distill-Llama-70B"),
        "tp": 8,
        "family": "standard",
    },
    "distill-qwen-7b": {
        "path": _llm_path("DeepSeek-R1-Distill-Qwen-7B"),
        "tp": 4,
        "family": "standard",
    },

    # --- Gemma ---
    "gemma-4b": {
        "path": _llm_path("google/gemma-3-4b-it"),
        "tp": 4,
        "family": "gemma",
    },
    "gemma-27b": {
        "path": _llm_path("google/gemma-3-27b-it"),
        "tp": 8,
        "family": "gemma",
    },

    # --- MedGemma (domain-specific) ---
    "medgemma-4b": {
        "path": _llm_path("google/medgemma-4b-it"),
        "tp": 4,
        "family": "gemma",
    },
    "medgemma-27b": {
        "path": _llm_path("google/medgemma-27b-it"),
        "tp": 8,
        "family": "gemma",
    },
}

# ============================================================
# Prompt Variants
# ============================================================
# "summary"  = detailed prompt asking for score + reasoning + clinical factors
# "simple"   = simplified prompt asking for score + brief explanation
PROMPT_VARIANTS = ["summary", "simple"]

# Dataset paths per prompt variant
#
# Each variant points at three JSONL files (one per random seed) that share
# the same prompt but were generated with different stochastic seeds during
# data construction, so we can report mean ± std across seeds.
#
# Override the directory or file pattern via env vars:
#   ZERO_SHOT_DATA_DIR      — directory containing the JSONL files
#   ZERO_SHOT_SUMMARY_GLOB  — filename template for the "summary" variant
#                             (must contain a "{seed}" placeholder)
#   ZERO_SHOT_SIMPLE_GLOB   — filename template for the "simple" variant
#   ZERO_SHOT_SEEDS         — comma-separated seeds (default: "41,42,43")
DEFAULT_SUMMARY_GLOB = "zero_shot_summary_seed{seed}.jsonl"
DEFAULT_SIMPLE_GLOB = "zero_shot_simple_seed{seed}.jsonl"


def get_datasets(prompt_variant):
    """Return list of dataset paths for a given prompt variant."""
    base = os.environ.get("ZERO_SHOT_DATA_DIR", "")
    seeds = [s.strip() for s in os.environ.get("ZERO_SHOT_SEEDS", "41,42,43").split(",")]
    if prompt_variant == "summary":
        pattern = os.environ.get("ZERO_SHOT_SUMMARY_GLOB", DEFAULT_SUMMARY_GLOB)
    elif prompt_variant == "simple":
        pattern = os.environ.get("ZERO_SHOT_SIMPLE_GLOB", DEFAULT_SIMPLE_GLOB)
    else:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    return [os.path.join(base, pattern.format(seed=s)) for s in seeds]

# ============================================================
# vLLM Engine Settings
# ============================================================
VLLM = {
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9,
    "enforce_eager": True,
    "enable_prefix_caching": True,
}

# ============================================================
# Sampling Parameters
# ============================================================
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 1024,
    "repetition_penalty": 1.2,
}

# ============================================================
# Output
# ============================================================
OUTPUT_DIR = os.environ.get(
    "ZERO_SHOT_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "results"),
)

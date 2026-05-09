#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge per-fold LoRA adapters into full model checkpoints.

For each fold, loads the LoRA adapter, applies it to the base model, and
saves the resulting full model alongside the score-head weights to
fold_X_merged/. This is a one-time CPU step run before inference.

Usage:
  python merge_lora.py
"""
import os
import shutil
import torch
import gc
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import BASE_MODEL_PATH, CHECKPOINT_DIR, FOLD_INDICES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_fold(fold_idx):
    lora_dir = CHECKPOINT_DIR / f"fold_{fold_idx}" / "checkpoint"
    merged_dir = CHECKPOINT_DIR / f"fold_{fold_idx}_merged"

    if not lora_dir.exists():
        logger.warning(f"  Checkpoint not found: {lora_dir}")
        return False

    if merged_dir.exists() and (merged_dir / "score_head.pt").exists():
        logger.info(f"  Already merged: fold_{fold_idx}")
        return True

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    logger.info(f"  Merging fold_{fold_idx}: {lora_dir} -> {merged_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_with_lora = PeftModel.from_pretrained(base_model, str(lora_dir))
    merged_model = model_with_lora.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(merged_dir))

    score_head_src = lora_dir / "score_head.pt"
    if score_head_src.exists():
        shutil.copy2(str(score_head_src), str(merged_dir / "score_head.pt"))

    logger.info(f"  Saved merged model to {merged_dir}")

    del base_model, model_with_lora, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    return True


def main():
    logger.info("=" * 60)
    logger.info("Merging LoRA adapters for 5 folds")
    logger.info("=" * 60)
    for fi in FOLD_INDICES:
        merge_fold(fi)
    logger.info("All folds merged.")


if __name__ == "__main__":
    main()

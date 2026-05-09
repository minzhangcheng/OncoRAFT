#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CFT inference: merge per-fold LoRA, generate text on external prompts, parse
the score, and average across folds (5-fold ensemble).

Steps:
  1. Merge LoRA adapters for each fold -> full models
  2. Generate text on TCGA prompts per fold
  3. Extract scores from generated text via regex
  4. Average per-fold scores -> ensemble score

Usage:
  # Merge only
  python inference_canonical.py --merge_only

  # Inference only (after merging)
  accelerate launch inference_canonical.py --skip_merge

  # Both
  accelerate launch inference_canonical.py
"""
import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
import re
import gc
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object

from config import (
    BASE_MODEL_PATH, CHECKPOINT_DIR, INFERENCE_OUTPUT,
    TCGA_PROMPTS, FOLD_INDICES, TRAIN, INFERENCE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Score extraction from generated text
# ============================================================
def extract_score_from_text(text):
    """Extract numeric score in [0,1] from generated text."""
    if not text:
        return None
    text = text.strip()
    # First token / first line number
    m = re.search(r'(?:^|Score:\s*)(\d+\.?\d*)', text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 1:
            return val
    # Any float in [0,1]
    for m in re.finditer(r'(\d+\.\d+)', text):
        val = float(m.group(1))
        if 0 <= val <= 1:
            return val
    return None


# ============================================================
# Prompt Dataset (left-padded for generation)
# ============================================================
class PromptDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length or INFERENCE["max_prompt_length"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"] + "\n\n"
        input_data = sample["input"] + "\n\nAnswer:\n"
        full_prompt = instruction + input_data

        patient_id = ""
        m = re.search(r"Patient ID: ([^\n]+)", input_data)
        if m:
            patient_id = m.group(1)

        return {
            "prompt": full_prompt,
            "patient_id": patient_id,
            "sample_idx": idx,
        }


def collate_fn_gen(batch, tokenizer, max_length):
    """Collate with left padding for generation."""
    prompts = [b["prompt"] for b in batch]

    tokenizer.padding_side = "left"
    encodings = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    )
    tokenizer.padding_side = "right"

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "patient_ids": [b["patient_id"] for b in batch],
        "sample_indices": [b["sample_idx"] for b in batch],
    }


# ============================================================
# Merge functions
# ============================================================
def merge_fold(fold_idx, ckpt_base):
    lora_dir = ckpt_base / f"fold_{fold_idx}" / "checkpoint"
    merged_dir = ckpt_base / f"fold_{fold_idx}_merged"

    if not lora_dir.exists():
        logger.warning(f"  Checkpoint not found: {lora_dir}")
        return False

    if merged_dir.exists() and (merged_dir / "config.json").exists():
        logger.info(f"  Already merged: fold_{fold_idx}")
        return True

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    logger.info(f"  Merging fold_{fold_idx}: {lora_dir} -> {merged_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_with_lora = PeftModel.from_pretrained(base_model, str(lora_dir))
    merged_model = model_with_lora.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(merged_dir))
    logger.info(f"  Saved merged model to {merged_dir}")

    del base_model, model_with_lora, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    return True


# ============================================================
# Text generation inference
# ============================================================
def run_inference(accelerator, merged_dir, prompt_file, max_new_tokens, batch_size, desc="inference"):
    """Load model, generate text on prompts, extract scores."""
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        logger.info(f"    Model loaded from {merged_dir}")

    # Load prompts
    samples = []
    with open(prompt_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    dataset = PromptDataset(samples, tokenizer)

    from functools import partial
    collate = partial(collate_fn_gen, tokenizer=tokenizer,
                      max_length=INFERENCE["max_prompt_length"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)
    dataloader = accelerator.prepare(dataloader)

    local_results = []
    for batch in tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(len(batch["patient_ids"])):
            generated = gen_ids[i][prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            score = extract_score_from_text(text)

            local_results.append({
                "sample_id": batch["sample_indices"][i],
                "patient_id": batch["patient_ids"][i],
                "predicted_score": score if score is not None else 0.5,
                "generated_text": text[:300],
                "score_extracted": score is not None,
            })

    all_results = gather_object(local_results)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        flat = []
        for r in all_results:
            if isinstance(r, list):
                flat.extend(r)
            else:
                flat.append(r)
        df = pd.DataFrame(flat)
        df = df.drop_duplicates(subset="sample_id").sort_values("sample_id").reset_index(drop=True)

        n_extracted = df['score_extracted'].sum()
        logger.info(f"    {desc}: {len(df)} predictions, {n_extracted}/{len(df)} scores extracted")
        return df
    return None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CFT: merge + generate + extract score")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=INFERENCE["batch_size"])
    args = parser.parse_args()

    ckpt_base = CHECKPOINT_DIR
    output_dir = INFERENCE_OUTPUT
    max_new_tokens = INFERENCE["max_new_tokens"]

    # ---- Step 1: Merge per-fold LoRA ----
    if not args.skip_merge:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            logger.info("=" * 60)
            logger.info("Merging LoRA for 5 folds")
            logger.info("=" * 60)

            for fi in FOLD_INDICES:
                merge_fold(fi, ckpt_base)

        if args.merge_only:
            return

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ---- Step 2: Per-fold inference + ensemble ----
    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    ext_datasets = {
        "tcga": TCGA_PROMPTS,
    }

    model_list = []
    for fi in FOLD_INDICES:
        merged_dir = ckpt_base / f"fold_{fi}_merged"
        if merged_dir.exists():
            model_list.append((merged_dir, f"F{fi}"))

    if accelerator.is_main_process:
        logger.info(f"\nModels: {[label for _, label in model_list]}")
        logger.info(f"max_new_tokens: {max_new_tokens}")

    for ds_name, prompt_file in ext_datasets.items():
        if not os.path.exists(prompt_file):
            if accelerator.is_main_process:
                logger.warning(f"Prompt file not found: {prompt_file}, skipping {ds_name}")
            continue

        if accelerator.is_main_process:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Inference on {ds_name}")
            logger.info("=" * 60)

        ds_out = output_dir / ds_name
        if accelerator.is_main_process:
            os.makedirs(ds_out, exist_ok=True)

        epoch_dfs = []
        valid_labels = []

        for merged_dir, label in model_list:
            if accelerator.is_main_process:
                logger.info(f"\n  --- {label} on {ds_name} ---")

            df = run_inference(
                accelerator, merged_dir, prompt_file,
                max_new_tokens=max_new_tokens,
                batch_size=args.batch_size,
                desc=f"{label} {ds_name}",
            )

            if accelerator.is_main_process and df is not None:
                df.to_csv(ds_out / f"{label}_predictions.csv", index=False)
                epoch_dfs.append(df.rename(columns={"predicted_score": f"score_{label}"}))
                valid_labels.append(label)

            accelerator.wait_for_everyone()

        # Combine and compute ensemble
        if accelerator.is_main_process and len(epoch_dfs) > 0:
            combined = epoch_dfs[0][["sample_id", "patient_id", f"score_{valid_labels[0]}"]].copy()
            for edf, label in zip(epoch_dfs[1:], valid_labels[1:]):
                combined = combined.merge(
                    edf[["sample_id", f"score_{label}"]],
                    on="sample_id", how="left",
                )

            fold_cols = [f"score_F{fi}" for fi in FOLD_INDICES if f"score_F{fi}" in combined.columns]
            if fold_cols:
                combined["score_Ensemble"] = combined[fold_cols].mean(axis=1)
                combined["score_std"] = combined[fold_cols].std(axis=1)

            out_file = ds_out / f"{ds_name}_predictions.csv"
            combined.to_csv(out_file, index=False)
            logger.info(f"\n  Saved: {out_file} ({combined.shape})")

            for col in [c for c in combined.columns if c.startswith("score_")]:
                logger.info(f"    {col}: mean={combined[col].mean():.4f}")

    if accelerator.is_main_process:
        logger.info("\nAll done! Canonical fine-tuning inference complete.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    main()

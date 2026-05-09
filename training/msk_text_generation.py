#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSK Held-Out Text Generation: per-fold score + reasoning generation.

For each fold, loads the fold's merged model and generates text for the
held-out validation samples (same split as training). This avoids data leakage.

Output per fold:
  - fold_{i}_generated.jsonl  (score + reasoning text)
  - fold_{i}_scores.csv       (scores only)
Combined:
  - msk_all_generated.jsonl
  - msk_all_scores.csv

Usage:
  # All folds
  accelerate launch --num_processes 8 msk_text_generation.py

  # Single fold
  accelerate launch --num_processes 8 msk_text_generation.py --fold 0

  # Custom checkpoint dir
  accelerate launch --num_processes 8 msk_text_generation.py \
    --checkpoint_dir /path/to/checkpoints
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object
from sklearn.model_selection import KFold
from collections import defaultdict
import gc
import re
import argparse
import logging

from config import (
    BASE_MODEL_PATH, CHECKPOINT_DIR, TRAINING_DATA, TRAIN, INFERENCE
)
from model import MultitaskScoreModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Patient-level CV split (must match training exactly)
# ============================================================
def group_samples_by_patient_id(samples):
    patient_groups = defaultdict(list)
    for sample in samples:
        patient_id = None
        if "input" in sample:
            match = re.search(r"Patient ID: ([^\n]+)", sample["input"])
            if match:
                patient_id = match.group(1)
        patient_groups[patient_id].append(sample)
    return patient_groups


def get_fold_val_samples(samples, fold_idx, n_folds=None):
    """Reproduce the exact same fold split as training."""
    n_folds = n_folds or TRAIN["n_folds"]
    patient_groups = group_samples_by_patient_id(samples)
    patient_ids = list(patient_groups.keys())

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=TRAIN["seed"])

    for fi, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        if fi == fold_idx:
            val_pids = [patient_ids[i] for i in val_idx]
            val_samples = []
            for pid in val_pids:
                val_samples.extend(patient_groups[pid])
            return val_samples

    raise ValueError(f"Fold {fold_idx} not found")


# ============================================================
# Dataset — left-padded for generation
# ============================================================
class GenerationDataset(Dataset):
    def __init__(self, samples, tokenizer, max_prompt_length=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length or INFERENCE["max_prompt_length"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"] + "\n\n"
        input_data = sample["input"] + "\n\nAnswer:\n"
        full_prompt = instruction + input_data

        patient_id = ""
        match = re.search(r"Patient ID: ([^\n]+)", input_data)
        if match:
            patient_id = match.group(1)

        # Parse true score from output
        output = sample.get("output", "")
        first_line = output.strip().split('\n')[0].strip()
        score_match = re.match(r'(?:Score:\s*)?(\d+\.?\d*)', first_line)
        true_score = float(score_match.group(1)) if score_match else 0.5
        true_score = min(1.0, max(0.0, true_score))
        true_binary = 1 if true_score >= 0.5 else 0

        encoding = self.tokenizer(
            full_prompt, add_special_tokens=False, truncation=True,
            max_length=self.max_prompt_length
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "prompt_length": len(encoding["input_ids"]),
            "score_position": len(encoding["input_ids"]) - 1,
            "patient_id": patient_id,
            "sample_idx": idx,
            "true_score": true_score,
            "true_binary": true_binary,
        }


def left_pad_collate(batch, pad_token_id):
    max_len = max(b["prompt_length"] for b in batch)

    padded_input_ids = []
    padded_attention_mask = []
    score_positions = []
    patient_ids = []
    sample_indices = []
    prompt_lengths = []
    true_scores = []
    true_binaries = []

    for b in batch:
        pad_len = max_len - b["prompt_length"]
        ids = [pad_token_id] * pad_len + b["input_ids"]
        mask = [0] * pad_len + b["attention_mask"]
        sp = b["score_position"] + pad_len

        padded_input_ids.append(ids)
        padded_attention_mask.append(mask)
        score_positions.append(sp)
        patient_ids.append(b["patient_id"])
        sample_indices.append(b["sample_idx"])
        prompt_lengths.append(b["prompt_length"])
        true_scores.append(b["true_score"])
        true_binaries.append(b["true_binary"])

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(padded_attention_mask),
        "score_positions": torch.tensor(score_positions),
        "patient_ids": patient_ids,
        "sample_indices": sample_indices,
        "prompt_lengths": prompt_lengths,
        "true_scores": true_scores,
        "true_binaries": true_binaries,
    }


# ============================================================
# Load model
# ============================================================
def load_model(model_dir, device):
    logger.info(f"Loading model from {model_dir}")

    lm_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    lm_model.eval()

    model = MultitaskScoreModel(lm_model)
    model.score_head = model.score_head.to(torch.bfloat16)

    score_head_path = Path(model_dir) / "score_head.pt"
    if score_head_path.exists():
        state_dict = torch.load(str(score_head_path), map_location=device)
        model.score_head.load_state_dict(state_dict)
        logger.info("  Loaded score_head.pt")

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


# ============================================================
# Per-fold text generation
# ============================================================
def run_fold_generation(accelerator, model_dir, val_samples, fold_idx, output_dir,
                        batch_size=None, max_new_tokens=None):
    batch_size = batch_size or INFERENCE["text_gen_batch_size"]
    max_new_tokens = max_new_tokens or INFERENCE["max_new_tokens"]

    device = accelerator.device
    model, tokenizer = load_model(model_dir, device)

    if accelerator.is_main_process:
        logger.info(f"  Fold {fold_idx}: {len(val_samples)} val samples, "
                     f"{accelerator.num_processes} GPUs")

    dataset = GenerationDataset(val_samples, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: left_pad_collate(b, pad_token_id),
        num_workers=2, pin_memory=True
    )
    dataloader = accelerator.prepare(dataloader)

    local_results = []

    for batch in tqdm(dataloader, desc=f"Fold {fold_idx} generating",
                      disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        score_positions = batch["score_positions"].to(device)
        prompt_lengths = batch["prompt_lengths"]
        batch_size_actual = input_ids.size(0)

        scores = model.predict_score(input_ids, attention_mask, score_positions)

        new_input_ids_list = []
        new_attention_mask_list = []
        for i in range(batch_size_actual):
            score_val = round(float(scores[i]), 2)
            score_prefix = f"{score_val}\nReasoning:\n"
            prefix_tokens = tokenizer.encode(score_prefix, add_special_tokens=False)

            orig_len = prompt_lengths[i]
            pad_offset = input_ids.size(1) - orig_len
            orig_ids = input_ids[i, pad_offset:].tolist()

            new_ids = orig_ids + prefix_tokens
            new_mask = [1] * len(new_ids)
            new_input_ids_list.append(new_ids)
            new_attention_mask_list.append(new_mask)

        max_new_len = max(len(ids) for ids in new_input_ids_list)
        for i in range(batch_size_actual):
            pad_len = max_new_len - len(new_input_ids_list[i])
            new_input_ids_list[i] = [tokenizer.pad_token_id] * pad_len + new_input_ids_list[i]
            new_attention_mask_list[i] = [0] * pad_len + new_attention_mask_list[i]

        gen_input_ids = torch.tensor(new_input_ids_list, device=device)
        gen_attention_mask = torch.tensor(new_attention_mask_list, device=device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=INFERENCE["repetition_penalty"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(batch_size_actual):
            gen_start = max_new_len
            new_tokens = generated_ids[i][gen_start:]
            reasoning = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            score_val = round(float(scores[i]), 2)
            full_text = f"{score_val}\nReasoning:\n{reasoning}"

            local_results.append({
                "sample_id": batch["sample_indices"][i],
                "patient_id": batch["patient_ids"][i],
                "predicted_score": float(scores[i]),
                "true_score": batch["true_scores"][i],
                "true_binary": batch["true_binaries"][i],
                "generated_text": full_text,
                "fold": fold_idx,
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

        # Save per-fold JSONL
        jsonl_path = output_dir / f"fold_{fold_idx}_generated.jsonl"
        with open(jsonl_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps({
                    "sample_id": int(row["sample_id"]),
                    "patient_id": row["patient_id"],
                    "predicted_score": row["predicted_score"],
                    "true_score": row["true_score"],
                    "true_binary": int(row["true_binary"]),
                    "generated_text": row["generated_text"],
                    "fold": fold_idx,
                }, ensure_ascii=False) + "\n")

        # Save per-fold scores CSV
        csv_path = output_dir / f"fold_{fold_idx}_scores.csv"
        df[["sample_id", "patient_id", "predicted_score", "true_score",
            "true_binary", "fold"]].to_csv(csv_path, index=False)

        logger.info(f"  Fold {fold_idx}: {len(df)} samples saved")
        logger.info(f"    Score mean: {df['predicted_score'].mean():.4f}")
        logger.info(f"    Avg text length: {df['generated_text'].str.len().mean():.0f} chars")

        return df
    return None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MSK held-out text generation (per-fold)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (default: all folds)")
    parser.add_argument("--batch_size", type=int, default=INFERENCE["text_gen_batch_size"])
    parser.add_argument("--max_new_tokens", type=int, default=INFERENCE["max_new_tokens"])
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR
    data_path = args.data_path or TRAINING_DATA
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "msk_text_generation"

    folds_to_run = [args.fold] if args.fold is not None else list(range(TRAIN["n_folds"]))

    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("MSK Held-Out Text Generation (Per-Fold)")
        logger.info(f"  Checkpoints: {checkpoint_dir}")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Folds: {folds_to_run}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max new tokens: {args.max_new_tokens}")
        logger.info(f"  GPUs: {accelerator.num_processes}")
        logger.info("=" * 60)
        os.makedirs(output_dir, exist_ok=True)

    # Load all samples
    if accelerator.is_main_process:
        logger.info(f"Loading data from {data_path}...")

    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(samples)} total samples")

    # Process each fold
    all_fold_dfs = []

    for fold_idx in folds_to_run:
        merged_dir = checkpoint_dir / f"fold_{fold_idx}_merged"
        if not merged_dir.exists():
            if accelerator.is_main_process:
                logger.warning(f"Fold {fold_idx} merged model not found: {merged_dir}, skipping")
            continue

        if accelerator.is_main_process:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Fold {fold_idx}")
            logger.info(f"  Model: {merged_dir}")
            logger.info("=" * 60)

        val_samples = get_fold_val_samples(samples, fold_idx)

        if accelerator.is_main_process:
            logger.info(f"  Val samples: {len(val_samples)}")

        df = run_fold_generation(
            accelerator=accelerator,
            model_dir=merged_dir,
            val_samples=val_samples,
            fold_idx=fold_idx,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        if df is not None:
            all_fold_dfs.append(df)

        accelerator.wait_for_everyone()

    # Combine all folds
    if accelerator.is_main_process and all_fold_dfs:
        combined = pd.concat(all_fold_dfs, ignore_index=True)
        combined = combined.sort_values(["fold", "sample_id"]).reset_index(drop=True)

        # Save combined JSONL
        jsonl_path = output_dir / "msk_all_generated.jsonl"
        with open(jsonl_path, 'w') as f:
            for _, row in combined.iterrows():
                f.write(json.dumps({
                    "sample_id": int(row["sample_id"]),
                    "patient_id": row["patient_id"],
                    "predicted_score": row["predicted_score"],
                    "true_score": row["true_score"],
                    "true_binary": int(row["true_binary"]),
                    "generated_text": row["generated_text"],
                    "fold": int(row["fold"]),
                }, ensure_ascii=False) + "\n")

        # Save combined scores CSV
        csv_path = output_dir / "msk_all_scores.csv"
        combined[["sample_id", "patient_id", "predicted_score", "true_score",
                  "true_binary", "fold"]].to_csv(csv_path, index=False)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Combined: {len(combined)} samples from {len(all_fold_dfs)} folds")
        logger.info(f"  JSONL: {jsonl_path}")
        logger.info(f"  CSV:   {csv_path}")
        logger.info(f"  Score mean: {combined['predicted_score'].mean():.4f}")
        logger.info(f"  Avg text len: {combined['generated_text'].str.len().mean():.0f} chars")

        # Show examples
        logger.info("\n  === Sample outputs ===")
        for _, row in combined.head(3).iterrows():
            logger.info(f"  Patient: {row['patient_id']}, Fold: {int(row['fold'])}")
            logger.info(f"  Score: {row['predicted_score']:.4f} (true: {row['true_score']:.2f})")
            text_preview = row['generated_text'][:400].replace('\n', ' | ')
            logger.info(f"  Text: {text_preview}...")
            logger.info("  ---")

        logger.info("\nAll done!")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    main()

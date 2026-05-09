#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External-validation inference: per-fold score + reasoning generation,
followed by ensemble aggregation across folds.

Outputs:
  {ds}_predictions_ensemble.csv   — per-sample mean score (and std) across folds
  {ds}_generated_ensemble.jsonl   — ensemble score + representative rationale
  {ds}_generated_F{i}.jsonl       — per-fold rationale + score
  {ds}_scores_F{i}.csv            — per-fold scores

Usage (multi-GPU):
  accelerate launch --num_processes 8 \\
      text_generation_inference.py --dataset tcga --batch_size 4
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
import gc
import re
import argparse
import logging

from config import (
    CHECKPOINT_DIR, TEXT_GENERATION_OUTPUT,
    TCGA_PROMPTS, INFERENCE
)
from model import MultitaskScoreModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Dataset — left-padded for causal LM generation
# ============================================================
class GenerationDataset(Dataset):
    """Left-padded dataset for text generation."""

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
        }


def left_pad_collate(batch, pad_token_id):
    """Left-pad batch for causal LM generation."""
    max_len = max(b["prompt_length"] for b in batch)

    padded_input_ids = []
    padded_attention_mask = []
    score_positions = []
    patient_ids = []
    sample_indices = []
    prompt_lengths = []

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

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(padded_attention_mask),
        "score_positions": torch.tensor(score_positions),
        "patient_ids": patient_ids,
        "sample_indices": sample_indices,
        "prompt_lengths": prompt_lengths,
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
# Inference
# ============================================================
def run_text_generation(accelerator, model_dir, prompt_file, output_dir,
                        batch_size=None, max_new_tokens=None, ds_name="dataset"):
    batch_size = batch_size or INFERENCE["text_gen_batch_size"]
    max_new_tokens = max_new_tokens or INFERENCE["max_new_tokens"]

    device = accelerator.device
    model, tokenizer = load_model(model_dir, device)

    if accelerator.is_main_process:
        logger.info(f"  Model loaded on {accelerator.num_processes} GPUs")
        os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    samples = []
    with open(prompt_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if accelerator.is_main_process:
        logger.info(f"  {ds_name}: {len(samples)} samples")

    dataset = GenerationDataset(samples, tokenizer)

    pad_token_id = tokenizer.pad_token_id
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: left_pad_collate(b, pad_token_id),
        num_workers=2, pin_memory=True
    )
    dataloader = accelerator.prepare(dataloader)

    local_results = []

    for batch in tqdm(dataloader, desc=f"Generating {ds_name}",
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

            # Extract original tokens (without left padding)
            orig_len = prompt_lengths[i]
            pad_offset = input_ids.size(1) - orig_len
            orig_ids = input_ids[i, pad_offset:].tolist()

            new_ids = orig_ids + prefix_tokens
            new_mask = [1] * len(new_ids)
            new_input_ids_list.append(new_ids)
            new_attention_mask_list.append(new_mask)

        # Left-pad the augmented prompts
        max_new_len = max(len(ids) for ids in new_input_ids_list)
        for i in range(batch_size_actual):
            pad_len = max_new_len - len(new_input_ids_list[i])
            new_input_ids_list[i] = [tokenizer.pad_token_id] * pad_len + new_input_ids_list[i]
            new_attention_mask_list[i] = [0] * pad_len + new_attention_mask_list[i]

        gen_input_ids = torch.tensor(new_input_ids_list, device=device)
        gen_attention_mask = torch.tensor(new_attention_mask_list, device=device)

        # Generate reasoning
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

        # Decode only newly generated tokens
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
                "generated_text": full_text,
            })

    # Gather from all GPUs
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

        # Save JSONL (with generated text)
        jsonl_path = output_dir / f"{ds_name}_generated.jsonl"
        with open(jsonl_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps({
                    "sample_id": int(row["sample_id"]),
                    "patient_id": row["patient_id"],
                    "predicted_score": row["predicted_score"],
                    "score_head_raw": row["predicted_score"],
                    "generated_text": row["generated_text"],
                }, ensure_ascii=False) + "\n")

        # Save scores CSV (for survival analysis)
        csv_path = output_dir / f"{ds_name}_scores.csv"
        df[["sample_id", "patient_id", "predicted_score"]].to_csv(csv_path, index=False)

        logger.info(f"  Saved {len(df)} results:")
        logger.info(f"    JSONL: {jsonl_path}")
        logger.info(f"    CSV:   {csv_path}")
        logger.info(f"    Score mean: {df['predicted_score'].mean():.4f}")
        logger.info(f"    Avg text length: {df['generated_text'].str.len().mean():.0f} chars")

        # Show examples
        logger.info("\n  === Sample outputs ===")
        for _, row in df.head(3).iterrows():
            logger.info(f"  Patient: {row['patient_id']}")
            logger.info(f"  Score: {row['predicted_score']:.4f}")
            text_preview = row['generated_text'][:400].replace('\n', ' | ')
            logger.info(f"  Text: {text_preview}...")
            logger.info("  ---")


# ============================================================
# Main
# ============================================================
def aggregate_ensemble(output_dir: Path, ds_name: str, fold_indices):
    """Combine per-fold {ds_name}_generated_F{i}.jsonl into an ensemble JSONL/CSV.

    - ensemble_score = mean of per-fold scores
    - generated_text = first fold's text (representative; full per-fold text
      retained in the per-fold JSONL files)
    """
    fold_dfs = {}
    for fi in fold_indices:
        fp = output_dir / f"{ds_name}_generated_F{fi}.jsonl"
        if not fp.exists():
            continue
        rows = [json.loads(line) for line in open(fp) if line.strip()]
        fdf = pd.DataFrame(rows)
        fold_dfs[fi] = fdf

    if not fold_dfs:
        logger.warning(f"  No per-fold files found for {ds_name}; skipping ensemble")
        return

    folds = sorted(fold_dfs)
    base = fold_dfs[folds[0]][["sample_id", "patient_id", "generated_text"]].copy()
    base = base.rename(columns={"generated_text": "generated_text_F0"})

    score_cols = []
    for fi in folds:
        fdf = fold_dfs[fi][["sample_id", "predicted_score"]].rename(
            columns={"predicted_score": f"score_F{fi}"}
        )
        base = base.merge(fdf, on="sample_id", how="left")
        score_cols.append(f"score_F{fi}")

    base["ensemble_score"] = base[score_cols].mean(axis=1)
    base["score_std"] = base[score_cols].std(axis=1)

    out_jsonl = output_dir / f"{ds_name}_generated_ensemble.jsonl"
    with open(out_jsonl, "w") as f:
        for _, row in base.iterrows():
            f.write(json.dumps({
                "sample_id": int(row["sample_id"]),
                "patient_id": row["patient_id"],
                "ensemble_score": float(row["ensemble_score"]),
                "score_std": float(row["score_std"]),
                "generated_text": row["generated_text_F0"],
            }, ensure_ascii=False) + "\n")

    out_csv = output_dir / f"{ds_name}_predictions_ensemble.csv"
    base[["sample_id", "patient_id", "ensemble_score", "score_std"] + score_cols].to_csv(
        out_csv, index=False
    )
    logger.info(f"  Ensemble: wrote {len(base)} rows -> {out_jsonl.name}, {out_csv.name}")


def main():
    parser = argparse.ArgumentParser(description="Per-fold text generation + ensemble score")
    parser.add_argument("--dataset", type=str, default="tcga",
                        choices=["tcga"])
    parser.add_argument("--batch_size", type=int, default=INFERENCE["text_gen_batch_size"])
    parser.add_argument("--max_new_tokens", type=int, default=INFERENCE["max_new_tokens"])
    parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                        help="Comma-separated fold indices to ensemble over")
    args = parser.parse_args()

    fold_indices = [int(f) for f in args.folds.split(",") if f.strip()]
    fold_dirs = [(fi, CHECKPOINT_DIR / f"fold_{fi}_merged") for fi in fold_indices]
    missing = [fi for fi, d in fold_dirs if not d.exists()]
    if missing:
        logger.error(f"Missing fold-merged dirs: {missing}")
        logger.error("Run merge_lora.py first to merge LoRA per fold.")
        return

    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Text Generation + Score (per-fold ensemble)")
        logger.info(f"  Folds:          {fold_indices}")
        logger.info(f"  Batch size:     {args.batch_size}")
        logger.info(f"  Max new tokens: {args.max_new_tokens}")
        logger.info(f"  GPUs:           {accelerator.num_processes}")
        logger.info("=" * 60)
        os.makedirs(TEXT_GENERATION_OUTPUT, exist_ok=True)

    datasets = {"tcga": TCGA_PROMPTS}

    for ds_name, prompt_file in datasets.items():
        if accelerator.is_main_process:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Dataset: {ds_name}")
            logger.info("=" * 60)

        for fi, model_dir in fold_dirs:
            if accelerator.is_main_process:
                logger.info(f"\n  --- Fold {fi}: {model_dir.name} ---")
            run_text_generation(
                accelerator=accelerator,
                model_dir=model_dir,
                prompt_file=prompt_file,
                output_dir=TEXT_GENERATION_OUTPUT,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                ds_name=f"{ds_name}_F{fi}",
            )
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            aggregate_ensemble(TEXT_GENERATION_OUTPUT, ds_name, fold_indices)

    if accelerator.is_main_process:
        logger.info("\nAll done! Per-fold + ensemble outputs written.")
        logger.info(f"Results: {TEXT_GENERATION_OUTPUT}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    main()

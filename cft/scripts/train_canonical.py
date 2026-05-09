#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Canonical Fine-Tuning (CFT).

Standard causal LM fine-tuning with LoRA over the training JSONL, using a
5-fold patient-level CV split. At inference time the score is parsed from
the model's generated text. Hyperparameters and module shapes are
configured in config.py.

Usage:
  for f in 0 1 2 3 4; do
    accelerate launch train_canonical.py --fold $f
  done
"""
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import re
import gc
import logging
import argparse
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
)
from sklearn.model_selection import KFold

from config import (
    BASE_MODEL_PATH, REASONING_DATA,
    CHECKPOINT_DIR, TRAIN, LORA,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(TRAIN["seed"])


# ============================================================
# Dataset
# ============================================================
class CanonicalDataset(Dataset):
    """Instruction-following dataset for causal LM fine-tuning.

    Labels are masked for the prompt portion (-100) so the model only
    learns to generate the output.
    """

    def __init__(self, samples, tokenizer, max_length=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length or TRAIN["max_length"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"] + "\n\n"
        input_data = sample["input"] + "\n\nAnswer:\n"
        output = sample["output"]

        patient_id = None
        match = re.search(r"Patient ID: ([^\n]+)", input_data)
        if match:
            patient_id = match.group(1)

        # Parse ground-truth score for evaluation (not used in loss)
        first_line = output.strip().split('\n')[0].strip()
        score_match = re.match(r'(?:Score:\s*)?(\d+\.?\d*)', first_line)
        true_score = min(1.0, max(0.0, float(score_match.group(1)))) if score_match else 0.5
        true_binary = 1.0 if true_score >= 0.5 else 0.0

        output_max_length = TRAIN["output_max_length"]
        prompt_max_length = self.max_length - output_max_length
        full_prompt = instruction + input_data

        prompt_enc = self.tokenizer(
            full_prompt, add_special_tokens=False, truncation=True,
            max_length=prompt_max_length,
        )
        output_enc = self.tokenizer(
            output, add_special_tokens=False, truncation=True,
            max_length=output_max_length,
        )

        input_ids = prompt_enc["input_ids"] + output_enc["input_ids"]
        attention_mask = prompt_enc["attention_mask"] + output_enc["attention_mask"]
        prompt_length = len(prompt_enc["input_ids"])

        # Pad or truncate
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Labels: mask prompt, only compute loss on output tokens
        labels = [-100] * prompt_length + output_enc["input_ids"]
        if len(labels) < self.max_length:
            labels += [-100] * (self.max_length - len(labels))
        else:
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "true_score": true_score,
            "true_binary": true_binary,
            "patient_id": patient_id,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "true_scores": torch.tensor([b["true_score"] for b in batch]),
        "true_binary": torch.tensor([b["true_binary"] for b in batch]),
        "patient_ids": [b["patient_id"] for b in batch],
    }


def load_data(data_path):
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


# ============================================================
# Patient-level CV split (identical logic to multi-task)
# ============================================================
def group_samples_by_patient_id(samples):
    groups = defaultdict(list)
    for sample in samples:
        pid = None
        if "input" in sample:
            m = re.search(r"Patient ID: ([^\n]+)", sample["input"])
            if m:
                pid = m.group(1)
        groups[pid].append(sample)
    return groups


def create_cv_folds(samples, n_folds=None):
    n_folds = n_folds or TRAIN["n_folds"]
    patient_groups = group_samples_by_patient_id(samples)
    patient_ids = list(patient_groups.keys())

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=TRAIN["seed"])

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids = [patient_ids[i] for i in val_idx]

        train_samples = [s for pid in train_pids for s in patient_groups[pid]]
        val_samples = [s for pid in val_pids for s in patient_groups[pid]]

        folds.append((train_samples, val_samples))
        logger.info(f"  Fold {fold_idx}: train={len(train_samples)} ({len(train_pids)} patients), "
                     f"val={len(val_samples)} ({len(val_pids)} patients)")

    return folds


# ============================================================
# Held-out evaluation via text generation
# ============================================================
def extract_score_from_text(text):
    """Extract numeric score from model-generated text."""
    if not text:
        return None
    text = text.strip()
    # Try first line / first number in [0,1]
    m = re.search(r'(?:^|Score:\s*)(\d+\.?\d*)', text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 1:
            return val
    # Fallback: any float in [0,1]
    for m in re.finditer(r'(\d+\.\d+)', text):
        val = float(m.group(1))
        if 0 <= val <= 1:
            return val
    return None


class HeldOutDataset(Dataset):
    """Dataset for held-out validation samples (used by multi-GPU generation)."""

    def __init__(self, val_samples):
        self.samples = val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"] + "\n\n"
        input_data = sample["input"] + "\n\nAnswer:\n"
        prompt = instruction + input_data

        pid = None
        m = re.search(r"Patient ID: ([^\n]+)", sample.get("input", ""))
        if m:
            pid = m.group(1)

        first_line = sample["output"].strip().split('\n')[0].strip()
        sm = re.match(r'(?:Score:\s*)?(\d+\.?\d*)', first_line)
        ts = min(1.0, max(0.0, float(sm.group(1)))) if sm else 0.5

        return {
            "prompt": prompt,
            "patient_id": pid or "",
            "true_score": ts,
            "true_binary": 1.0 if ts >= 0.5 else 0.0,
            "sample_idx": idx,
        }


def held_out_collate_fn(batch, tokenizer, max_length):
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
        "true_scores": [b["true_score"] for b in batch],
        "true_binaries": [b["true_binary"] for b in batch],
        "sample_indices": [b["sample_idx"] for b in batch],
    }


def predict_held_out(model, tokenizer, val_samples, accelerator, max_new_tokens=64):
    """Multi-GPU held-out prediction via DataLoader + accelerator."""
    from functools import partial
    from accelerate.utils import gather_object

    model.eval()

    dataset = HeldOutDataset(val_samples)
    collate = partial(held_out_collate_fn, tokenizer=tokenizer,
                      max_length=TRAIN["max_length"] - max_new_tokens)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)
    dataloader = accelerator.prepare(dataloader)

    local_results = []
    for batch in tqdm(dataloader, desc="Generating held-out predictions",
                      disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            gen_ids = accelerator.unwrap_model(model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(input_ids.shape[0]):
            generated = gen_ids[i][prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            pred_score = extract_score_from_text(text)
            if pred_score is None:
                pred_score = 0.5

            local_results.append({
                "sample_idx": batch["sample_indices"][i],
                "patient_id": batch["patient_ids"][i],
                "true_score": batch["true_scores"][i],
                "predicted_score": pred_score,
                "true_binary": int(batch["true_binaries"][i]),
                "generated_text": text[:200],
            })

    # Gather results from all GPUs
    all_results = gather_object(local_results)

    if accelerator.is_main_process:
        flat = []
        for r in all_results:
            if isinstance(r, list):
                flat.extend(r)
            else:
                flat.append(r)
        df = pd.DataFrame(flat)
        df = df.drop_duplicates(subset="sample_idx").sort_values("sample_idx").reset_index(drop=True)
        df = df[df["patient_id"].astype(str).str.strip() != ""].reset_index(drop=True)
        return df
    return None


def compute_metrics(df):
    pred_binary = (df['predicted_score'] >= 0.5).astype(int)
    true_binary = df['true_binary'].values
    pred_scores = df['predicted_score'].values

    return {
        'n': len(df),
        'auroc': roc_auc_score(true_binary, pred_scores),
        'auprc': average_precision_score(true_binary, pred_scores),
        'f1': f1_score(true_binary, pred_binary),
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division=0),
        'recall': recall_score(true_binary, pred_binary, zero_division=0),
        'mse': float(np.mean((pred_scores - df['true_score'].values) ** 2)),
    }


# ============================================================
# Training loop
# ============================================================
def train_fold(fold_idx, train_samples, val_samples, model_path, output_dir):
    accelerator = Accelerator()
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(fold_dir, "tensorboard"))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_dataset = CanonicalDataset(train_samples, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # Base model (bf16) + LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
    )

    lora_config = LoraConfig(
        r=LORA["r"], lora_alpha=LORA["alpha"],
        target_modules=LORA["target_modules"],
        lora_dropout=LORA["dropout"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Optimizer + scheduler (same as multi-task)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"],
    )

    num_training_steps = len(train_dataloader)
    num_warmup_steps = int(TRAIN["warmup_ratio"] * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler,
    )

    if accelerator.is_main_process:
        logger.info(f"{'=' * 60}")
        logger.info(f"Canonical Fine-Tuning | Fold {fold_idx}")
        logger.info(f"  Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")
        logger.info(f"  Steps: {num_training_steps}, Warmup: {num_warmup_steps}")
        logger.info(f"{'=' * 60}")

    # ---- Train ----
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        train_dataloader,
        desc=f"Fold {fold_idx} - Epoch 1/{TRAIN['num_epochs']}",
        disable=not accelerator.is_main_process,
    )

    for step, batch in enumerate(progress_bar):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=TRAIN["grad_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"lm_loss": f"{loss.item():.4f}"})

        if accelerator.is_main_process:
            writer.add_scalar(f"Fold{fold_idx}/Train/LM_Loss", loss.item(), step)
            writer.add_scalar(f"Fold{fold_idx}/Train/LR", scheduler.get_last_lr()[0], step)

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

    if accelerator.is_main_process:
        logger.info(f"Fold {fold_idx} done: LM_Loss={avg_loss:.4f}")

    # ---- Save checkpoint ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        ckpt_dir = os.path.join(fold_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        unwrapped.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Fold {fold_idx}: Checkpoint saved to {ckpt_dir}")

    # ---- Predict held-out via text generation (all GPUs) ----
    metrics = None
    if accelerator.is_main_process:
        logger.info(f"Fold {fold_idx}: Generating predictions on held-out set "
                     f"({accelerator.num_processes} GPUs)...")

    # All processes participate in generation
    held_out_df = predict_held_out(
        model, tokenizer, val_samples, accelerator,
        max_new_tokens=512,
    )

    if accelerator.is_main_process and held_out_df is not None:
        held_out_df['fold'] = fold_idx
        held_out_path = os.path.join(fold_dir, "held_out_predictions.csv")
        held_out_df.to_csv(held_out_path, index=False)
        logger.info(f"Fold {fold_idx}: Saved {len(held_out_df)} held-out predictions")

        # Score extraction success rate
        n_fallback = (held_out_df['generated_text'].apply(
            lambda t: extract_score_from_text(t) is None
        )).sum()
        logger.info(f"  Score extraction: {len(held_out_df) - n_fallback}/{len(held_out_df)} "
                     f"successful ({n_fallback} fallback to 0.5)")

        metrics = compute_metrics(held_out_df)
        metrics['fold'] = fold_idx
        metrics['train_lm_loss'] = avg_loss
        metrics['score_extraction_failures'] = int(n_fallback)

        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Fold {fold_idx} MSK Metrics:")
        logger.info(f"  AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
                     f"F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    # Cleanup
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    writer.close()

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Canonical Fine-Tuning (CFT)")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    if args.fold < 0 or args.fold > 4:
        raise ValueError("Fold index must be 0-4")

    data_path = REASONING_DATA
    output_dir = args.output_dir or str(CHECKPOINT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Logging to file
    log_file = os.path.join(output_dir, f"fold_{args.fold}_training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Canonical Fine-Tuning")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")

    # Load data
    all_samples = load_data(data_path)
    logger.info(f"Loaded {len(all_samples)} samples")

    # Create 5-fold split (same seed → same split as multi-task)
    logger.info("Creating 5-fold patient-level CV split:")
    folds = create_cv_folds(all_samples)
    train_samples, val_samples = folds[args.fold]

    # Save fold info
    fold_info = {
        "fold": args.fold,
        "data_path": data_path,
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "method": "canonical_finetuning",
        **TRAIN,
        "lora": LORA,
    }
    with open(os.path.join(output_dir, f"fold_{args.fold}_info.json"), "w") as f:
        json.dump(fold_info, f, indent=2)

    # Train
    torch.cuda.empty_cache()
    gc.collect()

    metrics = train_fold(
        args.fold, train_samples, val_samples,
        args.model_path, output_dir,
    )

    logger.info(f"Fold {args.fold} complete!")

    # Aggregate all folds if this is the last one
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0 and args.fold == 4:
        logger.info("\n" + "=" * 60)
        logger.info("Aggregating all 5 folds...")
        logger.info("=" * 60)

        all_dfs = []
        all_metrics = []
        for fi in range(5):
            pred_path = os.path.join(output_dir, f"fold_{fi}", "held_out_predictions.csv")
            metric_path = os.path.join(output_dir, f"fold_{fi}", "metrics.json")

            if os.path.exists(pred_path):
                all_dfs.append(pd.read_csv(pred_path))
            if os.path.exists(metric_path):
                with open(metric_path) as f:
                    all_metrics.append(json.load(f))

        if len(all_dfs) == 5:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(os.path.join(output_dir, "msk_all_held_out_predictions.csv"), index=False)

            overall_metrics = compute_metrics(combined)
            logger.info(f"\nOverall MSK (5-fold held-out, n={len(combined)}):")
            logger.info(f"  AUROC={overall_metrics['auroc']:.4f}, AUPRC={overall_metrics['auprc']:.4f}")
            logger.info(f"  F1={overall_metrics['f1']:.4f}, Accuracy={overall_metrics['accuracy']:.4f}")

            for m in all_metrics:
                logger.info(f"  Fold {m['fold']}: AUROC={m['auroc']:.4f}, F1={m['f1']:.4f}")

            avg_auroc = np.mean([m['auroc'] for m in all_metrics])
            std_auroc = np.std([m['auroc'] for m in all_metrics])
            logger.info(f"\n  AUROC: {avg_auroc:.4f} +/- {std_auroc:.4f}")

            # Score extraction success rate
            total_failures = sum(m.get('score_extraction_failures', 0) for m in all_metrics)
            logger.info(f"  Total score extraction failures: {total_failures}/{len(combined)}")

            summary = {
                "method": "canonical_finetuning",
                "overall": overall_metrics,
                "per_fold": all_metrics,
                "mean_auroc": avg_auroc,
                "std_auroc": std_auroc,
                "total_score_extraction_failures": total_failures,
            }
            with open(os.path.join(output_dir, "5fold_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        else:
            logger.info(f"Only {len(all_dfs)}/5 folds completed.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    main()

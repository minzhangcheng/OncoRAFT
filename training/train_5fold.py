#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OncoRAFT 5-fold patient-level CV training: base LLM + LoRA + score head.

Multitask objective: language-modeling loss on the rationale plus MSE loss
on the score head. Hyperparameters and module shapes are configured in
config.py.

Usage:
  # Train all folds sequentially
  for f in 0 1 2 3 4; do
    accelerate launch train_5fold.py --fold $f
  done

  # Train single fold
  accelerate launch train_5fold.py --fold 0

  # Custom paths
  accelerate launch train_5fold.py --fold 0 \
    --model_path /path/to/base_model \
    --data_path /path/to/training_data.jsonl \
    --output_dir /path/to/checkpoints
"""
import os
import json
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import gc
from tqdm import tqdm
import re
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import KFold
from collections import defaultdict

from config import BASE_MODEL_PATH, TRAINING_DATA, CHECKPOINT_DIR, TRAIN, LORA, SCORE_HEAD
from model import MultitaskLlamaModel, extract_response_score

# Set PyTorch memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
class MultitaskInstructionDataset(Dataset):
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
        if "Patient ID:" in input_data:
            match = re.search(r"Patient ID: ([^\n]+)", input_data)
            if match:
                patient_id = match.group(1)

        # Parse score from first line of output (e.g. "0.7\nReasoning:..." or "Score: 0.7\n...")
        first_line = output.strip().split('\n')[0].strip()
        score_match = re.match(r'(?:Score:\s*)?(\d+\.?\d*)', first_line)
        if score_match:
            true_score = min(1.0, max(0.0, float(score_match.group(1))))
        else:
            true_score = 0.5

        true_binary = 1.0 if true_score >= 0.5 else 0.0

        if not instruction or not input_data or not output:
            raise ValueError(f"Sample {idx} has empty fields")

        output_max_length = TRAIN["output_max_length"]
        prompt_max_length = self.max_length - output_max_length
        full_prompt = instruction + input_data

        prompt_encoding = self.tokenizer(
            full_prompt, add_special_tokens=False, truncation=True,
            max_length=prompt_max_length
        )
        output_encoding = self.tokenizer(
            output, add_special_tokens=False, truncation=True,
            max_length=output_max_length
        )

        input_ids = prompt_encoding["input_ids"] + output_encoding["input_ids"]
        attention_mask = prompt_encoding["attention_mask"] + output_encoding["attention_mask"]

        prompt_length = len(prompt_encoding["input_ids"])
        score_position = prompt_length - 1

        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        labels = [-100] * prompt_length + output_encoding["input_ids"]
        if len(labels) < self.max_length:
            labels += [-100] * (self.max_length - len(labels))
        else:
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "prompt": full_prompt,
            "true_output": output,
            "score_position": score_position,
            "true_score": true_score,
            "true_binary": true_binary,
            "patient_id": patient_id
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "true_outputs": [item["true_output"] for item in batch],
        "score_positions": torch.tensor([item["score_position"] for item in batch]),
        "true_scores": torch.tensor([item["true_score"] for item in batch]),
        "true_binary": torch.tensor([item["true_binary"] for item in batch]),
        "patient_ids": [item["patient_id"] for item in batch],
    }


def load_data(data_path):
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


# ============================================================
# Patient-level CV split
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


def create_cv_folds(samples, n_folds=None):
    n_folds = n_folds or TRAIN["n_folds"]
    patient_groups = group_samples_by_patient_id(samples)
    patient_ids = list(patient_groups.keys())

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=TRAIN["seed"])

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids = [patient_ids[i] for i in val_idx]

        train_samples = []
        for pid in train_pids:
            train_samples.extend(patient_groups[pid])

        val_samples = []
        for pid in val_pids:
            val_samples.extend(patient_groups[pid])

        folds.append((train_samples, val_samples))
        logger.info(f"  Fold {fold_idx}: train={len(train_samples)} ({len(train_pids)} patients), "
                     f"val={len(val_samples)} ({len(val_pids)} patients)")

    return folds


# ============================================================
# Held-out prediction + metrics
# ============================================================
def predict_held_out(model, dataloader, accelerator):
    model.eval()
    local_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting held-out",
                          disable=not accelerator.is_main_process):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                score_positions=batch["score_positions"]
            )
            pred_scores = outputs["predicted_scores"]

            gathered_pred = accelerator.gather(pred_scores)
            gathered_true = accelerator.gather(batch["true_scores"])
            gathered_bin = accelerator.gather(batch["true_binary"])

            from accelerate.utils import gather_object
            gathered_pids = gather_object(batch["patient_ids"])

            if accelerator.is_main_process:
                for i in range(len(gathered_pred)):
                    pid = gathered_pids[i] if i < len(gathered_pids) else ""
                    local_results.append({
                        "patient_id": pid,
                        "true_score": gathered_true[i].item(),
                        "predicted_score": gathered_pred[i].item(),
                        "true_binary": int(gathered_bin[i].item()),
                    })

    if accelerator.is_main_process:
        df = pd.DataFrame(local_results)
        df = df[df["patient_id"].astype(str).str.strip() != ""].reset_index(drop=True)
        return df
    return None


def compute_msk_metrics(df):
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
    train_dataset = MultitaskInstructionDataset(train_samples, tokenizer)
    val_dataset = MultitaskInstructionDataset(val_samples, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=TRAIN["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # Base model (bf16)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device}
    )

    # LoRA
    lora_config = LoraConfig(
        r=LORA["r"], lora_alpha=LORA["alpha"],
        target_modules=LORA["target_modules"],
        lora_dropout=LORA["dropout"], bias="none", task_type="CAUSAL_LM"
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable()

    # Multitask model
    model = MultitaskLlamaModel(base_model)
    model.score_head = model.score_head.to(torch.bfloat16)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"]
    )

    # Scheduler: cosine with warmup
    num_training_steps = len(train_dataloader)
    num_warmup_steps = int(TRAIN["warmup_ratio"] * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    if accelerator.is_main_process:
        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_idx}: Training {TRAIN['num_epochs']} epoch")
        logger.info(f"  Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")
        logger.info(f"  Steps: {num_training_steps}, Warmup: {num_warmup_steps}")
        logger.info(f"  LR: {TRAIN['lr']}, weight_decay: {TRAIN['weight_decay']}")
        logger.info(f"{'=' * 60}")

    # ---- Train ----
    model.train()
    epoch_loss = 0.0
    epoch_lm_loss = 0.0
    epoch_score_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        train_dataloader,
        desc=f"Fold {fold_idx} - Epoch 1/{TRAIN['num_epochs']}",
        disable=not accelerator.is_main_process
    )

    for step, batch in enumerate(progress_bar):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            score_positions=batch["score_positions"],
            true_scores=batch["true_scores"]
        )

        loss = outputs["loss"]
        lm_loss = outputs["lm_loss"]
        score_loss = outputs["score_loss"]

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=TRAIN["grad_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        if lm_loss is not None:
            epoch_lm_loss += lm_loss.item()
        if score_loss is not None:
            epoch_score_loss += score_loss.item()
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lm": f"{lm_loss.item() if lm_loss is not None else 0:.4f}",
            "score": f"{score_loss.item() if score_loss is not None else 0:.4f}"
        })

        if accelerator.is_main_process:
            writer.add_scalar(f"Fold{fold_idx}/Train/Loss", loss.item(), step)
            if lm_loss is not None:
                writer.add_scalar(f"Fold{fold_idx}/Train/LM_Loss", lm_loss.item(), step)
            if score_loss is not None:
                writer.add_scalar(f"Fold{fold_idx}/Train/Score_Loss", score_loss.item(), step)
            writer.add_scalar(f"Fold{fold_idx}/Train/LR", scheduler.get_last_lr()[0], step)

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    avg_lm = epoch_lm_loss / num_batches if num_batches > 0 else 0
    avg_score = epoch_score_loss / num_batches if num_batches > 0 else 0

    if accelerator.is_main_process:
        logger.info(f"Fold {fold_idx} done: Loss={avg_loss:.4f}, LM={avg_lm:.4f}, Score={avg_score:.4f}")

    # ---- Save checkpoint ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        ckpt_dir = os.path.join(fold_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        unwrapped.base_model.save_pretrained(ckpt_dir)
        torch.save(unwrapped.score_head.state_dict(), os.path.join(ckpt_dir, "score_head.pt"))
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Fold {fold_idx}: Checkpoint saved to {ckpt_dir}")

    # ---- Predict held-out ----
    if accelerator.is_main_process:
        logger.info(f"Fold {fold_idx}: Predicting on held-out validation set...")

    held_out_df = predict_held_out(model, val_dataloader, accelerator)

    if accelerator.is_main_process and held_out_df is not None:
        held_out_df['fold'] = fold_idx
        held_out_path = os.path.join(fold_dir, "held_out_predictions.csv")
        held_out_df.to_csv(held_out_path, index=False)
        logger.info(f"Fold {fold_idx}: Saved {len(held_out_df)} held-out predictions")

        metrics = compute_msk_metrics(held_out_df)
        metrics['fold'] = fold_idx
        metrics['train_loss'] = avg_loss
        metrics['train_lm_loss'] = avg_lm
        metrics['train_score_loss'] = avg_score

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

    return metrics if accelerator.is_main_process else None


# ============================================================
# Main
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="OncoRAFT 5-fold CV training")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=TRAINING_DATA)
    parser.add_argument("--output_dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    if args.fold < 0 or args.fold > 4:
        raise ValueError("Fold index must be 0-4")

    os.makedirs(args.output_dir, exist_ok=True)

    # File handler for logging
    log_file = os.path.join(args.output_dir, f"fold_{args.fold}_training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    all_samples = load_data(args.data_path)
    logger.info(f"Loaded {len(all_samples)} samples")

    # Create 5-fold split
    logger.info("Creating 5-fold patient-level CV split:")
    folds = create_cv_folds(all_samples)

    train_samples, val_samples = folds[args.fold]

    # Save fold info
    fold_info = {
        "fold": args.fold,
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        **TRAIN,
        "lora": LORA,
        "score_head": SCORE_HEAD,
    }
    with open(os.path.join(args.output_dir, f"fold_{args.fold}_info.json"), "w") as f:
        json.dump(fold_info, f, indent=2)

    # Train
    torch.cuda.empty_cache()
    gc.collect()

    metrics = train_fold(
        args.fold, train_samples, val_samples,
        args.model_path, args.output_dir
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
            pred_path = os.path.join(args.output_dir, f"fold_{fi}", "held_out_predictions.csv")
            metric_path = os.path.join(args.output_dir, f"fold_{fi}", "metrics.json")

            if os.path.exists(pred_path):
                all_dfs.append(pd.read_csv(pred_path))
            if os.path.exists(metric_path):
                with open(metric_path) as f:
                    all_metrics.append(json.load(f))

        if len(all_dfs) == 5:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(os.path.join(args.output_dir, "msk_all_held_out_predictions.csv"), index=False)

            overall_metrics = compute_msk_metrics(combined)
            logger.info(f"\nOverall MSK (5-fold held-out, n={len(combined)}):")
            logger.info(f"  AUROC={overall_metrics['auroc']:.4f}, AUPRC={overall_metrics['auprc']:.4f}")
            logger.info(f"  F1={overall_metrics['f1']:.4f}, Accuracy={overall_metrics['accuracy']:.4f}")

            for m in all_metrics:
                logger.info(f"  Fold {m['fold']}: AUROC={m['auroc']:.4f}, F1={m['f1']:.4f}")

            avg_auroc = np.mean([m['auroc'] for m in all_metrics])
            std_auroc = np.std([m['auroc'] for m in all_metrics])
            logger.info(f"\n  AUROC: {avg_auroc:.4f} +/- {std_auroc:.4f}")

            summary = {
                "overall": overall_metrics,
                "per_fold": all_metrics,
                "mean_auroc": avg_auroc,
                "std_auroc": std_auroc,
            }
            with open(os.path.join(args.output_dir, "5fold_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        else:
            logger.info(f"Only {len(all_dfs)}/5 folds completed.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Zero-Shot Inference
=============================
Single script to run any model × prompt × thinking-mode combination.

Usage:
    # Standard models
    python inference.py --model llama-8b --prompt summary
    python inference.py --model qwen-72b --prompt simple

    # Qwen3 with thinking mode
    python inference.py --model qwen3-235b --prompt summary --thinking

    # Domain-specific models
    python inference.py --model medgemma-4b --prompt summary

    # Custom settings
    python inference.py --model llama-8b --prompt summary --max_tokens 2048 --tp 8

    # List available models
    python inference.py --list_models
"""
import json
import os
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from config import MODELS, VLLM, SAMPLING, OUTPUT_DIR, get_datasets
from score_extraction import extract_score, parse_qwen3_thinking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Core inference
# ============================================================

def build_prompts(data, tokenizer, enable_thinking=False):
    """Build formatted prompts using model's chat template."""
    prompts = []
    for item in tqdm(data, desc="Building prompts"):
        if "instruction" not in item or "input" not in item:
            continue

        messages = [
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"}
        ]

        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking:
            kwargs["enable_thinking"] = True

        try:
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
            prompts.append(prompt)
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")

    return prompts


def process_outputs(batch_outputs, data, tokenizer, family, enable_thinking):
    """Process vLLM outputs: extract scores, compute metrics."""
    results = []

    for idx, output in enumerate(tqdm(batch_outputs, desc="Processing outputs")):
        original = data[idx]

        # --- Extract response text and score ---
        if family == "qwen3" and enable_thinking:
            output_ids = output.outputs[0].token_ids
            thinking, content = parse_qwen3_thinking(output_ids, tokenizer)
            score = extract_score(content)
            if score is None and thinking:
                score = extract_score(thinking)
            response_text = content
            thinking_text = thinking
        else:
            response_text = output.outputs[0].text.strip()
            score = extract_score(response_text)
            thinking_text = ""

        # --- Get ground-truth label ---
        if "label" in original:
            true_label = original["label"]
        elif "output" in original:
            output_score = extract_score(original["output"])
            true_label = "Positive" if output_score is not None and output_score >= 0.5 else "Negative"
        else:
            true_label = "Unknown"

        result = {
            "prompt": output.prompt,
            "reasoning_content": response_text,
            "response_score": score,
            "true_label": true_label,
        }
        if thinking_text:
            result["thinking_content"] = thinking_text

        results.append(result)

    return results


def compute_metrics(results, model_name, dataset_path, enable_thinking=False):
    """Compute classification metrics from results."""
    valid = [r for r in results if r["response_score"] is not None]

    if not valid:
        return {"dataset": dataset_path, "model": model_name,
                "error": "No valid predictions"}

    y_true = [1 if r["true_label"].lower() == "positive" else 0 for r in valid]
    y_score = [r["response_score"] for r in valid]
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except ValueError:
        tn = fp = fn = tp = total = precision = recall = specificity = accuracy = f1 = 0

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0

    try:
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = 0

    return {
        "dataset": dataset_path,
        "model": model_name,
        "thinking_mode": enable_thinking,
        "samples": len(valid),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auroc": auroc,
        "auprc": auprc,
        "missing_scores": len(results) - len(valid),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    }


def process_dataset(dataset_path, llm, tokenizer, sampling_params,
                    model_name, family, enable_thinking,
                    output_jsonl, metrics_file):
    """Process a single dataset: load, infer, evaluate, save."""
    logger.info(f"Processing: {dataset_path}")

    # Load data
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info(f"Loaded {len(data)} samples")

    # Build prompts
    prompts = build_prompts(data, tokenizer, enable_thinking)
    logger.info(f"Built {len(prompts)} prompts")

    # Inference
    logger.info("Running batch inference...")
    batch_outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Process outputs
    results = process_outputs(batch_outputs, data, tokenizer, family, enable_thinking)

    # Compute metrics
    metrics = compute_metrics(results, model_name, dataset_path, enable_thinking)

    # Save
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    logger.info(f"Results: {json.dumps({k: v for k, v in metrics.items() if k not in ('dataset', 'confusion_matrix')}, indent=2)}")
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Zero-shot inference with vLLM")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model key from registry: {list(MODELS.keys())}")
    parser.add_argument("--prompt", type=str, default="summary",
                        choices=["summary", "simple"],
                        help="Prompt variant")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (Qwen3 only)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens for generation")
    parser.add_argument("--tp", type=int, default=None,
                        help="Override tensor_parallel_size")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override dataset paths")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    # List models
    if args.list_models:
        print(f"\n{'Model Key':<25} {'Family':<10} {'TP':<5} {'Path'}")
        print("-" * 90)
        for key, cfg in sorted(MODELS.items()):
            print(f"{key:<25} {cfg['family']:<10} {cfg['tp']:<5} {cfg['path']}")
        return

    # Resolve model config
    if args.model not in MODELS:
        raise ValueError(f"Unknown model '{args.model}'. Use --list_models to see options.")

    model_cfg = MODELS[args.model]
    model_path = model_cfg["path"]
    tp = args.tp or model_cfg["tp"]
    family = model_cfg["family"]

    # Validate thinking mode
    if args.thinking and family != "qwen3":
        raise ValueError(f"--thinking is only supported for Qwen3 models, not {family}")

    # Resolve datasets
    datasets = args.datasets or get_datasets(args.prompt)

    # Build output naming
    thinking_suffix = "-thinking" if args.thinking else ""
    run_name = f"{args.model}_{args.prompt}{thinking_suffix}"
    run_output_dir = os.path.join(args.output_dir, run_name)

    logger.info(f"{'=' * 60}")
    logger.info(f"Zero-Shot Inference: {run_name}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Family: {family}, TP: {tp}")
    logger.info(f"  Prompt: {args.prompt}, Thinking: {args.thinking}")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Output: {run_output_dir}")
    logger.info(f"{'=' * 60}")

    # --- Initialize vLLM engine ---
    logger.info("Initializing vLLM engine...")
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tp,
        "dtype": VLLM["dtype"],
        "max_model_len": VLLM["max_model_len"],
        "gpu_memory_utilization": VLLM["gpu_memory_utilization"],
        "enforce_eager": VLLM["enforce_eager"],
        "enable_prefix_caching": VLLM["enable_prefix_caching"],
    }
    if args.thinking and family == "qwen3":
        llm_kwargs["enable_reasoning"] = True

    llm = LLM(**llm_kwargs)
    logger.info("vLLM engine ready.")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- Sampling params ---
    max_tokens = args.max_tokens or SAMPLING["max_tokens"]
    sampling_params = SamplingParams(
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        top_k=SAMPLING["top_k"],
        max_tokens=max_tokens,
        repetition_penalty=SAMPLING["repetition_penalty"],
    )

    # --- Process each dataset ---
    all_metrics = []
    for i, dataset_path in enumerate(datasets, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset {i}/{len(datasets)}: {os.path.basename(dataset_path)}")
        logger.info(f"{'=' * 60}")

        ds_name = os.path.basename(dataset_path).replace(".jsonl", "")
        output_jsonl = os.path.join(run_output_dir, f"scores-{ds_name}.jsonl")
        metrics_file = os.path.join(run_output_dir, f"metrics-{ds_name}.json")

        try:
            metrics = process_dataset(
                dataset_path, llm, tokenizer, sampling_params,
                args.model, family, args.thinking,
                output_jsonl, metrics_file,
            )
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Failed on {dataset_path}: {e}")
            all_metrics.append({"dataset": dataset_path, "model": args.model, "error": str(e)})

    # --- Save summary ---
    os.makedirs(run_output_dir, exist_ok=True)
    summary_file = os.path.join(run_output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "model_path": model_path,
            "prompt_variant": args.prompt,
            "thinking_mode": args.thinking,
            "max_tokens": max_tokens,
            "datasets_total": len(datasets),
            "datasets_success": len([m for m in all_metrics if "error" not in m]),
            "all_metrics": all_metrics,
        }, f, indent=4, ensure_ascii=False)

    # --- Print summary table ---
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Summary: {run_name}")
    logger.info(f"{'=' * 60}")
    successful = [m for m in all_metrics if "error" not in m]
    if successful:
        print(f"\n{'Dataset':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUROC':<8} {'AUPRC':<8}")
        print("-" * 68)
        for m in successful:
            ds = os.path.basename(m["dataset"]).split("_")[-1].replace(".jsonl", "")
            print(f"{ds:<20} {m['accuracy']:<8.4f} {m['precision']:<8.4f} "
                  f"{m['recall']:<8.4f} {m['f1']:<8.4f} {m['auroc']:<8.4f} {m['auprc']:<8.4f}")

        # Average across seeds
        import numpy as np
        avg_auroc = np.mean([m["auroc"] for m in successful])
        avg_auprc = np.mean([m["auprc"] for m in successful])
        avg_f1 = np.mean([m["f1"] for m in successful])
        print(f"\n  Mean AUROC: {avg_auroc:.4f}, Mean AUPRC: {avg_auprc:.4f}, Mean F1: {avg_f1:.4f}")

    logger.info(f"\nResults saved to: {run_output_dir}")


if __name__ == "__main__":
    main()

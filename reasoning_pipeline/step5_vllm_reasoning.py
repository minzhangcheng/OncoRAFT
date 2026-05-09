"""
Step 5 (vLLM variant): Generate reasoning using a local model via vLLM.

Replaces the DeepSeek API call with local vLLM offline batch inference.
Produces the same final output format as step5_api + step6 combined:
  - Filters INVESTIGATIONAL-only entries
  - Adds system role instruction
  - Output format: "{score}\\nReasoning:\\n{reasoning}"

Usage:
    python step5_vllm_reasoning.py
    python step5_vllm_reasoning.py --model /path/to/model --tag mymodel --tp 4
"""
import os
import re
import json
import time
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import BASE_DIR, BATCH_PROMPTS, INSTRUCTION_PROMPTS, SYSTEM_INSTRUCTION
from utils import load_jsonl

# Defaults
DEFAULT_MODEL = os.environ.get("ONCORAFT_BASE_MODEL", "")
DEFAULT_TAG = "llama3.1-8b"
DEFAULT_TP = 8
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_UTIL = 0.90
DEFAULT_BATCH_SIZE = 64


def build_prompts_with_chat_template(prompts, tokenizer):
    """Format prompts using the model's chat template."""
    formatted = []
    for item in tqdm(prompts, desc="Building chat template prompts"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        user_content = f"{instruction}\n\n{input_text}"
        messages = [{"role": "user", "content": user_content}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted.append(text)
        except Exception as e:
            print(f"Error applying chat template: {e}")
            formatted.append(user_content)
    return formatted


def extract_reasoning(response, original_score):
    """Extract reasoning from model response, replace score with original."""
    if not original_score:
        original_score = "0.0"
    response = response.strip()

    if "Reasoning:" in response:
        reasoning_part = response.split("Reasoning:", 1)[1].strip()
    else:
        score_pattern = re.compile(r"^Score:\s*[\d\.]+", re.MULTILINE)
        match = score_pattern.search(response)
        if match:
            reasoning_part = response[match.end():].strip()
        else:
            reasoning_part = response

    return f"{original_score}\nReasoning:\n{reasoning_part}"


def is_investigational_only(input_text):
    """Check if the entry contains only INVESTIGATIONAL drugs."""
    drug_match = re.search(r"Drug\(s\):\s*(.+)", input_text)
    return drug_match and drug_match.group(1).strip() == "INVESTIGATIONAL"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning via local vLLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Model tag for output filenames")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP, help="Tensor parallel size")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    output_file = os.path.join(BASE_DIR, f"{args.tag}-instruction_data_with_reasoning_postprocessed.jsonl")
    debug_file = os.path.join(BASE_DIR, f"{args.tag}-reasoning_outputs.jsonl")

    print(f"Model:      {args.model}")
    print(f"Tag:        {args.tag}")
    print(f"TP:         {args.tp}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output:     {output_file}")

    # Load data
    print("\nLoading data...")
    batch_prompts = load_jsonl(BATCH_PROMPTS)
    original_data = load_jsonl(INSTRUCTION_PROMPTS)

    min_len = min(len(batch_prompts), len(original_data))
    if len(batch_prompts) != len(original_data):
        print(f"WARNING: length mismatch ({len(batch_prompts)} vs {len(original_data)}), using first {min_len}")
    batch_prompts = batch_prompts[:min_len]
    original_data = original_data[:min_len]
    print(f"Loaded {min_len} samples")

    # Pre-filter INVESTIGATIONAL
    valid_indices = [i for i, item in enumerate(original_data) if not is_investigational_only(item.get("input", ""))]
    removed = min_len - len(valid_indices)
    print(f"After filtering INVESTIGATIONAL: {len(valid_indices)} samples (removed {removed})")

    # Extract original scores
    original_scores = []
    for item in original_data:
        output = item.get("output", "")
        match = re.search(r"Score:\s*([01](?:\.\d+)?)", output)
        original_scores.append(match.group(1) if match else None)

    # Initialize vLLM
    print(f"\nInitializing vLLM (tp={args.tp})...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    print(f"vLLM ready in {time.time() - t0:.1f}s")

    # Build prompts with chat template
    print("\nBuilding prompts with chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    valid_batch_prompts = [batch_prompts[i] for i in valid_indices]
    formatted_prompts = build_prompts_with_chat_template(valid_batch_prompts, tokenizer)
    print(f"Built {len(formatted_prompts)} formatted prompts")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Generate in batches
    print(f"\nGenerating reasoning for {len(formatted_prompts)} samples...")
    t0 = time.time()
    total_batches = (len(formatted_prompts) + args.batch_size - 1) // args.batch_size
    written = 0
    empty_count = 0

    with open(output_file, "w", encoding="utf-8") as f_out, \
         open(debug_file, "w", encoding="utf-8") as f_debug:

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(formatted_prompts))

            outputs = llm.generate(formatted_prompts[start:end], sampling_params)

            for i, output in enumerate(outputs):
                global_pos = start + i
                orig_idx = valid_indices[global_pos]
                generated = output.outputs[0].text.strip()
                score = original_scores[orig_idx]

                if not generated:
                    empty_count += 1
                    continue

                new_output = extract_reasoning(generated, score)
                final_item = {
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": original_data[orig_idx].get("input", ""),
                    "output": new_output,
                }
                f_out.write(json.dumps(final_item, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1

                debug_record = {
                    "index": orig_idx,
                    "original_score": score,
                    "full_response": generated,
                    "extracted_output": new_output[:200],
                }
                f_debug.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
                f_debug.flush()

    gen_time = time.time() - t0
    print(f"\nDone!")
    print(f"  Total input:     {min_len}")
    print(f"  INVESTIGATIONAL: {removed} (filtered)")
    print(f"  Empty responses: {empty_count}")
    print(f"  Written:         {written}")
    print(f"  Time:            {gen_time:.1f}s ({written/max(gen_time, 0.1):.1f} samples/s)")
    print(f"  Output: {output_file}")
    print(f"  Debug:  {debug_file}")


if __name__ == "__main__":
    main()

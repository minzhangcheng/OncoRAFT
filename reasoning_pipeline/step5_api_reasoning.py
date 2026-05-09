"""
Step 5 (API variant): Generate reasoning using DeepSeek API.

Calls DeepSeek-chat API concurrently to generate clinical reasoning for each
patient's drug response prediction. Preserves the original RECIST-based score
and replaces the output with detailed reasoning.

Usage:
    export DEEPSEEK_API_KEY=your_key
    python step5_api_reasoning.py
"""
import os
import time
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from config import (
    BATCH_PROMPTS, INSTRUCTION_PROMPTS, REASONING_DATA, API_OUTPUTS,
    API_KEY, API_BASE_URL, API_MODEL_REASONING,
    REASONING_TEMPERATURE, REASONING_TOP_P,
    MAX_WORKERS, RATE_LIMIT_DELAY,
)
from utils import RateLimiter, load_jsonl

# Global rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_DELAY)

# API client
if not API_KEY:
    raise ValueError("Environment variable DEEPSEEK_API_KEY must be set")
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def deepseek_api_request(prompt, max_retries=5):
    """Call DeepSeek API with streaming and return the full response text."""
    instruction = prompt.get("instruction", "")
    input_text = prompt.get("input", "")
    full_prompt = f"{instruction}\n\n{input_text}"

    for attempt in range(max_retries):
        try:
            rate_limiter.acquire()
            response = client.chat.completions.create(
                model=API_MODEL_REASONING,
                messages=[{'role': 'user', 'content': full_prompt}],
                stream=True,
                temperature=REASONING_TEMPERATURE,
                top_p=REASONING_TOP_P,
            )

            generated_text = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    generated_text += delta.content

            if not generated_text.strip():
                raise Exception("Empty output received.")
            return generated_text

        except Exception as e:
            error_str = str(e)
            if "TPM limit reached" in error_str or "429" in error_str:
                print(f"[Thread] Rate limit hit, waiting 60s (attempt {attempt+1})")
                time.sleep(60)
            else:
                wait_time = 2 ** attempt
                print(f"[Thread] Request failed: {error_str} (attempt {attempt+1}), waiting {wait_time}s")
                time.sleep(wait_time)

    return ""


def extract_reasoning(response, original_score):
    """Extract reasoning from model response, replace score with original."""
    if not original_score:
        original_score = "0.0"
    response = response.strip()

    if "Reasoning:" in response:
        reasoning_part = response.split("Reasoning:", 1)[1].strip()
    else:
        score_pattern = re.compile(r'^Score:\s*[\d\.]+', re.MULTILINE)
        match = score_pattern.search(response)
        if match:
            reasoning_part = response[match.end():].strip()
        else:
            reasoning_part = response

    return f"Score: {original_score}\nReasoning:\n{reasoning_part}"


def process_one_sample(idx, prompt, original_item):
    """Process a single prompt and return updated instruction data."""
    original_output = original_item.get("output", "")
    score_match = re.search(r"Score:\s*([01](?:\.\d+)?)", original_output)
    original_score = score_match.group(1) if score_match else None

    # Extract patient ID for logging
    input_text = original_item.get("input", "")
    patient_id = "unknown"
    pid_match = re.search(r"Patient ID:\s*([^\n]+)", input_text)
    if pid_match:
        patient_id = pid_match.group(1).strip()

    # Extract drug name for logging
    drug_name = "unknown"
    for pattern in [r"Drug:\s*([^\n]+)", r"Drug\(s\):\s*([^\n]+)"]:
        drug_match = re.search(pattern, input_text)
        if drug_match:
            drug_name = drug_match.group(1).strip()
            break

    generated_text = deepseek_api_request(prompt)
    new_output = extract_reasoning(generated_text, original_score)

    updated_item = original_item.copy()
    updated_item["output"] = new_output

    api_output = {
        "index": idx,
        "patient_id": patient_id,
        "drug_name": drug_name,
        "original_score": original_score,
        "full_api_response": generated_text,
        "extracted_reasoning": new_output,
    }

    return updated_item, api_output


def main():
    batch_prompts = load_jsonl(BATCH_PROMPTS)
    original_data = load_jsonl(INSTRUCTION_PROMPTS)

    print(f"Loaded {len(batch_prompts)} batch prompts and {len(original_data)} original data items.")

    if len(batch_prompts) != len(original_data):
        min_length = min(len(batch_prompts), len(original_data))
        print(f"WARNING: length mismatch, using first {min_length} items")
        batch_prompts = batch_prompts[:min_length]
        original_data = original_data[:min_length]

    updated_data = []

    with open(API_OUTPUTS, "w", encoding="utf-8") as api_file:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {}
            for idx, (prompt, orig) in enumerate(zip(batch_prompts, original_data)):
                future = executor.submit(process_one_sample, idx, prompt, orig)
                future_to_idx[future] = idx

            for future in tqdm(as_completed(future_to_idx), total=len(batch_prompts), desc="Generating reasoning"):
                idx = future_to_idx[future]
                try:
                    updated_item, api_output = future.result()
                    updated_data.append(updated_item)
                    api_file.write(json.dumps(api_output, ensure_ascii=False) + "\n")
                    api_file.flush()
                except Exception as e:
                    print(f"[Error] Sample {idx}: {e}")
                    updated_data.append(original_data[idx])
                    error_output = {"index": idx, "error": str(e), "full_api_response": ""}
                    api_file.write(json.dumps(error_output, ensure_ascii=False) + "\n")
                    api_file.flush()

    with open(REASONING_DATA, "w", encoding="utf-8") as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Updated instruction data saved to {REASONING_DATA}")
    print(f"API output data saved to {API_OUTPUTS}")


if __name__ == "__main__":
    main()

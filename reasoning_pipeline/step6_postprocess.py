"""
Step 6: Post-process reasoning data for fine-tuning.

Only needed when using step5_api_reasoning.py (the API variant).
The vLLM variant (step5_vllm_reasoning.py) already includes this post-processing.

Operations:
  - Add system role prefix to instruction
  - Remove "Score: " prefix from output (keep raw number)
"""
import json
import re
from config import REASONING_DATA, FINAL_TRAINING_DATA, SYSTEM_INSTRUCTION


def modify_jsonl_file(input_file, output_file):
    """Filter INVESTIGATIONAL entries, update instruction, clean output format."""
    modified_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())

                # Update instruction with system role prefix
                if 'instruction' in data:
                    data['instruction'] = SYSTEM_INSTRUCTION

                # Clean output: remove "Score: " prefix, keep raw number
                if 'output' in data:
                    output_text = data['output']
                    score_match = re.match(r'Score:\s*(\d+\.?\d*)', output_text)
                    if score_match:
                        score = score_match.group(1)
                        output_text = score + output_text[score_match.end():]
                        data['output'] = output_text

                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                modified_count += 1

            except json.JSONDecodeError as e:
                print(f"JSON parse error at line {line_num}: {e}")
            except Exception as e:
                print(f"Error at line {line_num}: {e}")

    print(f"Done! Modified {modified_count} records.")


def main():
    print("Step 6: Post-processing reasoning data...")
    print(f"Input:  {REASONING_DATA}")
    print(f"Output: {FINAL_TRAINING_DATA}")
    modify_jsonl_file(REASONING_DATA, FINAL_TRAINING_DATA)


if __name__ == "__main__":
    main()

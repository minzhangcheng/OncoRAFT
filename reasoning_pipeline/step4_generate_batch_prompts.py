"""
Step 4: Generate batch prompts for API reasoning.

Takes instruction-format data and creates prompts that instruct the LLM
to generate reasoning for each patient's drug response prediction.
"""
import json
from config import INSTRUCTION_PROMPTS, BATCH_PROMPTS
from utils import write_jsonl

BATCH_INSTRUCTION = (
    "Generate detailed reasoning for cancer drug response prediction\n\n"
    "You are an oncology AI specializing in precision medicine. "
    "For the given patient information and drug description:\n\n"
    "1. DO NOT change the provided prediction score (it is already correctly calculated)\n"
    "2. Generate detailed, scientifically accurate reasoning explaining WHY this patient received this specific score\n"
    "3. Analyze the specific genetic mutations and their relationship to the drug's mechanism\n"
    "4. Consider all relevant clinical factors including cancer type, staging, and patient demographics\n\n"
    "YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT:\n"
    "- First line must be exactly: \"Score: [score]\" where [score] is the provided score\n"
    "- Second line must be exactly: \"Reasoning:\"\n"
    "- Then provide 3-4 bullet points, each starting with \"-\" and focusing on different aspects:"
)

BATCH_OUTPUT_TEMPLATE = (
    "{score}\nReasoning:\n"
    "- [Genetic mutation analysis specific to this patient and their relationship to drug mechanism]\n"
    "- [Clinical factors analysis including cancer type, staging, and critical determinants]\n"
    "- [Additional factors that influence response prediction including treatment history]"
)


def main():
    print("Step 4: Generating batch prompts for API reasoning...")
    print(f"Input:  {INSTRUCTION_PROMPTS}")
    print(f"Output: {BATCH_PROMPTS}")

    batch_prompts = []
    with open(INSTRUCTION_PROMPTS, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                patient_data = item["input"]
                score_text = item["output"].strip()

                prompt = {
                    "instruction": BATCH_INSTRUCTION,
                    "input": f"{patient_data}\n\nPREDICTION SCORE: {score_text}",
                    "output": BATCH_OUTPUT_TEMPLATE.format(score=score_text),
                }
                batch_prompts.append(prompt)
            except Exception as e:
                print(f"Error processing item {idx}: {e}")

    write_jsonl(batch_prompts, BATCH_PROMPTS)
    print(f"Generated {len(batch_prompts)} batch prompts.")


if __name__ == "__main__":
    main()

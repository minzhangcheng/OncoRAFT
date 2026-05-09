"""
Step 3: Convert cleaned prompts to instruction-tuning format.

Extracts structured sections from raw prompts and reorganizes them into
a standard {instruction, input, output} format for fine-tuning.

Sections are merged:
  - "Clinical Information" + "Diagnosis Information" -> "Clinical and Diagnosis Information"
  - "Sample Information" + "Genetic Profile" -> "Sample and Genetic Profile"
"""
import json
from config import CLEANED_PROMPTS, INSTRUCTION_PROMPTS

INSTRUCTION_TEXT = """Drug Response Inference Task
Based on the patient's genetic profile, clinical and diagnosis Information, and the drug's mechanism of action, please provide:

1. Output a score representing the likelihood of positive treatment response (a number between 0 and 1).
 - Scores closer to 1 indicate higher likelihood of complete response(positive)
 - Scores closer to 0 indicate higher likelihood of disease progression(negative)
2. Key genetic factors influencing this prediction and why they matter for this treatment plan
3. Critical clinical determinants that significantly impact the efficacy of this treatment
"""


def find_next_section(prompt, start, candidates):
    """Find the earliest occurrence of any candidate section header after start position."""
    earliest = -1
    for candidate in candidates:
        idx = prompt.find(candidate, start)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    return earliest


def convert_to_instruction_format(input_path, output_path):
    """Convert cleaned prompts to instruction-tuning format."""
    count = 0

    with open(output_path, 'w', encoding='utf-8') as f_out, \
         open(input_path, 'r', encoding='utf-8') as f_in:

        for line in f_in:
            if not line.strip():
                continue

            prompt_data = json.loads(line)
            patient_id = prompt_data.get("patient_id", "")
            drug_name = prompt_data.get("drug_name", "")
            score = prompt_data.get("score", "0.0")
            original_prompt = prompt_data.get("prompt", "")

            input_parts = []
            input_parts.append(f"Patient ID: {patient_id}")
            input_parts.append(f"Drug(s): {drug_name}")

            # 1. Extract Drug Information section
            drug_start = original_prompt.find("# Drug Information")
            if drug_start != -1:
                drug_end = find_next_section(original_prompt, drug_start + len("# Drug Information"), [
                    "# Clinical and Diagnosis Information", "# Clinical Information",
                    "# Diagnosis Information", "# Sample Information",
                    "# Genetic Profile", "# Prediction Task",
                ])
                if drug_end == -1:
                    drug_end = len(original_prompt)
                input_parts.append(original_prompt[drug_start:drug_end].strip())

            # 2. Extract Clinical and Diagnosis Information (merged)
            clinical_index = original_prompt.find("# Clinical and Diagnosis Information")
            clinical_header = "# Clinical and Diagnosis Information"
            if clinical_index == -1:
                clinical_index = original_prompt.find("# Clinical Information")
                clinical_header = "# Clinical Information"

            clinical_content = ""
            if clinical_index != -1:
                clinical_end = find_next_section(original_prompt, clinical_index + len(clinical_header), [
                    "# Diagnosis Information", "# Sample Information",
                    "# Genetic Profile", "# Prediction Task",
                ])
                if clinical_end == -1:
                    clinical_end = len(original_prompt)
                clinical_content = original_prompt[clinical_index:clinical_end].strip()
                clinical_content = clinical_content.replace(clinical_header, "# Clinical and Diagnosis Information", 1)

            diagnosis_index = original_prompt.find("# Diagnosis Information")
            diagnosis_content = ""
            if diagnosis_index != -1:
                diagnosis_end = find_next_section(original_prompt, diagnosis_index + len("# Diagnosis Information"), [
                    "# Sample Information", "# Genetic Profile", "# Prediction Task",
                ])
                if diagnosis_end == -1:
                    diagnosis_end = len(original_prompt)
                diagnosis_content = original_prompt[diagnosis_index:diagnosis_end].strip()

            if clinical_content and diagnosis_content:
                diag_body = diagnosis_content.replace("# Diagnosis Information", "", 1).strip()
                clinical_content = f"{clinical_content}\n\n{diag_body}"
            elif diagnosis_content:
                clinical_content = diagnosis_content.replace("# Diagnosis Information", "# Clinical and Diagnosis Information", 1)

            if clinical_content:
                input_parts.append(clinical_content)

            # 3. Extract Sample Information and Genetic Profile (merged)
            sample_index = original_prompt.find("# Sample Information")
            sample_content = ""
            if sample_index != -1:
                sample_end = find_next_section(original_prompt, sample_index + len("# Sample Information"), [
                    "# Genetic Profile", "# Prediction Task",
                ])
                if sample_end == -1:
                    sample_end = len(original_prompt)
                sample_content = original_prompt[sample_index:sample_end].strip()

            genetic_index = original_prompt.find("# Genetic Profile")
            genetic_content = ""
            if genetic_index != -1:
                genetic_end = original_prompt.find("# Prediction Task", genetic_index + 1)
                if genetic_end == -1:
                    genetic_end = len(original_prompt)
                genetic_content = original_prompt[genetic_index:genetic_end].strip()

            if sample_content and genetic_content:
                sample_content = sample_content.replace("# Sample Information", "# Sample and Genetic Profile", 1)
                gen_body = genetic_content.replace("# Genetic Profile", "", 1).strip()
                sample_content = f"{sample_content}\n\n{gen_body}"
            elif genetic_content:
                sample_content = genetic_content.replace("# Genetic Profile", "# Sample and Genetic Profile", 1)

            if sample_content:
                input_parts.append(sample_content)

            input_text = "\n\n".join(input_parts)
            output_text = f"Score: {score}."

            instruction_prompt = {
                "instruction": INSTRUCTION_TEXT,
                "input": input_text,
                "output": output_text,
            }
            f_out.write(json.dumps(instruction_prompt, ensure_ascii=False) + '\n')
            count += 1

    return count


def main():
    print("Step 3: Converting to instruction-tuning format...")
    print(f"Input:  {CLEANED_PROMPTS}")
    print(f"Output: {INSTRUCTION_PROMPTS}")
    count = convert_to_instruction_format(CLEANED_PROMPTS, INSTRUCTION_PROMPTS)
    print(f"Converted {count} prompts")


if __name__ == "__main__":
    main()

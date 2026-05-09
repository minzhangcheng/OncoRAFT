"""
Step 2: Clean raw prompts.

- Filter out entries with only INVESTIGATIONAL drugs (no real drug name)
- Remove nan values
- Remove invalid Clinical/Pathological Group codes
- De-duplicate drug sections within each prompt
- De-duplicate drug names in drug_name field
"""
import json
import re
from collections import OrderedDict
from config import RAW_PROMPTS, CLEANED_PROMPTS


def clean_prompt_text(prompt_text):
    """Clean prompt text: remove nan values, invalid groups, and duplicate drug sections."""
    lines = prompt_text.split('\n')
    cleaned_lines = []
    seen_drug_sections = set()
    skip_drug_section = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Remove Clinical Group: 99 and Pathological Group: 99
        if re.match(r'Clinical Group:\s*99\s*$', line) or re.match(r'Pathological Group:\s*99\s*$', line):
            i += 1
            continue

        # Remove lines with nan values like "ECOG Score: nan (...)"
        if re.match(r'.*:\s*nan\s*\(.*\)', line):
            i += 1
            continue

        # Handle duplicate drug sections (## DRUG_NAME)
        drug_match = re.match(r'^##\s+(.+)$', line)
        if drug_match:
            drug_name = drug_match.group(1).strip()
            if drug_name in seen_drug_sections:
                skip_drug_section = True
                i += 1
                continue
            else:
                seen_drug_sections.add(drug_name)
                skip_drug_section = False
                cleaned_lines.append(line)
                i += 1
                continue

        # If inside a duplicate drug section, skip until next section header
        if skip_drug_section:
            if re.match(r'^#', line):
                skip_drug_section = False
                cleaned_lines.append(line)
            i += 1
            continue

        cleaned_lines.append(line)
        i += 1

    return '\n'.join(cleaned_lines)


def clean_drug_list(drug_string):
    """De-duplicate drug names while preserving order."""
    if not drug_string:
        return drug_string
    drugs = [drug.strip() for drug in drug_string.split(',')]
    return ', '.join(OrderedDict.fromkeys(drugs))


def is_investigational_only(data):
    """Check if entry has only INVESTIGATIONAL drugs (no real drug name).

    Catches both single 'INVESTIGATIONAL' and combos like
    'INVESTIGATIONAL, INVESTIGATIONAL, INVESTIGATIONAL'.
    """
    prompt = data.get('prompt', '')
    drug_match = re.search(r'Drug\(s\):\s*(.+)', prompt)
    if drug_match:
        drugs = [d.strip() for d in drug_match.group(1).split(',')]
        if all(d == 'INVESTIGATIONAL' for d in drugs):
            return True
    return False


def process_jsonl_file(input_file, output_file):
    """Process JSONL file: filter, clean prompts and drug names."""
    processed_count = 0
    total_count = 0
    removed_investigational = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1

                # Filter: remove entries with only INVESTIGATIONAL drug
                if is_investigational_only(data):
                    removed_investigational += 1
                    continue

                if 'drug_name' in data:
                    data['drug_name'] = clean_drug_list(data['drug_name'])

                if 'prompt' in data:
                    original_prompt = data['prompt']
                    cleaned_prompt = clean_prompt_text(original_prompt)
                    data['prompt'] = cleaned_prompt
                    if len(cleaned_prompt) < len(original_prompt) * 0.95:
                        processed_count += 1

                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                outfile.write(line)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                outfile.write(line)

    kept = total_count - removed_investigational
    print(f"Done! {total_count} total, removed {removed_investigational} INVESTIGATIONAL-only, "
          f"kept {kept}, {processed_count} significantly cleaned.")


def main():
    print("Step 2: Cleaning raw prompts...")
    print(f"Input:  {RAW_PROMPTS}")
    print(f"Output: {CLEANED_PROMPTS}")
    process_jsonl_file(RAW_PROMPTS, CLEANED_PROMPTS)


if __name__ == "__main__":
    main()

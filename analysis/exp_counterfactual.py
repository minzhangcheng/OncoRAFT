#!/usr/bin/env python3
"""
Counterfactual Perturbation Experiment

For each canonical gene-drug pair, mask the gene's mutation rows in the
input prompt and re-run the model. Two metrics:
  1. Gene mention disappearance rate: does the gene name vanish from the
     generated reasoning text after the mutation is masked?
  2. Δscore direction: does the predicted score shift in the expected
     pharmacological direction (down for direct targets, up for adverse
     biomarkers)?
"""
import os, sys, json, re, argparse, warnings, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
from config import (
    OUTPUT_DIR as _CFG_OUTPUT_DIR,
    TRAINING_DATA,
    ONCORAFT_CHECKPOINT_DIR,
    FEATURE_MATRIX_CSV,
)

BASE_DIR    = ONCORAFT_CHECKPOINT_DIR
MERGED_DIR  = lambda fold: f'{BASE_DIR}/fold_{fold}_merged'
DATA_FILE   = TRAINING_DATA
FM_CSV      = FEATURE_MATRIX_CSV
OUTPUT_DIR  = os.path.join(_CFG_OUTPUT_DIR, 'counterfactual')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPP = {'ZOLEDRONIC ACID','PAMIDRONATE','DENOSUMAB','LEUPROLIDE','GOSERELIN',
        'DEGARELIX','LEUCOVORIN','PREDNISONE','INVESTIGATIONAL','MEGESTROL'}

def norm_drug(d):
    if pd.isna(d): return None
    parts = [x.strip().upper() for x in d.split(',')]
    core = [x for x in parts if x not in SUPP]
    return ', '.join(sorted(core)) if core else None

# ── Experiment pairs ───────────────────────────────────────────────────
# Twenty canonical gene-drug pairs: ten direct drug targets (masked → score drops)
# and ten adverse biomarkers (masked → score rises).
PAIRS = [
    # (gene, cancer, regimen, expected_delta: 'down'=score should drop when masked)

    # === Direct drug targets (n=10) ===
    ('EGFR',  'NSCLC',       'OSIMERTINIB',                              'down'),
    ('EGFR',  'NSCLC',       'ERLOTINIB',                                'down'),
    ('EGFR',  'NSCLC',       'AFATINIB',                                 'down'),
    ('EGFR',  'NSCLC',       'DACOMITINIB',                              'down'),
    ('BRAF',  'NSCLC',       'DABRAFENIB, TRAMETINIB',                   'down'),
    ('BRAF',  'Colorectal',  'ENCORAFENIB',                              'down'),
    ('KRAS',  'NSCLC',       'SOTORASIB',                                'down'),
    ('MET',   'NSCLC',       'TEPOTINIB',                                'down'),
    ('MET',   'NSCLC',       'CAPMATINIB',                               'down'),
    ('BRCA2', 'Pancreatic',  'OLAPARIB',                                 'down'),

    # === Adverse biomarkers (n=10) ===
    ('STK11', 'NSCLC',       'PEMBROLIZUMAB',                            'up'),
    ('STK11', 'NSCLC',       'NIVOLUMAB',                                'up'),
    ('STK11', 'NSCLC',       'ATEZOLIZUMAB',                             'up'),
    ('KEAP1', 'NSCLC',       'PEMBROLIZUMAB',                            'up'),
    ('KEAP1', 'NSCLC',       'NIVOLUMAB',                                'up'),
    ('KEAP1', 'NSCLC',       'CARBOPLATIN, PEMBROLIZUMAB, PEMETREXED',   'up'),
    ('KRAS',  'Colorectal',  'FLUOROURACIL, OXALIPLATIN',                'up'),
    ('BRAF',  'Colorectal',  'FLUOROURACIL, OXALIPLATIN',                'up'),
    ('SMAD4', 'Colorectal',  'FLUOROURACIL, OXALIPLATIN',                'up'),
    ('ESR1',  'Breast',      'FULVESTRANT',                              'up'),
]

N_SAMPLES_PER_PAIR = 100

# Aliases for detecting gene mentions in generated text
GENE_KEYWORDS = {
    'EGFR':   [r'\bEGFR\b'],
    'STK11':  [r'\bSTK11\b', r'\bLKB1\b'],
    'ESR1':   [r'\bESR1\b', r'estrogen receptor.*mutation'],
    'MET':    [r'\bMET\b(?:\s+mutation|\s+amplif|\s+exon|\s+alter)', r'c-MET'],
    'ERBB2':  [r'\bERBB2\b', r'\bHER2\b'],
    'BRAF':   [r'\bBRAF\b'],
    'KEAP1':  [r'\bKEAP1\b'],
    'KRAS':   [r'\bKRAS\b'],
    'TP53':   [r'\bTP53\b', r'\bp53\b'],
    'PIK3CA': [r'\bPIK3CA\b', r'PI3K'],
    'BRCA2':  [r'\bBRCA2\b'],
    'BRCA1':  [r'\bBRCA1\b'],
    'APC':    [r'\bAPC\b'],
    'SMAD4':  [r'\bSMAD4\b', r'\bDPC4\b'],
    'ATM':    [r'\bATM\b'],
}


def gene_mentioned_in_text(text, gene):
    keywords = GENE_KEYWORDS.get(gene, [rf'\b{re.escape(gene)}\b'])
    for kw in keywords:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False


def mask_mutation_in_input(input_text, gene):
    """
    Remove a specific gene's mutation entries from the input text.
    The mutations section looks like:
      ## Mutations
      TP53 p.R273H Missense_Mutation
      EGFR p.L858R Missense_Mutation
      ...
    We remove lines containing the gene name in the mutations section.
    """
    # Find mutations section
    mut_start = input_text.find('## Mutations')
    if mut_start < 0:
        return input_text, False

    # Find end of mutations section
    mut_end = input_text.find('## Structural Variants', mut_start)
    if mut_end < 0:
        mut_end = input_text.find('\n# ', mut_start + 1)
    if mut_end < 0:
        mut_end = len(input_text)

    mutations_section = input_text[mut_start:mut_end]
    lines = mutations_section.split('\n')

    new_lines = []
    removed = False
    for line in lines:
        if re.search(rf'\b{re.escape(gene)}\b', line, re.IGNORECASE):
            removed = True  # skip this line
        else:
            new_lines.append(line)

    if not removed:
        return input_text, False

    new_mutations = '\n'.join(new_lines)

    # If no mutations left (all were this gene), add "No mutations detected"
    non_header_lines = [l for l in new_lines if l.strip() and not l.startswith('##')]
    if not non_header_lines:
        new_mutations = '## Mutations\nNo mutations detected'

    masked_text = input_text[:mut_start] + new_mutations + input_text[mut_end:]
    return masked_text, True


def run_inference(model, tokenizer, score_head, prompt_text, device, max_new_tokens=1024):
    """Run model inference: get score + generated text."""
    inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                       max_length=1536).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    score_pos = inputs['attention_mask'].sum().item() - 1
    score_hidden = hidden[0, score_pos, :]
    predicted_score = score_head(score_hidden.unsqueeze(0)).item()

    score_str = f"{predicted_score:.2f}"
    gen_prompt = prompt_text + score_str + "\nReasoning:\n"
    gen_inputs = tokenizer(gen_prompt, return_tensors='pt', truncation=True,
                           max_length=1536 + 20).to(device)

    with torch.no_grad():
        gen_outputs = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = gen_outputs[0][gen_inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    full_text = f"{score_str}\nReasoning:\n{generated_text}"

    return predicted_score, full_text


def run_counterfactual_experiment(device='cuda:0', max_new_tokens=1024):
    """Run the full counterfactual experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load data
    print("Loading data...")
    training_data = []
    with open(DATA_FILE) as f:
        for line in f:
            training_data.append(json.loads(line))

    fm = pd.read_csv(FM_CSV)
    fm['core_regimen'] = fm['drugs'].apply(norm_drug)

    all_results = []

    for gene, cancer, regimen, expected_delta in PAIRS:
        print(f"\n{'='*60}")
        print(f"Pair: {gene} → {regimen} ({cancer}), expected Δ={expected_delta}")
        print(f"{'='*60}")

        mut_col = f'mut_{gene}'

        # Find samples: cancer match + regimen match + gene mutated
        mask = (fm['cancer_type'] == cancer) & (fm['core_regimen'] == regimen)
        if mut_col in fm.columns:
            mask = mask & (fm[mut_col] == 1)

        candidates = fm[mask]
        print(f"  Candidates: {len(candidates)}")

        if len(candidates) < 10:
            print(f"  Too few candidates, skip")
            continue

        # Sample up to N_SAMPLES_PER_PAIR
        sampled = candidates.sample(min(N_SAMPLES_PER_PAIR, len(candidates)),
                                     random_state=42)

        # Group by fold for efficient model loading
        fold_groups = sampled.groupby('fold')

        for fold, fold_samples in fold_groups:
            fold = int(fold)
            print(f"\n  Fold {fold}: {len(fold_samples)} samples")

            # Load model
            model_path = MERGED_DIR(fold)
            print(f"  Loading model from {model_path}...")

            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16,
                device_map=device, output_hidden_states=True
            )
            model.eval()

            score_head = torch.nn.Sequential(
                torch.nn.Linear(4096, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 1),
                torch.nn.Sigmoid()
            ).to(device).half()
            score_head.load_state_dict(torch.load(f'{model_path}/score_head.pt',
                                                   map_location=device))
            score_head.eval()

            for idx, (_, row) in enumerate(fold_samples.iterrows()):
                sample_idx = int(row['sample_idx'])
                d = training_data[sample_idx]

                # Build prompt
                messages = [
                    {"role": "system", "content": d['instruction']},
                    {"role": "user", "content": d['input']},
                ]
                prompt_orig = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

                # Create masked input
                masked_input, was_masked = mask_mutation_in_input(d['input'], gene)
                if not was_masked:
                    continue

                messages_masked = [
                    {"role": "system", "content": d['instruction']},
                    {"role": "user", "content": masked_input},
                ]
                prompt_masked = tokenizer.apply_chat_template(
                    messages_masked, tokenize=False, add_generation_prompt=True)

                score_orig, text_orig = run_inference(
                    model, tokenizer, score_head, prompt_orig, device, max_new_tokens)
                score_masked, text_masked = run_inference(
                    model, tokenizer, score_head, prompt_masked, device, max_new_tokens)

                gene_in_orig = gene_mentioned_in_text(text_orig, gene)
                gene_in_masked = gene_mentioned_in_text(text_masked, gene)
                gene_disappeared = gene_in_orig and not gene_in_masked

                delta_score = score_masked - score_orig
                if expected_delta == 'down':
                    score_correct = delta_score < 0
                else:
                    score_correct = delta_score > 0

                all_results.append({
                    'gene': gene, 'cancer': cancer, 'regimen': regimen,
                    'sample_idx': sample_idx, 'fold': fold,
                    'score_orig': score_orig, 'score_masked': score_masked,
                    'delta_score': delta_score,
                    'gene_in_orig': gene_in_orig,
                    'gene_in_masked': gene_in_masked,
                    'gene_disappeared': gene_disappeared,
                    'score_correct': score_correct,
                    'expected_delta': expected_delta,
                })

                if (idx + 1) % 10 == 0:
                    print(f"    {idx+1}/{len(fold_samples)}: "
                          f"Δscore={delta_score:+.3f}, "
                          f"gene_in_orig={gene_in_orig}, "
                          f"gene_in_masked={gene_in_masked}")

            del model, score_head
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUTPUT_DIR}/counterfactual_results.csv', index=False)
    print(f"\nSaved {len(df)} results to counterfactual_results.csv")

    # Summary
    if len(df) > 0:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Gene':8s} {'Regimen':15s} {'N':>4s} {'Δscore_ok':>10s} "
              f"{'MentOrig':>9s} {'MentMask':>9s} {'Disapp':>7s}")
        print('-' * 65)
        for (gene, regimen), grp in df.groupby(['gene', 'regimen']):
            n = len(grp)
            score_ok = grp['score_correct'].mean()
            orig_mention = grp['gene_in_orig'].mean()
            mask_mention = grp['gene_in_masked'].mean()
            disappear = grp[grp['gene_in_orig']]['gene_disappeared'].mean() \
                        if grp['gene_in_orig'].sum() > 0 else float('nan')
            print(f"{gene:8s} {regimen[:15]:15s} {n:4d} {score_ok:10.0%} "
                  f"{orig_mention:9.0%} {mask_mention:9.0%} {disappear:7.0%}")

        orig_mentioned = df['gene_in_orig'].sum()
        disappeared = df[df['gene_in_orig']]['gene_disappeared'].sum()
        if orig_mentioned > 0:
            print(f"\nOverall gene disappearance rate: "
                  f"{disappeared}/{orig_mentioned} = {disappeared/orig_mentioned:.1%}")
        print(f"Overall Δscore correct: {df['score_correct'].mean():.1%}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()

    run_counterfactual_experiment(device=args.device,
                                  max_new_tokens=args.max_new_tokens)

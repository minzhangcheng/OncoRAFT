#!/usr/bin/env python3
"""Unified clinical / biomarker counterfactual experiment.

One script handles every single-variable clinical/biomarker perturbation,
each with two conditions (C0_original, C_swap). The output schema is
identical across variables so downstream aggregation runs in one sweep.

Supported variables (--variable):
  Disease state and extent:
    stage              : Derived Stage: Stage 4 -> Stage 1-3      (cohort: stage IV)
    metastatic         : Summary: Distant metastases -> Localized (cohort: distant-metastatic)
    liver_metastasis   : Delete Liver from Metastatic Site list   (cohort: any liver-mets)
  Predictive biomarker flip:
    hr                 : HR Status: Positive <-> Negative          (cohort: breast HR+)
    her2               : HER2 Status: Positive -> Negative         (cohort: breast HER2+)
    msi                : MSI-H -> MSS (flips Type/Score/MMR/Comment)(cohort: MSI-H)
    tmb                : TMB high (>=10) -> 2.0                    (cohort: TMB high)
  Prognostic markers:
    gleason            : Gleason Score: 9 -> 6                     (cohort: prostate high-grade)
    cea                : CEA: >10 -> 2.0                           (cohort: CRC CEA-high)
    ca199              : CA 19-9: >35 -> 10                        (cohort: pancreatic high)
    ca153              : CA 15-3: >30 -> 15                        (cohort: breast CA15-3 high)

Cohort selection:
  --cohort auto     : use the default cohort for the variable
  --cohort all      : no cohort filter
  --cohort <cancer> : restrict to a specific cancer type

CLI examples:
  python3 exp_clinical_swap.py --variable stage    --device cuda:0
  python3 exp_clinical_swap.py --variable hr       --device cuda:1
  python3 exp_clinical_swap.py --variable gleason  --device cuda:2
"""
import os, sys, json, re, argparse, warnings, gc
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

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
# ============================================================
# Per-variable swap functions. Each returns (new_text, changed).
# ============================================================
def swap_stage(text):
    """Stage 4 -> Stage 1-3 (localized)."""
    out, ch = text, False
    if 'Derived Stage: Stage 4' in out:
        out = out.replace('Derived Stage: Stage 4', 'Derived Stage: Stage 1-3'); ch = True
    # Also update AJCC stage if present
    out2 = re.sub(r'AJCC Stage:\s*IV[A-C]?', 'AJCC Stage: IA', out)
    if out2 != out:
        ch = True; out = out2
    # Summary if metastatic
    if 'Summary: Distant metastases/systemic disease' in out:
        out = out.replace('Summary: Distant metastases/systemic disease',
                          'Summary: Localized'); ch = True
    return out, ch


def swap_metastatic(text):
    """Distant metastases -> Localized."""
    out, ch = text, False
    pats = [
        ('Summary: Distant metastases/systemic disease', 'Summary: Localized'),
        ('Summary: Distant', 'Summary: Localized'),
        ('Summary: Regional to lymph nodes', 'Summary: Localized'),
        ('Summary: Regional, lymph nodes only', 'Summary: Localized'),
    ]
    for old, new in pats:
        if old in out:
            out = out.replace(old, new); ch = True
            break  # one replacement only
    return out, ch


def swap_hr(text):
    """HR Status: Positive <-> Negative."""
    out = text
    if 'HR Status: Positive' in out:
        return out.replace('HR Status: Positive', 'HR Status: Negative'), True
    if 'HR Status: Negative' in out:
        return out.replace('HR Status: Negative', 'HR Status: Positive'), True
    return text, False


def swap_gleason(text):
    """Gleason Score: 9 -> 6 (or any >=7 -> 6)."""
    m = re.search(r'Gleason Score:\s*(\d+)', text)
    if m and int(m.group(1)) >= 7:
        return re.sub(r'Gleason Score:\s*\d+',
                      'Gleason Score: 6', text, count=1), True
    return text, False


def swap_cea(text):
    """CEA: high (>10) -> 2.0 ng/ml."""
    pat = r'CEA:\s*([\d.]+)\s*ng/ml'
    m = re.search(pat, text)
    if m and float(m.group(1)) > 10:
        # Replace all occurrences with 2.0 ng/ml (patients may have multiple CEA measurements)
        return re.sub(pat, 'CEA: 2.0 ng/ml', text), True
    return text, False


def swap_ca199(text):
    pat = r'CA 19-9:\s*([\d.]+)\s*Units/ml'
    m = re.search(pat, text)
    if m and float(m.group(1)) > 35:
        return re.sub(pat, 'CA 19-9: 10 Units/ml', text), True
    return text, False


def swap_ca153(text):
    pat = r'CA 15-3:\s*([\d.]+)\s*Units/ml'
    m = re.search(pat, text)
    if m and float(m.group(1)) > 30:
        return re.sub(pat, 'CA 15-3: 15 Units/ml', text), True
    return text, False


def swap_liver_metastasis(text):
    """Delete 'Liver' from any 'Metastatic Site:' line. If liver is the sole
    site, replace the line value with 'Lymph Node'."""
    out, ch = text, False
    def _sub(m):
        nonlocal ch
        raw = m.group(1).strip()
        parts = [p.strip() for p in re.split(r'[,;/]', raw) if p.strip()]
        kept = [p for p in parts if p.lower() != 'liver' and 'hepatic' not in p.lower()]
        if len(kept) == len(parts):
            return m.group(0)  # no liver present, unchanged
        ch = True
        if kept:
            return 'Metastatic Site: ' + ', '.join(kept)
        return 'Metastatic Site: Lymph Node'
    out = re.sub(r'Metastatic Site:\s*([^\n]+)', _sub, text)
    return out, ch


def swap_msi(text):
    """MSI-H -> MSS. Flip MSI Type, MSI Score (to low), MMR (Absent=True->False),
    and the MSI Comment line if present."""
    out, ch = text, False
    if 'MSI Type: Instable' in out:
        out = out.replace('MSI Type: Instable', 'MSI Type: Stable'); ch = True
    # MSI Score: numeric -> 0.5 (well below 10 MSI-H threshold)
    m = re.search(r'MSI Score:\s*([\d.]+)', out)
    if m and float(m.group(1)) >= 10:
        out = re.sub(r'MSI Score:\s*[\d.]+', 'MSI Score: 0.50', out, count=1)
        ch = True
    # MSI Comment often says MICROSATELLITE INSTABILITY
    if 'MICROSATELLITE INSTABILITY' in out.upper() or 'MSI-H' in out.upper():
        out = re.sub(r'MSI Comment:\s*[^\n]+',
                     'MSI Comment: MICROSATELLITE STABLE (MSS).', out, count=1)
        ch = True
    # MMR Absent: True -> False
    out2 = re.sub(r'(MMR (?:Protein )?Absent:\s*)True',
                  r'\1False', out)
    if out2 != out:
        ch = True; out = out2
    return out, ch


def swap_tmb(text):
    """High TMB -> low TMB (drop to 2.0)."""
    m = re.search(r'TMB \(nonsynonymous\):\s*([\d.]+)', text)
    if not m:
        return text, False
    if float(m.group(1)) < 10:
        return text, False
    return re.sub(r'TMB \(nonsynonymous\):\s*[\d.]+',
                  'TMB (nonsynonymous): 2.00', text, count=1), True


def swap_her2(text):
    """HER2 Status: Positive -> Negative."""
    if 'HER2 Status: Positive' in text:
        return text.replace('HER2 Status: Positive', 'HER2 Status: Negative'), True
    return text, False


SWAPS = {
    'stage':            swap_stage,
    'metastatic':       swap_metastatic,
    'hr':               swap_hr,
    'gleason':          swap_gleason,
    'cea':              swap_cea,
    'ca199':            swap_ca199,
    'ca153':            swap_ca153,
    'liver_metastasis': swap_liver_metastasis,
    'msi':              swap_msi,
    'tmb':              swap_tmb,
    'her2':             swap_her2,
}


# ============================================================
# Cohort selection
# ============================================================
def build_cohort(variable, cohort_spec, fm, training, max_n, seed):
    """Return a DataFrame of (sample_idx, fold, cancer_type) for the experiment."""
    # Index helpers
    sidx_all = set(range(len(training)))

    def has(text_tests, idx):
        t = training[idx].get('input', '')
        return all(tt in t if isinstance(tt, str) else tt(t) for tt in text_tests)

    # Default cohort per variable
    default_cohorts = {
        'stage':      ('filter_field', ['Derived Stage: Stage 4']),
        'metastatic': ('filter_field', ['Summary: Distant metastases/systemic disease']),
        'hr':         ('cancer_and_field', ('Breast', ['HR Status: Positive'])),
        'gleason':    ('cancer_and_field', ('Prostate', [lambda t: bool(re.search(r'Gleason Score:\s*[7-9]', t)) or 'Gleason Score: 10' in t])),
        'cea':        ('cancer_and_field', ('Colorectal', [lambda t: any(float(m.group(1)) > 10 for m in re.finditer(r'CEA:\s*([\d.]+)\s*ng/ml', t))])),
        'ca199':      ('cancer_and_field', ('Pancreatic', [lambda t: any(float(m.group(1)) > 35 for m in re.finditer(r'CA 19-9:\s*([\d.]+)\s*Units/ml', t))])),
        'ca153':      ('cancer_and_field', ('Breast', [lambda t: any(float(m.group(1)) > 30 for m in re.finditer(r'CA 15-3:\s*([\d.]+)\s*Units/ml', t))])),
        'liver_metastasis': ('filter_field', [lambda t: bool(re.search(r'Metastatic Site:[^\n]*Liver', t))]),
        'msi':        ('filter_field', ['MSI Type: Instable']),
        'tmb':        ('filter_field', [lambda t: any(float(m.group(1)) >= 10 for m in re.finditer(r'TMB \(nonsynonymous\):\s*([\d.]+)', t))]),
        'her2':       ('filter_field', ['HER2 Status: Positive']),
    }

    mode, params = default_cohorts.get(variable, ('all', []))

    # Build eligible sample_idx set
    if mode == 'all':
        eligible = sidx_all
    elif mode == 'filter_field':
        tests = params
        eligible = set()
        for i in sidx_all:
            t = training[i].get('input', '')
            if all((tt in t) if isinstance(tt, str) else tt(t) for tt in tests):
                eligible.add(i)
    elif mode == 'cancer_and_field':
        cancer, field_tests = params
        eligible = set()
        for i in sidx_all:
            t = training[i].get('input', '')
            if f'Cancer Type: {cancer}' not in t and cancer not in str(
                    fm[fm['sample_idx'] == i]['cancer_type'].values):
                continue
            if all((tt in t) if isinstance(tt, str) else tt(t) for tt in field_tests):
                eligible.add(i)
    print(f'Eligible sample_idx count: {len(eligible)}')

    # Filter feature_matrix to eligible
    cohort = fm[fm['sample_idx'].isin(eligible)].copy()
    if len(cohort) > max_n:
        cohort = cohort.sample(n=max_n, random_state=seed)
        print(f'  subsampled to {len(cohort)}')
    return cohort


def run_inference(model, tokenizer, score_head, prompt_text, device,
                  max_new_tokens=1024):
    inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                       max_length=1536).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    pos = inputs['attention_mask'].sum().item() - 1
    predicted = score_head(hidden[0, pos, :].unsqueeze(0)).item()
    score_str = f"{predicted:.2f}"
    gen_prompt = prompt_text + score_str + "\nReasoning:\n"
    gen_inputs = tokenizer(gen_prompt, return_tensors='pt', truncation=True,
                           max_length=1536 + 20).to(device)
    with torch.no_grad():
        gen_outputs = model.generate(
            **gen_inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = gen_outputs[0][gen_inputs['input_ids'].shape[1]:]
    return predicted, f"{score_str}\nReasoning:\n{tokenizer.decode(new_tokens, skip_special_tokens=True)}"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--variable', required=True, choices=list(SWAPS.keys()))
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--tag', default='clinical_swap')
    p.add_argument('--max_n', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fold', type=int, default=-1)
    p.add_argument('--max_text_chars', type=int, default=1500)
    args = p.parse_args()

    SUPP = {'ZOLEDRONIC ACID','PAMIDRONATE','DENOSUMAB','LEUPROLIDE','GOSERELIN',
            'DEGARELIX','LEUCOVORIN','PREDNISONE','INVESTIGATIONAL','MEGESTROL'}
    def norm(d):
        if pd.isna(d): return None
        ps = [x.strip().upper() for x in d.split(',')]
        core = [x for x in ps if x not in SUPP]
        return ', '.join(sorted(core)) if core else None

    print(f"Variable: {args.variable}")
    print("Loading data...")
    training = []
    with open(DATA_FILE) as f:
        for line in f:
            training.append(json.loads(line))
    fm = pd.read_csv(FM_CSV)
    fm['core_regimen'] = fm['drugs'].apply(norm)

    cohort = build_cohort(args.variable, None, fm, training, args.max_n, args.seed)
    if args.fold >= 0:
        cohort = cohort[cohort['fold'] == args.fold]

    print(f"Cohort size: {len(cohort)}")
    if len(cohort) < 3:
        print("Too few, exit"); return
    print(cohort['cancer_type'].value_counts().head(5).to_string())

    out_dir = f"{OUTPUT_DIR}/{args.tag}_{args.variable}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = (f'{out_dir}/counterfactual_results_fold{args.fold}.csv'
               if args.fold >= 0 else
               f'{out_dir}/counterfactual_results.csv')

    conds = ['C0_original', 'C_swap']
    swap_fn = SWAPS[args.variable]
    all_results = []

    for fold, fold_samples in cohort.groupby('fold'):
        fold = int(fold)
        print(f"\n  Fold {fold}: {len(fold_samples)} samples")
        mp = MERGED_DIR(fold)
        tok = AutoTokenizer.from_pretrained(mp, padding_side='left')
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mp, dtype=torch.float16, device_map=args.device,
            output_hidden_states=True)
        model.eval()
        score_head = torch.nn.Sequential(
            torch.nn.Linear(4096, 512), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1), torch.nn.Sigmoid()
        ).to(args.device).half()
        score_head.load_state_dict(torch.load(
            f'{mp}/score_head.pt', map_location=args.device))
        score_head.eval()

        for pi, (_, row) in enumerate(fold_samples.iterrows()):
            sidx = int(row['sample_idx'])
            cancer = row['cancer_type']
            drug = row['core_regimen']
            d = training[sidx]
            for cond in conds:
                if cond == 'C0_original':
                    mod_input, ch = d['input'], True
                else:
                    mod_input, ch = swap_fn(d['input'])
                messages = [{"role": "system", "content": d['instruction']},
                            {"role": "user", "content": mod_input}]
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                score, text = run_inference(
                    model, tok, score_head, prompt, args.device,
                    args.max_new_tokens)
                all_results.append({
                    'sample_idx': sidx, 'fold': fold,
                    'variable': args.variable,
                    'drug': drug, 'cancer_type': cancer,
                    'condition': cond, 'score': score,
                    'text_changed': bool(ch),
                    'text': text[:args.max_text_chars],
                })
            if (pi + 1) % 10 == 0:
                print(f"    patient {pi+1}/{len(fold_samples)}")
                pd.DataFrame(all_results).to_csv(out_csv, index=False)

        del model, score_head
        gc.collect(); torch.cuda.empty_cache()
        pd.DataFrame(all_results).to_csv(out_csv, index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(out_csv, index=False)

    print(f"\n===== Summary: variable={args.variable} =====")
    pivot = df.pivot_table(index='sample_idx', columns='condition', values='score')
    changed_df = df[df['condition'] == 'C_swap'].drop_duplicates('sample_idx')
    n_changed = changed_df['text_changed'].sum()
    print(f"  text_changed rate: {n_changed}/{len(changed_df)} = {n_changed/max(len(changed_df),1):.0%}")
    if 'C0_original' in pivot.columns and 'C_swap' in pivot.columns:
        delta = (pivot['C_swap'] - pivot['C0_original'])
        # mask to only patients where swap actually changed text
        eff_idx = set(changed_df[changed_df['text_changed']]['sample_idx'].astype(int))
        delta_eff = delta[delta.index.isin(eff_idx)]
        print(f"  ΔS (all patients)             : mean={delta.mean():+.4f}  n={delta.notna().sum()}")
        print(f"  ΔS (text_changed=True only)   : mean={delta_eff.mean():+.4f}  n={delta_eff.notna().sum()}")
        print(f"  dir(down) all: {(delta<0).mean():.0%}  dir(down) eff: {(delta_eff<0).mean():.0%}")


if __name__ == '__main__':
    main()

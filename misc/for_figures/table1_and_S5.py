from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_fp(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)

def tanimoto(s1, s2):
    fp1, fp2 = get_fp(s1), get_fp(s2)
    if fp1 is None or fp2 is None:
        return float('nan')
    return DataStructs.TanimotoSimilarity(fp1, fp2)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = _REPO_ROOT / 'STRIPES2SMILES' / 'results_finetune'
datasets = ['JAK1', 'AR', 'PIM1', 'PPAR']

for dataset in datasets:
    path = f'{BASE}/{dataset}_finetuned/comparison/generated_beam15_N5_T1.4_step0.5.csv'
    df = pd.read_csv(path)

    # compute Tanimoto similarity between reference and generated SMILES
    df['tan_sim'] = [tanimoto(s1, s2) for s1, s2 in zip(df['can_smiles'], df['canonical_smiles'])]
    df.to_csv(path, index=False)
    print(f"[{dataset}] tan_sim computed and CSV updated.")

    valid = df.dropna(subset=['tan_sim'])
    total = len(df)
    n_valid = len(valid)
    print(f"\n=== Dataset: {dataset} ===")
    print(f"  Total rows: {total}  |  Valid (fp OK): {n_valid}")
    print(f"  {'Bin':<15} {'N':>6}  {'%':>7}")
    print(f"  {'-'*32}")

    for label, mask in [
        ('== 1.0',   valid['tan_sim'] == 1.0),
        ('>= 0.8',   valid['tan_sim'] >= 0.8),
        ('>= 0.6',   valid['tan_sim'] >= 0.6),
    ]:
        n = mask.sum()
        print(f"  {label:<15} {n:>6}  {n/n_valid*100:>6.2f}%")

    print(f"  Mean tan_sim: {valid['tan_sim'].mean():.4f}  |  Median: {valid['tan_sim'].median():.4f}")

# --- Validity / Uniqueness / Novelty per rank (1..5) ---
# Definitions (from generate.py):
#   validity   = is_valid / total
#   uniqueness = unique valid canonical_smiles / all valid canonical_smiles
#   novelty    = unique valid canonical_smiles NOT in reference (can_smiles) / unique valid canonical_smiles

def compute_metrics_for_subset(subset: pd.DataFrame, reference_set: set) -> dict:
    total        = len(subset)
    valid_df     = subset[subset['is_valid']]
    canon_smiles = valid_df['canonical_smiles'].dropna().tolist()
    unique_valid = set(canon_smiles)
    novel        = {s for s in unique_valid if s not in reference_set}
    return {
        'total':      total,
        'valid':      len(valid_df),
        'validity':   len(valid_df) / total if total else 0.0,
        'unique':     len(unique_valid),
        'uniqueness': len(unique_valid) / len(canon_smiles) if canon_smiles else 0.0,
        'novel':      len(novel),
        'novelty':    len(novel) / len(unique_valid) if unique_valid else 0.0,
    }

for dataset in datasets:
    path = f'{BASE}/{dataset}_finetuned/comparison/generated_beam15_N5_T1.4_step0.5.csv'
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[{dataset}] file not found, skipping.")
        continue

    reference_set = set(df['can_smiles'].dropna().unique())

    print(f"\n=== Dataset: {dataset} — Validity / Uniqueness / Novelty per rank ===")
    print(f"  {'rank':>5}  {'total':>6}  {'valid':>6}  {'validity':>9}  {'unique':>7}  {'uniqueness':>11}  {'novel':>7}  {'novelty':>9}")
    print(f"  {'-'*80}")

    for rank in sorted(df['rank'].unique()):
        subset = df[df['rank'] == rank]
        m = compute_metrics_for_subset(subset, reference_set)
        print(f"  {rank:>5}  {m['total']:>6}  {m['valid']:>6}  {m['validity']:>9.4f}  "
              f"{m['unique']:>7}  {m['uniqueness']:>11.4f}  {m['novel']:>7}  {m['novelty']:>9.4f}")

# --- Tanimoto similarity distribution per rank ---

for dataset in datasets:
    path = f'{BASE}/{dataset}_finetuned/comparison/generated_beam15_N5_T1.4_step0.5.csv'
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[{dataset}] file not found, skipping.")
        continue

    print(f"\n=== Dataset: {dataset} — Tanimoto similarity per rank ===")
    print(f"  {'rank':>5}  {'n_valid':>7}  {'mean':>7}  {'median':>7}  {'== 1.0':>14}  {'>= 0.8':>14}  {'>= 0.6':>14}")
    print(f"  {'-'*80}")

    for rank in sorted(df['rank'].unique()):
        sub = df[df['rank'] == rank].dropna(subset=['tan_sim'])
        n = len(sub)
        if n == 0:
            print(f"  {rank:>5}  {'0':>7}")
            continue

        ts = sub['tan_sim']
        c1  = (ts == 1.0).sum()
        c08 = (ts >= 0.8).sum()
        c06 = (ts >= 0.6).sum()

        def fmt(c): return f"{c:>5} ({c/n*100:>5.1f}%)"

        print(f"  {rank:>5}  {n:>7}  {ts.mean():>7.4f}  {ts.median():>7.4f}  "
              f"{fmt(c1):>14}  {fmt(c08):>14}  {fmt(c06):>14}")

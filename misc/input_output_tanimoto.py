"""
For each dataset, take the molecules found on PubChem/ChEMBL
(from PubChem_analysis/results/{DATASET}_bioactivity_results.csv),
find all (input can_smiles → output canonical_smiles) pairs across
every generated_*.csv in results_finetune/{DATASET}_finetuned/comparison/,
compute Tanimoto similarity between input and output, and save one CSV
per dataset with all unique pairs.

Output: PubChem_analysis/results/input_output_tanimoto/{DATASET}_input_output_tanimoto.csv
"""

import os
import glob
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

BIOACT_DIR     = str(BASE / 'PubChem_analysis' / 'results')
FINETUNE_DIR   = str(BASE / 'STRIPES2SMILES' / 'results_finetune')
OUT_DIR        = str(BASE / 'PubChem_analysis' / 'results' / 'input_output_tanimoto')

DATASETS = ['AR', 'JAK1', 'PIM1', 'PPAR']

os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def canonicalize(smi):
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(smi_a, smi_b):
    fp_a = morgan_fp(smi_a)
    fp_b = morgan_fp(smi_b)
    if fp_a is None or fp_b is None:
        return None
    return FingerprintSimilarity(fp_a, fp_b)


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    # 1. Load molecules found on PubChem / ChEMBL
    bioact_path = os.path.join(BIOACT_DIR, f'{dataset}_bioactivity_results.csv')
    if not os.path.isfile(bioact_path):
        print(f"  [SKIP] Bioactivity file not found: {bioact_path}")
        continue

    df_bio = pd.read_csv(bioact_path)

    # Keep only molecules with at least one bioactivity value
    ACTIVITY_COLS = ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']
    active_mask = df_bio[ACTIVITY_COLS].notna().any(axis=1)
    known_smiles = set(df_bio.loc[active_mask, 'canonical_smiles'].dropna().apply(canonicalize).dropna())
    print(f"  Molecules with at least one bioactivity value: {len(known_smiles)}")

    if not known_smiles:
        print("  [SKIP] No known molecules for this dataset.")
        continue

    # 2. Scan all generated_*.csv files in the comparison folder
    comparison_dir = os.path.join(FINETUNE_DIR, f'{dataset}_finetuned', 'comparison')
    csv_files = sorted(glob.glob(os.path.join(comparison_dir, 'generated_*.csv')))
    print(f"  Comparison files found: {len(csv_files)}")

    # Collect unique (can_smiles, canonical_smiles) pairs
    pairs = set()

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] Could not read {os.path.basename(csv_path)}: {e}")
            continue

        # Keep only valid generated molecules
        df = df[df['is_valid'] == True].copy()
        df['canonical_smiles'] = df['canonical_smiles'].apply(canonicalize)
        df['can_smiles']       = df['can_smiles'].apply(canonicalize)

        # Filter to molecules that are in PubChem/ChEMBL
        mask = df['canonical_smiles'].isin(known_smiles)
        matched = df[mask][['can_smiles', 'canonical_smiles']].dropna()

        for _, row in matched.iterrows():
            pairs.add((row['can_smiles'], row['canonical_smiles']))

    print(f"  Unique (input, output) pairs involving known molecules: {len(pairs)}")

    if not pairs:
        print("  [SKIP] No pairs found.")
        continue

    # 3. Compute Tanimoto similarity for each pair
    rows = []
    for idx, (inp, out) in enumerate(sorted(pairs)):
        sim = tanimoto(inp, out)
        rows.append({
            'dataset':            dataset,
            'input_can_smiles':   inp,
            'output_canonical_smiles': out,
            'tanimoto_input_output': sim,
        })
        if (idx + 1) % 20 == 0 or (idx + 1) == len(pairs):
            print(f"  [{idx+1}/{len(pairs)}] done")

    df_out = pd.DataFrame(rows)

    # 4. (Optional) Merge bioactivity columns from bioact file
    bioact_cols = ['canonical_smiles', 'exists_on_pubchem', 'exists_on_chembl',
                   'EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']
    available   = [c for c in bioact_cols if c in df_bio.columns]
    df_bio_merge = df_bio[available].copy()
    df_bio_merge['canonical_smiles'] = df_bio_merge['canonical_smiles'].apply(canonicalize)
    df_bio_merge = df_bio_merge.drop_duplicates(subset='canonical_smiles')

    df_out = df_out.merge(
        df_bio_merge.rename(columns={'canonical_smiles': 'output_canonical_smiles'}),
        on='output_canonical_smiles',
        how='left'
    )

    # 5. Save
    out_path = os.path.join(OUT_DIR, f'{dataset}_input_output_tanimoto.csv')
    df_out.to_csv(out_path, index=False)
    print(f"  Saved {len(df_out)} rows → {out_path}")

print("\nDone.")

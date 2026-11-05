"""Build an MD results.csv by joining the docking input table with the STRIPES
extracted from the MD trajectories.

    results.csv = MD/docking/<TARGET>/<TARGET>_filtered_for_docking.csv
                      (8 columns, carried over unchanged)
                + 'gen stripes'
                      (the 'STRIPES' column of the MD extraction)

joined on ligand_idx == ligand_id, inner: only the ligands that have an MD
stripe survive. Feed the result to similarity_function_hungarian.py to add the
'stripes_sim' column and obtain MD/<TARGET>/results_sim.csv.

    python MD/build_results.py \
        --docking    MD/docking/PIM1/PIM1_filtered_for_docking.csv \
        --md-stripes <md_run>/STRIPES/pim1_generated.csv \
        --output     MD/PIM1/results.csv
"""
import argparse
import pandas as pd

BASE_COLS = ['ligand_idx', 'mol_id', 'can_smiles', 'stripes',
             'rank', 'smiles', 'canonical_smiles', 'is_valid']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docking', required=True,
                        help="Path to <target>_filtered_for_docking.csv.")
    parser.add_argument('--md-stripes', required=True,
                        help="Path to the MD CSV with 'ligand_id' and 'STRIPES' columns.")
    parser.add_argument('--output', required=True,
                        help="Path to the output results.csv.")
    args = parser.parse_args()

    base = pd.read_csv(args.docking)
    md = pd.read_csv(args.md_stripes)

    missing = [c for c in BASE_COLS if c not in base.columns]
    if missing:
        raise SystemExit(f"Error: columns missing from {args.docking}: {missing}")
    for col in ['ligand_id', 'STRIPES']:
        if col not in md.columns:
            raise SystemExit(f"Error: column '{col}' not found in {args.md_stripes}. "
                             f"Available: {list(md.columns)}")

    md = md.rename(columns={'ligand_id': 'ligand_idx', 'STRIPES': 'gen stripes'})
    md = md[['ligand_idx', 'gen stripes']].drop_duplicates(subset='ligand_idx')

    df = base[BASE_COLS].merge(md, on='ligand_idx', how='inner')

    df.to_csv(args.output, index=False)

    print(f"Docking table: {len(base)} ligands")
    print(f"MD stripes:    {len(md)} ligands")
    print(f"Output:        {len(df)} rows -> {args.output}")

    dropped = sorted(set(base.ligand_idx) - set(md.ligand_idx))
    if dropped:
        print(f"Without MD stripes ({len(dropped)}): {dropped}")
    orphans = sorted(set(md.ligand_idx) - set(base.ligand_idx))
    if orphans:
        print(f"MD ids not in the docking table ({len(orphans)}): {orphans}")


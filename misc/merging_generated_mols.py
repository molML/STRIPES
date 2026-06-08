import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def merge_unique_molecules(folder: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(folder, "generated_*.csv"))

    seen = set()
    unique_smiles = []

    for f in sorted(csv_files):
        df = pd.read_csv(f)
        if "canonical_smiles" not in df.columns:
            print(f"Skipping {f}: no canonical_smiles column")
            continue
        valid = df[df["canonical_smiles"].notna() & (df["canonical_smiles"] != "")]
        before = len(unique_smiles)
        for smi in valid["canonical_smiles"]:
            if smi not in seen:
                seen.add(smi)
                unique_smiles.append(smi)
        print(f"{os.path.basename(f)}: +{len(unique_smiles) - before} new (total: {len(unique_smiles)})")

    return pd.DataFrame({"canonical_smiles": unique_smiles})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge generation-sweep CSVs into a single deduplicated list of unique canonical SMILES."
    )
    parser.add_argument(
        "--folder", required=True,
        help="Directory containing 'generated_*.csv' files "
             "(e.g. results_fine/<DATASET>_finetuned/comparison).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out_dir = Path(args.folder) / "all_generated_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_unique_molecules.csv"

    unique_df = merge_unique_molecules(args.folder)
    unique_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Total unique molecules: {len(unique_df)}")

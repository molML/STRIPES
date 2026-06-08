import pandas as pd
from pathlib import Path
from rdkit import Chem

_REPO_ROOT = Path(__file__).resolve().parent.parent


def to_canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def main():
    dataset_path = _REPO_ROOT / "data" / "PPAR" / "dataset.csv"

    df = pd.read_csv(dataset_path)

    df["can_smiles"] = df["smiles"].apply(to_canonical_smiles)

    n_failed = df["can_smiles"].isna().sum()
    if n_failed > 0:
        print(f"Warning: {n_failed} SMILES could not be parsed and were set to None.")

    df.to_csv(dataset_path, index=False)
    print(f"Saved {len(df)} rows to {dataset_path}")


if __name__ == "__main__":
    main()

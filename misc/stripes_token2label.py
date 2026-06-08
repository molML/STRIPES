import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"


def build_vocab(stripes_series: pd.Series) -> dict:
    vocab = {"<PAD>": 0}
    for stripes in stripes_series:
        for atom in stripes.split(';'):
            for token in atom.split('.'):
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "MISATO" / "dataset.csv")
    vocab = build_vocab(df['STRIPES'])

    out_path = DATA_DIR / "stripes_tokens2label.json"
    with open(out_path, 'w') as f:
        json.dump(vocab, f, indent=1)
    print(f"Saved {len(vocab)} tokens to {out_path}")

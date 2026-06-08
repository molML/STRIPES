"""
Run generation on test sets produced by run_finetuning.py, sweeping over
beam_size, n_molecules, initial_temperature, and temperature_increment.

The finetuned model is loaded ONCE per dataset and reused across all
parameter combinations — avoiding repeated checkpoint loading overhead.

Temperature strategy: descending — start at high T (diverse) and fall back
to lower T on retry if not enough valid molecules are found.

Constraint: beam_size >= n_molecules is enforced to ensure fair comparison
(you can't reliably fill N slots from fewer than N beam candidates).

Usage
-----
    python extract_test_and_generate.py \
        --pretrained_model results_pre/pretrained_stripes_encoder.pth \
        --pretrained_vocab results_pre/stripes_vocab.pkl \
        --results_dir      results_fine \
        --datasets         PIM1 JAK1 AR \
        --beam_sizes       5 10 15 \
        --n_molecules      5 10 \
        --temperatures     1.2 1.4 \
        --increments       -0.2 -0.3 -0.4 -0.5

Outputs (inside --results_dir/<DATASET>_finetuned/comparison/):
    generated_beam<b>_N<n>_T<t>_step<s>.csv
    generated_beam<b>_N<n>_T<t>_step<s>_metrics.json
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from finetuning import SMILESTranslator
from smiles_utils import is_valid_smiles, sanitize_smiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load test set saved by run_finetuning.py
# ---------------------------------------------------------------------------

def load_test_stripes(dataset_name: str, model_dir: Path) -> pd.DataFrame:
    """Load the test set saved by run_finetuning.py."""
    test_csv = model_dir / f"{dataset_name}_test_set.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test set not found: {test_csv}\n"
            "Run run_finetuning.py first to produce this file."
        )
    df = pd.read_csv(test_csv)
    if "STRIPES" in df.columns:
        df = df.rename(columns={"STRIPES": "stripes"})
    logger.info(f"{dataset_name}: {len(df)} test molecules loaded from {test_csv}")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, can_smiles_list: list) -> dict:
    """Compute validity, uniqueness, and novelty.

    - Validity:   fraction of all generated SMILES that are chemically valid.
    - Uniqueness: fraction of valid SMILES whose canonical form is unique across the batch.
    - Novelty:    fraction of unique canonical SMILES not found in the test set can_smiles.
    """
    reference_set = {s for s in can_smiles_list if s is not None}

    total        = len(df)
    valid_df     = df[df["is_valid"]]
    valid_smiles = valid_df["smiles"].tolist()
    canon_smiles = valid_df["canonical_smiles"].dropna().tolist()

    unique_valid = set(canon_smiles)
    novel        = {s for s in unique_valid if s not in reference_set}

    return {
        "total_generated":    total,
        "valid_count":        len(valid_smiles),
        "validity":           round(len(valid_smiles) / total,             4) if total        else 0.0,
        "unique_valid_count": len(unique_valid),
        "uniqueness":         round(len(unique_valid) / len(canon_smiles), 4) if canon_smiles else 0.0,
        "novel_count":        len(novel),
        "novelty":            round(len(novel) / len(unique_valid),        4) if unique_valid else 0.0,
        "novelty_reference":  "test set can_smiles",
    }


# ---------------------------------------------------------------------------
# Single combination
# ---------------------------------------------------------------------------

def run_combination(translator, test_df, out_csv, beam, n_mol, temp, step, max_attempts):
    """Run translate_batch for one parameter combination and save CSV + metrics."""
    stripes_seqs    = test_df["stripes"].tolist()
    can_smiles_list = test_df["can_smiles"].tolist() if "can_smiles" in test_df.columns else [None] * len(stripes_seqs)
    mol_ids         = test_df["mol_id"].tolist()     if "mol_id"     in test_df.columns else [None] * len(stripes_seqs)

    predictions = translator.translate_batch(
        stripes_seqs,
        beam_size=beam,
        n_molecules=n_mol,
        max_attempts=max_attempts,
        initial_temperature=temp,
        temperature_increment=step,
    )

    rows = []
    for mol_id, can_smi, stripes_seq, pred_list in zip(
        mol_ids, can_smiles_list, stripes_seqs, predictions
    ):
        for rank, smiles in enumerate(pred_list, start=1):
            valid    = is_valid_smiles(smiles) if smiles != "INVALID" else False
            canonical = sanitize_smiles(smiles, to_canonical=True) if valid else None
            rows.append({
                "mol_id":           mol_id,
                "can_smiles":       can_smi,
                "stripes":          stripes_seq,
                "rank":             rank,
                "smiles":           smiles,
                "canonical_smiles": canonical,
                "is_valid":         valid,
            })

    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    metrics = compute_metrics(df_out, can_smiles_list)
    metrics_path = out_csv.with_name(out_csv.stem + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"  → {out_csv.name}  "
        f"validity={metrics['validity']:.3f}  "
        f"uniqueness={metrics['uniqueness']:.3f}  "
        f"novelty={metrics['novelty']:.3f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    results_dir      = Path(args.results_dir)
    pretrained_model = args.pretrained_model
    pretrained_vocab = args.pretrained_vocab
    model_subdir     = args.model_subdir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build combinations — skip beam < n_mol (unfair comparison)
    combos = [
        (b, n, t, s)
        for b in args.beam_sizes
        for n in args.n_molecules
        if b >= n
        for t in args.temperatures
        for s in args.increments
    ]

    logger.info(f"{len(combos)} combinations per dataset (beam >= n_mol enforced)")

    for ds in args.datasets:
        # ── Locate finetuned model ──────────────────────────────────────
        model_dir  = results_dir / f"{ds}_finetuned"
        if model_subdir:
            model_dir = model_dir / model_subdir
        model_path = model_dir / f"{ds}_model.pth"
        if not model_path.exists():
            logger.warning(f"Model not found, skipping: {model_path}")
            continue

        # ── Load test set ───────────────────────────────────────────────
        try:
            test_df = load_test_stripes(ds, model_dir)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        out_dir = model_dir / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load model ONCE ─────────────────────────────────────────────
        logger.info(f"{ds}: loading model from {model_path}")
        checkpoint       = torch.load(model_path, map_location="cpu", weights_only=False)
        smiles_vocab     = checkpoint.get("smiles_vocab")
        pretrained_vocab_ckpt = checkpoint.get("pretrained_vocab")

        translator = SMILESTranslator(
            model_path=str(model_path),
            smiles_vocab=smiles_vocab     or {},
            pretrained_vocab=pretrained_vocab_ckpt or {},
            device=device,
            pretrained_encoder_path=pretrained_model,
            pretrained_vocab_path=pretrained_vocab,
            model_config=checkpoint.get("model_config"),
        )
        logger.info(f"{ds}: model loaded — running {len(combos)} combinations")

        # ── Sweep all combinations ──────────────────────────────────────
        for idx, (beam, n_mol, temp, step) in enumerate(combos, 1):
            step_abs = f"{abs(step):.1f}"
            out_csv  = out_dir / f"generated_beam{beam}_N{n_mol}_T{temp:.1f}_step{step_abs}.csv"

            logger.info(
                f"{ds} [{idx}/{len(combos)}] beam={beam} n={n_mol} T={temp:.1f} step={step:+.1f}"
            )
            run_combination(
                translator, test_df, out_csv,
                beam, n_mol, temp, step, args.max_attempts,
            )

        logger.info(f"{ds}: done.")

    logger.info("All datasets completed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SMILES from test sets sweeping generation parameters. "
                    "Model is loaded once per dataset."
    )
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--pretrained_vocab",  required=True)
    parser.add_argument("--results_dir",       required=True)
    parser.add_argument("--datasets", nargs="+", default=["PIM1", "JAK1", "AR"])
    parser.add_argument("--beam_sizes",  nargs="+", type=int,   default=[5, 10, 15])
    parser.add_argument("--n_molecules", nargs="+", type=int,   default=[5, 10],
                        help="beam_size >= n_molecules is enforced automatically.")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.2, 1.4],
                        help="Initial temperatures (descending strategy, default: 1.2 1.4).")
    parser.add_argument("--increments",   nargs="+", type=float, default=[-0.2, -0.3, -0.4, -0.5],
                        help="Temperature decrements per retry (default: -0.2 -0.3 -0.4 -0.5).")
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--model_subdir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

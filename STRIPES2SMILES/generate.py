"""
Standalone STRIPES → SMILES generation.

Load a finetuned model and generate SMILES molecules from STRIPES sequences
without running any training.

Usage
-----
Single sequence (command line):
    python generate.py \
        --finetuned_model  results_fine/PPAR_finetuned/PPAR_model.pth \
        --pretrained_model results_pre/pretrained_stripes_encoder.pth \
        --pretrained_vocab results_pre/stripes_vocab.pkl \
        --sequence "1.2.3.4.5.6.7.8.9.10.11;..." \
        --output generated.csv

From a CSV file (column 'stripes'):
    python generate.py \
        --finetuned_model  results_fine/PPAR_finetuned/PPAR_model.pth \
        --pretrained_model results_pre/pretrained_stripes_encoder.pth \
        --pretrained_vocab results_pre/stripes_vocab.pkl \
        --input_csv        my_sequences.csv \
        --output           generated.csv

From a plain text file (one STRIPES sequence per line):
    python generate.py \
        --finetuned_model  results_fine/PPAR_finetuned/PPAR_model.pth \
        --pretrained_model results_pre/pretrained_stripes_encoder.pth \
        --pretrained_vocab results_pre/stripes_vocab.pkl \
        --input_txt        my_sequences.txt \
        --output           generated.csv

The finetuned checkpoint already contains the SMILES vocabulary and the
pretrained STRIPES vocabulary, so no extra vocab files are needed beyond
--pretrained_vocab (required by STRIPESToSMILESModel to rebuild the encoder).

Output CSV columns
------------------
    stripes          — input STRIPES sequence
    rank             — molecule rank (1 = best beam score)
    smiles           — generated SMILES string
    is_valid         — True/False (RDKit validation)
"""

import argparse
import json
import logging
import pickle
import sys

import pandas as pd
import torch
from pathlib import Path

from finetuning import SMILESTranslator
from smiles_utils import is_valid_smiles, sanitize_smiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_sequences_from_csv(path: str):
    """Return (sequences, mol_ids, can_smiles_list) from a CSV file."""
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.lower() == "stripes"), None)
    if col is None:
        raise ValueError(f"CSV file '{path}' must contain a 'stripes' (or 'STRIPES') column.")
    df = df.dropna(subset=[col])
    sequences     = df[col].tolist()
    mol_ids       = df["mol_id"].tolist()      if "mol_id"     in df.columns else [None] * len(sequences)
    can_smiles    = df["can_smiles"].tolist()  if "can_smiles" in df.columns else [None] * len(sequences)
    logger.info(f"Loaded {len(sequences)} sequences from {path}")
    return sequences, mol_ids, can_smiles


def load_sequences_from_txt(path: str) -> list:
    with open(path) as f:
        seqs = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(seqs)} sequences from {path}")
    return seqs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, can_smiles_list: list) -> dict:
    """Compute validity, uniqueness, and novelty.

    - Validity:   fraction of all generated SMILES that are chemically valid.
    - Uniqueness: fraction of valid SMILES whose canonical form is unique across the batch.
    - Novelty:    fraction of unique canonical SMILES not found in the test set can_smiles.

    Uniqueness and novelty use canonical_smiles (not raw generated) to avoid counting
    the same molecule twice due to different SMILES representations.
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
        "validity":           round(len(valid_smiles) / total,            4) if total        else 0.0,
        "unique_valid_count": len(unique_valid),
        "uniqueness":         round(len(unique_valid) / len(canon_smiles), 4) if canon_smiles else 0.0,
        "novel_count":        len(novel),
        "novelty":            round(len(novel) / len(unique_valid),        4) if unique_valid else 0.0,
        "novelty_reference":  "test set can_smiles",
    }


def generate(args):
    # ── Collect input sequences ─────────────────────────────────────────
    mol_ids        = []
    can_smiles_list = []

    if args.sequence:
        sequences       = [args.sequence]
        mol_ids         = [None]
        can_smiles_list = [None]
    elif args.input_csv:
        sequences, mol_ids, can_smiles_list = load_sequences_from_csv(args.input_csv)
    elif args.input_txt:
        sequences       = load_sequences_from_txt(args.input_txt)
        mol_ids         = [None] * len(sequences)
        can_smiles_list = [None] * len(sequences)
    else:
        logger.error("Provide one of --sequence, --input_csv, or --input_txt.")
        sys.exit(1)

    if not sequences:
        logger.error("No sequences to process.")
        sys.exit(1)

    # ── Load vocabs from the finetuned checkpoint ───────────────────────
    # The checkpoint saved by FinetuningTrainer embeds both vocabularies.
    logger.info(f"Loading finetuned checkpoint: {args.finetuned_model}")
    checkpoint = torch.load(
        args.finetuned_model, map_location="cpu", weights_only=False
    )

    smiles_vocab     = checkpoint.get("smiles_vocab")
    pretrained_vocab = checkpoint.get("pretrained_vocab")

    if smiles_vocab is None or pretrained_vocab is None:
        # Fallback: load pretrained vocab from disk, SMILES vocab must be provided
        logger.warning(
            "Checkpoint does not embed vocabularies. "
            "Falling back to --pretrained_vocab. "
            "If you also need the SMILES vocab, pass --smiles_vocab."
        )
        with open(args.pretrained_vocab, "rb") as f:
            pretrained_vocab = pickle.load(f)

        if args.smiles_vocab:
            with open(args.smiles_vocab, "rb") as f:
                smiles_vocab = pickle.load(f)
        else:
            logger.error(
                "SMILES vocabulary not found in checkpoint and --smiles_vocab not provided."
            )
            sys.exit(1)

    model_config = checkpoint.get("model_config", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Build translator ────────────────────────────────────────────────
    translator = SMILESTranslator(
        model_path=args.finetuned_model,
        smiles_vocab=smiles_vocab,
        pretrained_vocab=pretrained_vocab,
        device=device,
        pretrained_encoder_path=args.pretrained_model,
        pretrained_vocab_path=args.pretrained_vocab,
        model_config=model_config,
    )

    # ── Generate ────────────────────────────────────────────────────────
    logger.info(
        f"Generating {args.n_molecules} molecule(s) per sequence "
        f"(beam_size={args.beam_size}, temperature={args.temperature})..."
    )
    all_predictions = translator.translate_batch(
        sequences,
        beam_size=args.beam_size,
        n_molecules=args.n_molecules,
        max_attempts=args.max_attempts,
        initial_temperature=args.temperature,
        temperature_increment=args.temperature_increment,
    )

    # ── Build output DataFrame ──────────────────────────────────────────
    rows = []
    for mol_id, can_smi, stripes_seq, pred_list in zip(
        mol_ids, can_smiles_list, sequences, all_predictions
    ):
        for rank, smiles in enumerate(pred_list, start=1):
            valid = is_valid_smiles(smiles) if smiles != "INVALID" else False
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    # ── Metrics (validity, uniqueness, novelty) ──────────────────────────
    metrics = compute_metrics(df_out, can_smiles_list)
    metrics_path = output_path.with_suffix("").with_suffix("") / (output_path.stem + "_metrics.json")
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(
        f"Validity={metrics['validity']:.3f}  "
        f"Uniqueness={metrics['uniqueness']:.3f}  "
        f"Novelty={metrics['novelty']:.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SMILES from STRIPES sequences using a finetuned model."
    )

    # Model paths (always required)
    parser.add_argument(
        "--finetuned_model", required=True,
        help="Path to the finetuned model checkpoint (.pth from run_finetuning.py).",
    )
    parser.add_argument(
        "--pretrained_model", required=True,
        help="Path to the pretrained encoder checkpoint (.pth from run_pretraining.py).",
    )
    parser.add_argument(
        "--pretrained_vocab", required=True,
        help="Path to the pretrained STRIPES vocabulary "
             "(stripes_vocab.pkl from run_pretraining.py).",
    )
    parser.add_argument(
        "--smiles_vocab", default=None,
        help="Path to the SMILES vocabulary (.pkl). "
             "Only needed if it is not embedded in the finetuned checkpoint.",
    )

    # Input (exactly one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence",
        help="A single STRIPES sequence string (passed directly on the command line).",
    )
    input_group.add_argument(
        "--input_csv",
        help="CSV file with a 'stripes' column.",
    )
    input_group.add_argument(
        "--input_txt",
        help="Plain text file with one STRIPES sequence per line.",
    )

    # Output
    parser.add_argument(
        "--output", default="generated_smiles.csv",
        help="Output CSV file path (default: generated_smiles.csv).",
    )

    # Generation parameters
    parser.add_argument(
        "--beam_size", type=int, default=15,
        help="Beam size for beam search (default: 15).",
    )
    parser.add_argument(
        "--n_molecules", type=int, default=5,
        help="Number of molecules to generate per sequence (default: 5).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for beam search (default: 1.0).",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=2,
        help="Max retry attempts if not enough valid molecules are found (default: 2).",
    )
    parser.add_argument(
        "--temperature_increment", type=float, default=-0.2,
        help="Temperature change per retry attempt. Negative = start diverse and fall back "
             "to focused (default: -0.2).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

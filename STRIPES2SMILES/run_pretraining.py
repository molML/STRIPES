"""
Usage

    python run_pretraining.py \
        --data_path  /path/to/MISATO \
        --output_dir ./results_pre

The directory passed to --data_path must contain a file named `dataset.csv`
with at least a column called `STRIPES` (or `stripes`).

Outputs (written to --output_dir)
-----------------------------------
    pretrained_stripes_encoder.pth  — best model checkpoint
    stripes_vocab.pkl               — vocabulary (needed for finetuning)
    training_metadata.json          — loss curves and model config
    pretraining_loss.png            — training/validation loss plot
"""

import argparse
import gc
import json
import logging
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.cuda
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pretraining import (
    STRIPESPretrainingDataset,
    STRIPESEncoder,
    STRIPESPretrainer,
    build_stripes_vocab,
    pretraining_collate_fn,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_stripes(data_path: str) -> list:
    dataset_path = Path(data_path) / "dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return []

    df = pd.read_csv(dataset_path)
    col = next((c for c in df.columns if c.lower() == "stripes"), None)
    if col is None:
        logger.error(f"Column 'STRIPES' (or 'stripes') not found in {dataset_path}")
        return []

    df_clean = df.dropna(subset=[col])
    df_clean = df_clean[df_clean[col].str.len() > 10]
    df_clean = df_clean.drop_duplicates(subset=[col])

    stripes = df_clean[col].tolist()
    logger.info(f"Collected {len(stripes)} unique STRIPES sequences")
    return stripes

def run_pretraining(args, device) -> tuple:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stripes = collect_stripes(args.data_path)
    if not all_stripes:
        logger.error("No STRIPES sequences found. Aborting.")
        return None, None

    vocab = build_stripes_vocab(all_stripes, min_freq=1, max_len=args.max_len)

    vocab_path = output_dir / "stripes_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"Vocabulary saved to {vocab_path}")

    train_seqs, val_seqs = train_test_split(
        all_stripes, test_size=0.1, random_state=args.seed
    )
    logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    # ── Datasets & DataLoaders ──────────────────────────────────────────
    train_dataset = STRIPESPretrainingDataset(
        train_seqs, vocab, max_len=args.max_len
    )
    val_dataset = STRIPESPretrainingDataset(
        val_seqs, vocab, max_len=args.max_len
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pretraining_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pretraining_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = STRIPESEncoder(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} trainable parameters")

    # ── Trainer (warmup + linear decay, BERT-style) ─────────────────────
    warmup_steps = int(0.06 * len(train_loader) * args.num_epochs)
    trainer = STRIPESPretrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        lr=args.lr,
        warmup_steps=warmup_steps,
        num_epochs=args.num_epochs,
        device=device,
    )

    model_path = output_dir / "pretrained_stripes_encoder.pth"
    train_losses, val_losses = trainer.pretrain(
        args.num_epochs, save_path=str(model_path)
    )

    # ── Metadata ─────────────────────────────────────────────────────────
    metadata = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_config": {
            "vocab_size": len(vocab),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dim_ff": args.dim_ff,
            "dropout": args.dropout,
            "max_len": args.max_len,
        },
        "training_stats": {
            "train_sequences": len(train_dataset),
            "val_sequences": len(val_dataset),
            "total_params": total_params,
        },
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("STRIPES MLM Pretraining")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.yscale("log")
    plt.legend()
    plt.title("STRIPES MLM Pretraining (log)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pretraining_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Pretraining completed. Model saved to {model_path}")
    return str(model_path), vocab



def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a STRIPES Transformer Encoder with MLM (BERT-style)."
    )
    # Required
    parser.add_argument(
        "--data_path", required=True,
        help="Directory containing dataset.csv with a 'STRIPES' (or 'stripes') column (e.g. MISATO/).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where checkpoints and artifacts will be saved.",
    )
    # Optional (with defaults)
    parser.add_argument("--max_len",    type=int,   default=1200)
    parser.add_argument("--d_model",   type=int,   default=512)
    parser.add_argument("--n_heads",   type=int,   default=8)
    parser.add_argument("--n_layers",  type=int,   default=8)
    parser.add_argument("--dim_ff",    type=int,   default=2048)
    parser.add_argument("--dropout",   type=float, default=0.1)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int,   default=100)
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    set_seed(args.seed)

    model_path, vocab = run_pretraining(args, device)
    if model_path and vocab:
        logger.info("=== PRETRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Model   : {model_path}")
        logger.info(f"Vocab   : {len(vocab)} tokens")
    else:
        logger.error("Pretraining failed.")

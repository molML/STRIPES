"""
Run STRIPES → SMILES Finetuning with Optuna hyperparameter search.

Pipeline
--------
1. Load the pretrained STRIPES encoder produced by run_pretraining.py
2. Build a SMILES vocabulary from all target datasets
3. Run Optuna (TPE + MedianPruner) to find the best hyperparameters
4. Final training with the best configuration

Generation and evaluation are handled separately by extract_test_and_generate.py.

Usage
-----
    python run_finetuning.py \
        --data_path       /path/to/data \
        --pretrained_model  results_pre/pretrained_stripes_encoder.pth \
        --pretrained_vocab  results_pre/stripes_vocab.pkl \
        --output_dir        results_fine \
        --datasets PPAR PIM1 JAK1 AR

Each dataset directory (e.g. data/PPAR/) must contain a `dataset.csv` file
with columns: stripes, can_smiles, mol_id, pKi.

Outputs (one sub-folder per dataset inside --output_dir)
----------------------------------------------------------
    <DATASET>_model.pth       — best finetuned model checkpoint
    <DATASET>_config.json     — model config + best Optuna params
    <DATASET>_optuna.json     — full Optuna trial log
    <DATASET>_split.json      — train/val/test split sizes
    <DATASET>_test_set.csv    — test set (mol_id, STRIPES, can_smiles) for generation
    <DATASET>_metrics.json    — training stats
    <DATASET>_plots.png       — training/validation loss curves
    smiles_vocab.pkl           — shared SMILES vocabulary
    summary.json               — aggregated metrics across all datasets
"""

import argparse
import gc
import json
import logging
import os
import pickle
import random
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from finetuning import (
    STRIPESToSMILESModel,
    FinetuningDataset,
    FinetuningTrainer,
    DATASET_MAX_LENGTHS,
)
from smiles_utils import segment_smiles, sanitize_smiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    stripes_seqs, smiles_seqs = zip(*batch)
    stripes_padded = pad_sequence(stripes_seqs, batch_first=True, padding_value=0)
    smiles_padded  = pad_sequence(smiles_seqs,  batch_first=True, padding_value=0)
    return stripes_padded, smiles_padded


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_smiles_vocab(dataset_names: list, data_base_path: str) -> dict:
    """Build a shared SMILES vocabulary from all target datasets."""
    smiles_tokens = set()
    for name in dataset_names:
        path = Path(data_base_path) / name / "dataset.csv"
        if not path.exists():
            logger.warning(f"Dataset not found, skipping for vocab: {path}")
            continue
        df = pd.read_csv(path)
        for smi in df.dropna(subset=["can_smiles"])["can_smiles"]:
            try:
                canonical = sanitize_smiles(smi, to_canonical=True)
                if canonical:
                    smiles_tokens.update(segment_smiles(canonical))
                else:
                    smiles_tokens.update(list(smi))
            except Exception:
                smiles_tokens.update(list(smi))

    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for token in sorted(smiles_tokens):
        if token and token.strip():
            vocab[token] = len(vocab)
    logger.info(f"SMILES vocabulary: {len(vocab)} tokens")
    return vocab


def load_and_split_by_mol_id(dataset_path: str, dataset_name: str):
    """Load dataset and split by mol_id to avoid data leakage."""
    df = pd.read_csv(dataset_path)
    if "mol_id" not in df.columns:
        df["mol_id"] = df.index

    df_clean = df.dropna(subset=["STRIPES", "can_smiles", "mol_id"])
    df_clean = df_clean[df_clean["STRIPES"].str.len() > 10]
    df_clean = df_clean[df_clean["can_smiles"].str.len() > 5]
    df_clean = df_clean[df_clean["pKi"] >= 6.0]
    df_clean = df_clean.reset_index(drop=True)

    unique_mols = df_clean["mol_id"].unique()
    train_val_mols, test_mols = train_test_split(
        unique_mols, test_size=0.15, random_state=42, shuffle=True
    )
    train_mols, val_mols = train_test_split(
        train_val_mols, test_size=0.1176, random_state=42, shuffle=True
    )

    train_idx = df_clean[df_clean["mol_id"].isin(train_mols)].index.tolist()
    val_idx   = df_clean[df_clean["mol_id"].isin(val_mols)].index.tolist()
    test_idx  = df_clean[df_clean["mol_id"].isin(test_mols)].index.tolist()

    logger.info(
        f"{dataset_name}: {len(unique_mols)} mol_ids → "
        f"train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}"
    )
    return df_clean, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, dataset_path, dataset_name, pretrained_model_path,
              pretrained_vocab_path, pretrained_vocab, smiles_vocab, device):
    lr              = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    freeze_layers   = trial.suggest_int("freeze_encoder_layers", 0, 4)
    n_decoder_layers = trial.suggest_int("n_decoder_layers", 4, 8)
    dropout         = trial.suggest_float("dropout", 0.1, 0.3)
    batch_size      = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay    = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.2)

    try:
        df_clean, train_idx, val_idx, _ = load_and_split_by_mol_id(
            dataset_path, dataset_name
        )
        if len(df_clean) < 100:
            return float("inf")

        stripes = df_clean["STRIPES"].tolist()
        smiles  = df_clean["can_smiles"].tolist()
        max_len = DATASET_MAX_LENGTHS.get(dataset_name, 700)

        train_ds = FinetuningDataset(
            [stripes[i] for i in train_idx], [smiles[i] for i in train_idx],
            pretrained_vocab, smiles_vocab, dataset_name, max_len,
        )
        val_ds = FinetuningDataset(
            [stripes[i] for i in val_idx], [smiles[i] for i in val_idx],
            pretrained_vocab, smiles_vocab, dataset_name, max_len,
        )

        g = torch.Generator().manual_seed(42)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2, generator=g,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2,
        )

        model = STRIPESToSMILESModel(
            pretrained_encoder_path=pretrained_model_path,
            pretrained_vocab_path=pretrained_vocab_path,
            smiles_vocab_size=len(smiles_vocab),
            freeze_encoder_layers=freeze_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
        )

        trainer = FinetuningTrainer(
            model, train_loader, val_loader,
            smiles_vocab, pretrained_vocab,
            device=device, lr=lr,
        )
        trainer.criterion = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing
        )

        enc_p, dec_p = [], []
        for name, p in trainer.model.named_parameters():
            if not p.requires_grad:
                continue
            (enc_p if "encoder" in name else dec_p).append(p)
        trainer.optimizer = optim.AdamW(
            [{"params": enc_p, "lr": lr * 0.1},
             {"params": dec_p, "lr": lr}],
            weight_decay=weight_decay,
        )

        warmup_steps = 3 * len(train_loader)
        trainer.warmup_scheduler = optim.lr_scheduler.LinearLR(
            trainer.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_steps,
        )
        trainer.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode="min", patience=5, factor=0.7, min_lr=1e-7,
        )
        trainer.scheduler    = trainer.plateau_scheduler
        trainer.warmup_steps = warmup_steps
        trainer.global_step  = 0

        best_val = float("inf")
        patience = 0
        for epoch in range(25):
            trainer.train_epoch()
            val_loss = trainer.validate()
            trainer.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if patience >= 3:
                break

        return best_val

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial error: {e}")
        traceback.print_exc()
        return float("inf")


# ---------------------------------------------------------------------------
# Main pipeline (single dataset)
# ---------------------------------------------------------------------------

def run_finetuning(dataset_name, dataset_path, pretrained_path,
                   pretrained_vocab_path, pretrained_vocab,
                   smiles_vocab, results_base_path, n_trials, device):
    """Full pipeline: Optuna search → final training → evaluation."""

    results_dir = Path(results_base_path) / f"{dataset_name}_finetuned"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Optuna ──────────────────────────────────────────────────────────
    logger.info(f"Starting Optuna optimisation for {dataset_name}...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{dataset_name}_optimisation",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=10),
        sampler=sampler,
    )
    study.optimize(
        lambda trial: objective(
            trial, dataset_path, dataset_name, pretrained_path,
            pretrained_vocab_path, pretrained_vocab, smiles_vocab, device,
        ),
        n_trials=n_trials,
        timeout=36000,
    )

    if not study.best_trial:
        logger.error("All Optuna trials failed.")
        return None

    best = study.best_trial.params
    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {best}")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    with open(results_dir / f"{dataset_name}_optuna.json", "w") as f:
        json.dump({
            "best_value":  study.best_trial.value,
            "best_params": best,
            "total_trials": len(study.trials),
            "completed":   len(completed),
            "pruned":      len(pruned),
            "completed_details": [
                {"params": t.params, "value": t.value, "number": t.number}
                for t in completed
            ],
        }, f, indent=2)

    # ── Final training ──────────────────────────────────────────────────
    logger.info(f"Final training for {dataset_name} with best params...")

    df_clean, train_idx, val_idx, test_idx = load_and_split_by_mol_id(
        dataset_path, dataset_name
    )
    stripes    = df_clean["STRIPES"].tolist()
    smiles_seqs = df_clean["can_smiles"].tolist()
    max_len    = DATASET_MAX_LENGTHS.get(dataset_name, 700)

    train_stripes = [stripes[i]     for i in train_idx]
    train_smiles  = [smiles_seqs[i] for i in train_idx]
    val_stripes   = [stripes[i]     for i in val_idx]
    val_smiles    = [smiles_seqs[i] for i in val_idx]
    test_stripes  = [stripes[i]     for i in test_idx]
    test_smiles   = [smiles_seqs[i] for i in test_idx]

    with open(results_dir / f"{dataset_name}_split.json", "w") as f:
        json.dump({"train": len(train_idx), "val": len(val_idx),
                   "test": len(test_idx), "max_len": max_len}, f, indent=2)

    # Save test set so extract_test_and_generate.py can load it directly
    test_df = df_clean.loc[test_idx, ["mol_id", "STRIPES", "can_smiles"]].copy()
    test_df = test_df.drop_duplicates(subset=["mol_id"])
    test_df.to_csv(results_dir / f"{dataset_name}_test_set.csv", index=False)
    logger.info(f"Test set saved: {len(test_df)} molecules")

    train_ds = FinetuningDataset(
        train_stripes, train_smiles, pretrained_vocab, smiles_vocab,
        dataset_name, max_len,
    )
    val_ds = FinetuningDataset(
        val_stripes, val_smiles, pretrained_vocab, smiles_vocab,
        dataset_name, max_len,
    )

    bs = best.get("batch_size", 32)
    g  = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=2, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    model = STRIPESToSMILESModel(
        pretrained_encoder_path=pretrained_path,
        pretrained_vocab_path=pretrained_vocab_path,
        smiles_vocab_size=len(smiles_vocab),
        freeze_encoder_layers=best.get("freeze_encoder_layers", 1),
        n_decoder_layers=best.get("n_decoder_layers", 6),
        dropout=best.get("dropout", 0.1),
    )

    trainer = FinetuningTrainer(
        model, train_loader, val_loader,
        smiles_vocab, pretrained_vocab,
        device=device, lr=best.get("lr", 1e-4),
    )
    trainer.criterion = nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=best.get("label_smoothing", 0.1),
    )

    enc_p, dec_p = [], []
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_p if "encoder" in name else dec_p).append(p)
    lr = best.get("lr", 1e-4)
    trainer.optimizer = optim.AdamW(
        [{"params": enc_p, "lr": lr * 0.1},
         {"params": dec_p, "lr": lr}],
        weight_decay=best.get("weight_decay", 1e-3),
    )

    warmup_steps = 5 * len(train_loader)
    trainer.warmup_scheduler = optim.lr_scheduler.LinearLR(
        trainer.optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_steps,
    )
    trainer.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", patience=5, factor=0.7, min_lr=1e-7,
    )
    trainer.scheduler    = trainer.plateau_scheduler
    trainer.warmup_steps = warmup_steps
    trainer.global_step  = 0

    model_path = results_dir / f"{dataset_name}_model.pth"
    trainer.fine_tune(
        num_epochs=500, save_path=str(model_path), early_stopping_patience=10,
    )

    model_config = {
        "pretrained_encoder_path": pretrained_path,
        "pretrained_vocab_path":   str(pretrained_vocab_path),
        "smiles_vocab_size":       len(smiles_vocab),
        "d_model":                 model.d_model,
        "n_decoder_layers":        best.get("n_decoder_layers", 6),
        "freeze_encoder_layers":   best.get("freeze_encoder_layers", 0),
        "dropout":                 best.get("dropout", 0.1),
        "max_len":                 max_len,
        "best_params":             best,
    }
    with open(results_dir / f"{dataset_name}_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(trainer.train_losses, label="Train")
    plt.plot(trainer.val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{dataset_name} — Finetuning Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / f"{dataset_name}_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {
        "best_val_loss":    trainer.best_val_loss,
        "epochs_trained":   len(trainer.train_losses),
        "train_sequences":  len(train_ds),
        "val_sequences":    len(val_ds),
        "test_sequences":   len(test_df),
        "best_params":      best,
    }
    with open(results_dir / f"{dataset_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"Done {dataset_name}: best_val_loss={metrics['best_val_loss']:.4f}, "
        f"epochs={metrics['epochs_trained']}"
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a STRIPES → SMILES encoder-decoder with Optuna HPO."
    )
    # Required
    parser.add_argument(
        "--data_path", required=True,
        help="Base directory containing one sub-folder per dataset "
             "(e.g. data/PPAR/dataset.csv, data/JAK1/dataset.csv, …).",
    )
    parser.add_argument(
        "--pretrained_model", required=True,
        help="Path to the pretrained encoder checkpoint "
             "(pretrained_stripes_encoder.pth from run_pretraining.py).",
    )
    parser.add_argument(
        "--pretrained_vocab", required=True,
        help="Path to the pretrained STRIPES vocabulary "
             "(stripes_vocab.pkl from run_pretraining.py).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where finetuned models and evaluation results are saved.",
    )
    # Optional
    parser.add_argument(
        "--datasets", nargs="+", default=["PPAR", "PIM1", "JAK1", "AR"],
        help="List of dataset names to finetune on (default: PPAR PIM1 JAK1 AR).",
    )
    parser.add_argument(
        "--n_trials", type=int, default=100,
        help="Number of Optuna trials per dataset (default: 100).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":


    args = parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    set_seed(args.seed)

    # Load pretrained STRIPES vocab
    try:
        with open(args.pretrained_vocab, "rb") as f:
            pretrained_vocab = pickle.load(f)
        logger.info(f"Pretrained vocab loaded: {len(pretrained_vocab)} tokens")
    except Exception as e:
        logger.error(f"Failed to load pretrained vocab: {e}")
        sys.exit(1)

    # Build shared SMILES vocabulary
    smiles_vocab = build_smiles_vocab(args.datasets, args.data_path)

    smiles_vocab_path = Path(args.output_dir) / "smiles_vocab.pkl"
    smiles_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(smiles_vocab_path, "wb") as f:
        pickle.dump(smiles_vocab, f)

    # Run each dataset
    all_results = {}
    for ds_name in args.datasets:
        ds_path = Path(args.data_path) / ds_name / "dataset.csv"
        if not ds_path.exists():
            logger.warning(f"Not found, skipping: {ds_path}")
            continue
        try:
            metrics = run_finetuning(
                ds_name, str(ds_path),
                args.pretrained_model, args.pretrained_vocab,
                pretrained_vocab, smiles_vocab,
                args.output_dir, args.n_trials, device,
            )
            if metrics:
                all_results[ds_name] = metrics
        except Exception as e:
            logger.error(f"Error on {ds_name}: {e}")
            traceback.print_exc()

    # Summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n=== RESULTS ===")
    for ds, m in all_results.items():
        logger.info(
            f"{ds}: best_val_loss={m['best_val_loss']:.4f}, "
            f"epochs={m['epochs_trained']}, "
            f"test_sequences={m['test_sequences']}"
        )
    logger.info(f"Summary saved to {summary_path}")
    logger.info("Run extract_test_and_generate.py to generate SMILES from the test sets.")

"""
Analyze STRIPES → SMILES generation results.

Loads all *_metrics.json files from the comparison folders, produces:
  - analysis_summary.csv   : full table (every combination, all metrics + raw counts)
  - analysis_summary.txt   : human-readable tables (per generation, T effect, beam effect, step effect)
  - plots/                 : heatmaps, parameter-effect lines

Usage
-----
    python analyze_generation.py \
        --results_dir  results_finetune \
        --datasets     PIM1 JAK1 AR \
        --output_dir   generation_analysis
"""

import argparse
import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["validity", "uniqueness", "novelty"]
COLORS  = {"validity": "#2196F3", "uniqueness": "#4CAF50", "novelty": "#FF9800"}

RAW_COUNTS = ["total_generated", "valid_count", "unique_valid_count", "novel_count"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(results_dir: Path, datasets: list) -> pd.DataFrame:
    """Load all *_metrics.json files and return a tidy DataFrame."""
    pattern = re.compile(
        r"generated_beam(\d+)_N(\d+)_T([\d.]+)_step([\d.]+)_metrics\.json"
    )
    records = []
    for ds in datasets:
        comp_dir = results_dir / f"{ds}_finetuned" / "comparison"
        if not comp_dir.exists():
            logger.warning(f"Comparison folder not found: {comp_dir}")
            continue
        for f in sorted(comp_dir.glob("*_metrics.json")):
            m = pattern.match(f.name)
            if not m:
                continue
            beam, n_mol, temp, step = m.groups()
            data = json.loads(f.read_text())
            records.append({
                "dataset":          ds,
                "beam":             int(beam),
                "n_mol":            int(n_mol),
                "temperature":      float(temp),
                "step":             float(step),
                "total_generated":  data.get("total_generated", np.nan),
                "valid_count":      data.get("valid_count", np.nan),
                "validity":         data.get("validity", np.nan),
                "unique_valid_count": data.get("unique_valid_count", np.nan),
                "uniqueness":       data.get("uniqueness", np.nan),
                "novel_count":      data.get("novel_count", np.nan),
                "novelty":          data.get("novelty", np.nan),
            })
    df = pd.DataFrame(records).sort_values(
        ["dataset", "beam", "n_mol", "temperature", "step"]
    ).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} records from {len(datasets)} dataset(s)")
    return df


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, out_dir: Path) -> None:
    csv_path = out_dir / "analysis_summary.csv"
    col_order = [
        "dataset", "beam", "n_mol", "temperature", "step",
        "total_generated", "valid_count", "validity",
        "unique_valid_count", "uniqueness",
        "novel_count", "novelty",
    ]
    df[col_order].to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Text tables
# ---------------------------------------------------------------------------

def _hline(widths, char="-"):
    return char * (sum(widths) + 3 * (len(widths) - 1) + 2) + "\n"


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    txt_path = out_dir / "analysis_summary.txt"
    with open(txt_path, "w") as fh:

        # ------------------------------------------------------------------
        # TABLE 1 — every single generation (beam × N × T × step)
        # ------------------------------------------------------------------
        fh.write("=" * 110 + "\n")
        fh.write("TABLE 1 — ALL GENERATIONS (each row = one specific run)\n")
        fh.write("=" * 110 + "\n\n")

        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            fh.write(f"Dataset: {ds}\n")
            header = (
                f"{'beam':>6} {'N':>4} {'T':>6} {'step':>6} "
                f"{'total':>7} {'valid':>7} {'valid%':>8} "
                f"{'uniq':>7} {'uniq%':>8} "
                f"{'novel':>7} {'novel%':>8}\n"
            )
            fh.write(header)
            fh.write("-" * 90 + "\n")
            for _, r in sub.iterrows():
                fh.write(
                    f"{int(r.beam):>6} {int(r.n_mol):>4} {r.temperature:>6.2f} {r.step:>6.2f} "
                    f"{int(r.total_generated):>7} {int(r.valid_count):>7} {r.validity:>8.4f} "
                    f"{int(r.unique_valid_count):>7} {r.uniqueness:>8.4f} "
                    f"{int(r.novel_count):>7} {r.novelty:>8.4f}\n"
                )
            fh.write("\n")

        # ------------------------------------------------------------------
        # TABLE 2 — effect of beam size (averaged over T, step, datasets)
        # ------------------------------------------------------------------
        fh.write("=" * 110 + "\n")
        fh.write("TABLE 2 — EFFECT OF BEAM SIZE (mean ± std over T, step, all datasets)\n")
        fh.write("=" * 110 + "\n")
        fh.write(f"{'beam':>6} {'N':>4}  {'validity':>18} {'uniqueness':>18} {'novelty':>18}\n")
        fh.write("-" * 70 + "\n")
        for (beam, n_mol), grp in df.groupby(["beam", "n_mol"]):
            row = f"{beam:>6} {n_mol:>4}  "
            for m in METRICS:
                mu, sd = grp[m].mean(), grp[m].std()
                row += f"  {mu:.4f} ± {sd:.4f}  "
            fh.write(row + "\n")
        fh.write("\n")

        # ------------------------------------------------------------------
        # TABLE 3 — effect of temperature (averaged over beam, step, datasets)
        # ------------------------------------------------------------------
        fh.write("=" * 110 + "\n")
        fh.write("TABLE 3 — EFFECT OF TEMPERATURE (mean ± std over beam, step, all datasets)\n")
        fh.write("=" * 110 + "\n")
        fh.write(f"{'T':>8}  {'validity':>18} {'uniqueness':>18} {'novelty':>18}\n")
        fh.write("-" * 70 + "\n")
        for temp, grp in df.groupby("temperature"):
            row = f"{temp:>8.2f}  "
            for m in METRICS:
                mu, sd = grp[m].mean(), grp[m].std()
                row += f"  {mu:.4f} ± {sd:.4f}  "
            fh.write(row + "\n")
        fh.write("\n")

        # ------------------------------------------------------------------
        # TABLE 4 — effect of step (averaged over beam, T, datasets)
        # ------------------------------------------------------------------
        fh.write("=" * 110 + "\n")
        fh.write("TABLE 4 — EFFECT OF STEP (mean ± std over beam, T, all datasets)\n")
        fh.write("=" * 110 + "\n")
        fh.write(f"{'step':>8}  {'validity':>18} {'uniqueness':>18} {'novelty':>18}\n")
        fh.write("-" * 70 + "\n")
        for step, grp in df.groupby("step"):
            row = f"{step:>8.2f}  "
            for m in METRICS:
                mu, sd = grp[m].mean(), grp[m].std()
                row += f"  {mu:.4f} ± {sd:.4f}  "
            fh.write(row + "\n")
        fh.write("\n")

        # ------------------------------------------------------------------
        # TABLE 5 — per-dataset overall summary
        # ------------------------------------------------------------------
        fh.write("=" * 110 + "\n")
        fh.write("TABLE 5 — OVERALL MEAN PER DATASET\n")
        fh.write("=" * 110 + "\n")
        fh.write(f"{'dataset':>12}  {'validity':>18} {'uniqueness':>18} {'novelty':>18}\n")
        fh.write("-" * 70 + "\n")
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            row = f"{ds:>12}  "
            for m in METRICS:
                mu, sd = sub[m].mean(), sub[m].std()
                row += f"  {mu:.4f} ± {sd:.4f}  "
            fh.write(row + "\n")
        fh.write("\n")

    logger.info(f"Text tables saved: {txt_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    """Heatmap: mean validity for each (beam, n_mol) averaged over T and step."""
    sub = df[df["dataset"] == dataset]
    piv = sub.groupby(["beam", "n_mol"])["validity"].mean().unstack("n_mol")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean validity")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"N={c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"beam={r}" for r in piv.index])
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="white" if val > 0.6 else "black", fontsize=10)
    ax.set_title(f"{dataset} — Mean validity (beam × N)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"heatmap_validity_{dataset}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_metrics_overview(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    """Bar chart: mean validity/uniqueness/novelty per (beam, n_mol), averaged over T and step."""
    sub    = df[df["dataset"] == dataset]
    groups = sub.groupby(["beam", "n_mol"])
    labels = [f"beam={b}\nN={n}" for b, n in groups.groups.keys()]
    x      = np.arange(len(labels))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    for i, metric in enumerate(METRICS):
        means = [g[metric].mean() for _, g in groups]
        stds  = [g[metric].std()  for _, g in groups]
        ax.bar(x + i * width, means, width, yerr=stds, label=metric,
               color=COLORS[metric], alpha=0.85, capsize=4)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend()
    ax.set_title(f"{dataset} — Metrics by beam × N (mean ± std over T and step)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"metrics_overview_{dataset}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_temperature_effect(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot: all metrics vs temperature, one line per metric, averaged over all conditions."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        for (beam, n_mol), grp in df.groupby(["beam", "n_mol"]):
            means = grp.groupby("temperature")[metric].mean()
            ax.plot(means.index, means.values, marker="o",
                    label=f"beam={beam} N={n_mol}", alpha=0.8)
        ax.set_xlabel("Temperature")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Effect of T on {metric}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle("Effect of Temperature on generation metrics", fontsize=12)
    plt.tight_layout()
    path = out_dir / "temperature_effect.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_beam_effect(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot: all metrics vs beam size, averaged over T and step."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        for n_mol, grp in df.groupby("n_mol"):
            means = grp.groupby("beam")[metric].mean()
            stds  = grp.groupby("beam")[metric].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker="o", label=f"N={n_mol}", capsize=4)
        ax.set_xlabel("Beam size")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Effect of beam on {metric}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Effect of Beam Size on generation metrics", fontsize=12)
    plt.tight_layout()
    path = out_dir / "beam_effect.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_step_effect(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot: validity vs step for each (beam, n_mol) combo, averaged over T."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for (beam, n_mol), grp in df.groupby(["beam", "n_mol"]):
        means = grp.groupby("step")["validity"].mean()
        ax.plot(means.index, means.values, marker="o", label=f"beam={beam} N={n_mol}")
    ax.set_xlabel("Step (temperature decrement per retry)")
    ax.set_ylabel("Mean validity")
    ax.set_title("Effect of step size on validity")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "step_effect_validity.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_dataset_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    if df["dataset"].nunique() < 2:
        return
    datasets = df["dataset"].unique()
    x        = np.arange(len(datasets))
    width    = 0.25
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, metric in enumerate(METRICS):
        means = [df[df["dataset"] == ds][metric].mean() for ds in datasets]
        stds  = [df[df["dataset"] == ds][metric].std()  for ds in datasets]
        ax.bar(x + i * width, means, width, yerr=stds, label=metric,
               color=COLORS[metric], alpha=0.85, capsize=4)
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Metrics by dataset (overall mean ± std)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "dataset_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    plots_dir   = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(results_dir, args.datasets)
    if df.empty:
        logger.error("No metrics files found. Check --results_dir and --datasets.")
        return

    save_csv(df, out_dir)
    save_tables(df, out_dir)

    for ds in df["dataset"].unique():
        plot_heatmap(df, ds, plots_dir)
        plot_metrics_overview(df, ds, plots_dir)

    plot_temperature_effect(df, plots_dir)
    plot_beam_effect(df, plots_dir)
    plot_step_effect(df, plots_dir)
    plot_dataset_comparison(df, plots_dir)

    logger.info(f"Analysis complete. Results in: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze generation metrics (validity, uniqueness, novelty)."
    )
    parser.add_argument("--results_dir",  required=True,
                        help="Directory containing <DATASET>_finetuned/ folders.")
    parser.add_argument("--datasets", nargs="+", default=["PIM1", "JAK1", "AR"])
    parser.add_argument("--output_dir", default="generation_analysis",
                        help="Where to save plots and tables (default: generation_analysis/).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

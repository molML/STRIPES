from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _REPO_ROOT / "t-SNE" / "results_tsne" / "best_configuration_results.csv"
OUT_PATH = _REPO_ROOT / "figures" / "spearman_correlation_heatmap.png"

PROPERTIES = [
    "lig_MW",
    "lig_logP",
    "lig_TPSA",
    "count_h_bonds",
]

LABELS = {
    "lig_MW": "Molecular Weight",
    "lig_logP": "logP",
    "lig_TPSA": "TPSA",
    "count_h_bonds": "H-bond Count",
}


def compute_spearman(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop in PROPERTIES:
        valid = df[["tsne_x", "tsne_y", prop]].dropna()
        n = len(valid)
        rho_x, p_x = spearmanr(valid[prop], valid["tsne_x"])
        rho_y, p_y = spearmanr(valid[prop], valid["tsne_y"])
        rows.append(
            {
                "property": prop,
                "label": LABELS[prop],
                "rho_x": rho_x,
                "p_x": p_x,
                "rho_y": rho_y,
                "p_y": p_y,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame, out_path: str) -> None:
    labels = results["label"].tolist()
    rho_matrix = results[["rho_x", "rho_y"]].values.T  # shape (2, n_props)

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["t-SNE X", "t-SNE Y"], fontsize=11)

    # annotate cells with rho and significance stars
    sig_thresholds = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
    for row_idx, p_col in enumerate(["p_x", "p_y"]):
        for col_idx, prop_row in results.iterrows():
            rho_val = rho_matrix[row_idx, col_idx]
            p_val = prop_row[p_col]
            stars = ""
            for thresh, mark in sig_thresholds:
                if p_val < thresh:
                    stars = mark
                    break
            ax.text(
                col_idx,
                row_idx,
                f"{rho_val:.2f}{stars}",
                ha="center",
                va="center",
                fontsize=10,
                color="black" if abs(rho_val) < 0.6 else "white",
            )

    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(
        "Spearman correlation: t-SNE coordinates vs physicochemical properties\n"
        "(*p<0.05  **p<0.01  ***p<0.001)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Heatmap saved → {out_path}")


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    for col in PROPERTIES + ["tsne_x", "tsne_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    results = compute_spearman(df)

    print("\nSpearman rank correlation: t-SNE X vs properties")
    print("-" * 60)
    print(f"{'Property':<22} {'ρ (X)':>8} {'p (X)':>12} {'ρ (Y)':>8} {'p (Y)':>12} {'n':>6}")
    for _, row in results.iterrows():
        print(
            f"{row['label']:<22} {row['rho_x']:>8.3f} {row['p_x']:>12.2e}"
            f" {row['rho_y']:>8.3f} {row['p_y']:>12.2e} {int(row['n']):>6}"
        )

    plot_heatmap(results, OUT_PATH)


if __name__ == "__main__":
    main()

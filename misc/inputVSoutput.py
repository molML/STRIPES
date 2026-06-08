import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from rdkit import Chem
from rdkit.Chem import Lipinski

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = str(Path(__file__).resolve().parent.parent / "STRIPES2SMILES" / "results_finetune")
TARGETS  = ["AR_finetuned", "PIM1_finetuned", "PPAR_finetuned", "JAK1_finetuned"]
OUT_DIR  = os.path.join(BASE_DIR, "all_results_merged")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Merge all valid CSVs ────────────────────────────────────────────────────
dfs = []
for target in TARGETS:
    pattern = os.path.join(BASE_DIR, target, "comparison", "generated_beam*.csv")
    for csv_file in sorted(glob.glob(pattern)):
        df = pd.read_csv(csv_file)
        df = df[df["is_valid"]].copy()
        df["target"]      = target.replace("_finetuned", "")
        df["source_file"] = os.path.basename(csv_file)
        dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged = merged[merged["canonical_smiles"].str.len() >= 10].copy()
merged.to_csv(os.path.join(OUT_DIR, "all_valid_merged.csv"), index=False)
print(f"Merged {len(merged):,} valid rows from {len(dfs)} files")
for t in TARGETS:
    n = (merged["target"] == t.replace("_finetuned", "")).sum()
    print(f"  {t}: {n:,} rows")

# ── 2. STRIPES feature extraction ─────────────────────────────────────────────
# Atoms are ; -separated blocks. Within each block tokens are . -separated.
# HBD/HBA: token contains "H(d" or "H(a" (catches compound tokens like [H(a2)C(p)])
# Pi-pi  : token contains uppercase P  –  C(p) is lowercase, not counted
# Persist: feature present in ≥4 of the 11 MD-snapshot tokens within the block

def stripes_features(stripes_str):
    blocks        = stripes_str.split(";")
    length        = len(blocks)
    hbd = hba = hbd_p = hba_p = pipi = pipi_p = 0
    for b in blocks:
        tokens     = b.split(".")
        n_hbd  = sum(1 for t in tokens if "H(d" in t)
        n_hba  = sum(1 for t in tokens if "H(a" in t)
        n_pipi = sum(1 for t in tokens if "P"   in t)
        if n_hbd  >= 1:
            hbd   += 1
        if n_hbd  >= 4:
            hbd_p += 1
        if n_hba  >= 1:
            hba   += 1
        if n_hba  >= 4:
            hba_p += 1
        if n_pipi >= 1:
            pipi   += 1
        if n_pipi >= 4:
            pipi_p += 1
    return pd.Series({"stripes_length":        length,
                      "stripes_hbd":           hbd,
                      "stripes_hbd_persist":   hbd_p,
                      "stripes_hba":           hba,
                      "stripes_hba_persist":   hba_p,
                      "stripes_pipi":          pipi,
                      "stripes_pipi_persist":  pipi_p})

# ── 3. SMILES / RDKit feature extraction ──────────────────────────────────────
def mol_features(smi, prefix):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return pd.Series({f"{prefix}_natoms":   np.nan,
                          f"{prefix}_hbd":      np.nan,
                          f"{prefix}_hba":      np.nan,
                          f"{prefix}_aromatic": np.nan})
    hbd      = Lipinski.NumHDonors(mol)
    hba      = Lipinski.NumHAcceptors(mol)
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    mol_h    = Chem.AddHs(mol)
    return pd.Series({f"{prefix}_natoms":   mol_h.GetNumAtoms(),
                      f"{prefix}_hbd":      hbd,
                      f"{prefix}_hba":      hba,
                      f"{prefix}_aromatic": aromatic})

print("Computing STRIPES features …")
stripes_feats = merged["stripes"].apply(stripes_features)
print("Computing generated SMILES features …")
gen_feats = merged["canonical_smiles"].apply(lambda s: mol_features(s, "gen"))
print("Computing reference SMILES features …")
ref_feats = merged["can_smiles"].apply(lambda s: mol_features(s, "ref"))

df_feats = pd.concat([merged[["target"]], stripes_feats, gen_feats, ref_feats], axis=1)
df_feats.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
print(f"Feature matrix saved: {df_feats.shape}")

# ── 4. Plotting helpers ────────────────────────────────────────────────────────
TARGET_COLORS = {t: c for t, c in zip(
    sorted(df_feats["target"].unique()),
    ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
)}

def plot_1to1(ax, sub, sc, mc, title, xlabel, ylabel):
    data = sub[[sc, mc, "target"]].dropna()
    for tgt, grp in data.groupby("target"):
        ax.scatter(grp[sc], grp[mc],
                   color=TARGET_COLORS.get(tgt, "grey"),
                   alpha=0.35, s=10, label=tgt, rasterized=True)
    lo = min(data[sc].min(), data[mc].min())
    hi = max(data[sc].max(), data[mc].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    exact_pct = (data[sc] == data[mc]).mean() * 100
    mae       = (data[sc] - data[mc]).abs().mean()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.text(0.03, 0.97,
            f"n={len(data):,}\nexact match: {exact_pct:.1f}%\nMAE: {mae:.2f}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

def save_panel(pairs, xlabel, ylabel, suptitle, fname):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
    for ax, (sc, mc, lbl) in zip(axes, pairs):
        plot_1to1(ax, df_feats, sc, mc, lbl, xlabel, ylabel)
    axes[-1].legend(title="Target", fontsize=7, markerscale=2,
                    loc="lower right", framealpha=0.8)
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# ── 5. STRIPES vs Reference SMILES (can_smiles) ───────────────────────────────
PAIRS_STRIPES_REF = [
    ("stripes_length",       "ref_natoms",   "# Atoms (incl. H)"),
    ("stripes_hbd",          "ref_hbd",      "HBD"),
    ("stripes_hbd_persist",  "ref_hbd",      "HBD persistent"),
    ("stripes_hba",          "ref_hba",      "HBA"),
    ("stripes_hba_persist",  "ref_hba",      "HBA persistent"),
    ("stripes_pipi",         "ref_aromatic", "Pi-Pi"),
    ("stripes_pipi_persist", "ref_aromatic", "Pi-Pi persistent"),
]
save_panel(PAIRS_STRIPES_REF,
           xlabel="STRIPES (reference pharmacophore)",
           ylabel="Reference SMILES (can_smiles)",
           suptitle="STRIPES vs Reference SMILES — 1:1 feature comparison",
           fname="stripes_vs_ref.png")

# ── 6. STRIPES vs Generated SMILES ────────────────────────────────────────────
PAIRS_STRIPES_GEN = [
    ("stripes_length",       "gen_natoms",   "# Atoms (incl. H)"),
    ("stripes_hbd",          "gen_hbd",      "HBD"),
    ("stripes_hbd_persist",  "gen_hbd",      "HBD persistent"),
    ("stripes_hba",          "gen_hba",      "HBA"),
    ("stripes_hba_persist",  "gen_hba",      "HBA persistent"),
    ("stripes_pipi",         "gen_aromatic", "Pi-Pi"),
    ("stripes_pipi_persist", "gen_aromatic", "Pi-Pi persistent"),
]
save_panel(PAIRS_STRIPES_GEN,
           xlabel="STRIPES (reference pharmacophore)",
           ylabel="Generated SMILES",
           suptitle="STRIPES vs Generated SMILES — 1:1 feature comparison",
           fname="stripes_vs_gen.png")

# ── 7. Reference SMILES vs Generated SMILES ───────────────────────────────────
PAIRS_REF_GEN = [
    ("ref_natoms",   "gen_natoms",   "# Atoms (incl. H)"),
    ("ref_hbd",      "gen_hbd",      "HBD"),
    ("ref_hba",      "gen_hba",      "HBA"),
    ("ref_aromatic", "gen_aromatic", "Pi-Pi"),
]
save_panel(PAIRS_REF_GEN,
           xlabel="Reference SMILES (can_smiles)",
           ylabel="Generated SMILES (canonical_smiles)",
           suptitle="Reference vs Generated SMILES — 1:1 feature comparison",
           fname="ref_vs_gen.png")

# ── 8. Summary table ──────────────────────────────────────────────────────────
rows = []
for label, pairs in [("STRIPES→REF", PAIRS_STRIPES_REF),
                     ("STRIPES→GEN", PAIRS_STRIPES_GEN),
                     ("REF→GEN",     PAIRS_REF_GEN)]:
    for sc, mc, lbl in pairs:
        d = df_feats[[sc, mc]].dropna()
        rows.append({
            "comparison": label,
            "feature":    lbl,
            "n":          len(d),
            "exact%":      round((d[sc] == d[mc]).mean() * 100, 1),
            "MAE":         round((d[sc] - d[mc]).abs().mean(), 2),
            "r_spearman":  round(d[sc].corr(d[mc], method="spearman"), 3),
        })
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
print("\n── Summary ──")
print(summary.to_string(index=False))

# ── 9. Spearman r heatmap ─────────────────────────────────────────────────────
FEAT_LABELS = {
    "# Atoms (incl. H)": "Atoms",
    "HBD":               "HBD",
    "HBD persistent":    "HBD persistent",
    "HBA":               "HBA",
    "HBA persistent":    "HBA persistent",
    "Pi-Pi":             "Pi-Pi",
    "Pi-Pi persistent":  "Pi-Pi persistent",
}
COL_ORDER = ["Atoms", "HBD", "HBD persistent", "HBA", "HBA persistent",
             "Pi-Pi", "Pi-Pi persistent"]
ROW_ORDER = ["STRIPES→REF", "STRIPES→GEN", "REF→GEN"]

pivot = (summary
         .assign(feat=summary["feature"].map(FEAT_LABELS))
         .pivot_table(index="comparison", columns="feat",
                      values="r_spearman", aggfunc="mean")
         .reindex(index=ROW_ORDER, columns=COL_ORDER))

# Discrete blue colormap — same style as reference figures
boundaries_r = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
blue_colors = ['#f4f7fb', '#e6edf4', '#d8e3ed', '#c9d9e6', '#b4cbdc',
               '#9ebdd2', '#88afc8', '#72a1be', '#5c93b4', '#4a829f']
blue_cmap = ListedColormap(blue_colors)
blue_cmap.set_bad(color='#e8e8e8')
norm_r = BoundaryNorm(boundaries_r, ncolors=len(blue_colors))

data_arr = pivot.values.astype(float)
n_rows, n_cols = data_arr.shape

fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(data_arr, aspect='auto', cmap=blue_cmap, norm=norm_r, origin='upper')

for i in range(n_rows):
    for j in range(n_cols):
        val = data_arr[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color='black', fontsize=9)

ax.set_xticks(np.arange(n_cols))
ax.set_yticks(np.arange(n_rows))
ax.set_xticklabels(COL_ORDER, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ROW_ORDER, fontsize=9)
ax.tick_params(which='minor', bottom=False, left=False)
ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

ax.set_title("Spearman r — feature transfer across representations", fontsize=11)
ax.set_xlabel("")
ax.set_ylabel("")

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Spearman r', rotation=90, labelpad=15, fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.subplots_adjust(right=0.88)
plt.savefig(os.path.join(OUT_DIR, "spearman_heatmap.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "spearman_heatmap.svg"), bbox_inches="tight")
plt.show()
print("Saved: spearman_heatmap.png")

# ── 10. Descriptive statistics: mean ± SD per feature source ──────────────────
FEAT_GROUPS = [
    ("N atoms (incl. H)", "stripes_length",      "ref_natoms",   "gen_natoms"),
    ("HBD",               "stripes_hbd",          "ref_hbd",      "gen_hbd"),
    ("HBD persistent",    "stripes_hbd_persist",  None,           None),
    ("HBA",               "stripes_hba",          "ref_hba",      "gen_hba"),
    ("HBA persistent",    "stripes_hba_persist",  None,           None),
    ("Pi-Pi",             "stripes_pipi",         "ref_aromatic", "gen_aromatic"),
    ("Pi-Pi persistent",  "stripes_pipi_persist", None,           None),
]

def fmt(col):
    if col is None:
        return "—"
    s = df_feats[col].dropna()
    return f"{s.mean():.2f} ± {s.std():.2f}"

def fmt_target(col, tgt):
    if col is None:
        return "—"
    s = df_feats.loc[df_feats["target"] == tgt, col].dropna()
    return f"{s.mean():.2f} ± {s.std():.2f}" if len(s) else "—"

# Overall summary
stat_rows = []
for feat_name, sc, rc, gc in FEAT_GROUPS:
    stat_rows.append({
        "Feature":  feat_name,
        "STRIPES":  fmt(sc),
        "REF":      fmt(rc),
        "GEN":      fmt(gc),
    })
stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"), index=False)

print("\n── Descriptive statistics (mean ± SD, all targets) ──")
print(stats_df.to_string(index=False))

# Per-target breakdown
targets_sorted = sorted(df_feats["target"].unique())
per_target_rows = []
for tgt in targets_sorted:
    for feat_name, sc, rc, gc in FEAT_GROUPS:
        per_target_rows.append({
            "Target":  tgt,
            "Feature": feat_name,
            "STRIPES": fmt_target(sc, tgt),
            "REF":     fmt_target(rc, tgt),
            "GEN":     fmt_target(gc, tgt),
        })
per_target_df = pd.DataFrame(per_target_rows)
per_target_df.to_csv(os.path.join(OUT_DIR, "descriptive_stats_per_target.csv"), index=False)

print("\n── Descriptive statistics (mean ± SD, per target) ──")
print(per_target_df.to_string(index=False))


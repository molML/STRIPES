"""
2D heatmaps of median ΔpKi (and ΔpKi/n/std) across STRIPES- and ECFP-similarity
bins, for each target dataset (Fig. 2b).
"""

from pathlib import Path

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import rdkit.DataStructs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import warnings
warnings.filterwarnings('ignore')

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

datasets = ['JAK1', 'PIM1', 'AR', 'PPAR']

#%% ============================================================
# LOAD AND PREPARE DATA FOR EACH DATASET
# ============================================================

all_data = {}

for dataset in datasets:
    print(f"Loading data for {dataset}...")

    # Load data
    df = pd.read_csv(_REPO_ROOT / 'data' / dataset / 'dataset.csv')
    smiles_list = df['smiles'].tolist()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [_morgan_gen.GetFingerprint(mol) for mol in mols]

    # Tanimoto matrix
    tanimoto_matrix = []
    for i, fp in enumerate(fps):
        similarities = rdkit.DataStructs.BulkTanimotoSimilarity(fp, fps)
        tanimoto_matrix.append(similarities)

    # Build unique-pair dataframe
    data = []
    for i, sim_row in enumerate(tanimoto_matrix):
        for j, similarity in enumerate(sim_row):
            if i < j:
                data.append({
                    'smiles1': smiles_list[i],
                    'pKi1': df.iloc[i]['pKi'],
                    'smiles2': smiles_list[j],
                    'pKi2': df.iloc[j]['pKi'],
                    'tanimoto': similarity
                })

    tanimoto_df = pd.DataFrame(data)

    # Load STRIPES
    df_stripes = pd.read_csv(_REPO_ROOT / 'STRIPES_similarity' / 'results' / dataset / 'results.csv')
    df_stripes = df_stripes[~df_stripes.isin(['ERROR']).any(axis=1)]
    df_stripes = df_stripes[df_stripes['smiles1'] != df_stripes['smiles2']]
    df_stripes['pair_key'] = df_stripes.apply(
        lambda r: tuple(sorted([r['smiles1'], r['smiles2']])), axis=1
    )
    df_stripes = df_stripes.drop_duplicates(subset='pair_key')

    # Merge
    tanimoto_df['pair_key'] = tanimoto_df.apply(
        lambda r: tuple(sorted([r['smiles1'], r['smiles2']])), axis=1
    )
    tanimoto_df = tanimoto_df.merge(
        df_stripes[['pair_key', 'similarity']],
        on='pair_key', how='left'
    )
    tanimoto_df.rename(columns={'similarity': 'stripes'}, inplace=True)
    tanimoto_df.drop(columns='pair_key', inplace=True)
    tanimoto_df['pKi_diff'] = abs(tanimoto_df['pKi1'] - tanimoto_df['pKi2'])
    tanimoto_df = tanimoto_df.dropna(subset=['stripes'])

    all_data[dataset] = tanimoto_df

print("Data loaded for all datasets.\n")

#%% ============================================================
# 2D HEATMAP 1: STRIPES vs Tanimoto with median ΔpKi ONLY (no n)
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(32, 8))

# Define bins for both axes
similarity_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_labels_heatmap = ['10-20%', '20-30%', '30-40%', '40-50%', '50-60%',
                       '60-70%', '70-80%', '80-90%', '90-100%']

# Discrete colormap with sharp boundaries - blue palette
boundaries = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
blue_colors = ['#f4f7fb', '#e6edf4', '#d8e3ed', '#c9d9e6', '#b4cbdc', '#9ebdd2',
               '#88afc8', '#72a1be', '#5c93b4', '#4a829f', '#3d6f89', '#335d73']
blue_cmap = ListedColormap(blue_colors)
norm = BoundaryNorm(boundaries, ncolors=len(blue_colors))

for dataset_idx, dataset in enumerate(datasets):
    ax = axes[dataset_idx]
    tanimoto_df = all_data[dataset]

    print(f"\n{'='*70}")
    print(f"HEATMAP 1 (ΔpKi only) FOR {dataset}")
    print(f"{'='*70}")

    # Create bins
    tanimoto_df_copy = tanimoto_df.copy()
    tanimoto_df_copy['tanimoto_bin'] = pd.cut(tanimoto_df_copy['tanimoto'],
                                               bins=similarity_bins,
                                               labels=bin_labels_heatmap,
                                               include_lowest=True)
    tanimoto_df_copy['stripes_bin'] = pd.cut(tanimoto_df_copy['stripes'],
                                              bins=similarity_bins,
                                              labels=bin_labels_heatmap,
                                              include_lowest=True)

    # Create matrices for heatmap
    n_bins = len(bin_labels_heatmap)
    heatmap_median = np.zeros((n_bins, n_bins))
    heatmap_median[:] = np.nan

    # Fill matrices
    for i, tan_label in enumerate(bin_labels_heatmap):
        for j, stripes_label in enumerate(bin_labels_heatmap):
            subset = tanimoto_df_copy[(tanimoto_df_copy['tanimoto_bin'] == tan_label) &
                                       (tanimoto_df_copy['stripes_bin'] == stripes_label)]
            if len(subset) >= 5:  # Minimum 5 pairs
                heatmap_median[i, j] = np.median(subset['pKi_diff'])

    # Plot heatmap
    im = ax.imshow(heatmap_median, aspect='equal', cmap=blue_cmap,
                   norm=norm, origin='lower')

    # Set ticks
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels_heatmap, rotation=45, ha='right', fontsize=22)
    if dataset_idx == 0:
        ax.set_yticklabels(bin_labels_heatmap, fontsize=22)
        ax.set_ylabel('Sim(ECFPs)', fontsize=34)
    else:
        ax.set_yticklabels([])

    # Add text annotations (median ΔpKi ONLY)
    for i in range(n_bins):
        for j in range(n_bins):
            if not np.isnan(heatmap_median[i, j]):
                ax.text(j, i, f'{heatmap_median[i, j]:.2f}',
                       ha="center", va="center", color="black",
                       fontsize=18)

    ax.set_xlabel('Sim(STRIPES)', fontsize=34)
    ax.set_title(f'{dataset}', fontsize=34)

    # Add grid
    ax.set_xticks(np.arange(n_bins) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_bins) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Median ΔpKi / ΔpEC50', rotation=90, labelpad=25,
               fontsize=34)
cbar.ax.tick_params(labelsize=22)

plt.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.15, wspace=0.01)
fig.savefig(_REPO_ROOT / 'figures' / 'heatmap1_median_dpki.svg', format='svg', bbox_inches='tight')
plt.show()

#%% ============================================================
# 2D HEATMAP 2: STRIPES vs Tanimoto with ΔpKi, n, and STD
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(32, 8))

for dataset_idx, dataset in enumerate(datasets):
    ax = axes[dataset_idx]
    tanimoto_df = all_data[dataset]

    print(f"\n{'='*70}")
    print(f"HEATMAP 2 (ΔpKi, n, std) FOR {dataset}")
    print(f"{'='*70}")

    tanimoto_df_copy = tanimoto_df.copy()
    tanimoto_df_copy['tanimoto_bin'] = pd.cut(tanimoto_df_copy['tanimoto'],
                                               bins=similarity_bins,
                                               labels=bin_labels_heatmap,
                                               include_lowest=True)
    tanimoto_df_copy['stripes_bin'] = pd.cut(tanimoto_df_copy['stripes'],
                                              bins=similarity_bins,
                                              labels=bin_labels_heatmap,
                                              include_lowest=True)

    n_bins = len(bin_labels_heatmap)
    heatmap_median = np.zeros((n_bins, n_bins))
    heatmap_count = np.zeros((n_bins, n_bins))
    heatmap_std = np.zeros((n_bins, n_bins))
    heatmap_median[:] = np.nan
    heatmap_std[:] = np.nan

    for i, tan_label in enumerate(bin_labels_heatmap):
        for j, stripes_label in enumerate(bin_labels_heatmap):
            subset = tanimoto_df_copy[(tanimoto_df_copy['tanimoto_bin'] == tan_label) &
                                       (tanimoto_df_copy['stripes_bin'] == stripes_label)]
            if len(subset) >= 5:  
                heatmap_median[i, j] = np.median(subset['pKi_diff'])
                heatmap_count[i, j] = len(subset)
                heatmap_std[i, j] = np.std(subset['pKi_diff'])

    im = ax.imshow(heatmap_median, aspect='equal', cmap=blue_cmap,
                   norm=norm, origin='lower')

    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels_heatmap, rotation=45, ha='right', fontsize=22)
    if dataset_idx == 0:
        ax.set_yticklabels(bin_labels_heatmap, fontsize=22)
        ax.set_ylabel('ECFP Similarity', fontsize=34)
    else:
        ax.set_yticklabels([])


    for i in range(n_bins):
        for j in range(n_bins):
            if not np.isnan(heatmap_median[i, j]):
                ax.text(j, i + 0.25, f'{heatmap_median[i, j]:.2f}',
                       ha="center", va="center", color="black",
                       fontsize=9)
                ax.text(j, i, f'n={int(heatmap_count[i, j])}',
                       ha="center", va="center", color="black",
                       fontsize=7)
                ax.text(j, i - 0.25, f'std={heatmap_std[i, j]:.2f}',
                       ha="center", va="center", color="black",
                       fontsize=7)

    ax.set_xlabel('STRIPES Similarity', fontsize=34)
    ax.set_title(f'{dataset}', fontsize=34)

    ax.set_xticks(np.arange(n_bins) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_bins) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)


cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Median ΔpKi / ΔpEC50', rotation=90, labelpad=25,
               fontsize=34)
cbar.ax.tick_params(labelsize=22)

plt.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.15, wspace=0.01)
fig.savefig(_REPO_ROOT / 'figures' / 'heatmap2_median_dpki_n_std.svg', format='svg', bbox_inches='tight')
plt.show()

#%% ============================================================
# PRINT EXAMPLE SMILES PAIRS (3 per dataset, per group)
# ============================================================

for dataset in datasets:
    df = all_data[dataset]

    print("\n" + "="*100)
    print(f"DATASET: {dataset}")
    print("="*100)

    print("\n" + "-"*80)
    print("3 PAIRS: HIGH ECFP, LOW STRIPES, HIGH ΔpKi")
    print("(Structurally similar but functionally different)")
    print("-"*80)

    # Highest Tanimoto possible, Low STRIPES (<0.5), High ΔpKi (>1.0)
    high_tan_low_stripes = df[
        (df['stripes'] < 0.5) &
        (df['pKi_diff'] > 1.0)
    ].sort_values('tanimoto', ascending=False).head(10)

    if len(high_tan_low_stripes) == 0:
        print("  No pairs found with these criteria")
    else:
        for i, (idx, row) in enumerate(high_tan_low_stripes.iterrows(), 1):
            print(f"\n  Pair {i} [{dataset}]:")
            print(f"    SMILES 1: {row['smiles1']}")
            print(f"    SMILES 2: {row['smiles2']}")
            print(f"    ECFP Similarity: {row['tanimoto']:.3f}")
            print(f"    STRIPES Similarity: {row['stripes']:.3f}")
            print(f"    ΔpKi: {row['pKi_diff']:.2f}")

    print("\n" + "-"*80)
    print("3 PAIRS: LOW ECFP, HIGH STRIPES, LOW ΔpKi")
    print("(Structurally different but functionally similar)")
    print("-"*80)

    # Low Tanimoto (<0.4), Highest STRIPES possible, Low ΔpKi (<0.5)
    low_tan_high_stripes = df[
        (df['tanimoto'] < 0.4) &
        (df['pKi_diff'] < 0.5)
    ].sort_values('stripes', ascending=False).head(3)

    if len(low_tan_high_stripes) == 0:
        print("  No pairs found with these criteria")
    else:
        for i, (idx, row) in enumerate(low_tan_high_stripes.iterrows(), 1):
            print(f"\n  Pair {i} [{dataset}]:")
            print(f"    SMILES 1: {row['smiles1']}")
            print(f"    SMILES 2: {row['smiles2']}")
            print(f"    ECFP Similarity: {row['tanimoto']:.3f}")
            print(f"    STRIPES Similarity: {row['stripes']:.3f}")
            print(f"    ΔpKi: {row['pKi_diff']:.2f}")


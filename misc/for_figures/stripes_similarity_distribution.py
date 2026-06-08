"""
Plot combinato:
  - Internal STRIPES similarity (distribuzione interna per dataset)
  - Two-stripes similarity (coppie di input che generano lo stesso output)
I due tipi sono distinti per colore unico; le coppie dello stesso dataset
sono affiancate, alternando internal → two-stripes.
"""
from pathlib import Path

import pandas as pd
import json
import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

datasets = ['JAK1', 'PIM1' ,'PPAR', 'AR' ]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_pim1_sim = pd.read_csv(_REPO_ROOT / 'MD' / 'PIM1' / 'results_sim.csv')
_pim1_sim = _pim1_sim[_pim1_sim['canonical_smiles'].fillna('').str.len() >= 10]
_jak1_sim = pd.read_csv(_REPO_ROOT / 'MD' / 'JAK1' / 'results_sim.csv')
_jak1_sim = _jak1_sim[_jak1_sim['canonical_smiles'].fillna('').str.len() >= 10]
_ar_sim = pd.read_csv(_REPO_ROOT / 'MD' / 'AR' / 'results_sim.csv')
_ar_sim = _ar_sim[_ar_sim['canonical_smiles'].fillna('').str.len() >= 10]
_ppar_sim = pd.read_csv(_REPO_ROOT / 'MD' / 'PPAR' / 'results_sim.csv')
_ppar_sim = _ppar_sim[_ppar_sim['canonical_smiles'].fillna('').str.len() >= 10]

similarity_MD = {
    'PIM1': _pim1_sim['stripes_sim'].dropna().tolist(),
    'JAK1': _jak1_sim['stripes_sim'].dropna().tolist(),
    'AR':   _ar_sim['stripes_sim'].dropna().tolist(),
    'PPAR': _ppar_sim['stripes_sim'].dropna().tolist(),
}

COLOR_INTERNAL   = '#5B8DB8'   # blu  – interno
COLOR_TWOSTRIPES = '#C96A4E'   # arancio – same predicted output
COLOR_MD         = '#5A9B6A'   # verde – MD similarity

TOKEN_DICT_PATH = _REPO_ROOT / 'data' / 'stripes_tokens2label.json'
BASE_PATH       = _REPO_ROOT / 'STRIPES_similarity' / 'results'


# ── Funzioni di similarità ────────────────────────────────────────────────────
def parse_stripe(stripes_str: str) -> List[List[str]]:
    if not stripes_str or stripes_str.strip() == '':
        return []
    try:
        atoms = stripes_str.split(';')
        parsed_atoms = []
        for atom in atoms:
            if atom.strip():
                tokens = atom.strip().split('.')
                parsed_atoms.append(tokens)
        return parsed_atoms
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        return []


def load_token_dictionary(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def atom_to_index_vector(atom_tokens: List[str], token_to_index: dict) -> List[int]:
    return [token_to_index[token] for token in atom_tokens if token in token_to_index]


def precompute_atom_vectors(parsed_stripe: List[List[str]], token_to_index: dict) -> List[List[int]]:
    return [atom_to_index_vector(atom, token_to_index) for atom in parsed_stripe]


def atom_similarity(vec1: List[int], vec2: List[int]) -> float:
    if not vec1 and not vec2:
        return 1.0
    if not vec1 or not vec2:
        return 0.0
    set1, set2 = set(vec1), set(vec2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 1.0


def hungarian_similarity(vecs1: List[List[int]], vecs2: List[List[int]]) -> float:
    n_A = len(vecs1)
    n_B = len(vecs2)
    if n_A == 0 and n_B == 0:
        return 1.0
    if n_A == 0 or n_B == 0:
        return 0.0
    sim_matrix = np.zeros((n_A, n_B))
    for i, vec_a in enumerate(vecs1):
        for j, vec_b in enumerate(vecs2):
            sim_matrix[i, j] = atom_similarity(vec_a, vec_b)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_sims = sim_matrix[row_ind, col_ind]
    unmatched = abs(n_A - n_B)
    total_pairs = max(n_A, n_B)
    sim_score = np.sum(matched_sims) / total_pairs
    sim_score *= (total_pairs - unmatched) / total_pairs
    return float(sim_score)


def calculate_similarity(stripes1: str, stripes2: str, token_to_index: dict) -> float:
    parsed1 = parse_stripe(stripes1)
    parsed2 = parse_stripe(stripes2)
    if not parsed1 or not parsed2:
        return 0.0
    vecs1 = precompute_atom_vectors(parsed1, token_to_index)
    vecs2 = precompute_atom_vectors(parsed2, token_to_index)
    return hungarian_similarity(vecs1, vecs2)


# ── Carica dizionario token ───────────────────────────────────────────────────
token_to_index = load_token_dictionary(TOKEN_DICT_PATH)
print(f"Dizionario caricato: {len(token_to_index)} token")


# ── 1) Distribuzione interna ──────────────────────────────────────────────────
dfs_internal = {}
for dataset in datasets:
    df = pd.read_csv(f'{BASE_PATH}/{dataset}/results.csv')
    df = df[df['STRIPES1'].fillna('') != 'ERROR']
    df = df[df['STRIPES2'].fillna('') != 'ERROR']
    df = df[df['similarity'] != 0]
    df = df[(df['pKi1'] >= 6) & (df['pKi2'] >= 6)]
    df = df[df['smiles1'] != df['smiles2']]
    dfs_internal[dataset] = df['similarity'].dropna().values
    print(f"[{dataset}] internal: {len(dfs_internal[dataset])} coppie  |  "
          f"median={np.median(dfs_internal[dataset]):.3f}  "
          f"mean={np.mean(dfs_internal[dataset]):.3f}")


# ── 2) Two-stripes similarity (same predicted output) ────────────────────────
all_similarity_results = []

for dataset in datasets:
    df_generated = pd.read_csv(
        _REPO_ROOT / 'STRIPES2SMILES' / 'results_finetune'
        / f'{dataset}_finetuned' / 'comparison' / 'generated_beam15_N5_T1.4_step0.5.csv'
    )
    df_generated = df_generated[df_generated['canonical_smiles'].fillna('').str.len() >= 10]
    df_grouped = df_generated.groupby('canonical_smiles').agg(
        can_smiles=('can_smiles', list)
    ).reset_index()
    df_grouped['n_unique_inputs'] = df_grouped['can_smiles'].apply(lambda x: len(set(x)))
    df_multi_input = df_grouped[df_grouped['n_unique_inputs'] > 1].copy()

    predicted_to_inputs = {
        row['canonical_smiles']: list(set(row['can_smiles']))
        for _, row in df_multi_input.iterrows()
    }
    print(f"\n[{dataset}] SMILES generate da più input: {len(predicted_to_inputs)}")

    df_stripes = pd.read_csv(_REPO_ROOT / 'data' / dataset / 'dataset.csv')
    smiles_to_stripes = dict(zip(df_stripes['can_smiles'], df_stripes['STRIPES']))

    predicted_to_inputs_with_stripes = {}
    for pred, can_smiles_list in predicted_to_inputs.items():
        inputs_with_stripes = [
            (smi, smiles_to_stripes[smi])
            for smi in can_smiles_list
            if smi in smiles_to_stripes
        ]
        if len(inputs_with_stripes) > 1:
            predicted_to_inputs_with_stripes[pred] = inputs_with_stripes

    print(f"[{dataset}] Gruppi con STRIPES: {len(predicted_to_inputs_with_stripes)}")

    results = []
    for pred_smiles, inputs in predicted_to_inputs_with_stripes.items():
        for (smi1, str1), (smi2, str2) in combinations(inputs, 2):
            similarity = calculate_similarity(str1, str2, token_to_index)
            results.append({
                'predicted_smiles': pred_smiles,
                'can_smiles_1': smi1,
                'can_smiles_2': smi2,
                'similarity':   similarity,
            })

    df_results = pd.DataFrame(results)
    df_results = df_results[df_results['predicted_smiles'] != 'INVALID']
    df_results['dataset'] = dataset
    print(f"[{dataset}] two-stripes: {len(df_results)} coppie")
    all_similarity_results.append(df_results)

df_all_stripes = pd.concat(all_similarity_results, ignore_index=True)


# ── 2b) Mann-Whitney U test (internal vs two-stripes) ────────────────────────
def pval_to_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

mw_results = {}
for dataset in datasets:
    data_int = dfs_internal[dataset]
    data_str = df_all_stripes[df_all_stripes['dataset'] == dataset]['similarity'].values
    stat, p = mannwhitneyu(data_int, data_str, alternative='two-sided')
    mw_results[dataset] = {'stat': stat, 'p': p, 'stars': pval_to_stars(p)}
    print(f"[{dataset}] Mann-Whitney (int vs two-stripes) U={stat:.0f}  p={p:.4g}  {pval_to_stars(p)}")

mw_results_md = {}
for dataset in datasets:
    data_int = dfs_internal[dataset]
    data_md  = np.array(similarity_MD[dataset])
    stat, p = mannwhitneyu(data_int, data_md, alternative='two-sided')
    mw_results_md[dataset] = {'stat': stat, 'p': p, 'stars': pval_to_stars(p)}
    print(f"[{dataset}] Mann-Whitney (int vs MD)         U={stat:.0f}  p={p:.4g}  {pval_to_stars(p)}")


# ── 3) Plot combinato ─────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
PAIR_OFFSET    = 0.60   # distanza intra-coppia
GROUP_SPACING  = 2.40   # distanza tra gruppi (aumentato per 3 violini)
VIOLIN_WIDTH   = 0.45
BOX_WIDTH      = 0.14
MAX_JITTER_PTS = 500    # campione massimo per i puntini jitter

rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(14, 6))

tick_positions = []
tick_labels    = []

for i, dataset in enumerate(datasets):
    pos_int = i * GROUP_SPACING
    pos_str = pos_int + PAIR_OFFSET
    pos_md  = pos_str + PAIR_OFFSET
    tick_positions.append((pos_int + pos_md) / 2)
    tick_labels.append(dataset)

    for pos, color, data in [
        (pos_int, COLOR_INTERNAL,   dfs_internal[dataset]),
        (pos_str, COLOR_TWOSTRIPES, df_all_stripes[df_all_stripes['dataset'] == dataset]['similarity'].values),
        (pos_md,  COLOR_MD,         np.array(similarity_MD[dataset])),
    ]:
        # Violin
        if len(data) >= 2:
            parts = ax.violinplot(data, positions=[pos], widths=VIOLIN_WIDTH,
                                  showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.30)

        # Boxplot sovrapposto
        if len(data) >= 2:
            ax.boxplot(data, positions=[pos], widths=BOX_WIDTH, patch_artist=True,
                       showfliers=False,
                       boxprops=dict(facecolor=color, alpha=0.85, linewidth=1.2),
                       medianprops=dict(color='white', linewidth=2.2),
                       whiskerprops=dict(color=color, linewidth=1.2),
                       capprops=dict(color=color, linewidth=1.2))
        else:
            ax.scatter([pos], data, color=color, alpha=0.85, s=80, zorder=5, marker='D')

        # Jitter (campionato per dataset grandi)
        sample = data if len(data) <= MAX_JITTER_PTS else rng.choice(data, MAX_JITTER_PTS, replace=False)
        jitter = rng.uniform(-0.09, 0.09, size=len(sample))
        ax.scatter(pos + jitter, sample, color=color, alpha=0.30, s=10, zorder=3, linewidths=0)

    # Significatività Mann-Whitney (internal vs Convergent inputs)
    data_int = dfs_internal[dataset]
    data_str = df_all_stripes[df_all_stripes['dataset'] == dataset]['similarity'].values
    y_top = 1.02
    h = 0.025
    stars = mw_results[dataset]['stars']
    ax.plot([pos_int, pos_int, pos_str, pos_str],
            [y_top, y_top + h, y_top + h, y_top],
            lw=1.0, color='black')

    # Significatività Mann-Whitney (internal vs MD)
    y_top_md = 1.07
    ax.plot([pos_int, pos_int, pos_md, pos_md],
            [y_top_md, y_top_md + h, y_top_md + h, y_top_md],
            lw=1.0, color='black')

# Linee verticali di separazione tra le coppie
for i in range(1, len(datasets)):
    sep = i * GROUP_SPACING - (GROUP_SPACING - 2 * PAIR_OFFSET) / 2
    ax.axvline(sep, color='grey', linewidth=0.6, linestyle='--', alpha=0.4)

# Legenda
legend_elements = [
    Patch(facecolor=COLOR_INTERNAL,   alpha=0.85, label='Internal distribution'),
    Patch(facecolor=COLOR_TWOSTRIPES, alpha=0.85, label='Convergent inputs'),
    Patch(facecolor=COLOR_MD,         alpha=0.85, label='MD similarity'),
]
ax.legend(handles=legend_elements, fontsize=14, framealpha=0.8,
          loc='lower right', bbox_to_anchor=(1.0, 1.02),
          bbox_transform=ax.transAxes, borderaxespad=0)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=16)
ax.set_xlim(-0.6, (len(datasets) - 1) * GROUP_SPACING + 2 * PAIR_OFFSET + 0.6)
ax.set_ylabel('STRIPES similarity', fontsize=16)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylim(0, 1.18)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(
    _REPO_ROOT / 'figures' / 'combined_similarity_plot.svg',
    dpi=150, bbox_inches='tight'
)
plt.show()

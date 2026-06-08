from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FINETUNE_DIR = _REPO_ROOT / "STRIPES2SMILES" / "results_finetune"
FIG_DIR = _REPO_ROOT / "figures"


def tanimoto(smi1, smi2):
    mol1, mol2 = Chem.MolFromSmiles(str(smi1)), Chem.MolFromSmiles(str(smi2))
    if mol1 is None or mol2 is None:
        return np.nan
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

datasets = ['JAK1', 'PIM1', 'PPAR', 'AR',]
COLOR = '#5B8DB8'

VIOLIN_WIDTH   = 0.45
BOX_WIDTH      = 0.14
MAX_JITTER_PTS = 500

rng = np.random.default_rng(42)

data = []
for dataset in datasets:
    df = pd.read_csv(RESULTS_FINETUNE_DIR / f'{dataset}_finetuned' / 'comparison' / 'generated_beam15_N5_T1.4_step0.5.csv')
    valid = df[df['is_valid']].copy()
    valid = valid[valid['canonical_smiles'].apply(
        lambda s: (m := Chem.MolFromSmiles(str(s))) is not None and m.GetNumAtoms() >= 10
    )]
    similarities = valid.apply(lambda row: tanimoto(row['can_smiles'], row['canonical_smiles']), axis=1)
    data.append(similarities.dropna().values)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(8, 6))

positions = list(range(len(datasets)))

for pos, vals in zip(positions, data):
    # Violin
    parts = ax.violinplot(vals, positions=[pos], widths=VIOLIN_WIDTH,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(COLOR)
        pc.set_alpha(0.30)

    # Boxplot sovrapposto
    ax.boxplot(vals, positions=[pos], widths=BOX_WIDTH, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor=COLOR, alpha=0.85, linewidth=1.2),
               medianprops=dict(color='white', linewidth=2.2),
               whiskerprops=dict(color=COLOR, linewidth=1.2),
               capprops=dict(color=COLOR, linewidth=1.2))

    # Jitter
    sample = vals if len(vals) <= MAX_JITTER_PTS else rng.choice(vals, MAX_JITTER_PTS, replace=False)
    jitter = rng.uniform(-0.09, 0.09, size=len(sample))
    ax.scatter(pos + jitter, sample, color=COLOR, alpha=0.30, s=10, zorder=3, linewidths=0)

ax.set_xticks(positions)
ax.set_xticklabels(datasets, fontsize=16)
ax.set_ylabel('Similarity', fontsize=16)
ax.tick_params(axis='y', labelsize=15)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / 'boxplot_similarity.svg', dpi=150, bbox_inches='tight')
plt.show()

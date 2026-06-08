"""
Analisi Descrittori Molecolari - Confronto Distribuzioni
=========================================================
Questo script calcola descrittori molecolari completi e confronta le distribuzioni tra:
  1. Molecole con pKi >= 6 (ATTIVE)
  2. Molecole con pKi < 6 (INATTIVE)
  3. Molecole generate (all_unique_molecules.csv)

Visualizzazione: Solo curve di densità (KDE)
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED
from rdkit.Chem import AllChem
from scipy.stats import gaussian_kde, mannwhitneyu, ttest_ind
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

DESCRIPTOR_NAMES = [
    'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds',
    'NumRings', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
    'NumHeteroatoms', 'NumStereocenters', 'FractionCSP3', 'NumHeavyAtoms',
    'QED', 'Lipinski_Violations', 'SASA', 'MolVolume', 'BertzCT',
    'NumRadicalElectrons', 'NumValenceElectrons', 'NumAmideBonds',
    'NumBridgeheadAtoms', 'NumSpiroAtoms', 'Aromatic_Ratio',
    'Heteroatom_Ratio', 'Polar_Ratio'
]

NAN_SERIES = pd.Series({d: np.nan for d in DESCRIPTOR_NAMES})


def calculate_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return NAN_SERIES.copy()

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
        num_saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)
        num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
        num_heavy_atoms = Lipinski.HeavyAtomCount(mol)
        try:
            fraction_csp3 = rdMolDescriptors.CalcFractionCsp3(mol)
        except AttributeError:
            fraction_csp3 = Lipinski.FractionCSP3(mol)

        num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        qed_score = QED.qed(mol)
        lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        try:
            sasa = rdMolDescriptors.CalcLabuteASA(mol)
        except Exception:
            sasa = np.nan

        try:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, randomSeed=42) == 0:
                mol_volume = AllChem.ComputeMolVolume(mol_3d, gridSpacing=0.2)
            else:
                mol_volume = np.nan
        except Exception:
            mol_volume = np.nan

        try:
            bertz_ct = Descriptors.BertzCT(mol)
        except Exception:
            bertz_ct = np.nan

        num_radical_electrons = Descriptors.NumRadicalElectrons(mol)
        num_valence_electrons = Descriptors.NumValenceElectrons(mol)

        amide_pattern = Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6]')
        num_amide_bonds = len(mol.GetSubstructMatches(amide_pattern)) if amide_pattern else 0

        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

        aromatic_ratio = num_aromatic_rings / num_rings if num_rings > 0 else 0
        heteroatom_ratio = num_heteroatoms / num_heavy_atoms if num_heavy_atoms > 0 else 0
        polar_ratio = (hbd + hba) / num_heavy_atoms if num_heavy_atoms > 0 else 0

        return pd.Series({
            'MW': mw, 'LogP': logp, 'HBD': hbd, 'HBA': hba,
            'TPSA': tpsa, 'RotatableBonds': rotatable,
            'NumRings': num_rings, 'NumAromaticRings': num_aromatic_rings,
            'NumAliphaticRings': num_aliphatic_rings, 'NumSaturatedRings': num_saturated_rings,
            'NumHeteroatoms': num_heteroatoms, 'NumStereocenters': num_stereocenters,
            'FractionCSP3': fraction_csp3, 'NumHeavyAtoms': num_heavy_atoms,
            'QED': qed_score, 'Lipinski_Violations': lipinski_violations,
            'SASA': sasa, 'MolVolume': mol_volume, 'BertzCT': bertz_ct,
            'NumRadicalElectrons': num_radical_electrons,
            'NumValenceElectrons': num_valence_electrons,
            'NumAmideBonds': num_amide_bonds,
            'NumBridgeheadAtoms': num_bridgehead, 'NumSpiroAtoms': num_spiro,
            'Aromatic_Ratio': aromatic_ratio, 'Heteroatom_Ratio': heteroatom_ratio,
            'Polar_Ratio': polar_ratio
        })

    except Exception as e:
        print(f"Errore nel calcolo dei descrittori: {e}")
        return NAN_SERIES.copy()


# ==============================================================================
# PIPELINE PER OGNI TARGET
# ==============================================================================

datasets = ['PIM1', 'JAK1', 'AR', 'PPAR']

for dataset in datasets:

    print("=" * 80)
    print(f"DATASET: {dataset}")
    print("=" * 80)

    # --- Caricamento dati ---
    df = pd.read_csv(_REPO_ROOT / 'data' / dataset / 'dataset.csv')
    print(f"✓ Dataset principale: {len(df)} molecole")

    df_docking = pd.read_csv(
        _REPO_ROOT / 'STRIPES2SMILES' / 'results_finetune'
        / f'{dataset}_finetuned' / 'comparison' / 'all_generated_combined' / 'all_unique_molecules.csv'
    )
    print(f"✓ Dataset molecole generate: {len(df_docking)} molecole")

    smiles_col = None
    for candidate in ('smiles', 'canonical_smiles', 'predicted_smiles'):
        if candidate in df_docking.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        print(f"⚠️  Nessuna colonna SMILES trovata. Colonne: {list(df_docking.columns)}")
        smiles_col = df_docking.columns[0]
    print(f"✓ Colonna SMILES molecole generate: '{smiles_col}'")

    # --- Calcolo descrittori ---
    print("\nCalcolo descrittori dataset principale...")
    descriptors_main = df['smiles'].apply(calculate_molecular_descriptors)
    df_descriptors = pd.concat([df[['smiles', 'pKi']], descriptors_main], axis=1)
    n_invalid = df_descriptors['MW'].isna().sum()
    if n_invalid > 0:
        print(f"⚠️  Rimosse {n_invalid} molecole con SMILES invalide")
        df_descriptors = df_descriptors.dropna(subset=['MW'])

    print("\nCalcolo descrittori molecole generate...")
    descriptors_generated = df_docking[smiles_col].apply(calculate_molecular_descriptors)
    df_generated_descriptors = pd.concat([df_docking[[smiles_col]], descriptors_generated], axis=1)
    n_invalid_gen = df_generated_descriptors['MW'].isna().sum()
    if n_invalid_gen > 0:
        print(f"⚠️  Rimosse {n_invalid_gen} molecole generate con SMILES invalide")
        df_generated_descriptors = df_generated_descriptors.dropna(subset=['MW'])

    # --- Split attive/inattive ---
    df_active = df_descriptors[df_descriptors['pKi'] >= 6].copy()
    df_inactive = df_descriptors[df_descriptors['pKi'] < 6].copy()

    print(f"\n✓ Attive (pKi >= 6): {len(df_active)}")
    print(f"✓ Inattive (pKi < 6): {len(df_inactive)}")
    print(f"✓ Generate: {len(df_generated_descriptors)}")

    all_descriptors = [col for col in df_descriptors.columns if col not in ['smiles', 'pKi']]

    # --- Visualizzazione KDE ---
    fixed_descriptors = [
        'HBA', 'TPSA', 'RotatableBonds', 'NumRings',
        'NumHeteroatoms', 'FractionCSP3', 'NumHeavyAtoms', 'SASA'
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(fixed_descriptors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for idx, descriptor in enumerate(fixed_descriptors):
        ax = axes[idx]

        data_active = df_active[descriptor].dropna()
        data_inactive = df_inactive[descriptor].dropna()
        data_generated = df_generated_descriptors[descriptor].dropna()

        if len(data_active) == 0 and len(data_inactive) == 0 and len(data_generated) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(descriptor)
            continue

        all_data = pd.concat([data_active, data_inactive, data_generated])
        min_val, max_val = all_data.min(), all_data.max()
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        x_range = np.linspace(min_val, max_val, 200)

        if len(data_inactive) > 5:
            try:
                ax.plot(x_range, gaussian_kde(data_inactive)(x_range),
                        color='coral', linewidth=2.5, alpha=0.9)
            except Exception:
                pass
        if len(data_active) > 5:
            try:
                ax.plot(x_range, gaussian_kde(data_active)(x_range),
                        color='green', linewidth=2.5, alpha=0.9)
            except Exception:
                pass
        if len(data_generated) > 5:
            try:
                ax.plot(x_range, gaussian_kde(data_generated)(x_range),
                        color='blue', linewidth=2.5, alpha=0.9)
            except Exception:
                pass

        if len(data_inactive) > 0:
            ax.axvline(data_inactive.mean(), color='darkred', linestyle='--', linewidth=1.5, alpha=0.6)
        if len(data_active) > 0:
            ax.axvline(data_active.mean(), color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.6)
        if len(data_generated) > 0:
            ax.axvline(data_generated.mean(), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_xlabel(descriptor, fontsize=22)
        ax.set_ylabel('Density', fontsize=22)
        ax.set_title(descriptor, fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', labelsize=19)
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(len(fixed_descriptors), len(axes)):
        axes[idx].axis('off')

    legend_handles = [
        Line2D([0], [0], color='coral', linewidth=2.5, label='Inactive'),
        Line2D([0], [0], color='green', linewidth=2.5, label='Active'),
        Line2D([0], [0], color='blue', linewidth=2.5, label='Generated'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=22, frameon=True, bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = _REPO_ROOT / 'figures' / 'descriptors' / f'{dataset}_descriptors_kde_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figura salvata: {output_path}")

    # --- Statistiche descrittive ---
    summary_data = []
    for descriptor in all_descriptors:
        da = df_active[descriptor].dropna()
        di = df_inactive[descriptor].dropna()
        dg = df_generated_descriptors[descriptor].dropna()
        summary_data.append({
            'Descriptor': descriptor,
            'Active_Mean': da.mean() if len(da) > 0 else np.nan,
            'Active_Std': da.std() if len(da) > 0 else np.nan,
            'Inactive_Mean': di.mean() if len(di) > 0 else np.nan,
            'Inactive_Std': di.std() if len(di) > 0 else np.nan,
            'Generated_Mean': dg.mean() if len(dg) > 0 else np.nan,
            'Generated_Std': dg.std() if len(dg) > 0 else np.nan,
            'Active_N': len(da),
            'Inactive_N': len(di),
            'Generated_N': len(dg),
        })
    df_summary = pd.DataFrame(summary_data)
    print("\nStatistiche descrittive (prime 10 righe):")
    print(df_summary.head(10).to_string(index=False))

    # --- Analisi significatività ---
    significance_data = []
    for descriptor in all_descriptors:
        da = df_active[descriptor].dropna()
        di = df_inactive[descriptor].dropna()
        dg = df_generated_descriptors[descriptor].dropna()
        if len(da) > 5 and len(di) > 5:
            try:
                _, p_mw = mannwhitneyu(da, di, alternative='two-sided')
                _, p_t = ttest_ind(da, di)
                pooled_std = np.sqrt(
                    ((len(da) - 1) * da.std()**2 + (len(di) - 1) * di.std()**2)
                    / (len(da) + len(di) - 2)
                )
                cohens_d = (da.mean() - di.mean()) / pooled_std if pooled_std > 0 else 0
                significance_data.append({
                    'Descriptor': descriptor,
                    'Active_Mean': da.mean(),
                    'Inactive_Mean': di.mean(),
                    'Generated_Mean': dg.mean() if len(dg) > 0 else np.nan,
                    'Diff_Active_Inactive': da.mean() - di.mean(),
                    'Cohens_d': cohens_d,
                    'p_value_MannWhitney': p_mw,
                    'p_value_ttest': p_t,
                    'Significant': 'YES' if p_mw < 0.05 else 'NO'
                })
            except Exception:
                pass

    df_significance = pd.DataFrame(significance_data)
    if len(df_significance) > 0:
        df_significance = df_significance.sort_values('p_value_MannWhitney')
        print("\nTOP 10 DESCRITTORI PIU' DISCRIMINANTI (Active vs Inactive):")
        print("-" * 100)
        header = "{:<6} {:<25} {:<12} {:<12} {:<12} {}".format(
            "Rank", "Descriptor", "p-value", "Cohen's d", "Diff Mean", "Significant"
        )
        print(header)
        print("-" * 100)
        for rank, (_, row) in enumerate(df_significance.head(10).iterrows(), start=1):
            print("{:<6} {:<25} {:<12.2e} {:<12.3f} {:<12.3f} {}".format(
                rank, row["Descriptor"], row["p_value_MannWhitney"],
                row["Cohens_d"], row["Diff_Active_Inactive"], row["Significant"]
            ))
    else:
        print("⚠️  Nessun dato sufficiente per l'analisi di significatività")

    print(f"\nDataset {dataset} completato:")
    print(f"  - Attive: {len(df_active)}  Inattive: {len(df_inactive)}  Generate: {len(df_generated_descriptors)}")
    print("=" * 80 + "\n")


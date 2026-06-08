from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

_REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVITY_COLS = ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']
DATASETS = ['AR', 'PIM1', 'PPAR', 'JAK1']


def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_max_sim(query_smiles_series, ref_fps, ref_smiles):
    max_sims = []
    most_similar = []
    for smi in query_smiles_series:
        query_fp = morgan_fp(smi)
        if query_fp is None or len(ref_fps) == 0:
            max_sims.append(None)
            most_similar.append(None)
            continue
        sims = BulkTanimotoSimilarity(query_fp, ref_fps)
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        max_sims.append(sims[best_idx])
        most_similar.append(ref_smiles[best_idx])
    return max_sims, most_similar


for dataset in DATASETS:
    bioact_path = _REPO_ROOT / 'PubChem_analysis' / 'results' / f'{dataset}_bioactivity_results.csv'
    dataset_path = _REPO_ROOT / 'data' / dataset / 'dataset.csv'
    test_set_path = (
        _REPO_ROOT / 'STRIPES2SMILES' / 'results_finetune'
        / f'{dataset}_finetuned' / f'{dataset}_test_set.csv'
    )

    df_bio = pd.read_csv(bioact_path)
    df_ref = pd.read_csv(dataset_path)
    df_test = pd.read_csv(test_set_path)

    test_mol_ids = set(df_test['mol_id'])
    df_ref_train = df_ref[~df_ref['mol_id'].isin(test_mol_ids)]

    active_mask = df_bio[ACTIVITY_COLS].notna().any(axis=1)
    df_active = df_bio[active_mask].copy()

    # build fingerprint lists for full dataset and train-only
    def build_fps(df):
        fps, smiles = [], []
        for smi in df['can_smiles']:
            fp = morgan_fp(smi)
            if fp is not None:
                fps.append(fp)
                smiles.append(smi)
        return fps, smiles

    ref_fps_all, ref_smiles_all = build_fps(df_ref)
    ref_fps_train, ref_smiles_train = build_fps(df_ref_train)

    max_sims_all, most_sim_all = compute_max_sim(df_active['canonical_smiles'], ref_fps_all, ref_smiles_all)
    max_sims_train, most_sim_train = compute_max_sim(df_active['canonical_smiles'], ref_fps_train, ref_smiles_train)

    df_active['max_tanimoto_full'] = max_sims_all
    df_active['most_similar_full'] = most_sim_all
    df_active['max_tanimoto_train'] = max_sims_train
    df_active['most_similar_train'] = most_sim_train

    out_path = _REPO_ROOT / 'PubChem_analysis' / 'results' / 'max_tan_similarity' / f'{dataset}_bioactivity_max_sim.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_active.to_csv(out_path, index=False)
    print(f'{dataset}: {len(df_active)} active molecules, train ref = {len(df_ref_train)}/{len(df_ref)} → {out_path}')


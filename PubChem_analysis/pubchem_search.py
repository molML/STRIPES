import argparse
import os
import pandas as pd
import requests
import urllib.parse
import time
from pathlib import Path
from rdkit import Chem

pd.set_option('display.max_colwidth', None)

_REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_RESULTS_DIR = str(_REPO_ROOT / "STRIPES2SMILES" / "results_finetune")
BASE_DATA_DIR    = str(_REPO_ROOT / "data")

TARGET_INFO = {
    'AR':   {'gene_id': '367',  'accession': 'P10275', 'keywords': ['androgen receptor', 'androgen'],
             'chembl_target': 'CHEMBL1871'},
    'JAK1': {'gene_id': '3716', 'accession': 'P23458', 'keywords': ['jak1', 'janus kinase 1'],
             'chembl_target': 'CHEMBL2835'},
    'PIM1': {'gene_id': '5292', 'accession': 'P11309', 'keywords': ['pim1', 'pim-1 kinase'],
             'chembl_target': 'CHEMBL2147'},
    'PPAR': {'gene_id': '5467', 'accession': 'Q03181', 'keywords': ['ppard', 'ppar delta', 'ppar-delta',
             'peroxisome proliferator-activated receptor delta'], 'chembl_target': 'CHEMBL3729'},
}


def canonicalize(smi):
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def pubchem_canonical_smiles(smiles):
    try:
        smiles_enc = urllib.parse.quote(smiles, safe='')
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_enc}/cids/TXT",
            timeout=5)
        if r.status_code != 200:
            return None
        cid = r.text.strip()
        if not cid.isdigit():
            return None
        r2 = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT",
            timeout=5)
        if r2.status_code != 200:
            return None
        return r2.text.strip()
    except requests.exceptions.RequestException:
        return None


def exists_exact_on_pubchem(smiles):
    smi_norm = canonicalize(smiles)
    if smi_norm is None:
        return False
    pubchem_smi = pubchem_canonical_smiles(smi_norm)
    if pubchem_smi is None:
        return False
    return canonicalize(pubchem_smi) == smi_norm


def get_cid_from_smiles(smiles):
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
               f"{urllib.parse.quote(smiles, safe='')}/cids/JSON")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                return data['IdentifierList']['CID'][0]
    except Exception as e:
        print(f"    CID error: {e}")
    return None


def get_chembl_id_from_smiles(smiles):
    try:
        url = (f"https://www.ebi.ac.uk/chembl/api/data/molecule.json"
               f"?molecule_structures__canonical_smiles__exact={urllib.parse.quote(smiles, safe='')}"
               f"&format=json")
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            molecules = r.json().get('molecules', [])
            if molecules:
                return molecules[0]['molecule_chembl_id']
    except requests.exceptions.RequestException:
        pass
    return None


def exists_exact_on_chembl(smiles):
    smi_norm = canonicalize(smiles)
    if smi_norm is None:
        return False
    return get_chembl_id_from_smiles(smi_norm) is not None


def get_bioactivity_for_target(cid, target_name):
    target    = TARGET_INFO.get(target_name, {})
    gene_id   = target.get('gene_id', '')
    accession = target.get('accession', '')
    keywords  = target.get('keywords', [])

    activity_data = {
        'EC50_uM': None, 'IC50_uM': None, 'Ki_uM': None,
        'Kd_uM': None, 'AC50_uM': None, 'found_target_data': False
    }

    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON"
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if 'Table' in data:
                columns = data['Table'].get('Columns', {}).get('Column', [])
                rows    = data['Table'].get('Row', [])
                for row in rows:
                    cells    = row.get('Cell', [])
                    row_dict = {col: cells[i] for i, col in enumerate(columns) if i < len(cells)}

                    is_target = (
                        str(row_dict.get('Target GeneID', '')) == gene_id or
                        str(row_dict.get('Target Accession', '')) == accession or
                        any(kw in str(row_dict.get('Assay Name', '')).lower() for kw in keywords)
                    )

                    if is_target:
                        activity_data['found_target_data'] = True
                        activity_name  = str(row_dict.get('Activity Name', '')).upper().strip()
                        activity_value = row_dict.get('Activity Value [uM]', '')
                        try:
                            value = float(activity_value) if activity_value else None
                        except (ValueError, TypeError):
                            value = None

                        if value is not None:
                            key_map = {'EC50': 'EC50_uM', 'IC50': 'IC50_uM',
                                       'KI': 'Ki_uM', 'KD': 'Kd_uM', 'AC50': 'AC50_uM'}
                            key = key_map.get(activity_name)
                            if key and (activity_data[key] is None or value < activity_data[key]):
                                activity_data[key] = value
    except Exception as e:
        print(f"    Bioactivity error: {e}")

    return activity_data


def get_bioactivity_from_chembl(chembl_id, target_name):
    chembl_target = TARGET_INFO.get(target_name, {}).get('chembl_target', '')
    activity_data = {
        'EC50_uM': None, 'IC50_uM': None, 'Ki_uM': None,
        'Kd_uM': None, 'AC50_uM': None, 'found_target_data': False
    }
    if not chembl_target:
        return activity_data

    try:
        url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
               f"?molecule_chembl_id={chembl_id}"
               f"&target_chembl_id={chembl_target}"
               f"&standard_type__in=IC50,EC50,Ki,Kd,AC50"
               f"&limit=100&format=json")
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return activity_data

        for act in r.json().get('activities', []):
            std_type  = str(act.get('standard_type', '')).upper().strip()
            std_value = act.get('standard_value')
            std_units = str(act.get('standard_units', '')).lower()
            if std_value is None:
                continue
            try:
                value = float(std_value)
            except (ValueError, TypeError):
                continue

            if 'nm' in std_units:
                value /= 1000
            elif 'mm' in std_units:
                value *= 1000
            elif 'pm' in std_units:
                value /= 1e6

            activity_data['found_target_data'] = True
            key_map = {'IC50': 'IC50_uM', 'EC50': 'EC50_uM',
                       'KI': 'Ki_uM', 'KD': 'Kd_uM', 'AC50': 'AC50_uM'}
            key = key_map.get(std_type)
            if key and (activity_data[key] is None or value < activity_data[key]):
                activity_data[key] = value

    except Exception as e:
        print(f"    ChEMBL bioactivity error: {e}")

    return activity_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="PubChem/ChEMBL bioactivity search on pre-generated unique molecules."
    )
    parser.add_argument(
        '--dataset', required=True,
        choices=list(TARGET_INFO.keys()),
        help="Target dataset to process (e.g. AR, JAK1, PIM1, PPAR)."
    )
    parser.add_argument(
        '--input', default=None,
        help="Direct path to all_unique_molecules.csv. "
             "If given, --results_dir is ignored."
    )
    parser.add_argument(
        '--results_dir', default=BASE_RESULTS_DIR,
        help=f"Root directory containing <DATASET>_finetuned subfolders. "
             f"Used only when --input is not provided. "
             f"(default: {BASE_RESULTS_DIR})"
    )
    parser.add_argument(
        '--data_dir', default=BASE_DATA_DIR,
        help=f"Root directory containing reference datasets (<DATASET>/dataset.csv). "
             f"(default: {BASE_DATA_DIR})"
    )
    parser.add_argument(
        '--output_dir', default=None,
        help="Directory to save the bioactivity CSV. "
             "Defaults to the same folder as the input CSV."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset

    if args.input:
        input_path = args.input
    else:
        input_path = os.path.join(
            args.results_dir,
            f"{dataset}_finetuned",
            "comparison",
            "all_generated_combined",
            "all_unique_molecules.csv",
        )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset}")
    print(f"Input   : {input_path}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path)
    if 'canonical_smiles' not in df.columns:
        raise ValueError(f"Expected column 'canonical_smiles' in {input_path}, got: {list(df.columns)}")

    df['canonical_smiles'] = df['canonical_smiles'].apply(canonicalize)
    df = df.dropna(subset=['canonical_smiles']).drop_duplicates(subset=['canonical_smiles']).reset_index(drop=True)
    print(f"Loaded {len(df)} unique valid molecules.")

    # ---- Load reference dataset and filter known molecules ----
    ref_path = os.path.join(args.data_dir, dataset, "dataset.csv")
    if not os.path.isfile(ref_path):
        print(f"[WARNING] Reference dataset not found at {ref_path} — skipping novelty filter.")
        reference_smiles_set = set()
    else:
        df_ref = pd.read_csv(ref_path)
        reference_smiles_set = set(df_ref['can_smiles'].apply(canonicalize).dropna())
        print(f"Reference dataset: {len(reference_smiles_set)} molecules from {ref_path}")

    before = len(df)
    df = df[~df['canonical_smiles'].isin(reference_smiles_set)].reset_index(drop=True)
    print(f"After novelty filter: {len(df)} molecules ({before - len(df)} removed as already in training set).")

    out_dir = args.output_dir or os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Check existence on PubChem and ChEMBL ----
    print(f"\nChecking existence on PubChem and ChEMBL ({len(df)} molecules)...")
    df['exists_on_pubchem'] = df['canonical_smiles'].apply(exists_exact_on_pubchem)
    df['exists_on_chembl']  = df['canonical_smiles'].apply(exists_exact_on_chembl)
    df['exists_in_db']      = df['exists_on_pubchem'] | df['exists_on_chembl']

    pubchem_count = df['exists_on_pubchem'].sum()
    chembl_count  = df['exists_on_chembl'].sum()
    both_count    = (df['exists_on_pubchem'] & df['exists_on_chembl']).sum()
    print(f"  Found in PubChem: {pubchem_count}, ChEMBL: {chembl_count}, both: {both_count}")

    df_to_search = df[df['exists_in_db']].copy()
    print(f"  Molecules to search for bioactivity: {len(df_to_search)}")

    # ---- Bioactivity search ----
    print(f"\nSearching {dataset} bioactivity on PubChem + ChEMBL...")
    bioactivity_results = []

    for idx, row_data in enumerate(df_to_search.itertuples(index=False)):
        smiles     = row_data.canonical_smiles
        on_pubchem = row_data.exists_on_pubchem
        on_chembl  = row_data.exists_on_chembl

        print(f"  [{idx+1}/{len(df_to_search)}] {smiles[:60]}...")

        row = {
            'canonical_smiles': smiles,
            'target': dataset,
            'exists_on_pubchem': on_pubchem,
            'exists_on_chembl': on_chembl,
            'EC50_uM': None, 'IC50_uM': None,
            'Ki_uM': None, 'Kd_uM': None, 'AC50_uM': None,
        }

        if on_pubchem:
            cid = get_cid_from_smiles(smiles)
            if cid:
                bioact_pub = get_bioactivity_for_target(cid, dataset)
                for k in ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']:
                    if bioact_pub[k] is not None:
                        row[k] = bioact_pub[k]
                if bioact_pub['found_target_data']:
                    vals = [f"{k}: {bioact_pub[k]}" for k in ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM'] if bioact_pub[k] is not None]
                    print(f"    [PubChem] {dataset}: {', '.join(vals) if vals else 'outcome only'}")
                else:
                    print(f"    [PubChem] No {dataset} assay found")

        if on_chembl:
            chembl_id = get_chembl_id_from_smiles(smiles)
            if chembl_id:
                bioact_che = get_bioactivity_from_chembl(chembl_id, dataset)
                for k in ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']:
                    if bioact_che[k] is not None and (row[k] is None or bioact_che[k] < row[k]):
                        row[k] = bioact_che[k]
                if bioact_che['found_target_data']:
                    vals = [f"{k}: {bioact_che[k]}" for k in ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM'] if bioact_che[k] is not None]
                    print(f"    [ChEMBL]  {dataset}: {', '.join(vals) if vals else 'outcome only'}")
                else:
                    print(f"    [ChEMBL]  No {dataset} assay found")

        if not any(row[k] is not None for k in ['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']):
            print(f"    [--] No {dataset} bioactivity data in either database")

        bioactivity_results.append(row)
        time.sleep(0.5)

    bioact_df     = pd.DataFrame(bioactivity_results)
    bioact_output = os.path.join(out_dir, f"{dataset}_bioactivity_results.csv")
    bioact_df.to_csv(bioact_output, index=False)

    has_bioact = bioact_df[bioact_df[['EC50_uM', 'IC50_uM', 'Ki_uM', 'Kd_uM', 'AC50_uM']].notna().any(axis=1)]
    print(f"\n  Molecules searched : {len(bioact_df)}")
    print(f"  With bioactivity   : {len(has_bioact)}")
    print(f"  Saved to           : {bioact_output}")


if __name__ == "__main__":
    main()

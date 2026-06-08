from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
pd.set_option('display.max_colwidth', None)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(_REPO_ROOT / 'STRIPES_similarity' / 'results' / 'PIM1' / 'results.csv')
df = df[df['smiles1'] != df['smiles2']]
def ecfp_tanimoto(smi1, smi2, radius=2, nbits=2048):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius, nBits=nbits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

df['tanimoto_ecfp4'] = df.apply(lambda row: ecfp_tanimoto(row['smiles1'], row['smiles2']), axis=1)
df

df_lowSTR_highTan = df[(df['similarity'] < 0.3) & (df['tanimoto_ecfp4'] < 0.10) & (abs(df['pKi1'] - df['pKi2']) > 2.35) & (abs(df['pKi1'] - df['pKi2']) < 2.45)]
df_lowSTR_highTan




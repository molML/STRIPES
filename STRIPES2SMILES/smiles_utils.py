import re
from typing import List, Optional
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|\<BEG\>|\<PAD\>|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"

__REGEXES = {
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
}

_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    """Segment SMILES string into tokens"""
    regex = _RE_PATTERNS["segmentation_sq"]
    return regex.findall(smiles)


def segment_smiles_batch(
    smiles_batch: List[str], segment_sq_brackets=True
) -> List[List[str]]:
    """Segment a batch of SMILES strings"""
    return [segment_smiles(smiles, segment_sq_brackets) for smiles in smiles_batch]


def sanitize_smiles(
    smiles: str,
    to_canonical=True,
) -> Optional[str]:
    """Sanitize and optionally canonicalize SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        sanitization_flag = Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
        )
        
        # SANITIZE_NONE is the "no error" flag of rdkit!
        if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
        
        return Chem.MolToSmiles(mol, canonical=to_canonical)
    except Exception:
        return None


def sanitize_smiles_batch(
    smiles_batch: List[str],
    to_canonical=True,
) -> List[Optional[str]]:
    """Sanitize a batch of SMILES strings"""
    return [
        sanitize_smiles(
            smiles,
            to_canonical=to_canonical,
        )
        for smiles in smiles_batch
    ]


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit with enhanced validation"""
    if not smiles or smiles == "INVALID" or not isinstance(smiles, str):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Additional validation: try to sanitize
        sanitization_flag = Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
        )
        
        # SANITIZE_NONE means no error
        return sanitization_flag == Chem.SanitizeFlags.SANITIZE_NONE
    except Exception:
        return False

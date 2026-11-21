"""Molecule library + RDKit property labels (molecular weight, chlorine, similarity)."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).parent / "data"
BUILTIN = DATA_DIR / "builtin_molecules.tsv"
CHEBI_LIBRARY = DATA_DIR / "chebi_library.tsv"   # written by scripts/00_fetch_chebi.py

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class Molecule:
    name: str
    smiles: str
    chebi_id: str = ""           # e.g. "CHEBI:17578"; empty if unknown
    mw: float = 0.0
    has_cl: bool = False
    n_cl: int = 0
    inchikey: str = ""

    @property
    def display_id(self) -> str:
        """Opaque identifier for ID-mode lists (no name cue)."""
        if self.chebi_id:
            return self.chebi_id
        h = int(hashlib.sha1(self.inchikey.encode()).hexdigest(), 16) % 10**6
        return f"MOL:{h:06d}"


def from_smiles(name: str, smiles: str, chebi_id: str = "") -> Molecule | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_cl = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    return Molecule(
        name=name,
        smiles=Chem.MolToSmiles(mol),
        chebi_id=chebi_id,
        mw=round(Descriptors.MolWt(mol), 2),
        has_cl=n_cl > 0,
        n_cl=n_cl,
        inchikey=Chem.MolToInchiKey(mol),
    )


@lru_cache(maxsize=200_000)
def _fp(smiles: str):
    return _FPGEN.GetFingerprint(Chem.MolFromSmiles(smiles))


def similarity(a: Molecule, b: Molecule) -> float:
    """Tanimoto similarity on Morgan (r=2, 2048-bit) fingerprints, rounded to 2 dp."""
    return round(DataStructs.TanimotoSimilarity(_fp(a.smiles), _fp(b.smiles)), 2)


def load_library(path: str | Path | None = None) -> list[Molecule]:
    """Load the ChEBI-derived library if present, else the built-in seed library."""
    if path is None:
        path = CHEBI_LIBRARY if CHEBI_LIBRARY.exists() else BUILTIN
    out, seen = [], set()
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m = from_smiles(row["name"], row["smiles"], row.get("chebi_id", "") or "")
            if m is None or m.inchikey in seen or m.name.lower() in seen:
                continue
            seen.update({m.inchikey, m.name.lower()})
            out.append(m)
    return out

"""Build data/chebi_library.tsv from ChEBI (run on a machine with internet, e.g. PACE login node).

Downloads the ChEBI 3-star SDF, keeps small neutral organic molecules with readable names,
and always includes the built-in seed molecules (mapped to ChEBI IDs by InChIKey).
"""
import csv
import gzip
import shutil
import urllib.request

from common import ROOT, base_parser  # noqa: F401
from rdkit import Chem
from rdkit.Chem import Descriptors

from wspmol.molecules import BUILTIN, CHEBI_LIBRARY, from_smiles

URLS = [
    "https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi_3_stars.sdf.gz",
    "https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/ChEBI_complete_3star.sdf.gz",
]
ALLOWED = {"C", "H", "N", "O", "S", "F", "Cl", "Br", "P"}


def download(dest):
    for u in URLS:
        try:
            print("downloading", u)
            with urllib.request.urlopen(u, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception as e:
            print("  failed:", e)
    raise SystemExit("could not download ChEBI SDF; pass --sdf with a local copy")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--sdf", default=None)
    ap.add_argument("--max-heavy", type=int, default=32)
    ap.add_argument("--max-name-len", type=int, default=32)
    a = ap.parse_args()
    sdf = a.sdf or str(ROOT / "data" / "chebi_3star.sdf.gz")
    if a.sdf is None and not (ROOT / "data" / "chebi_3star.sdf.gz").exists():
        download(sdf)
    rows, seen = {}, set()
    with gzip.open(sdf, "rb") as fh:
        for mol in Chem.ForwardSDMolSupplier(fh):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            cid = str(props.get("ChEBI ID", props.get("chebi_id", "")))
            name = str(props.get("ChEBI NAME", props.get("ChEBI Name", props.get("name", ""))))
            if not cid or not name:
                continue
            if not cid.startswith("CHEBI:"):
                cid = f"CHEBI:{cid}"
            if len(name) > a.max_name_len or any(ch in name for ch in "[]{}<>;"):
                continue
            if len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
                continue
            if any(at.GetFormalCharge() != 0 and at.GetSymbol() != "N" and at.GetSymbol() != "O"
                   for at in mol.GetAtoms()):
                continue
            if any(at.GetSymbol() not in ALLOWED for at in mol.GetAtoms()) or mol.GetNumHeavyAtoms() > a.max_heavy:
                continue
            if mol.GetNumHeavyAtoms() < 4 or not any(at.GetSymbol() == "C" for at in mol.GetAtoms()):
                continue
            if Descriptors.MolWt(mol) > 500:
                continue
            m = from_smiles(name, Chem.MolToSmiles(mol), cid)
            if m is None or m.inchikey in seen:
                continue
            seen.add(m.inchikey)
            rows[m.inchikey] = (m.name, m.smiles, cid)
    # seed molecules: keep, attach ChEBI ID when the structure matches
    with open(BUILTIN) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            m = from_smiles(r["name"], r["smiles"])
            cid = rows.get(m.inchikey, (None, None, ""))[2]
            rows[m.inchikey] = (m.name, m.smiles, cid)
    with open(CHEBI_LIBRARY, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["name", "smiles", "chebi_id"])
        for name, smi, cid in rows.values():
            w.writerow([name, smi, cid])
    print(f"wrote {len(rows)} molecules to {CHEBI_LIBRARY}")


if __name__ == "__main__":
    main()

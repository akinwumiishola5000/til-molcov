"""Chemistry for MolCov, the molecule converter in The Insilico Lab CADD Toolkit.

Everything that touches RDKit lives here, apart from the interface, so trainees
can read it, reuse it in Colab, and test it on its own.

    reading   : SMILES, SDF/MOL, MOL2, PDB, PDBQT, XYZ, SMI files
    geometry  : 2D depiction and 3D conformer (ETKDGv3 + MMFF94)
    writing   : SDF, MOL, PDB, MOL2, XYZ, SMILES
    properties: the same set as the Molecular Learning Workspace, plus identifiers
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

try:  # available in every RDKit release since 2022.09
    from rdkit.Chem import rdDetermineBonds
except ImportError:  # pragma: no cover
    rdDetermineBonds = None

RDLogger.DisableLog("rdApp.*")
rdDepictor.SetPreferCoordGen(True)

MAX_HEAVY_ATOMS_3D = 200

# Lipinski's rule of five (Lipinski et al., Adv. Drug Deliv. Rev. 1997)
LIPINSKI_LIMITS = {
    "Molecular weight": ("mw", 500.0),
    "LogP": ("logp", 5.0),
    "H-bond donors": ("hbd", 5),
    "H-bond acceptors": ("hba", 10),
}
# Veber's oral bioavailability rules (Veber et al., J. Med. Chem. 2002)
VEBER_LIMITS = {
    "Rotatable bonds": ("rotb", 10),
    "TPSA": ("tpsa", 140.0),
}

READABLE_EXTENSIONS = ["sdf", "sd", "mol", "mol2", "pdb", "ent", "pdbqt", "xyz", "smi", "smiles", "txt"]


# ---------------------------------------------------------------- reading
@dataclass
class LoadedMolecule:
    name: str
    mol: Chem.Mol
    has_3d: bool = False
    note: str = ""  # anything the user should know about how it was read


def from_smiles(smiles: str) -> Chem.Mol | None:
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def _is_3d(mol: Chem.Mol) -> bool:
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    return any(abs(conf.GetAtomPosition(i).z) > 1e-3 for i in range(mol.GetNumAtoms()))


def _name_of(mol: Chem.Mol, fallback: str) -> str:
    for key in ("_Name", "name", "Name", "TITLE"):
        if mol.HasProp(key) and mol.GetProp(key).strip():
            return mol.GetProp(key).strip()
    return fallback


def _gentle_sanitize(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Sanitise, and if that fails try again without the valence check."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:  # noqa: BLE001
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
            return mol
        except Exception:  # noqa: BLE001
            return None


def _smiles_ok(mol: Chem.Mol) -> bool:
    try:
        Chem.MolToSmiles(Chem.RemoveHs(mol))
        return True
    except Exception:  # noqa: BLE001
        return False


def _tidy(mol: Chem.Mol) -> Chem.Mol:
    """Drop square-planar and other unusual stereo tags that flat files can produce."""
    keep = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    for a in mol.GetAtoms():
        if a.GetChiralTag() not in keep:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return mol


def _bonds_from_coordinates(mol: Chem.Mol, charge: int = 0) -> tuple[Chem.Mol | None, str]:
    """Rebuild bonds (and bond orders where possible) from 3D coordinates."""
    if rdDetermineBonds is None:
        return None, "This RDKit version cannot work out bonds from coordinates."
    rw = Chem.RWMol(mol)
    for b in list(rw.GetBonds()):
        rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
    m = rw.GetMol()
    has_h = any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
    try:
        rdDetermineBonds.DetermineConnectivity(m)
    except Exception:  # noqa: BLE001
        return None, "Bonds could not be worked out from the coordinates."
    if has_h:
        for q in dict.fromkeys((charge, 0, 1, -1, 2, -2)):
            try:
                trial = Chem.Mol(m)
                rdDetermineBonds.DetermineBondOrders(trial, charge=q)
                Chem.SanitizeMol(trial)
                Chem.MolToSmiles(trial)
                return trial, "Bonds and bond orders were worked out from the atom positions. Check the structure."
            except Exception:  # noqa: BLE001
                continue
    for a in m.GetAtoms():
        a.SetNoImplicit(False)
    m = _gentle_sanitize(m)
    if m is not None and not _smiles_ok(m):
        return None, "Bonds could not be worked out from the coordinates."
    note = ("This file has no hydrogens, so only single bonds could be worked out from the atom positions. "
            "Give the correct SMILES as a template to restore double bonds and aromatic rings.")
    return m, note


def _fix_pdbqt(text: str) -> str:
    """Turn PDBQT into plain PDB: keep atom lines and write a proper element column."""
    ad_to_element = {"A": "C", "OA": "O", "NA": "N", "NS": "N", "SA": "S", "HD": "H", "HS": "H",
                     "OS": "O", "CG0": "C", "CG1": "C", "G0": "C", "G1": "C", "Z": "C"}
    out = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            ad = line[77:79].strip() if len(line) > 77 else ""
            elem = ad_to_element.get(ad, ad[:2].capitalize() if ad else "")
            if not elem:
                elem = re.sub(r"[^A-Za-z]", "", line[12:16])[:1]
            out.append(line[:66].ljust(76) + elem.rjust(2))
        elif line.startswith(("CONECT", "MODEL", "ENDMDL", "END")):
            out.append(line)
    return "\n".join(out) + "\n"


def _first_model(pdb_text: str) -> str:
    if "MODEL" not in pdb_text:
        return pdb_text
    m = re.search(r"^MODEL.*?$(.*?)^ENDMDL", pdb_text, flags=re.S | re.M)
    return m.group(1) if m else pdb_text


def _read_pdb(text: str, stem: str) -> list[LoadedMolecule]:
    text = _first_model(text)
    has_conect = bool(re.search(r"^CONECT", text, flags=re.M))
    raw = Chem.MolFromPDBBlock(text, removeHs=False, sanitize=False, proximityBonding=not has_conect)
    if raw is None or raw.GetNumAtoms() == 0:
        return []
    if has_conect:
        mol = _gentle_sanitize(Chem.Mol(raw))
        if mol is not None and _smiles_ok(mol):
            if any(b.GetBondTypeAsDouble() > 1 for b in mol.GetBonds()):
                return [LoadedMolecule(stem, mol, has_3d=True)]
            # Only single bonds in CONECT: see whether the hydrogens imply double bonds
            guess, note = _bonds_from_coordinates(raw, charge=Chem.GetFormalCharge(mol))
            if guess is not None and any(b.GetBondTypeAsDouble() > 1 for b in guess.GetBonds()):
                return [LoadedMolecule(stem, guess, has_3d=True, note=note)]
            return [LoadedMolecule(stem, mol, has_3d=True)]
    mol, note = _bonds_from_coordinates(raw)
    return [LoadedMolecule(stem, mol, has_3d=True, note=note)] if mol is not None else []


def _read_xyz(text: str, stem: str) -> list[LoadedMolecule]:
    raw = Chem.MolFromXYZBlock(text)
    if raw is None:
        return []
    lines = text.splitlines()
    title = lines[1].strip() if len(lines) > 1 and lines[1].strip() else stem
    mol, note = _bonds_from_coordinates(raw)
    return [LoadedMolecule(title, mol, has_3d=True, note=note)] if mol is not None else []


def _read_mol2(text: str, stem: str) -> list[LoadedMolecule]:
    blocks = [b for b in re.split(r"(?=@<TRIPOS>MOLECULE)", text) if "@<TRIPOS>ATOM" in b]
    out = []
    for i, block in enumerate(blocks, start=1):
        mol = Chem.MolFromMol2Block(block, removeHs=False)
        if mol is None:
            mol = _gentle_sanitize(Chem.MolFromMol2Block(block, removeHs=False, sanitize=False))
        if mol is None:
            continue
        lines = block.splitlines()
        title = lines[1].strip() if len(lines) > 1 and lines[1].strip() else f"{stem}_{i}"
        out.append(LoadedMolecule(title, mol, has_3d=_is_3d(mol)))
    return out


def _read_sdf(text: str, stem: str) -> list[LoadedMolecule]:
    if "$$$$" not in text:
        text = text.rstrip() + "\n$$$$\n"
    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, removeHs=False, sanitize=False)
    out = []
    for i, raw in enumerate(supplier, start=1):
        mol = _gentle_sanitize(raw)
        if mol is None:
            continue
        out.append(LoadedMolecule(_name_of(mol, f"{stem}_{i}"), mol, has_3d=_is_3d(mol)))
    return out


def _read_smiles_file(text: str, stem: str) -> list[LoadedMolecule]:
    out = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",", 1)]
        else:
            parts = line.split(None, 1)
        smi, name = parts[0], (parts[1] if len(parts) > 1 else f"{stem}_{i}")
        mol = from_smiles(smi)
        if mol is None and len(parts) > 1:  # maybe written as "name,SMILES"
            smi, name = parts[1], parts[0]
            mol = from_smiles(smi)
        if mol is not None:
            out.append(LoadedMolecule(name, mol))
    return out


def read_file(data: bytes, filename: str) -> list[LoadedMolecule]:
    """Read every molecule in an uploaded file. Returns an empty list if nothing could be read."""
    found = _read_any(data, filename)
    good = []
    for item in found:
        if item.mol is not None and _smiles_ok(item.mol):
            item.mol = _tidy(item.mol)
            good.append(item)
    return good


def _read_any(data: bytes, filename: str) -> list[LoadedMolecule]:
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    stem, _, ext = filename.rpartition(".")
    stem, ext = (stem or filename), ext.lower()
    if ext in ("sdf", "sd", "mol"):
        return _read_sdf(text, stem)
    if ext == "mol2":
        return _read_mol2(text, stem)
    if ext in ("pdb", "ent"):
        return _read_pdb(text, stem)
    if ext == "pdbqt":
        return _read_pdb(_fix_pdbqt(text), stem)
    if ext == "xyz":
        return _read_xyz(text, stem)
    if ext in ("smi", "smiles", "txt"):
        return _read_smiles_file(text, stem)
    # Unknown extension: try the formats one by one
    for reader in (_read_sdf, _read_mol2, _read_pdb, _read_xyz, _read_smiles_file):
        found = reader(text, stem)
        if found:
            return found
    return []


def apply_template(mol: Chem.Mol, template_smiles: str) -> tuple[Chem.Mol | None, str]:
    """Give a coordinate-only structure the bond orders of a SMILES template."""
    template = from_smiles(template_smiles)
    if template is None:
        return None, "The template SMILES could not be read."
    try:
        heavy = Chem.RemoveHs(mol, sanitize=False)
        fixed = AllChem.AssignBondOrdersFromTemplate(template, heavy)
        Chem.SanitizeMol(fixed)
        fixed = Chem.AddHs(fixed, addCoords=True)
        return fixed, "Bond orders were copied from your template SMILES."
    except Exception:  # noqa: BLE001
        return None, "The template does not match the atoms in the file. Check it is the same molecule."


# ---------------------------------------------------------------- identifiers
def canonical_smiles(mol: Chem.Mol, isomeric: bool = True) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=isomeric)


def identifiers(mol: Chem.Mol) -> dict:
    heavy = Chem.RemoveHs(mol)
    out = {
        "Canonical SMILES": Chem.MolToSmiles(heavy, isomericSmiles=False),
        "Isomeric SMILES": Chem.MolToSmiles(heavy),
        "Formula": rdMolDescriptors.CalcMolFormula(heavy),
        "InChI": "",
        "InChIKey": "",
    }
    try:
        out["InChI"] = Chem.MolToInchi(heavy) or ""
        out["InChIKey"] = Chem.MolToInchiKey(heavy) or ""
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------- geometry
def _tag(mol: Chem.Mol, name: str) -> Chem.Mol:
    mol.SetProp("_Name", name or "molecule")
    return mol


def make_2d(mol: Chem.Mol, hydrogens: bool = False, name: str = "") -> Chem.Mol:
    m = Chem.RemoveHs(mol)
    if hydrogens:
        m = Chem.AddHs(m)
    m = Chem.Mol(m)
    m.RemoveAllConformers()
    rdDepictor.Compute2DCoords(m)
    _scale_bonds(m, 1.5)  # usual 2D bond length, so other programs read the file correctly
    return _tag(m, name)


def _scale_bonds(mol: Chem.Mol, target: float) -> None:
    if mol.GetNumBonds() == 0:
        return
    conf = mol.GetConformer()
    lengths = [(conf.GetAtomPosition(b.GetBeginAtomIdx()) - conf.GetAtomPosition(b.GetEndAtomIdx())).Length()
               for b in mol.GetBonds()]
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return
    f = target / mean
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x * f, p.y * f, p.z * f))


def make_3d(mol: Chem.Mol, hydrogens: bool = True, name: str = "", seed: int = 0xF00D) -> tuple[Chem.Mol | None, str]:
    """Embed with ETKDGv3 and clean up with MMFF94 (UFF if MMFF has no parameters)."""
    heavy = Chem.RemoveHs(mol)
    if heavy.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS_3D:
        return None, f"3D generation is limited to {MAX_HEAVY_ATOMS_3D} heavy atoms."
    m = Chem.AddHs(heavy)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(m, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            return None, "RDKit could not build 3D coordinates for this structure."
    method = "ETKDGv3"
    try:
        if AllChem.MMFFHasAllMoleculeParams(m):
            AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
            method += " + MMFF94"
        elif AllChem.UFFHasAllMoleculeParams(m):
            AllChem.UFFOptimizeMolecule(m, maxIters=2000)
            method += " + UFF"
    except Exception:  # noqa: BLE001
        pass
    Chem.AssignStereochemistryFrom3D(m)
    if not hydrogens:
        m = Chem.RemoveHs(m)
    return _tag(m, name), method


def keep_file_3d(mol: Chem.Mol, hydrogens: bool = True, name: str = "") -> Chem.Mol:
    """Use coordinates that came with an uploaded file."""
    m = Chem.Mol(mol)
    if hydrogens:
        m = Chem.AddHs(m, addCoords=True)
    else:
        m = Chem.RemoveHs(m)
    return _tag(m, name)


# ---------------------------------------------------------------- writing
def _atom_names(mol: Chem.Mol) -> list[str]:
    counts: dict[str, int] = {}
    names = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
        names.append(f"{sym}{counts[sym]}"[:4])
    return names


def to_sdf(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    m.SetProp("SMILES", canonical_smiles(m))
    sio_block = Chem.MolToMolBlock(m)
    props = f">  <SMILES>\n{m.GetProp('SMILES')}\n\n"
    return sio_block + props + "$$$$\n"


def to_mol(mol: Chem.Mol) -> str:
    return Chem.MolToMolBlock(mol)


def to_pdb(mol: Chem.Mol, resname: str = "LIG") -> str:
    m = Chem.Mol(mol)
    for atom, nm in zip(m.GetAtoms(), _atom_names(m)):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f" {nm:<3}" if len(nm) < 4 else nm)
        info.SetResidueName(resname[:3].upper())
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetIsHeteroAtom(True)
        info.SetOccupancy(1.0)
        info.SetTempFactor(0.0)
        atom.SetMonomerInfo(info)
    return "AUTHOR    MolCov, The Insilico Lab\n" + Chem.MolToPDBBlock(m)


def xyz_block(mol: Chem.Mol) -> str:
    return Chem.MolToXYZBlock(mol)


def _sybyl_aromatic(m: Chem.Mol) -> tuple[set[int], set[int]]:
    """Atoms and bonds that SYBYL treats as aromatic.

    RDKit calls rings such as the pyrimidinedione of caffeine aromatic; SYBYL does not,
    because their ring atoms carry exocyclic C=O. Only rings where every bond is aromatic
    and no ring atom has a double bond leaving the ring are kept (m must be kekulised).
    """
    atoms: set[int] = set()
    bonds: set[int] = set()
    ri = m.GetRingInfo()
    for ring_bonds, ring_atoms in zip(ri.BondRings(), ri.AtomRings()):
        if not all(m.GetBondWithIdx(b).GetIsAromatic() for b in ring_bonds):
            continue
        inside = set(ring_atoms)
        exocyclic = any(
            b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtomIdx(a) not in inside
            for a in ring_atoms for b in m.GetAtomWithIdx(a).GetBonds()
        )
        if not exocyclic:
            atoms |= inside
            bonds |= set(ring_bonds)
    return atoms, bonds


def _double_bonded_o(atom: Chem.Atom) -> bool:
    return any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() in ("O", "S")
               for b in atom.GetBonds())


def _is_amide_n(atom: Chem.Atom) -> bool:
    if atom.GetSymbol() != "N":
        return False
    return any(nb.GetSymbol() == "C" and _double_bonded_o(nb) for nb in atom.GetNeighbors())


def _sybyl_type(atom: Chem.Atom, arom: bool) -> str:
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization()
    H = Chem.HybridizationType
    double = [b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    triple = [b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]
    if sym == "C":
        if arom:
            return "C.ar"
        if triple or hyb == H.SP:
            return "C.1"
        if (atom.GetFormalCharge() == 1
                and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "N") == 3):
            return "C.cat"
        if double or hyb == H.SP2:
            return "C.2"
        return "C.3"
    if sym == "N":
        if arom:
            # pyrrole-type N (three connections) reads back reliably as N.pl3
            return "N.pl3" if atom.GetDegree() + atom.GetTotalNumHs() == 3 else "N.ar"
        if atom.GetFormalCharge() == 1 and atom.GetDegree() + atom.GetTotalNumHs() == 4:
            return "N.4"
        if triple:
            return "N.1"
        if double and any(b.GetOtherAtom(atom).GetSymbol() == "O" for b in double) and atom.GetFormalCharge() == 1:
            return "N.pl3"  # nitro
        if double:
            return "N.2"
        if _is_amide_n(atom):
            return "N.am"
        if hyb == H.SP2 or any(nb.GetIsAromatic() for nb in atom.GetNeighbors()):
            return "N.pl3"
        return "N.3"
    if sym == "O":
        nbs = atom.GetNeighbors()
        if len(nbs) == 1 and nbs[0].GetSymbol() in ("C", "P"):
            centre = nbs[0]
            terminal_o = [n for n in centre.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1]
            if centre.GetSymbol() == "C" and len(terminal_o) == 2 and any(o.GetFormalCharge() == -1 for o in terminal_o):
                return "O.co2"
        if double or arom:
            return "O.2"
        return "O.3"
    if sym == "S":
        o_terminal = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1)
        if o_terminal >= 2:
            return "S.O2"
        if o_terminal == 1:
            return "S.O"
        if double or arom:
            return "S.2"
        return "S.3"
    if sym == "P":
        return "P.3"
    if sym == "H":
        return "H"
    return sym


def _sybyl_bond(bond: Chem.Bond, arom: bool) -> str:
    if arom:
        return "ar"
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        for n, c in ((a, b), (b, a)):
            if n.GetSymbol() == "N" and c.GetSymbol() == "C" and _double_bonded_o(c) and not n.GetIsAromatic():
                return "am"
        return "1"
    return {Chem.BondType.DOUBLE: "2", Chem.BondType.TRIPLE: "3"}.get(bt, "1")


def to_mol2(mol: Chem.Mol, resname: str = "LIG") -> str:
    """Tripos MOL2 with SYBYL atom types and Gasteiger partial charges.

    MOL2 files are expected to list every hydrogen, so missing ones are added.
    """
    m = Chem.Mol(mol)
    if any(a.GetTotalNumHs() for a in m.GetAtoms()):
        m = Chem.AddHs(m, addCoords=True)
    try:
        AllChem.ComputeGasteigerCharges(m)
        charges = [float(a.GetProp("_GasteigerCharge")) for a in m.GetAtoms()]
        charges = [0.0 if c != c else c for c in charges]  # NaN to zero
    except Exception:  # noqa: BLE001
        charges = [0.0] * m.GetNumAtoms()
    Chem.Kekulize(m, clearAromaticFlags=False)
    arom_atoms, arom_bonds = _sybyl_aromatic(m)
    conf = m.GetConformer()
    names = _atom_names(m)
    title = m.GetProp("_Name") if m.HasProp("_Name") else "molecule"
    res = resname[:3].upper()
    lines = [
        "@<TRIPOS>MOLECULE",
        title,
        f"{m.GetNumAtoms():>5d} {m.GetNumBonds():>5d}     1     0     0",
        "SMALL",
        "GASTEIGER",
        "",
        "@<TRIPOS>ATOM",
    ]
    for i, atom in enumerate(m.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(
            f"{i + 1:>7d} {names[i]:<8s}{p.x:>10.4f}{p.y:>10.4f}{p.z:>10.4f} "
            f"{_sybyl_type(atom, i in arom_atoms):<6s}{1:>4d}  {res:<6s}{charges[i]:>10.4f}"
        )
    lines.append("@<TRIPOS>BOND")
    for j, bond in enumerate(m.GetBonds()):
        lines.append(f"{j + 1:>6d}{bond.GetBeginAtomIdx() + 1:>6d}{bond.GetEndAtomIdx() + 1:>6d} "
                     f"{_sybyl_bond(bond, bond.GetIdx() in arom_bonds)}")
    lines += ["@<TRIPOS>SUBSTRUCTURE", f"{1:>6d} {res:<8s}{1:>6d} TEMP              0 ****  ****    0 ROOT", ""]
    return "\n".join(lines)


WRITERS = {
    "SDF": (to_sdf, "sdf", "chemical/x-mdl-sdfile"),
    "PDB": (to_pdb, "pdb", "chemical/x-pdb"),
    "MOL2": (to_mol2, "mol2", "chemical/x-mol2"),
    "MOL": (to_mol, "mol", "chemical/x-mdl-molfile"),
    "XYZ": (xyz_block, "xyz", "chemical/x-xyz"),
}


def write(mol: Chem.Mol, fmt: str) -> tuple[str, str, str]:
    """Return (text, file extension, MIME type)."""
    fn, ext, mime = WRITERS[fmt]
    return fn(mol), ext, mime


# ---------------------------------------------------------------- pictures
def svg_2d(mol: Chem.Mol, width: int = 520, height: int = 400) -> str:
    """2D drawing with a transparent background, so the viewer can colour it."""
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0 or _is_3d(m):
        m = make_2d(m, hydrogens=any(a.GetAtomicNum() == 1 for a in m.GetAtoms()))
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.addStereoAnnotation = True
    opts.padding = 0.08
    opts.bondLineWidth = 2
    opts.fixedFontSize = -1
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, m)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg"):]


# ---------------------------------------------------------------- properties
@dataclass
class Properties:
    mw: float = 0.0
    exact_mass: float = 0.0
    logp: float = 0.0
    mr: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotb: int = 0
    heavy_atoms: int = 0
    total_atoms: int = 0
    rings: int = 0
    aromatic_rings: int = 0
    fsp3: float = 0.0
    charge: int = 0
    stereocentres: int = 0
    lipinski_violations: list[str] = field(default_factory=list)
    veber_violations: list[str] = field(default_factory=list)
    multiple_fragments: bool = False

    @property
    def lipinski_pass(self) -> bool:
        return len(self.lipinski_violations) <= 1  # one violation is usually allowed

    @property
    def veber_pass(self) -> bool:
        return not self.veber_violations


def properties(mol: Chem.Mol) -> Properties:
    m = Chem.RemoveHs(mol)
    p = Properties()
    p.mw = Descriptors.MolWt(m)
    p.exact_mass = Descriptors.ExactMolWt(m)
    p.logp = Crippen.MolLogP(m)
    p.mr = Crippen.MolMR(m)
    p.tpsa = rdMolDescriptors.CalcTPSA(m)
    p.hbd = rdMolDescriptors.CalcNumLipinskiHBD(m)  # NH + OH, Lipinski's definition
    p.hba = rdMolDescriptors.CalcNumLipinskiHBA(m)  # N + O, Lipinski's definition
    p.rotb = rdMolDescriptors.CalcNumRotatableBonds(m)
    p.heavy_atoms = m.GetNumHeavyAtoms()
    p.total_atoms = Chem.AddHs(m).GetNumAtoms()
    p.rings = rdMolDescriptors.CalcNumRings(m)
    p.aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(m)
    p.fsp3 = rdMolDescriptors.CalcFractionCSP3(m)
    p.charge = Chem.GetFormalCharge(m)
    p.stereocentres = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
    p.multiple_fragments = len(Chem.GetMolFrags(m)) > 1
    for label, (attr, limit) in LIPINSKI_LIMITS.items():
        if getattr(p, attr) > limit:
            p.lipinski_violations.append(label)
    for label, (attr, limit) in VEBER_LIMITS.items():
        if getattr(p, attr) > limit:
            p.veber_violations.append(label)
    return p


def properties_row(name: str, ids: dict, p: Properties) -> dict:
    return {
        "Name": name,
        "Canonical SMILES": ids["Canonical SMILES"],
        "Isomeric SMILES": ids["Isomeric SMILES"],
        "InChIKey": ids["InChIKey"],
        "Formula": ids["Formula"],
        "MW (g/mol)": round(p.mw, 2),
        "Exact mass": round(p.exact_mass, 4),
        "LogP (Crippen)": round(p.logp, 2),
        "Molar refractivity": round(p.mr, 2),
        "TPSA (Å²)": round(p.tpsa, 2),
        "HBD": p.hbd,
        "HBA": p.hba,
        "Rotatable bonds": p.rotb,
        "Heavy atoms": p.heavy_atoms,
        "Rings": p.rings,
        "Aromatic rings": p.aromatic_rings,
        "Fraction sp3 C": round(p.fsp3, 2),
        "Formal charge": p.charge,
        "Stereocentres": p.stereocentres,
        "Lipinski violations": len(p.lipinski_violations),
        "Lipinski": "Pass" if p.lipinski_pass else "Fail",
        "Veber": "Pass" if p.veber_pass else "Fail",
    }

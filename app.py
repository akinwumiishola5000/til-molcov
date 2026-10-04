"""MolCov: draw, convert and download molecules. Part of The Insilico Lab CADD Toolkit.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import molcov_chem as mc

try:
    from streamlit_ketcher import st_ketcher
except ImportError:  # the app still works without the sketcher
    st_ketcher = None

APP_NAME = "MolCov"
VERSION = "1.0"
SITE_URL = "https://www.theinsilicolab.org/home"
TOOLKIT_URL = "https://www.theinsilicolab.org/cadd-toolkit"
WORKSPACE_URL = "https://theinsilicolab-workspace.streamlit.app"
HOLOSIFT_URL = "https://theinsilicolab-holosift.streamlit.app"
BASE_DIR = Path(__file__).parent
LOGO_FULL = BASE_DIR / "logo_full.png"
LOGO_ICON = BASE_DIR / "logo_icon.png"
VIEWER_HTML = BASE_DIR / "molcov_viewer.html"
HERO_HTML = BASE_DIR / "molcov_hero.html"

# Brand colours taken from The Insilico Lab logo, the same as the Workspace and HoloSift
NAVY, GREEN, BLUE = "#00133F", "#3F824A", "#1E9FC0"

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Ibuprofen (S)": "C[C@H](C(=O)O)c1ccc(CC(C)C)cc1",
    "Artemisinin": "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@]32OO4",
    "Chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "Osimertinib": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
    "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "Penicillin G": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
}
HERO_SET = ["Artemisinin", "Caffeine", "Aspirin", "Chloroquine", "Imatinib", "Paracetamol"]

st.set_page_config(
    page_title="MolCov | The Insilico Lab",
    page_icon=str(LOGO_ICON) if LOGO_ICON.exists() else ":material/hub:",
    layout="wide",
)
if LOGO_ICON.exists():
    st.logo(str(LOGO_ICON), size="large", link=SITE_URL, icon_image=str(LOGO_ICON))

st.markdown(f"""
<style>
h1, h2, h3 {{ color: {NAVY}; }}
h1 {{ font-weight: 800; letter-spacing: -0.01em; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; }}
[data-testid="stSidebar"] {{ border-right: 3px solid {GREEN}; }}
.til-rule {{ height: 5px; width: 100%; margin: 0.2rem 0 1.2rem;
  background: linear-gradient(90deg, {NAVY} 0 40%, {BLUE} 40% 70%, {GREEN} 70% 100%); border-radius: 2px; }}
.til-hero-title {{ font-size: clamp(2.8rem, 5vw, 4rem) !important; font-weight: 800; color: {NAVY}; letter-spacing: -0.02em;
  line-height: 1; margin: 0.4rem 0 0.4rem; }}
.til-hero-sub {{ color: {GREEN}; font-weight: 700; font-size: 1.2rem; margin: 0 0 0.8rem; }}
.til-lead {{ font-size: 1.1rem; line-height: 1.55; max-width: 34em; }}
.til-id {{ font-size: 0.85rem; color: #5F6E86; margin: 0.6rem 0 0.1rem; font-weight: 600; }}
iframe {{ border: 0; }}
</style>
""", unsafe_allow_html=True)


def brand_rule():
    st.markdown('<div class="til-rule"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- cached helpers
@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False, max_entries=300)
def prepare(molblock: str, name: str, file_has_3d: bool) -> dict:
    """Everything the results panel needs, worked out once per molecule."""
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    ids = mc.identifiers(mol)
    props = mc.properties(mol)
    gen3d, method = mc.make_3d(mol, hydrogens=True, name=name)
    out = {
        "ids": ids,
        "row": mc.properties_row(name, ids, props),
        "props": props,
        "svg": mc.svg_2d(mc.make_2d(mol, hydrogens=False)),
        "gen3d": Chem.MolToMolBlock(gen3d) if gen3d is not None else "",
        "method": method if gen3d is not None else "",
        "note3d": "" if gen3d is not None else method,
        "file3d": Chem.MolToMolBlock(mc.keep_file_3d(mol, True, name)) if file_has_3d else "",
        "unassigned_stereo": any(c[1] == "?" for c in Chem.FindMolChiralCenters(
            Chem.RemoveHs(mol), includeUnassigned=True, useLegacyImplementation=False)),
    }
    return out


@st.cache_data(show_spinner=False)
def hero_data() -> list[dict]:
    items = []
    for name in HERO_SET:
        mol = mc.from_smiles(EXAMPLES[name])
        mol3d, _ = mc.make_3d(mol, hydrogens=True, name=name)
        items.append({"name": name, "smiles": EXAMPLES[name],
                      "svg": mc.svg_2d(mc.make_2d(mol), 520, 400),
                      "sdf": Chem.MolToMolBlock(mol3d)})
    return items


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def pubchem_smiles(name: str) -> tuple[str | None, str]:
    """Look up a compound name on PubChem. Returns (smiles, message)."""
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"
    for prop in ("SMILES", "IsomericSMILES"):
        try:
            r = requests.get(base.format(quote(name), prop), timeout=10)
        except requests.RequestException:
            return None, "PubChem could not be reached. Check your connection or type the SMILES instead."
        if r.status_code == 404:
            return None, f"PubChem has no compound called “{name}”. Check the spelling or try another name."
        if r.ok:
            props = r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            smi = props.get("SMILES") or props.get("IsomericSMILES")
            if smi:
                return smi, f"Found on PubChem (CID {props.get('CID', '?')})."
    return None, "PubChem returned an unexpected answer. Type the SMILES instead."


@st.cache_data(show_spinner=False, max_entries=50)
def read_upload(data: bytes, filename: str) -> list[dict]:
    found = mc.read_file(data, filename)
    return [{"name": m.name, "molblock": Chem.MolToMolBlock(m.mol), "has_3d": m.has_3d, "note": m.note,
             "formula": rdMolDescriptors.CalcMolFormula(Chem.RemoveHs(m.mol))} for m in found]


def embed_html(html: str, height: int):
    """Show a self-contained HTML page in an iframe (st.iframe on new Streamlit, components.html before)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)  # fixed height: the 3D canvas needs a stable size
    else:  # Streamlit older than 1.52
        components.html(html, height=height, scrolling=False)


def slug(text: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (text or "molecule").strip())
    return keep.strip("_")[:60] or "molecule"


def viewer(data: dict, name: str, use_file_3d: bool, height: int = 440):
    sdf3d = data["file3d"] if use_file_3d and data["file3d"] else data["gen3d"]
    payload = {"name": name, "svg": data["svg"], "sdf3d": sdf3d, "start": "3d" if sdf3d else "2d",
               "height": height, "note3d": data["note3d"],
               "method": "Coordinates from your file" if use_file_3d and data["file3d"] else
               (f"3D built with {data['method']}" if data["method"] else "")}
    html = read_text(str(VIEWER_HTML)).replace(
        "/*MOLCOV_DATA*/", "window.MOLCOV = " + json.dumps(payload).replace("</", "<\\/") + ";", 1)
    embed_html(html, height + 175)


def rule_table(p: mc.Properties, limits: dict) -> pd.DataFrame:
    rows = []
    for label, (attr, limit) in limits.items():
        value = getattr(p, attr)
        unit = " Å²" if attr == "tpsa" else (" g/mol" if attr == "mw" else "")
        shown = f"{value:.2f}" if isinstance(value, float) else str(value)
        rows.append({"Rule": f"{label} ≤ {limit:g}{unit}", "This molecule": shown,
                     "Result": "Pass" if value <= limit else "Fail"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- results panel
def show_results(molblock: str, name: str, file_has_3d: bool, key: str):
    """Structure, identifiers, viewer, downloads and properties for one molecule."""
    with st.spinner("Building 2D and 3D structures…"):
        data = prepare(molblock, name, file_has_3d)
    ids, p = data["ids"], data["props"]

    st.subheader(name or "Your molecule")
    use_file_3d = False
    if data["file3d"]:
        use_file_3d = st.toggle("Use the 3D coordinates from my file", value=True, key=f"{key}_file3d",
                                help="Turn off to rebuild the 3D structure from scratch with RDKit.")

    left, right = st.columns([4, 8], gap="large")
    with left:
        st.markdown('<p class="til-id">Canonical SMILES</p>', unsafe_allow_html=True)
        st.code(ids["Canonical SMILES"], language=None, wrap_lines=True)
        if ids["Isomeric SMILES"] != ids["Canonical SMILES"]:
            st.markdown('<p class="til-id">Isomeric SMILES (keeps stereochemistry)</p>', unsafe_allow_html=True)
            st.code(ids["Isomeric SMILES"], language=None, wrap_lines=True)
        st.markdown('<p class="til-id">Formula</p>', unsafe_allow_html=True)
        st.code(ids["Formula"], language=None)
        if ids["InChIKey"]:
            st.markdown('<p class="til-id">InChIKey</p>', unsafe_allow_html=True)
            st.code(ids["InChIKey"], language=None)
            with st.expander("InChI"):
                st.code(ids["InChI"], language=None, wrap_lines=True)
        if p.multiple_fragments:
            st.warning("This structure has more than one fragment, often a salt or solvent. "
                       "Remove the counter-ion if you want to describe the drug itself.")
        if data["unassigned_stereo"]:
            st.info("Some stereocentres are not defined. The 3D structure picks one arrangement for each, "
                    "so define them in the SMILES or drawing if they matter.")
    with right:
        viewer(data, name, use_file_3d)

    # ---- downloads
    with st.container(border=True):
        st.markdown("**Download the structure**")
        c1, c2, c3 = st.columns([3, 3, 2], gap="medium")
        geometry = c1.segmented_control("Coordinates", ["3D", "2D"], default="3D", key=f"{key}_geo",
                                        help="3D for docking and simulations, 2D for drawings and databases.")
        geometry = geometry or "3D"
        formats = ["SDF", "PDB", "MOL2", "MOL"] + (["XYZ"] if geometry == "3D" else [])
        fmt = c2.segmented_control("Format", formats, default="SDF", key=f"{key}_fmt") or "SDF"
        hydrogens = c3.checkbox("Include hydrogens", value=geometry == "3D", key=f"{key}_h",
                                disabled=fmt in ("MOL2", "XYZ"),
                                help="MOL2 and XYZ files always include every hydrogen.")
        if fmt in ("MOL2", "XYZ"):
            hydrogens = True

        base = Chem.MolFromMolBlock(molblock, removeHs=False)
        if geometry == "2D":
            out_mol = mc.make_2d(base, hydrogens=hydrogens, name=name)
        else:
            block = data["file3d"] if use_file_3d and data["file3d"] else data["gen3d"]
            out_mol = Chem.MolFromMolBlock(block, removeHs=False) if block else None
            if out_mol is not None:
                out_mol = out_mol if hydrogens else Chem.RemoveHs(out_mol)
                out_mol.SetProp("_Name", name or "molecule")
        if out_mol is None:
            st.error(data["note3d"] or "No 3D structure is available. Download the 2D version instead.")
        else:
            text, ext, mime = mc.write(out_mol, fmt)
            d1, d2, d3 = st.columns([2, 2, 3])
            d1.download_button(f"Download {fmt} ({geometry})", text.encode("utf-8"),
                               file_name=f"{slug(name)}_{geometry}.{ext}", mime=mime, type="primary",
                               icon=":material/download:", key=f"{key}_dl")
            d2.download_button("Download SMILES (.smi)", f"{ids['Isomeric SMILES']}\t{name}\n".encode("utf-8"),
                               file_name=f"{slug(name)}.smi", mime="chemical/x-daylight-smiles",
                               key=f"{key}_smi")
            if geometry == "2D" and fmt in ("PDB", "MOL2"):
                d3.caption("2D PDB and MOL2 files are flat and cannot record stereochemistry. "
                           "Use SDF or 3D if it matters.")
            with st.expander(f"Preview the {fmt} file"):
                st.code(text[:6000] + ("\n…" if len(text) > 6000 else ""), language=None)

    # ---- properties
    st.subheader("Molecular properties")
    m = st.columns(4)
    m[0].metric("Mol. weight (g/mol)", f"{p.mw:.2f}")
    m[1].metric("LogP (Crippen)", f"{p.logp:.2f}")
    m[2].metric("TPSA (Å²)", f"{p.tpsa:.1f}")
    m[3].metric("Exact mass", f"{p.exact_mass:.4f}")
    m = st.columns(4)
    m[0].metric("H-bond donors", p.hbd)
    m[1].metric("H-bond acceptors", p.hba)
    m[2].metric("Rotatable bonds", p.rotb)
    m[3].metric("Heavy atoms", p.heavy_atoms)
    m = st.columns(4)
    m[0].metric("Aromatic rings", p.aromatic_rings)
    m[1].metric("Fraction sp3 C", f"{p.fsp3:.2f}")
    m[2].metric("Formal charge", p.charge)
    m[3].metric("Stereocentres", p.stereocentres)

    lcol, vcol = st.columns(2, gap="large")
    with lcol:
        st.markdown("**Lipinski's rule of five**")
        n = len(p.lipinski_violations)
        if n == 0:
            st.success("Passes all four rules.")
        elif n == 1:
            st.info(f"One violation ({p.lipinski_violations[0]}). One is usually allowed, so this still counts as drug-like.")
        else:
            st.error(f"{n} violations: {', '.join(p.lipinski_violations)}. Oral absorption may be poor, "
                     "though some approved drugs sit outside these rules.")
        st.dataframe(rule_table(p, mc.LIPINSKI_LIMITS), hide_index=True, width="stretch")
    with vcol:
        st.markdown("**Veber's rules for oral bioavailability**")
        if p.veber_pass:
            st.success("Passes both rules.")
        else:
            st.error(f"Fails: {', '.join(p.veber_violations)}.")
        st.dataframe(rule_table(p, mc.VEBER_LIMITS), hide_index=True, width="stretch")

    st.download_button("Download properties (CSV)", pd.DataFrame([data["row"]]).to_csv(index=False).encode("utf-8"),
                       file_name=f"{slug(name)}_properties.csv", mime="text/csv", key=f"{key}_csv",
                       icon=":material/table:")
    st.caption("Properties are calculated with RDKit for the structure without its counter-ions removed. "
               f"For many molecules at once, use the [Molecular Learning Workspace]({WORKSPACE_URL}).")


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("sk_smiles", EXAMPLES["Aspirin"])
ss.setdefault("sk_name", "Aspirin")
ss.setdefault("sk_nonce", 0)


def set_molecule(smiles: str, name: str, from_sketch: bool = False, rerun: bool = True):
    """Make a molecule current on the sketch page. The sketcher is reloaded unless it was the source."""
    ss.sk_smiles, ss.sk_name = smiles, name
    ss.sk_sync_box = True
    if not from_sketch:
        ss.sk_nonce += 1
    if rerun:
        st.rerun()


def _pick_example():
    choice = ss.get("sk_example")
    if choice:
        set_molecule(EXAMPLES[choice], choice, rerun=False)


# ---------------------------------------------------------------- pages
def home():
    text_col, stage_col = st.columns([5, 6], gap="large", vertical_alignment="center")
    with text_col:
        if LOGO_FULL.exists():
            st.image(str(LOGO_FULL), width=230)
        st.markdown('<div class="til-hero-title">MolCov</div>', unsafe_allow_html=True)
        st.markdown('<p class="til-hero-sub">Draw it. Convert it. Take it into your next tool.</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="til-lead">Sketch a molecule or paste its SMILES, see it in 2D and 3D, colour it, '
            'rotate it and save a picture. Download it as SDF, PDB or MOL2, or upload a structure file '
            'to get its SMILES and properties.</p>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.page_link(pages["convert"], label="Draw or type a molecule", icon=":material/draw:")
        b2.page_link(pages["files"], label="Convert a file", icon=":material/upload_file:")
        st.caption(f"Version {VERSION}. Free for members of The Insilico Lab.")
    with stage_col:
        payload = json.dumps(hero_data()).replace("</", "<\\/")
        embed_html(read_text(str(HERO_HTML)).replace(
            "/*MOLCOV_HERO*/", "window.MOLCOV_HERO = " + payload + ";", 1), 470)
    brand_rule()

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("#### :material/draw: Draw or type")
        st.write("Draw in the sketcher, paste a SMILES, search PubChem by name, or pick an example. "
                 "MolCov gives you the canonical SMILES, InChIKey and formula.")
        st.page_link(pages["convert"], label="Open the sketcher", icon=":material/arrow_forward:")
    with c2:
        st.markdown("#### :material/upload_file: Convert a file")
        st.write("Upload SDF, MOL, MOL2, PDB, PDBQT, XYZ or SMI. Pick a molecule from multi-molecule files "
                 "and docking outputs, then convert it.")
        st.page_link(pages["files"], label="Upload a structure", icon=":material/arrow_forward:")
    with c3:
        st.markdown("#### :material/menu_book: Know your formats")
        st.write("What each file format stores, when to use 2D or 3D, and how MolCov builds 3D structures.")
        st.page_link(pages["learn"], label="Read the guide", icon=":material/arrow_forward:")
    st.divider()
    st.write(f"Part of the [CADD Toolkit]({TOOLKIT_URL}) from [The Insilico Lab]({SITE_URL}). "
             f"Also try the [Molecular Learning Workspace]({WORKSPACE_URL}) and [HoloSift]({HOLOSIFT_URL}).")


def convert():
    st.title("Draw and convert")
    brand_rule()

    # A teaching link such as ?smiles=CCO opens with that molecule
    qs = str(st.query_params.get("smiles", "")).strip()
    if qs and ss.get("sk_query_used") != qs and mc.from_smiles(qs) is not None:
        ss.sk_query_used = qs
        set_molecule(qs, str(st.query_params.get("name", "")).strip() or "Shared molecule")
    if ss.pop("sk_sync_box", False) or "sk_box" not in ss:
        ss.sk_box = ss.sk_smiles

    with st.container(border=True):
        t_draw, t_smiles, t_name, t_example = st.tabs(
            [":material/draw: Draw", ":material/text_fields: Type SMILES", ":material/search: Search by name",
             ":material/science: Examples"])
        with t_draw:
            if st_ketcher is None:
                st.warning("The sketcher is not installed. Add streamlit-ketcher to requirements.txt.")
            else:
                st.caption("Draw your molecule, then press **Apply** under the editor to send it to MolCov. "
                           "Use the templates on the bottom toolbar for rings.")
                drawn = st_ketcher(ss.sk_smiles, key=f"ketcher_{ss.sk_nonce}", height=520)
                if drawn is not None and drawn != ss.sk_smiles:
                    if not drawn.strip():
                        st.info("The canvas is empty. Draw a structure and press Apply.")
                    else:
                        new = mc.from_smiles(drawn)
                        old = mc.from_smiles(ss.sk_smiles)
                        if new is None:
                            st.error("This drawing is not a valid structure. Check atom valences and charges.")
                        elif old is None or mc.canonical_smiles(new) != mc.canonical_smiles(old):
                            set_molecule(drawn, "Drawn structure", from_sketch=True)
        with t_smiles:
            st.text_input("SMILES", key="sk_box",
                          help="A text code for a molecule, e.g. CC(=O)Oc1ccccc1C(=O)O for aspirin.")
            nm = st.text_input("Name (optional)", key="sk_name_box", placeholder="e.g. Aspirin")
            if st.button("Convert", type="primary", icon=":material/sync_alt:"):
                if mc.from_smiles(ss.sk_box) is None:
                    st.error("RDKit could not read this SMILES. Check brackets, ring numbers and atom valences.")
                else:
                    set_molecule(ss.sk_box.strip(), nm.strip() or "My molecule")
        with t_name:
            query = st.text_input("Compound name", placeholder="e.g. imatinib, artemisinin, quercetin")
            if st.button("Search PubChem", icon=":material/search:") and query.strip():
                with st.spinner("Searching PubChem…"):
                    found, msg = pubchem_smiles(query.strip())
                if found:
                    set_molecule(found, query.strip().title())
                else:
                    st.error(msg)
        with t_example:
            st.selectbox("Example molecules", list(EXAMPLES), index=None, placeholder="Choose one",
                         key="sk_example", on_change=_pick_example)

    mol = mc.from_smiles(ss.sk_smiles)
    if mol is None:
        st.error("The current structure could not be read. Draw it again or type a new SMILES.")
        return
    name = st.text_input("Molecule name, used for the file names", value=ss.sk_name, key="nm_" + hashlib.md5(f"{ss.sk_nonce}{ss.sk_smiles}{ss.sk_name}".encode()).hexdigest()[:10])
    show_results(Chem.MolToMolBlock(mol), name.strip() or "molecule", file_has_3d=False, key="sk")


def files():
    st.title("Convert a file")
    brand_rule()
    st.write("Upload a structure to get its SMILES, view it, and save it in another format. "
             "Multi-molecule SDF and MOL2 files and docking outputs are fine; you choose which molecule to convert.")

    with st.container(border=True):
        up = st.file_uploader("Structure file", type=mc.READABLE_EXTENSIONS,
                              help="SDF, MOL, MOL2, PDB, PDBQT, XYZ, or a text file with one SMILES per line.")
        if up is None:
            st.caption("No file to hand? Download a sample, then upload it.")
            sample = mc.make_3d(mc.from_smiles(EXAMPLES["Erlotinib"]), True, "Erlotinib")[0]
            st.download_button("Sample: erlotinib 3D (PDB)", mc.to_pdb(sample).encode(), file_name="erlotinib.pdb",
                               mime="chemical/x-pdb", icon=":material/download:")
            return
        data = up.getvalue()
        if len(data) > 20 * 1024 * 1024:
            st.error("This file is larger than 20 MB. Split it into smaller files first.")
            return
        mols = read_upload(data, up.name)

    if not mols:
        st.error("No molecule could be read from this file. Check that the extension matches the contents. "
                 "Protein structures belong in HoloSift; MolCov is for small molecules.")
        return
    if len(mols) > 1:
        labels = [f"{i + 1}. {m['name']} ({m['formula']})" for i, m in enumerate(mols)]
        pick = st.selectbox(f"This file has {len(mols)} molecules. Choose one", range(len(mols)),
                            format_func=lambda i: labels[i])
    else:
        pick = 0
    item = mols[pick]
    molblock = item["molblock"]

    if item["note"]:
        st.info(item["note"])
        if "template" in item["note"]:
            with st.form(f"tpl_{pick}"):
                tpl = st.text_input("Template SMILES", placeholder="SMILES of the same molecule, e.g. from PubChem")
                if st.form_submit_button("Fix bond orders"):
                    fixed, msg = mc.apply_template(Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=False), tpl)
                    if fixed is None:
                        st.error(msg)
                    else:
                        ss[f"tpl_fixed_{hashlib.md5(data).hexdigest()}_{pick}"] = Chem.MolToMolBlock(fixed)
                        st.success(msg)
            molblock = ss.get(f"tpl_fixed_{hashlib.md5(data).hexdigest()}_{pick}", molblock)

    key = "f" + hashlib.md5((up.name + str(pick)).encode()).hexdigest()[:8]
    show_results(molblock, item["name"], file_has_3d=item["has_3d"], key=key)


def learn():
    st.title("Formats and terms")
    brand_rule()
    st.write("Short explanations of what MolCov shows and writes. Open any one you need.")
    guide = [
        ("SMILES", "A line of text that describes a molecule's atoms, bonds and rings. Lowercase letters are aromatic "
         "atoms, numbers open and close rings, and brackets show branches. The canonical SMILES is the one RDKit "
         "always writes for the same molecule, so you can compare molecules by text. The isomeric SMILES also keeps "
         "stereochemistry: @ and @@ for chiral centres, / and \\ for double-bond geometry."),
        ("InChI and InChIKey", "IUPAC's standard identifiers. The InChIKey is a fixed-length code made from the InChI "
         "and is the easiest way to search for a compound in PubChem, ChEMBL or Google."),
        ("2D or 3D?", "2D coordinates are a flat drawing: good for pictures, databases and reports. 3D coordinates give "
         "every atom a position in space: you need them for docking, pharmacophores and molecular dynamics. A 3D "
         "structure from MolCov is one reasonable low-energy shape, not the bound pose or the only conformer."),
        ("How MolCov builds 3D", "Hydrogens are added, RDKit's ETKDGv3 method embeds the molecule using known bond "
         "lengths, angles and torsion preferences, and the MMFF94 force field relaxes it (UFF if MMFF94 has no "
         "parameters for an atom). The seed is fixed, so the same input always gives the same structure."),
        ("SDF and MOL", "The MDL formats used by PubChem, ChEMBL and most cheminformatics tools. They store atoms, "
         "bonds, bond orders, charges and stereochemistry. An SDF file can hold many molecules with data fields; "
         "MOL holds one. Use SDF when you are unsure."),
        ("PDB", "The Protein Data Bank format. It lists atoms with names, residues and coordinates; bonds go in CONECT "
         "lines. Many docking and simulation tools want a ligand as PDB. MolCov names the residue LIG and writes "
         "double bonds as repeated CONECT lines so the file reads back correctly."),
        ("MOL2", "The Tripos format, with SYBYL atom types (such as C.ar or N.am), bond types and partial charges. "
         "MolCov writes Gasteiger charges and always includes hydrogens. It is used by Chimera, DOCK, GOLD and as "
         "input for AMBER's antechamber, which will usually recalculate the charges (for example AM1-BCC)."),
        ("XYZ", "Just elements and coordinates, no bonds. Tools have to guess bonds from distances, so MolCov offers "
         "XYZ for 3D structures only."),
        ("PDBQT", "AutoDock's format: PDB plus charges and AutoDock atom types. MolCov can read it but does not write "
         "it, because docking preparation needs choices about protonation and rotatable bonds. Download SDF from "
         "MolCov and prepare it with Meeko or Open Babel."),
        ("Files without bond orders", "PDB files without CONECT lines, PDBQT and XYZ files only give positions. MolCov "
         "works the bonds out from distances, which is reliable when hydrogens are present. Without hydrogens, give "
         "the correct SMILES as a template to restore double bonds and aromatic rings."),
    ]
    for term, text in guide:
        with st.expander(term):
            st.write(text)
    st.caption("References: Weininger, J. Chem. Inf. Comput. Sci. 1988 (SMILES); Heller et al., J. Cheminform. 2015 "
               "(InChI); Wang and Riniker, J. Chem. Inf. Model. 2020 (ETKDGv3); Halgren, J. Comput. Chem. 1996 (MMFF94); "
               "Gasteiger and Marsili, Tetrahedron 1980; Lipinski et al., Adv. Drug Deliv. Rev. 1997; "
               "Veber et al., J. Med. Chem. 2002. Molecule editor: Ketcher (EPAM). 3D graphics: 3Dmol.js "
               "(Rego and Koes, Bioinformatics 2015). Chemistry: RDKit.")


# ---------------------------------------------------------------- navigation
pages = {
    "home": st.Page(home, title="Home", icon=":material/home:", default=True),
    "convert": st.Page(convert, title="Draw and convert", icon=":material/draw:", url_path="convert"),
    "files": st.Page(files, title="Convert a file", icon=":material/upload_file:", url_path="files"),
    "learn": st.Page(learn, title="Formats and terms", icon=":material/menu_book:", url_path="learn"),
}
nav = st.navigation(list(pages.values()))
with st.sidebar:
    if LOGO_FULL.exists():
        st.image(str(LOGO_FULL), width="stretch")
    st.markdown(f"[theinsilicolab.org]({SITE_URL})")
    st.markdown(f"**More from the [CADD Toolkit]({TOOLKIT_URL})**  \n"
                f"[Molecular Learning Workspace]({WORKSPACE_URL})  \n[HoloSift]({HOLOSIFT_URL})")
    st.caption(f"{APP_NAME} v{VERSION}")
nav.run()

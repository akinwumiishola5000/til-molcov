# MolCov

A molecule sketcher and converter for members of The Insilico Lab, part of the CADD Toolkit at <https://www.theinsilicolab.org/cadd-toolkit>.

The name comes from *mol*ecule and *conv*ert.

## What it does

**Draw and convert**

- Draw a structure in the Ketcher editor, type a SMILES, search PubChem by name, or pick an example.
- Shows the canonical and isomeric SMILES, formula, InChI and InChIKey, each ready to copy.
- Shows the molecule in 2D and 3D. You can change the 3D style, colour by element, set your own carbon colour or use one colour, change the background, show hydrogens and atom labels, spin it, rotate the 2D drawing, and save a PNG (or SVG for 2D).
- Downloads 2D or 3D coordinates as SDF, PDB, MOL2, MOL or XYZ, with or without hydrogens, plus a SMILES file. 3D structures are built with RDKit's ETKDGv3 and relaxed with MMFF94 (UFF as a fallback).
- Calculates the same properties as the Molecular Learning Workspace, with Lipinski's and Veber's rules, and downloads them as CSV.

**Convert a file**

- Reads SDF, MOL, MOL2, PDB, PDBQT, XYZ and SMI files, including multi-molecule files and docking outputs.
- Keeps the 3D coordinates from the file, or rebuilds them.
- For files with positions but no bond orders (PDB without CONECT, PDBQT, XYZ), works the bonds out from distances, and accepts a template SMILES to restore double bonds when hydrogens are missing.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit app: pages, inputs, downloads and properties |
| `molcov_chem.py` | All the chemistry (reading, 2D/3D, writing, properties). Usable on its own in Colab |
| `molcov_viewer.html` | The 2D/3D viewer with colour, background, spin and picture controls |
| `molcov_hero.html` | The animated homepage stage (SMILES, then 2D, then 3D) |
| `requirements.txt` | Python packages for Streamlit Cloud |
| `packages.txt` | System libraries RDKit needs for drawing |
| `.streamlit/config.toml` | Brand colours, the same as the Workspace and HoloSift |
| `logo_full.png`, `logo_icon.png` | The Insilico Lab logo and browser-tab icon |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a public GitHub repository called `til-molcov` and upload every file in this folder, including the `.streamlit` folder.
2. At <https://share.streamlit.io> choose **Create app** → **Deploy a public app from GitHub**. Repository `akinwumiishola5000/til-molcov`, branch `main`, main file `app.py`, App URL `theinsilicolab-molcov`.
3. In Google Sites, add a MolCov subpage under CADD Toolkit and embed `https://theinsilicolab-molcov.streamlit.app/?embed=true`. Make the embed about 1,800 px tall.

## Links for teaching

- Open a molecule directly: `https://theinsilicolab-molcov.streamlit.app/convert?smiles=CC(=O)Oc1ccccc1C(=O)O&name=Aspirin`
- Open the file converter: `https://theinsilicolab-molcov.streamlit.app/files`

In a link, write `#` in a SMILES as `%23` (for example a triple bond `C#C` becomes `C%23C`).

## Outside services

PubChem for name search; 3Dmol.js 2.4.2 from `cdn.jsdelivr.net`; Source Sans 3 and Material Symbols from Google Fonts. Uploaded files are processed in the app's session and are not stored.

## Credits

Built for The Insilico Lab (theinsilicolab.org). Chemistry: RDKit. Molecule editor: Ketcher (EPAM) through streamlit-ketcher. 3D graphics: 3Dmol.js (Rego and Koes, Bioinformatics 2015). 3D generation: ETKDGv3 (Wang and Riniker, J. Chem. Inf. Model. 2020) and MMFF94 (Halgren, J. Comput. Chem. 1996).

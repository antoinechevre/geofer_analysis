"""Exécute Notebook_corridor.ipynb pour chaque corridor listé dans l'onglet
"corridor" de gares_geofer.xlsx (colonnes gare_depart, gare_arrivee).

Chaque exécution calcule le corridor (gares intermédiaires, aire
d'influence, flux, population, charge par tronçon, carte) et pousse ses
fichiers caches sur le dataset HF antoinechevre/Analyse_gare (dernière
cellule du notebook) — utile pour préchauffer en lot le cache partagé que
l'app Corridor Analyse réutilise ensuite (sidebar "Corridor déjà identifié").

Usage :
    python lancer_corridors.py [gares_geofer.xlsx] [Notebook_corridor.ipynb]
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

XLSX_PAR_DEFAUT = "gares_geofer.xlsx"
NOTEBOOK_PAR_DEFAUT = "Notebook_corridor.ipynb"
ONGLET_CORRIDOR = "corridor"
KERNEL = "gtfs-app"


def lister_corridors(chemin_xlsx: Path) -> pd.DataFrame:
    corridors = pd.read_excel(chemin_xlsx, sheet_name=ONGLET_CORRIDOR)
    return corridors.dropna(subset=["gare_depart", "gare_arrivee"])


def parametrer_notebook(chemin_notebook: Path, gare_depart: str, gare_arrivee: str) -> dict:
    notebook = json.loads(chemin_notebook.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    source = re.sub(r'GARE_DEPART = ".*"', f'GARE_DEPART = "{gare_depart}"', source, count=1)
    source = re.sub(r'GARE_ARRIVEE = ".*"', f'GARE_ARRIVEE = "{gare_arrivee}"', source, count=1)
    notebook["cells"][0]["source"] = source.splitlines(keepends=True)
    for cell in notebook["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
    return notebook


def executer_corridor(chemin_notebook: Path, gare_depart: str, gare_arrivee: str) -> bool:
    print(f"\n=== {gare_depart} -> {gare_arrivee} ===")
    notebook = parametrer_notebook(chemin_notebook, gare_depart, gare_arrivee)
    with tempfile.NamedTemporaryFile(
        suffix=".ipynb", mode="w", encoding="utf-8", delete=False, dir=chemin_notebook.parent,
    ) as f:
        json.dump(notebook, f)
        chemin_tmp = Path(f.name)
    try:
        resultat = subprocess.run(
            [
                "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                f"--ExecutePreprocessor.kernel_name={KERNEL}", str(chemin_tmp),
            ],
            capture_output=True, text=True,
        )
        if resultat.returncode != 0:
            print("  ÉCHEC :")
            print("  " + resultat.stderr.strip().replace("\n", "\n  ")[-2000:])
            return False
        print("  OK")
        return True
    finally:
        chemin_tmp.unlink(missing_ok=True)


def main():
    chemin_xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(XLSX_PAR_DEFAUT)
    chemin_notebook = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(NOTEBOOK_PAR_DEFAUT)

    corridors = lister_corridors(chemin_xlsx)
    print(f"{len(corridors)} corridor(s) à traiter depuis {chemin_xlsx} (onglet {ONGLET_CORRIDOR!r})")

    echecs = []
    for ligne in corridors.itertuples():
        ok = executer_corridor(chemin_notebook, ligne.gare_depart, ligne.gare_arrivee)
        if not ok:
            echecs.append((ligne.gare_depart, ligne.gare_arrivee))

    print(f"\nTerminé : {len(corridors) - len(echecs)}/{len(corridors)} corridors traités avec succès.")
    if echecs:
        print("Échecs :")
        for depart, arrivee in echecs:
            print(f"  - {depart} -> {arrivee}")


if __name__ == "__main__":
    main()

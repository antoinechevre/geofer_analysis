"""Récupère les bases de flux de mobilité INSEE (domicile-travail 2022,
domicile-études 2021) et les met en cache sur le dataset HF
antoinechevre/accessibility-data, pour un chargement à la demande par
app.py (flèches proportionnelles domicile-travail/études par commune).

Usage : python extraire_flux_mobilite.py

Sources (pages de téléchargement, pas d'API — fichiers ZIP direct) :
- https://www.insee.fr/fr/statistiques/8582949 (domicile-travail 2022)
- https://www.insee.fr/fr/statistiques/8201894 (domicile-études 2021)
"""

import io
import os
import zipfile

import requests
from huggingface_hub import HfApi

SOURCES = {
    "flux_domicile_travail.csv": (
        "https://www.insee.fr/fr/statistiques/fichier/8582949/"
        "base-flux-mobilite-domicile-lieu-travail-2022_csv.zip"
    ),
    "flux_domicile_etudes.csv": (
        "https://www.insee.fr/fr/statistiques/fichier/8201894/"
        "base-flux-mobilite-domicile-lieu-etude-2021-csv.zip"
    ),
}

OUTPUT_DIR = "Data_INSEE"
HF_DATASET_REPO = "antoinechevre/accessibility-data"
HF_PREFIX = "extracted"


def telecharger_et_extraire(nom_sortie: str, url: str) -> str:
    print(f"Téléchargement {url}...")
    reponse = requests.get(url, timeout=120)
    reponse.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(reponse.content)) as archive:
        # Une seule entrée CSV par zip source, quel que soit son nom exact.
        nom_entree = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        contenu = archive.read(nom_entree)

    chemin_local = os.path.join(OUTPUT_DIR, nom_sortie)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(chemin_local, "wb") as f:
        f.write(contenu)
    print(f"✓ {chemin_local} ({len(contenu) / 1e6:.1f} Mo)")
    return chemin_local


def envoyer_vers_hf(chemin_local: str, nom_fichier: str):
    try:
        HfApi().upload_file(
            path_or_fileobj=chemin_local,
            path_in_repo=f"{HF_PREFIX}/{nom_fichier}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"✓ envoyé vers {HF_DATASET_REPO}/{HF_PREFIX}/{nom_fichier}")
    except Exception as e:
        print(f"[hf] échec envoi {nom_fichier} : {type(e).__name__}: {e}")


def main():
    for nom_sortie, url in SOURCES.items():
        chemin_local = telecharger_et_extraire(nom_sortie, url)
        envoyer_vers_hf(chemin_local, nom_sortie)


if __name__ == "__main__":
    main()

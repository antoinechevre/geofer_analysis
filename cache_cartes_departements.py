"""Pré-calcule les carreaux INSEE 200m de chaque département et les publie
sur le dataset HuggingFace antoinechevre/Analyse_gare.

C'est la partie la plus lente de l'appli (load_insee_carreaux dans app.py) :
une lecture par bbox du gpkg national Filosofi 2019, suivie d'un filtre
intersects et du calcul des indicateurs dérivés (pop, niveau_vie,
taux_pauvrete, part_65p) — 10 à 15s pour un département avec ses voisins.

Ce script fait ce travail une fois pour toutes, département par département
(pas par zone dept+voisins, pour que les fichiers restent combinables :
"51+02+08" est simplement la concaténation des trois fichiers), et les
dépose sur le dataset au format GeoParquet. app.py les télécharge ensuite
directement (cf. load_carreaux_dept_cache) au lieu de relire le gpkg
national à chaque changement de département.

Usage :
    HF_TOKEN=hf_xxx_écriture python cache_cartes_departements.py [CODE ...]

Sans code en argument, traite tous les départements de
Data_admin/departements.geojson. HF_TOKEN doit avoir un accès en écriture
sur le dataset antoinechevre/Analyse_gare (à recréer/relancer quand la
source Filosofi ou les contours des départements changent).
"""

import argparse
import os
import sys
import tempfile

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402 — réutilise load_departements/get_insee_local_path/load_insee_carreaux

CARTES_DATASET_REPO = "antoinechevre/Analyse_gare"
CARTES_DEPT_DIR = "carreaux_departements"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codes", nargs="*", help="Codes département à (re)générer (défaut : tous)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(f"HF_TOKEN manquant (accès en écriture requis sur {CARTES_DATASET_REPO})")

    api = HfApi(token=token)
    api.create_repo(CARTES_DATASET_REPO, repo_type="dataset", exist_ok=True)

    departements = app.load_departements()
    insee_path = app.get_insee_local_path()
    codes = args.codes or departements["code"].tolist()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for code in codes:
            matches = departements[departements["code"] == code]
            if matches.empty:
                print(f"{code} : département inconnu, ignoré")
                continue

            dept_geom = matches.iloc[0].geometry
            carreaux = app.load_insee_carreaux(insee_path, dept_geom, code)
            if carreaux.empty:
                print(f"{code} : aucun carreau INSEE (hors couverture Filosofi métropole ?), ignoré")
                continue

            local_path = os.path.join(tmp_dir, f"{code}.parquet")
            carreaux.to_parquet(local_path)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"{CARTES_DEPT_DIR}/{code}.parquet",
                repo_id=CARTES_DATASET_REPO,
                repo_type="dataset",
            )
            print(f"{code} : {len(carreaux)} carreaux -> {CARTES_DATASET_REPO}/{CARTES_DEPT_DIR}/{code}.parquet")

    print("Terminé.")


if __name__ == "__main__":
    main()

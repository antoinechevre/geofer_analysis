"""Récupère la fréquentation moyenne annuelle par gare et la joint aux gares
de Data_geofer/geofer_gares.csv via codeUic — à la suite de
extraire_passages_gares_gtfs.py (source différente, même principe de
jointure par identifiant de gare).

Usage : python extraire_frequentation_gares.py

Note : le dataset data.gouv.fr "Fréquentation moyenne annuelle par gare"
(https://www.data.gouv.fr/datasets/frequentation-moyenne-annuelle-par-gare)
n'est agrégé qu'au niveau géographique (commune/EPCI/département...) : ses
fichiers n'ont ni nom de gare ni identifiant de gare, seulement un code
commune INSEE — inexploitable pour une jointure par gare. Ce script utilise
donc directement la source primaire citée sur cette page : l'API
OpenDataSoft de SNCF Gares & Connexions ("frequentation-gares"), qui expose
un code_uic_complet correspondant au codeUic de geofer_gares.csv.
"""

import re

import pandas as pd
import requests

API_URL = "https://data.sncf.com/api/records/1.0/search/"
DATASET = "frequentation-gares"
GARES_PATH = "Data_geofer/geofer_gares.csv"
OUTPUT_PATH = "Data_SNCF/frequentation_gares.csv"
PAGE_SIZE = 1000


def recuperer_frequentation() -> pd.DataFrame:
    """Pagine sur l'API OpenDataSoft pour récupérer tous les enregistrements
    du dataset (~3000 gares, 4 pages à 1000 lignes)."""
    lignes = []
    start = 0
    while True:
        reponse = requests.get(
            API_URL, params={"dataset": DATASET, "rows": PAGE_SIZE, "start": start}, timeout=30
        )
        reponse.raise_for_status()
        data = reponse.json()
        records = data["records"]
        if not records:
            break
        for rec in records:
            lignes.append(rec["fields"])
        start += PAGE_SIZE
        if start >= data["nhits"]:
            break
    return pd.DataFrame(lignes)


def colonnes_annees_voyageurs(colonnes) -> dict[int, str]:
    """{année: nom_colonne} pour les colonnes "voyageurs" (hors
    "non_voyageurs", qui compte aussi les personnes entrant en gare sans
    voyager). Gère la coquille du jeu de données sur 2017, exposé en
    "totalvoyageurs2017" sans séparateurs plutôt que "total_voyageurs_2017"
    comme les autres années — sans quoi 2017 disparaît silencieusement."""
    annees = {}
    for col in colonnes:
        if "non_voyageurs" in col:
            continue
        m = re.fullmatch(r"total_?voyageurs_?(\d{4})", col)
        if m:
            annees[int(m.group(1))] = col
    return annees


def main():
    print(f"Récupération de {DATASET} depuis l'API SNCF Gares & Connexions...")
    frequentation = recuperer_frequentation()
    print(f"✓ {len(frequentation)} gares récupérées")

    frequentation["code_uic_complet"] = frequentation["code_uic_complet"].astype(str)
    annees = colonnes_annees_voyageurs(frequentation.columns)

    # Jeu de données incomplet par gare (années manquantes selon les cas) :
    # une ligne par (gare, année) avec une valeur connue, plutôt qu'une
    # colonne par année — c'est l'historique demandé, pas juste l'année la
    # plus récente.
    historique = pd.concat(
        [
            frequentation[["code_uic_complet", col]]
            .rename(columns={col: "voyageurs"})
            .assign(annee=annee)
            for annee, col in sorted(annees.items())
        ],
        ignore_index=True,
    )
    historique = historique.dropna(subset=["voyageurs"])
    historique["voyageurs"] = historique["voyageurs"].astype(int)

    gares = pd.read_csv(GARES_PATH, dtype={"codeUic": str})
    gares = gares[gares["siOuverte"]][["codeUic", "nomGare", "nomCommune"]]

    resultat = gares.merge(historique, left_on="codeUic", right_on="code_uic_complet", how="inner")
    resultat = resultat.drop(columns=["code_uic_complet"]).sort_values(["codeUic", "annee"])

    resultat.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ {len(resultat)} lignes (gare x année) écrites dans {OUTPUT_PATH}")
    print(f"  {resultat['codeUic'].nunique()} gares, années {min(annees)}-{max(annees)}")


if __name__ == "__main__":
    main()

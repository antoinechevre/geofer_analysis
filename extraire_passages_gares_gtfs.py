"""Extrait le nombre de passages de train par gare (TER / Intercités / TGV)
depuis le GTFS national SNCF, pour un jour ouvré de référence (JOB).

Usage : python extraire_passages_gares_gtfs.py

Nécessite gtfs-kit (pip install gtfs-kit) en plus des dépendances de
requirements.txt — outil de préparation de données, pas utilisé par l'app
Streamlit elle-même (cf. Notebook_dowload_geofer.ipynb, même esprit).

Classification TER / Intercités / TGV : le stop_id de chaque StopPoint (un
arrêt physique par mode desservant une gare) embarque le mode commercial
exploitant, ex. "StopPoint:OCETrain TER-87481184" ou
"StopPoint:OCETGV INOUI-71043075" — pas un champ standard GTFS, propre à cet
export SNCF (vérifié sur l'export du 2026-09-04). Plus fiable qu'une
classification par ligne : des lignes Intercités classiques (Paris-Toulouse,
Nantes-Bordeaux...) partagent le même format de code que des lignes TGV/OUIGO
sans aucun marqueur commun les distinguant au niveau ligne.
"""

import re
import warnings
from datetime import datetime

import gtfs_kit as gk
import pandas as pd

# compute_trip_activity insère ses colonnes date une par une : bruit sans
# rapport avec ce script, propre à l'implémentation interne de gtfs_kit.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

GTFS_PATH = "Data_SNCF/National_GTFS.zip"
GARES_PATH = "Data_geofer/geofer_gares.csv"
OUTPUT_PATH = "Data_SNCF/passages_gares_par_mode.csv"

MODES_PAR_CATEGORIE = {
    "ter": {"Train TER", "Car TER", "TramTrain"},
    "intercites": {"INTERCITES", "INTERCITES de nuit"},
    "tgv": {"TGV INOUI", "OUIGO", "ICE", "Lyria", "Car à réservation"},
}
CATEGORIES = ["ter", "intercites", "tgv", "autre"]


def categorie_depuis_stop_id(stop_id: str) -> str:
    m = re.match(r"StopPoint:OCE(.+)-\d+$", stop_id)
    mode = m.group(1) if m else None
    for categorie, modes in MODES_PAR_CATEGORIE.items():
        if mode in modes:
            return categorie
    return "autre"


def determiner_jour_reference(feed: gk.Feed) -> str:
    """Dernier mardi ou jeudi avec un niveau de service fiable (>= 70% du
    nombre de trips/jour maximum observé) — même logique que le reste de
    l'écosystème (cf. antoinechevre/Accessibility_analysis, src/info_reseau.py),
    sans son repli "hors vacances scolaires" (nécessite une géolocalisation
    académique hors périmètre ici)."""
    dates = feed.get_dates()
    activite = feed.compute_trip_activity(dates)
    comptages = activite.set_index("trip_id")[dates].sum()
    seuil = 0.7 * comptages.max()
    dates_fiables = comptages[comptages >= seuil].index.tolist()

    dates_parsees = [datetime.strptime(d, "%Y%m%d") for d in dates_fiables]
    mar_jeu = [d for d in dates_parsees if d.weekday() in (1, 3)]
    reference = max(mar_jeu) if mar_jeu else max(dates_parsees)
    return reference.strftime("%Y%m%d")


def main():
    print(f"Chargement du GTFS {GTFS_PATH}...")
    feed = gk.read_feed(GTFS_PATH, dist_units="km")

    date_reference = determiner_jour_reference(feed)
    print(f"Jour de référence (JOB) : {date_reference}")

    trips_actifs = feed.get_trips(date=date_reference)["trip_id"]
    stop_times = feed.stop_times[feed.stop_times["trip_id"].isin(trips_actifs)]
    print(f"{len(trips_actifs):,} trips actifs, {len(stop_times):,} passages ce jour-là".replace(",", " "))

    stops = feed.stops[["stop_id", "parent_station", "location_type"]].copy()
    stop_points = stops[stops["location_type"].astype(str) == "0"].copy()
    stop_points["categorie"] = stop_points["stop_id"].map(categorie_depuis_stop_id)

    passages = stop_times.merge(
        stop_points[["stop_id", "parent_station", "categorie"]], on="stop_id", how="left"
    )
    passages["uic"] = passages["parent_station"].str.replace("StopArea:OCE", "", regex=False)

    par_gare = passages.groupby(["uic", "categorie"]).size().unstack(fill_value=0)
    for categorie in CATEGORIES:
        if categorie not in par_gare.columns:
            par_gare[categorie] = 0
    par_gare = par_gare[CATEGORIES].reset_index()
    par_gare["totalPassages"] = par_gare[CATEGORIES].sum(axis=1)
    par_gare = par_gare.rename(
        columns={
            "ter": "passagesTer",
            "intercites": "passagesIntercites",
            "tgv": "passagesTgv",
            "autre": "passagesAutre",
        }
    )

    gares = pd.read_csv(GARES_PATH, dtype={"codeUic": str})
    gares = gares[gares["siOuverte"]][["codeUic", "nomGare", "nomCommune"]]

    resultat = gares.merge(par_gare, left_on="codeUic", right_on="uic", how="left").drop(columns=["uic"])
    colonnes_passages = ["passagesTer", "passagesIntercites", "passagesTgv", "passagesAutre", "totalPassages"]
    for col in colonnes_passages:
        resultat[col] = resultat[col].fillna(0).astype(int)
    resultat.insert(0, "dateReference", date_reference)
    resultat = resultat.sort_values("totalPassages", ascending=False)

    resultat.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ {len(resultat)} gares écrites dans {OUTPUT_PATH}")
    print(f"  dont {(resultat['totalPassages'] > 0).sum()} avec au moins un passage ce jour-là")


if __name__ == "__main__":
    main()

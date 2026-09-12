"""Corridor Analyse — flux domicile-travail/études et desserte le long d'un
corridor ferroviaire entre deux gares Géofer.

Choisissez deux gares : l'app suit la vraie ligne de chemin de fer (données
OpenStreetMap) entre les deux, détecte les gares intermédiaires (une seule
par commune — la plus fréquentée quand une agglomération en a plusieurs),
étend chaque gare à son aire d'influence réelle Géofer 10 min en voiture,
fusionne les communes de cette aire sur leur gare avant de calculer les
flux domicile-travail/domicile-études, et affiche une carte HTML avec
toutes les couches sélectionnables (mêmes briques de rendu que l'app
principale Géofer Analysis).

Les résultats intermédiaires sont mis en cache sur le dataset HF
antoinechevre/Analyse_gare : un corridor déjà analysé par un autre
visiteur se recharge instantanément, sans repasser par Overpass/le
routage réseau/l'API découpage administratif.
"""

import io
import math
import os
import tempfile
import time

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
import requests
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import substring, unary_union, voronoi_diagram

GEOFER_DIR = "Data_geofer"
SNCF_DIR = "Data_SNCF"

OFFRE_PATH = f"{SNCF_DIR}/passages_gares_par_mode.csv"
OFFRE_CATEGORIES = {
    "passagesTer": ("TER", "#2ca25f"),
    "passagesIntercites": ("Intercités", "#fd8d3c"),
    "passagesTgv": ("TGV", "#e34a33"),
}
OFFRE_MIN_RADIUS_PX = 5
OFFRE_MAX_RADIUS_PX = 42

FREQUENTATION_PATH = f"{SNCF_DIR}/frequentation_gares.csv"
FREQUENTATION_ANNEE = 2024
FREQUENTATION_COLOR = "#6a3d9a"
FREQUENTATION_MIN_RADIUS_PX = 7
FREQUENTATION_MAX_RADIUS_PX = 64

ISOCHRONE_FILES = {
    "10 min en voiture": (f"{GEOFER_DIR}/iso_10min_voiture.geojson", "#9cd0ed"),
    "10 min à vélo": (f"{GEOFER_DIR}/iso_10min_velo.geojson", "#1992d4"),
    "15 min à pied": (f"{GEOFER_DIR}/iso_15min_pieton.geojson", "#0a3a55"),
}

CARREAUX_COLOR_SCALE = [
    "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272",
    "#fb6a4a", "#de2d26", "#a50f15", "#67000d",
]
CARREAUX_COLOR_GAMMA = 0.45
CHARGE_COLOR_SCALE = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]

# Mêmes sources que app.py (Space Geofer_analysis) : flux INSEE et carreaux
# 200m sont trop volumineux pour être commités dans ce repo, donc récupérés
# depuis le même dataset HF au lieu d'être dupliqués.
INSEE_DATASET_REPO = "antoinechevre/accessibility-data"
INSEE_LEGER_REMOTE_FILE = "extracted/carreaux_200m_met_leger.parquet"
INSEE_DIR = "Data_INSEE"
FLUX_TRAVAIL_FILE = "flux_domicile_travail.csv"
FLUX_ETUDES_FILE = "flux_domicile_etudes.csv"
FLUX_THEMES = {
    "travail": ("Domicile-travail (2022)", "CODGEO", "LIBGEO", "DCLT", "L_DCLT", "NBFLUX_C22_ACTOCC15P"),
    "etudes": ("Domicile-études (2021)", "CODGEO", "LIBGEO", "DCETU", "L_DCETU", "NBFLUX_C21_SCOL02P"),
}
FLUX_MIN_WEIGHT_PX = 1
FLUX_MAX_WEIGHT_PX = 10
FLUX_COULEURS = {"travail": "#1f78b4", "etudes": "#e6550d", "cumul": "#1a9850"}
COURBURE_PAR_THEME = {"travail": 0.18, "etudes": 0.30, "cumul": 0.24}

# Paris/Lyon/Marseille : chaque arrondissement est une commune INSEE à part
# dans les bases de flux, ce qui dilue le signal "vers Paris" — fusionnés
# vers le code commune globale avant tout calcul (cf. app.py).
ARRONDISSEMENTS_A_FUSIONNER = {
    **{f"751{i:02d}": "75056" for i in range(1, 21)},
    **{f"6938{i}": "69123" for i in range(1, 10)},
    **{f"132{i:02d}": "13055" for i in range(1, 17)},
}
VILLES_FUSIONNEES = {"75056": "Paris", "69123": "Lyon", "13055": "Marseille"}

# Cache des résultats intermédiaires par corridor, partagé entre tous les
# visiteurs de l'app (cf. charger_depuis_cache_hf / sauvegarder_cache_hf).
CACHE_DATASET_REPO = "antoinechevre/Analyse_gare"
CACHE_PREFIX = "corridors"
CACHE_FICHIERS = [
    "gares_corridor.csv", "communes_influence.csv", "flux_corridor.csv",
    "population_gares.csv", "charge_troncons.csv", "meta.csv",
]

MARGE_BBOX_DEG = 0.4  # marge autour des deux gares pour la requête Overpass
SEUIL_DISTANCE_ALERTE_KM = 200  # message d'erreur dédié au-delà, si aucune voie ne relie les deux gares
TAMPON_VOIE_M_DEFAUT = 300  # distance max à la voie réelle pour qu'une gare soit retenue

OVERPASS_MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

CARTO_API_KEY = os.environ.get("CARTO_API_KEY")
CARTO_ATTR = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
    '&copy; <a href="https://carto.com/attributions">CARTO</a>'
)


def carto_tile_url(variant: str) -> str:
    url = f"https://{{s}}.basemaps.cartocdn.com/{variant}/{{z}}/{{x}}/{{y}}{{r}}.png"
    return f"{url}?key={CARTO_API_KEY}" if CARTO_API_KEY else url


def nom_corridor_de(gare_depart: str, gare_arrivee: str) -> str:
    return f"{gare_depart}_{gare_arrivee}_corridor".replace(" ", "_")


# --------------------------------------------------------------------------
# Chargement des données de base (mise en cache, comme app.py)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des gares Géofer...")
def load_gares() -> gpd.GeoDataFrame:
    df = pd.read_csv(f"{GEOFER_DIR}/geofer_gares.csv")
    df = df[df["siOuverte"]].copy()
    df["codeUic"] = df["codeUic"].astype(str)
    df["inseeCommune"] = df["inseeCommune"].astype(str).str.zfill(5)
    df["inseeDepartement"] = df["inseeDepartement"].astype(str)
    df["label"] = df["nomGare"] + " — " + df["nomCommune"] + " (" + df["codeUic"] + ")"
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["wgs84Lon"], df["wgs84Lat"]), crs="EPSG:4326",
    )
    return gdf.sort_values("label")


@st.cache_resource(show_spinner="Chargement d'un isochrone Géofer...")
def load_isochrones(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


@st.cache_data(show_spinner="Chargement de l'offre 2026...")
def load_offre_2026() -> pd.DataFrame:
    df = pd.read_csv(OFFRE_PATH, dtype={"codeUic": str})
    df["totalClasse"] = df[list(OFFRE_CATEGORIES)].sum(axis=1)
    return df[df["totalClasse"] > 0]


@st.cache_data(show_spinner="Chargement de la fréquentation annuelle...")
def load_frequentation_annee() -> pd.DataFrame:
    df = pd.read_csv(FREQUENTATION_PATH, dtype={"codeUic": str})
    df = df[df["annee"] == FREQUENTATION_ANNEE]
    return df[df["voyageurs"] > 0]


@st.cache_data(show_spinner="Chargement de l'historique de fréquentation...")
def load_frequentation_historique() -> pd.DataFrame:
    return pd.read_csv(FREQUENTATION_PATH, dtype={"codeUic": str})


def get_insee_leger_path() -> str:
    local_path = os.path.join(INSEE_DIR, os.path.basename(INSEE_LEGER_REMOTE_FILE))
    if os.path.exists(local_path):
        return local_path
    return hf_hub_download(
        repo_id=INSEE_DATASET_REPO, repo_type="dataset",
        filename=INSEE_LEGER_REMOTE_FILE, token=os.environ.get("HF_TOKEN"),
    )


@st.cache_resource(show_spinner="Chargement des carreaux INSEE (France entière, une fois par session)...")
def load_carreaux_france() -> gpd.GeoDataFrame:
    return gpd.read_parquet(get_insee_leger_path())


def get_flux_local_path(nom_fichier: str) -> str:
    local_path = os.path.join(INSEE_DIR, nom_fichier)
    if os.path.exists(local_path):
        return local_path
    return hf_hub_download(
        repo_id=INSEE_DATASET_REPO, repo_type="dataset",
        filename=f"extracted/{nom_fichier}", token=os.environ.get("HF_TOKEN"),
    )


@st.cache_resource(show_spinner="Chargement des flux de mobilité...")
def load_flux(theme: str) -> pd.DataFrame:
    nom_fichier = FLUX_TRAVAIL_FILE if theme == "travail" else FLUX_ETUDES_FILE
    chemin = get_flux_local_path(nom_fichier)
    _, col_origine, col_label_origine, col_dest, col_label_dest, col_flux = FLUX_THEMES[theme]
    df = pd.read_csv(chemin, sep=";", dtype={col_origine: str, col_dest: str})
    df = df.rename(columns={
        col_origine: "origine", col_label_origine: "label_origine",
        col_dest: "destination", col_label_dest: "label_destination", col_flux: "flux",
    })
    df = df[["origine", "label_origine", "destination", "label_destination", "flux"]]

    df["origine"] = df["origine"].map(lambda c: ARRONDISSEMENTS_A_FUSIONNER.get(c, c))
    df["destination"] = df["destination"].map(lambda c: ARRONDISSEMENTS_A_FUSIONNER.get(c, c))
    df["label_origine"] = df["origine"].map(VILLES_FUSIONNEES).fillna(df["label_origine"])
    df["label_destination"] = df["destination"].map(VILLES_FUSIONNEES).fillna(df["label_destination"])
    return df.groupby(
        ["origine", "label_origine", "destination", "label_destination"], as_index=False
    )["flux"].sum()


@st.cache_data(show_spinner="Récupération des communes du département...")
def load_communes_departement(code_dept: str) -> pd.DataFrame:
    r = requests.get(
        "https://geo.api.gouv.fr/communes",
        params={"codeDepartement": code_dept, "geometry": "contour", "format": "geojson", "fields": "nom,code"},
        timeout=30,
    )
    r.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
    return pd.DataFrame({"code": gdf["code"], "nom": gdf["nom"], "geometry_wkt": gdf.geometry.to_wkt()})


# --------------------------------------------------------------------------
# Cache partagé (dataset HF antoinechevre/Analyse_gare)
# --------------------------------------------------------------------------

def charger_depuis_cache_hf(nom_corridor: str):
    """Tente de recharger un corridor déjà analysé par un autre visiteur.
    Retourne None si le cache est incomplet ou absent (calcul normal)."""
    token = os.environ.get("HF_TOKEN")
    # Codes commune/gare INSEE zero-paddés (ex. "01001", "87485003") : lus
    # comme entiers sinon, ce qui casserait tous les merges/isin() par la
    # suite (cf. app.py::load_gares pour le même piège).
    dtypes_par_fichier = {
        "gares_corridor.csv": {"codeUic": str, "inseeCommune": str, "inseeDepartement": str},
        "communes_influence.csv": {"gare_corridor_inseeCommune": str, "inseeCommune": str},
        "flux_corridor.csv": {"origine": str, "destination": str},
    }
    fichiers = {}
    try:
        for nom_fichier in CACHE_FICHIERS:
            chemin = hf_hub_download(
                repo_id=CACHE_DATASET_REPO, repo_type="dataset",
                filename=f"{CACHE_PREFIX}/{nom_corridor}/{nom_fichier}", token=token,
            )
            fichiers[nom_fichier] = pd.read_csv(chemin, dtype=dtypes_par_fichier.get(nom_fichier))
    except Exception:
        return None

    gares_corridor = fichiers["gares_corridor.csv"]
    gares_corridor = gpd.GeoDataFrame(
        gares_corridor, geometry=gpd.GeoSeries.from_wkt(gares_corridor["geometry_wkt"]), crs="EPSG:2154",
    ).drop(columns="geometry_wkt")

    meta = fichiers["meta.csv"].iloc[0]
    ligne_voie = shapely_wkt.loads(meta["ligne_voie_wkt"])
    aire_influence_totale = calculer_aire_influence_totale_simple(gares_corridor)
    departements_geom = calculer_departements_geom_simple(gares_corridor)

    return {
        "gares_corridor": gares_corridor,
        "communes_influence": fichiers["communes_influence.csv"],
        "flux_corridor": fichiers["flux_corridor.csv"],
        "population_corridor": fichiers["population_gares.csv"],
        "charge_troncons": fichiers["charge_troncons.csv"],
        "ligne_voie": ligne_voie,
        "aire_influence_totale": aire_influence_totale,
        "departements_geom": departements_geom,
        "distance_directe_km": float(meta["distance_directe_km"]),
    }


def sauvegarder_cache_hf(nom_corridor: str, gare_depart: str, gare_arrivee: str, resultat: dict):
    """Best-effort : publie les résultats sur le dataset HF pour que le
    prochain visiteur demandant le même corridor n'ait pas à tout
    recalculer. N'interrompt jamais l'analyse en cours si ça échoue (droit
    d'écriture absent/insuffisant, dataset indisponible...) — token=None
    laisse huggingface_hub résoudre l'authentification lui-même (variable
    HF_TOKEN sur le Space, ou session CLI déjà connectée en local)."""
    token = os.environ.get("HF_TOKEN")
    try:
        with tempfile.TemporaryDirectory() as dossier_tmp:
            gares_a_exporter = resultat["gares_corridor"].copy()
            gares_a_exporter["geometry_wkt"] = gares_a_exporter.geometry.to_wkt()
            gares_a_exporter.drop(columns="geometry").to_csv(
                os.path.join(dossier_tmp, "gares_corridor.csv"), index=False
            )
            resultat["communes_influence"].to_csv(os.path.join(dossier_tmp, "communes_influence.csv"), index=False)
            resultat["flux_corridor"].to_csv(os.path.join(dossier_tmp, "flux_corridor.csv"), index=False)
            resultat["population_corridor"].to_csv(os.path.join(dossier_tmp, "population_gares.csv"), index=False)
            resultat["charge_troncons"].to_csv(os.path.join(dossier_tmp, "charge_troncons.csv"), index=False)
            pd.DataFrame([{
                "distance_directe_km": resultat["distance_directe_km"],
                "ligne_voie_wkt": resultat["ligne_voie"].wkt,
                "gare_depart": gare_depart,
                "gare_arrivee": gare_arrivee,
            }]).to_csv(os.path.join(dossier_tmp, "meta.csv"), index=False)

            HfApi().upload_folder(
                repo_id=CACHE_DATASET_REPO, repo_type="dataset",
                folder_path=dossier_tmp, path_in_repo=f"{CACHE_PREFIX}/{nom_corridor}",
                token=token,
            )

        # Met à jour l'index global des corridors déjà mis en cache (liste
        # "Corridor déjà identifié" dans la sidebar) — fichier à part, à la
        # racine de CACHE_PREFIX, distinct des fichiers propres à ce corridor.
        try:
            chemin_index = hf_hub_download(
                repo_id=CACHE_DATASET_REPO, repo_type="dataset",
                filename=f"{CACHE_PREFIX}/index.csv", token=token,
            )
            index = pd.read_csv(chemin_index)
        except Exception:
            index = pd.DataFrame(columns=["nom_corridor", "gare_depart", "gare_arrivee"])
        index = index[index["nom_corridor"] != nom_corridor]
        nouvelle_ligne = pd.DataFrame([{
            "nom_corridor": nom_corridor, "gare_depart": gare_depart, "gare_arrivee": gare_arrivee,
        }])
        index = pd.concat([index, nouvelle_ligne], ignore_index=True)
        with tempfile.TemporaryDirectory() as dossier_tmp:
            chemin_local_index = os.path.join(dossier_tmp, "index.csv")
            index.to_csv(chemin_local_index, index=False)
            HfApi().upload_file(
                path_or_fileobj=chemin_local_index, path_in_repo=f"{CACHE_PREFIX}/index.csv",
                repo_id=CACHE_DATASET_REPO, repo_type="dataset", token=token,
            )
    except Exception as exc:
        st.caption(f"(cache partagé non mis à jour : {exc})")


@st.cache_data(ttl=300, show_spinner="Recherche des corridors déjà analysés...")
def lister_corridors_caches() -> pd.DataFrame:
    """Corridors déjà mis en cache par un visiteur précédent (index tenu à
    jour par sauvegarder_cache_hf) — alimente le sélecteur "Corridor déjà
    identifié" de la sidebar. TTL court : un nouveau corridor mis en cache
    par un autre visiteur doit apparaître sans attendre un redéploiement."""
    try:
        chemin = hf_hub_download(
            repo_id=CACHE_DATASET_REPO, repo_type="dataset",
            filename=f"{CACHE_PREFIX}/index.csv", token=os.environ.get("HF_TOKEN"),
        )
        return pd.read_csv(chemin)
    except Exception:
        return pd.DataFrame(columns=["nom_corridor", "gare_depart", "gare_arrivee"])


# --------------------------------------------------------------------------
# Routage réel sur le réseau ferré (OpenStreetMap)
# --------------------------------------------------------------------------

def requete_overpass(query: str, essais: int = 2) -> dict:
    headers = {"User-Agent": "corridor-analyse-fr/1.0 (contact: antoine.chevre@gmail.com)"}
    derniere_erreur = None
    for url in OVERPASS_MIRRORS:
        for _ in range(essais):
            try:
                r = requests.post(url, data={"data": query}, headers=headers, timeout=60)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                derniere_erreur = exc
                time.sleep(1)
    raise RuntimeError(f"Overpass indisponible sur tous les miroirs : {derniere_erreur}")


def _arrondir(coord, precision=0.5):
    return (round(coord[0] / precision) * precision, round(coord[1] / precision) * precision)


@st.cache_data(show_spinner="Récupération du tracé réel de la voie ferrée (OpenStreetMap)...")
def calculer_ligne_voie(gare_depart: str, gare_arrivee: str) -> LineString:
    """Plus court chemin réel entre les deux gares sur le réseau ferré
    (données OSM) : une simple distance à une voie ne suffit pas (une gare
    peut être aussi proche d'une ligne différente), donc on construit un
    graphe du réseau local et on calcule le vrai chemin, pas une
    approximation géométrique (ellipse/détour)."""
    gares = load_gares()
    ligne_depart = gares.loc[gares["nomGare"] == gare_depart].iloc[0]
    ligne_arrivee = gares.loc[gares["nomGare"] == gare_arrivee].iloc[0]
    sud = min(ligne_depart["wgs84Lat"], ligne_arrivee["wgs84Lat"]) - MARGE_BBOX_DEG
    nord = max(ligne_depart["wgs84Lat"], ligne_arrivee["wgs84Lat"]) + MARGE_BBOX_DEG
    ouest = min(ligne_depart["wgs84Lon"], ligne_arrivee["wgs84Lon"]) - MARGE_BBOX_DEG
    est = max(ligne_depart["wgs84Lon"], ligne_arrivee["wgs84Lon"]) + MARGE_BBOX_DEG

    data_osm = requete_overpass(f'[out:json][timeout:60];way["railway"="rail"]({sud},{ouest},{nord},{est});out geom;')

    lignes_osm, service_osm = [], []
    for el in data_osm["elements"]:
        if el["type"] != "way" or "geometry" not in el:
            continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2:
            continue
        lignes_osm.append(LineString(coords))
        service_osm.append(el.get("tags", {}).get("service"))

    voies = gpd.GeoDataFrame({"service": service_osm}, geometry=lignes_osm, crs="EPSG:4326").to_crs("EPSG:2154")
    voies_principales = voies[voies["service"].isna()]
    if voies_principales.empty:
        raise RuntimeError("Aucune voie ferrée OSM trouvée dans la zone du corridor.")

    troncons = unary_union(list(voies_principales.geometry))
    troncons = list(troncons.geoms) if hasattr(troncons, "geoms") else [troncons]

    reseau = nx.Graph()
    for troncon in troncons:
        points = list(troncon.coords)
        for a, b in zip(points[:-1], points[1:]):
            na, nb = _arrondir(a), _arrondir(b)
            longueur = Point(a).distance(Point(b))
            if reseau.has_edge(na, nb) and reseau[na][nb]["length"] <= longueur:
                continue
            reseau.add_edge(na, nb, length=longueur)

    gares_2154 = gares.to_crs("EPSG:2154")
    point_depart = gares_2154.loc[gares["nomGare"] == gare_depart, "geometry"].iloc[0]
    point_arrivee = gares_2154.loc[gares["nomGare"] == gare_arrivee, "geometry"].iloc[0]
    distance_vol_oiseau_km = point_depart.distance(point_arrivee) / 1000

    noeuds = list(reseau.nodes())
    points_noeuds = gpd.GeoSeries([Point(n) for n in noeuds], crs="EPSG:2154")

    def noeud_le_plus_proche(point):
        distances = points_noeuds.distance(point)
        idx = distances.idxmin()
        return noeuds[idx]

    noeud_depart = noeud_le_plus_proche(point_depart)
    noeud_arrivee = noeud_le_plus_proche(point_arrivee)
    if not nx.has_path(reseau, noeud_depart, noeud_arrivee):
        if distance_vol_oiseau_km > SEUIL_DISTANCE_ALERTE_KM:
            raise RuntimeError(
                f"{gare_depart} et {gare_arrivee} sont à {distance_vol_oiseau_km:.0f} km à vol d'oiseau "
                f"(> {SEUIL_DISTANCE_ALERTE_KM} km) et aucune voie ferroviaire continue ne les relie dans les "
                "données OpenStreetMap récupérées : vérifiez qu'il existe bien une ligne directe entre ces "
                "deux gares."
            )
        raise RuntimeError(
            f"Aucun chemin ferré continu trouvé entre {gare_depart} et {gare_arrivee} dans les données "
            "OpenStreetMap récupérées."
        )
    chemin_noeuds = nx.shortest_path(reseau, noeud_depart, noeud_arrivee, weight="length")
    return LineString(chemin_noeuds)


# --------------------------------------------------------------------------
# Pipeline corridor (détection, aire d'influence, flux, population, charge)
# --------------------------------------------------------------------------

def detecter_corridor(gares: gpd.GeoDataFrame, ligne_voie: LineString, tampon_voie_m: float):
    """Gares du corridor : celles à moins de `tampon_voie_m` du tracé réel,
    une seule par commune (la plus fréquentée si une agglomération en a
    plusieurs), ordonnées par position réelle le long de la voie."""
    gares_2154 = gares.to_crs("EPSG:2154")
    zone_voie = ligne_voie.buffer(tampon_voie_m)
    gares_2154 = gares_2154[gares_2154.geometry.within(zone_voie)].copy()
    gares_2154["distance_depart_km"] = gares_2154.geometry.apply(lambda g: ligne_voie.project(g) / 1000)
    gares_2154 = gares_2154.sort_values("distance_depart_km")

    frequentation_recente = load_frequentation_annee()[["codeUic", "voyageurs"]]
    gares_2154 = gares_2154.merge(frequentation_recente, on="codeUic", how="left")
    gares_2154["voyageurs"] = gares_2154["voyageurs"].fillna(0)
    gares_corridor = (
        gares_2154.sort_values("voyageurs", ascending=False)
        .drop_duplicates(subset="inseeCommune", keep="first")
        .sort_values("distance_depart_km")
        .reset_index(drop=True)
    )
    return gares_corridor


def calculer_aire_influence(gares_corridor: gpd.GeoDataFrame):
    isochrones = load_isochrones(ISOCHRONE_FILES["10 min en voiture"][0]).to_crs("EPSG:2154")

    isochrones_corridor = isochrones[isochrones["code_uic"].isin(gares_corridor["codeUic"])].merge(
        gares_corridor[["codeUic", "inseeCommune", "nomGare", "distance_depart_km", "geometry"]].rename(
            columns={"geometry": "point_gare", "inseeCommune": "gare_corridor_inseeCommune", "nomGare": "gare_corridor"}
        ),
        left_on="code_uic", right_on="codeUic",
    )

    departements_corridor = gares_corridor["inseeDepartement"].unique()
    communes_brutes = pd.concat(
        [load_communes_departement(dept) for dept in departements_corridor], ignore_index=True,
    )
    # Union de toutes les communes des départements traversés (pas juste
    # l'aire d'influence des gares) : sert à cadrer la couche carreaux
    # population sur la carte, cf. construire_carte.
    departements_geom = gpd.GeoSeries.from_wkt(communes_brutes["geometry_wkt"], crs="EPSG:4326").union_all()
    communes = gpd.GeoDataFrame(
        communes_brutes, geometry=gpd.GeoSeries.from_wkt(communes_brutes["geometry_wkt"]), crs="EPSG:4326",
    ).drop(columns="geometry_wkt").to_crs("EPSG:2154")

    communes_influence = []
    for _, gare_row in isochrones_corridor.iterrows():
        recoupe = communes[communes.intersects(gare_row["geometry"])].copy()
        recoupe["gare_corridor"] = gare_row["gare_corridor"]
        recoupe["gare_corridor_inseeCommune"] = gare_row["gare_corridor_inseeCommune"]
        recoupe["gare_corridor_position_km"] = gare_row["distance_depart_km"]
        recoupe["distance_a_la_gare_corridor_km"] = recoupe.geometry.centroid.distance(gare_row["point_gare"]) / 1000
        communes_influence.append(recoupe)
    communes_influence = pd.concat(communes_influence, ignore_index=True)
    # La commune d'une gare est toujours dans sa PROPRE isochrone (le point de
    # la gare y est), mais deux gares de corridor très proches (ex. Bordeaux/
    # Cenon) peuvent faire gagner la gare voisine sur le seul critère de
    # distance au centroïde — la commune d'une gare doit toujours lui rester
    # rattachée, sans quoi ses propres flux domicile-travail/études seraient
    # comptés sur la mauvaise gare (jusqu'à un faux "Cenon -> Cenon" après
    # fusion, deux communes distinctes devenant le même nœud par erreur).
    communes_influence["propre_gare"] = communes_influence["code"] == communes_influence["gare_corridor_inseeCommune"]
    communes_influence = communes_influence.sort_values(
        ["propre_gare", "distance_a_la_gare_corridor_km"], ascending=[False, True]
    ).drop_duplicates(subset="code", keep="first").drop(columns="propre_gare")
    communes_influence = communes_influence.rename(columns={"code": "inseeCommune", "nom": "nomCommune"})
    communes_influence = communes_influence.sort_values("gare_corridor_position_km")[
        ["gare_corridor_position_km", "gare_corridor", "gare_corridor_inseeCommune",
         "inseeCommune", "nomCommune", "distance_a_la_gare_corridor_km"]
    ].reset_index(drop=True)

    isochrones_corridor_completes = isochrones[isochrones["code_uic"].isin(gares_corridor["codeUic"])].merge(
        gares_corridor[["codeUic", "nomGare", "nomCommune", "distance_depart_km"]],
        left_on="code_uic", right_on="codeUic",
    )  # encore en EPSG:2154, nécessaire pour un Voronoï correct en mètres

    return communes_influence, isochrones_corridor_completes, departements_geom


def calculer_departements_geom_simple(gares_corridor: gpd.GeoDataFrame):
    """Reconstruit departements_geom (union des communes des départements
    traversés) depuis gares_corridor seul — utilisé sur le chemin de cache
    HF, qui ne conserve pas la géométrie des communes elle-même."""
    departements_corridor = gares_corridor["inseeDepartement"].unique()
    communes_brutes = pd.concat(
        [load_communes_departement(dept) for dept in departements_corridor], ignore_index=True,
    )
    return gpd.GeoSeries.from_wkt(communes_brutes["geometry_wkt"], crs="EPSG:4326").union_all()


def calculer_flux_corridor(communes_influence: pd.DataFrame) -> pd.DataFrame:
    fusion_codes = communes_influence.set_index("inseeCommune")["gare_corridor_inseeCommune"]
    fusion_labels = communes_influence.set_index("inseeCommune")["gare_corridor"]

    def fusionner_sur_gare(df):
        df = df.copy()
        for col_code, col_label in [("origine", "label_origine"), ("destination", "label_destination")]:
            df[col_code] = df[col_code].map(lambda c: fusion_codes.get(c, c))
            df[col_label] = df[col_code].map(fusion_labels).fillna(df[col_label])
        return df.groupby(
            ["origine", "label_origine", "destination", "label_destination"], as_index=False
        )["flux"].sum()

    gares_corridor_codes = set(communes_influence["gare_corridor_inseeCommune"])
    flux_corridor = pd.concat(
        [
            fusionner_sur_gare(load_flux(theme))[
                lambda df: df["origine"].isin(gares_corridor_codes) & df["destination"].isin(gares_corridor_codes)
            ].assign(theme=theme)
            for theme in ["travail", "etudes"]
        ],
        ignore_index=True,
    )
    return flux_corridor[flux_corridor["origine"] != flux_corridor["destination"]].reset_index(drop=True)


def calculer_aire_influence_totale_simple(gares_corridor: gpd.GeoDataFrame):
    """Union brute (sans partage Voronoï) des isochrones 10 min voiture des
    gares du corridor — sert uniquement à cadrer la couche carreaux
    population sur la carte, pas au calcul de population (qui a besoin du
    partage Voronoï pour éviter le double compte, cf. calculer_population_corridor).
    Ne dépend que de gares_corridor : reconstructible même depuis le cache
    HF, qui ne conserve pas isochrones_corridor_completes."""
    isochrones = load_isochrones(ISOCHRONE_FILES["10 min en voiture"][0])
    isochrones_corridor = isochrones[isochrones["code_uic"].isin(gares_corridor["codeUic"])]
    return isochrones_corridor.geometry.union_all()


def calculer_population_corridor(gares_corridor: gpd.GeoDataFrame, isochrones_corridor_completes: gpd.GeoDataFrame):
    """Population par gare, en partageant les zones qui se recoupent par la
    médiane (diagramme de Voronoï, en mètres) pour éviter le double compte."""
    carreaux_france = load_carreaux_france()

    points_gares = MultiPoint(list(gares_corridor.geometry))
    cellules_voronoi = list(voronoi_diagram(points_gares).geoms)
    cellule_par_gare = {
        gare["codeUic"]: next(c for c in cellules_voronoi if gare.geometry.within(c))
        for _, gare in gares_corridor.iterrows()
    }

    isochrones_corridor_completes = isochrones_corridor_completes.copy()
    isochrones_corridor_completes["geometry"] = isochrones_corridor_completes.apply(
        lambda row: row["geometry"].intersection(cellule_par_gare[row["code_uic"]]), axis=1
    )
    isochrones_corridor_completes = isochrones_corridor_completes.to_crs("EPSG:4326")  # carreaux_france est en WGS84

    def population_dans_isochrone(polygon):
        if polygon.is_empty:
            return 0
        minx, miny, maxx, maxy = polygon.bounds
        sous = carreaux_france.cx[minx:maxx, miny:maxy]
        if sous.empty:
            return 0
        return sous[sous.intersects(polygon)]["pop"].sum()

    isochrones_corridor_completes["population_10min_voiture"] = isochrones_corridor_completes["geometry"].apply(
        population_dans_isochrone
    )
    population_corridor = isochrones_corridor_completes[
        ["distance_depart_km", "nomGare", "nomCommune", "population_10min_voiture"]
    ].sort_values("distance_depart_km").reset_index(drop=True)

    aire_influence_totale = isochrones_corridor_completes.union_all()
    return population_corridor, aire_influence_totale


def calculer_charge_troncons(gares_corridor: gpd.GeoDataFrame, df_flux: pd.DataFrame) -> pd.DataFrame:
    """Charge cumulée par tronçon (segment entre deux gares consécutives) :
    somme des flux dont le trajet traverse ce tronçon — même principe que
    creer_carte_troncons dans github.com/antoinechevre/GTFS_analysis_fr,
    appliqué aux flux INSEE plutôt qu'aux passages GTFS."""
    positions_gares = gares_corridor.set_index("inseeCommune")["distance_depart_km"]
    gares_ordonnees = gares_corridor.sort_values("distance_depart_km").reset_index(drop=True)

    lignes = []
    for i in range(len(gares_ordonnees) - 1):
        gare_a, gare_b = gares_ordonnees.iloc[i], gares_ordonnees.iloc[i + 1]
        pos_a, pos_b = gare_a["distance_depart_km"], gare_b["distance_depart_km"]
        pos_origine = df_flux["origine"].map(positions_gares)
        pos_destination = df_flux["destination"].map(positions_gares)
        borne_min = pd.concat([pos_origine, pos_destination], axis=1).min(axis=1)
        borne_max = pd.concat([pos_origine, pos_destination], axis=1).max(axis=1)
        traverse = (borne_min <= pos_a) & (borne_max >= pos_b)
        lignes.append({
            "troncon": f"{gare_a['nomGare']} — {gare_b['nomGare']}",
            "position_depart_km": pos_a,
            "position_arrivee_km": pos_b,
            "charge": df_flux.loc[traverse, "flux"].sum(),
        })
    return pd.DataFrame(lignes)


@st.cache_data(show_spinner="Analyse du corridor...")
def analyser_corridor(gare_depart: str, gare_arrivee: str, tampon_voie_m: float):
    nom_corridor = nom_corridor_de(gare_depart, gare_arrivee)

    depuis_cache = charger_depuis_cache_hf(nom_corridor)
    if depuis_cache is not None:
        depuis_cache["depuis_cache"] = True
        return depuis_cache

    gares = load_gares()
    ligne_voie = calculer_ligne_voie(gare_depart, gare_arrivee)
    distance_directe_km = ligne_voie.length / 1000

    gares_corridor = detecter_corridor(gares, ligne_voie, tampon_voie_m)
    communes_influence, isochrones_corridor_completes, departements_geom = calculer_aire_influence(gares_corridor)
    flux_corridor = calculer_flux_corridor(communes_influence)
    population_corridor, aire_influence_totale = calculer_population_corridor(
        gares_corridor, isochrones_corridor_completes
    )
    flux_cumul_od = flux_corridor.groupby(
        ["origine", "label_origine", "destination", "label_destination"], as_index=False
    )["flux"].sum()
    charge_troncons = calculer_charge_troncons(gares_corridor, flux_cumul_od)

    resultat = {
        "gares_corridor": gares_corridor,
        "ligne_voie": ligne_voie,
        "distance_directe_km": distance_directe_km,
        "communes_influence": communes_influence,
        "flux_corridor": flux_corridor,
        "population_corridor": population_corridor,
        "aire_influence_totale": aire_influence_totale,
        "departements_geom": departements_geom,
        "charge_troncons": charge_troncons,
        "depuis_cache": False,
    }
    sauvegarder_cache_hf(nom_corridor, gare_depart, gare_arrivee, resultat)
    return resultat


# --------------------------------------------------------------------------
# Rendu de la carte HTML (couches sélectionnables, mêmes briques que app.py)
# --------------------------------------------------------------------------

def offre_pie_svg(gare_offre, rayon_px: float) -> str:
    total = gare_offre["totalClasse"]
    stops = []
    cumule = 0.0
    for col, (_, couleur) in OFFRE_CATEGORIES.items():
        valeur = gare_offre[col]
        if valeur <= 0:
            continue
        debut = cumule / total * 360
        cumule += valeur
        fin = cumule / total * 360
        stops.append(f"{couleur} {debut:.1f}deg {fin:.1f}deg")
    taille = rayon_px * 2
    return (
        f'<div style="width:{taille:.0f}px;height:{taille:.0f}px;border-radius:50%;'
        f'background:conic-gradient({", ".join(stops)});'
        f'border:1px solid rgba(0,0,0,0.5);box-shadow:0 0 3px rgba(0,0,0,0.35);"></div>'
    )


def offre_popup(gare_offre) -> str:
    lignes = [f"<b>{gare_offre['nomGare']}</b>"]
    for col, (label, _) in OFFRE_CATEGORIES.items():
        if gare_offre[col] > 0:
            lignes.append(f"{label} : {int(gare_offre[col])} trains/jour")
    lignes.append(f"<b>Total : {int(gare_offre['totalClasse'])} trains/jour</b>")
    return "<br>".join(lignes)


def frequentation_bubble_svg(rayon_px: float) -> str:
    taille = rayon_px * 2
    return (
        f'<div style="width:{taille:.0f}px;height:{taille:.0f}px;border-radius:50%;'
        f'background:{FREQUENTATION_COLOR};opacity:0.7;'
        f'border:1px solid rgba(0,0,0,0.5);box-shadow:0 0 3px rgba(0,0,0,0.35);"></div>'
    )


def bezier_arc(origine: tuple, destination: tuple, courbure: float, n: int = 24) -> list:
    lat1, lon1 = origine
    lat2, lon2 = destination
    dlat, dlon = lat2 - lat1, lon2 - lon1
    longueur = math.hypot(dlat, dlon) or 1e-9
    perp_lat, perp_lon = -dlon / longueur, dlat / longueur
    ctrl_lat = (lat1 + lat2) / 2 + perp_lat * longueur * courbure
    ctrl_lon = (lon1 + lon2) / 2 + perp_lon * longueur * courbure

    points = []
    for i in range(n + 1):
        t = i / n
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * ctrl_lat + t**2 * lat2
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * ctrl_lon + t**2 * lon2
        points.append([lat, lon])
    return points


def arrowhead_svg(bearing_deg: float, couleur: str, taille_px: int = 14) -> str:
    return (
        f'<div style="width:0;height:0;'
        f"border-left:{taille_px // 2}px solid transparent;"
        f"border-right:{taille_px // 2}px solid transparent;"
        f"border-bottom:{taille_px}px solid {couleur};"
        f'transform:rotate({bearing_deg}deg);transform-origin:50% 50%;"></div>'
    )


def angle_entre_points(p1: tuple, p2: tuple) -> float:
    lat1, lon1 = p1
    lat2, lon2 = p2
    return math.degrees(math.atan2(lon2 - lon1, lat2 - lat1))


def formater_colonnes_entieres(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    """Copie d'affichage avec ces colonnes en entiers, espace comme
    séparateur de milliers (ex. "25 000") — les données sous-jacentes
    (CSV téléchargés, calculs) restent des nombres, seul l'affichage change."""
    df = df.copy()
    for colonne in colonnes:
        df[colonne] = df[colonne].map(lambda v: f"{v:,.0f}".replace(",", " "))
    return df


def formater_colonnes_distance(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    """Copie d'affichage avec ces colonnes (km) à une décimale."""
    df = df.copy()
    for colonne in colonnes:
        df[colonne] = df[colonne].map(lambda v: f"{v:.1f}")
    return df


# Bouton d'export PNG de la carte, reflétant exactement les couches
# actuellement affichées (celles cochées dans le LayerControl) — via
# leaflet-image (rasterise les tuiles + calques visibles dans un canvas).
# cross_origin=True sur les TileLayer (cf. plus bas) est nécessaire pour
# que ça fonctionne sans "tainter" le canvas.
LIB_LEAFLET_IMAGE = '<script src="https://cdn.jsdelivr.net/npm/leaflet-image@0.4.0/leaflet-image.js"></script>'

# Ce bloc référence la variable JS de la carte ({nom_carte} = L.map(...)),
# mais l'ordre exact des <script> générés par folium ne garantit pas que
# cette ligne s'exécute avant la nôtre (elle peut même passer après) — d'où
# le DOMContentLoaded : par définition, il ne se déclenche qu'une fois tout
# le JS de la page déjà exécuté, {nom_carte} est donc forcément assigné.
# Sans ça, le bouton n'apparaissait pas (TypeError silencieuse sur
# `undefined.addControl`, la variable valant encore `undefined`).
SCRIPT_EXPORT_PNG = """
document.addEventListener('DOMContentLoaded', function() {{
    var carte = {nom_carte};
    var BoutonExport = L.Control.extend({{
        options: {{position: 'topright'}},
        onAdd: function(map) {{
            var bouton = L.DomUtil.create('button');
            bouton.innerHTML = '📷 Export PNG';
            bouton.title = "Exporte la carte telle qu'affichée (couches cochées) en PNG";
            bouton.style.cssText = 'background:white;padding:6px 10px;cursor:pointer;font-size:13px;' +
                'border:2px solid rgba(0,0,0,0.2);border-radius:4px;';
            bouton.onclick = function(e) {{
                L.DomEvent.stopPropagation(e);
                bouton.innerHTML = '⏳ Export...';
                leafletImage(map, function(err, canvas) {{
                    bouton.innerHTML = '📷 Export PNG';
                    if (err) {{ alert("Export impossible : " + err); return; }}
                    var lien = document.createElement('a');
                    lien.download = 'carte_corridor.png';
                    lien.href = canvas.toDataURL('image/png');
                    document.body.appendChild(lien);
                    lien.click();
                    document.body.removeChild(lien);
                }});
            }};
            return bouton;
        }},
    }});
    carte.addControl(new BoutonExport());
}});
"""


def construire_carte(resultat: dict, seuil_min_flux: float) -> folium.Map:
    gares_corridor = resultat["gares_corridor"]
    aire_influence_totale = resultat["aire_influence_totale"]
    departements_geom = resultat["departements_geom"]
    flux_corridor = resultat["flux_corridor"]
    charge_troncons = resultat["charge_troncons"]
    ligne_voie = resultat["ligne_voie"]
    carreaux_france = load_carreaux_france()

    # La couche population couvre tous les départements traversés par le
    # corridor (pas seulement l'aire d'influence des gares), pour donner le
    # contexte démographique complet autour de la ligne.
    minx_dept, miny_dept, maxx_dept, maxy_dept = departements_geom.bounds
    carreaux_corridor = carreaux_france.cx[minx_dept:maxx_dept, miny_dept:maxy_dept]
    carreaux_corridor = carreaux_corridor[carreaux_corridor.intersects(departements_geom)]

    # Le cadrage initial de la carte reste celui du corridor lui-même (pas
    # les départements entiers, sinon la ligne devient minuscule à l'écran).
    minx, miny, maxx, maxy = aire_influence_totale.bounds
    centre_carte = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=centre_carte, tiles=None, prefer_canvas=True, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", cross_origin=True).add_to(m)
    folium.TileLayer(
        carto_tile_url("light_all"), name="CartoDB Positron", attr=CARTO_ATTR, cross_origin=True,
    ).add_to(m)
    folium.TileLayer(
        carto_tile_url("dark_all"), name="CartoDB Dark Matter", attr=CARTO_ATTR, cross_origin=True,
    ).add_to(m)

    if not carreaux_corridor.empty:
        vmin = float(carreaux_corridor["pop"].min())
        vmax = float(carreaux_corridor["pop"].max())
        colormap = folium.LinearColormap(
            colors=CARREAUX_COLOR_SCALE, vmin=vmin, vmax=vmax, caption="Population (carreau 200m)"
        )

        def style_carreau(feature, lo=vmin, hi=vmax):
            value = feature["properties"]["pop"]
            if hi > lo:
                frac = ((value - lo) / (hi - lo)) ** CARREAUX_COLOR_GAMMA
                value = lo + frac * (hi - lo)
            return {"fillColor": colormap(value), "color": "#581012", "weight": 0, "fillOpacity": 1.0}

        folium.GeoJson(
            carreaux_corridor[["pop", "geometry"]],
            name="Carreaux INSEE 200m (population)",
            style_function=style_carreau,
            tooltip=folium.GeoJsonTooltip(fields=["pop"], aliases=["Population"], localize=True),
        ).add_to(m)
        colormap.add_to(m)

    codes_uic_corridor = set(gares_corridor["codeUic"])
    for mode, (path, couleur) in ISOCHRONE_FILES.items():
        gdf = load_isochrones(path)
        gdf = gdf[gdf["code_uic"].isin(codes_uic_corridor)]
        if gdf.empty:
            continue
        folium.GeoJson(
            gdf, name=mode,
            style_function=lambda feature, c=couleur: {
                "color": c, "weight": 2, "fill": True, "fillColor": c, "fillOpacity": 0.3,
            },
        ).add_to(m)

    frequentation = load_frequentation_annee()
    frequentation_corridor = gares_corridor[["codeUic", "wgs84Lat", "wgs84Lon"]].merge(
        frequentation, on="codeUic", how="inner"
    )
    if not frequentation_corridor.empty:
        freq_max = frequentation_corridor["voyageurs"].max()
        freq_layer = folium.FeatureGroup(name=f"Fréquentation {FREQUENTATION_ANNEE} (voyageurs/an)", show=False)
        for _, gare_freq in frequentation_corridor.iterrows():
            rayon = FREQUENTATION_MIN_RADIUS_PX + (FREQUENTATION_MAX_RADIUS_PX - FREQUENTATION_MIN_RADIUS_PX) * math.sqrt(
                gare_freq["voyageurs"] / freq_max
            )
            folium.Marker(
                [gare_freq["wgs84Lat"], gare_freq["wgs84Lon"]],
                icon=folium.DivIcon(
                    html=frequentation_bubble_svg(rayon), icon_size=(rayon * 2, rayon * 2), icon_anchor=(rayon, rayon)
                ),
                tooltip=(
                    f"{gare_freq['nomGare']} — {int(gare_freq['voyageurs']):,} voyageurs/an "
                    f"({FREQUENTATION_ANNEE})"
                ).replace(",", " "),
            ).add_to(freq_layer)
        freq_layer.add_to(m)

    offre = load_offre_2026()
    offre_corridor = gares_corridor[["codeUic", "wgs84Lat", "wgs84Lon"]].merge(offre, on="codeUic", how="inner")
    if not offre_corridor.empty:
        offre_max = offre_corridor["totalClasse"].max()
        offre_layer = folium.FeatureGroup(name="Offre 2026 (TER / Intercités / TGV)", show=False)
        for _, gare_offre in offre_corridor.iterrows():
            rayon = OFFRE_MIN_RADIUS_PX + (OFFRE_MAX_RADIUS_PX - OFFRE_MIN_RADIUS_PX) * math.sqrt(
                gare_offre["totalClasse"] / offre_max
            )
            folium.Marker(
                [gare_offre["wgs84Lat"], gare_offre["wgs84Lon"]],
                icon=folium.DivIcon(
                    html=offre_pie_svg(gare_offre, rayon), icon_size=(rayon * 2, rayon * 2), icon_anchor=(rayon, rayon)
                ),
                tooltip=offre_popup(gare_offre),
            ).add_to(offre_layer)
        offre_layer.add_to(m)

    coords_gare_corridor = gares_corridor.set_index("inseeCommune")[["wgs84Lat", "wgs84Lon"]]
    flux_cumul_od = flux_corridor.groupby(
        ["origine", "label_origine", "destination", "label_destination"], as_index=False
    )["flux"].sum()
    flux_layers = {
        "travail": (flux_corridor[flux_corridor["theme"] == "travail"], FLUX_COULEURS["travail"], "Flèches domicile-travail"),
        "etudes": (flux_corridor[flux_corridor["theme"] == "etudes"], FLUX_COULEURS["etudes"], "Flèches domicile-études"),
        "cumul": (flux_cumul_od, FLUX_COULEURS["cumul"], "Flèches cumulées domicile-travail + domicile-études"),
    }
    for theme, (df_theme, couleur, nom_calque) in flux_layers.items():
        df_theme = df_theme[df_theme["flux"] >= seuil_min_flux]
        if df_theme.empty:
            continue
        flux_max = df_theme["flux"].max()
        couche = folium.FeatureGroup(name=nom_calque, show=(theme == "cumul"))
        for _, flux in df_theme.iterrows():
            origine_latlon = tuple(coords_gare_corridor.loc[flux["origine"]])
            destination_latlon = tuple(coords_gare_corridor.loc[flux["destination"]])
            poids = FLUX_MIN_WEIGHT_PX + (FLUX_MAX_WEIGHT_PX - FLUX_MIN_WEIGHT_PX) * math.sqrt(flux["flux"] / flux_max)
            courbe = bezier_arc(origine_latlon, destination_latlon, courbure=COURBURE_PAR_THEME[theme])
            folium.PolyLine(
                courbe, color=couleur, weight=poids, opacity=0.75,
                tooltip=f"{flux['label_origine']} → {flux['label_destination']} — {flux['flux']:.0f} personnes",
            ).add_to(couche)
            angle = angle_entre_points(courbe[-2], courbe[-1])
            taille = poids + 8
            folium.Marker(
                courbe[-1],
                icon=folium.DivIcon(
                    html=arrowhead_svg(angle, couleur, taille_px=int(taille)),
                    icon_size=(taille, taille), icon_anchor=(taille / 2, taille / 2),
                ),
            ).add_to(couche)
        couche.add_to(m)

    if not charge_troncons.empty and charge_troncons["charge"].max() > 0:
        charge_min = charge_troncons["charge"].min()
        charge_max = charge_troncons["charge"].max()
        colormap_charge = folium.LinearColormap(
            colors=CHARGE_COLOR_SCALE, vmin=charge_min, vmax=charge_max,
            caption="Charge cumulée par tronçon (domicile-travail + domicile-études)",
        )
        couche_charge = folium.FeatureGroup(name="Charge cumulée par tronçon", show=False)
        for _, troncon in charge_troncons.iterrows():
            segment = substring(ligne_voie, troncon["position_depart_km"] * 1000, troncon["position_arrivee_km"] * 1000)
            segment_wgs84 = gpd.GeoSeries([segment], crs="EPSG:2154").to_crs("EPSG:4326").iloc[0]
            coords = [(lat, lon) for lon, lat in segment_wgs84.coords]
            poids = (
                6 + (troncon["charge"] - charge_min) / (charge_max - charge_min) * 16
                if charge_max > charge_min else 8
            )
            folium.PolyLine(
                coords, color=colormap_charge(troncon["charge"]), weight=poids, opacity=0.85,
                tooltip=f"{troncon['troncon']} — {troncon['charge']:.0f} personnes (cumul)",
            ).add_to(couche_charge)
        couche_charge.add_to(m)
        colormap_charge.add_to(m)

    couche_gares = folium.FeatureGroup(name="Gares du corridor")
    for _, gare in gares_corridor.iterrows():
        folium.CircleMarker(
            [gare["wgs84Lat"], gare["wgs84Lon"]], radius=4, color="#000", fill=True, fill_opacity=1,
            tooltip=gare["nomGare"],
        ).add_to(couche_gares)
    couche_gares.add_to(m)

    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().header.add_child(folium.Element(LIB_LEAFLET_IMAGE))
    m.get_root().script.add_child(folium.Element(SCRIPT_EXPORT_PNG.format(nom_carte=m.get_name())))
    return m


# --------------------------------------------------------------------------
# App Streamlit
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Corridor Analyse", page_icon="🚉", layout="wide")
    st.markdown(
        """<style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
            scrollbar-gutter: stable;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("🚉 Corridor Analyse — flux et desserte entre deux gares")
    st.caption(
        "Choisissez deux gares Géofer : l'app suit la vraie ligne de chemin de fer (OpenStreetMap) entre les "
        "deux, calcule la population desservie, les flux domicile-travail/domicile-études et la charge par "
        "tronçon, et affiche une carte détaillée."
    )

    gares = load_gares()
    options_gares = gares["nomGare"].drop_duplicates().sort_values().tolist()

    corridors_caches = lister_corridors_caches()

    def _appliquer_corridor_cache():
        choix = st.session_state.get("corridor_cache_select")
        if choix and choix in corridors_par_label:
            depart, arrivee = corridors_par_label[choix]
            st.session_state["gare_depart_select"] = depart
            st.session_state["gare_arrivee_select"] = arrivee

    with st.sidebar:
        st.header("Corridor")
        if not corridors_caches.empty:
            corridors_par_label = {
                f"{ligne.gare_depart} → {ligne.gare_arrivee}": (ligne.gare_depart, ligne.gare_arrivee)
                for ligne in corridors_caches.itertuples()
            }
            st.selectbox(
                "Corridor déjà identifié", list(corridors_par_label), index=None,
                placeholder="— ou choisir un corridor déjà analysé —",
                key="corridor_cache_select", on_change=_appliquer_corridor_cache,
                help="Recharge instantanément un corridor déjà analysé par un précédent visiteur.",
            )
        gare_depart = st.selectbox(
            "Gare 1 (départ)", options_gares, index=None, placeholder="Choisir une gare", key="gare_depart_select",
        )
        gare_arrivee = st.selectbox(
            "Gare 2 (arrivée)", options_gares, index=None, placeholder="Choisir une gare", key="gare_arrivee_select",
        )
        tampon_voie_m = st.slider(
            "Tampon autour de la voie réelle (m)", 100, 1000, TAMPON_VOIE_M_DEFAUT, step=50,
            help="Distance maximale à la voie ferrée réelle pour qu'une gare soit retenue dans le corridor.",
        )
        seuil_min_flux = st.slider(
            "Seuil minimum de représentation d'un flux DT/DE", 0, 50, 10,
            help="Masque les flèches (et n'affiche pas) les flux domicile-travail/domicile-études inférieurs à ce seuil.",
        )
        nb_top_od = st.slider("Nombre d'origines-destinations affichées", 5, 30, 10)

    if not gare_depart or not gare_arrivee:
        st.info("Choisissez les deux gares du corridor dans le menu de gauche.")
        return
    if gare_depart == gare_arrivee:
        st.warning("Choisissez deux gares différentes.")
        return

    try:
        resultat = analyser_corridor(gare_depart, gare_arrivee, tampon_voie_m)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if resultat.get("depuis_cache"):
        st.caption("♻️ Résultats rechargés depuis le cache partagé (déjà analysé précédemment).")

    gares_corridor = resultat["gares_corridor"]
    population_corridor = resultat["population_corridor"]
    flux_corridor = resultat["flux_corridor"]
    charge_troncons = resultat["charge_troncons"]
    nom_corridor = nom_corridor_de(gare_depart, gare_arrivee)

    onglet_resultats, onglet_carte = st.tabs(["📊 Résultats", "🗺️ Carte"])

    with onglet_resultats:
        st.header(f"Corridor {gare_depart} – {gare_arrivee}")
        st.caption(f"{resultat['distance_directe_km']:.1f} km le long de la voie")

        gares_intermediaires = gares_corridor[
            ~gares_corridor["nomGare"].isin([gare_depart, gare_arrivee])
        ]["nomGare"].tolist()
        st.subheader(f"Gares intermédiaires ({len(gares_intermediaires)})")
        st.write(", ".join(gares_intermediaires) if gares_intermediaires else "Aucune")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Population desservie (10 min voiture, sans double comptage)",
            f"{int(population_corridor['population_10min_voiture'].sum()):,}".replace(",", " "),
        )
        flux_travail_total = flux_corridor.loc[flux_corridor["theme"] == "travail", "flux"].sum()
        flux_etudes_total = flux_corridor.loc[flux_corridor["theme"] == "etudes", "flux"].sum()
        col2.metric("Somme domicile-travail", f"{flux_travail_total:,.0f}".replace(",", " "))
        col3.metric("Somme domicile-études", f"{flux_etudes_total:,.0f}".replace(",", " "))

        st.subheader("Population par gare (aire d'influence 10 min voiture)")
        population_affichee = formater_colonnes_entieres(population_corridor, ["population_10min_voiture"])
        population_affichee = formater_colonnes_distance(population_affichee, ["distance_depart_km"])
        st.dataframe(population_affichee, width="stretch", hide_index=True)

        st.subheader("Charge cumulée par tronçon")
        charge_affichee = formater_colonnes_entieres(charge_troncons, ["charge"])
        charge_affichee = formater_colonnes_distance(charge_affichee, ["position_depart_km", "position_arrivee_km"])
        st.dataframe(charge_affichee, width="stretch", hide_index=True)

        st.subheader(f"Top {nb_top_od} origines-destinations (cumul domicile-travail + domicile-études)")
        flux_cumul_od = flux_corridor.groupby(
            ["origine", "label_origine", "destination", "label_destination"], as_index=False
        )["flux"].sum().sort_values("flux", ascending=False)
        top_od = flux_cumul_od.head(nb_top_od).reset_index(drop=True)
        st.dataframe(formater_colonnes_entieres(top_od, ["flux"]), width="stretch", hide_index=True)

        st.subheader("Téléchargements")
        frequentation_historique = load_frequentation_historique()
        frequentation_export = frequentation_historique[
            frequentation_historique["codeUic"].isin(gares_corridor["codeUic"])
        ].merge(
            gares_corridor[["codeUic", "distance_depart_km"]], on="codeUic",
        ).sort_values(["distance_depart_km", "annee"])

        dl1, dl2, dl3, dl4 = st.columns(4)
        dl1.download_button(
            "Fréquentation par gare (.csv)",
            data=frequentation_export.to_csv(index=False).encode("utf-8"),
            file_name=f"{nom_corridor}_frequentation_gares.csv", mime="text/csv",
        )
        dl2.download_button(
            "Population par gare (.csv)",
            data=population_corridor.to_csv(index=False).encode("utf-8"),
            file_name=f"{nom_corridor}_population_gares.csv", mime="text/csv",
        )
        dl3.download_button(
            "Flux domicile-travail/études (.csv)",
            data=flux_corridor.to_csv(index=False).encode("utf-8"),
            file_name=f"{nom_corridor}_flux.csv", mime="text/csv",
        )
        dl4.download_button(
            "Charge par tronçon (.csv)",
            data=charge_troncons.to_csv(index=False).encode("utf-8"),
            file_name=f"{nom_corridor}_charge_troncons.csv", mime="text/csv",
        )

    with onglet_carte:
        carte = construire_carte(resultat, seuil_min_flux)
        st.iframe(carte.get_root().render(), height=750)
        st.download_button(
            "Télécharger la carte (.html)",
            data=carte.get_root().render().encode("utf-8"),
            file_name=f"{nom_corridor}_carte.html", mime="text/html",
        )
        st.caption(
            "L'export PNG (bouton 📷 en haut à droite de la carte) capture exactement les couches "
            "actuellement cochées."
        )


if __name__ == "__main__":
    main()

"""Isochrone GTFS — accessibilité en transport collectif à l'heure de pointe.

Choisit un réseau GTFS parmi ceux stockés dans le dataset HF
antoinechevre/accessibility-data, détermine son JOB (jour ouvré de base,
même logique que antoinechevre/Accessibility_analysis : dernier mardi ou
jeudi avec un niveau de service fiable — sans le repli "hors vacances
scolaires" de ce projet, qui nécessite une géolocalisation académique hors
périmètre ici), puis calcule depuis un arrêt de départ choisi les arrêts
atteignables en transport collectif à une heure de pointe donnée, dans un
budget de temps donné.

Le calcul de correspondance est un RAPTOR simplifié limité aux arrêts
référencés dans le GTFS : à chaque tour, on ne considère que les trajets
passant par les arrêts atteints au tour précédent, avec un nombre de
correspondances plafonné — et uniquement des correspondances au même
arrêt (pas de correspondance piétonne entre deux arrêts distincts proches,
hors périmètre de cette version).

Chaque arrêt atteint est habillé d'un isochrone piéton (API
Isochrone/Isodistance de la Géoplateforme IGN, data.geopf.fr) pour donner
un rendu de zone de chalandise plutôt qu'un simple nuage de points.
"""

import os
import time
from datetime import datetime, time as dt_time

import folium
import gtfs_kit as gk
import pandas as pd
import requests
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download

HF_DATA_REPO_ID = "antoinechevre/accessibility-data"
GTFS_PREFIX = "GTFS/"

# CARTO exige désormais une clé API sur ses fonds raster (sinon un filigrane
# "API KEY REQUIRED" recouvre les tuiles) : chargée depuis le secret
# CARTO_API_KEY plutôt que codée en dur, ce fichier étant public.
CARTO_API_KEY = os.environ.get("CARTO_API_KEY")
CARTO_ATTR = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
    '&copy; <a href="https://carto.com/attributions">CARTO</a>'
)


def carto_tile_url(variant: str) -> str:
    url = f"https://{{s}}.basemaps.cartocdn.com/{variant}/{{z}}/{{x}}/{{y}}{{r}}.png"
    return f"{url}?key={CARTO_API_KEY}" if CARTO_API_KEY else url

GEOPF_ISOCHRONE_URL = "https://data.geopf.fr/navigation/isochrone"
GEOPF_MAX_REQ_PER_SEC = 5  # limite documentée de l'API Géoplateforme
GEOPF_MAX_STOPS_CALLED = 150  # au-delà, repli sur un cercle approximatif (pas d'appel API)

TRANSFER_BUFFER_SECONDS = 120  # temps de correspondance minimal au même arrêt

DEFAULT_BUDGET_MIN = 30
DEFAULT_MAX_TRANSFERS = 1
DEFAULT_WALK_BUFFER_MIN = 5

# Charte visuelle reprise de app.py (même charte que geofer.cerema.fr)
GEOFER_PRIMARY = "#1992D4"
GEOFER_SURFACE_GROUND = "#eff3f8"
GEOFER_TEXT_COLOR = "#495057"
DUREE_COLOR_SCALE = [
    "#084594", "#2171b5", "#4292c6", "#6baed6",
    "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff",
]

CUSTOM_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{
    font-family: 'Lato', Helvetica, sans-serif;
    color: {GEOFER_TEXT_COLOR};
}}
[data-testid="stSidebar"] {{
    background-color: {GEOFER_SURFACE_GROUND};
}}
h1, h2, h3 {{
    color: {GEOFER_PRIMARY};
    font-weight: 900;
}}
[data-testid="stMetricValue"] {{
    color: {GEOFER_PRIMARY};
}}
</style>
"""


def to_seconds(hhmmss: str) -> int:
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def to_hhmm(total_seconds: float) -> str:
    total_seconds = int(total_seconds) % 86400
    return f"{total_seconds // 3600:02d}h{(total_seconds % 3600) // 60:02d}"


@st.cache_data(show_spinner="Recherche des réseaux GTFS disponibles...")
def list_gtfs_networks() -> list[tuple[str, str]]:
    """Liste (label, chemin_hf) des GTFS sous GTFS/ dans le dataset HF."""
    fichiers = HfApi().list_repo_files(HF_DATA_REPO_ID, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    gtfs_files = sorted(f for f in fichiers if f.startswith(GTFS_PREFIX) and f.lower().endswith(".zip"))
    labels = [(f[len(GTFS_PREFIX):].rsplit(".", 1)[0].replace("_", " "), f) for f in gtfs_files]
    return sorted(labels)


@st.cache_resource(show_spinner="Téléchargement du GTFS...")
def download_gtfs(hf_path: str) -> str:
    return hf_hub_download(
        repo_id=HF_DATA_REPO_ID,
        repo_type="dataset",
        filename=hf_path,
        token=os.environ.get("HF_TOKEN"),
    )


@st.cache_resource(show_spinner="Chargement du GTFS...")
def load_feed(local_zip_path: str):
    return gk.read_feed(local_zip_path, dist_units="km")


@st.cache_data(show_spinner="Détermination du JOB (jour ouvré de base)...")
def determiner_job(_feed, hf_path_reseau: str) -> str:
    """Dernier mardi ou jeudi avec un niveau de service fiable (>= 70% du
    nombre de trips/jour maximum observé sur le GTFS)."""
    dates = _feed.get_dates()
    activite = _feed.compute_trip_activity(dates)
    comptages = activite.set_index("trip_id")[dates].sum()
    seuil = 0.7 * comptages.max()
    dates_fiables = comptages[comptages >= seuil].index.tolist()

    dates_parsees = [datetime.strptime(d, "%Y%m%d") for d in dates_fiables]
    mar_jeu = [d for d in dates_parsees if d.weekday() in (1, 3)]
    if mar_jeu:
        return max(mar_jeu).strftime("%Y%m%d")
    return max(dates_parsees).strftime("%Y%m%d")


@st.cache_data(show_spinner="Calcul des arrêts atteignables...")
def calculer_arrets_atteignables(
    _feed, hf_path_reseau: str, date_job: str, origin_stop_id: str,
    depart_s: int, budget_min: int, max_correspondances: int,
) -> pd.DataFrame:
    """RAPTOR simplifié : à chaque tour, ne considère que les trajets
    passant par les arrêts atteints au tour précédent (frontière),
    correspondances au même arrêt uniquement (cf. docstring du module)."""
    activite = _feed.compute_trip_activity([date_job])
    trip_ids_actifs = set(activite.loc[activite[date_job] == 1, "trip_id"])

    st_df = _feed.stop_times[_feed.stop_times["trip_id"].isin(trip_ids_actifs)].copy()
    st_df = st_df.dropna(subset=["departure_time", "arrival_time"])
    st_df["dep_s"] = st_df["departure_time"].map(to_seconds)
    st_df["arr_s"] = st_df["arrival_time"].map(to_seconds)
    st_df = st_df.sort_values(["trip_id", "stop_sequence"])

    trajets = {
        trip_id: list(zip(grp["stop_id"], grp["dep_s"], grp["arr_s"]))
        for trip_id, grp in st_df.groupby("trip_id")
    }
    trips_par_arret = st_df.groupby("stop_id")["trip_id"].unique().to_dict()

    max_arrivee = depart_s + budget_min * 60
    arrivee_au_plus_tot = {origin_stop_id: depart_s}
    correspondances_utilisees = {origin_stop_id: 0}
    frontiere = {origin_stop_id}

    for tour in range(max_correspondances + 1):
        trips_a_scanner: set = set()
        for arret in frontiere:
            trips_a_scanner.update(trips_par_arret.get(arret, []))

        nouvelle_frontiere = set()
        for trip_id in trips_a_scanner:
            embarque = False
            for stop_id, dep_s, arr_s in trajets[trip_id]:
                if not embarque:
                    if stop_id in frontiere:
                        tampon = TRANSFER_BUFFER_SECONDS if correspondances_utilisees[stop_id] > 0 else 0
                        if dep_s >= arrivee_au_plus_tot[stop_id] + tampon:
                            embarque = True
                    continue
                if arr_s <= max_arrivee and arr_s < arrivee_au_plus_tot.get(stop_id, max_arrivee + 1):
                    arrivee_au_plus_tot[stop_id] = arr_s
                    correspondances_utilisees[stop_id] = tour
                    nouvelle_frontiere.add(stop_id)

        frontiere = nouvelle_frontiere
        if not frontiere:
            break

    del arrivee_au_plus_tot[origin_stop_id]
    if not arrivee_au_plus_tot:
        return pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon", "arrivee_s", "correspondances", "duree_min"])

    resultat = pd.DataFrame({
        "stop_id": list(arrivee_au_plus_tot.keys()),
        "arrivee_s": list(arrivee_au_plus_tot.values()),
    })
    resultat["correspondances"] = resultat["stop_id"].map(correspondances_utilisees)
    resultat["duree_min"] = (resultat["arrivee_s"] - depart_s) / 60
    return resultat.merge(_feed.stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]], on="stop_id")


def fetch_geopf_isochrone(lon: float, lat: float, minutes: float):
    """Isochrone piéton Géoplateforme (geometry GeoJSON) autour d'un point,
    ou None en cas d'échec (réseau, timeout, arrêt hors zone couverte...)."""
    try:
        r = requests.get(
            GEOPF_ISOCHRONE_URL,
            params={
                "resource": "bdtopo-valhalla",
                "point": f"{lon},{lat}",
                "direction": "departure",
                "costType": "time",
                "costValue": max(60, round(minutes * 60)),
                "profile": "pedestrian",
                "geometryFormat": "geojson",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["geometry"]
    except Exception:
        return None


def recuperer_buffers_marche(arrets: pd.DataFrame, minutes: float) -> dict:
    """Isochrone piéton par arrêt atteint, avec cache session + limitation
    à GEOPF_MAX_REQ_PER_SEC requêtes/s (limite documentée de l'API)."""
    cache = st.session_state.setdefault("_geopf_cache", {})
    buffers = {}
    arrets_a_appeler = arrets.head(GEOPF_MAX_STOPS_CALLED)
    for _, arret in arrets_a_appeler.iterrows():
        cle = (round(arret["stop_lon"], 5), round(arret["stop_lat"], 5), minutes)
        if cle not in cache:
            cache[cle] = fetch_geopf_isochrone(arret["stop_lon"], arret["stop_lat"], minutes)
            time.sleep(1 / GEOPF_MAX_REQ_PER_SEC)
        buffers[arret["stop_id"]] = cache[cle]
    return buffers


def build_map(origine, arrets: pd.DataFrame, buffers: dict, budget_min: int, rayon_marche_min: int) -> folium.Map:
    center = [origine["stop_lat"], origine["stop_lon"]]
    m = folium.Map(location=center, zoom_start=13, tiles=None, prefer_canvas=True, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", cross_origin=True).add_to(m)
    folium.TileLayer(
        carto_tile_url("light_all"), name="CartoDB Positron", attr=CARTO_ATTR, cross_origin=True,
    ).add_to(m)

    if not arrets.empty:
        colormap = folium.LinearColormap(
            colors=DUREE_COLOR_SCALE, vmin=0, vmax=budget_min,
            caption="Durée de trajet depuis l'arrêt de départ (min)",
        )
        colormap.add_to(m)

        buffers_layer = folium.FeatureGroup(name="Zones de marche autour des arrêts atteints")
        for _, arret in arrets.iterrows():
            geometry = buffers.get(arret["stop_id"])
            couleur = colormap(arret["duree_min"])
            if geometry is not None:
                folium.GeoJson(
                    geometry,
                    style_function=lambda _f, c=couleur: {"fillColor": c, "color": c, "weight": 1, "fillOpacity": 0.35},
                ).add_to(buffers_layer)
            else:
                folium.Circle(
                    [arret["stop_lat"], arret["stop_lon"]],
                    radius=rayon_marche_min * 80,  # ~80 m/min à pied, repli grossier si l'API échoue
                    color=couleur, weight=1, dash_array="4", fill=True, fillColor=couleur, fillOpacity=0.25,
                ).add_to(buffers_layer)
        buffers_layer.add_to(m)

        arrets_layer = folium.FeatureGroup(name="Arrêts atteints")
        for _, arret in arrets.iterrows():
            folium.CircleMarker(
                [arret["stop_lat"], arret["stop_lon"]],
                radius=4,
                color="#334",
                weight=1,
                fill=True,
                fillColor=colormap(arret["duree_min"]),
                fillOpacity=0.9,
                tooltip=(
                    f"{arret['stop_name']}<br>{arret['duree_min']:.0f} min "
                    f"({int(arret['correspondances'])} correspondance{'s' if arret['correspondances'] > 1 else ''})"
                ),
            ).add_to(arrets_layer)
        arrets_layer.add_to(m)

    folium.Marker(
        center,
        tooltip=f"Départ : {origine['stop_name']}",
        icon=folium.Icon(color="darkred", icon="play", prefix="fa"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    st.set_page_config(page_title="Isochrone GTFS", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🚌 Isochrone GTFS — accessibilité en transport collectif")
    st.caption(
        "Arrêts atteignables en transport collectif depuis un arrêt de départ, à une heure de "
        "pointe donnée, sur le jour ouvré de base (JOB) du réseau choisi. Correspondances au même "
        "arrêt uniquement (pas de correspondance piétonne entre arrêts distincts). Chaque arrêt "
        "atteint est habillé d'un isochrone piéton (API Isochrone/Isodistance de la Géoplateforme IGN)."
    )

    with st.sidebar:
        st.header("Réseau")
        reseaux = list_gtfs_networks()
        reseau_label = st.selectbox(
            "GTFS", [label for label, _ in reseaux], index=None, placeholder="Choisir un réseau",
        )

        origine = None
        date_job = None
        feed = None
        if reseau_label is not None:
            hf_path = dict(reseaux)[reseau_label]
            local_zip = download_gtfs(hf_path)
            feed = load_feed(local_zip)
            date_job = determiner_job(feed, hf_path)
            st.caption(f"JOB détecté : {datetime.strptime(date_job, '%Y%m%d').strftime('%A %d/%m/%Y')}")

            stops = feed.stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
            stops["label"] = stops["stop_name"] + " (" + stops["stop_id"].astype(str) + ")"
            stops = stops.sort_values("label")
            arret_label = st.selectbox(
                "Arrêt de départ", stops["label"], index=None, placeholder="Choisir un arrêt",
            )
            if arret_label is not None:
                origine = stops[stops["label"] == arret_label].iloc[0]

        st.header("Heure de pointe et budget")
        heure_pointe = st.time_input("Heure de départ", value=dt_time(8, 0))
        budget_min = st.slider("Budget de trajet (minutes)", 5, 90, DEFAULT_BUDGET_MIN, step=5)
        max_correspondances = st.slider(
            "Correspondances max", 0, 2, DEFAULT_MAX_TRANSFERS,
            help="Plafonné à 2 pour contenir le temps de calcul.",
        )
        rayon_marche_min = st.slider(
            "Isochrone piéton autour de chaque arrêt atteint (min à pied)", 0, 15, DEFAULT_WALK_BUFFER_MIN,
            help="0 = afficher uniquement les arrêts (sans zone de marche).",
        )

        calculer = st.button("Calculer l'isochrone", type="primary", disabled=origine is None)

    if origine is None:
        st.info("Choisissez un réseau puis un arrêt de départ dans le menu de gauche.")
        return

    if not calculer and "derniers_resultats" not in st.session_state:
        st.info("Réglez les paramètres puis cliquez sur \"Calculer l'isochrone\".")
        return

    if calculer:
        depart_s = heure_pointe.hour * 3600 + heure_pointe.minute * 60
        arrets = calculer_arrets_atteignables(
            feed, dict(reseaux)[reseau_label], date_job, origine["stop_id"],
            depart_s, budget_min, max_correspondances,
        )
        buffers = recuperer_buffers_marche(arrets, rayon_marche_min) if rayon_marche_min > 0 and not arrets.empty else {}
        st.session_state["derniers_resultats"] = (origine, arrets, buffers, budget_min, rayon_marche_min)

    origine, arrets, buffers, budget_min, rayon_marche_min = st.session_state["derniers_resultats"]

    col_map, col_stats = st.columns([3, 1])
    with col_map:
        m = build_map(origine, arrets, buffers, budget_min, rayon_marche_min)
        st.iframe(m.get_root().render(), height=650)

    with col_stats:
        st.subheader(origine["stop_name"])
        if arrets.empty:
            st.warning("Aucun arrêt atteignable avec ces paramètres (pas de service à cette heure, budget trop court...).")
        else:
            st.metric("Arrêts atteints", f"{len(arrets):,}".replace(",", " "))
            st.metric("Durée médiane", f"{arrets['duree_min'].median():.0f} min")
            st.metric("Avec correspondance", f"{int((arrets['correspondances'] > 0).sum()):,}".replace(",", " "))
            if len(arrets) > GEOPF_MAX_STOPS_CALLED:
                st.caption(
                    f"Plus de {GEOPF_MAX_STOPS_CALLED} arrêts atteints : au-delà, les zones de marche "
                    "sont approximées par un cercle plutôt qu'un appel à l'API Géoplateforme (limite de "
                    f"{GEOPF_MAX_REQ_PER_SEC} requêtes/s)."
                )


if __name__ == "__main__":
    main()

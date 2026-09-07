"""Géofer Analysis — potentiel territorial des gares ferroviaires.

Carte unique : toutes les gares sont affichées en cluster (comme
geofer.cerema.fr), et les isochrones Géofer ainsi que la densité de
population des carreaux INSEE 200x200 m (Filosofi 2019) se chargent
pour le département choisi — sur l'ensemble du territoire, pas
seulement autour d'une gare.

La carte est rendue en HTML statique (st.components.v1.html), pas via
st_folium : c'est ce qui donne un zoom/pan fluide et 100% côté
navigateur (technique reprise de github.com/antoinechevre/
Accessibility_analysis, onglet Cartographie INSEE) plutôt qu'un
aller-retour Streamlit à chaque interaction.
"""

import json
import math
import os

import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from folium.plugins import MarkerCluster
from huggingface_hub import hf_hub_download
from shapely.ops import unary_union
from shapely.validation import make_valid

GEOFER_DIR = "Data_geofer"
INSEE_DIR = "Data_INSEE"
ADMIN_DIR = "Data_admin"
SNCF_DIR = "Data_SNCF"

# Offre 2026 : passages de train par gare et par catégorie, extraits du GTFS
# national SNCF sur un jour ouvré de référence (cf. extraire_passages_gares_gtfs.py).
OFFRE_PATH = f"{SNCF_DIR}/passages_gares_par_mode.csv"
OFFRE_CATEGORIES = {
    "passagesTer": ("TER", "#2ca25f"),
    "passagesIntercites": ("Intercités", "#fd8d3c"),
    "passagesTgv": ("TGV", "#e31a1c"),
}
OFFRE_MIN_RADIUS_PX = 5
OFFRE_MAX_RADIUS_PX = 42

# Fréquentation annuelle par gare (cf. extraire_frequentation_gares.py) : la
# source (API SNCF Gares & Connexions) n'a pas encore d'année 2026 publiée à
# la date d'écriture — 2024 est la dernière année disponible, utilisée ici
# plutôt que 2026.
FREQUENTATION_PATH = f"{SNCF_DIR}/frequentation_gares.csv"
FREQUENTATION_ANNEE = 2024
FREQUENTATION_COLOR = "#6a3d9a"
FREQUENTATION_MIN_RADIUS_PX = 7
FREQUENTATION_MAX_RADIUS_PX = 64

# Flux domicile-travail/études par commune (cf. extraire_flux_mobilite.py) :
# flèches proportionnelles depuis la commune de la gare choisie vers ses N
# plus grosses destinations. NBFLUX est une estimation pondérée (recensement
# complémentaire), donc décimale plutôt qu'un effectif entier.
FLUX_TRAVAIL_FILE = "flux_domicile_travail.csv"
FLUX_ETUDES_FILE = "flux_domicile_etudes.csv"
FLUX_THEMES = {
    # nom_affiché, colonne origine, colonne destination, colonne label destination, colonne flux, couleur
    "travail": ("Domicile-travail (2022)", "CODGEO", "DCLT", "L_DCLT", "NBFLUX_C22_ACTOCC15P", "#1f78b4"),
    "etudes": ("Domicile-études (2021)", "CODGEO", "DCETU", "L_DCETU", "NBFLUX_C21_SCOL02P", "#e6550d"),
}
FLUX_MIN_WEIGHT_PX = 1
FLUX_MAX_WEIGHT_PX = 10
FLUX_COURBURE = 0.15  # 0 = ligne droite, cf. github.com/ANGEKOTIN/Flux_mapper
COMMUNE_CENTROID_API = "https://geo.api.gouv.fr/communes/{code}"

# Quarts canoniques utilisés par Notebook_cartes_departements.ipynb pour son
# export PNG/HTML par département (impression/aperçu, indépendant de l'app) :
# un département peut n'en avoir que 1 à 4 réellement (decouper_en_quadrants
# n'en génère pas pour un coin de son rectangle englobant hors de sa forme
# réelle, ex. le 54 n'a pas de NE).
QUADRANTS = ["NO", "NE", "SO", "SE"]

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

# Dégradé de bleus repris du thème PrimeNG de geofer.cerema.fr : plus la zone
# est locale, plus le bleu est soutenu.
ISOCHRONE_FILES = {
    "10 min en voiture": (f"{GEOFER_DIR}/iso_10min_voiture.geojson", "#9cd0ed"),
    "10 min à vélo": (f"{GEOFER_DIR}/iso_10min_velo.geojson", "#1992d4"),
    "15 min à pied": (f"{GEOFER_DIR}/iso_15min_pieton.geojson", "#0a3a55"),
}

# Carreaux INSEE 200m (Filosofi 2019, France métropolitaine) : version légère
# (4 colonnes déjà calculées + géométrie, WGS84, ~148 Mo en Parquet) du gpkg
# source (1,1 Go, 39 colonnes) — cf. extraire_carreaux_leger.py. Téléchargée
# une fois depuis le dataset HF qui l'héberge et chargée entièrement en
# mémoire (load_carreaux_france) plutôt que lue par bbox à chaque zone comme
# avant : plus rapide (un seul téléchargement, un seul chargement disque par
# session) et permet un filtrage en mémoire par n'importe quelle géométrie.
INSEE_DATASET_REPO = "antoinechevre/accessibility-data"
INSEE_LEGER_REMOTE_FILE = "extracted/carreaux_200m_met_leger.parquet"

COLOR_VARIABLES = {
    "Population": "pop",
    "Revenu moyen par habitant (€ SNV)": "niveau_vie",
    "Taux de pauvreté (%)": "taux_pauvrete",
    "Part de 65 ans et + (%)": "part_65p",
}

FRANCE_CENTER = [46.6, 2.5]
FRANCE_ZOOM = 6
FRANCE_BOUNDS = (-5.5, 41.2, 9.8, 51.3)  # minx, miny, maxx, maxy

# Charte visuelle reprise de geofer.cerema.fr (thème PrimeNG bleu, police Lato)
GEOFER_PRIMARY = "#1992D4"
# Rouge séquentiel ColorBrewer "Reds" (8 paliers) : rendu proche des outils de
# cartographie de carroyage population de référence (ex. Géofer Cerema).
CARREAUX_COLOR_SCALE = [
    "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272",
    "#fb6a4a", "#de2d26", "#a50f15", "#67000d",
]
# < 1 = pousse les carreaux moyennement peuplés vers les rouges foncés (cf.
# style_carreau) plutôt que de les laisser dans les tons pâles de l'échelle.
CARREAUX_COLOR_GAMMA = 0.45
GEOFER_SURFACE_GROUND = "#eff3f8"
GEOFER_TEXT_COLOR = "#495057"

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


@st.cache_data(show_spinner="Chargement des gares Géofer...")
def load_gares() -> pd.DataFrame:
    df = pd.read_csv(f"{GEOFER_DIR}/geofer_gares.csv")
    df = df[df["siOuverte"]].copy()
    df["codeUic"] = df["codeUic"].astype(str)
    # Code commune INSEE à 5 caractères (avec zéro de tête, ex. "01001") : lu
    # comme entier sinon, ce qui casse la comparaison aux codes texte des
    # bases de flux (cf. FLUX_THEMES) même quand la valeur numérique matche.
    df["inseeCommune"] = df["inseeCommune"].astype(str).str.zfill(5)
    df["label"] = df["nomGare"] + " — " + df["nomCommune"] + " (" + df["codeUic"] + ")"
    return df.sort_values("label")


@st.cache_data(show_spinner="Chargement des contours des départements...")
def load_departements() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(f"{ADMIN_DIR}/departements.geojson")
    gdf["label"] = gdf["code"] + " — " + gdf["nom"]
    return gdf.sort_values("code")


def departements_limitrophes(dept_code: str, departements: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    dept_geom = departements.loc[departements["code"] == dept_code, "geometry"].iloc[0]
    autres = departements[departements["code"] != dept_code]
    voisins = autres[autres.geometry.touches(dept_geom)]
    if voisins.empty:
        # Repli : des géométries simplifiées peuvent laisser un micro-espace
        # entre deux départements limitrophes, faisant échouer "touches".
        voisins = autres[autres.geometry.intersects(dept_geom.buffer(0.005))]
    return voisins


@st.cache_data(show_spinner=False)
def load_isochrones(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf["code_uic"] = gdf["code_uic"].astype(str)
    return gdf


@st.cache_resource(show_spinner="Récupération des carreaux INSEE (premier chargement)...")
def get_insee_leger_path() -> str:
    local_path = os.path.join(INSEE_DIR, os.path.basename(INSEE_LEGER_REMOTE_FILE))
    if os.path.exists(local_path):
        return local_path
    # antoinechevre/accessibility-data est un dataset privé : le Space a besoin
    # d'un secret HF_TOKEN (lecture) configuré dans ses variables d'environnement.
    try:
        return hf_hub_download(
            repo_id=INSEE_DATASET_REPO,
            repo_type="dataset",
            filename=INSEE_LEGER_REMOTE_FILE,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        st.error(
            "Impossible de récupérer les carreaux INSEE depuis "
            f"{INSEE_DATASET_REPO} : {exc}\n\n"
            "Vérifiez que le secret HF_TOKEN (lecture sur ce dataset privé) "
            "est configuré dans les paramètres du Space."
        )
        st.stop()


@st.cache_resource(show_spinner="Chargement des carreaux INSEE (France entière, une fois par session)...")
def load_carreaux_france() -> gpd.GeoDataFrame:
    return gpd.read_parquet(get_insee_leger_path())


@st.cache_data(show_spinner="Filtrage des carreaux INSEE de la zone...")
def load_insee_carreaux(_carreaux_france: gpd.GeoDataFrame, _dept_geom, dept_code: str) -> gpd.GeoDataFrame:
    """Carreaux de la zone (_dept_geom), tous statuts confondus — filtrés en
    mémoire depuis _carreaux_france (déjà chargée une fois pour la session,
    cf. load_carreaux_france) plutôt que relus sur disque à chaque zone."""
    minx, miny, maxx, maxy = _dept_geom.bounds
    sous_ensemble = _carreaux_france.cx[minx:maxx, miny:maxy]
    if sous_ensemble.empty:
        return sous_ensemble
    return sous_ensemble[sous_ensemble.intersects(_dept_geom)].copy()


@st.cache_data(show_spinner="Chargement de l'offre ferroviaire 2026...")
def load_offre_2026() -> pd.DataFrame:
    df = pd.read_csv(OFFRE_PATH, dtype={"codeUic": str})
    df["totalClasse"] = df[list(OFFRE_CATEGORIES)].sum(axis=1)
    return df[df["totalClasse"] > 0]


def offre_pie_svg(gare_offre, rayon_px: float) -> str:
    """Camembert CSS (conic-gradient) : parts TER/Intercités/TGV d'une gare
    (catégorie "Autre" résiduelle du GTFS exclue, cf. extraire_passages_gares_gtfs.py,
    donc les parts totalisent toujours 360°)."""
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


def script_legende_offre():
    """Légende statique (camemberts non colorables via LinearColormap)."""
    items = "".join(
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{couleur};margin-right:4px;"></span>{label}<br>'
        for _, (label, couleur) in OFFRE_CATEGORIES.items()
    )
    return f"""
    <div style="position:fixed; bottom:28px; left:10px; z-index:1000; background:white;
        border:2px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px;
        font-family:'Lato',Helvetica,sans-serif; font-size:12px; line-height:1.5;">
        <b>Offre 2026</b><br>{items}
    </div>
    """


@st.cache_data(show_spinner="Chargement de la fréquentation annuelle...")
def load_frequentation() -> pd.DataFrame:
    df = pd.read_csv(FREQUENTATION_PATH, dtype={"codeUic": str})
    df = df[df["annee"] == FREQUENTATION_ANNEE]
    return df[df["voyageurs"] > 0]


def frequentation_bubble_svg(rayon_px: float) -> str:
    """Simple bulle proportionnelle (pas un camembert : une seule grandeur,
    pas de répartition par catégorie)."""
    taille = rayon_px * 2
    return (
        f'<div style="width:{taille:.0f}px;height:{taille:.0f}px;border-radius:50%;'
        f'background:{FREQUENTATION_COLOR};opacity:0.7;'
        f'border:1px solid rgba(0,0,0,0.5);box-shadow:0 0 3px rgba(0,0,0,0.35);"></div>'
    )


@st.cache_resource(show_spinner="Récupération des flux de mobilité...")
def get_flux_local_path(nom_fichier: str) -> str:
    local_path = os.path.join(INSEE_DIR, nom_fichier)
    if os.path.exists(local_path):
        return local_path
    try:
        return hf_hub_download(
            repo_id=INSEE_DATASET_REPO,
            repo_type="dataset",
            filename=f"extracted/{nom_fichier}",
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        st.error(f"Impossible de récupérer {nom_fichier} depuis {INSEE_DATASET_REPO} : {exc}")
        return None


@st.cache_resource(show_spinner="Chargement des flux de mobilité...")
def load_flux(theme: str) -> pd.DataFrame:
    nom_fichier = FLUX_TRAVAIL_FILE if theme == "travail" else FLUX_ETUDES_FILE
    chemin = get_flux_local_path(nom_fichier)
    if chemin is None:
        return pd.DataFrame()
    _, col_origine, col_dest, col_label, col_flux, _ = FLUX_THEMES[theme]
    df = pd.read_csv(chemin, sep=";", dtype={col_origine: str, col_dest: str})
    return df[[col_origine, col_dest, col_label, col_flux]]


@st.cache_data(show_spinner=False, ttl=86400)
def commune_centroid(code_insee: str):
    """(lat, lon) du centre de la commune (API découpage administratif de
    data.gouv.fr) ou None en cas d'échec (code invalide, réseau...) — les
    arrondissements de Paris/Lyon/Marseille (75101, 69381, 13201...) y sont
    bien référencés, comme dans les bases de flux."""
    try:
        r = requests.get(
            COMMUNE_CENTROID_API.format(code=code_insee), params={"fields": "centre"}, timeout=5,
        )
        r.raise_for_status()
        lon, lat = r.json()["centre"]["coordinates"]
        return lat, lon
    except Exception:
        return None


def bezier_arc(origine: tuple, destination: tuple, courbure: float = FLUX_COURBURE, n: int = 24) -> list:
    """Points [lat, lon] d'une courbe de Bézier quadratique entre deux points
    — évite que les flèches entre les mêmes communes se superposent
    exactement à une ligne droite (même principe que
    github.com/ANGEKOTIN/Flux_mapper, réimplémenté ici en shapely/folium)."""
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
    """Triangle CSS pointant vers le nord (0°) par défaut, tourné à bearing_deg
    (mesuré depuis le nord, sens horaire — cf. angle_entre_points)."""
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


def top_flux_avec_centroides(theme: str, code_origine: str, n: int) -> list:
    """Les n plus gros flux depuis code_origine (hors "reste dans sa
    commune", qui n'a pas de sens comme flèche), avec centroïdes résolus."""
    df = load_flux(theme)
    if df.empty:
        return []
    _, col_origine, col_dest, col_label, col_flux, _ = FLUX_THEMES[theme]
    origine_centre = commune_centroid(code_origine)
    if origine_centre is None:
        return []

    sous = df[(df[col_origine] == code_origine) & (df[col_dest] != code_origine)]
    sous = sous.sort_values(col_flux, ascending=False).head(n)

    resultats = []
    for _, ligne in sous.iterrows():
        dest_centre = commune_centroid(ligne[col_dest])
        if dest_centre is None:
            continue
        resultats.append(
            {"origine": origine_centre, "destination": dest_centre, "label": ligne[col_label], "flux": ligne[col_flux]}
        )
    return resultats


def station_popup(gare) -> str:
    lines = [f"<b>{gare['nomGare']}</b><br>{gare['nomCommune']}"]
    for mode_label, col in [
        ("15 min à pied", "habitants15MinAPied2023"),
        ("10 min à vélo", "habitants10MinAVelo2023"),
        ("10 min en voiture", "habitants10MinEnVoiture2023"),
    ]:
        if col in gare and pd.notna(gare[col]):
            lines.append(f"{mode_label} : {int(gare[col]):,} habitants".replace(",", " "))
    return "<br>".join(lines)


def script_reajuster_si_masque(m, bounds):
    """<script> qui réajuste une carte Leaflet (invalidateSize + fitBounds)
    une fois son conteneur stabilisé en taille, sans animation.

    L'iframe (st.iframe) n'a pas de largeur fixée : sa largeur réelle dépend
    de la mise en page Streamlit (colonnes, sidebar...) qui se stabilise en
    plusieurs passes après le premier paint. Réagir à chaque redimensionnement
    applique un fitBounds animé à chaque passe, ce qui fait visiblement
    "trembler" la carte pendant la stabilisation — d'où le debounce : la
    correction ne part, sans animation, qu'une fois les redimensionnements
    retombés au calme pendant DEBOUNCE_MS (technique reprise de
    antoinechevre/Accessibility_analysis, src/cartographie.py).

    bounds: (minx, miny, maxx, maxy).
    """
    nom_carte = m.get_name()
    minx, miny, maxx, maxy = bounds
    bounds_json = json.dumps([[float(miny), float(minx)], [float(maxy), float(maxx)]])
    return f"""
    <script>
    window.addEventListener("load", function() {{
        var carte = {nom_carte};
        var DEBOUNCE_MS = 250;
        var minuteur = null;
        var observer = new ResizeObserver(function(entries) {{
            for (var entree of entries) {{
                if (entree.contentRect.width > 0 && entree.contentRect.height > 0) {{
                    clearTimeout(minuteur);
                    minuteur = setTimeout(function() {{
                        carte.invalidateSize({{animate: false}});
                        carte.fitBounds({bounds_json}, {{animate: false}});
                        observer.disconnect();
                    }}, DEBOUNCE_MS);
                    return;
                }}
            }}
        }});
        observer.observe(carte.getContainer());
    }});
    </script>
    """


def script_export_png(m):
    """Bouton flottant qui exporte la vue actuelle de la carte (zoom/pan en
    cours) en PNG, via html2canvas.

    Utilisait leaflet-image auparavant : cette librairie (2016, plus
    maintenue) re-télécharge chaque tuile visible image par image plutôt que
    de lire le rendu déjà affiché, et gère mal les gros clusters de
    marqueurs (nos 3589 gares) — export mesuré à plusieurs minutes, parfois
    quasi bloqué. html2canvas lit directement le DOM/canvas déjà rendu par
    le navigateur, ce qui est nettement plus rapide et fiable.
    """
    nom_carte = m.get_name()
    return f"""
    <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
    <script>
    window.addEventListener("load", function() {{
        var carte = {nom_carte};
        var bouton = document.createElement("button");
        bouton.innerHTML = "\\u2b07\\ufe0f Export PNG";
        bouton.style.cssText = "position:absolute; bottom:28px; right:10px; z-index:1000; "
            + "background:white; border:2px solid rgba(0,0,0,0.2); border-radius:4px; "
            + "padding:6px 10px; font-family:'Lato',Helvetica,sans-serif; font-size:13px; "
            + "cursor:pointer; box-shadow:0 1px 4px rgba(0,0,0,0.2);";
        bouton.onclick = function() {{
            bouton.disabled = true;
            var texte_origine = bouton.innerHTML;
            bouton.innerHTML = "Export en cours...";
            bouton.style.visibility = "hidden";
            html2canvas(carte.getContainer(), {{useCORS: true, allowTaint: false}}).then(function(canvas) {{
                bouton.disabled = false;
                bouton.innerHTML = texte_origine;
                bouton.style.visibility = "visible";
                var lien = document.createElement("a");
                lien.download = "geofer_carte.png";
                lien.href = canvas.toDataURL("image/png");
                document.body.appendChild(lien);
                lien.click();
                document.body.removeChild(lien);
            }}).catch(function(err) {{
                bouton.disabled = false;
                bouton.innerHTML = texte_origine;
                bouton.style.visibility = "visible";
                console.error(err);
                alert("Export PNG impossible : " + err);
            }});
        }};
        carte.getContainer().appendChild(bouton);
    }});
    </script>
    """


def build_map(
    gares, center, zoom, bounds, isochrones_in_dept, selected_modes, carreaux, color_field, color_label,
    show_served, offre_in_dept, offre_max_total, frequentation_in_dept, frequentation_max, flux_par_theme,
):
    # prefer_canvas : rendu canvas plutôt que SVG, indispensable pour garder un
    # zoom/pan fluide avec plusieurs milliers de polygones (carreaux INSEE).
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, prefer_canvas=True, control_scale=True)
    # cross_origin : nécessaire pour que leaflet-image (export PNG) puisse lire
    # les tuiles sans que le canvas soit "taint" par la politique cross-origin.
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", cross_origin=True).add_to(m)
    folium.TileLayer(
        carto_tile_url("light_all"), name="CartoDB Positron", attr=CARTO_ATTR, cross_origin=True,
    ).add_to(m)
    folium.TileLayer(
        carto_tile_url("dark_all"), name="CartoDB Dark Matter", attr=CARTO_ATTR, cross_origin=True,
    ).add_to(m)

    if carreaux is not None and not carreaux.empty:
        unserved = carreaux[~carreaux["desservi"]]
        colorscale_source = unserved if not unserved.empty else carreaux
        vmin = float(colorscale_source[color_field].min())
        vmax = float(colorscale_source[color_field].max())
        colormap = folium.LinearColormap(
            colors=CARREAUX_COLOR_SCALE,
            vmin=vmin,
            vmax=vmax,
            caption=f"{color_label} — carreaux non desservis",
        )

        display_carreaux = carreaux if show_served else unserved

        def style_carreau(feature, cf=color_field, cm=colormap, lo=vmin, hi=vmax):
            if feature["properties"]["desservi"]:
                return {"fillColor": "#c8ced6", "color": "#9aa3af", "weight": 0, "fillOpacity": 0.35}
            value = feature["properties"][cf]
            if hi > lo:
                # Distribution de population par carreau très asymétrique (beaucoup
                # de carreaux peu peuplés, peu de carreaux très denses) : une
                # interpolation linéaire min-max laisse la majorité des carreaux
                # résidentiels dans les tons pâles. Le gamma < 1 pousse davantage
                # de carreaux vers le rouge foncé pour un rendu plus contrasté.
                frac = ((value - lo) / (hi - lo)) ** CARREAUX_COLOR_GAMMA
                value = lo + frac * (hi - lo)
            return {"fillColor": cm(value), "color": "#581012", "weight": 0, "fillOpacity": 1.0}

        folium.GeoJson(
            display_carreaux,
            name="Carreaux INSEE 200m",
            style_function=style_carreau,
            tooltip=folium.GeoJsonTooltip(
                fields=["pop", "niveau_vie", "taux_pauvrete", "part_65p", "desservi"],
                aliases=["Population", "Revenu moyen (€)", "Taux de pauvreté (%)", "Part 65 ans+ (%)", "Desservi"],
                localize=True,
            ),
        ).add_to(m)
        colormap.add_to(m)

    for mode in selected_modes:
        gdf = isochrones_in_dept.get(mode)
        if gdf is None or gdf.empty:
            continue
        _, color = ISOCHRONE_FILES[mode]
        folium.GeoJson(
            gdf,
            name=mode,
            style_function=lambda feature, c=color: {
                "color": c,
                "weight": 2,
                "fill": True,
                "fillColor": c,
                "fillOpacity": 0.3,
            },
        ).add_to(m)

    # offre_max_total / frequentation_max : maximum du département affiché
    # (calculé dans main()), pas un maximum national — sinon la gare la plus
    # fréquentée de France écrase l'échelle et les gares d'un département
    # quelconque se tassent quasi toutes au rayon minimal.
    if offre_in_dept is not None and not offre_in_dept.empty:
        offre_layer = folium.FeatureGroup(name="Offre 2026 (TER / Intercités / TGV)")
        for _, gare_offre in offre_in_dept.iterrows():
            rayon = OFFRE_MIN_RADIUS_PX + (OFFRE_MAX_RADIUS_PX - OFFRE_MIN_RADIUS_PX) * math.sqrt(
                gare_offre["totalClasse"] / offre_max_total
            )
            folium.Marker(
                [gare_offre["wgs84Lat"], gare_offre["wgs84Lon"]],
                icon=folium.DivIcon(
                    html=offre_pie_svg(gare_offre, rayon),
                    icon_size=(rayon * 2, rayon * 2),
                    icon_anchor=(rayon, rayon),
                ),
                tooltip=f"{gare_offre['nomGare']} — {int(gare_offre['totalClasse'])} trains/jour",
                popup=folium.Popup(offre_popup(gare_offre), max_width=220),
            ).add_to(offre_layer)
        offre_layer.add_to(m)
        m.get_root().html.add_child(folium.Element(script_legende_offre()))

    if frequentation_in_dept is not None and not frequentation_in_dept.empty:
        frequentation_layer = folium.FeatureGroup(name=f"Fréquentation {FREQUENTATION_ANNEE} (voyageurs/an)")
        for _, gare_freq in frequentation_in_dept.iterrows():
            rayon = FREQUENTATION_MIN_RADIUS_PX + (FREQUENTATION_MAX_RADIUS_PX - FREQUENTATION_MIN_RADIUS_PX) * math.sqrt(
                gare_freq["voyageurs"] / frequentation_max
            )
            folium.Marker(
                [gare_freq["wgs84Lat"], gare_freq["wgs84Lon"]],
                icon=folium.DivIcon(
                    html=frequentation_bubble_svg(rayon),
                    icon_size=(rayon * 2, rayon * 2),
                    icon_anchor=(rayon, rayon),
                ),
                tooltip=(
                    f"{gare_freq['nomGare']} — {int(gare_freq['voyageurs']):,} voyageurs/an "
                    f"({FREQUENTATION_ANNEE})".replace(",", " ")
                ),
            ).add_to(frequentation_layer)
        frequentation_layer.add_to(m)

    for theme, flux_liste in flux_par_theme.items():
        if not flux_liste:
            continue
        nom_theme, _, _, _, _, couleur = FLUX_THEMES[theme]
        flux_layer = folium.FeatureGroup(name=f"Flux {nom_theme}")
        flux_max = max(f["flux"] for f in flux_liste)
        for flux in flux_liste:
            poids = FLUX_MIN_WEIGHT_PX + (FLUX_MAX_WEIGHT_PX - FLUX_MIN_WEIGHT_PX) * math.sqrt(
                flux["flux"] / flux_max
            )
            courbe = bezier_arc(flux["origine"], flux["destination"])
            folium.PolyLine(
                courbe, color=couleur, weight=poids, opacity=0.75,
                tooltip=f"{flux['label']} — {flux['flux']:.0f} personnes",
            ).add_to(flux_layer)
            angle = angle_entre_points(courbe[-2], courbe[-1])
            taille = poids + 8
            folium.Marker(
                courbe[-1],
                icon=folium.DivIcon(
                    html=arrowhead_svg(angle, couleur, taille_px=int(taille)),
                    icon_size=(taille, taille),
                    icon_anchor=(taille / 2, taille / 2),
                ),
            ).add_to(flux_layer)
        flux_layer.add_to(m)

    cluster = MarkerCluster(name="Gares").add_to(m)
    for _, gare in gares.iterrows():
        folium.Marker(
            [gare["wgs84Lat"], gare["wgs84Lon"]],
            tooltip=gare["nomGare"],
            popup=folium.Popup(station_popup(gare), max_width=250),
            icon=folium.Icon(color="darkred", icon="train", prefix="fa"),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(script_reajuster_si_masque(m, bounds)))
    m.get_root().html.add_child(folium.Element(script_export_png(m)))
    return m


def main():
    st.set_page_config(page_title="Géofer Analysis", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🚉 Analyse Gare - accessibilité (géofer) / fréquentation / offre")
    st.caption(
        "Toutes les gares sont affichées (regroupées en clusters) ; choisissez un département "
        "pour afficher les isochrones Géofer et la densité de population des carreaux INSEE "
        "200x200 m (Filosofi 2019), sur l'ensemble de son territoire."
    )

    gares = load_gares()
    departements = load_departements()
    offre = load_offre_2026()
    frequentation = load_frequentation()
    carreaux_france = load_carreaux_france()

    with st.sidebar:
        st.header("Zone à charger")
        dept_label = st.selectbox("Département", departements["label"], index=None, placeholder="Choisir un département")
        include_voisins = st.checkbox("Inclure les départements limitrophes", value=True)

        st.header("Isochrones d'accès Gare")
        selected_modes = [mode for mode in ISOCHRONE_FILES if st.checkbox(mode, value=True)]

        st.header("Données Gares")
        show_offre = st.checkbox("Offre 2026 (camembert TER / Intercités / TGV)", value=True)
        show_frequentation = st.checkbox(f"Fréquentation {FREQUENTATION_ANNEE} (voyageurs/an)", value=False)

        st.header("Filtres carreaux INSEE")
        color_label = st.selectbox("Colorer les carreaux non desservis selon", list(COLOR_VARIABLES.keys()))
        color_field = COLOR_VARIABLES[color_label]
        pop_min = st.slider("Population minimale du carreau", 0, 200, 1, step=1)
        show_served = st.checkbox("Afficher aussi les carreaux desservis (en gris)", value=True)

    isochrones_in_dept = {}
    carreaux = None
    offre_in_dept = None
    offre_max_total = None
    frequentation_in_dept = None
    frequentation_max = None
    flux_par_theme = {theme: [] for theme in FLUX_THEMES}
    center, zoom, bounds = FRANCE_CENTER, FRANCE_ZOOM, FRANCE_BOUNDS

    if dept_label is None:
        st.info("Choisissez un département dans le menu de gauche pour afficher les isochrones et les carreaux INSEE.")
    else:
        dept = departements[departements["label"] == dept_label].iloc[0]

        if include_voisins:
            voisins = departements_limitrophes(dept["code"], departements)
        else:
            voisins = departements.iloc[0:0]

        zone_geoms = [make_valid(dept.geometry)] + [make_valid(g) for g in voisins.geometry]
        zone_geom = unary_union(zone_geoms)
        bounds = zone_geom.bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        zoom = 9 if voisins.empty else 8

        if not voisins.empty:
            st.caption("Départements limitrophes inclus : " + ", ".join(sorted(voisins["nom"])))

        in_dept = gares[
            gares["wgs84Lon"].between(bounds[0], bounds[2]) & gares["wgs84Lat"].between(bounds[1], bounds[3])
        ]
        codes_in_dept = set(in_dept["codeUic"])

        communes_options = (
            in_dept[["inseeCommune", "nomCommune"]].dropna().drop_duplicates().sort_values("nomCommune")
        )
        with st.sidebar:
            st.header("Flux domicile-travail / études")
            commune_choisie = st.selectbox(
                "Commune (parmi les gares affichées)", communes_options["nomCommune"],
                index=None, placeholder="Choisir une commune",
            )
            show_flux_travail = st.checkbox("Domicile-travail (2022)", value=False)
            show_flux_etudes = st.checkbox("Domicile-études (2021)", value=False)
            nb_flux = st.slider("Nombre de flux affichés (par thème)", 5, 30, 10)

        if commune_choisie is not None:
            code_commune = communes_options.loc[
                communes_options["nomCommune"] == commune_choisie, "inseeCommune"
            ].iloc[0]
            if show_flux_travail:
                flux_par_theme["travail"] = top_flux_avec_centroides("travail", code_commune, nb_flux)
            if show_flux_etudes:
                flux_par_theme["etudes"] = top_flux_avec_centroides("etudes", code_commune, nb_flux)

        for mode, (path, _) in ISOCHRONE_FILES.items():
            full = load_isochrones(path)
            isochrones_in_dept[mode] = full[full["code_uic"].isin(codes_in_dept)]

        if show_offre:
            offre_in_dept = in_dept[["codeUic", "wgs84Lat", "wgs84Lon"]].merge(
                offre, on="codeUic", how="inner"
            )
            if not offre_in_dept.empty:
                offre_max_total = offre_in_dept["totalClasse"].max()
        if show_frequentation:
            frequentation_in_dept = in_dept[["codeUic", "wgs84Lat", "wgs84Lon"]].merge(
                frequentation, on="codeUic", how="inner"
            )
            if not frequentation_in_dept.empty:
                frequentation_max = frequentation_in_dept["voyageurs"].max()

        zone_key = "+".join(sorted([dept["code"]] + voisins["code"].tolist()))
        carreaux = load_insee_carreaux(carreaux_france, zone_geom, zone_key)

        if carreaux.empty:
            st.warning("Aucun carreau INSEE trouvé dans cette zone.")
        else:
            carreaux = carreaux[carreaux["pop"] >= pop_min].copy()

            served_geoms = [
                make_valid(geom)
                for mode in selected_modes
                for geom in isochrones_in_dept.get(mode, gpd.GeoDataFrame(geometry=[])).geometry
            ]
            if served_geoms:
                union_geom = unary_union(served_geoms)
                carreaux["desservi"] = carreaux.intersects(union_geom)
            else:
                carreaux["desservi"] = False

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m = build_map(
            gares, center, zoom, bounds, isochrones_in_dept, selected_modes, carreaux, color_field, color_label,
            show_served, offre_in_dept, offre_max_total, frequentation_in_dept, frequentation_max, flux_par_theme,
        )
        st.iframe(m.get_root().render(), height=650)

    with col_stats:
        st.subheader(dept_label if dept_label else "Aucun département")
        if carreaux is not None and not carreaux.empty:
            unserved = carreaux[~carreaux["desservi"]]
            st.metric("Population non desservie", f"{int(unserved['pop'].sum()):,}".replace(",", " "))
            st.metric("Carreaux peuplés non desservis", f"{len(unserved):,}".replace(",", " "))
            st.metric("Population desservie (isochrones)", f"{int(carreaux[carreaux['desservi']]['pop'].sum()):,}".replace(",", " "))
        else:
            st.caption("Pas encore de données pour cette zone.")


if __name__ == "__main__":
    main()

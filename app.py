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
import streamlit as st
from folium.plugins import MarkerCluster
from huggingface_hub import HfApi, hf_hub_download
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
OFFRE_MIN_RADIUS_PX = 10
OFFRE_MAX_RADIUS_PX = 42

# Fréquentation annuelle par gare (cf. extraire_frequentation_gares.py) : la
# source (API SNCF Gares & Connexions) n'a pas encore d'année 2026 publiée à
# la date d'écriture — 2024 est la dernière année disponible, utilisée ici
# plutôt que 2026.
FREQUENTATION_PATH = f"{SNCF_DIR}/frequentation_gares.csv"
FREQUENTATION_ANNEE = 2024
FREQUENTATION_COLOR = "#6a3d9a"
FREQUENTATION_MIN_RADIUS_PX = 14
FREQUENTATION_MAX_RADIUS_PX = 64

# Cartes par quart de département pré-générées et mises en cache (cf.
# Notebook_cartes_departements.ipynb) : utilisées à la place du calcul live
# quand les 4 quarts existent pour le département choisi — plus rapide, mais
# ne prend pas en compte les départements limitrophes (le notebook ne les
# gère pas) et peut être en retard sur les dernières données si le cache
# n'a pas été régénéré.
HF_CARTES_DATASET = "antoinechevre/Analyse_gare"
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

# Carreaux INSEE 200m (Filosofi 2019, France métropolitaine) : trop volumineux
# pour tenir dans le quota de stockage du Space (1 Go), donc téléchargés à la
# demande depuis le dataset HF qui les héberge déjà — avec repli sur une copie
# locale si présente (développement local, cf. Data_INSEE/).
INSEE_DATASET_REPO = "antoinechevre/accessibility-data"
INSEE_REMOTE_FILE_METROPOLE = "extracted/carreaux_200m_met.gpkg"

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


@st.cache_data(show_spinner="Chargement des carreaux INSEE de la zone (peut prendre 10-15s avec les départements limitrophes)...")
def load_insee_carreaux(insee_path: str, _dept_geom, dept_code: str) -> gpd.GeoDataFrame:
    """Tous les carreaux du département (_dept_geom), pas seulement ceux desservis."""
    bbox_geom = gpd.GeoSeries([_dept_geom.envelope], crs="EPSG:4326")
    gdf = gpd.read_file(insee_path, bbox=bbox_geom)
    if gdf.empty:
        return gdf
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.intersects(_dept_geom)].copy()
    if gdf.empty:
        return gdf

    gdf["pop"] = gdf["ind"]
    gdf["niveau_vie"] = gdf["ind_snv"] / gdf["ind"]
    gdf["taux_pauvrete"] = (gdf["men_pauv"] / gdf["men"]).clip(upper=1) * 100
    gdf["part_65p"] = ((gdf["ind_65_79"] + gdf["ind_80p"]) / gdf["ind"]).clip(upper=1) * 100
    # Les dizaines de colonnes INSEE brutes (ind_snv, men_pauv, log_45_70...)
    # ne servent qu'à calculer les 4 champs ci-dessus : les conserver dans le
    # GeoJSON envoyé au navigateur multiplie inutilement sa taille par ~8
    # (39 colonnes contre 5), au point de dépasser la limite de message de
    # Streamlit sur les départements les plus peuplés (ex. 280 Mo sur le 17).
    return gdf[["geometry", "pop", "niveau_vie", "taux_pauvrete", "part_65p"]]


@st.cache_resource(show_spinner="Récupération des carreaux INSEE (premier chargement, peut prendre une minute)...")
def get_insee_local_path() -> str:
    local_path = os.path.join(INSEE_DIR, os.path.basename(INSEE_REMOTE_FILE_METROPOLE))
    if os.path.exists(local_path):
        return local_path
    # antoinechevre/accessibility-data est un dataset privé : le Space a besoin
    # d'un secret HF_TOKEN (lecture) configuré dans ses variables d'environnement.
    try:
        return hf_hub_download(
            repo_id=INSEE_DATASET_REPO,
            repo_type="dataset",
            filename=INSEE_REMOTE_FILE_METROPOLE,
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


@st.cache_data(show_spinner=False, ttl=3600)
def lister_cartes_cache() -> set:
    """Fichiers disponibles dans HF_CARTES_DATASET. TTL 1h (pas de cache
    indéfini) : un nouveau lot de cartes générées par le notebook doit finir
    par apparaître sans redéploiement du Space."""
    try:
        return set(HfApi().list_repo_files(HF_CARTES_DATASET, repo_type="dataset", token=os.environ.get("HF_TOKEN")))
    except Exception:
        return set()


@st.cache_resource(show_spinner="Récupération de la carte pré-générée...")
def telecharger_carte_cache(nom_fichier: str) -> str:
    return hf_hub_download(
        repo_id=HF_CARTES_DATASET, repo_type="dataset", filename=nom_fichier, token=os.environ.get("HF_TOKEN"),
    )


def afficher_cartes_cache(dept, cartes_dept: dict):
    st.info(
        "Cartes pré-générées trouvées pour ce département (cache) : affichage instantané, mais sans les "
        "départements limitrophes et potentiellement en retard sur les dernières données."
    )
    col_no, col_ne = st.columns(2)
    col_so, col_se = st.columns(2)
    for quadrant, col in zip(QUADRANTS, [col_no, col_ne, col_so, col_se]):
        with col:
            st.caption(f"{dept['nom']} — {quadrant}")
            chemin = telecharger_carte_cache(cartes_dept[quadrant])
            with open(chemin, encoding="utf-8") as f:
                st.iframe(f.read(), height=420)


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
    show_served, offre_in_dept, offre_max_total, frequentation_in_dept, frequentation_max,
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
    center, zoom, bounds = FRANCE_CENTER, FRANCE_ZOOM, FRANCE_BOUNDS

    if dept_label is None:
        st.info("Choisissez un département dans le menu de gauche pour afficher les isochrones et les carreaux INSEE.")
    else:
        dept = departements[departements["label"] == dept_label].iloc[0]

        fichiers_cache = lister_cartes_cache()
        cartes_dept = {q: f"cartes/{dept['code']}_{q}.html" for q in QUADRANTS}
        if all(f in fichiers_cache for f in cartes_dept.values()):
            afficher_cartes_cache(dept, cartes_dept)
            return

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

        insee_path = get_insee_local_path()
        zone_key = "+".join(sorted([dept["code"]] + voisins["code"].tolist()))
        carreaux = load_insee_carreaux(insee_path, zone_geom, zone_key)

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
            show_served, offre_in_dept, offre_max_total, frequentation_in_dept, frequentation_max,
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

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

import os

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from huggingface_hub import hf_hub_download
from shapely.ops import unary_union

GEOFER_DIR = "Data_geofer"
INSEE_DIR = "Data_INSEE"
ADMIN_DIR = "Data_admin"

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
MAX_CARREAUX_RENDER = 10000

# Charte visuelle reprise de geofer.cerema.fr (thème PrimeNG bleu, police Lato)
GEOFER_PRIMARY = "#1992D4"
CARREAUX_COLOR_SCALE = ["#fdf4f5", "#f0a2a5", "#e25055", "#db272d", "#581012"]
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


@st.cache_data(show_spinner=False)
def load_isochrones(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf["code_uic"] = gdf["code_uic"].astype(str)
    return gdf


@st.cache_data(show_spinner="Chargement des carreaux INSEE du département...")
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
    return gdf


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


def build_map(gares, center, zoom, isochrones_in_dept, selected_modes, carreaux, color_field, color_label, show_served):
    # prefer_canvas : rendu canvas plutôt que SVG, indispensable pour garder un
    # zoom/pan fluide avec plusieurs milliers de polygones (carreaux INSEE).
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron").add_to(m)

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

        def style_carreau(feature, cf=color_field, cm=colormap):
            if feature["properties"]["desservi"]:
                return {"fillColor": "#c8ced6", "color": "#9aa3af", "weight": 0, "fillOpacity": 0.35}
            return {"fillColor": cm(feature["properties"][cf]), "color": "#581012", "weight": 0, "fillOpacity": 0.92}

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

    cluster = MarkerCluster(name="Gares").add_to(m)
    for _, gare in gares.iterrows():
        folium.Marker(
            [gare["wgs84Lat"], gare["wgs84Lon"]],
            tooltip=gare["nomGare"],
            popup=folium.Popup(station_popup(gare), max_width=250),
            icon=folium.Icon(color="darkred", icon="train", prefix="fa"),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    st.set_page_config(page_title="Géofer Analysis", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🚉 Géofer Analysis — potentiel territorial des gares")
    st.caption(
        "Toutes les gares sont affichées (regroupées en clusters) ; choisissez un département "
        "pour afficher les isochrones Géofer et la densité de population des carreaux INSEE "
        "200x200 m (Filosofi 2019), sur l'ensemble de son territoire."
    )

    gares = load_gares()
    departements = load_departements()

    with st.sidebar:
        st.header("Zone à charger")
        dept_label = st.selectbox("Département", departements["label"], index=None, placeholder="Choisir un département")

        st.header("Isochrones affichées")
        selected_modes = [mode for mode in ISOCHRONE_FILES if st.checkbox(mode, value=True)]

        st.header("Filtres carreaux INSEE")
        color_label = st.selectbox("Colorer les carreaux non desservis selon", list(COLOR_VARIABLES.keys()))
        color_field = COLOR_VARIABLES[color_label]
        pop_min = st.slider("Population minimale du carreau", 0, 200, 1, step=1)
        show_served = st.checkbox("Afficher aussi les carreaux desservis (en gris)", value=True)

    isochrones_in_dept = {}
    carreaux = None
    truncated = False
    center, zoom = FRANCE_CENTER, FRANCE_ZOOM

    if dept_label is None:
        st.info("Choisissez un département dans le menu de gauche pour afficher les isochrones et les carreaux INSEE.")
    else:
        dept = departements[departements["label"] == dept_label].iloc[0]
        dept_geom = dept.geometry
        bounds = dept_geom.bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        zoom = 9

        in_dept = gares[
            gares["wgs84Lon"].between(bounds[0], bounds[2]) & gares["wgs84Lat"].between(bounds[1], bounds[3])
        ]
        codes_in_dept = set(in_dept["codeUic"])

        for mode, (path, _) in ISOCHRONE_FILES.items():
            full = load_isochrones(path)
            isochrones_in_dept[mode] = full[full["code_uic"].isin(codes_in_dept)]

        insee_path = get_insee_local_path()
        carreaux = load_insee_carreaux(insee_path, dept_geom, dept["code"])

        if carreaux.empty:
            st.warning("Aucun carreau INSEE trouvé dans ce département.")
        else:
            carreaux = carreaux[carreaux["pop"] >= pop_min].copy()
            if len(carreaux) > MAX_CARREAUX_RENDER:
                carreaux = carreaux.nlargest(MAX_CARREAUX_RENDER, "pop")
                truncated = True

            served_geoms = [
                geom
                for mode in selected_modes
                for geom in isochrones_in_dept.get(mode, gpd.GeoDataFrame(geometry=[])).geometry
            ]
            if served_geoms:
                union_geom = unary_union(served_geoms)
                carreaux["desservi"] = carreaux.intersects(union_geom)
            else:
                carreaux["desservi"] = False

        if truncated:
            st.warning(
                f"Trop de carreaux dans ce département : limité aux {MAX_CARREAUX_RENDER:,} "
                "les plus peuplés.".replace(",", " ")
            )

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m = build_map(
            gares, center, zoom, isochrones_in_dept, selected_modes, carreaux, color_field, color_label,
            show_served,
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

"""Géofer Analysis — potentiel territorial des gares ferroviaires.

Carte unique et navigable (comme geofer.cerema.fr) : toutes les gares
sont affichées en cluster, et les isochrones Géofer ainsi que la
densité de population des carreaux INSEE 200x200 m (Filosofi 2019) se
chargent pour la zone actuellement visible à l'écran, sur l'ensemble
du territoire — pas seulement autour d'une gare choisie.
"""

import os

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from huggingface_hub import hf_hub_download
from shapely.geometry import box
from shapely.ops import unary_union
from streamlit_folium import st_folium

GEOFER_DIR = "Data_geofer"
INSEE_DIR = "Data_INSEE"

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
MIN_ZOOM_FOR_DETAIL = 12
MAX_CARREAUX_RENDER = 8000

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


@st.cache_data(show_spinner=False)
def load_isochrones(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf["code_uic"] = gdf["code_uic"].astype(str)
    return gdf


@st.cache_data(show_spinner="Chargement des carreaux INSEE de la zone affichée...")
def load_insee_carreaux(insee_path: str, bounds: tuple) -> gpd.GeoDataFrame:
    """Tous les carreaux dans les limites de la carte (bounds), pas seulement ceux desservis."""
    west, south, east, north = bounds
    bbox_geom = gpd.GeoSeries([box(west, south, east, north)], crs="EPSG:4326")
    gdf = gpd.read_file(insee_path, bbox=bbox_geom)
    if gdf.empty:
        return gdf
    gdf = gdf.to_crs("EPSG:4326")

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


def build_map(gares, isochrones_in_view, selected_modes, carreaux, color_field, color_label, show_served, center, zoom):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)
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
                return {"fillColor": "#c8ced6", "color": "#9aa3af", "weight": 0.2, "fillOpacity": 0.35}
            return {"fillColor": cm(feature["properties"][cf]), "color": "#581012", "weight": 0.2, "fillOpacity": 0.92}

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
        gdf = isochrones_in_view.get(mode)
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
        "Carte navigable : les isochrones Géofer et la densité de population des carreaux "
        "INSEE 200x200 m (Filosofi 2019) se chargent pour la zone affichée, sur tout le territoire."
    )

    gares = load_gares()

    if "map_center" not in st.session_state:
        st.session_state.map_center = FRANCE_CENTER
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = FRANCE_ZOOM
    if "map_bounds" not in st.session_state:
        st.session_state.map_bounds = None

    with st.sidebar:
        st.header("Rechercher une gare")
        search = st.text_input("Nom de gare ou commune")
        if search:
            matches = gares[gares["label"].str.contains(search, case=False, na=False)]
            if matches.empty:
                st.caption("Aucun résultat.")
            else:
                pick = st.selectbox("Résultats", matches["label"])
                if st.button("Centrer la carte sur cette gare"):
                    row = matches[matches["label"] == pick].iloc[0]
                    st.session_state.map_center = [row["wgs84Lat"], row["wgs84Lon"]]
                    st.session_state.map_zoom = 14
                    st.session_state.map_bounds = None

        st.header("Isochrones affichées")
        selected_modes = [mode for mode in ISOCHRONE_FILES if st.checkbox(mode, value=True)]

        st.header("Filtres carreaux INSEE")
        color_label = st.selectbox("Colorer les carreaux non desservis selon", list(COLOR_VARIABLES.keys()))
        color_field = COLOR_VARIABLES[color_label]
        pop_min = st.slider("Population minimale du carreau", 0, 200, 1, step=1)
        show_served = st.checkbox("Afficher aussi les carreaux desservis (en gris)", value=True)

    zoom = st.session_state.map_zoom
    bounds = st.session_state.map_bounds
    show_detail = bounds is not None and zoom >= MIN_ZOOM_FOR_DETAIL

    isochrones_in_view = {}
    carreaux = None
    truncated = False

    if not show_detail:
        st.info(
            "Zoomez sur une zone (échelle ville/agglomération) pour afficher les isochrones "
            "et les carreaux INSEE. Toutes les gares sont visibles, regroupées en clusters, "
            "quel que soit le niveau de zoom."
        )
    else:
        west, south, east, north = bounds
        in_view = gares[
            gares["wgs84Lon"].between(west, east) & gares["wgs84Lat"].between(south, north)
        ]
        codes_in_view = set(in_view["codeUic"])

        for mode, (path, _) in ISOCHRONE_FILES.items():
            full = load_isochrones(path)
            isochrones_in_view[mode] = full[full["code_uic"].isin(codes_in_view)]

        insee_path = get_insee_local_path()
        carreaux = load_insee_carreaux(insee_path, bounds)

        if carreaux.empty:
            st.warning("Aucun carreau INSEE trouvé sur cette zone.")
        else:
            carreaux = carreaux[carreaux["pop"] >= pop_min].copy()
            if len(carreaux) > MAX_CARREAUX_RENDER:
                carreaux = carreaux.nlargest(MAX_CARREAUX_RENDER, "pop")
                truncated = True

            served_geoms = [
                geom
                for mode in selected_modes
                for geom in isochrones_in_view.get(mode, gpd.GeoDataFrame(geometry=[])).geometry
            ]
            if served_geoms:
                union_geom = unary_union(served_geoms)
                carreaux["desservi"] = carreaux.intersects(union_geom)
            else:
                carreaux["desservi"] = False

        if truncated:
            st.warning(
                f"Trop de carreaux dans la zone affichée : limité aux {MAX_CARREAUX_RENDER:,} "
                "les plus peuplés. Zoomez davantage pour un rendu exhaustif.".replace(",", " ")
            )

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m = build_map(
            gares, isochrones_in_view, selected_modes, carreaux, color_field, color_label,
            show_served, st.session_state.map_center, st.session_state.map_zoom,
        )
        st_data = st_folium(m, width=None, height=650, returned_objects=["bounds", "zoom", "center"])

    if st_data:
        if st_data.get("zoom") is not None:
            st.session_state.map_zoom = st_data["zoom"]
        if st_data.get("center"):
            st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
        if st_data.get("bounds"):
            b = st_data["bounds"]
            sw, ne = b["_southWest"], b["_northEast"]
            new_bounds = (sw["lng"], sw["lat"], ne["lng"], ne["lat"])
            if new_bounds != st.session_state.map_bounds:
                st.session_state.map_bounds = new_bounds
                st.rerun()

    with col_stats:
        st.subheader("Zone affichée")
        if carreaux is not None and not carreaux.empty:
            unserved = carreaux[~carreaux["desservi"]]
            st.metric("Population non desservie", f"{int(unserved['pop'].sum()):,}".replace(",", " "))
            st.metric("Carreaux peuplés non desservis", f"{len(unserved):,}".replace(",", " "))
            st.metric("Population desservie (isochrones)", f"{int(carreaux[carreaux['desservi']]['pop'].sum()):,}".replace(",", " "))
        else:
            st.caption("Pas encore de données pour cette zone.")


if __name__ == "__main__":
    main()

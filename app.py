"""Géofer Analysis — potentiel territorial des gares ferroviaires.

Superpose un fond de carte OSM, les isochrones Géofer (10 min voiture,
10 min vélo, 15 min à pied) autour d'une gare, et la densité de
population des carreaux INSEE 200x200 m (Filosofi 2019), avec des
filtres sur les caractéristiques de chaque carreau.
"""

import os

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from shapely.geometry import Point
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

# Carreaux INSEE 200m (Filosofi 2019) : trop volumineux pour tenir dans le
# quota de stockage du Space (1 Go), donc téléchargés à la demande depuis le
# dataset HF qui les héberge déjà — avec repli sur une copie locale si
# présente (développement local, cf. Data_INSEE/).
INSEE_DATASET_REPO = "antoinechevre/accessibility-data"
INSEE_REMOTE_FILE_BY_DEP_PREFIX = {
    "972": "extracted/carreaux_200m_mart.gpkg",
    "974": "extracted/carreaux_200m_reun.gpkg",
}
INSEE_REMOTE_FILE_METROPOLE = "extracted/carreaux_200m_met.gpkg"

COLOR_VARIABLES = {
    "Population": "pop",
    "Revenu moyen par habitant (€ SNV)": "niveau_vie",
    "Taux de pauvreté (%)": "taux_pauvrete",
    "Part de 65 ans et + (%)": "part_65p",
}

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


def station_buffer(gare, radius_km: float):
    point = gpd.GeoSeries([Point(gare["wgs84Lon"], gare["wgs84Lat"])], crs="EPSG:4326")
    buffer_l93 = point.to_crs("EPSG:2154").buffer(radius_km * 1000)
    return gpd.GeoSeries(buffer_l93, crs="EPSG:2154").to_crs("EPSG:4326").iloc[0]


@st.cache_data(show_spinner="Chargement des carreaux INSEE...")
def load_insee_carreaux(insee_path: str, _clip_geom, clip_bounds: tuple) -> gpd.GeoDataFrame:
    """Tous les carreaux dans le rayon d'affichage (_clip_geom), pas seulement ceux desservis."""
    bbox_geom = gpd.GeoSeries([_clip_geom.envelope], crs="EPSG:4326")
    gdf = gpd.read_file(insee_path, bbox=bbox_geom)
    if gdf.empty:
        return gdf
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.intersects(_clip_geom)].copy()

    gdf["pop"] = gdf["ind"]
    gdf["niveau_vie"] = gdf["ind_snv"] / gdf["ind"]
    gdf["taux_pauvrete"] = (gdf["men_pauv"] / gdf["men"]).clip(upper=1) * 100
    gdf["part_65p"] = ((gdf["ind_65_79"] + gdf["ind_80p"]) / gdf["ind"]).clip(upper=1) * 100
    return gdf


@st.cache_resource(show_spinner="Récupération des carreaux INSEE (premier chargement, peut prendre une minute)...")
def get_insee_local_path(remote_filename: str) -> str:
    local_path = os.path.join(INSEE_DIR, os.path.basename(remote_filename))
    if os.path.exists(local_path):
        return local_path
    # antoinechevre/accessibility-data est un dataset privé : le Space a besoin
    # d'un secret HF_TOKEN (lecture) configuré dans ses variables d'environnement.
    try:
        return hf_hub_download(
            repo_id=INSEE_DATASET_REPO,
            repo_type="dataset",
            filename=remote_filename,
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


def insee_file_for_departement(dep: str) -> str:
    dep = str(dep)[:3]
    remote_filename = INSEE_REMOTE_FILE_BY_DEP_PREFIX.get(dep, INSEE_REMOTE_FILE_METROPOLE)
    return get_insee_local_path(remote_filename)


def build_map(gare, isochrones_by_mode, selected_modes, carreaux, color_field, color_label, show_served):
    m = folium.Map(location=[gare["wgs84Lat"], gare["wgs84Lon"]], zoom_start=13, tiles=None)
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
        gdf = isochrones_by_mode[mode]
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

    folium.Marker(
        [gare["wgs84Lat"], gare["wgs84Lon"]],
        tooltip=gare["nomGare"],
        icon=folium.Icon(color="darkred", icon="train", prefix="fa"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    st.set_page_config(page_title="Géofer Analysis", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🚉 Géofer Analysis — potentiel territorial des gares")
    st.caption(
        "Isochrones Géofer autour d'une gare superposées à la densité de population "
        "des carreaux INSEE 200x200 m (Filosofi 2019)."
    )

    gares = load_gares()

    with st.sidebar:
        st.header("Gare")
        search = st.text_input("Rechercher une gare ou une commune")
        options = gares
        if search:
            mask = gares["label"].str.contains(search, case=False, na=False)
            options = gares[mask]
        if options.empty:
            st.warning("Aucune gare ne correspond à la recherche.")
            st.stop()
        label = st.selectbox("Gare", options["label"], index=0)
        gare = options[options["label"] == label].iloc[0]

        st.header("Isochrones affichées")
        selected_modes = [mode for mode in ISOCHRONE_FILES if st.checkbox(mode, value=True)]

        st.header("Zone d'affichage")
        radius_km = st.slider("Rayon d'affichage des carreaux (km)", 1, 20, 5)
        st.caption("Tous les carreaux du rayon sont affichés, y compris hors de l'aire d'attraction des gares.")

        st.header("Filtres carreaux INSEE")
        color_label = st.selectbox("Colorer les carreaux non desservis selon", list(COLOR_VARIABLES.keys()))
        color_field = COLOR_VARIABLES[color_label]
        pop_min = st.slider("Population minimale du carreau", 0, 200, 1, step=1)
        show_served = st.checkbox("Afficher aussi les carreaux desservis (en gris)", value=True)

    isochrones_by_mode = {
        mode: load_isochrones(path).query("code_uic == @gare.codeUic")
        for mode, (path, _) in ISOCHRONE_FILES.items()
    }

    display_geom = station_buffer(gare, radius_km)
    insee_path = insee_file_for_departement(gare["inseeDepartement"])
    carreaux = load_insee_carreaux(insee_path, display_geom, display_geom.bounds)

    if carreaux.empty:
        st.warning("Aucun carreau INSEE trouvé sur cette zone.")
    else:
        carreaux = carreaux[carreaux["pop"] >= pop_min].copy()
        if selected_modes:
            union_geom = unary_union([isochrones_by_mode[m].geometry.iloc[0] for m in selected_modes])
            carreaux["desservi"] = carreaux.intersects(union_geom)
        else:
            carreaux["desservi"] = False

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m = build_map(gare, isochrones_by_mode, selected_modes, carreaux, color_field, color_label, show_served)
        st_folium(m, width=None, height=650, returned_objects=[])

    with col_stats:
        st.subheader(gare["nomGare"])
        st.write(f"{gare['nomCommune']} ({gare['inseeDepartement']})")
        if carreaux is not None and not carreaux.empty:
            unserved = carreaux[~carreaux["desservi"]]
            st.metric("Population non desservie (rayon affiché)", f"{int(unserved['pop'].sum()):,}".replace(",", " "))
            st.metric("Carreaux peuplés non desservis", f"{len(unserved):,}".replace(",", " "))
            st.metric("Population desservie (isochrones)", f"{int(carreaux[carreaux['desservi']]['pop'].sum()):,}".replace(",", " "))
        st.divider()
        st.caption("Potentiel Géofer déclaré (dans le rayon officiel de chaque mode) :")
        for mode_label, col in [
            ("15 min à pied", "habitants15MinAPied2023"),
            ("10 min à vélo", "habitants10MinAVelo2023"),
            ("10 min en voiture", "habitants10MinEnVoiture2023"),
        ]:
            if col in gare and pd.notna(gare[col]):
                st.write(f"**{mode_label}** : {int(gare[col]):,} habitants".replace(",", " "))


if __name__ == "__main__":
    main()

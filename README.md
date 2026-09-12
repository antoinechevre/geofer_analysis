---
title: Geofer Analysis
emoji: 🚉
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.59.2"
app_file: app.py
pinned: false
---

# Géofer Analysis

Application Streamlit qui superpose :

- un fond de carte OpenStreetMap ;
- les isochrones [Géofer](https://www.data.gouv.fr/datasets/geofer-donnees-de-potentiel-territorial-des-gares-ferroviaires)
  autour d'une gare (10 min en voiture, 10 min à vélo, 15 min à pied) ;
- la densité de population des carreaux INSEE 200x200 m (Filosofi 2019),
  filtrable selon les caractéristiques de chaque carreau (population,
  revenu moyen, taux de pauvreté, part de 65 ans et plus) ;
- l'offre ferroviaire 2026 par gare (camembert TER / Intercités / TGV) et la
  fréquentation annuelle par gare, par département.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Données

- `Data_geofer/` : gares, isochrones (téléchargées via `Notebook_dowload_geofer.ipynb`).
- `Data_INSEE/` : carreaux 200 m Filosofi 2019 (métropole, Martinique, Réunion),
  suivis avec Git LFS en raison de leur taille.
- `Data_SNCF/` : offre (`passages_gares_par_mode.csv`, cf.
  `extraire_passages_gares_gtfs.py`) et fréquentation par gare
  (`frequentation_gares.csv`, cf. `extraire_frequentation_gares.py`). Le
  GTFS national SNCF source (`National_GTFS.zip`, requis pour régénérer le
  premier fichier) n'est pas versionné (gros fichier tiers, cf. `.gitignore`).

## Déploiement sur Hugging Face Spaces

```bash
git remote add hf https://huggingface.co/spaces/antoinechevre/Geofer_analysis
git push hf main
```

## Corridor Analyse

Deuxième application du même dépôt : `app_corridor.py` (déployée à part sur
[Corridor_Analyse_fr](https://huggingface.co/spaces/antoinechevre/Corridor_Analyse_fr)),
issue de `Notebook_corridor.ipynb`. Étant donné deux gares Géofer, elle :

- suit la vraie ligne de chemin de fer entre les deux (données OpenStreetMap
  via l'API Overpass + plus court chemin `networkx`, pas une approximation
  géométrique) et détecte les gares intermédiaires (une seule par commune —
  la plus fréquentée quand une agglomération en a plusieurs) ;
- calcule la population desservie (carreaux INSEE 200 m dans l'aire
  d'influence Géofer 10 min en voiture de chaque gare, partagée par
  diagramme de Voronoï entre gares voisines pour éviter le double compte) ;
- calcule les flux domicile-travail/domicile-études le long du corridor et
  la charge cumulée par tronçon (même principe que
  [GTFS_analysis_fr](https://github.com/antoinechevre/GTFS_analysis_fr)) ;
- affiche une carte HTML avec toutes les couches sélectionnables
  (fonds de carte, carreaux, isochrones, offre, fréquentation, flux, charge
  par tronçon), exportable en PNG telle qu'affichée.

Les résultats intermédiaires sont mis en cache sur le dataset HF
[antoinechevre/Analyse_gare](https://huggingface.co/datasets/antoinechevre/Analyse_gare) :
un corridor déjà analysé se recharge instantanément (sidebar « Corridor
déjà identifié ») pour les visiteurs suivants.

```bash
streamlit run app_corridor.py
```

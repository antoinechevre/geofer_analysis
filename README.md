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

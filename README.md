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
  revenu moyen, taux de pauvreté, part de 65 ans et plus).

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Données

- `Data_geofer/` : gares, isochrones (téléchargées via `Notebook_dowload_geofer.ipynb`).
- `Data_INSEE/` : carreaux 200 m Filosofi 2019 (métropole, Martinique, Réunion),
  suivis avec Git LFS en raison de leur taille.

## Déploiement sur Hugging Face Spaces

```bash
git remote add hf https://huggingface.co/spaces/antoinechevre/Geofer_analysis
git push hf main
```

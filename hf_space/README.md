---
title: Aviator Predictor
emoji: 🎰
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Aviator Predictor

Modele XGBoost predicant le prochain multiplicateur d'une partie Aviator a
partir des 5 derniers multiplicateurs observes.

Entrez les 5 derniers multiplicateurs (le plus recent en premier) et l'interface
calcule automatiquement les 11 features (lags, moyennes/ecarts rolling, EWM)
utilisees par le modele, puis affiche la prediction.
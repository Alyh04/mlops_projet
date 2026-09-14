# Aviator Prediction — MLOps

Pipeline MLOps complet : data versioning (DVC), entrainement reproductible
(XGBoost + MLflow), tests CI stricts, deploiement sur Railway et monitoring
de drift (Evidently + Slack).

## Architecture

```
GitHub Actions
  1. test    : pytest strict + restauration des donnees via DVC (GDrive)
  2. train   : python train.py -> models/Model.pkl + tracking MLflow
  3. build   : image Docker (xgb + FastAPI) -> GHCR
  4. deploy  : deploiement Railway
  (option)   : monitor.yml quotidien -> detection de drift Evidently
```

- **API** : `main.py` (FastAPI) — `GET /`, `GET /model-info`, `POST /predict`
- **Modele** : `XGBRegressor` (11 features lag/rolling sur multiplicateurs)
- **Donnees** : versionnees par DVC (remote Google Drive)

## Pourquoi rester sur Railway (et laisser Hugging Face de cote)

Le choix de deploiement initial visait aussi un espace Gradio sur Hub
(Hugging Face) en plus de l'API Railway. Nous restons finalement sur Railway
seul, pour deux raisons :

1. **Cout** : depuis 2026, Hugging Face exige un abonnement **PRO** (~9$/mois,
   ou Team/Enterprise pour les orgs) pour creer et heberger un Space
   Gradio/Docker. L'offre gratuite ne couvre que les Spaces statiques et
   l'exception ZeroGPU (indisponible/limitante pour notre cas et notre quota).
   Sans budget alloue, cette option est ecartee.

2. **Simplicite** : Railway heberge deja l'API FastAPI en production (avec le
   modele embarque dans l'image Docker). Tout reste au meme endroit : un seul
   point de deploiement, une seule chaîne CI/CD, pas de duplication de
   l'interface ni de dependance a une deuxieme plateforme.

Si un budget apparait plus tard, la reprise est simple : un Space Gradio ne
demande qu'un `app.py` + le `Model.pkl` pousse via `huggingface_hub`.

## Deployer

Chaque push sur `main` : branche `main` -> GitHub Actions (test, train, build)
-> Railway. Le deploiement Railway peut se faire aussi automatiquement via
l'app GitHub "Railway" ou via le CLI (secrets `RAILWAY_TOKEN`,
`RAILWAY_SERVICE_ID`).

## Tester en local

```bash
dvc pull          # restaure data/aviator_dataset.csv et data/multipliers.csv
pip install -r requirements-dev.txt
pytest tests/ -v  # tests API + entrainement
uvicorn main:app --reload   # API locale sur http://127.0.0.1:8000
```
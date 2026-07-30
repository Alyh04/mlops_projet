#!/usr/bin/env python3
"""
init_project.py — Script d'initialisation du pipeline MLOps Aviator.
Verifie l'arborescence, inspecte les CSV, genere les fichiers manquants.
"""

import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# 1.  Verification de la structure
# -------------------------------------------------
REQUIRED_DIRS = [
    "data",
    "models",
    ".github",
    ".github/workflows",
]

REQUIRED_DATA = [
    "data/aviator_dataset.csv",
    "data/multipliers.csv",
]

SCRIPTS_TO_GENERATE = {
    "requirements.txt": None,
    "train.py": None,
    "main.py": None,
    "monitor.py": None,
    "Dockerfile": None,
    ".github/workflows/ci-cd.yml": None,
}

# -------------------------------------------------
# 2.  Contenu des fichiers a generer
# -------------------------------------------------

REQUIREMENTS_TXT = """\
pandas==2.2.3
scikit-learn==1.6.1
xgboost==2.1.4
fastapi==0.115.12
uvicorn[standard]==0.34.2
mlflow==2.19.0
evidently==0.4.38
joblib==1.4.2
pydantic==2.10.6
"""

TRAIN_PY = (
    '"""\n'
    'train.py - Entrainement du modele Aviator.\n'
    'Charge et fusionne aviator_dataset.csv + multipliers.csv,\n'
    'applique du feature engineering, entraine un XGBoost,\n'
    'enregistre les metriques dans MLflow et exporte Model.pkl.\n'
    '"""\n'
    '\n'
    'import warnings\n'
    'import joblib\n'
    'import mlflow\n'
    'import mlflow.sklearn\n'
    'import pandas as pd\n'
    'import numpy as np\n'
    'from sklearn.model_selection import train_test_split\n'
    'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n'
    'from xgboost import XGBRegressor\n'
    '\n'
    'warnings.filterwarnings("ignore")\n'
    '\n'
    'DATA_DIR = "data"\n'
    'MODEL_DIR = "models"\n'
    'MODEL_PATH = f"{MODEL_DIR}/Model.pkl"\n'
    'TARGET = "target"\n'
    'MLFLOW_EXPERIMENT = "Aviator_Prediction"\n'
    'RANDOM_STATE = 42\n'
    'TEST_SIZE = 0.2\n'
    '\n'
    '\n'
    'def load_data():\n'
    '    """Charge les deux jeux de donnees et les fusionne."""\n'
    '    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")\n'
    '    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")\n'
    '\n'
    '    print("[INFO] aviator_dataset.csv - shape:", aviator.shape)\n'
    '    print("[INFO] multipliers.csv   - shape:", multipliers.shape)\n'
    '\n'
    '    all_multipliers = pd.concat(\n'
    '        [aviator[TARGET], multipliers["Multiplier"]],\n'
    '        axis=0,\n'
    '        ignore_index=True,\n'
    '    ).dropna()\n'
    '\n'
    '    print(f"[INFO] Total multiplicateurs unifies : {len(all_multipliers)}")\n'
    '\n'
    '    feat = pd.DataFrame({"multiplier": all_multipliers})\n'
    '    feat = feat.sort_index()\n'
    '\n'
    '    feat["mult_lag1"] = feat["multiplier"].shift(1)\n'
    '    feat["mult_lag2"] = feat["multiplier"].shift(2)\n'
    '    feat["mult_lag3"] = feat["multiplier"].shift(3)\n'
    '    feat["roll_mean_3"] = feat["multiplier"].rolling(3).mean()\n'
    '    feat["roll_mean_5"] = feat["multiplier"].rolling(5).mean()\n'
    '    feat["roll_std_3"] = feat["multiplier"].rolling(3).std()\n'
    '    feat["roll_std_5"] = feat["multiplier"].rolling(5).std()\n'
    '    feat["roll_max_3"] = feat["multiplier"].rolling(3).max()\n'
    '    feat["roll_min_3"] = feat["multiplier"].rolling(3).min()\n'
    '    feat["ewm_mean"] = feat["multiplier"].ewm(span=5).mean()\n'
    '    feat["mult_ratio_lag1"] = feat["multiplier"] / (feat["mult_lag1"] + 1e-9)\n'
    '\n'
    '    feat = feat.dropna().reset_index(drop=True)\n'
    '\n'
    '    X = feat.drop(columns=["multiplier"])\n'
    '    y = feat["multiplier"]\n'
    '\n'
    '    print(f"[INFO] Features shape        : {X.shape}")\n'
    '    print(f"[INFO] Features columns       : {list(X.columns)}")\n'
    '    return X, y\n'
    '\n'
    '\n'
    'def train():\n'
    '    """Entraine un XGBoost et logge dans MLflow."""\n'
    '    X, y = load_data()\n'
    '\n'
    '    X_train, X_test, y_train, y_test = train_test_split(\n'
    '        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE\n'
    '    )\n'
    '\n'
    '    params = {\n'
    '        "n_estimators": 300,\n'
    '        "max_depth": 6,\n'
    '        "learning_rate": 0.05,\n'
    '        "subsample": 0.8,\n'
    '        "colsample_bytree": 0.8,\n'
    '        "random_state": RANDOM_STATE,\n'
    '    }\n'
    '\n'
    '    mlflow.set_experiment(MLFLOW_EXPERIMENT)\n'
    '\n'
    '    with mlflow.start_run():\n'
    '        model = XGBRegressor(**params, verbosity=0)\n'
    '        model.fit(X_train, y_train)\n'
    '\n'
    '        y_pred = model.predict(X_test)\n'
    '\n'
    '        mae = mean_absolute_error(y_test, y_pred)\n'
    '        mse = mean_squared_error(y_test, y_pred)\n'
    '        r2 = r2_score(y_test, y_pred)\n'
    '\n'
    '        mlflow.log_params(params)\n'
    '        mlflow.log_metrics({"mae": mae, "mse": mse, "r2": r2})\n'
    '        mlflow.sklearn.log_model(model, "model")\n'
    '        mlflow.set_tag("model", "XGBRegressor")\n'
    '\n'
    '        os.makedirs(MODEL_DIR, exist_ok=True)\n'
    '        joblib.dump(model, MODEL_PATH)\n'
    '\n'
    '        print(f"\\n[OK] Modele sauvegarde -> {MODEL_PATH}")\n'
    '        print(f"    MAE : {mae:.4f}  |  MSE : {mse:.4f}  |  R2 : {r2:.4f}")\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    train()\n'
)

MAIN_PY = (
    '"""\n'
    'main.py - API FastAPI de prediction Aviator.\n'
    'Charge models/Model.pkl et expose:\n'
    '  GET  /        -> health-check\n'
    '  POST /predict -> prediction du multiplicateur\n'
    '"""\n'
    '\n'
    'import joblib\n'
    'import pandas as pd\n'
    'import numpy as np\n'
    'from fastapi import FastAPI, HTTPException\n'
    'from pydantic import BaseModel, Field\n'
    '\n'
    'MODEL_PATH = "models/Model.pkl"\n'
    '\n'
    'app = FastAPI(title="Aviator Predictor API", version="1.0.0")\n'
    '\n'
    'try:\n'
    '    model = joblib.load(MODEL_PATH)\n'
    '    FEATURE_NAMES = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else None\n'
    'except Exception as e:\n'
    '    model = None\n'
    '    FEATURE_NAMES = None\n'
    '    print(f"[WARN] Impossible de charger le modele : {e}")\n'
    '\n'
    '\n'
    'class PredictInput(BaseModel):\n'
    '    mult_lag1: float = Field(..., description="Multiplicateur a t-1")\n'
    '    mult_lag2: float = Field(..., description="Multiplicateur a t-2")\n'
    '    mult_lag3: float = Field(..., description="Multiplicateur a t-3")\n'
    '    roll_mean_3: float = Field(..., description="Moyenne mobile sur 3")\n'
    '    roll_mean_5: float = Field(..., description="Moyenne mobile sur 5")\n'
    '    roll_std_3: float = Field(..., description="Ecart-type mobile 3")\n'
    '    roll_std_5: float = Field(..., description="Ecart-type mobile 5")\n'
    '    roll_max_3: float = Field(..., description="Max mobile sur 3")\n'
    '    roll_min_3: float = Field(..., description="Min mobile sur 3")\n'
    '    ewm_mean: float = Field(..., description="Moyenne exponentielle span=5")\n'
    '    mult_ratio_lag1: float = Field(..., description="Ratio multiplicateur / lag1")\n'
    '\n'
    '\n'
    'class PredictOutput(BaseModel):\n'
    '    prediction: float\n'
    '    features_used: list[str]\n'
    '\n'
    '\n'
    '@app.get("/", tags=["Health"])\n'
    'def health():\n'
    '    return {\n'
    '        "status": "ok",\n'
    '        "model_loaded": model is not None,\n'
    '        "timestamp": str(pd.Timestamp.now()),\n'
    '    }\n'
    '\n'
    '\n'
    '@app.post("/predict", response_model=PredictOutput, tags=["Prediction"])\n'
    'def predict(input_data: PredictInput):\n'
    '    if model is None:\n'
    '        raise HTTPException(status_code=503, detail="Modele non charge")\n'
    '\n'
    '    df = pd.DataFrame([input_data.model_dump()])\n'
    '\n'
    '    if FEATURE_NAMES is not None:\n'
    '        missing = [c for c in FEATURE_NAMES if c not in df.columns]\n'
    '        if missing:\n'
    '            raise HTTPException(\n'
    '                status_code=400,\n'
    '                detail=f"Colonnes manquantes : {missing}",\n'
    '            )\n'
    '        df = df[FEATURE_NAMES]\n'
    '\n'
    '    pred = model.predict(df)[0]\n'
    '    return PredictOutput(prediction=round(float(pred), 4), features_used=list(df.columns))\n'
)

MONITOR_PY = (
    '"""\n'
    'monitor.py - Detection de Data Drift avec Evidently AI.\n'
    'Compare les donnees d\'apprentissage (reference) aux nouvelles predictions (current).\n'
    '"""\n'
    '\n'
    'import warnings\n'
    'import joblib\n'
    'import pandas as pd\n'
    'import numpy as np\n'
    'from sklearn.model_selection import train_test_split\n'
    'from evidently.report import Report\n'
    'from evidently.metric_preset import DataDriftPreset, RegressionPreset\n'
    'from evidently.test_suite import TestSuite\n'
    'from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns\n'
    '\n'
    'warnings.filterwarnings("ignore")\n'
    '\n'
    'DATA_DIR = "data"\n'
    'MODEL_PATH = "models/Model.pkl"\n'
    'TARGET = "target"\n'
    'RANDOM_STATE = 42\n'
    'DRIFT_OUTPUT = "drift_report.html"\n'
    '\n'
    '\n'
    'def load_reference_data():\n'
    '    """Construit le jeu de reference (features + target)."""\n'
    '    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")\n'
    '    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")\n'
    '\n'
    '    all_mult = pd.concat(\n'
    '        [aviator[TARGET], multipliers["Multiplier"]], axis=0, ignore_index=True\n'
    '    ).dropna()\n'
    '\n'
    '    feat = pd.DataFrame({"multiplier": all_mult}).sort_index()\n'
    '    feat["mult_lag1"] = feat["multiplier"].shift(1)\n'
    '    feat["mult_lag2"] = feat["multiplier"].shift(2)\n'
    '    feat["mult_lag3"] = feat["multiplier"].shift(3)\n'
    '    feat["roll_mean_3"] = feat["multiplier"].rolling(3).mean()\n'
    '    feat["roll_mean_5"] = feat["multiplier"].rolling(5).mean()\n'
    '    feat["roll_std_3"] = feat["multiplier"].rolling(3).std()\n'
    '    feat["roll_std_5"] = feat["multiplier"].rolling(5).std()\n'
    '    feat["roll_max_3"] = feat["multiplier"].rolling(3).max()\n'
    '    feat["roll_min_3"] = feat["multiplier"].rolling(3).min()\n'
    '    feat["ewm_mean"] = feat["multiplier"].ewm(span=5).mean()\n'
    '    feat["mult_ratio_lag1"] = feat["multiplier"] / (feat["mult_lag1"] + 1e-9)\n'
    '\n'
    '    feat = feat.dropna().reset_index(drop=True)\n'
    '\n'
    '    X = feat.drop(columns=["multiplier"])\n'
    '    y = feat["multiplier"]\n'
    '\n'
    '    _, X_ref, _, y_ref = train_test_split(\n'
    '        X, y, test_size=0.3, random_state=RANDOM_STATE\n'
    '    )\n'
    '    ref = X_ref.copy()\n'
    '    ref[TARGET] = y_ref.values\n'
    '    return ref\n'
    '\n'
    '\n'
    'def load_current_data():\n'
    '    """Simule des donnees current recentes a partir du dataset."""\n'
    '    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")\n'
    '    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")\n'
    '\n'
    '    all_mult = pd.concat(\n'
    '        [aviator[TARGET], multipliers["Multiplier"]], axis=0, ignore_index=True\n'
    '    ).dropna()\n'
    '\n'
    '    feat = pd.DataFrame({"multiplier": all_mult}).sort_index()\n'
    '    feat["mult_lag1"] = feat["multiplier"].shift(1)\n'
    '    feat["mult_lag2"] = feat["multiplier"].shift(2)\n'
    '    feat["mult_lag3"] = feat["multiplier"].shift(3)\n'
    '    feat["roll_mean_3"] = feat["multiplier"].rolling(3).mean()\n'
    '    feat["roll_mean_5"] = feat["multiplier"].rolling(5).mean()\n'
    '    feat["roll_std_3"] = feat["multiplier"].rolling(3).std()\n'
    '    feat["roll_std_5"] = feat["multiplier"].rolling(5).std()\n'
    '    feat["roll_max_3"] = feat["multiplier"].rolling(3).max()\n'
    '    feat["roll_min_3"] = feat["multiplier"].rolling(3).min()\n'
    '    feat["ewm_mean"] = feat["multiplier"].ewm(span=5).mean()\n'
    '    feat["mult_ratio_lag1"] = feat["multiplier"] / (feat["mult_lag1"] + 1e-9)\n'
    '\n'
    '    feat = feat.dropna().reset_index(drop=True)\n'
    '\n'
    '    X = feat.drop(columns=["multiplier"])\n'
    '    y = feat["multiplier"]\n'
    '\n'
    '    _, X_cur, _, y_cur = train_test_split(\n'
    '        X, y, test_size=0.15, random_state=RANDOM_STATE + 1\n'
    '    )\n'
    '    cur = X_cur.copy()\n'
    '    cur[TARGET] = y_cur.values\n'
    '    return cur\n'
    '\n'
    '\n'
    'def run_drift_detection():\n'
    '    """Execute les rapports Evidently et sauvegarde le HTML."""\n'
    '    print("[INFO] Chargement des donnees de reference...")\n'
    '    reference = load_reference_data()\n'
    '\n'
    '    print("[INFO] Chargement des donnees courantes (simulation)...")\n'
    '    current = load_current_data()\n'
    '\n'
    '    print("[INFO] Generation du rapport de Data Drift...")\n'
    '    drift_report = Report(metrics=[DataDriftPreset()])\n'
    '    drift_report.run(reference_data=reference, current_data=current)\n'
    '    drift_report.save_html(DRIFT_OUTPUT)\n'
    '    print(f"[OK] Rapport de drift sauvegarde -> {DRIFT_OUTPUT}")\n'
    '\n'
    '    print("[INFO] Generation du rapport de performance regression...")\n'
    '    try:\n'
    '        model = joblib.load(MODEL_PATH)\n'
    '        perf_report = Report(metrics=[RegressionPreset()])\n'
    '        perf_report.run(\n'
    '            reference_data=reference,\n'
    '            current_data=current,\n'
    '            model=model,\n'
    '        )\n'
    '        perf_report.save_html("regression_performance.html")\n'
    '        print("[OK] Rapport de performance sauvegarde -> regression_performance.html")\n'
    '    except Exception as e:\n'
    '        print(f"[WARN] Rapport de performance ignore : {e}")\n'
    '\n'
    '    print("[INFO] Execution des tests de drift...")\n'
    '    suite = TestSuite(tests=[TestColumnDrift(), TestShareOfDriftedColumns()])\n'
    '    suite.run(reference_data=reference, current_data=current)\n'
    '    suite.save_html("drift_tests.html")\n'
    '    print("[OK] Tests de drift sauvegardes -> drift_tests.html")\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    run_drift_detection()\n'
)

DOCKERFILE = """\
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

CI_CD_YML = """\
name: CI/CD Pipeline

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: |
          if [ -d tests/ ]; then pytest tests/ -v; else echo "No tests/ directory found - skipping"; fi

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        run: docker build -t aviator-api:latest .
"""


# -------------------------------------------------
# 3.  Fonction utilitaire : analyse d'un CSV
# -------------------------------------------------
def inspect_csv(rel_path: str) -> None:
    full = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.isfile(full):
        print(f"  [ABSENT]  {rel_path}")
        return
    import pandas as pd

    df = pd.read_csv(full)
    print(f"\n  -- {rel_path} ({len(df)} lignes, {len(df.columns)} colonnes) --")
    print(f"  Colonnes  : {list(df.columns)}")
    print(f"  Types     :\n{df.dtypes.to_string()}")
    print(f"  Nuls      :\n{df.isnull().sum().to_string()}")
    print(f"  Aperçu    :")
    for i, row in df.head(5).iterrows():
        print(f"    {i}: {dict(row)}")


# -------------------------------------------------
# 4.  Generation des scripts
# -------------------------------------------------
CONTENT_MAP = {
    "requirements.txt": REQUIREMENTS_TXT,
    "train.py": TRAIN_PY,
    "main.py": MAIN_PY,
    "monitor.py": MONITOR_PY,
    "Dockerfile": DOCKERFILE,
    ".github/workflows/ci-cd.yml": CI_CD_YML,
}


def ensure_dirs():
    print("\n" + "=" * 50)
    print("  1. VERIFICATION DE L'ARBORESCENCE")
    print("=" * 50)
    for d in REQUIRED_DIRS:
        path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(path):
            print(f"  [OK]  {d}/")
        else:
            os.makedirs(path, exist_ok=True)
            print(f"  [CRÉÉ] {d}/")


def inspect_data():
    print("\n" + "=" * 50)
    print("  2. ANALYSE DES DONNEES")
    print("=" * 50)
    for f in REQUIRED_DATA:
        inspect_csv(f)


def generate_scripts():
    print("\n" + "=" * 50)
    print("  3. GENERATION DES FICHIERS MANQUANTS")
    print("=" * 50)
    count = 0
    for rel, content in CONTENT_MAP.items():
        full = os.path.join(PROJECT_ROOT, rel)
        if os.path.isfile(full) and os.path.getsize(full) > 0:
            print(f"  [EXISTE] {rel} ({os.path.getsize(full)} octets)")
        else:
            with open(full, "w", encoding="utf-8") as f:
                f.write(content.lstrip("\n"))
            print(f"  [GENERE] {rel}")
            count += 1

    if count == 0:
        print("  Aucun fichier a generer - tous presents.")
    else:
        print(f"\n  ok {count} fichier(s) genere(s).")


def summary():
    print("\n" + "=" * 50)
    print("  RECAPITULATIF - Structure du projet")
    print("=" * 50 + "\n")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue
        level = root.replace(PROJECT_ROOT, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub = "  " * (level + 1)
        for f in sorted(files):
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            print(f"{sub}{f} ({size} octets)")


def main():
    print(f"\n  Projet MLOps - Aviator")
    print(f"  Racine : {PROJECT_ROOT}")
    print(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_dirs()
    inspect_data()
    generate_scripts()

    print("\n" + "=" * 50)
    print("  4. PROCHAINES ETAPES")
    print("=" * 50)
    print("""
  python init_project.py         # (deja fait)
  pip install -r requirements.txt
  python train.py                 # entraine et sauvegarde models/Model.pkl
  uvicorn main:app --reload      # lance l'API FastAPI
  python monitor.py               # detection de data drift

  git init && git add . && git commit -m "Initial MLOps pipeline"
  # Pousser sur GitHub -> CI/CD automatique (GitHub Actions)
""")

    summary()

    print("\n  [OK] Initialisation terminee.\n")


if __name__ == "__main__":
    main()

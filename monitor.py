"""
monitor.py - Detection de Data Drift avec Evidently AI.
Reference  = donnees d'entrainement (CSV versionnes par DVC).
Current    = predictions reelles collectees par l'API (predictions/predictions.jsonl).
En cas de drift > seuil : alerte (webhook) et retrain optionnel.
"""

import json
import os
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns

from train import build_features, DATA_DIR, TARGET

warnings.filterwarnings("ignore")

MODEL_PATH = "models/Model.pkl"
RANDOM_STATE = 42
CURRENT_DATA = os.environ.get("CURRENT_DATA", "predictions/predictions.jsonl")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.2"))


def load_reference_data():
    """Construit le jeu de reference a partir des donnees d'entrainement."""
    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")
    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")

    s1 = aviator[TARGET].dropna().reset_index(drop=True)
    s2 = multipliers["Multiplier"].dropna().reset_index(drop=True)
    blocks = np.repeat([0, 1], [len(s1), len(s2)])

    feat = build_features(pd.concat([s1, s2], ignore_index=True), blocks)
    X = feat.drop(columns=["multiplier", "block"])
    y = feat["multiplier"]

    _, X_ref, _, y_ref = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    ref = X_ref.copy()
    ref[TARGET] = y_ref.values
    return ref


def load_current_data():
    """Lit les predictions RELLES collectees par l'API."""
    if os.path.isfile(CURRENT_DATA):
        rows = [json.loads(l) for l in open(CURRENT_DATA, encoding="utf-8") if l.strip()]
        df = pd.DataFrame([r["features"] for r in rows])
        df[TARGET] = [r["prediction"] for r in rows]
        print(f"[INFO] {len(df)} predictions reelles chargees depuis {CURRENT_DATA}")
        return df

    print(f"[WARN] {CURRENT_DATA} introuvable -> simulation depuis le dataset")
    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")
    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")

    s1 = aviator[TARGET].dropna().reset_index(drop=True)
    s2 = multipliers["Multiplier"].dropna().reset_index(drop=True)
    blocks = np.repeat([0, 1], [len(s1), len(s2)])

    feat = build_features(pd.concat([s1, s2], ignore_index=True), blocks)
    X = feat.drop(columns=["multiplier", "block"])
    y = feat["multiplier"]

    _, X_cur, _, y_cur = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE + 1)
    cur = X_cur.copy()
    cur[TARGET] = y_cur.values
    return cur


def send_alert(drift_share: float):
    """Alerte console + webhook Slack optionnel (SLACK_WEBHOOK)."""
    print(f"[ALERTE] Data drift au-dessus du seuil : {drift_share:.2%} (seuil {DRIFT_THRESHOLD:.0%})")
    webhook = os.environ.get("SLACK_WEBHOOK")
    if webhook:
        import urllib.request

        payload = json.dumps({"text": f"[Aviator] Drift detecte : {drift_share:.2%}"}).encode()
        req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)


def retrain_and_version():
    """Relance l'entrainement puis versionne les nouvelles donnees avec DVC."""
    print("[INFO] Retrain : train.py")
    subprocess.run([sys.executable, "train.py"], check=True)
    print("[INFO] Versionnement DVC")
    subprocess.run(
        ["dvc", "add", "data/aviator_dataset.csv", "data/multipliers.csv"], check=True
    )
    subprocess.run(["dvc", "commit"], check=True)


def run_drift_detection():
    print("[INFO] Chargement de la reference...")
    reference = load_reference_data()

    print("[INFO] Chargement des donnees courantes...")
    current = load_current_data()

    print("[INFO] Generation du rapport de Data Drift...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")
    print("[OK] Rapport de drift sauvegarde -> drift_report.html")

    print("[INFO] Generation du rapport de performance regression...")
    try:
        model = joblib.load(MODEL_PATH)
        perf_report = Report(metrics=[RegressionPreset()])
        perf_report.run(reference_data=reference, current_data=current, model=model)
        perf_report.save_html("regression_performance.html")
        print("[OK] Rapport de performance sauvegarde -> regression_performance.html")
    except Exception as e:
        print(f"[WARN] Rapport de performance ignore : {e}")

    print("[INFO] Execution des tests de drift...")
    suite = TestSuite(tests=[TestShareOfDriftedColumns(lte=DRIFT_THRESHOLD), TestColumnDrift()])
    suite.run(reference_data=reference, current_data=current)
    suite.save_html("drift_tests.html")

    drift_share = None
    for t in suite.as_dict()["tests"]:
        if "drift_share" in t.get("parameters", {}).get("features", {}):
            drift_share = t["parameters"]["features"]["drift_share"]

    if drift_share is None:
        print("[WARN] Seuil de drift introuvable dans les resultats")
        return

    print(f"[INFO] Colonnes driftes : {drift_share:.2%} (seuil {DRIFT_THRESHOLD:.0%})")
    if drift_share > DRIFT_THRESHOLD:
        send_alert(drift_share)
        if os.environ.get("RETRAIN_ON_DRIFT", "0") == "1":
            retrain_and_version()
        raise SystemExit(1)


if __name__ == "__main__":
    run_drift_detection()

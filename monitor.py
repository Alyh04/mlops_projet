"""
monitor.py - Detection de Data Drift avec Evidently AI.
Compare les donnees d'apprentissage (reference) aux nouvelles predictions (current).
"""

import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_PATH = "models/Model.pkl"
TARGET = "target"
RANDOM_STATE = 42
DRIFT_OUTPUT = "drift_report.html"


def load_reference_data():
    """Construit le jeu de reference (features + target)."""
    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")
    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")

    all_mult = pd.concat(
        [aviator[TARGET], multipliers["Multiplier"]], axis=0, ignore_index=True
    ).dropna()

    feat = pd.DataFrame({"multiplier": all_mult}).sort_index()
    feat["mult_lag1"] = feat["multiplier"].shift(1)
    feat["mult_lag2"] = feat["multiplier"].shift(2)
    feat["mult_lag3"] = feat["multiplier"].shift(3)
    feat["roll_mean_3"] = feat["multiplier"].rolling(3).mean()
    feat["roll_mean_5"] = feat["multiplier"].rolling(5).mean()
    feat["roll_std_3"] = feat["multiplier"].rolling(3).std()
    feat["roll_std_5"] = feat["multiplier"].rolling(5).std()
    feat["roll_max_3"] = feat["multiplier"].rolling(3).max()
    feat["roll_min_3"] = feat["multiplier"].rolling(3).min()
    feat["ewm_mean"] = feat["multiplier"].ewm(span=5).mean()
    feat["mult_ratio_lag1"] = feat["multiplier"] / (feat["mult_lag1"] + 1e-9)

    feat = feat.dropna().reset_index(drop=True)

    X = feat.drop(columns=["multiplier"])
    y = feat["multiplier"]

    _, X_ref, _, y_ref = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    ref = X_ref.copy()
    ref[TARGET] = y_ref.values
    return ref


def load_current_data():
    """Simule des donnees current recentes a partir du dataset."""
    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")
    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")

    all_mult = pd.concat(
        [aviator[TARGET], multipliers["Multiplier"]], axis=0, ignore_index=True
    ).dropna()

    feat = pd.DataFrame({"multiplier": all_mult}).sort_index()
    feat["mult_lag1"] = feat["multiplier"].shift(1)
    feat["mult_lag2"] = feat["multiplier"].shift(2)
    feat["mult_lag3"] = feat["multiplier"].shift(3)
    feat["roll_mean_3"] = feat["multiplier"].rolling(3).mean()
    feat["roll_mean_5"] = feat["multiplier"].rolling(5).mean()
    feat["roll_std_3"] = feat["multiplier"].rolling(3).std()
    feat["roll_std_5"] = feat["multiplier"].rolling(5).std()
    feat["roll_max_3"] = feat["multiplier"].rolling(3).max()
    feat["roll_min_3"] = feat["multiplier"].rolling(3).min()
    feat["ewm_mean"] = feat["multiplier"].ewm(span=5).mean()
    feat["mult_ratio_lag1"] = feat["multiplier"] / (feat["mult_lag1"] + 1e-9)

    feat = feat.dropna().reset_index(drop=True)

    X = feat.drop(columns=["multiplier"])
    y = feat["multiplier"]

    _, X_cur, _, y_cur = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE + 1
    )
    cur = X_cur.copy()
    cur[TARGET] = y_cur.values
    return cur


def run_drift_detection():
    """Execute les rapports Evidently et sauvegarde le HTML."""
    print("[INFO] Chargement des donnees de reference...")
    reference = load_reference_data()

    print("[INFO] Chargement des donnees courantes (simulation)...")
    current = load_current_data()

    print("[INFO] Generation du rapport de Data Drift...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html(DRIFT_OUTPUT)
    print(f"[OK] Rapport de drift sauvegarde -> {DRIFT_OUTPUT}")

    print("[INFO] Generation du rapport de performance regression...")
    try:
        model = joblib.load(MODEL_PATH)
        perf_report = Report(metrics=[RegressionPreset()])
        perf_report.run(
            reference_data=reference,
            current_data=current,
            model=model,
        )
        perf_report.save_html("regression_performance.html")
        print("[OK] Rapport de performance sauvegarde -> regression_performance.html")
    except Exception as e:
        print(f"[WARN] Rapport de performance ignore : {e}")

    print("[INFO] Execution des tests de drift...")
    suite = TestSuite(tests=[TestColumnDrift(), TestShareOfDriftedColumns()])
    suite.run(reference_data=reference, current_data=current)
    suite.save_html("drift_tests.html")
    print("[OK] Tests de drift sauvegardes -> drift_tests.html")


if __name__ == "__main__":
    run_drift_detection()

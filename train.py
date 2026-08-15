"""
train.py - Entrainement du modele Aviator.
Charge et fusionne aviator_dataset.csv + multipliers.csv,
applique du feature engineering, entraine un XGBoost,
enregistre les metriques dans MLflow et exporte Model.pkl.
"""

import os
import hashlib
import warnings
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/Model.pkl"
TARGET = "target"
MLFLOW_EXPERIMENT = "Aviator_Prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def file_md5(path: str) -> str:
    """Empreinte MD5 = version des donnees (a remplacer par dvc.get_hash si besoin)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_features(series: pd.Series, blocks: np.ndarray) -> pd.DataFrame:
    """Features lag/rolling calculees SANS croiser la frontiere entre les 2 blocs."""
    feat = pd.DataFrame({"multiplier": series, "block": blocks})
    parts = []
    for _, sub in feat.groupby("block", sort=False):
        m = sub["multiplier"]
        p = pd.DataFrame(index=sub.index)
        p["mult_lag1"] = m.shift(1)
        p["mult_lag2"] = m.shift(2)
        p["mult_lag3"] = m.shift(3)
        p["roll_mean_3"] = m.rolling(3).mean()
        p["roll_mean_5"] = m.rolling(5).mean()
        p["roll_std_3"] = m.rolling(3).std()
        p["roll_std_5"] = m.rolling(5).std()
        p["roll_max_3"] = m.rolling(3).max()
        p["roll_min_3"] = m.rolling(3).min()
        p["ewm_mean"] = m.ewm(span=5).mean()
        parts.append(p)
    feats = pd.concat(parts).sort_index()
    feats["mult_ratio_lag1"] = series / (feats["mult_lag1"] + 1e-9)
    return pd.concat([feat, feats], axis=1).dropna().reset_index(drop=True)


def load_data():
    """Charge les deux jeux, fusionne et construit les features."""
    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")
    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")

    s1 = aviator[TARGET].dropna().reset_index(drop=True)
    s2 = multipliers["Multiplier"].dropna().reset_index(drop=True)
    blocks = np.repeat([0, 1], [len(s1), len(s2)])

    feat = build_features(pd.concat([s1, s2], ignore_index=True), blocks)
    X = feat.drop(columns=["multiplier", "block"])
    y = feat["multiplier"]
    print(f"[INFO] Features shape : {X.shape}")
    return X, y


def train():
    X, y = load_data()

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        model = XGBRegressor(**params, verbosity=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics({"mae": mae, "mse": mse, "r2": r2})
        mlflow.sklearn.log_model(
            model, "model",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
        )
        mlflow.set_tag("model", "XGBRegressor")

        for f in ("aviator_dataset.csv", "multipliers.csv"):
            mlflow.log_param(f"data_md5/{f}", file_md5(f"{DATA_DIR}/{f}"))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"\n[OK] Modele sauvegarde -> {MODEL_PATH}")
        print(f"    MAE : {mae:.4f}  |  MSE : {mse:.4f}  |  R2 : {r2:.4f}")


if __name__ == "__main__":
    train()

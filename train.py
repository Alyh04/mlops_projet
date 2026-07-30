"""
train.py - Entrainement du modele Aviator.
Charge et fusionne aviator_dataset.csv + multipliers.csv,
applique du feature engineering, entraine un XGBoost,
enregistre les metriques dans MLflow et exporte Model.pkl.
"""

import os
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


def load_data():
    """Charge les deux jeux de donnees et les fusionne."""
    aviator = pd.read_csv(f"{DATA_DIR}/aviator_dataset.csv")
    multipliers = pd.read_csv(f"{DATA_DIR}/multipliers.csv")

    print("[INFO] aviator_dataset.csv - shape:", aviator.shape)
    print("[INFO] multipliers.csv   - shape:", multipliers.shape)

    all_multipliers = pd.concat(
        [aviator[TARGET], multipliers["Multiplier"]],
        axis=0,
        ignore_index=True,
    ).dropna()

    print(f"[INFO] Total multiplicateurs unifies : {len(all_multipliers)}")

    feat = pd.DataFrame({"multiplier": all_multipliers})
    feat = feat.sort_index()

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

    print(f"[INFO] Features shape        : {X.shape}")
    print(f"[INFO] Features columns       : {list(X.columns)}")
    return X, y


def train():
    """Entraine un XGBoost et logge dans MLflow."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

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

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"\n[OK] Modele sauvegarde -> {MODEL_PATH}")
        print(f"    MAE : {mae:.4f}  |  MSE : {mse:.4f}  |  R2 : {r2:.4f}")


if __name__ == "__main__":
    train()

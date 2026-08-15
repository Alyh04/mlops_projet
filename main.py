"""
main.py - API FastAPI de prediction Aviator.
Charge models/Model.pkl et expose:
  GET  /        -> health-check
  GET  /model-info -> metadonnees du modele
  POST /predict -> prediction du multiplicateur
Chaque requete est journalisee dans predictions/predictions.jsonl
pour alimenter le monitoring (feedback loop).
"""

import json
import os
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/Model.pkl"
PREDICTIONS_DIR = "predictions"
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "predictions.jsonl")

app = FastAPI(title="Aviator Predictor API", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
    FEATURE_NAMES = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else None
except Exception as e:
    model = None
    FEATURE_NAMES = None
    print(f"[WARN] Impossible de charger le modele : {e}")


def log_prediction(input_data: dict, prediction: float) -> str:
    """Sauvegarde de chaque requete -> source de la boucle de retrain."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    record = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": input_data,
        "prediction": round(prediction, 4),
    }
    with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record["prediction_id"]


class PredictInput(BaseModel):
    mult_lag1: float = Field(..., description="Multiplicateur a t-1")
    mult_lag2: float = Field(..., description="Multiplicateur a t-2")
    mult_lag3: float = Field(..., description="Multiplicateur a t-3")
    roll_mean_3: float = Field(..., description="Moyenne mobile sur 3")
    roll_mean_5: float = Field(..., description="Moyenne mobile sur 5")
    roll_std_3: float = Field(..., description="Ecart-type mobile 3")
    roll_std_5: float = Field(..., description="Ecart-type mobile 5")
    roll_max_3: float = Field(..., description="Max mobile sur 3")
    roll_min_3: float = Field(..., description="Min mobile sur 3")
    ewm_mean: float = Field(..., description="Moyenne exponentielle span=5")
    mult_ratio_lag1: float = Field(..., description="Ratio multiplicateur / lag1")


class PredictOutput(BaseModel):
    prediction: float
    prediction_id: str
    features_used: list[str]


@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": str(pd.Timestamp.now()),
    }


@app.get("/model-info", tags=["Health"])
def model_info():
    return {"model_loaded": model is not None, "features": FEATURE_NAMES}


@app.post("/predict", response_model=PredictOutput, tags=["Prediction"])
def predict(input_data: PredictInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    df = pd.DataFrame([input_data.model_dump()])

    if FEATURE_NAMES is not None:
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes : {missing}",
            )
        df = df[FEATURE_NAMES]

    pred = float(model.predict(df)[0])
    pred_id = log_prediction(input_data.model_dump(), pred)
    return PredictOutput(
        prediction=round(pred, 4),
        prediction_id=pred_id,
        features_used=list(df.columns),
    )

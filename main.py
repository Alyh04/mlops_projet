"""
main.py - API FastAPI de prediction Aviator.
Charge models/Model.pkl et expose:
  GET  /        -> health-check
  POST /predict -> prediction du multiplicateur
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/Model.pkl"

app = FastAPI(title="Aviator Predictor API", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
    FEATURE_NAMES = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else None
except Exception as e:
    model = None
    FEATURE_NAMES = None
    print(f"[WARN] Impossible de charger le modele : {e}")


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
    features_used: list[str]


@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": str(pd.Timestamp.now()),
    }


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

    pred = model.predict(df)[0]
    return PredictOutput(prediction=round(float(pred), 4), features_used=list(df.columns))

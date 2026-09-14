"""
app.py - Interface Gradio pour le modele Aviator.

Reproduit exactement l'ingenierie de features de train.py.build_features
(les 11 features lag/rolling calculees SANS croiser la frontiere de bloc)
puis appelle le XGBRegressor exporte dans Model.pkl.
"""

import os

import numpy as np
import gradio as gr
import joblib

MODEL_CANDIDATES = ["Model.pkl", "models/Model.pkl"]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), MODEL_CANDIDATES[0])
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "mult_lag1",
    "mult_lag2",
    "mult_lag3",
    "roll_mean_3",
    "roll_mean_5",
    "roll_std_3",
    "roll_std_5",
    "roll_max_3",
    "roll_min_3",
    "ewm_mean",
    "mult_ratio_lag1",
]


def _ewm_span5(values):
    """EWM(span=5).mean() de pandas (adjust=True) au dernier point."""
    alpha = 2.0 / 6.0  # 2 / (span + 1)
    decay = 1.0 - alpha
    num = 0.0
    den = 0.0
    weight = 1.0
    for x in reversed(values):
        num += weight * x
        den += weight
        weight *= decay
    return num / den


def compute_features(values):
    """values : les 5 derniers multiplicateurs, du plus ancien au plus recent."""
    v = np.asarray([float(x) for x in values], dtype=float)
    if len(v) < 5:
        raise gr.Error("5 valeurs de multiplicateurs sont requises.")
    feats = {
        "mult_lag1": float(v[-2]),
        "mult_lag2": float(v[-3]),
        "mult_lag3": float(v[-4]),
        "roll_mean_3": float(np.mean(v[-3:])),
        "roll_mean_5": float(np.mean(v[-5:])),
        "roll_std_3": float(np.std(v[-3:], ddof=1)),
        "roll_std_5": float(np.std(v[-5:], ddof=1)),
        "roll_max_3": float(np.max(v[-3:])),
        "roll_min_3": float(np.min(v[-3:])),
        "ewm_mean": float(_ewm_span5(v)),
        "mult_ratio_lag1": float(v[-1] / (v[-2] + 1e-9)),
    }
    return [feats[f] for f in FEATURE_NAMES]


def predict(mult_n1, mult_n2, mult_n3, mult_n4, mult_n5):
    """mult_n1 = plus recent, mult_n5 = plus ancien des 5."""
    values = [mult_n5, mult_n4, mult_n3, mult_n2, mult_n1]
    feats = compute_features(values)
    prediction = float(model.predict([feats])[0])
    return round(prediction, 4)


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Multiplicateur n-1 (le plus recent)", value=1.08),
        gr.Number(label="Multiplicateur n-2", value=1.39),
        gr.Number(label="Multiplicateur n-3", value=0.96),
        gr.Number(label="Multiplicateur n-4", value=2.11),
        gr.Number(label="Multiplicateur n-5 (le plus ancien)", value=1.05),
    ],
    outputs=gr.Number(label="Multiplicateur predit"),
    title="Aviator Predictor",
    description=(
        "Entrez les 5 derniers multiplicateurs observes pour obtenir le "
        "multiplicateur predit par le modele XGBoost."
    ),
    examples=[[1.08, 1.39, 0.96, 2.11, 1.05], [1.22, 1.18, 0.88, 1.6, 0.97]],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
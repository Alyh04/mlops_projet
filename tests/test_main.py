"""
test_main.py - Tests unitaires des endpoints FastAPI (/ et /predict).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from fastapi.testclient import TestClient

client = TestClient(main.app)

VALID_PAYLOAD = {
    "mult_lag1": 2.0,
    "mult_lag2": 1.5,
    "mult_lag3": 1.2,
    "roll_mean_3": 1.57,
    "roll_mean_5": 1.60,
    "roll_std_3": 0.40,
    "roll_std_5": 0.50,
    "roll_max_3": 2.8,
    "roll_min_3": 1.1,
    "ewm_mean": 1.70,
    "mult_ratio_lag1": 1.20,
}


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_valid_payload():
    r = client.post("/predict", json=VALID_PAYLOAD)
    if main.model is None:
        assert r.status_code == 503
    else:
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert "prediction_id" in body
        assert set(body["features_used"]) == set(main.FEATURE_NAMES)


def test_predict_missing_fields():
    r = client.post("/predict", json={"mult_lag1": 1.0})
    assert r.status_code == 422


def test_predict_wrong_types():
    payload = dict(VALID_PAYLOAD)
    payload["mult_lag1"] = "not-a-number"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

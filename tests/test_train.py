"""
test_train.py - Tests unitaires de la preparation des donnees (train.py).
Sont ignores en CI si les donnees brutes ne sont pas restaurees (dvc pull).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PRESENT = os.path.isfile(os.path.join("data", "aviator_dataset.csv")) and os.path.isfile(
    os.path.join("data", "multipliers.csv")
)

pytestmark = pytest.mark.skipif(
    not DATA_PRESENT,
    reason="Donnees brutes absentes : executer 'dvc pull' au prealable",
)

from train import load_data  # noqa: E402

FEATURE_COLS = [
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


def test_load_data_shape():
    X, y = load_data()
    assert len(X) == len(y) > 0
    assert list(X.columns) == FEATURE_COLS


def test_load_data_no_nan():
    X, _ = load_data()
    assert X.isna().sum().sum() == 0

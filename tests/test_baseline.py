"""Synthetic checks for development-only model evaluation."""

import json

import numpy as np
import pandas as pd

from heart_ml import baseline
from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA, TARGET
from heart_ml.split import create_membership


def test_development_evaluation_excludes_holdout_and_predicts_valid_probabilities(tmp_path):
    config = json.loads(open("configs/default.json").read())
    rows = 120
    frame = pd.DataFrame({"id": np.arange(rows)})
    rng = np.random.default_rng(4)
    for column in NUMERIC:
        frame[column] = rng.normal(size=rows)
    for column in CATEGORICAL:
        frame[column] = rng.integers(0, 3, size=rows)
    frame[TARGET] = np.where(np.arange(rows) % 2, "Presence", "Absence")
    frame = frame[SCHEMA + [TARGET]]
    membership = create_membership(frame, config)
    development = baseline.development_rows(frame, membership, config)
    folds, predictions = baseline.evaluate_model(development, config, "logistic")
    holdout_ids = set(membership.loc[membership.partition == "holdout", "id"])
    assert set(predictions.id).isdisjoint(holdout_ids)
    assert predictions.probability_presence.between(0, 1).all()
    assert len(folds) == 3
    assert folds.roc_auc.notna().all()


def test_cli_models_are_exactly_four():
    assert baseline.SUPPORTED_MODELS == ("logistic", "decision_tree", "adaboost", "random_forest")

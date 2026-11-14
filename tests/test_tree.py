"""Small synthetic checks of selection, leakage protection and saved tree behavior."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA, TARGET
from heart_ml.preprocess import build_pipeline
from heart_ml.tree_experiment import candidates_from_plan, run_search, select_candidate


def test_plan_validation_and_exact_ties():
    plan = json.loads(open("configs/decision_tree_search.json").read())
    assert len(candidates_from_plan(plan)) == 8
    for bad in ([0], [True], [3, 3], []):
        with pytest.raises(ValueError):
            candidates_from_plan({**plan, "max_depth": bad})
    with pytest.raises(ValueError, match="eight"):
        candidates_from_plan({**plan, "max_depth": [1, 2, 3, 4, 5]})
    scores = pd.DataFrame([
        {"roc_auc_mean": 0.8, "max_depth": 8, "min_samples_leaf": 2},
        {"roc_auc_mean": 0.8, "max_depth": 3, "min_samples_leaf": 2},
        {"roc_auc_mean": 0.8, "max_depth": 3, "min_samples_leaf": 5},
    ])
    assert select_candidate(scores).name == 2


def test_search_saves_reproducible_model_and_excludes_holdout(tmp_path):
    config = json.loads(open("configs/default.json").read())
    plan = json.loads(open("configs/decision_tree_search.json").read())
    plan.update(max_depth=[1, 3], min_samples_leaf=[2])
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"id": np.arange(150)})
    for column in NUMERIC:
        frame[column] = rng.normal(size=len(frame))
    for column in CATEGORICAL:
        frame[column] = rng.integers(0, 3, size=len(frame))
    frame[TARGET] = np.where(frame["Age"] > 0, "Presence", "Absence")
    source = tmp_path / "train.csv"
    frame[SCHEMA + [TARGET]].to_csv(source, index=False)
    config["data"].update(train=str(source), test=str(tmp_path / "absent.csv"))
    config["splits_dir"] = str(tmp_path / "splits")
    output, result = run_search(config, plan, tmp_path / "run")
    assert result["selected_parameters"]["max_depth"] == 1
    assert result["selected_summary"]["metrics"]["roc_auc"]["mean"] > 0.95
    assert result["selected_fold_repeat_matches"]
    metadata = json.loads((output / "selected/run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["holdout_evaluated"] is metadata["test_csv_used"] is False
    membership = pd.read_csv(tmp_path / "splits/membership.csv")
    oof = pd.read_csv(output / "selected/development_predictions.csv")
    holdout_ids = set(membership.loc[membership.partition == "holdout", "id"])
    assert holdout_ids.isdisjoint(oof.id)
    assert len(pd.read_csv(output / "search_folds.csv")) == 6
    pipeline = joblib.load(output / "selected/decision_tree_pipeline.joblib")
    assert "scale" not in pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps
    features = config["numeric_features"] + config["categorical_features"]
    probability = pipeline.predict_proba(frame[features])
    np.testing.assert_array_equal(probability, joblib.load(
        output / "selected/decision_tree_pipeline.joblib"
    ).predict_proba(frame[features]))
    assert np.unique(probability[:, 1]).size > 1
    assert build_pipeline("decision_tree", config).named_steps["model"].random_state == 42
    with pytest.raises(FileExistsError):
        run_search(config, plan, output)

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier

from heart_ml.adaboost_experiment import candidates_from_plan, run_search, select_candidate
from heart_ml.preprocess import build_pipeline
from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA, TARGET


def test_adaboost_pipeline_is_cloneable_and_uses_stumps():
    config = json.loads(open("configs/default.json").read())
    pipeline = build_pipeline("adaboost", config)
    assert isinstance(clone(pipeline).named_steps["model"], AdaBoostClassifier)
    assert pipeline.named_steps["model"].estimator is None
    assert pipeline.named_steps["model"].n_estimators == 50


def test_candidate_validation_and_deterministic_tie_break():
    plan = json.loads(open("configs/adaboost_search.json").read())
    assert len(candidates_from_plan(plan)) == 6
    with pytest.raises(ValueError, match="At most eight"):
        candidates_from_plan({**plan, "n_estimators": [1, 2, 3, 4, 5]})
    table = pd.DataFrame([
        {"roc_auc_mean": .8, "n_estimators": 50, "learning_rate": .5},
        {"roc_auc_mean": .8, "n_estimators": 25, "learning_rate": 1.0},
    ])
    assert select_candidate(table)["n_estimators"] == 25


@pytest.mark.parametrize("counts", [[0], [-1], [True], [1, 1]])
def test_invalid_estimator_counts_rejected(counts):
    plan = json.loads(open("configs/adaboost_search.json").read())
    with pytest.raises(ValueError):
        candidates_from_plan({**plan, "n_estimators": counts})


@pytest.mark.parametrize("rates", [[0], [-0.1], [True], [float("nan")], [float("inf")], [0.5, 0.5]])
def test_invalid_learning_rates_rejected(rates):
    plan = json.loads(open("configs/adaboost_search.json").read())
    with pytest.raises(ValueError):
        candidates_from_plan({**plan, "learning_rate": rates})


def test_pipeline_handles_missing_numeric_and_unseen_categories():
    config = json.loads(open("configs/default.json").read())
    config["models"]["adaboost"]["n_estimators"] = 3
    columns = config["numeric_features"] + config["categorical_features"]
    data = pd.DataFrame(np.arange(4 * len(columns)).reshape(4, len(columns)), columns=columns)
    data.loc[0, config["numeric_features"][0]] = np.nan
    y = [0, 1, 0, 1]
    pipeline = build_pipeline("adaboost", config).fit(data, y)
    new = data.iloc[:2].copy()
    new.loc[:, config["categorical_features"][0]] = 999
    probabilities = pipeline.predict_proba(new)[:, np.flatnonzero(pipeline.classes_ == 1).item()]
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()


def test_small_search_persists_selected_pipeline_without_test_input(tmp_path, monkeypatch):
    config = json.loads(open("configs/default.json").read())
    rows = 120
    frame = pd.DataFrame({"id": np.arange(rows)})
    for column in NUMERIC:
        frame[column] = np.arange(rows, dtype=float)
    for column in CATEGORICAL:
        frame[column] = np.arange(rows) % 3
    frame[TARGET] = np.where(np.arange(rows) % 2, "Presence", "Absence")
    source = tmp_path / "train.csv"
    frame[SCHEMA + [TARGET]].to_csv(source, index=False)
    config["data"]["train"] = str(source)
    config["data"]["test"] = str(tmp_path / "intentionally-missing-test.csv")
    config["splits_dir"] = str(tmp_path / "splits")
    monkeypatch.chdir(tmp_path)
    plan = {"selection_metric": "roc_auc", "tie_break": "fewer estimators, then lower learning rate",
            "n_estimators": [2, 3], "learning_rate": [0.5]}
    (tmp_path / "reports/generated/adaboost_search").mkdir(parents=True)
    output, selection = run_search(config, plan)
    table = pd.read_csv(output / "search_results.csv")
    assert len(table) == 2
    assert selection["holdout_evaluated"] is False
    pipeline_file = Path(selection["selected_pipeline"])
    if not pipeline_file.is_absolute():
        pipeline_file = tmp_path / pipeline_file
    assert pipeline_file.exists()
    assert (output / "search_folds.csv").exists()
    pipeline = joblib.load(pipeline_file)
    features = config["numeric_features"] + config["categorical_features"]
    membership = pd.read_csv(tmp_path / "splits/membership.csv")
    holdout_ids = set(membership.loc[membership.partition == "holdout", "id"])
    predictions = pd.read_csv(output / "selected/development_predictions.csv")
    assert set(predictions["id"]).isdisjoint(holdout_ids)
    development = frame.loc[frame["id"].isin(predictions["id"]), features]
    selected_config = json.loads((output / "selected/config.json").read_text())
    labels = frame.loc[frame["id"].isin(predictions["id"]), TARGET].eq("Presence").astype(int)
    fresh_pipeline = build_pipeline("adaboost", selected_config).fit(development, labels)
    persisted_positive = np.flatnonzero(pipeline.classes_ == 1).item()
    fresh_positive = np.flatnonzero(fresh_pipeline.classes_ == 1).item()
    np.testing.assert_allclose(
        pipeline.predict_proba(development)[:, persisted_positive],
        fresh_pipeline.predict_proba(development)[:, fresh_positive],
    )
    inference = frame.loc[:1, features].copy()
    inference.loc[:, config["categorical_features"][0]] = 999
    inference.loc[0, config["numeric_features"][0]] = np.nan
    assert np.isfinite(pipeline.predict_proba(inference)).all()
    assert not (tmp_path / "intentionally-missing-test.csv").exists()
    second_output, _ = run_search(config, plan)
    assert second_output != output


def test_later_candidate_failure_keeps_completed_outputs(tmp_path, monkeypatch):
    import heart_ml.adaboost_experiment as experiment
    from heart_ml.baseline import run_baseline as real_run

    config = json.loads(open("configs/default.json").read())
    rows = 120
    frame = pd.DataFrame({"id": np.arange(rows)})
    for column in NUMERIC:
        frame[column] = np.arange(rows, dtype=float)
    for column in CATEGORICAL:
        frame[column] = np.arange(rows) % 3
    frame[TARGET] = np.where(np.arange(rows) % 2, "Presence", "Absence")
    source = tmp_path / "train.csv"
    frame[SCHEMA + [TARGET]].to_csv(source, index=False)
    config["data"]["train"] = str(source)
    config["data"]["test"] = str(tmp_path / "missing-test.csv")
    config["splits_dir"] = str(tmp_path / "splits")
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second-candidate failure")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(experiment, "run_baseline", fail_second)
    output = tmp_path / "failed-search"
    plan = {"selection_metric": "roc_auc", "tie_break": "fewer estimators, then lower learning rate",
            "n_estimators": [2, 3], "learning_rate": [0.5]}
    with pytest.raises(RuntimeError, match="synthetic second"):
        experiment.run_search(config, plan, output)
    assert (output / "candidate_0/cv_results.csv").exists()
    status = json.loads((output / "search_metadata.json").read_text())
    assert status["status"] == "failed"

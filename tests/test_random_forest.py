import json
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA
from heart_ml.baseline import evaluate_model
from heart_ml.preprocess import build_pipeline
from heart_ml.random_forest_experiment import candidates_from_plan, select_candidate
from heart_ml.random_forest_experiment import run_search


def test_constructor_is_cloneable_and_has_bounded_cpu_defaults():
    config = json.loads(open("configs/default.json").read())
    pipeline = build_pipeline("random_forest", config)
    assert clone(pipeline).named_steps["model"].random_state == 42
    assert pipeline.named_steps["model"].n_jobs == 1
    assert pipeline.named_steps["model"].n_estimators == 100
    numeric = pipeline.named_steps["preprocess"].transformers[0][1]
    assert "scale" not in numeric.named_steps
    assert not hasattr(pipeline.named_steps["model"], "estimators_")
    config["threads"] = 2
    config["models"]["random_forest"].pop("n_jobs")
    assert build_pipeline("random_forest", config).named_steps["model"].n_jobs == 2
    for value in (-1, True, 1):
        bad = json.loads(open("configs/default.json").read())
        bad["threads"] = 2
        bad["models"]["random_forest"]["n_jobs"] = value
        with pytest.raises(ValueError, match="n_jobs"):
            build_pipeline("random_forest", bad)


def test_search_grid_validation_and_tie_break():
    plan = json.loads(open("configs/random_forest_search.json").read())
    candidates = candidates_from_plan(plan, 1)
    assert len(candidates) == 4
    assert all(c["n_jobs"] == 1 for c in candidates)
    with pytest.raises(ValueError):
        candidates_from_plan({**plan, "max_depth": [40]}, 1)
    for invalid in ("4", True, 0, 9):
        with pytest.raises(ValueError, match="max_candidates"):
            candidates_from_plan({**plan, "max_candidates": invalid}, 1)
    with pytest.raises(ValueError, match="max_features"):
        candidates_from_plan({**plan, "max_features": 0.5}, 1)
    table = pd.DataFrame([
        {"roc_auc_mean": .8, "max_depth": 12, "min_samples_leaf": 50},
        {"roc_auc_mean": .8, "max_depth": 8, "min_samples_leaf": 200},
    ])
    assert select_candidate(table).max_depth == 8


def test_fitted_pipeline_missing_unseen_values_and_fold_probability_shape():
    config = json.loads(open("configs/default.json").read())
    config["models"]["random_forest"].update(n_estimators=3, max_depth=3, min_samples_leaf=1)
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({"id": np.arange(90), "partition": "development",
                          "fold": np.tile([0, 1, 2], 30)})
    for name in NUMERIC:
        frame[name] = rng.normal(size=len(frame))
    for name in CATEGORICAL:
        frame[name] = rng.integers(0, 3, size=len(frame))
    frame.loc[0, NUMERIC[0]] = np.nan
    frame.loc[0, CATEGORICAL[0]] = np.nan
    frame.loc[1, CATEGORICAL[0]] = 99
    frame["Heart Disease"] = np.where(frame["Age"].fillna(0) > 0, "Presence", "Absence")
    folds, predictions = evaluate_model(frame, config, "random_forest")
    assert len(folds) == 3
    assert predictions.probability_presence.between(0, 1).all()
    fitted = build_pipeline("random_forest", config)
    features = NUMERIC + CATEGORICAL
    y = (frame["Heart Disease"] == "Presence").astype(int)
    fitted.fit(frame[features], y)
    buffer = BytesIO()
    joblib.dump(fitted, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)
    inference = frame[features].copy()
    inference.loc[2, CATEGORICAL[0]] = 101
    p = loaded.predict_proba(inference)[:, list(loaded.classes_).index(1)]
    assert p.shape == (len(frame),)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p, fitted.predict_proba(inference)[:, list(fitted.classes_).index(1)])


def _synthetic_data(tmp_path):
    config = json.loads(open("configs/default.json").read())
    rng = np.random.default_rng(22)
    n = 120
    frame = pd.DataFrame({"id": np.arange(n)})
    for name in NUMERIC:
        frame[name] = rng.normal(size=n)
    for name in CATEGORICAL:
        frame[name] = rng.integers(0, 3, size=n)
    frame["Heart Disease"] = np.where(frame["Age"] > 0, "Presence", "Absence")
    source = tmp_path / "train.csv"
    frame[SCHEMA + ["Heart Disease"]].to_csv(source, index=False)
    config["data"].update(train=str(source), test=str(tmp_path / "absent-test.csv"))
    config["splits_dir"] = str(tmp_path / "splits")
    config["models"]["random_forest"].update(n_estimators=2, max_depth=3, min_samples_leaf=1)
    plan = json.loads(open("configs/random_forest_search.json").read())
    plan.update(max_depth=[2, 3], min_samples_leaf=[1], n_estimators=2)
    return config, plan, frame


def test_search_outputs_and_refuses_existing_output(tmp_path):
    config, plan, frame = _synthetic_data(tmp_path)
    config["models"]["random_forest"].update(criterion="entropy", bootstrap=False)
    parent = tmp_path / "runs"
    parent.mkdir()
    output, result = run_search(config, plan, parent / "first")
    with pytest.raises(FileExistsError):
        run_search(config, plan, output)
    selected_config = json.loads((output / "selected/config.json").read_text())
    selected = selected_config["models"]["random_forest"]
    assert selected["n_estimators"] == 2
    assert selected["criterion"] == "entropy" and selected["bootstrap"] is False
    assert result["selected_parameters"] == selected
    pipeline = joblib.load(output / "selected/random_forest_pipeline.joblib")
    features = NUMERIC + CATEGORICAL
    membership = pd.read_csv(tmp_path / "splits/membership.csv")
    development = frame[frame.id.isin(membership.loc[membership.partition == "development", "id"])]
    fresh = build_pipeline("random_forest", selected_config)
    y = (development["Heart Disease"] == "Presence").astype(int)
    fresh.fit(development[features], y)
    np.testing.assert_allclose(
        pipeline.predict_proba(development[features]),
        fresh.predict_proba(development[features]), rtol=0, atol=1e-12,
    )
    loaded = joblib.load(output / "selected/random_forest_pipeline.joblib")
    np.testing.assert_allclose(pipeline.predict_proba(development[features]),
                               loaded.predict_proba(development[features]))


def test_search_default_parent_allocates_fresh_run_directories(tmp_path, monkeypatch):
    config, plan, _ = _synthetic_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    first, _ = run_search(config, plan)
    second, _ = run_search(config, plan)
    assert first.parent == second.parent
    assert first != second
    assert (first / "plan.json").is_file() and (second / "plan.json").is_file()


def test_search_preserves_non_search_settings_and_records_later_failure(tmp_path, monkeypatch):
    config, plan, _ = _synthetic_data(tmp_path)
    config["models"]["random_forest"].update(criterion="entropy", bootstrap=False)
    original_evaluate = __import__("heart_ml.random_forest_experiment", fromlist=["evaluate_model"]).evaluate_model
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic candidate failure")
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr("heart_ml.random_forest_experiment.evaluate_model", fail_second)
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        run_search(config, plan, output)
    assert len(pd.read_csv(output / "search_results.csv")) == 1
    metadata = json.loads((output / "search_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["plan_sha256"] and metadata["config_sha256"]
    assert metadata["elapsed_seconds"] >= 0

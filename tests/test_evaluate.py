import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from importlib.metadata import version
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from heart_ml.audit import FEATURES, ID, SCHEMA, TARGET, sha256
from heart_ml.baseline import write_json
from heart_ml.evaluate import positive_probabilities, run_evaluation
import heart_ml.evaluate as evaluation
from heart_ml.preprocess import build_pipeline
from heart_ml.split import create_membership, expected_metadata


def _fixture(tmp_path):
    rng = np.random.default_rng(5)
    n = 120
    config = {
        "data": {"train": str(tmp_path / "train.csv"), "test": str(tmp_path / "missing-test.csv"),
                 "target": TARGET, "id": ID},
        "seed": 42, "holdout_fraction": 0.2, "folds": 3, "threshold": 0.5,
        "selection_metric": "roc_auc", "positive_label": "Presence", "negative_label": "Absence",
        "numeric_features": ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"],
        "categorical_features": ["Sex", "Chest pain type", "FBS over 120", "EKG results",
                                 "Exercise angina", "Slope of ST", "Thallium"],
        "models": {"adaboost": {"n_estimators": 50, "learning_rate": 1.0}},
        "threads": 1, "splits_dir": str(tmp_path / "splits"),
    }
    data = {name: rng.integers(0, 5, n) for name in config["categorical_features"]}
    data.update({name: rng.normal(size=n) for name in config["numeric_features"]})
    data[ID] = np.arange(1000, 1000 + n)
    data[TARGET] = np.where(np.arange(n) % 2, "Presence", "Absence")
    frame = pd.DataFrame(data)[[*SCHEMA, TARGET]]
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    membership = create_membership(frame, config)
    membership_path = split_dir / "membership.csv"
    membership.to_csv(membership_path, index=False)
    split_meta = expected_metadata(config, train_path)
    split_meta["membership_sha256"] = sha256(membership_path)
    (split_dir / "metadata.json").write_text(json.dumps(split_meta))

    ids = membership.loc[membership.partition.eq("development"), ID]
    dev = frame[frame[ID].isin(ids)]
    pipeline = build_pipeline("adaboost", config)
    pipeline.fit(dev[FEATURES], dev[TARGET].map({"Absence": 0, "Presence": 1}))

    run = tmp_path / "source"
    run.mkdir()
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    import heart_ml.baseline as baseline
    code = baseline.code_metadata()
    packages = {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl")}
    meta = {
        "status": "completed", "evaluation": "development_cross_validation", "model": "adaboost",
        "feature_set": "original", "holdout_evaluated": False, "test_csv_used": False,
        "pipeline_fit_partition": "development_only", "threshold": 0.5, "positive_class": 1,
        "config_sha256": sha256(config_path), "configured_parameters": config["models"]["adaboost"],
        "packages": packages, "code": code, "train_sha256": sha256(train_path),
        "membership_sha256": sha256(membership_path), "split_metadata": split_meta,
        "development_rows": int((membership.partition == "development").sum()),
        "holdout_rows_excluded": int((membership.partition == "holdout").sum()),
    }
    (run / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    import joblib
    joblib.dump(pipeline, run / "adaboost_pipeline.joblib")
    return run, config, frame, membership


class FakePipeline:
    classes_ = np.array([1, 0])

    def __init__(self, values):
        self.values = np.asarray(values)

    def predict_proba(self, X):
        return self.values


def test_positive_probability_uses_class_order_and_threshold_inclusive():
    values = positive_probabilities(FakePipeline([[0.8, 0.2], [0.5, 0.5]]), [1, 2])
    np.testing.assert_array_equal(values, [0.8, 0.5])
    np.testing.assert_array_equal(values >= 0.5, [1, 1])


def test_frozen_candidate_inference_accepts_exact_11_feature_schema():
    columns = [
        "Age", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro",
        "Sex", "Chest pain type", "EKG results", "Exercise angina", "Slope of ST", "Thallium",
    ]

    class SchemaCheckingPipeline(FakePipeline):
        def predict_proba(self, X):
            assert list(X.columns) == columns
            assert X.shape == (2, 11)
            return self.values

    values = positive_probabilities(
        SchemaCheckingPipeline([[0.2, 0.8], [0.7, 0.3]]),
        pd.DataFrame(np.zeros((2, 11)), columns=columns),
    )
    np.testing.assert_array_equal(values, [0.2, 0.7])


@pytest.mark.parametrize("values", [[[0.2]], [[float("nan"), 0.8]], [[1.1, -0.1]]])
def test_positive_probability_rejects_bad_output(values):
    with pytest.raises(ValueError):
        positive_probabilities(FakePipeline(values), [1])


def test_end_to_end_load_only_holdout_evaluation(tmp_path, monkeypatch):
    run, config, frame, membership = _fixture(tmp_path)
    output = tmp_path / "result"
    observed = []
    original_predict = Pipeline.predict_proba

    def recording_predict(self, X):
        observed.append((len(X), list(X.columns)))
        return original_predict(self, X)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Evaluation attempted to fit")

    monkeypatch.setattr(Pipeline, "predict_proba", recording_predict)
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    import pandas as pandas
    original_read_csv = pandas.read_csv

    def guarded_read(path, *args, **kwargs):
        assert "missing-test.csv" not in str(path)
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pandas, "read_csv", guarded_read)
    summary = run_evaluation(run, output, confirm_holdout=True, artifact_path=tmp_path / "artifacts/evaluated_pipeline.joblib")
    held = frame[frame[ID].isin(membership.loc[membership.partition.eq("holdout"), ID])]
    result = pd.read_csv(output / "holdout_predictions.csv")
    assert observed == [(len(held), FEATURES)]
    assert set(result[ID]) == set(held[ID])
    assert len(result) == len(held)
    assert (result.predicted_label == (result.probability_presence >= 0.5)).all()
    labels = held[TARGET].map({"Absence": 0, "Presence": 1}).to_numpy()
    pred = result.predicted_label.to_numpy()
    assert summary["metrics"]["roc_auc"] == roc_auc_score(labels, result.probability_presence)
    assert summary["metrics"]["average_precision"] == average_precision_score(labels, result.probability_presence)
    assert summary["metrics"]["accuracy"] == accuracy_score(labels, pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    assert summary["metrics"]["confusion_matrix"] == {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    assert (tmp_path / "artifacts/evaluated_pipeline.joblib").read_bytes() == (run / "adaboost_pipeline.joblib").read_bytes()
    assert not Path(config["data"]["test"]).exists()
    assert json.loads((run / "holdout_evaluation.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("bad_field,bad_value", [
    ("status", "failed"), ("pipeline_fit_partition", "all_labeled_data"),
    ("configured_parameters", {"n_estimators": 25, "learning_rate": 0.5}),
])
def test_source_metadata_mismatch_rejected_before_claim(tmp_path, bad_field, bad_value):
    run, _, _, _ = _fixture(tmp_path)
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[bad_field] = bad_value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        run_evaluation(run, tmp_path / "result", True, artifact_path=tmp_path / "artifact.joblib")
    assert not (run / "holdout_evaluation.json").exists()


def test_config_hash_mismatch_rejected(tmp_path):
    run, _, _, _ = _fixture(tmp_path)
    path = run / "config.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="configuration hash"):
        run_evaluation(run, tmp_path / "result", True, artifact_path=tmp_path / "artifact.joblib")


def test_missing_split_is_never_created(tmp_path):
    run, config, _, _ = _fixture(tmp_path)
    membership = Path(config["splits_dir"]) / "membership.csv"
    metadata = membership.parent / "metadata.json"
    membership.unlink()
    with pytest.raises(ValueError):
        run_evaluation(run, tmp_path / "result", True, artifact_path=tmp_path / "artifact.joblib")
    assert not membership.exists()
    assert metadata.exists()


@pytest.mark.parametrize("corrupt", ["train", "membership"])
def test_input_hash_corruption_rejected_before_claim(tmp_path, corrupt):
    run, config, _, _ = _fixture(tmp_path)
    if corrupt == "train":
        with open(config["data"]["train"], "a") as stream:
            stream.write("\n")
    else:
        path = Path(config["splits_dir"]) / "membership.csv"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        run_evaluation(run, tmp_path / "result", True, artifact_path=tmp_path / "artifact.joblib")
    assert not (run / "holdout_evaluation.json").exists()


def test_duplicate_claim_refuses_an_alternate_output(tmp_path):
    run, _, _, _ = _fixture(tmp_path)
    (run / "holdout_evaluation.json").write_text('{"status":"completed"}\n')
    with pytest.raises(ValueError, match="already has"):
        run_evaluation(run, tmp_path / "different-output", True, artifact_path=tmp_path / "artifact.joblib")
    assert not (tmp_path / "different-output").exists()


def test_default_artifact_sidecars_are_protected(tmp_path):
    run, _, _, _ = _fixture(tmp_path)
    artifact = tmp_path / "artifacts" / "new-pipeline.joblib"
    artifact.parent.mkdir()
    (artifact.parent / "evaluated_pipeline_config.json").write_text("{}")
    with pytest.raises(ValueError, match="sidecar"):
        run_evaluation(run, tmp_path / "result", True, artifact_path=artifact)
    assert not (run / "holdout_evaluation.json").exists()


def test_second_authorization_hash_mismatch_rejected_before_claim(tmp_path):
    run, config, _, _ = _fixture(tmp_path)
    config["models"]["adaboost"] = {"n_estimators": 65, "learning_rate": 1.1}
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["configured_parameters"] = config["models"]["adaboost"]
    metadata["config_sha256"] = sha256(config_path)
    metadata_path.write_text(json.dumps(metadata))
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "authorized": True,
        "model": {"n_estimators": 65, "learning_rate": 1.1},
        "candidate_pipeline_sha256": "not-the-pipeline",
    }))
    with pytest.raises(ValueError, match="authorization/hash"):
        run_evaluation(
            run, tmp_path / "result", True, config_path=config_path,
            artifact_path=tmp_path / "new-artifact.joblib",
            authorization_path=authorization,
        )
    assert not (run / "holdout_evaluation.json").exists()


def test_failure_after_inference_keeps_claim_predictions_and_failed_metadata(tmp_path, monkeypatch):
    run, _, _, _ = _fixture(tmp_path)

    def fail_plot(*args, **kwargs):
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(evaluation.plt, "subplots", fail_plot)
    output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="synthetic"):
        run_evaluation(run, output, True, artifact_path=tmp_path / "artifact.joblib")
    assert (output / "holdout_predictions.csv").exists()
    assert (output / "summary.json").exists()
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["elapsed_seconds"] >= 0
    assert json.loads((run / "holdout_evaluation.json").read_text())["status"] == "failed_after_claim"

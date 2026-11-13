import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA, TARGET, audit, validate_frame
from heart_ml.preprocess import build_pipeline
from heart_ml.split import (
    create_membership,
    encode_labels,
    protocol_hash,
    verify_existing,
    write_or_verify_splits,
)


@pytest.fixture
def config():
    with open("configs/default.json") as config_file:
        return json.load(config_file)


def synthetic_frame(config, rows=60):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"id": np.arange(rows)})
    for column in NUMERIC:
        frame[column] = rng.normal(size=rows)
    for column in CATEGORICAL:
        frame[column] = rng.integers(0, 4, size=rows)
    frame[TARGET] = np.tile(["Absence", "Presence"], rows // 2)
    return frame[SCHEMA + [TARGET]]


def test_audit_rejects_invalid_schema_and_numeric(tmp_path, config):
    frame = synthetic_frame(config)
    with pytest.raises(ValueError, match="expected columns"):
        validate_frame(frame.rename(columns={"id": "wrong"}), True)
    frame[NUMERIC[0]] = frame[NUMERIC[0]].astype(object)
    frame.loc[0, NUMERIC[0]] = "not numeric"
    with pytest.raises(ValueError, match="finite numbers"):
        validate_frame(frame, True)


def test_missing_predictors_allowed_but_missing_id_target_rejected(tmp_path, config):
    frame = synthetic_frame(config)
    frame.loc[0, NUMERIC[0]] = np.nan
    frame.loc[1, CATEGORICAL[0]] = np.nan
    assert validate_frame(frame, True).isna().sum().sum() == 2
    for column in ("id", TARGET):
        broken = frame.copy()
        broken.loc[0, column] = np.nan
        with pytest.raises(ValueError, match="missing"):
            validate_frame(broken, True)


def test_duplicate_ids_and_invalid_categorical_codes_rejected(config):
    frame = synthetic_frame(config)
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicate ID"):
        validate_frame(frame, True)
    frame = synthetic_frame(config)
    frame[CATEGORICAL[0]] = frame[CATEGORICAL[0]].astype(object)
    frame.loc[0, CATEGORICAL[0]] = "bad code"
    with pytest.raises(ValueError, match="numeric codes"):
        validate_frame(frame, True)


def test_audit_reports_missing_and_duplicate_feature_rows(tmp_path, config):
    frame = synthetic_frame(config, 4)
    frame.loc[1, NUMERIC[0]] = np.nan
    frame.loc[2, SCHEMA[1:]] = frame.loc[0, SCHEMA[1:]].to_numpy()
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    report = audit(path, True)
    assert report["missing_by_column"][NUMERIC[0]] == 1
    assert report["duplicate_feature_rows"] == 1


def test_explicit_target_mapping(config):
    labels = pd.Series(["Absence", "Presence"])
    assert encode_labels(labels, config).tolist() == [0, 1]


def test_pipeline_is_cloneable_unfitted_and_predicts_unseen_codes(config):
    frame = synthetic_frame(config)
    X = frame[config["numeric_features"] + config["categorical_features"]]
    X.loc[0, config["numeric_features"][0]] = np.nan
    X.loc[1, config["categorical_features"][0]] = np.nan
    y = encode_labels(frame[TARGET], config)
    unseen = X.iloc[:3].copy()
    unseen[config["categorical_features"][0]] = 999
    for name in config["models"]:
        pipeline = build_pipeline(name, config)
        assert not hasattr(pipeline, "classes_")
        fitted = clone(pipeline).fit(X, y)
        probabilities = fitted.predict_proba(unseen)
        positive_column = np.flatnonzero(fitted.classes_ == 1).item()
        positive_probabilities = probabilities[:, positive_column]
        assert positive_probabilities.shape == (3,)
        assert np.isfinite(positive_probabilities).all()
        assert ((positive_probabilities >= 0) & (positive_probabilities <= 1)).all()


def test_membership_deterministic_complete_disjoint_and_holdout_excluded(config):
    frame = synthetic_frame(config)
    first = create_membership(frame, config)
    second = create_membership(frame, config)
    pd.testing.assert_frame_equal(first, second)
    assert first["id"].is_unique and set(first["id"]) == set(frame["id"])
    holdout = first[first.partition == "holdout"]
    development = first[first.partition == "development"]
    assert len(holdout) == 12 and len(development) == 48
    assert (holdout.fold == -1).all()
    assert set(development.fold) == {0, 1, 2}
    assert development.groupby("fold").size().sum() == len(development)
    assert set(holdout.id).isdisjoint(development.id)


def test_split_create_then_reuse_and_protocol_only_hash(tmp_path, config):
    frame = synthetic_frame(config)
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    cfg = copy.deepcopy(config)
    cfg["data"]["train"] = str(source)
    cfg["splits_dir"] = str(tmp_path / "splits")
    path, created = write_or_verify_splits(cfg)
    original = pd.read_csv(path)
    assert created
    changed_model = copy.deepcopy(cfg)
    changed_model["models"]["logistic"]["C"] = 0.5
    changed_model["threshold"] = 0.6
    assert protocol_hash(cfg) == protocol_hash(changed_model)
    assert verify_existing(path, path.parent / "metadata.json", changed_model, source)
    reused_path, created_again = write_or_verify_splits(changed_model)
    assert not created_again
    pd.testing.assert_frame_equal(original, pd.read_csv(reused_path))


@pytest.mark.parametrize("key,value", [("seed", 12), ("folds", 4)])
def test_split_protocol_change_rejected(tmp_path, config, key, value):
    frame = synthetic_frame(config)
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    cfg = copy.deepcopy(config)
    cfg["data"]["train"] = str(source)
    cfg["splits_dir"] = str(tmp_path / "splits")
    path, _ = write_or_verify_splits(cfg)
    changed = copy.deepcopy(cfg)
    changed[key] = value
    with pytest.raises(ValueError, match="protocol mismatch"):
        write_or_verify_splits(changed)


def test_split_rejects_corruption_and_incomplete_pair(tmp_path, config):
    frame = synthetic_frame(config)
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    cfg = copy.deepcopy(config)
    cfg["data"]["train"] = str(source)
    cfg["splits_dir"] = str(tmp_path / "splits")
    path, _ = write_or_verify_splits(cfg)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="membership hash"):
        write_or_verify_splits(cfg)
    path.unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        write_or_verify_splits(cfg)


def test_split_rejects_source_and_metadata_mismatch(tmp_path, config):
    frame = synthetic_frame(config)
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    cfg = copy.deepcopy(config)
    cfg["data"]["train"] = str(source)
    cfg["splits_dir"] = str(tmp_path / "splits")
    path, _ = write_or_verify_splits(cfg)
    metadata_path = path.parent / "metadata.json"

    changed_frame = frame.copy()
    changed_frame.loc[0, NUMERIC[0]] += 1
    changed_frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="source/protocol mismatch"):
        write_or_verify_splits(cfg)

    changed_frame.to_csv(source, index=False)
    metadata = json.loads(metadata_path.read_text())
    metadata["protocol_sha256"] = "incorrect"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source/protocol mismatch"):
        write_or_verify_splits(cfg)


def test_split_rejects_bad_schema_and_missing_id(tmp_path, config):
    frame = synthetic_frame(config)
    frame.loc[0, "id"] = np.nan
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    cfg = copy.deepcopy(config)
    cfg["data"]["train"] = str(source)
    cfg["splits_dir"] = str(tmp_path / "splits")
    with pytest.raises(ValueError, match="missing IDs"):
        write_or_verify_splits(cfg)

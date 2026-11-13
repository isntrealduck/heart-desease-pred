"""Shared input-schema validation and a compact, non-clinical data audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "Age", "BP", "Cholesterol", "Max HR", "ST depression",
    "Number of vessels fluro",
]
CATEGORICAL = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Thallium",
]
FEATURES = NUMERIC + CATEGORICAL
ID = "id"
TARGET = "Heart Disease"
LABELS = {"Absence", "Presence"}
SCHEMA = [
    ID, "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro",
    "Thallium",
]


def sha256(path):
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    """Constrain role/schema declarations to the agreed dataset contract."""
    expected_roles = {
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
    }
    for key, expected in expected_roles.items():
        if config.get(key) != expected:
            raise ValueError(f"Configuration {key} must match the fixed schema")
    data = config.get("data", {})
    if data.get("id") != ID or data.get("target") != TARGET:
        raise ValueError("Configuration id/target must match the fixed schema")
    if (config.get("negative_label"), config.get("positive_label")) != (
        "Absence", "Presence"
    ):
        raise ValueError("Configuration labels must be Absence and Presence")
    if config.get("selection_metric") != "roc_auc":
        raise ValueError("Configuration selection_metric must be roc_auc")
    forbidden = {ID, TARGET}
    if forbidden.intersection(config["numeric_features"] + config["categorical_features"]):
        raise ValueError("Identifier and target cannot be model features")


def validate_frame(frame, labeled, source="<data>"):
    """Validate exact column schema, IDs, labels, and nonmissing value types.

    Predictor missingness is allowed for the configured imputers. IDs and
    targets are required; numeric values must be finite whenever they occur.
    """
    expected = SCHEMA + ([TARGET] if labeled else [])
    if list(frame.columns) != expected:
        raise ValueError(f"{source}: expected columns {expected}; got {list(frame.columns)}")
    if frame[ID].isna().any():
        raise ValueError(f"{source}: missing IDs")
    if not frame[ID].is_unique:
        raise ValueError(f"{source}: duplicate ID values")
    if labeled:
        if frame[TARGET].isna().any():
            raise ValueError(f"{source}: missing target labels")
        unexpected = set(frame[TARGET].unique()) - LABELS
        if unexpected:
            raise ValueError(f"{source}: unexpected target labels {unexpected}")
        if set(frame[TARGET].unique()) != LABELS:
            raise ValueError(f"{source}: both target classes are required")
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if invalid.any() or not np.isfinite(values.dropna()).all():
            raise ValueError(f"{source}: {column} must contain finite numbers or missing values")
    for column in CATEGORICAL:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if invalid.any() or not np.isfinite(values.dropna()).all():
            raise ValueError(f"{source}: {column} must contain finite numeric codes or missing values")
    return frame


def validate_csv(path, labeled):
    """Read and validate a project CSV."""
    try:
        return validate_frame(pd.read_csv(path), labeled, str(path))
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read {path}: {error}") from error


def audit(path, labeled):
    """Return descriptive schema/content checks, not clinical ranges."""
    frame = validate_csv(path, labeled)
    report = {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "id_unique": True,
        "missing_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count
        },
        "numeric_ranges": {
            column: [
                float(frame[column].min()), float(frame[column].max())
            ]
            for column in NUMERIC if frame[column].notna().any()
        },
        "categories": {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL
        },
        "duplicate_feature_rows": int(frame[FEATURES].duplicated().sum()),
    }
    if labeled:
        report["label_counts"] = {
            str(label): int(count)
            for label, count in frame[TARGET].value_counts().items()
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="DATA_MANIFEST.json")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = json.load(config_file)
    validate_config(config)
    reports = {
        name: audit(config["data"][name], name == "train")
        for name in ("train", "test")
    }
    Path(args.output).write_text(json.dumps({
        "provenance": "unverified; candidate source noted in PROJECT_PLAN.md",
        "reports": reports,
    }, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

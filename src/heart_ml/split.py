"""Create and verify immutable stratified train/development memberships."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .audit import ID, TARGET, sha256, validate_config, validate_csv

PROTOCOL_KEYS = (
    "seed", "holdout_fraction", "folds", "positive_label", "negative_label",
)


def encode_labels(labels, config):
    """Map the declared negative/positive labels explicitly to 0/1."""
    mapping = {config["negative_label"]: 0, config["positive_label"]: 1}
    encoded = labels.map(mapping)
    if encoded.isna().any():
        raise ValueError("Labels contain values outside the configured mapping")
    return encoded.astype("int8")


def protocol_hash(config):
    """Hash only choices that determine membership, not model tuning settings."""
    protocol = {
        "id": config["data"]["id"],
        "target": config["data"]["target"],
        **{key: config[key] for key in PROTOCOL_KEYS},
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_membership(frame, config):
    """Assign one stratified holdout and disjoint shuffled CV folds.

    Fold numbers are defined only for development rows. Holdout membership is
    split first; cross-validation then partitions the development indices.
    """
    labels = encode_labels(frame[TARGET], config)
    indices = np.arange(len(frame))
    development, holdout = train_test_split(
        indices,
        test_size=config["holdout_fraction"],
        stratify=labels,
        random_state=config["seed"],
    )
    folds = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=config["folds"], shuffle=True, random_state=config["seed"]
    )
    for fold_number, (_, validation_positions) in enumerate(
        splitter.split(np.zeros(len(development)), labels.iloc[development])
    ):
        folds[development[validation_positions]] = fold_number
    partition = np.full(len(frame), "development", dtype=object)
    partition[holdout] = "holdout"
    return pd.DataFrame({
        ID: frame[ID].to_numpy(),
        "partition": partition,
        "fold": folds,
    })


def expected_metadata(config, source):
    return {
        "source_sha256": sha256(source),
        "protocol_sha256": protocol_hash(config),
    }


def verify_existing(path, metadata_path, config, source):
    """Reject partial, stale, or corrupted split files rather than replacing."""
    if not path.exists() and not metadata_path.exists():
        return False
    if not path.exists() or not metadata_path.exists():
        raise ValueError("Incomplete existing split files; refusing overwrite")
    metadata = json.loads(metadata_path.read_text())
    expected = expected_metadata(config, source)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing split source/protocol mismatch; refusing overwrite")
    if metadata.get("membership_sha256") != sha256(path):
        raise ValueError("Existing membership hash mismatch; refusing overwrite")
    return True


def write_or_verify_splits(config):
    """Create a split once, or verify the existing source/protocol/output hashes."""
    validate_config(config)
    source = config["data"]["train"]
    frame = validate_csv(source, labeled=True)
    output_dir = Path(config["splits_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "membership.csv"
    metadata_path = output_dir / "metadata.json"
    if verify_existing(membership_path, metadata_path, config, source):
        return membership_path, False

    membership = create_membership(frame, config)
    membership.to_csv(membership_path, index=False)
    metadata = expected_metadata(config, source)
    metadata["membership_sha256"] = sha256(membership_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return membership_path, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = json.load(config_file)
    path, created = write_or_verify_splits(config)
    if created:
        print(f"Wrote {path}")
    else:
        print(f"Verified existing splits: {path}")


if __name__ == "__main__":
    main()

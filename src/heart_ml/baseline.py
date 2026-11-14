"""Compare baselines and feature hypotheses using saved development folds only."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

from .audit import ID, TARGET, sha256, validate_config, validate_csv
from .preprocess import build_pipeline
from .features import FEATURE_DEFINITIONS, FEATURE_SETS
from .split import encode_labels, write_or_verify_splits

SUPPORTED_MODELS = ("dummy", "logistic", "decision_tree")
METRICS = ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1")


def development_rows(frame, membership, config):
    """Join by ID, validate membership, and remove holdout before model use."""
    if list(membership.columns) != [ID, "partition", "fold"]:
        raise ValueError("Expected membership columns: id, partition, fold")
    if membership[ID].isna().any() or not membership[ID].is_unique:
        raise ValueError("Membership IDs must be present and unique")
    joined = frame.merge(
        membership, on=ID, how="outer", validate="one_to_one", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Membership must cover exactly the training CSV IDs")
    if set(joined["partition"]) != {"development", "holdout"}:
        raise ValueError("Membership must contain development and holdout partitions")
    holdout = joined["partition"].eq("holdout")
    if not joined.loc[holdout, "fold"].eq(-1).all():
        raise ValueError("Holdout rows must have fold -1")
    development = joined.loc[~holdout].drop(columns="_merge").reset_index(drop=True)
    if set(development["fold"]) != set(range(config["folds"])):
        raise ValueError("Development fold assignments do not match configuration")
    return development


def _feature_pipeline(model_name, config, feature_set):
    # Retain the two-argument builder API for existing baseline callers/tests.
    if feature_set == "original":
        return build_pipeline(model_name, config)
    return build_pipeline(model_name, config, feature_set=feature_set)


def evaluate_model(development, config, model_name, feature_set="original"):
    """Return per-fold results and out-of-fold development predictions.

    Each fold fits a fresh complete pipeline, including learned preprocessing.
    Probabilities, rather than thresholded labels, are used for ROC-AUC/AP.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Supported models are dummy, logistic and decision_tree")
    if feature_set not in FEATURE_SETS or (feature_set != "original" and model_name != "logistic"):
        raise ValueError("Unknown feature set, or derived features requested for a non-logistic model")
    if not development["partition"].eq("development").all():
        raise ValueError("Only development rows may enter baseline evaluation")
    threshold = config["threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number between 0 and 1")
    if set(development["fold"]) != set(range(config["folds"])):
        raise ValueError("Development fold assignments do not match configuration")
    features = config["numeric_features"] + config["categorical_features"]
    X = development[features]
    y = encode_labels(development[TARGET], config)
    predictions = development[[ID, "fold"]].copy()
    predictions["true_label"] = y.to_numpy()
    probabilities = np.full(len(development), np.nan)
    results = []
    for fold in range(config["folds"]):
        validation = development["fold"].eq(fold).to_numpy()
        training = ~validation
        if y.loc[training].nunique() != 2 or y.loc[validation].nunique() != 2:
            raise ValueError(f"Fold {fold}: training and validation need both classes")
        started = perf_counter()
        pipeline = _feature_pipeline(model_name, config, feature_set)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X.loc[training], y.loc[training])
            positive_column = np.flatnonzero(pipeline.classes_ == 1).item()
            probability = pipeline.predict_proba(X.loc[validation])[:, positive_column]
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError(f"Fold {fold}: invalid predicted probabilities")
        predicted = (probability >= threshold).astype("int8")
        actual = y.loc[validation]
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average="binary", pos_label=1, zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        notes = [str(item.message) for item in caught]
        if not predicted.any():
            notes.append("No positive predictions; undefined precision is reported as 0.")
        results.append({
            "model": model_name, "strategy": "prior" if model_name == "dummy" else "",
            "feature_set": feature_set,
            "fold": fold,
            "training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
            "training_positive_fraction": float(y.loc[training].mean()),
            "validation_positive_fraction": float(actual.mean()),
            "threshold": threshold,
            "roc_auc": float(roc_auc_score(actual, probability)),
            "average_precision": float(average_precision_score(actual, probability)),
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "solver_iterations": int(np.max(getattr(pipeline.named_steps["model"], "n_iter_", [0]))),
            "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
            "elapsed_seconds": perf_counter() - started,
            "notes": " | ".join(notes),
        })
        probabilities[validation] = probability
    predictions["probability_presence"] = probabilities
    predictions["predicted_label"] = (probabilities >= threshold).astype("int8")
    return pd.DataFrame(results), predictions


def evaluate_dummy(development, config):
    """Preserve the original dummy-only API while sharing evaluation logic."""
    return evaluate_model(development, config, "dummy")


def code_metadata():
    """Identify both the base commit and current source, including local edits."""
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = git("rev-parse", "HEAD")
    # The slim Docker image has no Git executable, but the bind-mounted clone
    # still contains HEAD and its refs. Source hashes also capture local edits.
    if commit is None and (root / ".git").is_dir():
        git_dir = root / ".git"
        try:
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref: refs/"):
                reference = head.removeprefix("ref: ")
                ref_path = git_dir / reference
                if ref_path.is_file():
                    commit = ref_path.read_text().strip()
                elif (git_dir / "packed-refs").is_file():
                    for line in (git_dir / "packed-refs").read_text().splitlines():
                        if line.endswith(" " + reference):
                            commit = line.split()[0]
                            break
            elif len(head) in (40, 64) and all(char in "0123456789abcdef" for char in head):
                commit = head
        except OSError:
            commit = None
    status = git("status", "--porcelain")
    return {
        "commit": commit,
        "working_tree_dirty": None if status is None else bool(status),
        "requirements_lock_sha256": sha256(root / "requirements.lock"),
        "source_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_baseline(config, model_name="dummy", output_dir=None, feature_set="original"):
    """Verify/create saved splits and save a reproducible development-only run."""
    validate_config(config)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Supported models are dummy, logistic and decision_tree")
    if feature_set not in FEATURE_SETS or (feature_set != "original" and model_name != "logistic"):
        raise ValueError("Unknown feature set, or derived features requested for a non-logistic model")
    threads = config.get("threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir is not None else Path("reports/generated") / model_name / run_id
    # Refuse to overwrite a reference run, including one with a failed status.
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id, "status": "running", "evaluation": "development_cross_validation",
        "model": model_name, "strategy": "prior" if model_name == "dummy" else None,
        "feature_set": feature_set, "feature_definitions": FEATURE_DEFINITIONS[feature_set],
        "configured_parameters": config["models"][model_name],
        "threshold": config["threshold"],
        "positive_class": 1, "threshold_rule": "probability_presence >= threshold",
        "zero_division": 0, "holdout_evaluated": False, "test_csv_used": False,
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "threads": threads,
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl")},
        "code": code_metadata(),
    }
    write_json(output / "config.json", config)
    metadata["config_sha256"] = sha256(output / "config.json")
    started = perf_counter()
    try:
        with threadpool_limits(limits=threads):
            membership_path, created = write_or_verify_splits(config)
            frame = validate_csv(config["data"]["train"], labeled=True)
            membership = pd.read_csv(membership_path)
            development = development_rows(frame, membership, config)
            metadata.update({
                "train_sha256": sha256(config["data"]["train"]),
                "membership_sha256": sha256(membership_path),
                "split_metadata": json.loads((membership_path.parent / "metadata.json").read_text()),
                "split_created": created, "development_rows": len(development),
                "holdout_rows_excluded": len(frame) - len(development),
            })
            folds, predictions = evaluate_model(development, config, model_name, feature_set=feature_set)
            folds.to_csv(output / "cv_results.csv", index=False)
            predictions.to_csv(output / "development_predictions.csv", index=False)
            summary = {
                "evaluation": "development_cross_validation", "model": model_name,
                "strategy": "prior" if model_name == "dummy" else None,
                "feature_set": feature_set, "feature_definitions": FEATURE_DEFINITIONS[feature_set],
                "configured_parameters": config["models"][model_name],
                "folds": len(folds), "threshold": config["threshold"],
                "development_rows": len(development),
                "metrics": {name: {
                    "mean": float(folds[name].mean()),
                    "fold_sd": float(folds[name].std(ddof=1)),
                } for name in METRICS},
                "note": "Unweighted fold means; sample SD (ddof=1), not a confidence interval. No pooled OOF ROC-AUC is reported.",
            }
            write_json(output / "summary.json", summary)
            # This is a development-trained baseline, not a holdout-evaluated final model.
            pipeline = _feature_pipeline(model_name, config, feature_set)
            features = config["numeric_features"] + config["categorical_features"]
            with warnings.catch_warnings(record=True) as final_warnings:
                warnings.simplefilter("always")
                pipeline.fit(development[features], encode_labels(development[TARGET], config))
            metadata["final_fit_warnings"] = [str(item.message) for item in final_warnings]
            metadata["final_fit_convergence_warning"] = any(
                issubclass(item.category, ConvergenceWarning) for item in final_warnings
            )
            metadata["resolved_model_parameters"] = pipeline.named_steps["model"].get_params(deep=False)
            metadata["final_fit_solver_iterations"] = int(
                np.max(getattr(pipeline.named_steps["model"], "n_iter_", [0]))
            )
            joblib.dump(pipeline, output / f"{model_name}_pipeline.joblib")
        metadata["status"] = "completed"
        metadata["pipeline_fit_partition"] = "development_only"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        metadata["elapsed_seconds"] = perf_counter() - started
        write_json(output / "run_metadata.json", metadata)
    return output, summary


def run_dummy_baseline(config, output_dir=None):
    """Keep existing callers and the default dummy command working."""
    return run_baseline(config, model_name="dummy", output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="dummy")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="original")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output, summary = run_baseline(
        config, model_name=args.model, output_dir=args.output_dir, feature_set=args.feature_set
    )
    print(f"{args.model.capitalize()} / {args.feature_set}: development cross-validation only")
    for metric, values in summary["metrics"].items():
        print(f"{metric}: {values['mean']:.6f} (fold SD {values['fold_sd']:.6f})")
    print(f"Saved results and development-trained pipeline to {output}")


if __name__ == "__main__":
    main()

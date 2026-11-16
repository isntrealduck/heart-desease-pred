"""One-time, load-only holdout evaluation for the frozen AdaBoost run."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
from importlib.metadata import version
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
    roc_curve,
)

from .audit import FEATURES, ID, TARGET, sha256, validate_config, validate_csv
from .baseline import code_metadata, development_rows, write_json
from .split import encode_labels, protocol_hash, verify_existing


def positive_probabilities(pipeline, X):
    """Return the class-1 column without assuming estimator class ordering."""
    classes = np.asarray(pipeline.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Fitted pipeline must contain exactly one positive class 1")
    values = np.asarray(pipeline.predict_proba(X), dtype=float)
    if values.ndim != 2 or values.shape != (len(X), len(classes)):
        raise ValueError("predict_proba returned an unexpected shape")
    probability = values[:, matches[0]]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("predict_proba returned invalid probabilities")
    return probability


def _preflight(run_dir, output_dir, config_path, artifact_path=None, authorization_path=None):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    artifact_path = Path(artifact_path) if artifact_path else Path("artifacts/evaluated_pipeline.joblib")
    sidecars = [artifact_path, artifact_path.with_name("evaluated_pipeline_config.json"),
                artifact_path.with_name("evaluated_pipeline_metadata.json")]
    if any(path.exists() for path in sidecars):
        raise ValueError(f"Evaluated artifact or sidecar already exists: {artifact_path}")
    config_path = Path(config_path) if config_path else run_dir / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    if config["threshold"] != 0.5:
        raise ValueError("Frozen holdout threshold must be 0.5")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    if (meta.get("status"), meta.get("evaluation"), meta.get("model"),
        meta.get("feature_set"), meta.get("holdout_evaluated"), meta.get("test_csv_used"),
        meta.get("pipeline_fit_partition"), meta.get("threshold"), meta.get("positive_class")) != (
        "completed", "development_cross_validation", "adaboost", "original",
        False, False, "development_only", 0.5, 1
    ):
        raise ValueError("Source run is not the expected completed development-only AdaBoost run")
    if meta.get("config_sha256") != sha256(config_path) or sha256(run_dir / "config.json") != sha256(config_path):
        raise ValueError("Source configuration hash mismatch")
    model_config = config["models"].get("adaboost", {})
    second = authorization_path is not None
    if second:
        authorization = json.loads(Path(authorization_path).read_text())
        original = Path("reports/generated/adaboost/20261114T223109215876Z")
        candidate_pipeline = run_dir / "adaboost_pipeline.joblib"
        required = {
            "authorized": True, "candidate_pipeline_sha256": sha256(candidate_pipeline),
            "candidate_config_sha256": sha256(config_path),
            "candidate_metadata_sha256": sha256(run_dir / "run_metadata.json"),
            "train_sha256": sha256(config["data"]["train"]),
            "membership_sha256": sha256(Path(config["splits_dir"]) / "membership.csv"),
            "original_claim_sha256": sha256(original / "holdout_evaluation.json"),
            "original_summary_sha256": sha256(original / "holdout-result/summary.json"),
            "original_membership_sha256": sha256(Path(config["splits_dir"]) / "membership.csv"),
        }
        if any(authorization.get(key) != value for key, value in required.items()):
            raise ValueError("Second-evaluation authorization/hash binding mismatch")
        if authorization.get("model") != {"n_estimators": 65, "learning_rate": 1.1}:
            raise ValueError("Authorization does not identify the exact frozen candidate")
        if sum(len(config.get(k, [])) for k in ("numeric_features", "categorical_features")) != 11:
            raise ValueError("Authorized candidate must use exactly 11 predictors")
    elif model_config != {"n_estimators": 50, "learning_rate": 1.0}:
        raise ValueError("Unexpected frozen AdaBoost configuration")
    if meta.get("configured_parameters") != model_config:
        raise ValueError("Source run configured parameters mismatch")
    if (config["seed"], config["holdout_fraction"], config["folds"],
        config["positive_label"], config["negative_label"]) != (42, 0.2, 3, "Presence", "Absence"):
        raise ValueError("Saved split protocol differs from the frozen protocol")
    expected_packages = ("numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl")
    actual_packages = {name: version(name) for name in expected_packages}
    if any(meta.get("packages", {}).get(name) != value for name, value in actual_packages.items()):
        raise ValueError("Evaluation environment package versions differ from training")
    lock_hash = sha256(Path("requirements.lock"))
    if meta.get("code", {}).get("requirements_lock_sha256") != lock_hash:
        raise ValueError("Requirements lock differs from the training environment")
    train_path = Path(config["data"]["train"])
    membership_path = Path(config["splits_dir"]) / "membership.csv"
    split_meta_path = membership_path.parent / "metadata.json"
    if not verify_existing(membership_path, split_meta_path, config, train_path):
        raise ValueError("Frozen split files are required; evaluation never creates them")
    membership_hash = sha256(membership_path)
    split_metadata = json.loads(split_meta_path.read_text())
    for key, actual in (("train_sha256", sha256(train_path)),
                        ("membership_sha256", membership_hash)):
        if meta.get(key) != actual:
            raise ValueError(f"Source run {key} mismatch")
    if meta.get("split_metadata") != split_metadata:
        raise ValueError("Source split metadata mismatch")
    if split_metadata.get("protocol_sha256") != protocol_hash(config):
        raise ValueError("Frozen split protocol mismatch")
    frame = validate_csv(train_path, labeled=True)
    membership = pd.read_csv(membership_path)
    development = development_rows(frame, membership, config)
    holdout_ids = set(membership.loc[membership.partition.eq("holdout"), ID])
    if meta.get("development_rows") != len(development) or meta.get("holdout_rows_excluded") != len(holdout_ids):
        raise ValueError("Source run row counts mismatch")
    holdout = frame.loc[frame[ID].isin(holdout_ids)].copy()
    if len(holdout) != len(holdout_ids):
        raise ValueError("Holdout ID join failed")
    if set(holdout[TARGET]) != {"Absence", "Presence"}:
        raise ValueError("Holdout must contain both target classes")
    pipeline_path = run_dir / "adaboost_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    model = pipeline.named_steps.get("model")
    if not isinstance(model, AdaBoostClassifier) or not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Artifact is not the expected fitted AdaBoost pipeline")
    expected_estimators, expected_rate = (65, 1.1) if second else (50, 1.0)
    if model.n_estimators != expected_estimators or model.learning_rate != expected_rate or model.random_state != 42:
        raise ValueError("Fitted AdaBoost parameters do not match frozen settings")
    if len(model.estimators_) != expected_estimators or any(
        not isinstance(tree, DecisionTreeClassifier) or tree.get_depth() != 1
        for tree in model.estimators_
    ):
        raise ValueError(f"Fitted weak learners differ from the expected {expected_estimators} stumps")
    expected_features = config["numeric_features"] + config["categorical_features"]
    if not hasattr(pipeline, "feature_names_in_") or list(pipeline.feature_names_in_) != expected_features:
        raise ValueError("Pipeline input features differ from the frozen schema")
    selectors = pipeline.named_steps["preprocess"].transformers
    if selectors[0][2] != config["numeric_features"] or selectors[1][2] != config["categorical_features"]:
        raise ValueError("Fitted preprocessing selectors differ from the frozen schema")
    marker = run_dir / "holdout_evaluation.json"
    if marker.exists():
        raise ValueError("This source run already has a holdout evaluation claim")
    return config, meta, frame, membership, holdout, pipeline, marker, run_dir, output_dir, artifact_path, second


def _claim(marker, info):
    """Exclusive creation prevents a second invocation through another output path."""
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2)
        stream.write("\n")


def run_evaluation(run_dir, output_dir, confirm_holdout=False, config_path=None, artifact_path=None, authorization_path=None):
    if not confirm_holdout:
        raise ValueError("Explicit --confirm-holdout acknowledgment is required")
    (config, source_meta, frame, membership, holdout, pipeline, marker,
     run_dir, output_dir, artifact_path, second) = _preflight(
         run_dir, output_dir, config_path, artifact_path, authorization_path
     )
    output_dir.mkdir(parents=True, exist_ok=False)
    claim = {
        "status": "claimed", "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir), "source_pipeline_sha256": sha256(run_dir / "adaboost_pipeline.joblib"),
        "membership_sha256": sha256(Path(config["splits_dir"]) / "membership.csv"),
        "source_metadata_sha256": sha256(run_dir / "run_metadata.json"),
        "threshold": 0.5, "output_dir": str(output_dir),
    }
    if authorization_path is not None:
        claim.update({
            "authorization_path": str(authorization_path),
            "authorization_sha256": sha256(authorization_path),
            "reused_previous_holdout": True,
            "previous_evaluation": "reports/generated/adaboost/20261114T223109215876Z/holdout_evaluation.json",
        })
    _claim(marker, claim)
    started = perf_counter()
    metadata = {
        "status": "running", "stage": "inference", "started_at_utc": claim["claimed_at_utc"],
        "source_run": str(run_dir), "source_metadata_sha256": claim["source_metadata_sha256"],
        "source_pipeline_sha256": claim["source_pipeline_sha256"],
        "evaluation_code": code_metadata(), "training_code": source_meta.get("code"),
        "training_packages": source_meta.get("packages"), "test_csv_used": False,
        "pipeline_refit": False, "threshold": 0.5,
        "reused_previous_holdout": second,
        "authorization_path": str(authorization_path) if second else None,
        "authorization_sha256": sha256(authorization_path) if second else None,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    try:
        X = holdout[config["numeric_features"] + config["categorical_features"]]
        y = encode_labels(holdout[TARGET], config).to_numpy()
        probability = positive_probabilities(pipeline, X)
        predicted = (probability >= 0.5).astype("int8")
        predictions = pd.DataFrame({
            ID: holdout[ID].to_numpy(), "true_label": y,
            "probability_presence": probability, "predicted_label": predicted,
        })
        predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
        fpr, tpr, _ = roc_curve(y, probability)
        precision_curve, recall_curve, _ = precision_recall_curve(y, probability)
        pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}).to_csv(output_dir / "roc_curve.csv", index=False)
        pd.DataFrame({"recall": recall_curve, "precision": precision_curve}).to_csv(output_dir / "precision_recall_curve.csv", index=False)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        p, r, f, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
        metrics = {
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "rows": len(y), "positive_rows": int(y.sum()), "negative_rows": int(len(y) - y.sum()),
            "predicted_positive_rows": int(predicted.sum()), "threshold": 0.5,
        }
        summary = {
            "evaluation": "final_holdout", "model": "adaboost",
            "source_run": str(run_dir), "source_config_sha256": source_meta["config_sha256"],
            "source_metadata_sha256": claim["source_metadata_sha256"],
            "source_pipeline_sha256": claim["source_pipeline_sha256"],
            "train_sha256": sha256(config["data"]["train"]),
            "membership_sha256": claim["membership_sha256"], "metrics": metrics,
            "note": ("Authorized second evaluation reuses the previously evaluated holdout; prior 50-stump evaluation disclosed. No tuning permitted after this result. " if second else "One-time evaluation of the frozen development-only pipeline. ") + "AP is average precision, not trapezoidal PR-AUC.",
        }
        write_json(output_dir / "summary.json", summary)
        metadata.update({
            **summary, "status": "running", "stage": "plotting_and_artifacts",
            "packages": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl")},
        })
        write_json(output_dir / "run_metadata.json", metadata)
        for name, x, yy, xlabel, ylabel in (
            ("roc_curve.png", fpr, tpr, "False positive rate", "True positive rate"),
            ("precision_recall_curve.png", recall_curve, precision_curve, "Recall", "Precision"),
        ):
            fig, ax = plt.subplots()
            ax.plot(x, yy)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            fig.tight_layout()
            fig.savefig(output_dir / name, dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots()
        ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"], xlabel="Predicted", ylabel="Actual")
        for (i, j), value in np.ndenumerate([[tn, fp], [fn, tp]]):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "adaboost_pipeline.joblib", artifact_path)
        shutil.copyfile(run_dir / "config.json", output_dir / "config.json")
        shutil.copyfile(run_dir / "config.json", artifact_path.with_name("evaluated_pipeline_config.json"))
        (output_dir / "report.md").write_text(
            "# AdaBoost final holdout evaluation\n\n"
            + ("This is an explicitly authorized second evaluation reusing the previously evaluated holdout; the earlier 50-stump evaluation remains preserved. No tuning is permitted after this result. " if second else "This is the one-time evaluation of the frozen development-only pipeline. ")
            + "The unlabeled test file was not opened; no model was fit or selected.\n\n"
            f"Rows: {metrics['rows']}; threshold: 0.5.\n\n"
            "| Metric | Value |\n|---|---:|\n" +
            "".join(f"| {k} | {metrics[k]:.6f} |\n" for k in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1")) +
            "\nAP is average precision, not trapezoidal PR-AUC.\n"
        )
        metadata.update({
            "status": "completed", "stage": "complete",
            "evaluated_pipeline": str(artifact_path),
            "evaluated_pipeline_sha256": sha256(artifact_path),
            "python": platform.python_version(),
            "elapsed_seconds": perf_counter() - started,
        })
        write_json(output_dir / "run_metadata.json", metadata)
        shutil.copyfile(output_dir / "run_metadata.json", artifact_path.with_name("evaluated_pipeline_metadata.json"))
        claim.update({"status": "completed", "output_dir": str(output_dir)})
        write_json(marker, claim)
        return summary
    except Exception as error:
        metadata.update({
            "status": "failed", "stage": metadata.get("stage"),
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": perf_counter() - started,
        })
        write_json(output_dir / "run_metadata.json", metadata)
        claim.update({"status": "failed_after_claim", "error_type": type(error).__name__, "error": str(error)})
        write_json(marker, claim)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="Optional config path; defaults to source run config")
    parser.add_argument("--artifact-path", help="Destination for a separate evaluated pipeline artifact")
    parser.add_argument("--authorization", help="Hash-bound authorization JSON for the specifically approved second evaluation")
    parser.add_argument("--confirm-holdout", action="store_true", required=True,
                        help="Acknowledge this consumes the one permitted holdout evaluation")
    args = parser.parse_args()
    result = run_evaluation(args.run_dir, args.output_dir, args.confirm_holdout, args.config,
                            args.artifact_path, args.authorization)
    print(f"Completed final holdout evaluation; outputs saved to {args.output_dir}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

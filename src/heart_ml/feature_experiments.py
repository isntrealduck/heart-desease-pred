"""Run a predeclared, development-only feature-engineering comparison."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .audit import NUMERIC, TARGET, sha256, validate_csv
from .baseline import code_metadata, development_rows, run_baseline, write_json
from .features import FEATURE_DEFINITIONS, FEATURE_SETS
from .split import encode_labels, write_or_verify_splits


def development_eda(development, config, output):
    """Describe only development rows; plotted bins never become model features."""
    if not development["partition"].eq("development").all():
        raise ValueError("EDA accepts development rows only")
    view = development[["Age", "ST depression"]].copy()
    if view["Age"].notna().sum() == 0 or view["ST depression"].notna().sum() == 0:
        raise ValueError("These hypotheses require observed development Age and ST depression")
    view["positive"] = encode_labels(development[TARGET], config).to_numpy()
    age_bins = pd.cut(view["Age"], bins=6)
    st_bins = pd.cut(view["ST depression"], bins=5)
    age_rates = view.groupby(age_bins, observed=True)["positive"].agg(["size", "mean"])
    grid = view.pivot_table(
        index=age_bins, columns=st_bins, values="positive", aggfunc="mean", observed=True
    )
    grid.index = grid.index.astype(str)
    grid.columns = grid.columns.astype(str)
    audit = {
        "rows": len(development),
        "positive_fraction": float(view["positive"].mean()),
        "numeric_description": json.loads(development[NUMERIC].describe().to_json()),
        "missing_numeric": {name: int(development[name].isna().sum()) for name in NUMERIC},
        "age_bin_rates": [
            {"bin": str(label), "rows": int(row["size"]), "presence_fraction": float(row["mean"])}
            for label, row in age_rates.iterrows()
        ],
        "age_st_bin_rates": json.loads(grid.to_json(orient="split")),
        "note": "Descriptive bins use development data only; they are not clinical cutoffs or training inputs.",
    }
    write_json(output / "development_eda.json", audit)
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    age_axis, grid_axis = figure.subplots(1, 2)
    age_axis.plot(np.arange(len(age_rates)), age_rates["mean"], marker="o")
    age_axis.set_xticks(np.arange(len(age_rates)), [str(value) for value in age_rates.index], rotation=40, ha="right")
    age_axis.set(xlabel="Exploratory age bin", ylabel="Fraction labeled Presence", ylim=(0, 1))
    image = grid_axis.imshow(grid.to_numpy(), aspect="auto", vmin=0, vmax=1)
    grid_axis.set_xticks(np.arange(len(grid.columns)), grid.columns, rotation=40, ha="right")
    grid_axis.set_yticks(np.arange(len(grid.index)), grid.index)
    grid_axis.set(xlabel="Exploratory ST-depression bin", ylabel="Exploratory age bin")
    figure.colorbar(image, ax=grid_axis, label="Fraction labeled Presence")
    figure.suptitle("Development data only: descriptive label rates")
    figure.savefig(output / "feature_engineering_eda.png", dpi=160)
    return audit


def choose_variant(records, minimum_gain):
    """Apply the frozen practical-gain/paired-fold rule, not a significance test."""
    completed = [row for row in records if row["status"] == "completed"]
    original = next(row for row in completed if row["feature_set"] == "original")
    eligible = [
        row for row in completed
        if row["feature_set"] != "original"
        and row["mean_delta_roc_auc"] >= minimum_gain
        and row["all_folds_improved"]
        and row["convergence_warnings"] == 0
    ]
    selected = max(eligible, key=lambda row: (row["roc_auc"], -row["added_features"])) if eligible else original
    return {
        "recommended_feature_set": selected["feature_set"],
        "highest_mean_roc_auc_feature_set": max(completed, key=lambda row: row["roc_auc"])["feature_set"],
        "minimum_mean_roc_auc_gain": minimum_gain,
        "require_positive_gain_in_every_fold": True,
        "exclude_convergence_warnings": True,
        "eligible_engineered_variants": [row["feature_set"] for row in eligible],
        "scope": "Development recommendation only; no final holdout evaluation or significance claim.",
    }


def comparison_figure(records, selection, output):
    completed = [row for row in records if row["status"] == "completed"]
    figure = Figure(figsize=(11, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    score_axis, gain_axis = figure.subplots(1, 2)
    labels = [row["feature_set"].replace("_", " ") for row in completed]
    positions = np.arange(len(completed))
    score_axis.errorbar(
        positions, [row["roc_auc"] for row in completed],
        yerr=[row["roc_auc_fold_sd"] for row in completed], fmt="o", capsize=4,
    )
    score_axis.set_xticks(positions, labels, rotation=30, ha="right")
    score_axis.set(ylabel="Mean development ROC-AUC", title="Error bars: fold sample SD")
    for position, row in enumerate(completed):
        gains = row["fold_roc_auc_gains"]
        gain_axis.scatter(np.full(len(gains), position), gains, s=30)
        gain_axis.plot(position, row["mean_delta_roc_auc"], marker="_", markersize=18, color="black")
    gain_axis.axhline(0, color="gray", linewidth=1)
    gain_axis.axhline(selection["minimum_mean_roc_auc_gain"], color="gray", linestyle=":", label="Required mean gain")
    gain_axis.set_xticks(positions, labels, rotation=30, ha="right")
    gain_axis.set(ylabel="Paired ROC-AUC gain over original", title="Dots: folds; black mark: mean")
    gain_axis.legend(fontsize=8)
    figure.suptitle("Feature hypotheses: identical development folds and fixed logistic settings")
    figure.savefig(output / "feature_engineering_comparison.png", dpi=160)


def write_report(output, record):
    selection = record["selection"]
    lines = [
        "# Feature engineering: development experiments", "",
        "These are development-only selection experiments. The final holdout and test.csv were not evaluated.", "",
        "## Predeclared hypotheses and protocol", "",
        "The original logistic baseline is compared with Age squared, Age times ST depression, and both terms together. "
        "These test curvature and an interaction in the log-odds; they are not clinical formulas or causal claims. "
        "Candidates and the retention rule were saved before examining their scores.", "",
        "Every engineered variant first fits numeric median imputation on its training fold, creates the specified "
        "terms from those imputed values, and fits standard scaling on that fold. Categorical mode imputation and "
        "one-hot encoding also stay inside the fold. Original raw feature roles, IDs, labels, saved split, folds, "
        "C=1.0, max_iter=1000, seed 42, and threshold 0.5 remain fixed. No parameter search was performed.", "",
        "The predeclared retention rule requires mean ROC-AUC gain of at least "
        f"{selection['minimum_mean_roc_auc_gain']:.4f}, positive gain in every fold, and no convergence warnings. "
        "Otherwise the original feature set is retained. This is a practical course-project rule, not a statistical test.", "",
        f"Development rows: {record['data']['development_rows']}; holdout rows excluded: "
        f"{record['data']['holdout_rows_excluded']}.", "",
        "## Development exploration", "",
        "The plots describe age and age/ST-depression label-rate patterns in the development pool. "
        "Their bins are exploratory and never used as model inputs. Such associations can reflect other predictors; "
        "they do not establish a causal effect or prove an engineered term will help.", "",
        "![Development exploration](feature_engineering_eda.png)", "",
        "## Complete comparison", "",
        "| Feature set | Status | Mean ROC-AUC | Fold SD | Mean gain | Accuracy | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in record["comparison"]:
        if row["status"] == "completed":
            lines.append(
                f"| {row['feature_set']} | completed | {row['roc_auc']:.6f} | {row['roc_auc_fold_sd']:.6f} | "
                f"{row['mean_delta_roc_auc']:+.6f} | {row['accuracy']:.6f} | {row['recall']:.6f} |"
            )
        else:
            lines.append(f"| {row['feature_set']} | failed | - | - | - | - | - |")
    lines += [
        "", "Values are proportions, not percentages. Scores are unweighted means across the same three folds. "
        "Fold SD is not a confidence interval; paired gains compare corresponding folds. "
        "All weak/negative and failed outcomes are retained in the full reference JSON and comparison CSV.", "",
        "![Feature comparison](feature_engineering_comparison.png)", "",
        "## Development recommendation", "",
        f"Recommended feature set under the frozen rule: **{selection['recommended_feature_set']}**. "
        f"Highest mean ROC-AUC: **{selection['highest_mean_roc_auc_feature_set']}**.", "",
        "These scores were used to compare feature sets and may be optimistic as selection estimates. "
        "The eventual selected method still needs the untouched final holdout after remaining project choices are frozen. "
        "No clinical-validity or statistical-significance claim follows from this experiment.", "",
        "## Reproduction and traceability", "",
        "[Full reference record](feature_engineering_reference.json) includes every candidate, paired gains, plan, "
        "data/split hashes, environment and source fingerprint. Individual runs retain fold CSVs, development "
        "predictions, warnings, and complete development-trained pipelines. Failed runs retain their errors.", "",
        "Run the feature_experiments command in the README using a new output directory. "
        "The original baselines and experiment inputs are preserved.",
    ]
    (output / "feature_engineering.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(config, plan, output):
    names = plan["feature_sets"]
    minimum_gain = plan["minimum_mean_roc_auc_gain"]
    if (
        not names or names[0] != "original" or len(set(names)) != len(names)
        or any(name not in FEATURE_SETS for name in names)
        or not isinstance(minimum_gain, (int, float)) or not np.isfinite(minimum_gain)
        or minimum_gain <= 0 or plan.get("require_positive_gain_in_every_fold") is not True
    ):
        raise ValueError("Invalid predeclared feature-experiment plan")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    started = perf_counter()
    records = []
    metadata = {"status": "running", "plan": plan, "code": code_metadata(), "holdout_evaluated": False, "test_csv_used": False}
    try:
        membership_path, _ = write_or_verify_splits(config)
        frame = validate_csv(config["data"]["train"], labeled=True)
        development = development_rows(frame, pd.read_csv(membership_path), config)
        source_hash = sha256(config["data"]["train"])
        membership_hash = sha256(membership_path)
        metadata["data"] = {
            "train_sha256": source_hash, "membership_sha256": membership_hash,
            "development_rows": len(development), "holdout_rows_excluded": len(frame) - len(development),
        }
        development_eda(development, config, output)
        original_scores = None
        for name in names:
            print(f"Running logistic / {name}", flush=True)
            try:
                run_dir, summary = run_baseline(config, "logistic", output / name, feature_set=name)
                run_metadata = json.loads((run_dir / "run_metadata.json").read_text())
                if run_metadata["train_sha256"] != source_hash or run_metadata["membership_sha256"] != membership_hash:
                    raise ValueError("Input data or split changed during the comparison")
                folds = pd.read_csv(run_dir / "cv_results.csv").sort_values("fold")
                scores = folds["roc_auc"].to_numpy()
                if name == "original":
                    original_scores = scores.copy()
                    metadata["environment"] = {key: run_metadata[key] for key in ("python", "platform", "machine", "threads", "packages")}
                gains = scores - original_scores
                row = {
                    "feature_set": name, "definitions": FEATURE_DEFINITIONS[name],
                    "added_features": len(FEATURE_DEFINITIONS[name]), "status": "completed",
                    **{metric: values["mean"] for metric, values in summary["metrics"].items()},
                    "roc_auc_fold_sd": summary["metrics"]["roc_auc"]["fold_sd"],
                    "mean_delta_roc_auc": float(gains.mean()),
                    "fold_roc_auc_gains": gains.tolist(),
                    "all_folds_improved": bool((gains > 0).all()),
                    "convergence_warnings": int(folds["convergence_warning"].sum()) + int(run_metadata["final_fit_convergence_warning"]),
                    "elapsed_seconds": run_metadata["elapsed_seconds"],
                }
                records.append(row)
                print(f"Mean ROC-AUC {row['roc_auc']:.6f}; gain {row['mean_delta_roc_auc']:+.6f}", flush=True)
            except Exception as error:
                records.append({"feature_set": name, "status": "failed", "error": str(error)})
                if name == "original":
                    raise
        selection = choose_variant(records, minimum_gain)
        pd.DataFrame(records).to_csv(output / "comparison.csv", index=False)
        comparison_figure(records, selection, output)
        metadata.update({
            "status": "completed" if all(row["status"] == "completed" for row in records) else "completed_with_failures",
            "selection": selection, "comparison": records, "elapsed_seconds": perf_counter() - started,
            "generated_directory": str(output),
        })
        write_json(output / "feature_engineering_reference.json", metadata)
        write_report(output, metadata)
    except Exception as error:
        metadata.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    finally:
        metadata["elapsed_seconds"] = perf_counter() - started
        write_json(output / "experiment_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--plan", default="configs/feature_experiments.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    plan = json.loads(Path(args.plan).read_text())
    output = args.output_dir or Path("reports/generated/feature-engineering") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = run_experiments(config, plan, output)
    print(f"Development recommendation: {result['selection']['recommended_feature_set']}")
    print(f"Saved report to {Path(output) / 'feature_engineering.md'}")
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

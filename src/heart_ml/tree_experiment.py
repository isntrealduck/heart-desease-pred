"""A small, recorded decision-tree search using frozen development folds."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .audit import sha256, validate_config, validate_csv
from .baseline import METRICS, code_metadata, development_rows, evaluate_model, run_baseline, write_json
from .split import write_or_verify_splits


def candidates_from_plan(plan):
    """Keep the search bounded and its selection rule explicit."""
    if plan.get("criterion") != "gini" or plan.get("selection_metric") != "roc_auc":
        raise ValueError("The tree plan requires Gini splits and ROC-AUC selection")
    if plan.get("tie_break") != "shallower depth, then larger minimum leaf size":
        raise ValueError("Unexpected tie-break rule")
    dimensions = []
    for key in ("max_depth", "min_samples_leaf"):
        values = plan.get(key)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in values
        ) or len(set(values)) != len(values):
            raise ValueError(f"{key} must contain unique positive integers")
        dimensions.append(values)
    candidates = [
        {"criterion": "gini", "max_depth": depth, "min_samples_leaf": leaf}
        for depth in dimensions[0] for leaf in dimensions[1]
    ]
    if len(candidates) > 8:
        raise ValueError("At most eight tree candidates are allowed")
    return candidates


def select_candidate(table):
    """Select mean CV ROC-AUC; exact ties prefer a smaller, smoother tree."""
    if table.empty or not np.isfinite(table["roc_auc_mean"]).all():
        raise ValueError("Candidate ROC-AUC values must be finite")
    return table.sort_values(
        ["roc_auc_mean", "max_depth", "min_samples_leaf"],
        ascending=[False, True, False], kind="stable",
    ).iloc[0]


def run_search(config, plan, output_dir=None):
    validate_config(config)
    candidates = candidates_from_plan(plan)
    threads = config.get("threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else Path("reports/generated/decision-tree") / run_id
    output.mkdir(parents=True, exist_ok=False)
    # Save the grid and rule before fitting any candidate.
    write_json(output / "plan.json", plan)
    write_json(output / "config.json", config)
    metadata = {
        "status": "running", "evaluation": "development_cross_validation",
        "holdout_evaluated": False, "test_csv_used": False,
        "code": code_metadata(), "plan_sha256": sha256(output / "plan.json"),
        "config_sha256": sha256(output / "config.json"),
    }
    try:
        membership_path, _ = write_or_verify_splits(config)
        frame = validate_csv(config["data"]["train"], labeled=True)
        development = development_rows(frame, pd.read_csv(membership_path), config)
        metadata.update({
            "train_sha256": sha256(config["data"]["train"]),
            "membership_sha256": sha256(membership_path),
            "development_rows": len(development),
            "holdout_rows_excluded": len(frame) - len(development),
        })
        del frame
        records, fold_tables = [], []
        with threadpool_limits(limits=threads):
            for number, params in enumerate(candidates):
                candidate_config = deepcopy(config)
                candidate_config["models"]["decision_tree"] = params
                folds, _ = evaluate_model(development, candidate_config, "decision_tree")
                folds.insert(0, "candidate", number)
                fold_tables.append(folds)
                record = {"candidate": number, **params}
                for metric in METRICS:
                    record[metric + "_mean"] = float(folds[metric].mean())
                    record[metric + "_fold_sd"] = float(folds[metric].std(ddof=1))
                records.append(record)
                # Persist every completed candidate, including weak results.
                pd.DataFrame(records).to_csv(output / "search_results.csv", index=False)
                pd.concat(fold_tables, ignore_index=True).to_csv(output / "search_folds.csv", index=False)
                print(f"Candidate {number}: {params}; ROC-AUC={record['roc_auc_mean']:.6f}", flush=True)
        table = pd.DataFrame(records)
        selected = select_candidate(table)
        selected_params = candidates[int(selected["candidate"])]
        chosen_config = deepcopy(config)
        chosen_config["models"]["decision_tree"] = selected_params
        selected_output, summary = run_baseline(chosen_config, "decision_tree", output / "selected")
        # Independently rerun the chosen folds while saving its full pipeline/predictions.
        original_folds = fold_tables[int(selected["candidate"])].drop(columns=["candidate", "elapsed_seconds"])
        repeated_folds = pd.read_csv(selected_output / "cv_results.csv", keep_default_na=False).drop(columns="elapsed_seconds")
        pd.testing.assert_frame_equal(original_folds, repeated_folds, check_dtype=False, rtol=1e-14, atol=1e-14)
        result = {
            "selection_metric": "mean development CV ROC-AUC",
            "selected_candidate": int(selected["candidate"]),
            "selected_parameters": selected_params,
            "candidate_count": len(candidates),
            "selected_summary": summary,
            "holdout_evaluated": False,
            "selection_caveat": "These folds selected the hyperparameters; scores are not an independent final evaluation.",
            "selected_fold_repeat_matches": True,
        }
        write_json(output / "selection.json", result)
        metadata["status"] = "completed"
        metadata["selected_candidate"] = int(selected["candidate"])
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_json(output / "search_metadata.json", metadata)
    return output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--plan", default="configs/decision_tree_search.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output, result = run_search(
        json.loads(Path(args.config).read_text(encoding="utf-8")),
        json.loads(Path(args.plan).read_text(encoding="utf-8")),
        args.output_dir,
    )
    print("Selected:", result["selected_parameters"])
    for metric, value in result["selected_summary"]["metrics"].items():
        print(f"{metric}: {value['mean']:.6f} (fold SD {value['fold_sd']:.6f})")
    print(f"Outputs: {output}; final holdout NOT evaluated")


if __name__ == "__main__":
    main()

"""Bounded Random Forest search on the frozen development folds."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.audit import sha256, validate_config, validate_csv
from src.baseline import METRICS, code_metadata, development_rows, evaluate_model, run_baseline, write_json
from src.split import write_or_verify_splits


def _thread_count(config):
    threads = config.get("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    return threads


def candidates_from_plan(plan, threads):
    """Expand the deliberately small forest grid after validating its bounds."""
    if plan.get("selection_metric") != "roc_auc":
        raise ValueError("Random Forest selection metric must be roc_auc")
    if plan.get("tie_break") != "shallower depth, then larger minimum leaf size":
        raise ValueError("Unexpected tie-break rule")
    depths, leaves = plan.get("max_depth"), plan.get("min_samples_leaf")
    trees = plan.get("n_estimators")
    max_candidates = plan.get("max_candidates", 8)
    max_features = plan.get("max_features", "sqrt")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 8:
        raise ValueError("max_candidates must be an integer from 1 to 8")
    if max_features != "sqrt":
        raise ValueError('max_features is fixed to "sqrt" for this search')
    if (not isinstance(depths, list) or not depths or
        any(type(v) is not int or not 1 <= v <= 30 for v in depths) or len(set(depths)) != len(depths)):
        raise ValueError("max_depth must contain unique integers from 1 to 30")
    if (not isinstance(leaves, list) or not leaves or
        any(type(v) is not int or not 1 <= v <= 10000 for v in leaves) or len(set(leaves)) != len(leaves)):
        raise ValueError("min_samples_leaf must contain unique positive integers")
    if type(trees) is not int or not 1 <= trees <= 500:
        raise ValueError("n_estimators must be between 1 and 500")
    candidates = [
        {"n_estimators": trees, "max_depth": d, "min_samples_leaf": l,
         "max_features": plan.get("max_features", "sqrt"), "n_jobs": threads}
        for d in depths for l in leaves
    ]
    if len(candidates) > max_candidates:
        raise ValueError("At most eight Random Forest candidates are allowed")
    return candidates


def select_candidate(table):
    if table.empty or not np.isfinite(table["roc_auc_mean"]).all():
        raise ValueError("Candidate ROC-AUC values must be finite")
    return table.sort_values(["roc_auc_mean", "max_depth", "min_samples_leaf"],
                             ascending=[False, True, False], kind="stable").iloc[0]


def run_search(config, plan, output_dir=None):
    """Evaluate candidates on saved development folds, then refit the winner."""
    validate_config(config)
    threads = _thread_count(config)
    candidates = candidates_from_plan(plan, threads)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else Path("reports/generated/random-forest") / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "config.json", config)
    metadata = {"status": "running", "evaluation": "development_cross_validation",
                "holdout_evaluated": False, "test_csv_used": False, "code": code_metadata()}
    started = perf_counter()
    try:
        metadata["plan_sha256"] = sha256(output / "plan.json")
        metadata["config_sha256"] = sha256(output / "config.json")
        membership_path, _ = write_or_verify_splits(config)
        frame = validate_csv(config["data"]["train"], labeled=True)
        development = development_rows(frame, pd.read_csv(membership_path), config)
        metadata.update(train_sha256=sha256(config["data"]["train"]),
                        membership_sha256=sha256(membership_path),
                        development_rows=len(development),
                        holdout_rows_excluded=len(frame)-len(development))
        records, folds_all = [], []
        with threadpool_limits(limits=threads):
            for i, params in enumerate(candidates):
                candidate_config = deepcopy(config)
                merged_params = {**candidate_config["models"].get("random_forest", {}), **params}
                candidate_config["models"]["random_forest"] = merged_params
                candidate_started = perf_counter()
                folds, _ = evaluate_model(development, candidate_config, "random_forest")
                candidate_elapsed = perf_counter() - candidate_started
                folds.insert(0, "candidate", i)
                folds_all.append(folds)
                record = {"candidate": i, **merged_params, "elapsed_seconds": candidate_elapsed}
                for metric in METRICS:
                    record[metric+"_mean"] = float(folds[metric].mean())
                    record[metric+"_fold_sd"] = float(folds[metric].std(ddof=1))
                records.append(record)
                pd.DataFrame(records).to_csv(output / "search_results.csv", index=False)
                pd.concat(folds_all, ignore_index=True).to_csv(output / "search_folds.csv", index=False)
                print(f"Candidate {i}: ROC-AUC={record['roc_auc_mean']:.6f}", flush=True)
        table = pd.DataFrame(records)
        selected = select_candidate(table)
        params = candidates[int(selected["candidate"])]
        chosen = deepcopy(config)
        selected_params = {**chosen["models"].get("random_forest", {}), **params}
        chosen["models"]["random_forest"] = selected_params
        selected_output, summary = run_baseline(chosen, "random_forest", output / "selected")
        result = {"selection_metric": "mean development CV ROC-AUC",
                  "selected_candidate": int(selected["candidate"]),
                  "selected_parameters": selected_params, "candidate_count": len(candidates),
                  "selected_summary": summary, "holdout_evaluated": False,
                  "selection_caveat": "Search folds select hyperparameters; not an independent final evaluation."}
        write_json(output / "selection.json", result)
        metadata.update(status="completed", selected_candidate=result["selected_candidate"])
    except Exception as error:
        metadata.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        metadata["elapsed_seconds"] = perf_counter() - started
        write_json(output / "search_metadata.json", metadata)
    return output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--plan", default="configs/random_forest_search.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output, result = run_search(json.loads(Path(args.config).read_text()),
                                json.loads(Path(args.plan).read_text()), args.output_dir)
    print("Selected:", result["selected_parameters"])
    print("Outputs:", output, "(holdout NOT evaluated)")


if __name__ == "__main__":
    main()

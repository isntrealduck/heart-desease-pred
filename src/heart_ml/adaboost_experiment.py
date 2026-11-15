"""Small AdaBoost search using frozen development folds only."""

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd

from .audit import validate_config
from .baseline import run_baseline, write_json


def candidates_from_plan(plan):
    if (
        plan.get("selection_metric") != "roc_auc"
        or plan.get("tie_break") != "fewer estimators, then lower learning rate"
    ):
        raise ValueError("Unexpected AdaBoost selection rule")
    counts, rates = plan.get("n_estimators"), plan.get("learning_rate")
    if (
        not isinstance(counts, list)
        or not counts
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("n_estimators must contain unique positive integers")
    if (
        not isinstance(rates, list)
        or not rates
        or any(
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not np.isfinite(x)
            or x <= 0
            for x in rates
        )
        or len(set(rates)) != len(rates)
    ):
        raise ValueError("learning_rate must contain unique positive finite numbers")
    candidates = [
        {"n_estimators": n, "learning_rate": rate}
        for n, rate in itertools.product(counts, rates)
    ]
    if len(candidates) > 8:
        raise ValueError("At most eight AdaBoost candidates are allowed")
    return candidates


def select_candidate(table):
    if table.empty or not np.isfinite(table["roc_auc_mean"]).all():
        raise ValueError("Candidate ROC-AUC values must be finite")
    return table.sort_values(
        ["roc_auc_mean", "n_estimators", "learning_rate"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]


def run_search(config, plan, output_dir=None):
    """Record candidate fold results, then persist the selected development pipeline."""
    validate_config(config)
    candidates = candidates_from_plan(plan)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else Path("reports/generated/adaboost_search") / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "config.json", config)
    records, folds_all = [], []
    try:
        # Reuse baseline's fixed split, fold-fitted transforms, metrics and serialization.
        # Small development-only selection runs are performed as isolated runs per candidate.
        for number, params in enumerate(candidates):
            candidate_config = deepcopy(config)
            candidate_config["models"]["adaboost"] = params
            candidate_dir = output / f"candidate_{number}"
            candidate_started = perf_counter()
            _, _ = run_baseline(candidate_config, "adaboost", candidate_dir)
            folds = pd.read_csv(candidate_dir / "cv_results.csv")
            folds.insert(0, "candidate", number)
            folds_all.append(folds)
            record = {"candidate": number, **params}
            for metric in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"):
                record[metric + "_mean"] = float(folds[metric].mean())
                record[metric + "_fold_sd"] = float(folds[metric].std(ddof=1))
            record["cv_elapsed_seconds"] = float(folds["elapsed_seconds"].sum())
            record["elapsed_seconds"] = perf_counter() - candidate_started
            record["warnings"] = " | ".join(folds["notes"].dropna().astype(str).tolist())
            records.append(record)
            pd.DataFrame(records).to_csv(output / "search_results.csv", index=False)
            pd.concat(folds_all, ignore_index=True).to_csv(output / "search_folds.csv", index=False)
            print(f"Candidate {number}: {params}; ROC-AUC={record['roc_auc_mean']:.6f}", flush=True)
        table = pd.DataFrame(records)
        chosen = select_candidate(table)
        params = candidates[int(chosen["candidate"])]
        chosen_config = deepcopy(config)
        chosen_config["models"]["adaboost"] = params
        selected_dir, summary = run_baseline(chosen_config, "adaboost", output / "selected")
        result = {"selection_metric": "mean development CV ROC-AUC",
                  "selected_candidate": int(chosen["candidate"]), "selected_parameters": params,
                  "candidate_count": len(candidates), "selected_summary": summary,
                  "selected_pipeline": str(selected_dir / "adaboost_pipeline.joblib"),
                  "holdout_evaluated": False,
                  "selection_caveat": "The same development folds selected hyperparameters; scores are not independent.",
                  "runtime_note": "Each candidate baseline runs CV and a full-development refit; the selected run repeats CV and refits its pipeline."}
        write_json(output / "selection.json", result)
        return output, result
    except Exception as error:
        write_json(output / "search_metadata.json", {"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    finally:
        if not (output / "search_metadata.json").exists():
            write_json(output / "search_metadata.json", {"status": "completed" if (output / "selection.json").exists() else "failed",
                                                         "holdout_evaluated": False, "test_csv_used": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--plan", default="configs/adaboost_search.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output, result = run_search(json.loads(Path(args.config).read_text()),
                                json.loads(Path(args.plan).read_text()), args.output_dir)
    print("Selected:", result["selected_parameters"])
    print("Outputs:", output, "(holdout not evaluated)")


if __name__ == "__main__":
    main()

# Project instructions

- Read `PROJECT_PLAN.md` before ML or evaluation changes. It is a plan, not evidence of implementation; verify current files, dependencies, and commands.
- This is a small two-person university ML project. Prefer clear, documented implementations both teammates can understand; avoid enterprise process overhead.
- Keep instructions neutral and shared. Plan ownership labels are coordination suggestions, not assumptions about who is working.

## Scientific protocol

- Unless teammates agree to a justified change, exclude `id` from features and explicitly map `Absence` to 0 and `Presence` to 1.
- Keep `test.csv` unlabeled: generate predictions only, never local performance metrics from it.
- Reserve a stratified 20% train holdout with seed 42; use shuffled 3-fold stratified CV with seed 42 on development data. Select by ROC-AUC, use threshold 0.5 for primary thresholded metrics, and never tune on the holdout.
- Fit preprocessing within each fold using scikit-learn `Pipeline`/`ColumnTransformer`; do not leak validation or holdout information.
- Keep scope bounded: dummy baseline, logistic regression, and CPU `HistGradientBoostingClassifier`; use small, recorded tuning only.
- Persist holdout and fold membership by `id` alongside fixed configuration.
- Save the entire fitted preprocessing-plus-model pipeline. Prediction must load it without retraining and use `classes_` to locate the positive-class probability.
- Distinguish development CV from final holdout results. Any optional all-labeled-data refit is a separate artifact, not a newly evaluated model.

## Implementation and data care

- Docker is required for setup, training, evaluation, prediction, and tests. This overrides conflicting Docker-optional wording in `PROJECT_PLAN.md`. Follow the README for tested commands. Setup, audit, split, synthetic tests, and development-only dummy/logistic baselines are implemented; final holdout evaluation and test-file prediction are not. Use `python -m heart_ml.baseline --model dummy` or `--model logistic` for the shared original-feature workflow.
- Python 3.12 is the plan's proposed starting version, not a verified compatibility claim. Verify dependencies and Docker build/run behavior before calling commands tested; document exact tested commands in the README when implemented.
- Explain preprocessing and metric choices in the README. Use comments/docstrings for non-obvious logic, not to restate code.
- Add small synthetic tests for split non-overlap, schema validation, and prediction shape/range. Routine tests should not require downloaded data or full expensive searches.
- Record experiment settings and results, including weak outcomes and failures, so report claims trace to saved outputs. Do not make clinical claims; data provenance is unconfirmed.
- Preserve supplied CSV originals. Keep raw data and generated artifacts/predictions out of commits unless redistribution or tracking is explicitly agreed and permitted.

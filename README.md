# Heart disease classification project

Implemented: input audit, saved ID-based split, shared pipeline constructors, and tested development-only dummy and original-feature logistic-regression experiments. **Final model selection, holdout evaluation, and test-file predictions are not implemented.** See the [dummy report](reports/dummy_baseline.md) and [logistic comparison report](reports/logistic_baseline.md) for measured reference results.

## Data and limitations

Place the supplied datasets at `data/raw/train.csv` and `data/raw/test.csv`; preserve their original contents. The local files appear consistent with Kaggle Playground Series S6E2, but provenance, access terms, units, category meanings, patient/group identity, and row independence are unconfirmed. `id` is an identifier, not a verified patient identifier. No clinical conclusions are supported. Audit output records hashes, dimensions, label counts, numeric ranges, categories, and ID uniqueness; it does not establish data quality beyond these checks.

`configs/default.json` fixes the 20% stratified holdout (seed 42), shuffled three-fold CV on development rows, ROC-AUC selection metric, threshold 0.5, features, and CPU model defaults. `id` is excluded from model inputs; labels map explicitly Absence=0, Presence=1. Numeric median imputation/scaling (for logistic regression) and categorical mode imputation/one-hot encoding are composed in the pipeline and thus can be fitted within folds. HistGradientBoosting uses dense one-hot inputs. Model constructors return unfitted pipelines. Holdout rows have fold `-1` and never enter development folds.

Audit allows missing predictors (reported per column for later imputation), but rejects missing IDs/labels and nonnumeric or nonfinite values. It reports observed ranges/categories and duplicate feature-row count without imposing clinical ranges or dropping rows. Coded categorical values are validated as finite numeric codes; no permitted value ranges are asserted.

## Docker setup and tested commands

Docker is required. The initial setup was documented as tested on linux/arm64. Both baselines were verified on 14 November 2026 using Docker Engine 29.9.3, Python 3.12.16, and linux/amd64 under WSL2 on Windows. The updated image built successfully and all 24 synthetic tests passed. Repeat real-data runs produced identical metric summaries and byte-identical development predictions for each model. Lock versions are explicit. Run from the repository root; the braced volume syntax works in PowerShell and POSIX shells:

```sh
docker build -t heart-ml .
docker run --rm -v "${PWD}:/app" heart-ml python -m pytest -q
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.audit --config configs/default.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.split --config configs/default.json
```

The image installs the package editable. Bind-mounting the project ensures source edits and generated outputs are immediately visible and persistent; the repository's `src` shadows the image copy. `DATA_MANIFEST.json` and `data/splits/` are generated, git-ignored outputs; raw CSVs and these generated files are excluded from Docker build context as well. The audit manifest includes SHA-256 hashes.

If PowerShell cannot find Docker after a per-user installation, add its executable folder to the current terminal's PATH before running these commands:

```powershell
$env:Path = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;" + $env:Path
docker version
```

The folder must contain both docker.exe and its credential helper. This changes only the current terminal environment. Installations in another location should use their actual executable folder. Docker Desktop's Linux engine must be running; a working CLI alone is insufficient.

Split metadata records the training CSV hash and a protocol hash of ID/target/label mapping/seed/holdout fraction/fold count, plus a membership-file hash. Model parameters, threshold, and thread settings do not change split identity. Split creation rejects incomplete files, protocol/source mismatches, and membership corruption rather than replacing them. If resetting a setup-only split, first back up both split files and explicitly remove the pair before recreating; do not reset after experiments rely on it. Inputs must match expected schemas exactly.

## Dummy classifier: development baseline

The existing constructor uses `DummyClassifier(strategy="prior")`. It learns the training class proportions and assigns the same disease probability to every row, regardless of its features. This is a no-feature benchmark for logistic regression and boosting. The dummy needs no scaling or encoding because it ignores predictor values. The learned-model constructors retain their fold-fitted preprocessing pipelines.

After placing train.csv at `data/raw/train.csv`, run:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --config configs/default.json
```

The command creates or verifies the saved split using the existing split code. It joins membership by ID, removes holdout rows, and fits a fresh dummy pipeline on the other two development folds before predicting the remaining fold. Class proportions are learned separately from each training fold. It never opens test.csv and never evaluates the final holdout. It then fits and saves a baseline pipeline on all development rows for later reuse; that artifact has no final holdout score.

Each run gets a new directory under `reports/generated/dummy/`:

| File | Contents |
| --- | --- |
| `cv_results.csv` | Per-fold scores, class proportions, row counts, confusion counts, elapsed time, warnings/notes |
| `development_predictions.csv` | ID, validation fold, true label, out-of-fold probability of Presence, predicted label |
| `summary.json` | Unweighted metric means and sample SD across folds (`ddof=1`) |
| `dummy_pipeline.joblib` | Complete dummy pipeline fitted on development rows only |
| `config.json` | Settings used for this run |
| `run_metadata.json` | Completion/failure status, hashes, split metadata, source fingerprint, versions, and timing |

For a chosen new directory, add `--output-dir reports/generated/my-dummy-run`. Existing output directories are refused rather than overwritten. Failed experiments record an error in run metadata; check its status before using any partial outputs.

ROC-AUC measures ranking; average precision (AP) summarizes precision over recall increments. Both use positive-class probabilities. Accuracy, positive-class precision/recall/F1, and the confusion matrix use `probability_presence >= threshold`, initially 0.5. Undefined precision/recall/F1 values use `zero_division=0`; a fold with no positive predictions records a note. AP differs from trapezoidal PR-AUC. Fold SD is not a confidence interval.

A constant-probability classifier has ROC-AUC 0.5 when both validation classes occur. If each training fold's positive fraction is below 0.5, it predicts no positives at the default threshold; its positive-class recall is then zero. These are mathematical expectations, not measured project results. Keep such weak results in the report. Do not report pooled out-of-fold ROC-AUC for this baseline: small differences in fold-specific priors can produce a misleading ranking across folds. Use the recorded per-fold scores and their mean.

Synthetic tests check fold-specific prior probabilities, ID-based joins, holdout exclusion from fit/predict, insensitivity to changed holdout values, saved-pipeline behavior, and recorded failures. Run them with the Docker pytest command above.

The measured reference run used 504,000 development rows and excluded all 126,000 holdout rows. Mean CV ROC-AUC was 0.500000, accuracy was 55.1661%, and positive-class precision, recall, and F1 were all zero. The [generated baseline report](reports/dummy_baseline.md), [full-precision summary](reports/dummy_reference.json), and [run metadata](reports/dummy_run_metadata.json) retain the values and traceability information. This is repeatability within the tested image; a fresh-clone release reproduction and cross-platform verification remain future work.

## Logistic regression: original-feature baseline

Run the same evaluation workflow with the learned classifier:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model logistic --config configs/default.json
```

This uses the existing 13 predictors with numeric imputation/scaling and categorical imputation/one-hot encoding inside each fold's pipeline. It adds no derived predictors, nonlinear terms, or interactions. Fixed settings are L2 regularization, C=1.0, solver=lbfgs, max_iter=1000, tolerance=0.0001, seed 42, and one numeric-library thread. No hyperparameters or thresholds were tuned. This provides the learned baseline needed to measure later feature-engineering gains.

The command uses the same persisted development folds, metrics, threshold 0.5, and output format as the dummy. Default outputs go under `reports/generated/logistic/<run-id>/`, with `logistic_pipeline.joblib` storing the full preprocessing/classifier pipeline fitted on all development rows. For a named new directory, add `--output-dir reports/generated/my-logistic-run`. The original dummy command remains supported; `--model dummy` is optional.

CV results also record solver iterations and convergence warnings; metadata records final-fit warnings and resolved classifier parameters. A completed run with warnings must be interpreted with those warnings, rather than assuming successful convergence.

The measured logistic reference used 504,000 development rows and excluded all 126,000 holdout rows. Mean development ROC-AUC was 0.952470, accuracy was 88.4966%, and positive-class recall was 85.9287%. All three fold fits and the development-wide fit converged in 15 iterations without recorded warnings. A repeat run in a fresh container produced identical summaries and byte-identical predictions.

See the [comparison report](reports/logistic_baseline.md), [full-precision logistic summary](reports/logistic_reference.json), and [run metadata](reports/logistic_run_metadata.json). The comparison verifies matching training-data, configuration, and split hashes for dummy and logistic. These are development CV results, not final holdout scores or external clinical validation. The saved pipeline is development-trained and has no final holdout score.

## Status and next steps

Confirm data provenance/independence assumptions, then use the original-feature logistic results as the reference for feature-engineering experiments and boosting comparison on the same saved folds. Final model selection, holdout evaluation, test prediction, a full-project reproduction command, and clean-clone release verification remain future work.

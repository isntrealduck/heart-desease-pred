# Heart disease classification project

Implemented: input audit, saved ID-based split, shared pipeline constructors, tested dummy/logistic baselines, and a bounded development-only feature-engineering study, and a decision-tree search. **Final model selection, holdout evaluation, and test-file predictions are not implemented.** See the [dummy report](reports/dummy_baseline.md) and [logistic comparison report](reports/logistic_baseline.md) for measured reference results.

## Data and limitations

Place the supplied datasets at `data/raw/train.csv` and `data/raw/test.csv`; preserve their original contents. The local files appear consistent with Kaggle Playground Series S6E2, but provenance, access terms, units, category meanings, patient/group identity, and row independence are unconfirmed. `id` is an identifier, not a verified patient identifier. No clinical conclusions are supported. Audit output records hashes, dimensions, label counts, numeric ranges, categories, and ID uniqueness; it does not establish data quality beyond these checks.

`configs/default.json` fixes the 20% stratified holdout (seed 42), shuffled three-fold CV on development rows, ROC-AUC selection metric, threshold 0.5, features, and CPU model defaults. `id` is excluded from model inputs; labels map explicitly Absence=0, Presence=1. Numeric median imputation/scaling (for logistic regression) and categorical mode imputation/one-hot encoding are composed in the pipeline and thus can be fitted within folds. HistGradientBoosting uses dense one-hot inputs. Model constructors return unfitted pipelines. Holdout rows have fold `-1` and never enter development folds.

Audit allows missing predictors (reported per column for later imputation), but rejects missing IDs/labels and nonnumeric or nonfinite values. It reports observed ranges/categories and duplicate feature-row count without imposing clinical ranges or dropping rows. Coded categorical values are validated as finite numeric codes; no permitted value ranges are asserted.

## Docker setup and tested commands

Docker is required. The initial setup was documented as tested on linux/arm64. Both baselines were verified on 14 November 2026 using Docker Engine 29.9.3, Python 3.12.16, and linux/amd64 under WSL2 on Windows. The updated image built successfully; the current project passes all 39 synthetic tests. Repeat real-data runs produced identical metric summaries and byte-identical development predictions for each model. Lock versions are explicit. Run from the repository root; the braced volume syntax works in PowerShell and POSIX shells:

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

## Feature-engineering experiments

Run the fixed comparison with the original logistic classifier settings:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.feature_experiments --config configs/default.json --plan configs/feature_experiments.json
```

The plan compares original features, Age squared, Age times ST depression, and both additions together. These are limited hypotheses about curvature and interaction in the log-odds. Numeric median imputation is fitted inside each training fold, then the new terms are created from the imputed values and scaled inside that fold. Original raw feature roles and categorical handling remain fixed; no raw CSVs are changed. C, solver, seed, threshold, saved split, and CV folds remain unchanged.

The plan is saved before candidate scores are examined. Retaining an engineered variant requires mean ROC-AUC gain of at least 0.0005, positive gain in every fold, and no convergence warnings. This is a predeclared practical rule, not a significance test. Exploration uses development rows only; its bins are descriptive and never model inputs.

Measured mean development ROC-AUC:

| Variant | ROC-AUC | Gain over original |
| --- | ---: | ---: |
| Original | 0.952470 | 0.000000 |
| Age squared | 0.952480 | +0.000010 |
| Age times ST depression | 0.952472 | +0.000002 |
| Both additions | 0.952479 | +0.000009 |

**Retain the original features under the predeclared rule.** All fits converged, but none met the required gain. These outcomes do not imply that every possible feature-engineering approach is ineffective. All results and paired fold changes are retained in the [feature report](reports/feature_engineering.md) and [full reference JSON](reports/feature_engineering_reference.json), with [exploration values](reports/feature_engineering_eda.json) and [verification evidence](reports/feature_engineering_verification.json).

Outputs go to a new directory under reports/generated/feature-engineering/. Each variant retains its own metrics, development predictions, warnings, settings, metadata, and full development-trained pipeline. The suite also saves the exploration/comparison figures and report. Existing output directories are never overwritten.

For a single reproducible variant:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model logistic --feature-set age_squared --config configs/default.json
```

All 36 synthetic tests passed. The original control reproduced the published logistic predictions exactly, and a fresh-container Age-squared repeat produced identical summaries and prediction bytes. The final holdout and test.csv remain unevaluated. Feature-set selection on development CV is not an independent final performance estimate.

## Decision tree: bounded development search

Train and compare eight trees on the same saved development folds:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.tree_experiment --config configs/default.json --plan configs/decision_tree_search.json
```

The plan crosses maximum depths 3, 5, 8 and 12 with minimum leaf sizes 50 and 200. It is saved before training. Selection uses highest mean CV ROC-AUC; exact ties prefer shallower depth, then larger minimum leaf size. The selected folds are rerun to check repeatability, and the complete pipeline is fitted on development rows only. No final holdout or test-file evaluation occurs.

The tree learns feature thresholds and interactions using greedy Gini-impurity reduction. Depth and leaf-size limits control complexity. Numeric median imputation and categorical mode imputation/one-hot encoding are fitted inside each fold. No numeric scaling or derived features are used. See [scikit-learn's decision-tree guide](https://scikit-learn.org/1.7/modules/tree.html).

Each new output directory under `reports/generated/decision-tree/` contains the frozen plan/config, `search_results.csv` (all candidate summaries), `search_folds.csv` (all fold metrics/confusion counts), `selection.json`, and `search_metadata.json`. The `selected/` folder contains the chosen settings, six metric summaries, per-fold confusion counts, out-of-fold predictions, run metadata, and `decision_tree_pipeline.joblib`. Existing output folders are refused.

Repeat just a selected tree with its saved configuration:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model decision_tree --config reports/generated/decision-tree-reference/selected/config.json
```

That path is from the local reference run; for another search use its own selected/config.json. Running `--model decision_tree` with configs/default.json instead uses the initial depth-8/leaf-50 preset, not necessarily the search winner. All thresholded scores use positive probability >= 0.5, including exact ties; calling the estimator's predict method can handle exact ties differently.

Measured reference: maximum depth 12, minimum leaf size 200, mean CV ROC-AUC **0.949041**, AP **0.939705**, accuracy **88.0242%**, precision **86.9890%**, recall **86.1783%**, and F1 **0.865816**. Logistic remains stronger by ROC-AUC (0.952470), accuracy and F1; tree recall is slightly higher at the shared threshold. All 39 tests passed. A fresh-container repeat produced identical summaries and byte-identical development predictions; all fold metrics and confusion counts were checked against the saved OOF predictions.

See the [decision-tree report](reports/decision_tree.md), [full-precision summary](reports/decision_tree_reference.json), [all candidate results](reports/decision_tree_search_results.csv), and [verification evidence](reports/decision_tree_verification.json). To reproduce the selected settings without searching, use `--model decision_tree --config reports/decision_tree_selected_config.json` with the shared baseline command.

The final report must label these scores development CV: eight configurations were compared on these folds, whereas logistic regression used fixed settings. These scores are not an independent final performance estimate.

## Status and next steps

Confirm data provenance/independence assumptions. The bounded feature study retains the original features under its frozen rule; compare that logistic baseline with the decision tree and the teammate-led AdaBoost experiment on the same saved folds. Final model selection, holdout evaluation, test prediction, a full-project reproduction command, and clean-clone release verification remain future work.

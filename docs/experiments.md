# Heart disease classification project

Implemented: input audit, saved ID-based split, tested baselines and bounded studies, and two explicitly documented evaluations reusing the same holdout. These are not independent validation results. The pipelines and results are saved locally; test-file predictions are not implemented. See the [dummy report](../reports/dummy_baseline.md) and [logistic comparison report](../reports/logistic_baseline.md) for development-CV results.

## Data and limitations

Place the supplied datasets at `data/raw/train.csv` and `data/raw/test.csv`; preserve their original contents. The local files appear consistent with Kaggle Playground Series S6E2, but provenance, access terms, units, category meanings, patient/group identity, and row independence are unconfirmed. `id` is an identifier, not a verified patient identifier. No clinical conclusions are supported. Audit output records hashes, dimensions, label counts, numeric ranges, categories, and ID uniqueness; it does not establish data quality beyond these checks.

`configs/default.json` fixes the 20% stratified holdout (seed 42), shuffled three-fold CV on development rows, ROC-AUC selection metric, threshold 0.5, features, and CPU model defaults. `id` is excluded from model inputs; labels map explicitly Absence=0, Presence=1. Numeric median imputation/scaling (for logistic regression) and categorical mode imputation/one-hot encoding are composed in the pipeline and thus can be fitted within folds. HistGradientBoosting uses dense one-hot inputs. Model constructors return unfitted pipelines. Holdout rows have fold `-1` and never enter development folds.

Audit allows missing predictors (reported per column for later imputation), but rejects missing IDs/labels and nonnumeric or nonfinite values. It reports observed ranges/categories and duplicate feature-row count without imposing clinical ranges or dropping rows. Coded categorical values are validated as finite numeric codes; no permitted value ranges are asserted.

## Docker setup and tested commands

Docker is required. The initial setup was documented as tested on linux/arm64. Both baselines were verified on 14 November 2026 using Docker Engine 29.9.3, Python 3.12.16, and linux/amd64 under WSL2 on Windows. The current 76-test synthetic suite has also passed independently on linux/arm64. Repeat development-data runs produced identical metric summaries and byte-identical predictions for each model. Lock versions are explicit. Run from the repository root; the braced volume syntax works in PowerShell and POSIX shells:

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

The measured reference run used 504,000 development rows and excluded all 126,000 holdout rows. Mean CV ROC-AUC was 0.500000, accuracy was 55.1661%, and positive-class precision, recall, and F1 were all zero. The [generated baseline report](../reports/dummy_baseline.md), [full-precision summary](../reports/dummy_reference.json), and [run metadata](../reports/dummy_run_metadata.json) retain the values and traceability information. This is repeatability within the tested image; a fresh-clone release reproduction and cross-platform verification remain future work.

## Logistic regression: original-feature baseline

Run the same evaluation workflow with the learned classifier:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model logistic --config configs/default.json
```

This uses the existing 13 predictors with numeric imputation/scaling and categorical imputation/one-hot encoding inside each fold's pipeline. It adds no derived predictors, nonlinear terms, or interactions. Fixed settings are L2 regularization, C=1.0, solver=lbfgs, max_iter=1000, tolerance=0.0001, seed 42, and one numeric-library thread. No hyperparameters or thresholds were tuned. This provides the learned baseline needed to measure later feature-engineering gains.

The command uses the same persisted development folds, metrics, threshold 0.5, and output format as the dummy. Default outputs go under `reports/generated/logistic/<run-id>/`, with `logistic_pipeline.joblib` storing the full preprocessing/classifier pipeline fitted on all development rows. For a named new directory, add `--output-dir reports/generated/my-logistic-run`. The original dummy command remains supported; `--model dummy` is optional.

CV results also record solver iterations and convergence warnings; metadata records final-fit warnings and resolved classifier parameters. A completed run with warnings must be interpreted with those warnings, rather than assuming successful convergence.

The measured logistic reference used 504,000 development rows and excluded all 126,000 holdout rows. Mean development ROC-AUC was 0.952470, accuracy was 88.4966%, and positive-class recall was 85.9287%. All three fold fits and the development-wide fit converged in 15 iterations without recorded warnings. A repeat run in a fresh container produced identical summaries and byte-identical predictions.

See the [comparison report](../reports/logistic_baseline.md), [full-precision logistic summary](../reports/logistic_reference.json), and [run metadata](../reports/logistic_run_metadata.json). The comparison verifies matching training-data, configuration, and split hashes for dummy and logistic. These are development CV results, not final holdout scores or external clinical validation. The saved pipeline is development-trained and has no final holdout score.

## Feature association heatmaps

To inspect how predictors relate to one another, run:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.feature_associations --config configs/default.json --output-dir reports/feature-associations
```

The analysis uses only the saved development partition; it does not read `test.csv` or the reserved final holdout. It writes heatmaps and matching CSV matrices for numeric-feature Pearson and Spearman correlations, an exploratory Spearman matrix of categorical codes, categorical-feature bias-corrected Cramer's V, and categorical/numeric eta association. Pearson and Spearman values are signed; Cramer's V and eta are unsigned association strengths from 0 to 1. Category-code Spearman values are only interpretable where codes have a meaningful order; use Cramer's V for general categorical association. Associations are descriptive and do not establish causality. Choose a new output directory for each run because existing output directories are refused.

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

**Retain the original features under the predeclared rule.** All fits converged, but none met the required gain. These outcomes do not imply that every possible feature-engineering approach is ineffective. All results and paired fold changes are retained in the [feature report](../reports/feature_engineering.md) and [full reference JSON](../reports/feature_engineering_reference.json), with [exploration values](../reports/feature_engineering_eda.json) and [verification evidence](../reports/feature_engineering_verification.json).

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

Measured reference: maximum depth 12, minimum leaf size 200, mean CV ROC-AUC **0.949041**, AP **0.939705**, accuracy **88.0242%**, precision **86.9890%**, recall **86.1783%**, and F1 **0.865816**. Logistic remains stronger by ROC-AUC (0.952470), accuracy and F1; tree recall is slightly higher at the shared threshold. At the time of the original tree verification, all 39 then-existing tests passed. A fresh-container repeat produced identical summaries and byte-identical development predictions; all fold metrics and confusion counts were checked against the saved OOF predictions.

See the [decision-tree report](../reports/decision_tree.md), [full-precision summary](../reports/decision_tree_reference.json), [all candidate results](../reports/decision_tree_search_results.csv), and [verification evidence](../reports/decision_tree_verification.json). To reproduce the selected settings without searching, use `--model decision_tree --config reports/decision_tree_selected_config.json` with the shared baseline command.

The final report must label tree scores as development CV: the eight tree configurations were compared on these folds, whereas logistic regression used fixed settings. These scores are not an independent final performance estimate.

## AdaBoost experiment

AdaBoost is available in the shared original-feature baseline and uses decision-tree stumps (scikit-learn's default depth-1 weak learner). Each round increases the influence of examples misclassified by earlier learners; `n_estimators` bounds rounds and training may stop early if a learner makes no errors. Numeric values are median-imputed, categorical codes are mode-imputed and one-hot encoded inside each fold; tree models are not scaled. The default is 50 estimators and learning rate 1.0. No SAMME.R or explicit estimator object is configured, preserving compatibility with the pinned scikit-learn API.

Run the baseline or bounded six-candidate search:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model adaboost --config configs/default.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.adaboost_experiment --config configs/default.json --plan configs/adaboost_search.json
```

The search crosses 25/50/100 estimators with learning rates 0.5/1.0, selects by mean ROC-AUC on the saved three development folds (ties: fewer estimators, then lower learning rate), and records candidate/fold metrics and warnings. It does not read the unlabeled test CSV or evaluate the holdout. Generated candidate outputs contain each development-fitted pipeline; `selected/` contains the selected pipeline and run outputs. Each candidate performs CV and a full-development refit; the selected run repeats CV and refits, so a full-data search can be expensive. A separate fixed-default baseline was run on the full development set: mean ROC-AUC 0.952684, AP 0.945670, accuracy 88.5393%, precision 88.1322%, recall 86.0216%, F1 0.870639. This is not a tuned search result or holdout score; its saved summary is `reports/generated/adaboost/20261114T223109215876Z/summary.json`.

### Completed one-time frozen AdaBoost holdout evaluation

AdaBoost was selected as the final model based on the highest mean development-CV ROC-AUC in the comparison below. The frozen model uses 50 depth-1 decision-tree stumps, learning rate 1.0, seed 42, and was fit on the 504,000 development rows. It was loaded without retraining for the single authorized evaluation of the saved 126,000-row stratified holdout; the threshold remained 0.5. The evaluator verified the saved membership and source hashes, and did not create splits, fit/search, or access `test.csv`. The one-time claim is recorded at `reports/generated/adaboost/20261114T223109215876Z/holdout_evaluation.json`. That claim has been consumed: **do not rerun the command below or attempt another holdout evaluation.**

The completed output is local at `reports/generated/adaboost/20261114T223109215876Z/holdout-result/summary.json` and the unchanged evaluated pipeline is `artifacts/evaluated_pipeline.joblib`. Final holdout results:

| Holdout metric | Result |
|---|---:|
| ROC-AUC | 0.9534865885689886 |
| Average precision (AP) | 0.9464313815418444 |
| Accuracy | 0.8867777777777778 |
| Precision | 0.8834940875156667 |
| Recall | 0.8610044077817706 |
| F1 | 0.872104281718425 |

At threshold 0.5 (`probability_presence >= 0.5`), the confusion counts were **TN 63,095; FP 6,414; FN 7,852; TP 48,639** (126,000 rows total). ROC-AUC and AP use positive-class probabilities; AP is average precision, not trapezoidal PR-AUC. These are final holdout metrics for the frozen development-trained artifact, not development CV estimates. The evaluator did not retrain the model. The original training metadata predates a pipeline checksum; evaluation metadata records the current artifact hash and source-run metadata hash.

### Authorized second evaluation: same holdout, 11 predictors

After explicit team authorization, the same 126,000 labeled holdout was reused once to report the frozen 65-stump AdaBoost model (`learning_rate=1.1`, seed 42) with 11 predictors (BP and FBS over 120 excluded). This is a protocol amendment and a **second look at the same holdout**, not independent validation, a replacement for the first result, or a new untouched test. The earlier 50-stump/13-predictor evaluation above remains preserved and must be disclosed alongside it. Both runs used threshold 0.5. The second run loaded the already development-fitted pipeline; it did not fit or tune, and did not read `test.csv`. No further tuning or holdout evaluation is permitted based on either result.

The authorization record is `reports/generated/adaboost_search/20261116T105636099417Z/selected/second_holdout_authorization.json`; the exclusive claim is beside it as `holdout_evaluation.json`. Output is in `selected/authorized-second-holdout/`, and the separate copied pipeline is `artifacts/authorized_second_holdout_65_1p1/evaluated_pipeline.joblib`. Authorization binds the candidate config, metadata, and pipeline hashes, source data and membership hashes, and the original claim and summary hashes.

| Holdout metric | Result |
|---|---:|
| ROC-AUC | 0.9538633348884222 |
| Average precision (AP) | 0.9468909238037428 |
| Accuracy | 0.8868492063492064 |
| Precision | 0.8849673679221205 |
| Recall | 0.8593227239737303 |
| F1 | 0.87195653150119 |

At threshold 0.5, confusion counts were **TN 63,199; FP 6,310; FN 7,947; TP 48,544** (126,000 total). Do not interpret differences between these reused-holdout scores as independent evidence of generalization or use them for model selection.

## Random Forest: original-feature baseline and optional search

The baseline uses a seeded 100-tree `RandomForestClassifier` (`max_depth=12`, `min_samples_leaf=50`, `max_features="sqrt"`) on the original 13 predictors. Numeric median imputation and categorical mode imputation/one-hot encoding are fitted within each fold; there is no scaling or feature engineering. `n_jobs` is derived from positive configured `threads` when omitted, and any supplied value must match (default 1); unbounded `-1` is rejected.

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model random_forest --config configs/default.json
```

The optional four-candidate search is explicitly opt-in:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.random_forest_experiment --config configs/default.json --plan configs/random_forest_search.json
```

It crosses depths 8/12 with leaf sizes 50/200 (100 trees fixed), selects mean development-fold ROC-AUC, and breaks exact ties toward shallower depth then larger leaves. CV evaluates four candidates (12 forest fits); selected baseline output repeats three folds and fits once on development data (4 more fits), for 16 forest fits total. Forests can be expensive and artifacts larger than the other models; the search is not run automatically. Outputs are development-only; no holdout is evaluated and test.csv is not read. The fixed-default baseline has been run; see the comparison below. The optional search has not been run.

Baseline outputs default to `reports/generated/random_forest/<run-id>/`; search outputs default to `reports/generated/random-forest/<run-id>/`, with the selected run under `selected/`. Explicit output directories must be new. Search metadata records plan/config hashes and elapsed time; fold result rows retain warning notes.

## Development CV comparison

All results below are unweighted means across the same three saved development folds, using threshold 0.5 for thresholded metrics. Accuracy, precision, and recall are percentages; other metrics are on the 0–1 scale. Bold marks the best displayed value in each column. The tree row is the winner of the bounded eight-candidate search; AdaBoost and Random Forest are fixed-default baselines, not tuned results.

| Model | ROC-AUC | Average precision | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.500000 | 0.448339 | 55.1661% | 0.0000% | 0.0000% | 0.000000 |
| Logistic regression | 0.952470 | 0.945306 | 88.4966% | 88.1183% | 85.9287% | 0.870097 |
| Decision tree | 0.949041 | 0.939705 | 88.0242% | 86.9890% | 86.1783% | 0.865816 |
| AdaBoost | **0.952684** | **0.945670** | **88.5393%** | **88.1322%** | 86.0216% | **0.870639** |
| Random Forest | 0.951362 | 0.943743 | 88.3732% | 87.5988% | **86.2818%** | 0.869353 |

### Fold-to-fold sample standard deviation

SDs are on the 0–1 scale for all metrics (including accuracy, precision, and recall); they describe variation among these three folds and are not confidence intervals.

| Model | ROC-AUC | Average precision | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Logistic regression | 0.000383 | 0.000274 | 0.000906 | 0.000947 | 0.001402 | 0.001061 |
| Decision tree | 0.000352 | 0.000316 | 0.001077 | 0.000678 | 0.002479 | 0.001368 |
| AdaBoost | 0.000368 | 0.000224 | 0.000510 | 0.002059 | 0.001628 | 0.000411 |
| Random Forest | 0.000338 | 0.000076 | 0.000871 | 0.001138 | 0.000826 | 0.000956 |

The dummy's tiny nonzero AP SD rounds to 0.000000. These are development-CV comparisons, not independent final performance estimates; the single final AdaBoost holdout result is reported separately above. Unlabeled `test.csv` was not used. Small score differences do not establish a statistically meaningful winner, and the held-out result must not be used for further model/threshold selection. Source summaries: [`dummy_reference.json`](../reports/dummy_reference.json), [`logistic_reference.json`](../reports/logistic_reference.json), [`decision_tree_reference.json`](../reports/decision_tree_reference.json), and the local generated summaries for [AdaBoost](../reports/generated/adaboost/20261114T223109215876Z/summary.json) and [Random Forest](../reports/generated/random_forest/20261115T130554382868Z/summary.json). Generated outputs are local/ignored and may not be present in a fresh clone.

## Results showcase

A small dependency-free static page presents the development-CV comparison and the separately reported one-time AdaBoost holdout results. Its self-contained `web/results.json` is a display snapshot of the project reference summaries; update it only when the source results are intentionally revised. It does not load data, train a model, or make predictions.

The static-server command below serves only the results showcase. To use the single-record form, stop the static server and start the prediction app in the next section; it provides the required local prediction API.

From the repository root, serve it with Docker:

```sh
docker run --rm -p 8000:8000 -v "${PWD}/web:/site:ro" -w /site heart-ml python -m http.server 8000 --bind 0.0.0.0
```

Open <http://localhost:8000>. The display notes that CV and holdout are distinct and that the results do not establish clinical validity. No CSVs, predictions, or model artifacts are served.

## Status and next steps

Confirm data provenance/independence assumptions and report limitations. The holdout was evaluated twice under the specific authorized amendment above; do not evaluate it again or tune from either result. The local single-record demo loads a frozen pipeline without retraining. Batch test-file prediction, a full-project reproduction command, and clean-clone release verification remain future work.

### Exploratory BP, FBS, and cholesterol ablations

These were explicitly requested **post-finalization exploratory development checks**, after the one-time holdout evaluation. That is a deviation from the plan's usual prohibition on post-holdout feature tuning; the comparisons use the same saved development folds and 50-stump AdaBoost settings (`learning_rate=1.0`, seed 42). They are exploratory comparisons on folds already used during model development, not independent validation or new final performance estimates; no holdout or unlabeled test data was used. Removing BP alone did not change the per-fold metrics. Removing FBS over 120 as well also produced exactly zero per-fold differences for all six metrics (ROC-AUC, average precision, accuracy, precision, recall, and F1), relative to BP-only removal:

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| ROC-AUC | 0.9526840414038346 | 0.0003679353543932212 |
| Average precision | 0.945670135906247 | 0.00022357033482875378 |
| Accuracy | 0.8853928571428571 | 0.0005103450768898003 |
| Precision | 0.8813220979580695 | 0.0020592696820395146 |
| Recall | 0.8602160530706354 | 0.001628001853902608 |
| F1 | 0.8706386945569816 | 0.0004111175063692303 |

Removing cholesterol in addition to BP gave fold ROC-AUCs **0.952208118275172**, **0.952250625053061**, and **0.952798578012118** (mean **0.952419107113450**, sample SD **0.000329317975130**). Relative to the BP-removed version, the paired folds each decreased (by approximately 0.00020, 0.00029, and 0.00030; mean decrease **0.000264934290384**). This small observed difference is not evidence of statistical significance. Cholesterol is retained. BP and FBS over 120 are excluded from the current 11-input config and local form; raw CSV schemas remain unchanged.

The original 13-input holdout numbers above remain unchanged and belong only to that artifact. The subsequent authorized 11-input evaluation is recorded separately above and reuses the same holdout. The form can load current 11-input, prior 12-input, or original 13-input schemas. For either legacy schema, it supplies missing values for excluded features only after verifying from fitted transformed feature names and every fitted tree that neither BP nor any FBS one-hot column is used; otherwise it fails closed. This compatibility behavior does not alter the artifact.

## Single-record prediction demo

The local web form uses the saved `artifacts/evaluated_pipeline.joblib` AdaBoost pipeline and never fits or changes it. It accepts 11 inputs (excluding BP and FBS over 120), reports the model's Presence probability and frozen 0.5-threshold class, and does not access either CSV. Blank compatibility placeholders use only the artifact's already-fitted imputers. Categorical choices use descriptive labels and the backend maps them to the model's stored codes. Feature explanations and documented units follow the [UCI Heart Disease documentation](https://archive.ics.uci.edu/dataset/45/heart+disease), because the local schema appears related; local file provenance and code meanings have not been independently verified. Where that documentation does not state a unit, the form leaves it unspecified. The model artifact is ignored by Git, so this demo runs only where the authorized evaluated artifact is already available; do not rerun the one-time holdout evaluation to recreate it.

Build the image if needed, then run from the repository root:

```sh
docker build -t heart-ml .
docker run --rm -p 127.0.0.1:8000:8000 -v "${PWD}:/app" heart-ml python -m heart_ml.predict_web
```

Open <http://localhost:8000> and choose “Single-record demo”. The server binds inside the container, while Docker publishes the port only to the local machine. Form submissions are not written to disk or included in server logs. This is a student-project classification of a supplied dataset label, not a diagnosis, medical advice, or future-risk estimate. Do not use it for health decisions. Batch prediction for `test.csv` remains a separate, unimplemented workflow.

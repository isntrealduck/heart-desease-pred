# Heart-disease label classification

Coursework comparing four classifiers for the supplied `Heart Disease` label:
logistic regression, a decision tree, AdaBoost, and Random Forest. This is a
classification experiment, not a clinical tool. Data provenance is unconfirmed;
the task does not establish diagnosis, future risk, or patient-level validity.

## Data

Place the labeled file at `data/raw/train.csv`. It must contain, in order,
`id`, `Age`, `Sex`, `Chest pain type`, `BP`, `Cholesterol`, `FBS over 120`,
`EKG results`, `Max HR`, `Exercise angina`, `ST depression`, `Slope of ST`,
`Number of vessels fluro`, `Thallium`, and `Heart Disease`. Labels are exactly
`Absence` and `Presence`. The current model uses 11 predictors: it excludes
`id`, `BP`, and `FBS over 120`; labels are mapped explicitly to 0 and 1.
Preserve supplied CSV originals. Data appear consistent with Kaggle Playground
Series S6E2, but source and redistribution terms are unverified.

`test.csv` is not needed or used. It is unlabeled and this project does not
produce batch predictions or local metrics from it.

## Protocol and models

Use a stratified 20% holdout (seed 42) and shuffled stratified 3-fold CV
(seed 42) on the remaining development data. The holdout is excluded from all
experiments below. Candidate selection uses mean fold ROC-AUC; other recorded
metrics include average precision and thresholded metrics at 0.5. Imputation,
scaling (logistic regression), and categorical one-hot encoding are inside
scikit-learn pipelines fitted separately within each training fold. Split and
fold membership are stored by ID. The prior holdout has already been used for
two authorized evaluations and is consumed: do not evaluate it again or use it
to select models/settings. These workflows are development-only.

The compact searches are configured in version control:

| Experiment | Candidates |
| --- | --- |
| Logistic regression | Fixed baseline: `C=1.0`, `max_iter=1000` |
| Decision tree | Gini; `max_depth` 3, 8, 12 crossed with `min_samples_leaf` 50, 200 (6) |
| AdaBoost stump count | `n_estimators` 25 through 75 by 5; `learning_rate=1.0` (11) |
| AdaBoost learning rate | 65 estimators; learning rates 0.8, 0.9, 1.0, 1.1, 1.2 (5) |
| Random Forest | `max_depth` 8, 12 crossed with `min_samples_leaf` 50, 200 (4); 100 trees, square-root feature sampling, one worker |

AdaBoost uses decision stumps. Selection ties prefer fewer estimators then
lower learning rate; tree ties prefer shallower depth then larger leaf size.
The two AdaBoost grids are separate experiments. Each candidate runs CV and a
full-development refit; selected CV and refit are repeated after selection.

## Docker workflow

Docker is required for setup and training. From the repository root:

```sh
docker build -t heart-ml .
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.audit --config configs/default.json --output data/audit.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.split --config configs/default.json
```

Development-only experiments:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model logistic
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.tree_experiment --plan configs/decision_tree_search.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.adaboost_experiment --plan configs/adaboost_stump_count_search.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.adaboost_experiment --plan configs/adaboost_learning_rate_65_search.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.random_forest_experiment --plan configs/random_forest_search.json
```

For a fixed-configuration development CV run, `baseline --model` accepts only
`logistic`, `decision_tree`, `adaboost`, or `random_forest`. All commands use
`configs/default.json` unless overridden. Run a baseline or search only when
the labeled training file is available; none requires `test.csv`.

Runs and generated split membership are written locally under ignored
`reports/` and `data/splits/` paths. Experiment output includes configuration,
fold metrics/predictions, metadata, and a fitted development pipeline.
Generated outputs and data are not submission files and should not be
committed.

## Limitations

Development CV scores are selection estimates, not independent final
performance; fold standard deviation is not a confidence interval. The
previously used holdout is consumed and is not part of reproducible workflows
in this submission. Data source, patient identity, row independence, and
clinical interpretation remain unverified. No clinical claims are made.

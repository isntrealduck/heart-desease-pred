# Heart-disease label classification

This project compares simple machine-learning models for predicting the supplied
`Heart Disease` label (`Absence` or `Presence`). It is a coursework experiment,
not a clinical tool. The provenance of the local data is unconfirmed, so results
do not establish clinical validity, patient-level generalization, diagnosis, or
future risk.

## Data and protocol

Place the supplied files at `data/raw/train.csv` and `data/raw/test.csv`.
Preserve the originals. The files appear consistent with Kaggle Playground
Series S6E2, but that identification and its access/redistribution terms have
not been verified. `test.csv` is unlabeled; no local metrics are computed from
it, and batch prediction for it is not implemented.

The saved split reserves 20% of labeled training rows (126,000) as a
stratified holdout (seed 42); shuffled stratified three-fold CV (seed 42) uses
the other 504,000 development rows. `id` is excluded from predictors, and the
target is explicitly mapped `Absence=0`, `Presence=1`. Preprocessing is part
of each scikit-learn pipeline and is fitted within training folds. Models are
compared by ROC-AUC; thresholded metrics use 0.5. Split and fold membership
are persisted by ID. The current `configs/default.json` uses 11 predictors,
excluding BP and FBS over 120. The historical CV comparison below used the
original 13 predictors; do not conflate those feature schemas.

## Recorded results

Historical development-CV snapshot: unweighted means across the same three
development folds. Thresholded metrics use 0.5. AdaBoost and Random Forest are
fixed-default runs; the tree is the winner of a bounded search. These
comparisons are not independent estimates, and small differences do not
establish a meaningful winner.

| Model (13-feature historical comparison) | ROC-AUC | AP | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.500000 | 0.448339 | 55.1661% | 0.0000% | 0.0000% | 0.000000 |
| Logistic regression | 0.952470 | 0.945306 | 88.4966% | 88.1183% | 85.9287% | 0.870097 |
| Decision tree | 0.949041 | 0.939705 | 88.0242% | 86.9890% | 86.1783% | 0.865816 |
| AdaBoost | **0.952684** | **0.945670** | **88.5393%** | **88.1322%** | 86.0216% | **0.870639** |
| Random Forest | 0.951362 | 0.943743 | 88.3732% | 87.5988% | **86.2818%** | 0.869353 |

Two AdaBoost evaluations reused the **same 126,000-row holdout** under a
specifically authorized protocol amendment. They are not independent of one
another; neither demonstrates superiority or provides an untouched final
estimate. The holdout is consumed: no further holdout evaluation, tuning, or
selection is permitted.

| Frozen artifact | Predictors / configuration | ROC-AUC | AP | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original evaluation | 13; 50 stumps, learning rate 1.0 | 0.953487 | 0.946431 | 88.6778% | 88.3494% | 86.1004% | 0.872104 |
| Authorized second look | 11 (BP/FBS excluded); 65 stumps, learning rate 1.1 | 0.953863 | 0.946891 | 88.6849% | 88.4967% | 85.9323% | 0.871957 |

Both used threshold 0.5 and loaded already-fitted development pipelines; there
was no holdout fitting. Score differences are not evidence of independent
generalization or established superiority. Detailed precision, confusion
counts, provenance, authorization record, experiment methods, and historical
commands are in [docs/experiments.md](docs/experiments.md). The specifically
authorized second-evaluation files are:

- Authorization and exclusive claim:
  `reports/generated/adaboost_search/20261116T105636099417Z/selected/second_holdout_authorization.json`
  and sibling `holdout_evaluation.json`.
- Output: `reports/generated/adaboost_search/20261116T105636099417Z/selected/authorized-second-holdout/`.
- Copied pipeline: `artifacts/authorized_second_holdout_65_1p1/evaluated_pipeline.joblib`.
- Original evaluation and artifact:
  `reports/generated/adaboost/20261114T223109215876Z/holdout_evaluation.json`,
  `reports/generated/adaboost/20261114T223109215876Z/holdout-result/summary.json`,
  `artifacts/evaluated_pipeline.joblib`.

These outputs/artifacts are local and git-ignored. `PROJECT_PLAN.md` records
the authorization amendment and its safeguards. Do not rerun either evaluator.
The static `web/results.json` is intentionally a historical display snapshot
of the original-model CV and original holdout; it is not the second-look
result and must not be silently relabeled.

## Docker: build, tests, audit, and split

Docker is required for project workflows. Run at the repository root:

```sh
docker build -t heart-ml .
docker run --rm -v "${PWD}:/app" heart-ml python -m pytest -q
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.audit --config configs/default.json
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.split --config configs/default.json
```

Data files, generated split files, and fitted artifacts are not committed.
See [docs/experiments.md](docs/experiments.md) for the fuller command reference,
experiment details, and output interpretation.

## Results page and local prediction demo

The static results page is informational only:

```sh
docker run --rm -p 8000:8000 -v "${PWD}/web:/site:ro" -w /site heart-ml \
  python -m http.server 8000 --bind 0.0.0.0
```

Visit <http://localhost:8000>. For the single-record form, stop that server and
run the local frozen-pipeline demo instead:

```sh
docker run --rm -p 127.0.0.1:8000:8000 -v "${PWD}:/app" heart-ml \
  python -m heart_ml.predict_web
```

It loads `artifacts/evaluated_pipeline.joblib` (the original 50-stump,
13-feature evaluation artifact), without fitting or changing it. The form
accepts the current 11 inputs and supports legacy schemas only after checks
that excluded features are unused. The 65-stump alternative can be selected
with `HEART_ML_PIPELINE=artifacts/authorized_second_holdout_65_1p1/evaluated_pipeline.joblib`
when that local artifact is available; the environment override changes no
artifact and does not authorize another evaluation. Do not use this demo for
health decisions. Batch `test.csv` prediction remains unimplemented.

## Reports and limitations

- [Dummy baseline](reports/dummy_baseline.md)
- [Logistic comparison](reports/logistic_baseline.md)
- [Decision-tree report](reports/decision_tree.md)
- [Detailed experiment and holdout record](docs/experiments.md)

Development CV is not independent final evaluation; fold variability is not a
confidence interval. The two holdout results reuse one partition. Data
provenance, clinical interpretation, patient identity, and row independence
remain unverified. The results support no clinical claim.

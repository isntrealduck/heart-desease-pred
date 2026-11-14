# Decision tree: development cross-validation

Reference experiment: 14 November 2026 (Asia/Qyzylorda). These are development selection scores, not final test scores.

## Method and leakage controls

Used all 504,000 development rows and the existing three stratified folds (336,000 training / 168,000 validation rows per fold). All 126,000 final holdout rows were excluded; unlabeled test.csv was not opened. Original 13 predictors, excluding id; Absence=0, Presence=1, seed 42, one numeric-library thread.

Numeric median imputation and categorical mode imputation/one-hot encoding were fitted inside each training fold. No scaling or engineered terms were used.

The tree captures nonlinear thresholds and interactions using inspectable rules. Gini impurity is G = 1 - sum_k(p_k squared); greedy splits reduce sample-weighted child impurity. Leaf class proportions supply probabilities. Depth and leaf-size limits control complexity. See [scikit-learn's tree guide](https://scikit-learn.org/1.7/modules/tree.html).

The saved plan fixed max_depth in {3,5,8,12} crossed with min_samples_leaf in {50,200}: eight Gini candidates. Selection maximizes mean CV ROC-AUC; exact ties prefer shallower depth, then larger minimum leaf size. No threshold tuning or class weighting.

## All candidate results

| Maximum depth | Minimum leaf size | Mean ROC-AUC | Fold SD | Mean accuracy | Mean F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 50 | 0.905119 | 0.000572 | 0.841677 | 0.811922 |
| 3 | 200 | 0.905119 | 0.000572 | 0.841677 | 0.811922 |
| 5 | 50 | 0.932259 | 0.000122 | 0.862704 | 0.843658 |
| 5 | 200 | 0.932259 | 0.000122 | 0.862704 | 0.843658 |
| 8 | 50 | 0.946349 | 0.000146 | 0.877599 | 0.862251 |
| 8 | 200 | 0.946272 | 0.000190 | 0.876925 | 0.861867 |
| 12 | 50 | 0.948366 | 0.000089 | 0.881187 | 0.866649 |
| 12 | 200 | 0.949041 | 0.000352 | 0.880242 | 0.865816 |

Selected max_depth=12, min_samples_leaf=200, criterion=gini. The development-wide fitted tree has depth 12 and 1137 leaves.

## Selected tree versus logistic regression

| Metric | Tree mean | Tree fold SD | Logistic mean | Tree minus logistic |
| --- | ---: | ---: | ---: | ---: |
| roc_auc | 0.949041 | 0.000352 | 0.952470 | -0.003430 |
| average_precision | 0.939705 | 0.000316 | 0.945306 | -0.005601 |
| accuracy | 0.880242 | 0.001077 | 0.884966 | -0.004724 |
| precision | 0.869890 | 0.000678 | 0.881183 | -0.011293 |
| recall | 0.861783 | 0.002479 | 0.859287 | +0.002496 |
| f1 | 0.865816 | 0.001368 | 0.870097 | -0.004281 |

Unweighted fold means and sample SD (ddof=1); SD is not a confidence interval. ROC-AUC and AP use probabilities; other metrics use probability_presence >= 0.5, including exact ties. The tree estimator's predict method can handle ties differently. Undefined divisions use zero_division=0.

ROC-AUC evaluates ranking across thresholds. Average precision (AP) summarizes precision over recall increments and complements ROC-AUC for the positive class; it differs from trapezoidal PR-AUC. Accuracy measures total correctness, precision measures correct positive predictions, recall measures detected positives, and F1 is the harmonic mean of precision and recall. Together they show false-positive/false-negative tradeoffs without establishing clinical usefulness.

## Confusion counts

| Fold | True negatives | False positives | False negatives | True positives |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 82,984 | 9,695 | 10,626 | 64,695 |
| 1 | 82,906 | 9,773 | 10,293 | 65,028 |
| 2 | 83,021 | 9,658 | 10,313 | 65,008 |
| Total OOF | 248,911 | 29,126 | 31,232 | 194,731 |

Each development row is predicted once by a model trained without that row. These totals pool confusion counts only; reported metrics remain fold means.

## Interpretation and verification

The selected tree trails logistic ROC-AUC by 0.003430, and has lower accuracy and F1. Recall is slightly higher. Keep the weaker overall result. This is the best of eight tested trees by ROC-AUC; the depth-12/leaf-50 candidate has slightly higher accuracy and F1 but was not selected by the frozen primary metric.

The tree was tuned on these folds; logistic used fixed settings. This descriptive comparison does not establish statistical significance. Development scores are subject to selection optimism. Wait for AdaBoost and freeze the final pipeline/threshold before evaluating the untouched holdout. Unconfirmed provenance, independence and possible synthetic data limit external conclusions.

All 24 search fold fits and the selected repeat/full-development fits completed without recorded warnings. The internal selected-fold rerun matched. Another fresh-container run produced identical metric summaries and byte-identical predictions. All 39 synthetic tests passed. Saved OOF probabilities independently reproduce each fold metric and confusion count within 1e-14. This demonstrates repeatability in the tested Docker environment; clean-clone and cross-platform reproduction remain future work.

## Reproduce

Follow README Docker setup and data placement, then run:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.tree_experiment --config configs/default.json --plan configs/decision_tree_search.json
```

Repeat selected settings directly:

```sh
docker run --rm -v "${PWD}:/app" heart-ml python -m heart_ml.baseline --model decision_tree --config reports/decision_tree_selected_config.json
```

The full pipeline is locally at reports/generated/decision-tree-reference/selected/decision_tree_pipeline.joblib. Large predictions/model artifacts remain ignored by Git. Full-precision reference, selected config, all-candidate/fold tables, metadata, selection and verification JSON accompany this report. Source fingerprints identify the code used; the recorded base Git commit predates these uncommitted additions.

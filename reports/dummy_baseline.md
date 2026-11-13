# Dummy classifier: measured development baseline

Run date: 14 November 2026 (Asia/Qyzylorda). This report is generated from the saved experiment outputs. No final holdout or unlabeled-test performance is reported.

## Protocol

The prior-probability dummy classifier ignores all predictors. It is fitted separately on the training rows of each saved development fold, then evaluated on that fold's validation rows. The three folds use seed 42. Presence is positive class 1, Absence is class 0, and labels are predicted at probability >= 0.5. Undefined precision/recall/F1 are assigned zero.

Development rows: 504000. Final holdout rows excluded: 126000. Neither the holdout nor test.csv was evaluated.

## Results

| Metric | Mean across development folds | Fold SD |
| --- | ---: | ---: |
| roc_auc | 0.500000 | 0.000000 |
| average_precision | 0.448339 | 0.000000 |
| accuracy | 0.551661 | 0.000000 |
| precision | 0.000000 | 0.000000 |
| recall | 0.000000 | 0.000000 |
| f1 | 0.000000 | 0.000000 |

Means are unweighted across the three folds. SD is the sample SD (ddof=1), not a confidence interval. Average precision is not trapezoidal PR-AUC. The saved JSON retains full-precision results.

The model predicts Absence for every development validation row at threshold 0.5. Its majority-class accuracy therefore coexists with zero positive-class recall. This is the intended no-feature benchmark, not evidence of a useful disease detector. ROC-AUC 0.5 reflects constant probabilities within each validation fold.

| Fold | Training rows | Validation rows | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 336000 | 168000 | 92679 | 0 | 75321 | 0 |
| 1 | 336000 | 168000 | 92679 | 0 | 75321 | 0 |
| 2 | 336000 | 168000 | 92679 | 0 | 75321 | 0 |

## Verification and environment

The Docker image built successfully. All 19 synthetic tests passed in 6.09 seconds, including six dummy-baseline tests. A second run in a new container reused the existing split and produced identical summary values and a byte-identical development-prediction CSV.

Prediction CSV SHA-256: d7487a9c3ade5eb28fbccecd65b64c3e7e469d4a31ef65c5c16581f20ea845b9

Reference run ID (UTC): 20261113T195611913611Z
Reproduction run ID (UTC): 20261113T195726636803Z
Reference baseline command runtime: 13.33 seconds, excluding image build and container startup.

Tested with Docker Engine 29.9.3, Python 3.12.16, scikit-learn 1.7.2, Linux x86_64 under WSL2 on Windows, and one numeric-library thread.
Image ID: sha256:341c895017d60dd128b24112dda7ce6b8980dc4163abd3425cce59ea0ed676c3

This checks repeatability within the tested image/environment. It is not yet a fresh-clone verification of a published project release or a cross-platform result.

## Files and traceability

- [Full-precision summary](dummy_reference.json)
- [Run metadata, data/split/configuration hashes, package versions, and source hashes](dummy_run_metadata.json)
- Detailed reference outputs: reports/generated/dummy-reference/
- Repeat-run outputs: reports/generated/dummy-reproduction/
- Saved development-trained pipeline: reports/generated/dummy-reference/dummy_pipeline.joblib

The recorded Git commit is the setup base revision. This reference was measured before the baseline implementation was committed; module SHA-256 hashes identify the actual executed source. Git working-tree cleanliness is unknown inside the slim image because Git is not installed there.

The saved pipeline is fitted on development rows only and has no final holdout score. Use the same saved folds for the subsequent logistic-regression and boosting comparisons.

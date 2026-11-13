# Logistic regression: original-feature baseline

Measured on 14 November 2026 (Asia/Qyzylorda). All scores below are development cross-validation results. The final holdout and supplied unlabeled test file were not evaluated.

## Method

This experiment uses the original 13 predictors, excluding ID and target. It adds no nonlinear terms, interactions, or other derived predictors. Numeric features use median imputation and standard scaling; categorical codes use mode imputation and one-hot encoding with unknown categories ignored. Every fold fits the complete preprocessing/classifier pipeline using only its training rows.

Logistic regression supplies a learned probabilistic baseline: log-odds are linear in the transformed inputs. L2 regularization controls coefficient size. Fixed settings were C=1.0, solver=lbfgs, max_iter=1000, tolerance=0.0001, seed=42, and one numeric-library thread. No hyperparameter search or threshold optimization was performed.

The run reused the dummy baseline's saved ID-based split and three development folds: 504000 development rows; 126000 holdout rows excluded. Positive class: Presence=1. Threshold: probability >= 0.5. Undefined threshold-based metrics use zero_division=0.

## Comparison on identical development folds

| Metric | Dummy mean | Logistic mean | Logistic fold SD |
| --- | ---: | ---: | ---: |
| roc_auc | 0.500000 | 0.952470 | 0.000383 |
| average_precision | 0.448339 | 0.945306 | 0.000274 |
| accuracy | 0.551661 | 0.884966 | 0.000906 |
| precision | 0.000000 | 0.881183 | 0.000947 |
| recall | 0.000000 | 0.859287 | 0.001402 |
| f1 | 0.000000 | 0.870097 | 0.001061 |

Values are proportions, not percentages. Means are unweighted across folds; SD is the sample SD (ddof=1), not a confidence interval. ROC-AUC and average precision use positive-class probabilities. Average precision differs from trapezoidal PR-AUC. No pooled out-of-fold ROC-AUC is used for comparison.

Logistic regression improved over the no-feature benchmark on these development folds. This establishes the original-feature learned baseline for later feature-engineering comparisons. It does not establish external clinical validity, statistical significance of future small gains, or final holdout performance.

## Per-fold checks

| Fold | Training rows | Validation rows | TN | FP | FN | TP | Solver iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 336000 | 168000 | 83933 | 8746 | 10717 | 64604 | 15 |
| 1 | 336000 | 168000 | 83892 | 8787 | 10565 | 64756 | 15 |
| 2 | 336000 | 168000 | 84031 | 8648 | 10514 | 64807 | 15 |

All three fold fits and the development-wide fit converged without recorded warnings. The development-wide fit used 15 iterations. The complete pipeline is saved as reports/generated/logistic-reference/logistic_pipeline.joblib; it has no final holdout score.

## Reproducibility

The updated Docker image built successfully and all 24 synthetic tests passed in 10.56 seconds, including checks that logistic preprocessing sees only its training fold and that holdout changes cannot alter development scores.
A repeat run in a fresh container reused the saved split and produced identical summary values and a byte-identical development-prediction CSV. This verifies repeatability within the tested image/environment; clean-clone release and cross-platform verification remain future work.

Reference run ID (UTC): 20261113T211731398899Z
Repeat run ID (UTC): 20261113T211839925963Z
Development-prediction SHA-256: 7ac4d6342c4f43e0f29f5a374f8bb54d508fd28eefe4c2e8682e05b7e19f96cb
Baseline command runtime: 20.27 seconds, excluding image build and container startup.

Tested with Docker Engine 29.9.3, Python 3.12.16, scikit-learn 1.7.2, and Linux x86_64 under WSL2 on Windows.
Image ID: sha256:c4b85270e6a9cad9da8a075367b5012eb4b7581178cf613a0f6de2234425052b

## Reference files

- [Full-precision logistic summary](logistic_reference.json)
- [Logistic run metadata and source/data/split/configuration hashes](logistic_run_metadata.json)
- [Dummy reference summary](dummy_reference.json)
- Detailed logistic outputs: reports/generated/logistic-reference/
- Repeat-run outputs: reports/generated/logistic-reproduction/

The saved Git commit is the setup base revision. This reference was measured before the baseline implementation was committed; module hashes identify the executed source. Git working-tree status is unavailable inside the slim image.

Reproduce using the logistic command in the README. Use a new output directory for each run. The original-feature configuration and results should remain available when feature-engineering experiments are introduced.

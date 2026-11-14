# Heart disease classification: two-person project plan

This document records the project plan. See README.md for implemented commands, current status, and measured results. The shared repository is https://github.com/isntrealduck/heart-desease-pred.

## 1. Agree on the question and definition of done - together

**Question:** Can we predict the supplied `Heart Disease` label (`Presence` or `Absence`) from the 13 provided features?

This is binary classification of a supplied disease-status label. It does not establish future disease risk, a diagnosis, or performance on real patients. Several features describe diagnostic examinations, so explain when these inputs would be available. Confirm the professor's rubric, allowed algorithms, dataset source, submission format, and deadline before expanding scope.

The minimum successful project contains:

- A validated data description and documented source/access instructions.
- A fixed evaluation protocol that prevents leakage.
- A dummy baseline, logistic regression, and one stronger tabular model.
- A reproducible comparison, one selected pipeline, and an untouched final evaluation.
- A command that rebuilds the analysis, model, metrics, figures, and predictions.
- A README that your teammate successfully follows in a clean environment.
- A report that justifies methods/metrics and matches saved experiment outputs.
- A tagged GitHub release with both contributors credited.

Complete this experiment before adding an app, API, neural network, or large ensemble.


## 3. Confirm the data and preserve it - you lead, friend checks

| Property | train.csv | test.csv |
| --- | ---: | ---: |
| Rows, excluding header | 630,000 | 270,000 |
| Columns | 15 | 14 |
| Predictor columns, excluding ID and target | 13 | 13 |
| Empty cells | 0 | 0 |
| Repeated feature rows beyond first occurrence | 0 | 0 |
| Contains Heart Disease | Yes | No |

Training labels: `Presence` = 282,454 (44.8%); `Absence` = 347,546 (55.2%). IDs are unique across the two files. No exact feature vectors were shared between train and test. Duplicate checks used CSV feature values as written; they do not establish patient independence or exclude near-duplicates. Python validation should also check parsed missing values and unexpected sentinel codes.

These filenames, schema, and sizes appear consistent with Kaggle's **Predicting Heart Disease, Playground Series S6E2**. This is an inference, not verified provenance of your local copies. Confirm where you obtained them. If confirmed, document the synthetic nature and access/redistribution terms; do not describe the rows as a verified clinical cohort.

Store untouched files in `data/raw/`. Keep them out of Git initially. Record the exact download page, dataset version/date, access requirements, and SHA-256 checksums. The supplied copies' checksums are in `DATA_MANIFEST.json`. If redistribution is permitted and desired, decide explicitly whether to publish the files; otherwise give precise download instructions. A path on your own computer is insufficient.

Build an audit that verifies schema, row counts, labels, unique IDs, parsed missing values, feature ranges, category values, and duplicates. Fail clearly for incorrect files.

| Role | Columns | Treatment |
| --- | --- | --- |
| Identifier | `id` | Exclude from inputs; retain for splits and predictions |
| Target | `Heart Disease` | Map Absence to 0 and Presence to 1 explicitly |
| Numeric/count | `Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`, `Number of vessels fluro` | Numeric; the last is a small count |
| Categorical/binary | `Sex`, `Chest pain type`, `FBS over 120`, `EKG results`, `Exercise angina`, `Slope of ST`, `Thallium` | Treat codes as categories unless the source supports an ordered interpretation |

Verify code meanings and units from the source. Keep original names, including `Number of vessels fluro`, or use one documented rename map throughout.

## 4. Freeze the evaluation protocol - together, before experiments

1. Reserve **20% of train.csv** as a stratified final holdout using `random_state=42`: 126,000 rows. The remaining 504,000 rows form the development pool.
2. Save split membership by `id` and its checksum. Investigate repeated patients, groups, or time ordering first; use a group/time-aware protocol instead if row independence is not justified.
3. Use **3-fold stratified cross-validation**, shuffled with seed 42, within the development pool. Save fold membership too.
4. Choose the model by mean validation **ROC-AUC**, retaining fold scores and SD. Define any simpler-model tie-breaking rule before selection.
5. Use threshold **0.5** for the primary precision/recall/F1/confusion-matrix report. If optimizing a threshold, choose it using development predictions only, document the objective, and freeze it before opening the holdout.
6. Evaluate the selected configuration on the holdout once. Do not tune features, parameters, preprocessing choices, or thresholds after seeing its results.

**The supplied test.csv is unlabeled. It cannot provide local accuracy, ROC-AUC, or recall.** Use it for final predictions only. A competition leaderboard, if available, is separate from local holdout evaluation.

Learn imputation, scaling, encodings, feature selection, calibration, and any resampling within the development procedure. Use a scikit-learn `Pipeline` and `ColumnTransformer` so preprocessing is fitted within each training fold. Never fit transformations on all data before splitting/CV. With the observed class balance, do not add oversampling by default.

## 5. Set up the environment and shared interfaces - friend leads, you review

Use Python 3.12 in DOCKER as a proposed starting version, a project virtual environment, and CPU training. Verify dependency compatibility. Use pandas, NumPy, scikit-learn, matplotlib, joblib, and pytest; add Jupyter/seaborn only if needed.

Pin the exact tested Python patch version and all resolved dependency versions. Commit a lock file from a clean, successfully tested project environment. Avoid unbounded package versions and do not invent a version list before installing/testing it.

Before working independently, agree on:

- Target mapping, feature roles/types, split/fold files, metric, and threshold.
- A common `build_pipeline(model_name, config)` returning an **unfitted** pipeline.
- An experiment result format: model, parameters, fold IDs, fold scores, mean/SD, runtime, and failure/warning notes.
- CLI names and artifact paths below.

Store seed, split fractions, fold count, features, model settings, threshold, and thread settings in `configs/default.json`. Pass fixed integer seeds to splitters, stochastic estimators, and randomized searches. Record Python/library versions, OS, CPU, thread count, data/split/configuration hashes, and Git commit for every final run. Set thread limits before numeric libraries load if required.

Seeds alone do not guarantee byte-identical outputs across hardware/OS. Define numerical comparison tolerances and demonstrate the scope actually tested. Start with absolute tolerance `1e-6` for probabilities/metrics on the reference environment; investigate discrepancies rather than silently relaxing it.

## 6. Explore development data and build baselines - you lead

After fixing the split:

- Plot development label balance, numeric distributions, category frequencies, and relationships with the label.
- Explain missingness/sentinels, plausible ranges, and possible leakage. Do not remove unusual values without justification.
- Train `DummyClassifier(strategy="prior")` as a no-feature benchmark.
- Train logistic regression with numeric imputation/scaling and categorical imputation/one-hot encoding inside the pipeline.
- Evaluate both with the agreed folds and result format.

The dummy model tests whether features add predictive value. Logistic regression is an interpretable binary-classification baseline; its log-odds are linear in the transformed inputs unless nonlinear terms/interactions are supplied.

Keep exploration in a notebook and reusable code in the package. Restart and run all notebook cells before submission. Training must work without opening the notebook.

## 7. Compare course-covered tree models - split between teammates

The team chose a standalone decision tree (you lead) and AdaBoost (friend leads), alongside the existing dummy and logistic baselines, so both can explain the algorithms covered in the course. HistGradientBoosting remains an optional existing constructor rather than a required experiment.

The decision-tree study uses the original features, Gini splits, seed 42, and eight configurations: maximum depth in {3, 5, 8, 12} crossed with minimum leaf size in {50, 200}. Select the highest mean development CV ROC-AUC; exact ties prefer shallower depth and then larger minimum leaf size. Keep threshold 0.5 and all saved folds fixed. Save every candidate's scores and the complete selected development-trained pipeline.

Your friend should evaluate a small recorded AdaBoost search with the same data, folds, label mapping, threshold and metrics. The best standalone tree need not be the weak learner used inside AdaBoost; small trees are appropriate starting weak learners. Agree on the candidate grid before examining scores. Logistic settings remain the existing fixed reference unless a separate tuning experiment is explicitly recorded.

A fixed development-only sample is useful for debugging; label its results provisional. Avoid broad searches and further model families until the comparison works. Keep weak results, warnings and failed runs. Because the same folds choose hyperparameters, development scores are selection estimates; reserve the untouched holdout for the final frozen model.

## 8. Evaluate and explain the selected pipeline - together

Freeze the configuration based on development CV, fit a fresh pipeline on the development pool, and evaluate the untouched holdout.

Save:

- ROC-AUC from positive-class scores/probabilities.
- Average precision (AP) using `average_precision_score`. AP summarizes precision over recall increments; it is not identical to trapezoidal PR-AUC.
- Accuracy, positive-class precision, recall, F1, and confusion matrix at the frozen threshold.
- ROC and precision-recall curves; calibration plot/Brier score if discussing probability quality.
- Development comparison with both baselines and a review of false positives/negatives.
- Permutation importance or carefully interpreted coefficients. Associations are not causal evidence.

Label development CV and holdout results separately. Fold SD is not a confidence interval. Document methods/assumptions if adding uncertainty estimates.

Save the **entire fitted preprocessing-plus-classifier pipeline** as `artifacts/evaluated_pipeline.joblib`, with schema, threshold, and run metadata.

Generate 270,000 predictions for test.csv in original order. Use `classes_` to identify the positive-class probability column. Output `id`, `probability_presence`, and `predicted_label`. Check unique IDs, row count, non-missing predictions, and probabilities in [0, 1]. Inspect the actual sample submission before producing a competition-specific format.

An optional refit on all 630,000 labeled rows must produce a **separate** artifact. The existing holdout score belongs to the evaluated pipeline; the refitted model has no new untouched local test score.

## 9. Make the workflow runnable and document it - friend leads, both contribute

Proposed structure; most files still need implementation:

```text
heart-disease-prediction/
  README.md
  PROJECT_PLAN.md
  REPORT_TEMPLATE.md
  DATA_MANIFEST.json
  .gitignore
  pyproject.toml
  requirements.lock
  configs/default.json
  data/raw/                      # downloaded input; ignored by Git
  data/splits/                   # holdout/fold membership
  notebooks/01_eda.ipynb
  src/heart_ml/
    __init__.py
    audit.py
    split.py
    preprocess.py
    train.py
    evaluate.py
    predict.py
    reproduce.py
  tests/
  reports/                       # metrics, tables, figures, report
  artifacts/                     # fitted pipelines; ignored by Git
  predictions/                   # generated CSVs; ignored by Git
```

Implement the planned command:

```text
python -m heart_ml.reproduce --config configs/default.json
```

It should verify input hashes/schema, create/verify fixed splits/folds, rerun the recorded candidate comparison, select/refit the development pipeline, evaluate the holdout, save outputs/metadata, and predict test.csv. Create output folders automatically and require no manual notebook execution. Preserve reference outputs using separate run directories or a reference snapshot.

Add a separate prediction command that loads the saved pipeline without retraining. Test split non-overlap, data validation, consistent inference transformations, and prediction shape/range. Use small synthetic fixtures in CI; expensive searches and downloaded data need not run on every PR.

The README must include actual prerequisites, data access/hashes, exact setup/train/predict commands, outputs, measured runtime/hardware, results, threshold, tolerance, limitations, and contributors. Replace draft markers before submission.

## 10. Reproduce from a clean clone and release - swap roles

1. Freeze a candidate commit and dependency lock.
2. Each teammate uses a clean clone and fresh environment.
3. Follow only the README and obtain data through its documented instructions.
4. Verify exact data/split hashes, selected settings, output counts, and numerical agreement with reference probabilities/metrics within the declared tolerance.
5. Record hardware, OS, environment, elapsed time, and discrepancies.
6. Fix missing dependencies, absolute paths, missing files, and unclear instructions; repeat affected checks.
7. Tag the verified commit `v1.0.0`, retain small reference outputs in Git, and provide repository/tag plus data/artifact access instructions.

State which platforms were tested. Use the required Docker workflow and locked dependencies for setup, experiments, prediction, and tests.

## Recommended division of work

| Deliverable | Lead | Reviewer | Completion condition |
| --- | --- | --- | --- |
| Repository, issues, access | You | Friend | Both clone, branch, and open PRs |
| Provenance, audit, data dictionary | You | Friend | Verified manifest and audit report |
| Split protocol and metric choice | Together | Together | Committed split/fold specification |
| Environment, package, config | Friend | You | Fresh-environment install works |
| Preprocessing and dummy/logistic models | You | Friend | Comparable pipeline CV results |
| Boosting and bounded tuning | Friend | You | CV results and settings retained |
| Final selection and evaluation | Together | Together | Frozen pipeline and honest evaluation |
| Orchestration, save/load, prediction CLI | Friend | You | One command regenerates outputs |
| README data/method/results | You | Friend | Claims match experiment |
| README setup, tests, CI | Friend | You | Commands tested from clean clone |
| Report data/method rationale and limitations | You | Friend | Justified claims trace to evidence |
| Report evaluation/results and figures | Friend | You | Tables/captions match saved outputs |
| Reproduction and presentation | Together | Together | Both explain and reproduce the project |

Suggested seven-session schedule:

| Session | You | Friend | Shared checkpoint |
| --- | --- | --- | --- |
| 1 | Source/audit and repository | Environment/package | Freeze schema, interfaces, split, metrics |
| 2 | Development EDA/preprocessing | Split persistence/runner | Same development rows for both |
| 3 | Dummy/logistic models | Boosting model | First end-to-end run |
| 4 | Baseline tuning/interpretation | Bounded boosting search | Freeze winning settings |
| 5 | Report narrative/plots | Holdout/output integration | Save final evaluation/artifacts |
| 6 | Clean-clone reproduction | Clean-clone documentation check | Fix reproducibility failures |
| 7 | Final report/README | CI/release verification | Review evidence and submit together |

Use short session check-ins and prompt PR reviews. Keep the experiment table in version control.

## Professor's report requirements

Use [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) for the academic report. The report explains and evaluates the experiment; the README provides reproduction instructions.

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Clear, concise, technically accurate language | Define target, positive class, data roles, and metrics; distinguish observations from explanations. | Reviewed report with consistent terminology |
| Explain methods and metrics | Justify the dummy/logistic/boosting comparison and ROC-AUC plus complementary metrics. | Methods/evaluation sections and settings |
| Separate test data from selection/tuning | Tune within development; freeze model, transformations, and threshold before holdout evaluation. | Split/fold files, experiment log, frozen config |
| Report weak or negative results | Include baseline/candidate results, unsuccessful tuning, warnings/failures, and simpler-model wins. | Full search log and concise comparison |
| Keep numbers, figures, tables, conclusions consistent with code | Generate report assets from one identified final run, round only for display, and verify report/README against saved outputs. | Run ID, commit, hashes, metrics, predictions, figures |

Describe actual methods if implementation departs from this plan. Use probabilities for ROC-AUC/AP and thresholded labels for precision/recall/F1. Do not use post-evaluation feature explanations to justify undocumented retuning.

Before submission, one teammate verifies numbers/captions against outputs; the other checks method descriptions against code. Confusion-matrix counts must sum to evaluated sample count and reproduce the threshold-based metrics. Failed runs must remain marked failed. If the holdout was reused for tuning, disclose that limitation; it can no longer be described as untouched.

## References

- [GitHub repository creation](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [GitHub collaborator invitations](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)
- [scikit-learn leakage and random-state practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [Model persistence and environment](https://scikit-learn.org/stable/model_persistence.html)
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Candidate data source, pending confirmation](https://www.kaggle.com/competitions/playground-series-s6e2/data)
# Heart disease classification project

Initial setup milestone only: validated environment, inputs, immutable ID-based split, and shared unfitted model pipelines. **No training, model selection, holdout evaluation, or test predictions have been implemented.** Do not interpret setup outputs as model results.

## Data and limitations

Place the supplied datasets at `data/raw/train.csv` and `data/raw/test.csv`; preserve their original contents. The local files appear consistent with Kaggle Playground Series S6E2, but provenance, access terms, units, category meanings, patient/group identity, and row independence are unconfirmed. `id` is an identifier, not a verified patient identifier. No clinical conclusions are supported. Audit output records hashes, dimensions, label counts, numeric ranges, categories, and ID uniqueness; it does not establish data quality beyond these checks.

`configs/default.json` fixes the 20% stratified holdout (seed 42), shuffled three-fold CV on development rows, ROC-AUC selection metric, threshold 0.5, features, and CPU model defaults. `id` is excluded from model inputs; labels map explicitly Absence=0, Presence=1. Numeric median imputation/scaling (for logistic regression) and categorical mode imputation/one-hot encoding are composed in the pipeline and thus can be fitted within folds. HistGradientBoosting uses dense one-hot inputs. Model constructors return unfitted pipelines. Holdout rows have fold `-1` and never enter development folds.

Audit allows missing predictors (reported per column for later imputation), but rejects missing IDs/labels and nonnumeric or nonfinite values. It reports observed ranges/categories and duplicate feature-row count without imposing clinical ranges or dropping rows. Coded categorical values are validated as finite numeric codes; no permitted value ranges are asserted.

## Docker setup and tested commands

Docker is required. Tested with Docker 29.9.3, Python 3.12.16, linux/arm64. Lock versions are explicit; rebuild and run the checks:

```sh
docker build -t heart-ml .
docker run --rm -v "$PWD:/app" heart-ml python -m pytest
docker run --rm -v "$PWD:/app" heart-ml python -m heart_ml.audit --config configs/default.json
docker run --rm -v "$PWD:/app" heart-ml python -m heart_ml.split --config configs/default.json
```

The image installs the package editable. Bind-mounting the project ensures source edits and generated outputs are immediately visible and persistent; the repository's `src` shadows the image copy. `DATA_MANIFEST.json` and `data/splits/` are generated, git-ignored outputs; raw CSVs and these generated files are excluded from Docker build context as well. The audit manifest includes SHA-256 hashes.

Split metadata records the training CSV hash and a protocol hash of ID/target/label mapping/seed/holdout fraction/fold count, plus a membership-file hash. Model parameters, threshold, and thread settings do not change split identity. Split creation rejects incomplete files, protocol/source mismatches, and membership corruption rather than replacing them. If resetting a setup-only split, first back up both split files and explicitly remove the pair before recreating; do not reset after experiments rely on it. Inputs must match expected schemas exactly.

## Status and next steps

Setup Docker, audit and split commands are available; no training/evaluation/prediction/reproduction command exists yet. Next, confirm data provenance and independence assumptions, review the persisted split together, then add fold-based model comparison, evaluation, artifact persistence, and test prediction within the agreed protocol. No runtime or performance results are claimed here.

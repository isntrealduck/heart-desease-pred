"""Explore feature-to-feature associations on development rows only."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .audit import CATEGORICAL, ID, NUMERIC, validate_csv
from .split import write_or_verify_splits


def cramers_v(left, right):
    """Bias-corrected Cramer's V for two categorical variables."""
    table = pd.crosstab(left, right).to_numpy(dtype=float)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return np.nan
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = np.sum((table - expected) ** 2 / expected)
    phi2 = chi2 / n
    rows, columns = table.shape
    correction = (columns - 1) * (rows - 1) / (n - 1)
    phi2_corrected = max(0, phi2 - correction)
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    cols_corrected = columns - (columns - 1) ** 2 / (n - 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else np.nan


def correlation_ratio(categories, values):
    """Eta: association of a categorical feature with a numeric feature."""
    categories = pd.Series(categories)
    values = pd.to_numeric(values, errors="coerce")
    valid = categories.notna() & values.notna()
    categories, values = categories[valid], values[valid]
    if len(values) == 0 or values.nunique() < 2:
        return np.nan
    overall = values.mean()
    between = sum(
        len(group) * (group.mean() - overall) ** 2
        for _, group in values.groupby(categories)
    )
    total = ((values - overall) ** 2).sum()
    return float(np.sqrt(between / total)) if total else 0.0


def draw_heatmap(matrix, title, path, scale, cmap, annotation_color="black"):
    figure = Figure(figsize=(max(8, len(matrix.columns) * 0.9), max(6, len(matrix) * 0.75)), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=scale[0], vmax=scale[1], cmap=cmap)
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for col in range(len(matrix.columns)):
            value = matrix.iat[row, col]
            if pd.notna(value):
                axis.text(
                    col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=8, color=annotation_color,
                )
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=160)


def analyze(config, output):
    membership_path, _ = write_or_verify_splits(config)
    frame = validate_csv(config["data"]["train"], labeled=True)
    membership = pd.read_csv(membership_path)
    development_ids = membership.loc[membership["partition"] == "development", ID]
    data = frame.loc[frame[ID].isin(development_ids)].copy()
    if len(data) != len(development_ids):
        raise ValueError("Could not match all development IDs to training rows")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    # Numeric correlation captures linear (Pearson) and monotonic (Spearman)
    # relationships. Coded categories are deliberately excluded here.
    numeric = data[NUMERIC].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    categorical = data[CATEGORICAL].astype("string")
    categorical_codes = data[CATEGORICAL].apply(pd.to_numeric, errors="coerce")
    categorical_spearman = categorical_codes.corr(method="spearman")
    cat_assoc = pd.DataFrame(index=CATEGORICAL, columns=CATEGORICAL, dtype=float)
    for left in CATEGORICAL:
        for right in CATEGORICAL:
            cat_assoc.loc[left, right] = 1.0 if left == right else cramers_v(categorical[left], categorical[right])
    mixed = pd.DataFrame(index=CATEGORICAL, columns=NUMERIC, dtype=float)
    for category in CATEGORICAL:
        for measure in NUMERIC:
            mixed.loc[category, measure] = correlation_ratio(categorical[category], numeric[measure])

    matrices = {
        "numeric_pearson": pearson,
        "numeric_spearman": spearman,
        "categorical_code_spearman": categorical_spearman,
        "categorical_cramers_v": cat_assoc,
        "categorical_numeric_eta": mixed,
    }
    for name, matrix in matrices.items():
        matrix.to_csv(output / f"{name}.csv")
    draw_heatmap(pearson, "Numeric features: Pearson correlation", output / "numeric_pearson.png", (-1, 1), "coolwarm")
    draw_heatmap(spearman, "Numeric features: Spearman correlation", output / "numeric_spearman.png", (-1, 1), "coolwarm")
    draw_heatmap(categorical_spearman, "Categorical codes: Spearman correlation (exploratory)", output / "categorical_code_spearman.png", (-1, 1), "coolwarm")
    draw_heatmap(cat_assoc, "Categorical features: Cramer's V", output / "categorical_cramers_v.png", (0, 1), "viridis", annotation_color="white")
    draw_heatmap(mixed, "Categorical–numeric association: eta", output / "categorical_numeric_eta.png", (0, 1), "viridis")
    record = {
        "rows": len(data),
        "scope": "Development partition only; final holdout and unlabeled test.csv were not read.",
        "feature_groups": {"numeric": NUMERIC, "categorical_codes_treated_as_categories": CATEGORICAL},
        "interpretation": {
            "pearson": "Linear association, signed; not evidence of causation.",
            "spearman": "Monotonic rank association, signed; not evidence of causation.",
            "categorical_code_spearman": "Exploratory rank correlation of stored category codes; interpret only where codes have a meaningful order.",
            "cramers_v": "Categorical association strength from 0 to 1; not signed.",
            "eta": "Categorical/numeric association strength from 0 to 1; not signed.",
        },
        "files": list(matrices),
    }
    (output / "analysis.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default="reports/feature-associations")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    record = analyze(config, args.output_dir)
    print(f"Analyzed {record['rows']} development rows; saved heatmaps and tables to {args.output_dir}")


if __name__ == "__main__":
    main()

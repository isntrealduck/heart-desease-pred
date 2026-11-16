"""Render a readable shallow view of the selected development-trained tree."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
from sklearn.tree import plot_tree

from .audit import TARGET, validate_csv
from .baseline import development_rows
from .preprocess import build_pipeline
from .split import encode_labels, write_or_verify_splits


def render_tree(config, output, display_depth=3):
    """Fit only on saved development rows and render the tree's top levels."""
    if not isinstance(display_depth, int) or display_depth < 1:
        raise ValueError("display_depth must be a positive integer")
    membership_path, _ = write_or_verify_splits(config)
    frame = validate_csv(config["data"]["train"], labeled=True)
    development = development_rows(frame, pd.read_csv(membership_path), config)
    X = development[config["numeric_features"] + config["categorical_features"]]
    y = encode_labels(development[TARGET], config)
    pipeline = build_pipeline("decision_tree", config)
    pipeline.fit(X, y)

    prep = pipeline.named_steps["preprocess"]
    feature_names = prep.get_feature_names_out()
    tree = pipeline.named_steps["model"]
    figure = Figure(figsize=(30, 16), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=[config["negative_label"], config["positive_label"]],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        max_depth=display_depth,
        fontsize=8,
        ax=axis,
    )
    axis.set_title(
        f"Selected decision tree — first {display_depth} levels\n"
        f"Fit on {len(development):,} development rows only; "
        f"max_depth={tree.max_depth}, leaves={tree.get_n_leaves():,}"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    return {
        "output": str(output),
        "development_rows": len(development),
        "display_depth": display_depth,
        "fitted_tree_depth": int(tree.get_depth()),
        "leaves": int(tree.get_n_leaves()),
        "holdout_evaluated": False,
        "test_csv_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="reports/decision_tree_selected_config.json")
    parser.add_argument("--output", default="reports/decision_tree.png")
    parser.add_argument("--display-depth", type=int, default=3)
    args = parser.parse_args()
    result = render_tree(
        json.loads(Path(args.config).read_text(encoding="utf-8")),
        args.output,
        args.display_depth,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Plot every fitted depth-1 learner from a saved AdaBoost pipeline."""

import argparse
from pathlib import Path

from joblib import load
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.tree import plot_tree


def render_stumps(pipeline_path, output):
    """Render saved estimators and their AdaBoost weights without fitting."""
    pipeline = load(pipeline_path)
    model = pipeline.named_steps["model"]
    estimators = getattr(model, "estimators_", None)
    weights = getattr(model, "estimator_weights_", None)
    if estimators is None or weights is None or len(estimators) == 0:
        raise ValueError("Saved pipeline does not contain fitted AdaBoost estimators")

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    count = len(estimators)
    columns = 5
    rows = int(np.ceil(count / columns))
    figure = Figure(figsize=(columns * 4.2, rows * 3.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.axis("off")
            continue
        estimator = estimators[index]
        plot_tree(
            estimator,
            feature_names=names,
            class_names=["Absence", "Presence"],
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            precision=2,
            fontsize=6,
            ax=axis,
        )
        axis.set_title(
            f"Stump {index + 1} | weight={weights[index]:.4f}",
            fontsize=9,
        )
    figure.suptitle(
        f"Fitted AdaBoost decision stumps ({count} learners)\n"
        "Loaded from the saved pipeline; no training or evaluation performed",
        fontsize=15,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    return {
        "output": str(output),
        "fitted_estimators": count,
        "pipeline": str(pipeline_path),
        "trained_or_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        default="artifacts/evaluated_pipeline.joblib",
    )
    parser.add_argument("--output", default="reports/adaboost_stumps.png")
    args = parser.parse_args()
    import json

    print(json.dumps(render_stumps(args.pipeline, args.output), indent=2))


if __name__ == "__main__":
    main()

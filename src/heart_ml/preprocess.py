"""Leakage-safe preprocessing and classifier construction."""

from numbers import Integral

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .audit import validate_config
from .features import FEATURE_SETS, NumericFeatureEngineer


def _random_forest_params(config):
    """Resolve a bounded CPU worker count from the shared thread setting."""
    threads = config.get("threads", 1)
    if not isinstance(threads, Integral) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    params = dict(config["models"].get("random_forest", {}))
    requested = params.get("n_jobs", threads)
    if not isinstance(requested, Integral) or isinstance(requested, bool) or requested != threads:
        raise ValueError("random_forest n_jobs must be a positive integer matching threads")
    params["n_jobs"] = threads
    return params


def build_pipeline(model_name, config, feature_set="original"):
    """Return an unfitted pipeline; derived terms are logistic-only experiments."""
    validate_config(config)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set != "original" and model_name != "logistic":
        raise ValueError("Derived features are supported only for logistic regression")
    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    params = (_random_forest_params(config) if model_name == "random_forest"
              else config["models"][model_name])
    if model_name == "dummy":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    # Preserve the original path exactly; engineered variants keep all six
    # numeric columns even if one is entirely missing in a training fold.
    numeric_steps = [("impute", SimpleImputer(
        strategy="median", keep_empty_features=feature_set != "original"
    ))]
    if feature_set != "original":
        numeric_steps.append(("engineer", NumericFeatureEngineer(feature_set)))
    if model_name == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", cat_pipe, categorical),
    ])
    if model_name == "logistic":
        model = LogisticRegression(**params, random_state=config["seed"])
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(**params, random_state=config["seed"])
    elif model_name == "adaboost":
        model = AdaBoostClassifier(**params, random_state=config["seed"])
    elif model_name == "random_forest":
        model = RandomForestClassifier(**params, random_state=config["seed"])
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**params, random_state=config["seed"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", model)])

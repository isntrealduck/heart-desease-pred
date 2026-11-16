"""Leakage-safe preprocessing and classifier construction."""

from numbers import Integral

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .audit import validate_config


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


def build_pipeline(model_name, config):
    """Return an unfitted, fold-safe pipeline for one supported classifier."""
    validate_config(config)
    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    params = (_random_forest_params(config) if model_name == "random_forest"
              else config["models"][model_name])
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", model)])

"""Leakage-safe preprocessing and baseline model construction."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .audit import validate_config


def build_pipeline(model_name, config):
    """Return an unfitted preprocessing/model pipeline from project config."""
    validate_config(config)
    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    params = config["models"][model_name]
    if model_name == "dummy":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    if model_name == "logistic":
        numeric_pipe.steps.append(("scale", StandardScaler()))
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", cat_pipe, categorical)])
    if model_name == "logistic":
        model = LogisticRegression(**params, random_state=config["seed"])
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**params, random_state=config["seed"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", model)])

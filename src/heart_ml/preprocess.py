"""Leakage-safe preprocessing and classifier construction."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .audit import validate_config
from .features import FEATURE_SETS, NumericFeatureEngineer


def build_pipeline(model_name, config, feature_set="original"):
    """Return an unfitted pipeline; derived terms are logistic-only experiments."""
    validate_config(config)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set != "original" and model_name != "logistic":
        raise ValueError("Derived features are supported only for logistic regression")
    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    params = config["models"][model_name]
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
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**params, random_state=config["seed"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", model)])

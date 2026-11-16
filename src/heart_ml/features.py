"""Small numeric feature hypotheses applied after fold-fitted imputation."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .audit import NUMERIC

FEATURE_SETS = ("original", "age_squared", "age_st_interaction", "combined")
FEATURE_DEFINITIONS = {
    "original": [],
    "age_squared": ["Age_squared = Age ** 2"],
    "age_st_interaction": ["Age_x_ST_depression = Age * ST depression"],
    "combined": ["Age_squared = Age ** 2", "Age_x_ST_depression = Age * ST depression"],
}


class NumericFeatureEngineer(TransformerMixin, BaseEstimator):
    """Append a fixed square/interaction; learn no statistics from labels."""

    def __init__(self, feature_set="original", numeric_features=None):
        self.feature_set = feature_set
        self.numeric_features = numeric_features

    def _numeric_names(self):
        return list(NUMERIC if self.numeric_features is None else self.numeric_features)

    def fit(self, X, y=None):
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {self.feature_set}")
        array = check_array(X, dtype=float)
        if array.shape[1] != len(self._numeric_names()):
            raise ValueError("Numeric feature count does not match configured names")
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        array = check_array(X, dtype=float)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("Numeric feature count changed")
        extra = []
        names = self._numeric_names()
        age = array[:, names.index("Age")]
        st = array[:, names.index("ST depression")]
        if self.feature_set in ("age_squared", "combined"):
            extra.append(age ** 2)
        if self.feature_set in ("age_st_interaction", "combined"):
            extra.append(age * st)
        result = np.column_stack([array, *extra]) if extra else array.copy()
        if not np.isfinite(result).all():
            raise ValueError("Engineered numeric features must be finite")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        names = list(self._numeric_names() if input_features is None else input_features)
        if len(names) != self.n_features_in_:
            raise ValueError("Numeric feature names do not match the fitted input")
        if self.feature_set in ("age_squared", "combined"):
            names.append("Age_squared")
        if self.feature_set in ("age_st_interaction", "combined"):
            names.append("Age_x_ST_depression")
        return np.asarray(names, dtype=object)

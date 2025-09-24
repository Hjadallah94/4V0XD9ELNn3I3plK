"""
features.py
Feature preprocessing pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def build_preprocessor(feature_names):
    """
    Build preprocessing: impute missing values + scale features.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, list(feature_names))],
        remainder="drop",
    )
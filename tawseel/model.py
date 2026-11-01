
"""Failure-risk baseline for Tawseel COD parcels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

TARGET = "failed"

# Only information available when the parcel is created.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]

HISTORY_COLUMNS = [
    "customer_prior_parcels",
    "customer_prior_fail_rate",
]

NUMERIC_COLUMNS = [
    "cod_amount",
    "n_items",
    "customer_prior_parcels",
    "customer_prior_fail_rate",
]

CATEGORICAL_COLUMNS = [
    "customer_id",
    "merchant_id",
    "area",
    "promised_slot",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time customer history without modifying df."""

    result = df.copy()

    work = df[
        ["customer_id", "created_at", "failed"]
    ].copy()

    work["_position"] = np.arange(len(work))

    prior_counts = np.zeros(len(work), dtype=int)
    prior_fail_rates = np.full(len(work), np.nan, dtype=float)

    # Process each customer independently.
    for _, customer_rows in work.groupby(
        "customer_id", sort=False, dropna=False
    ):

        previous_count = 0
        previous_failures = 0

        # All parcels with the same timestamp are processed together.
        for _, same_time_rows in customer_rows.groupby(
            "created_at", sort=True, dropna=False
        ):

            positions = same_time_rows["_position"].to_numpy()

            # Only strictly earlier parcels contribute.
            prior_counts[positions] = previous_count

            if previous_count > 0:
                prior_fail_rates[positions] = (
                    previous_failures / previous_count
                )

            # Update history AFTER assigning the current timestamp.
            previous_count += len(same_time_rows)

            failures = pd.to_numeric(
                same_time_rows["failed"], errors="coerce"
            ).fillna(0)

            previous_failures += int(failures.sum())

    result["customer_prior_parcels"] = prior_counts
    result["customer_prior_fail_rate"] = prior_fail_rates

    return result


def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically without separating equal timestamps."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = math.ceil(len(df) * test_fraction)

    sorted_df = df.sort_values(time_col, kind="mergesort")

    boundary = sorted_df.iloc[len(sorted_df) - n_test][time_col]

    train = df.loc[df[time_col] < boundary].copy()
    test = df.loc[df[time_col] >= boundary].copy()

    return train, test


def _extract_time_features(values):
    """Extract useful numeric features from parcel creation time."""

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    timestamps = pd.to_datetime(values, errors="coerce")

    return np.column_stack([
        timestamps.dt.hour.to_numpy(dtype=float),
        timestamps.dt.dayofweek.to_numpy(dtype=float),
        timestamps.dt.day.to_numpy(dtype=float),
        timestamps.dt.month.to_numpy(dtype=float),
    ])


def _clean_address_text(values):
    """Convert missing address values to empty strings."""

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    elif isinstance(values, np.ndarray) and values.ndim == 2:
        values = values[:, 0]

    return (
        pd.Series(values)
        .fillna("")
        .astype(str)
        .to_numpy()
    )


def build_pipeline(seed: int = 0) -> Pipeline:
    """Build an unfitted preprocessing and classification pipeline."""

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__missing__",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    time_pipeline = Pipeline([
        (
            "extract",
            FunctionTransformer(
                _extract_time_features,
                validate=False,
            ),
        ),
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    address_pipeline = Pipeline([
        (
            "clean",
            FunctionTransformer(
                _clean_address_text,
                validate=False,
            ),
        ),
        (
            "vectorizer",
            HashingVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                n_features=1024,
                alternate_sign=False,
                norm="l2",
            ),
        ),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
            (
                "time",
                time_pipeline,
                ["created_at"],
            ),
            (
                "address",
                address_pipeline,
                ["address_text"],
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=seed,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate a fitted model on an unseen test DataFrame."""

    y_true = test_df[TARGET]
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "base_rate": float(y_true.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Compare Logistic Regression against a prior-probability baseline."""

    X_train = train_df
    y_train = train_df[TARGET]

    model = build_pipeline(seed=seed)
    model.fit(X_train, y_train)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)

    model_results = evaluate(model, test_df)
    dummy_results = evaluate(dummy, test_df)

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": (
            model_results["pr_auc"] - dummy_results["pr_auc"]
        ),
    }

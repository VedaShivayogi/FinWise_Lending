"""
data_pipeline.py
=================
FinWise Lending - Credit Risk Project
Checkpoint 2 (Week 3): Data Ingestion + Preprocessing Pipeline

This module is the single source of truth for turning raw loan application
data into a model-ready feature matrix. It is used by:
  - the baseline model notebook (Checkpoint 2)
  - the full XGBoost/SHAP notebook (later checkpoint)
  - the FastAPI serving layer (production)

Design goals:
  - Deterministic: same input -> same output, every time.
  - No leakage: all fitted transforms (encoders, imputers, scalers) are fit
    on TRAIN data only and reused (never refit) on validation/test/serving data.
  - Auditable: every step logs what it did and how many rows/cols it touched.
  - Reusable: the same `FinWisePipeline` object that trained the baseline
    model is pickled and reused unchanged when training the champion model,
    so the two are compared on identical features.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("finwise_pipeline")

# ---------------------------------------------------------------------------
# 1. INGESTION
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "application_id", "age", "income_annual_inr", "employment_type",
    "employment_years", "existing_loans", "existing_emi_inr", "credit_score",
    "loan_amount_inr", "loan_purpose", "ltv_ratio", "dti_ratio", "default_flag",
]

VALID_RANGES = {
    "age": (18, 100),
    "income_annual_inr": (0, None),
    "employment_years": (0, None),
    "existing_loans": (0, None),
    "existing_emi_inr": (0, None),
    "credit_score": (300, 900),
    "loan_amount_inr": (0, None),
    "ltv_ratio": (0, 2.0),
    "dti_ratio": (0, None),
    "default_flag": (0, 1),
}


@dataclass
class IngestionReport:
    """Everything a reviewer needs to trust the ingested data."""
    source: str
    rows_read: int = 0
    rows_after_validation: int = 0
    missing_columns: list = field(default_factory=list)
    dropped_rows_by_reason: dict = field(default_factory=dict)
    duplicate_ids_removed: int = 0

    def to_dict(self):
        return self.__dict__


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV. Fails loudly rather than silently on a bad path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find data file at '{path}'. Place your CSV there or "
            f"pass the correct path to load_raw_data()."
        )
    df = pd.read_csv(p)
    logger.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {path}")
    return df


def validate_schema(df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
    """Check required columns exist. Missing columns are a hard failure -
    the pipeline should not silently proceed with an incomplete schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.missing_columns = missing
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean_and_validate_rows(df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
    """Row-level data quality checks. Every row we drop is logged with a reason
    so the report is auditable rather than a silent black box."""
    df = df.copy()
    initial_n = len(df)

    # 1. Drop exact duplicate application IDs (keep first occurrence)
    n_before = len(df)
    df = df.drop_duplicates(subset="application_id", keep="first")
    report.duplicate_ids_removed = n_before - len(df)

    # 2. Drop rows with nulls in required numeric/categorical fields
    n_before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    report.dropped_rows_by_reason["null_required_field"] = n_before - len(df)

    # 3. Range checks - drop rows with physically impossible values
    dropped_range = 0
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        mask = pd.Series(True, index=df.index)
        if lo is not None:
            mask &= df[col] >= lo
        if hi is not None:
            mask &= df[col] <= hi
        n_before = len(df)
        df = df[mask]
        dropped_range += n_before - len(df)
    report.dropped_rows_by_reason["out_of_range_value"] = dropped_range

    report.rows_read = initial_n
    report.rows_after_validation = len(df)
    logger.info(
        f"Validation complete: {initial_n:,} -> {len(df):,} rows "
        f"({initial_n - len(df):,} dropped, "
        f"{(initial_n - len(df)) / max(initial_n,1):.1%})"
    )
    return df.reset_index(drop=True)


def ingest(path: str) -> tuple[pd.DataFrame, IngestionReport]:
    """Full ingestion entrypoint: load -> validate schema -> clean rows."""
    report = IngestionReport(source=path)
    df = load_raw_data(path)
    df = validate_schema(df, report)
    df = clean_and_validate_rows(df, report)
    return df, report


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (stateless - safe to apply to any split)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features computed from raw columns only (no fitted state,
    so this is safe to apply identically to train/val/test/production)."""
    df = df.copy()

    df["emi_to_income_ratio"] = (
        df["existing_emi_inr"] / (df["income_annual_inr"] / 12 + 1)
    ).clip(0, 10)

    df["loan_to_income_ratio"] = df["loan_amount_inr"] / (df["income_annual_inr"] + 1)
    df["total_debt"] = df["existing_emi_inr"] * 12 + df["loan_amount_inr"]
    df["debt_to_income"] = df["total_debt"] / (df["income_annual_inr"] + 1)

    df["credit_score_band"] = pd.cut(
        df["credit_score"], bins=[0, 580, 670, 740, 800, 900],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
    )
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 40, 50, 60, 100],
        labels=["Young", "Early Career", "Mid Career", "Senior", "Elder"]
    )

    df["employment_stability"] = df["employment_years"] / (df["age"] - 17 + 1)

    df["high_dti"] = (df["dti_ratio"] > 0.5).astype(int)
    df["high_ltv"] = (df["ltv_ratio"] > 0.8).astype(int)
    df["low_credit"] = (df["credit_score"] < 600).astype(int)
    df["multiple_loans"] = (df["existing_loans"] >= 2).astype(int)
    df["risk_score"] = df["high_dti"] + df["high_ltv"] + df["low_credit"] + df["multiple_loans"]

    return df


# ---------------------------------------------------------------------------
# 3. FITTED PREPROCESSING (stateful - fit on train, reused elsewhere)
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["employment_type", "loan_purpose", "credit_score_band", "age_group"]
DROP_COLUMNS = ["application_id", "default_flag"]
TARGET_COLUMN = "default_flag"


class FinWisePreprocessor:
    """Stateful preprocessing: label encoders + scaler, fit ONCE on the
    training split and reused (transform-only) everywhere else. This is the
    object that gets pickled and shipped to serving so train/serve skew
    cannot happen."""

    def __init__(self, scale_numeric: bool = True):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.scale_numeric = scale_numeric
        self.feature_names_: Optional[list] = None
        self.is_fitted = False

    def fit(self, df_engineered: pd.DataFrame) -> "FinWisePreprocessor":
        df = df_engineered.copy()
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        X = self._encode(df)
        if self.scale_numeric:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        self.feature_names_ = X.columns.tolist()
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {len(df):,} rows -> {len(self.feature_names_)} features")
        return self

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        drop_cols = list(DROP_COLUMNS)
        for col in CATEGORICAL_FEATURES:
            le = self.label_encoders[col]
            # unseen categories at inference time map to a safe fallback (-1)
            # rather than crashing the pipeline
            df[col + "_encoded"] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
            drop_cols.append(col)
        drop_cols = [c for c in set(drop_cols) if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        return X

    def transform(self, df_engineered: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        X = self._encode(df_engineered)
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return X

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> "FinWisePreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 4. TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """Three-way stratified split: train / validation / test.
    Validation is held out from train specifically for baseline vs.
    champion-model comparison later; test stays untouched until final report."""
    y = df[TARGET_COLUMN]
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative_size, random_state=random_state,
        stratify=train_val[TARGET_COLUMN]
    )
    logger.info(
        f"Split: train={len(train):,} ({len(train)/len(df):.1%})  "
        f"val={len(val):,} ({len(val)/len(df):.1%})  "
        f"test={len(test):,} ({len(test)/len(df):.1%})"
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. END-TO-END CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str, artifacts_dir: str = "artifacts"):
    """Runs ingestion -> feature engineering -> split -> fit preprocessor ->
    transform all splits. Returns everything downstream code needs.
    This is the function Checkpoint 2's baseline notebook calls."""
    Path(artifacts_dir).mkdir(exist_ok=True, parents=True)

    df_raw, ingestion_report = ingest(csv_path)
    df_fe = engineer_features(df_raw)
    train_df, val_df, test_df = split_data(df_fe)

    preprocessor = FinWisePreprocessor().fit(train_df)

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
    y_val = val_df[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_df[TARGET_COLUMN].reset_index(drop=True)

    preprocessor.save(f"{artifacts_dir}/preprocessor.pkl")
    with open(f"{artifacts_dir}/ingestion_report.json", "w") as f:
        json.dump(ingestion_report.to_dict(), f, indent=2, default=str)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "preprocessor": preprocessor,
        "ingestion_report": ingestion_report,
        "raw_train_df": train_df, "raw_val_df": val_df, "raw_test_df": test_df,
    }


if __name__ == "__main__":
    result = run_pipeline("finwise_loan_applications.csv")
    print(f"Pipeline complete. Features: {result['X_train'].shape[1]}")
    print(f"Train/Val/Test sizes: {len(result['X_train'])}/{len(result['X_val'])}/{len(result['X_test'])}")

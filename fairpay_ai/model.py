"""
model.py
--------
Trains an Isolation Forest anomaly detection model on the synthetic FAIRPAY AI
pricing dataset, generates predictions and anomaly scores on the held-out test
set, and persists the fitted model to disk.
"""

import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "pricing_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
TEST_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "test_predictions.csv")

# ── Model hyper-parameters ─────────────────────────────────────────────────────
IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.08
IF_MAX_SAMPLES = "auto"
IF_RANDOM_STATE = 42

TEST_SIZE = 0.20
STRATIFY_COLUMN = "is_anomaly"

CATEGORICAL_FEATURES = ["service_type", "region"]
NUMERIC_FEATURES = ["tenure_months", "new_customer_price", "renewal_price", "price_ratio"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the pricing dataset from CSV and return as a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run generate_dataset.py first."
        )
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features and apply label encoding to categorical columns.

    Returns a copy of the DataFrame with:
    - price_ratio: renewal_price / new_customer_price
    - service_type and region columns replaced by their integer label encodings
    """
    df = df.copy()
    df["price_ratio"] = (df["renewal_price"] / df["new_customer_price"]).round(4)

    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = IF_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split on is_anomaly.

    Returns (X_train, X_test, y_train, y_test).
    """
    X = df[FEATURE_COLUMNS]
    y = df[STRATIFY_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    n_estimators: int = IF_N_ESTIMATORS,
    contamination: float = IF_CONTAMINATION,
    max_samples: str | int = IF_MAX_SAMPLES,
    random_state: int = IF_RANDOM_STATE,
) -> IsolationForest:
    """
    Fit an Isolation Forest on the training feature matrix.

    Returns the fitted estimator.
    """
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train)
    return clf


def predict(
    clf: IsolationForest,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate binary predictions and raw anomaly scores for the test set.

    Isolation Forest uses the convention: -1 → anomaly, +1 → normal.
    We convert to a boolean array where True = predicted anomaly.

    Returns (binary_predictions, anomaly_scores).
    """
    raw_labels = clf.predict(X_test)                    # -1 or +1
    binary_preds = raw_labels == -1                      # True = anomaly

    # decision_function returns the negative of the anomaly score;
    # lower (more negative) = more anomalous.  We negate to get a score
    # where higher = more anomalous, consistent with AUC-ROC convention.
    anomaly_scores = -clf.decision_function(X_test)

    return binary_preds, anomaly_scores


def save_model(clf: IsolationForest, path: str = MODEL_PATH) -> None:
    """Persist the fitted model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)
    print(f"Model saved to: {path}")


def save_test_predictions(
    df_raw: pd.DataFrame,
    test_indices: pd.Index,
    y_test: pd.Series,
    binary_preds: np.ndarray,
    anomaly_scores: np.ndarray,
    path: str = TEST_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """
    Combine original test-set rows with ground truth and predictions,
    then save to CSV.

    Returns the merged DataFrame for downstream evaluation.
    """
    results = df_raw.loc[test_indices].copy()
    results["y_true"] = y_test.values
    results["y_pred"] = binary_preds
    results["anomaly_score"] = anomaly_scores
    results.to_csv(path, index=False)
    print(f"Test predictions saved to: {path}")
    return results


def main() -> None:
    """
    End-to-end pipeline: load → engineer features → split →
    train Isolation Forest → predict → save model & predictions.
    """
    print("=" * 60)
    print("FAIRPAY AI — Isolation Forest Training Pipeline")
    print("=" * 60)

    # 1. Load & engineer features
    print("\n[1/5] Loading dataset...")
    df_raw = load_data()
    print(f"      Loaded {len(df_raw):,} records.")

    print("[2/5] Engineering features...")
    df = engineer_features(df_raw)

    # 2. Split
    print("[3/5] Splitting into train / test sets (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      Train: {len(X_train):,} records  "
          f"({y_train.sum():,} anomalies, {y_train.mean() * 100:.1f}%)")
    print(f"      Test : {len(X_test):,} records  "
          f"({y_test.sum():,} anomalies, {y_test.mean() * 100:.1f}%)")

    # 3. Train
    print("[4/5] Training Isolation Forest "
          f"(n_estimators={IF_N_ESTIMATORS}, contamination={IF_CONTAMINATION})...")
    clf = train_model(X_train)

    # 4. Predict
    print("[5/5] Generating predictions on test set...")
    binary_preds, anomaly_scores = predict(clf, X_test)
    n_flagged = binary_preds.sum()
    print(f"      Flagged {n_flagged:,} records as anomalies "
          f"({n_flagged / len(X_test) * 100:.1f}% of test set).")

    # 5. Persist
    save_model(clf)
    save_test_predictions(
        df_raw, X_test.index, y_test, binary_preds, anomaly_scores
    )

    print("\nTraining complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

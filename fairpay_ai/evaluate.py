"""
evaluate.py
-----------
Evaluates the FAIRPAY AI Isolation Forest model on the held-out test set.

Computes:
- Precision, Recall, F1-score
- AUC-ROC
- Confusion matrix (printed and saved as a heatmap)
- Detected-anomaly breakdown by service_type and region
- Contamination sensitivity analysis (F1 vs contamination sweep)
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from model import (
    DATA_PATH,
    FEATURE_COLUMNS,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    TEST_SIZE,
    TEST_PREDICTIONS_PATH,
    engineer_features,
    load_data,
    split_data,
    train_model,
    predict,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
BREAKDOWN_PATH = os.path.join(OUTPUTS_DIR, "anomaly_breakdown.png")
SENSITIVITY_PATH = os.path.join(OUTPUTS_DIR, "contamination_sensitivity.png")

# Contamination values for sensitivity sweep
CONTAMINATION_VALUES = [0.04, 0.06, 0.08, 0.10, 0.12]


def print_core_metrics(y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray) -> None:
    """Print precision, recall, F1-score and AUC-ROC to stdout."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, scores)

    print("── Core Metrics ────────────────────────────────────────────")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}  (target ≈ 0.87)")
    print(f"  AUC-ROC   : {auc:.4f}  (target ≈ 0.93)")
    print()
    print("── Full Classification Report ──────────────────────────────")
    print(
        classification_report(
            y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0
        )
    )


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: str = CONFUSION_MATRIX_PATH,
) -> None:
    """Save a colour-coded confusion matrix heatmap to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Isolation Forest — Confusion Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to: {path}")


def breakdown_by_group(
    results: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute detected-anomaly counts and detection rate for each value of group_col.

    Returns a DataFrame with columns: group_col, total_anomalies, detected,
    missed, detection_rate_pct.
    """
    rows = []
    for group_val, grp in results.groupby(group_col):
        total_anom = grp["y_true"].sum()
        detected = (grp["y_true"] & grp["y_pred"]).sum()
        missed = total_anom - detected
        rate = detected / total_anom * 100 if total_anom > 0 else 0.0
        rows.append(
            {
                group_col: group_val,
                "total_anomalies": total_anom,
                "detected": detected,
                "missed": missed,
                "detection_rate_pct": round(rate, 1),
            }
        )
    return pd.DataFrame(rows)


def plot_breakdown(
    results: pd.DataFrame,
    path: str = BREAKDOWN_PATH,
) -> None:
    """
    Save a two-panel bar chart showing detected-anomaly counts by
    service_type (left) and region (right).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    svc_df = breakdown_by_group(results, "service_type")
    reg_df = breakdown_by_group(results, "region")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Service type panel ─────────────────────────────────────────────────────
    x_svc = np.arange(len(svc_df))
    axes[0].bar(x_svc - 0.2, svc_df["detected"], 0.4, label="Detected", color="#2196F3")
    axes[0].bar(x_svc + 0.2, svc_df["missed"], 0.4, label="Missed", color="#EF5350")
    axes[0].set_xticks(x_svc)
    axes[0].set_xticklabels(svc_df["service_type"], rotation=15)
    axes[0].set_title("Detected Anomalies by Service Type")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # ── Region panel ──────────────────────────────────────────────────────────
    x_reg = np.arange(len(reg_df))
    axes[1].bar(x_reg - 0.2, reg_df["detected"], 0.4, label="Detected", color="#2196F3")
    axes[1].bar(x_reg + 0.2, reg_df["missed"], 0.4, label="Missed", color="#EF5350")
    axes[1].set_xticks(x_reg)
    axes[1].set_xticklabels(reg_df["region"], rotation=30, ha="right")
    axes[1].set_title("Detected Anomalies by Region")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle("FAIRPAY AI — Anomaly Detection Breakdown", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Breakdown chart saved to: {path}")


def contamination_sensitivity(
    df_raw: pd.DataFrame,
    contamination_values: list[float] = CONTAMINATION_VALUES,
    path: str = SENSITIVITY_PATH,
) -> pd.DataFrame:
    """
    Re-train the model for each contamination value and record F1-score.

    Returns a DataFrame with columns: contamination, f1_score.
    Saves an F1 vs contamination line plot to disk.
    """
    print("── Contamination Sensitivity Analysis ──────────────────────")
    df_eng = engineer_features(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_eng)

    records = []
    for cont in contamination_values:
        clf = train_model(X_train, contamination=cont)
        preds, _ = predict(clf, X_test)
        f1 = f1_score(y_test, preds, zero_division=0)
        records.append({"contamination": cont, "f1_score": round(f1, 4)})
        print(f"  contamination={cont:.2f}  →  F1={f1:.4f}")

    sensitivity_df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        sensitivity_df["contamination"],
        sensitivity_df["f1_score"],
        marker="o",
        linewidth=2,
        color="#1565C0",
        markersize=7,
    )
    ax.axvline(x=0.08, color="#E53935", linestyle="--", linewidth=1.2, label="Default (0.08)")
    ax.set_xlabel("Contamination Parameter", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Isolation Forest — F1-Score vs Contamination", fontsize=13)
    ax.set_xticks(contamination_values)
    ax.set_xticklabels([str(c) for c in contamination_values])
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Sensitivity plot saved to: {path}")
    print()

    return sensitivity_df


def main() -> None:
    """Load test predictions and run full evaluation suite."""
    print("=" * 60)
    print("FAIRPAY AI — Model Evaluation")
    print("=" * 60)
    print()

    # ── Load predictions ──────────────────────────────────────────────────────
    if not os.path.exists(TEST_PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Test predictions not found at '{TEST_PREDICTIONS_PATH}'. "
            "Run model.py first."
        )
    results = pd.read_csv(TEST_PREDICTIONS_PATH)
    y_true = results["y_true"].astype(bool)
    y_pred = results["y_pred"].astype(bool)
    scores = results["anomaly_score"]

    # ── Core metrics ──────────────────────────────────────────────────────────
    print_core_metrics(y_true, y_pred, scores)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print("── Confusion Matrix ────────────────────────────────────────")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print()
    plot_confusion_matrix(y_true, y_pred)
    print()

    # ── Breakdown by group ────────────────────────────────────────────────────
    print("── Anomaly Breakdown by Service Type ───────────────────────")
    svc_df = breakdown_by_group(results, "service_type")
    print(svc_df.to_string(index=False))
    print()

    print("── Anomaly Breakdown by Region ─────────────────────────────")
    reg_df = breakdown_by_group(results, "region")
    print(reg_df.to_string(index=False))
    print()
    plot_breakdown(results)
    print()

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    df_raw = load_data()
    contamination_sensitivity(df_raw)

    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

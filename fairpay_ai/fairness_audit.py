"""
fairness_audit.py
-----------------
Assesses whether the FAIRPAY AI Isolation Forest model disproportionately
flags consumers from particular regional groups as false positives.

Computes the False Positive Rate (FPR) per region and flags any region where
FPR deviates more than 5 percentage points from the overall FPR.

Outputs:
- Console fairness summary
- Bar chart of FPR by region saved to fairpay_ai/outputs/fairness_fpr_by_region.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import TEST_PREDICTIONS_PATH

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FPR_CHART_PATH = os.path.join(OUTPUTS_DIR, "fairness_fpr_by_region.png")

# Threshold: flag if FPR deviates by more than this many percentage points
FPR_DEVIATION_THRESHOLD_PP = 5.0


def compute_fpr(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute the False Positive Rate.

    FPR = FP / (FP + TN)

    Returns 0.0 if there are no true negatives (edge case).
    """
    fp = ((~y_true) & y_pred).sum()
    tn = ((~y_true) & (~y_pred)).sum()
    if (fp + tn) == 0:
        return 0.0
    return fp / (fp + tn)


def compute_fpr_by_region(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the FPR for each region in the test set.

    Returns a DataFrame with columns: region, fp, tn, fpr.
    """
    y_true = results["y_true"].astype(bool)
    y_pred = results["y_pred"].astype(bool)

    rows = []
    for region, grp in results.groupby("region"):
        gt = grp["y_true"].astype(bool)
        pred = grp["y_pred"].astype(bool)
        fp = ((~gt) & pred).sum()
        tn = ((~gt) & (~pred)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append(
            {
                "region": region,
                "n_normal": int(fp + tn),
                "fp": int(fp),
                "tn": int(tn),
                "fpr": round(fpr, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("fpr", ascending=False).reset_index(drop=True)


def flag_deviating_regions(
    fpr_df: pd.DataFrame,
    overall_fpr: float,
    threshold_pp: float = FPR_DEVIATION_THRESHOLD_PP,
) -> pd.DataFrame:
    """
    Add a 'flagged' column — True for regions whose FPR deviates more than
    threshold_pp percentage points from the overall FPR.
    """
    fpr_df = fpr_df.copy()
    fpr_df["deviation_pp"] = ((fpr_df["fpr"] - overall_fpr) * 100).round(2)
    fpr_df["flagged"] = fpr_df["deviation_pp"].abs() > threshold_pp
    return fpr_df


def plot_fpr_by_region(
    fpr_df: pd.DataFrame,
    overall_fpr: float,
    path: str = FPR_CHART_PATH,
) -> None:
    """
    Save a bar chart of FPR by region with a reference line for the overall FPR.

    Bars for flagged regions are highlighted in orange/red.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    colours = [
        "#EF5350" if flagged else "#42A5F5"
        for flagged in fpr_df["flagged"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(fpr_df["region"], fpr_df["fpr"] * 100, color=colours, edgecolor="white")

    # Reference line — overall FPR
    ax.axhline(
        y=overall_fpr * 100,
        color="#1B5E20",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall FPR ({overall_fpr * 100:.1f}%)",
    )

    # Threshold bands
    upper = (overall_fpr + FPR_DEVIATION_THRESHOLD_PP / 100) * 100
    lower = max(0, (overall_fpr - FPR_DEVIATION_THRESHOLD_PP / 100) * 100)
    ax.axhspan(lower, upper, alpha=0.08, color="#1B5E20", label=f"±{FPR_DEVIATION_THRESHOLD_PP}pp band")

    # Value labels on top of each bar
    for bar, val in zip(bars, fpr_df["fpr"] * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax.set_title(
        "FAIRPAY AI — Fairness Audit: False Positive Rate by Region\n"
        "(Red bars exceed ±5 pp threshold)",
        fontsize=12,
    )
    ax.set_xticks(range(len(fpr_df)))
    ax.set_xticklabels(fpr_df["region"].tolist(), rotation=25, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  FPR chart saved to: {path}")


def print_fairness_summary(
    fpr_df: pd.DataFrame,
    overall_fpr: float,
) -> None:
    """Print a plain-language fairness summary to stdout."""
    n_flagged = fpr_df["flagged"].sum()
    flagged_regions = fpr_df[fpr_df["flagged"]]["region"].tolist()

    print("── Fairness Summary ────────────────────────────────────────")
    print(
        f"  Overall False Positive Rate : {overall_fpr * 100:.2f}%\n"
        f"  Deviation threshold         : ±{FPR_DEVIATION_THRESHOLD_PP} percentage points\n"
    )

    if n_flagged == 0:
        print(
            "  RESULT: No regional disparity detected.\n"
            "  The model's false positive rate is consistent across all regions.\n"
            "  No region deviates by more than "
            f"{FPR_DEVIATION_THRESHOLD_PP} pp from the overall FPR.\n"
            "  The model does not disproportionately burden any regional group."
        )
    else:
        region_list = ", ".join(flagged_regions)
        print(
            f"  RESULT: Potential fairness concern detected.\n"
            f"  {n_flagged} region(s) deviate by more than "
            f"{FPR_DEVIATION_THRESHOLD_PP} pp from the overall FPR:\n"
            f"  → {region_list}\n\n"
            "  These regions experience a disproportionate rate of false positive\n"
            "  anomaly flags. Consumers in these areas may receive incorrect\n"
            "  loyalty-penalty alerts at a higher rate than the general population.\n"
            "  Recommend investigating regional pricing data distributions and\n"
            "  considering region-stratified re-calibration of the model."
        )

    print()
    print("── FPR by Region (sorted by FPR descending) ────────────────")
    display_df = fpr_df[["region", "n_normal", "fp", "tn", "fpr", "deviation_pp", "flagged"]].copy()
    display_df["fpr_pct"] = (display_df["fpr"] * 100).round(2)
    print(
        display_df[["region", "n_normal", "fp", "tn", "fpr_pct", "deviation_pp", "flagged"]]
        .rename(columns={"fpr_pct": "fpr_%", "deviation_pp": "dev_pp"})
        .to_string(index=False)
    )


def main() -> None:
    """Load test predictions and run the regional fairness audit."""
    print("=" * 60)
    print("FAIRPAY AI — Fairness Audit")
    print("=" * 60)
    print()

    if not os.path.exists(TEST_PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Test predictions not found at '{TEST_PREDICTIONS_PATH}'. "
            "Run model.py first."
        )

    results = pd.read_csv(TEST_PREDICTIONS_PATH)

    y_true_all = results["y_true"].astype(bool)
    y_pred_all = results["y_pred"].astype(bool)
    overall_fpr = compute_fpr(y_true_all, y_pred_all)

    fpr_df = compute_fpr_by_region(results)
    fpr_df = flag_deviating_regions(fpr_df, overall_fpr)

    print_fairness_summary(fpr_df, overall_fpr)
    print()
    plot_fpr_by_region(fpr_df, overall_fpr)

    print("\nFairness audit complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
report.py
---------
Generates plain-language consumer alert strings for every record in the test
set that has been flagged as a loyalty-penalty anomaly.

For each flagged record the alert compares the consumer's renewal price against
the median renewal price of comparable consumers (same service_type + region)
in the full dataset.

Outputs:
    fairpay_ai/outputs/consumer_alerts.csv
        Columns: consumer_id, service_type, region, renewal_price,
                 peer_median_price, overpayment_amount, overpayment_pct,
                 alert_text
"""

import os

import pandas as pd
import numpy as np

from model import DATA_PATH, TEST_PREDICTIONS_PATH

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ALERTS_PATH = os.path.join(OUTPUTS_DIR, "consumer_alerts.csv")


def compute_peer_medians(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the median renewal price for each (service_type, region) peer group
    using the full dataset (all 10,000 records).

    Returns a DataFrame with columns: service_type, region, peer_median_price.
    """
    peer_medians = (
        df_full
        .groupby(["service_type", "region"])["renewal_price"]
        .median()
        .reset_index()
        .rename(columns={"renewal_price": "peer_median_price"})
    )
    return peer_medians


def build_alert_text(row: pd.Series) -> str:
    """
    Construct a plain-language consumer alert string for a single flagged record.

    Example output:
        "Alert: Your broadband renewal price of £42.10 appears to be £11.20
         (36%) above the median price paid by comparable customers in your
         region. You may be paying a loyalty penalty. Consider reviewing your
         contract."
    """
    renewal = row["renewal_price"]
    median = row["peer_median_price"]
    overpayment = row["overpayment_amount"]
    pct = row["overpayment_pct"]
    svc = row["service_type"]
    region = row["region"]

    return (
        f"Alert: Your {svc} renewal price of £{renewal:.2f} appears to be "
        f"£{overpayment:.2f} ({pct:.0f}%) above the median price paid by "
        f"comparable customers in {region}. "
        "You may be paying a loyalty penalty. "
        "Consider reviewing your contract."
    )


def generate_alerts(
    test_predictions: pd.DataFrame,
    df_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the consumer alerts DataFrame for all flagged anomalies.

    Steps:
    1. Filter test predictions to flagged anomalies (y_pred == True).
    2. Join with peer-median prices on (service_type, region).
    3. Compute overpayment_amount and overpayment_pct.
    4. Generate alert_text for each row.

    Returns the alerts DataFrame.
    """
    # ── 1. Filter to flagged anomalies ─────────────────────────────────────────
    flagged = test_predictions[test_predictions["y_pred"].astype(bool)].copy()

    # ── 2. Join peer medians ───────────────────────────────────────────────────
    peer_medians = compute_peer_medians(df_full)
    flagged = flagged.merge(peer_medians, on=["service_type", "region"], how="left")

    # ── 3. Overpayment metrics ─────────────────────────────────────────────────
    flagged["overpayment_amount"] = (
        flagged["renewal_price"] - flagged["peer_median_price"]
    ).round(2)
    flagged["overpayment_pct"] = (
        (flagged["overpayment_amount"] / flagged["peer_median_price"]) * 100
    ).round(1)

    # Some flagged records may actually be at or below the peer median
    # (false positives). We still generate an alert — the model flagged them —
    # but we cap overpayment_pct at 0 for readability.
    flagged["overpayment_amount"] = flagged["overpayment_amount"].clip(lower=0)
    flagged["overpayment_pct"] = flagged["overpayment_pct"].clip(lower=0)

    # ── 4. Alert text ──────────────────────────────────────────────────────────
    flagged["alert_text"] = flagged.apply(build_alert_text, axis=1)

    output_cols = [
        "consumer_id",
        "service_type",
        "region",
        "renewal_price",
        "peer_median_price",
        "overpayment_amount",
        "overpayment_pct",
        "alert_text",
    ]
    return flagged[output_cols].reset_index(drop=True)


def print_sample_alerts(alerts: pd.DataFrame, n: int = 5) -> None:
    """Print up to n sample alert strings to stdout."""
    print("── Sample Consumer Alerts ──────────────────────────────────")
    for _, row in alerts.head(n).iterrows():
        print(f"  [{row['consumer_id']}]  {row['alert_text']}")
        print()


def main() -> None:
    """Load predictions and full dataset, generate alerts, and save to CSV."""
    print("=" * 60)
    print("FAIRPAY AI — Consumer Alert Report")
    print("=" * 60)
    print()

    for path, label in [
        (TEST_PREDICTIONS_PATH, "Test predictions"),
        (DATA_PATH, "Full dataset"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at '{path}'. "
                "Ensure generate_dataset.py and model.py have been run."
            )

    test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    df_full = pd.read_csv(DATA_PATH)

    print("[1/2] Generating consumer alerts...")
    alerts = generate_alerts(test_predictions, df_full)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    alerts.to_csv(ALERTS_PATH, index=False)

    n_flagged = len(alerts)
    n_true_positives = (
        test_predictions[test_predictions["y_pred"].astype(bool)]["y_true"]
        .astype(bool)
        .sum()
    )
    n_false_positives = n_flagged - n_true_positives

    print(f"[2/2] Alerts saved to: {ALERTS_PATH}")
    print()
    print(f"  Total alerts generated : {n_flagged:,}")
    print(f"  True positive alerts   : {n_true_positives:,}  (genuine anomalies)")
    print(f"  False positive alerts  : {n_false_positives:,}  (incorrectly flagged)")
    print()

    print_sample_alerts(alerts)

    print("Report generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
generate_dataset.py
-------------------
Generates 10,000 synthetic consumer pricing records for the FAIRPAY AI
proof-of-concept anomaly detection system.

The dataset simulates loyalty penalty pricing anomalies in UK digital service
markets (broadband, mobile, digital subscriptions).
"""

import os
import numpy as np
import pandas as pd

# Reproducibility
RNG_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "pricing_data.csv")

# ── Dataset dimensions ─────────────────────────────────────────────────────────
N_RECORDS = 10_000
ANOMALY_RATE = 0.08          # 8% → 800 injected anomaly records
ANOMALY_TENURE_THRESHOLD = 24  # anomalies only injected for tenure > 24 months

# ── Service-type distribution (target ≈ 40 / 35 / 25 %) ───────────────────────
SERVICE_TYPES = ["broadband", "mobile", "subscription"]
SERVICE_PROBS = [0.40, 0.35, 0.25]

# New-customer price parameters (log-normal, £)
# log-normal parameterised via target mean & std on the *natural* scale
SERVICE_PRICE_PARAMS = {
    #                mean_£   std_£
    "broadband":    (29.20,   6.00),
    "mobile":       (22.50,   5.00),
    "subscription": (12.80,   3.00),
}

# ── Region distribution ────────────────────────────────────────────────────────
REGIONS = [
    "London", "South East", "North West", "Scotland",
    "Wales", "Midlands", "South West", "North East",
]
# Approximate UK population weights for the regions above
REGION_PROBS = [0.20, 0.18, 0.14, 0.09, 0.06, 0.16, 0.10, 0.07]

# ── Renewal price multipliers ──────────────────────────────────────────────────
NORMAL_RENEWAL_MEAN_MULTIPLIER = 1.05
NORMAL_RENEWAL_SIGMA = 0.08          # log-normal sigma

ANOMALY_UPLIFT_MEAN = 1.28           # right-skewed target mean
ANOMALY_UPLIFT_MIN = 1.15
ANOMALY_UPLIFT_MAX = 1.55


def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """Convert natural-scale mean & std to log-normal mu and sigma parameters."""
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / mean ** 2))
    return mu, sigma


def generate_consumer_ids(n: int) -> list[str]:
    """Return a list of zero-padded pseudonymous consumer IDs."""
    return [f"CONS_{i:05d}" for i in range(1, n + 1)]


def generate_service_types(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample service types according to the 40/35/25 target split."""
    return rng.choice(SERVICE_TYPES, size=n, p=SERVICE_PROBS)


def generate_regions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample regions according to approximate UK population weights."""
    probs = np.array(REGION_PROBS, dtype=float)
    probs /= probs.sum()          # normalise to sum exactly to 1.0
    return rng.choice(REGIONS, size=n, p=probs)


def generate_tenure(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate tenure in months as a mixture of:
    - Uniform(1, 24) for short-tenure consumers  (~50 % of records)
    - Exponential(mean=36) for long-tenure consumers (~50 % of records)

    Values are capped at 120 months (10 years).
    """
    short_mask = rng.random(n) < 0.50
    tenure = np.empty(n, dtype=float)
    n_short = short_mask.sum()
    n_long = n - n_short

    tenure[short_mask] = rng.uniform(1, 24, size=n_short)
    tenure[~short_mask] = rng.exponential(scale=36, size=n_long)
    tenure = np.clip(tenure, 1, 120)
    return np.round(tenure).astype(int)


def generate_new_customer_price(
    service_types: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate new-customer prices from per-service log-normal distributions."""
    prices = np.empty(len(service_types), dtype=float)
    for svc, (mean, std) in SERVICE_PRICE_PARAMS.items():
        mask = service_types == svc
        mu, sigma = _lognormal_params(mean, std)
        prices[mask] = rng.lognormal(mean=mu, sigma=sigma, size=mask.sum())
    return np.round(prices, 2)


def generate_anomaly_mask(
    n: int,
    tenure: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inject anomalies at exactly 8 % of all records.

    Anomalies are drawn exclusively from consumers with tenure > 24 months,
    reflecting the real-world pattern where loyal customers face higher renewal
    prices.
    """
    eligible_indices = np.where(tenure > ANOMALY_TENURE_THRESHOLD)[0]
    n_anomalies = int(round(N_RECORDS * ANOMALY_RATE))

    if len(eligible_indices) < n_anomalies:
        raise ValueError(
            f"Insufficient eligible long-tenure consumers "
            f"({len(eligible_indices)}) to inject {n_anomalies} anomalies."
        )

    chosen = rng.choice(eligible_indices, size=n_anomalies, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True
    return mask


def generate_renewal_price(
    new_prices: np.ndarray,
    anomaly_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate renewal prices:
    - Normal consumers: renewal ≈ new_price × LogNormal(mean=1.05, σ=0.08)
    - Anomalies: renewal = new_price × uplift, where uplift is right-skewed
      with mean ≈ 1.28, drawn from a truncated lognormal in [1.15, 1.55].
    """
    n = len(new_prices)
    renewal_prices = np.empty(n, dtype=float)

    # ── Normal consumers ───────────────────────────────────────────────────────
    normal_mask = ~anomaly_mask
    normal_mu, normal_sigma = _lognormal_params(
        NORMAL_RENEWAL_MEAN_MULTIPLIER, NORMAL_RENEWAL_MEAN_MULTIPLIER * NORMAL_RENEWAL_SIGMA
    )
    normal_multipliers = rng.lognormal(
        mean=normal_mu, sigma=normal_sigma, size=normal_mask.sum()
    )
    renewal_prices[normal_mask] = new_prices[normal_mask] * normal_multipliers

    # ── Anomalous consumers — right-skewed loyalty uplift ─────────────────────
    # Use a lognormal truncated to [1.15, 1.55] by rejection sampling.
    uplift_mu, uplift_sigma = _lognormal_params(ANOMALY_UPLIFT_MEAN, 0.09)
    n_anom = anomaly_mask.sum()
    uplift = np.empty(n_anom, dtype=float)
    generated = 0
    while generated < n_anom:
        batch = rng.lognormal(mean=uplift_mu, sigma=uplift_sigma, size=(n_anom - generated) * 3)
        valid = batch[(batch >= ANOMALY_UPLIFT_MIN) & (batch <= ANOMALY_UPLIFT_MAX)]
        take = min(len(valid), n_anom - generated)
        uplift[generated: generated + take] = valid[:take]
        generated += take

    renewal_prices[anomaly_mask] = new_prices[anomaly_mask] * uplift
    return np.round(renewal_prices, 2)


def build_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Assemble the full synthetic pricing dataset as a DataFrame."""
    consumer_ids = generate_consumer_ids(N_RECORDS)
    service_types = generate_service_types(N_RECORDS, rng)
    regions = generate_regions(N_RECORDS, rng)
    tenure = generate_tenure(N_RECORDS, rng)
    new_prices = generate_new_customer_price(service_types, rng)
    anomaly_mask = generate_anomaly_mask(N_RECORDS, tenure, rng)
    renewal_prices = generate_renewal_price(new_prices, anomaly_mask, rng)

    return pd.DataFrame(
        {
            "consumer_id": consumer_ids,
            "service_type": service_types,
            "region": regions,
            "tenure_months": tenure,
            "new_customer_price": new_prices,
            "renewal_price": renewal_prices,
            "is_anomaly": anomaly_mask,
        }
    )


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset statistics and anomaly distribution to stdout."""
    n = len(df)
    n_anom = df["is_anomaly"].sum()

    print("=" * 60)
    print("FAIRPAY AI — Synthetic Dataset Summary")
    print("=" * 60)
    print(f"Total records      : {n:,}")
    print(f"Total anomalies    : {n_anom:,} ({n_anom / n * 100:.1f}%)")
    print()

    print("── Service-type distribution ──────────────────────────────")
    svc_counts = df["service_type"].value_counts()
    for svc, cnt in svc_counts.items():
        print(f"  {svc:<14}: {cnt:,} ({cnt / n * 100:.1f}%)")
    print()

    print("── Anomaly distribution by service type ───────────────────")
    anom_by_svc = df[df["is_anomaly"]].groupby("service_type").size()
    for svc, cnt in anom_by_svc.items():
        total_svc = svc_counts[svc]
        print(f"  {svc:<14}: {cnt:,} anomalies ({cnt / total_svc * 100:.1f}% of {svc})")
    print()

    print("── Price statistics by service type (new customer price) ──")
    price_stats = df.groupby("service_type")["new_customer_price"].agg(["mean", "std", "min", "max"])
    print(price_stats.round(2).to_string())
    print()

    print("── Tenure distribution ────────────────────────────────────")
    print(df["tenure_months"].describe().round(1).to_string())
    print("=" * 60)


def main() -> None:
    """Generate dataset, save to CSV, and print summary statistics."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("Generating synthetic dataset...")
    df = build_dataframe(rng)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to: {OUTPUT_PATH}")
    print()
    print_summary(df)


if __name__ == "__main__":
    main()

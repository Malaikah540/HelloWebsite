"""
main.py
-------
FAIRPAY AI — Main Runner

Executes the complete proof-of-concept pipeline in order:

    Step 1 — generate_dataset.py : Synthetic dataset generation
    Step 2 — model.py            : Isolation Forest training & prediction
    Step 3 — evaluate.py         : Metrics, confusion matrix, sensitivity plot
    Step 4 — fairness_audit.py   : Regional FPR fairness audit
    Step 5 — report.py           : Consumer alert generation

Run with:
    python main.py
"""

import sys
import time

# ── Import pipeline modules ────────────────────────────────────────────────────
import generate_dataset
import model as model_module
import evaluate
import fairness_audit
import report

PIPELINE_STEPS = [
    ("Step 1 — Dataset Generation",          generate_dataset.main),
    ("Step 2 — Isolation Forest Training",    model_module.main),
    ("Step 3 — Model Evaluation",             evaluate.main),
    ("Step 4 — Fairness Audit",               fairness_audit.main),
    ("Step 5 — Consumer Alert Report",        report.main),
]

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          FAIRPAY AI — Anomaly Detection PoC                  ║
║  Detecting loyalty penalty pricing in UK digital markets     ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_step(step_name: str, step_fn) -> float:
    """
    Run a single pipeline step, printing a header and elapsed time.

    Returns elapsed wall-clock seconds.
    """
    separator = "─" * 62
    print(f"\n{separator}")
    print(f"  {step_name}")
    print(f"{separator}\n")

    t0 = time.perf_counter()
    step_fn()
    elapsed = time.perf_counter() - t0

    print(f"\n  ✓ {step_name} completed in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    """Run all pipeline steps sequentially and print a final summary."""
    print(BANNER)

    total_start = time.perf_counter()
    timings: list[tuple[str, float]] = []

    for step_name, step_fn in PIPELINE_STEPS:
        try:
            elapsed = run_step(step_name, step_fn)
            timings.append((step_name, elapsed))
        except Exception as exc:
            print(f"\n  ERROR in {step_name}: {exc}", file=sys.stderr)
            raise

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "═" * 62)
    print("  FAIRPAY AI Pipeline — Complete")
    print("═" * 62)
    for step_name, elapsed in timings:
        print(f"  {elapsed:5.1f}s  {step_name}")
    print(f"  {'─' * 40}")
    print(f"  {total_elapsed:5.1f}s  Total")
    print("═" * 62)
    print()
    print("  Outputs written to: fairpay_ai/outputs/")
    print("    • confusion_matrix.png")
    print("    • anomaly_breakdown.png")
    print("    • contamination_sensitivity.png")
    print("    • fairness_fpr_by_region.png")
    print("    • consumer_alerts.csv")
    print()


if __name__ == "__main__":
    main()

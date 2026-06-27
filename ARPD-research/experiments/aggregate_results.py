"""
Aggregate multi-seed results and compute statistical significance.

Usage:
  python experiments/aggregate_results.py

Reads: results/arpd_results.csv, results/baseline_results.csv
Outputs: console summary + results/significance_report.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_and_pivot(path: Path, model_col: str = "model") -> dict[str, list[float]]:
    """Load CSV, group by model, return {model_name: [f1_macro per seed]}."""
    df = pd.read_csv(path)
    out = {}
    for model, group in df.groupby(model_col):
        out[str(model)] = group["test_f1_macro"].tolist()
    return out


def paired_ttest_onesided(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    One-sided paired t-test: H1: mean(a) > mean(b).
    Returns (t_statistic, p_value).
    p_value from two-sided / 2 if t > 0, else 1.0.
    """
    t, p_two = stats.ttest_rel(a, b)
    p_one = p_two / 2 if t > 0 else 1.0
    return float(t), float(p_one)


def wilcoxon_onesided(a: list[float], b: list[float]) -> float:
    """Wilcoxon signed-rank test (less parametric than t-test)."""
    if len(set(np.array(a) - np.array(b))) < 2:
        return float("nan")
    _, p_two = stats.wilcoxon(a, b)
    diffs = np.array(a) - np.array(b)
    return float(p_two / 2) if np.sum(diffs) > 0 else 1.0


def main() -> None:
    arpd_path = RESULTS_DIR / "arpd_results.csv"
    base_path = RESULTS_DIR / "baseline_results.csv"

    if not arpd_path.exists():
        print(f"ERROR: {arpd_path} not found. Run experiments first.")
        sys.exit(1)

    arpd_df = pd.read_csv(arpd_path)

    # Aggregate ARPD variants
    print("\n" + "=" * 70)
    print("ARPD results by variant (mean ± std across seeds)")
    print("=" * 70)
    fmt = f"{'Variant':<30} {'Seeds':>5} {'F1-macro':>10} {'Acc':>8} {'F1-fake':>8}"
    print(fmt)
    print("-" * 70)

    # Group by config (augmentation, retrieve, use_speaker, use_ensemble)
    group_cols = [c for c in ["augmentation", "retrieve", "use_speaker", "use_ensemble"]
                  if c in arpd_df.columns]
    groups = {}
    for keys, grp in arpd_df.groupby(group_cols):
        label = "_".join(str(k) for k in (keys if isinstance(keys, tuple) else [keys]))
        groups[label] = grp

    rows = []
    for label, grp in groups.items():
        f1s = grp["test_f1_macro"].tolist()
        accs = grp["test_acc"].tolist()
        ffs = grp["test_f1_fake"].tolist()
        print(
            f"{label:<30} {len(f1s):>5} "
            f"{np.mean(f1s):>8.4f}±{np.std(f1s):.4f} "
            f"{np.mean(accs):>8.4f} "
            f"{np.mean(ffs):>8.4f}"
        )
        rows.append({
            "variant": label,
            "n_seeds": len(f1s),
            "f1_macro_mean": np.mean(f1s),
            "f1_macro_std": np.std(f1s),
            "acc_mean": np.mean(accs),
            "f1_fake_mean": np.mean(ffs),
        })

    # Significance test: Full ARPD vs each baseline
    print("\n" + "=" * 70)
    print("Statistical significance (one-sided paired t-test, H1: ARPD > baseline)")
    print("=" * 70)

    # Identify "Full ARPD" row (retrieve=True, augmentation='synonym', use_ensemble=True)
    full_mask = (arpd_df.get("retrieve", True) == True)
    if "augmentation" in arpd_df.columns:
        full_mask &= arpd_df["augmentation"] == "synonym"
    if "use_ensemble" in arpd_df.columns:
        full_mask &= arpd_df["use_ensemble"] == True
    arpd_full_f1 = arpd_df.loc[full_mask, "test_f1_macro"].tolist()

    if not arpd_full_f1:
        print("No 'Full ARPD' rows found. Skipping significance tests.")
    else:
        if base_path.exists():
            base_df = pd.read_csv(base_path)
            for model, grp in base_df.groupby("model"):
                base_f1 = grp["test_f1_macro"].tolist()
                if len(base_f1) != len(arpd_full_f1):
                    print(f"  {model}: seed count mismatch ({len(base_f1)} vs {len(arpd_full_f1)}) — skipping")
                    continue
                t, p = paired_ttest_onesided(arpd_full_f1, base_f1)
                sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
                print(
                    f"  Full ARPD ({np.mean(arpd_full_f1):.4f}) vs {model} ({np.mean(base_f1):.4f}): "
                    f"t={t:+.3f}, p={p:.4f}  [{sig}]"
                )
        else:
            print(f"  {base_path} not found — run run_baseline.py first.")

    # Save summary CSV
    summary_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "significance_report.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()

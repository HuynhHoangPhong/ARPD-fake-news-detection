"""
Build Wikipedia evidence cache CSVs for all three LIAR splits.

Generates:
    cached_train_evidence.csv
    cached_val_evidence.csv
    cached_test_evidence.csv

Each CSV has columns:
    claim              — original claim text (key for lookup)
    k_used             — k value from UncertaintyScorer
    retrieved_evidence — passages joined with ' [SEP] ' (empty string if none)

Usage:
    python build_evidence_cache.py [--splits train val test] [--sleep 0.5]

Timing estimates (measured on LIAR train, varies with network):
  --sleep 0.5 (default, polite): ~10-15h on Colab T4, ~40-48h on local machine.
  --sleep 0.1 (faster, may hit rate limits): ~3-5h on Colab T4.
Run on Colab with --sleep 0.1 for best speed; resume is safe (checkpoints every 50 claims).
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

from src.uncertainty_scorer import UncertaintyScorer
from src.adaptive_retriever import AdaptiveRetriever

DATA_DIR = Path(__file__).parent / "data" / "processed"
OUT_DIR = Path(__file__).parent  # cache files go in repo root


def build_cache(split: str, sleep_between: float = 0.5, resume: bool = True,
                 max_workers: int = 16) -> None:
    split_file = DATA_DIR / f"liar_{split}.csv"
    if not split_file.exists():
        # Handle 'validation' vs 'val' naming
        alt = DATA_DIR / f"liar_validation.csv" if split == "val" else None
        if alt and alt.exists():
            split_file = alt
        else:
            print(f"[{split}] Data file not found at {split_file}. Skipping.")
            return

    out_path = OUT_DIR / f"cached_{split}_evidence.csv"

    df = pd.read_csv(split_file)
    claims = df["claim"].tolist()
    print(f"[{split}] {len(claims)} claims → {out_path}")

    # Resume support: skip already-processed claims
    done: dict[str, dict] = {}
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        for _, row in existing.iterrows():
            done[row["claim"]] = row.to_dict()
        print(f"[{split}] Resuming — {len(done)} already cached, {len(claims) - len(done)} remaining")

    scorer = UncertaintyScorer(k_min=1, k_max=5)
    retriever = AdaptiveRetriever(sleep_between=sleep_between, sim_threshold=0.25,
                                   max_workers=max_workers)

    rows = []
    for claim in tqdm(claims, desc=split):
        if claim in done:
            rows.append(done[claim])
            continue

        k = scorer.compute_k(claim)
        try:
            passages = retriever.retrieve(claim, k)
        except Exception as e:
            print(f"  ERROR on claim '{claim[:60]}': {e}")
            passages = []

        evidence_str = " [SEP] ".join(passages) if passages else ""
        rows.append({"claim": claim, "k_used": k, "retrieved_evidence": evidence_str})

        # Checkpoint every 50 claims so at most ~3-4 min of work is lost on interruption.
        # (500 was too coarse: at 4s/claim that's 33 min to first save.)
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)

    # Report fill rate
    result_df = pd.DataFrame(rows)
    non_empty = result_df["retrieved_evidence"].apply(lambda v: isinstance(v, str) and v.strip() != "").sum()
    print(f"[{split}] Done. Evidence fill rate: {non_empty}/{len(rows)} ({100*non_empty/len(rows):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build Wikipedia evidence cache for LIAR splits")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds between Wikipedia API calls (default 0.5)")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Số thread song song khi fetch page summaries cho 1 claim "
                             "(default 16). Đặt 1 để chạy tuần tự như bản gốc.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Rebuild from scratch even if partial cache exists")
    args = parser.parse_args()

    for split in args.splits:
        build_cache(split, sleep_between=args.sleep, resume=not args.no_resume,
                    max_workers=args.max_workers)


if __name__ == "__main__":
    main()

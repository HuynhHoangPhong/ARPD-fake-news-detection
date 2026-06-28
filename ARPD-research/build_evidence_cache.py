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
    python build_evidence_cache.py [--splits train val test] [--max-claims N]

Note: This uses the optimized Batch API retriever. It natively handles rate limits 
and does not require artificial sleep or threading parameters.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

from src.uncertainty_scorer import UncertaintyScorer
from src.adaptive_retriever import AdaptiveRetriever

DATA_DIR = Path(__file__).parent / "data" / "processed"
OUT_DIR = Path(__file__).parent  # cache files go in repo root


def build_cache(split: str, resume: bool = True, max_new_claims: int | None = None) -> None:
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
    retriever = AdaptiveRetriever(sim_threshold=0.25)

    rows = []
    new_processed = 0
    stopped_early = False
    
    for claim in tqdm(claims, desc=split):
        if claim in done:
            rows.append(done[claim])
            continue

        if max_new_claims is not None and new_processed >= max_new_claims:
            stopped_early = True
            break

        k = scorer.compute_k(claim)
        try:
            passages = retriever.retrieve(claim, k)
        except Exception as e:
            print(f"  ERROR on claim '{claim[:60]}': {e}")
            passages = []

        evidence_str = " [SEP] ".join(passages) if passages else ""
        rows.append({"claim": claim, "k_used": k, "retrieved_evidence": evidence_str})
        new_processed += 1

        # Checkpoint every 50 claims so at most ~3-4 min of work is lost on interruption.
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)

    if stopped_early:
        remaining = len(claims) - len(rows)
        print(f"[{split}] Đã xử lý {new_processed} claim mới trong lô này, dừng theo --max-claims. "
              f"Còn {remaining} claim chưa xử lý — chạy lại cùng lệnh để tiếp tục (tự resume).")
        return

    # Report fill rate
    result_df = pd.DataFrame(rows)
    non_empty = result_df["retrieved_evidence"].apply(lambda v: isinstance(v, str) and v.strip() != "").sum()
    print(f"[{split}] Done. Evidence fill rate: {non_empty}/{len(rows)} ({100*non_empty/len(rows):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build Wikipedia evidence cache for LIAR splits")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--no-resume", action="store_true",
                        help="Rebuild from scratch even if partial cache exists")
    parser.add_argument("--max-claims", type=int, default=None,
                        help="Max number of NEW claims to process in one run. Use to split "
                             "work into small batches for safe sync/checkpoint on Colab. "
                             "Re-run the same command to resume from where it stopped.")
    args = parser.parse_args()

    for split in args.splits:
        build_cache(split, resume=not args.no_resume, max_new_claims=args.max_claims)


if __name__ == "__main__":
    main()

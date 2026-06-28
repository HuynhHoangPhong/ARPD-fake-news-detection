"""
Run ARPD full pipeline on LIAR dataset.
Results saved to results/arpd_results.csv (appended across runs for ablation).

Usage:
  python experiments/run_arpd.py                         # full ARPD, seed 42
  python experiments/run_arpd.py --no-retrieve           # ablation: no evidence
  python experiments/run_arpd.py --no-speaker            # ablation: no speaker context
  python experiments/run_arpd.py --no-ensemble           # ablation: MLP only
  python experiments/run_arpd.py --augmentation none     # ablation: no augmentation
  python experiments/run_arpd.py --seed 123              # different seed
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all_splits, load_processed, save_processed
from src.pipeline import ARPDPipeline


RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CACHE_DIR = Path(__file__).parent.parent


def load_or_download() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {s: DATA_DIR / f"liar_{s}.csv" for s in ["train", "validation", "test"]}
    if all(p.exists() for p in paths.values()):
        print("Loading cached LIAR splits...")
        return tuple(pd.read_csv(p) for p in paths.values())
    print("Downloading LIAR dataset...")
    splits = load_all_splits()
    for name, df in splits.items():
        save_processed(df, DATA_DIR / f"liar_{name}.csv")
    return splits["train"], splits["validation"], splits["test"]


def main(
    augmentation: str = "synonym",
    retrieve: bool = True,
    use_speaker: bool = True,
    use_ensemble: bool = True,
    epochs: int = 20,
    k_min: int = 1,
    k_max: int = 5,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    RESULTS_DIR.mkdir(exist_ok=True)

    train_cache = CACHE_DIR / "cached_train_evidence.csv" if retrieve else None
    val_cache   = CACHE_DIR / "cached_val_evidence.csv"   if retrieve else None
    test_cache  = CACHE_DIR / "cached_test_evidence.csv"  if retrieve else None

    print("Loading LIAR splits...")
    train, val, test = load_or_download()

    pipeline = ARPDPipeline(
        k_min=k_min,
        k_max=k_max,
        augmentation_method=augmentation,
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        cached_train_path=train_cache,
        cached_val_path=val_cache,
        use_speaker_context=use_speaker,
        use_ensemble=use_ensemble,
        grid_search_weight=use_ensemble,  # always tune on val when ensemble is on
    )

    # Extract speaker/subject lists (None if speaker context disabled)
    def _cols(df: pd.DataFrame, col: str) -> list[str] | None:
        if not use_speaker or col not in df.columns:
            return None
        return df[col].fillna("").tolist()

    print("\nFitting ARPD pipeline...")
    pipeline.fit(
        train["claim"].tolist(), train["label"].tolist(),
        val["claim"].tolist(), val["label"].tolist(),
        train_speakers=_cols(train, "speaker"),
        train_subjects=_cols(train, "subject"),
        val_speakers=_cols(val, "speaker"),
        val_subjects=_cols(val, "subject"),
        epochs=epochs,
        verbose=True,
    )

    print("\nEvaluating on test set...")
    preds = pipeline.predict(
        test["claim"].tolist(),
        speakers=_cols(test, "speaker"),
        subjects=_cols(test, "subject"),
        cached_test_path=test_cache,
    )
    y_test = test["label"].values

    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    result = {
        "model": "ARPD",
        "augmentation": augmentation,
        "retrieve": retrieve,
        "use_speaker": use_speaker,
        "use_ensemble": use_ensemble,
        "ensemble_weight": pipeline.ensemble_weight,
        "k_min": k_min,
        "k_max": k_max,
        "seed": seed,
        "test_acc": accuracy_score(y_test, preds),
        "test_f1_macro": f1_score(y_test, preds, average="macro"),
        "test_f1_fake": f1_score(y_test, preds, pos_label=0, average="binary"),
        "test_f1_real": f1_score(y_test, preds, pos_label=1, average="binary"),
    }

    out_path = RESULTS_DIR / "arpd_results.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        pd.concat([existing, pd.DataFrame([result])], ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"\nResult saved -> {out_path}")
    print(f"Test Acc={result['test_acc']:.4f}  F1-macro={result['test_f1_macro']:.4f}")

    pipeline.save(RESULTS_DIR / "checkpoints")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", default="synonym",
                        choices=["none", "synonym", "backtranslate", "both"])
    parser.add_argument("--no-retrieve",  action="store_true")
    parser.add_argument("--no-speaker",   action="store_true")
    parser.add_argument("--no-ensemble",  action="store_true")
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--k-min",        type=int, default=1)
    parser.add_argument("--k-max",        type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    main(
        augmentation=args.augmentation,
        retrieve=not args.no_retrieve,
        use_speaker=not args.no_speaker,
        use_ensemble=not args.no_ensemble,
        epochs=args.epochs,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )

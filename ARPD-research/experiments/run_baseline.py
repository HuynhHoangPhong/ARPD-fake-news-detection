"""
Baselines:
  1. TF-IDF + Logistic Regression (with speaker-context features)
  2. Frozen all-distilroberta-v1 encoder + trained MLP head
     (NOTE: this is NOT DistilBERT fine-tuned end-to-end.
      The encoder weights are frozen; only the MLP classifier is trained.)

Results appended to results/baseline_results.csv (one row per seed per model).
Run across multiple seeds:
  for seed in 42 123 456 789 2025; do
      python experiments/run_baseline.py --seed $seed
  done
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_liar, load_all_splits, load_processed, save_processed


RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_or_download() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {s: DATA_DIR / f"liar_{s}.csv" for s in ["train", "validation", "test"]}
    if all(p.exists() for p in paths.values()):
        print("Loading cached data...")
        return tuple(pd.read_csv(p) for p in paths.values())
    print("Downloading LIAR dataset...")
    splits = load_all_splits()
    for name, df in splits.items():
        save_processed(df, DATA_DIR / f"liar_{name}.csv")
    return splits["train"], splits["validation"], splits["test"]


def _safe_str(value) -> str:
    """Convert value to str, returning '' for None/NaN (avoids .strip() on float)."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _add_speaker_context(df: pd.DataFrame) -> list[str]:
    """Format "[speaker] [subject] claim" for TF-IDF input (mirrors ARPD encoder)."""
    return [
        f"[{_safe_str(row.get('speaker')) or 'unknown'}] "
        f"[{_safe_str(row.get('subject')) or 'unknown'}] "
        f"{row['claim']}"
        for _, row in df.iterrows()
    ]


def run_tfidf_lr(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """TF-IDF + Logistic Regression baseline (with speaker-context features)."""
    print(f"\n--- Baseline 1: TF-IDF + LR [seed={seed}] ---")
    # Bug A fix: TfidfVectorizer has no 'C' parameter — that belongs to LogisticRegression.
    vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
    X_train = vec.fit_transform(_add_speaker_context(train))
    X_val   = vec.transform(_add_speaker_context(val))
    X_test  = vec.transform(_add_speaker_context(test))

    clf = LogisticRegression(max_iter=1000, C=0.5, random_state=seed)
    clf.fit(X_train, train["label"])

    result = {"model": "TF-IDF+LR", "seed": seed}
    for split_name, X, y in [
        ("val",  X_val,  val["label"]),
        ("test", X_test, test["label"]),
    ]:
        preds = clf.predict(X)
        result[f"{split_name}_acc"]      = accuracy_score(y, preds)
        result[f"{split_name}_f1_macro"] = f1_score(y, preds, average="macro")
        result[f"{split_name}_f1_fake"]  = f1_score(y, preds, pos_label=0, average="binary")
        result[f"{split_name}_f1_real"]  = f1_score(y, preds, pos_label=1, average="binary")
        print(
            f"  {split_name}: acc={result[f'{split_name}_acc']:.4f}"
            f"  f1={result[f'{split_name}_f1_macro']:.4f}"
        )
    return result


def run_distilbert(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    """
    Frozen all-distilroberta-v1 encoder + trained MLP classifier.

    The encoder (all-distilroberta-v1, ~82M params) is frozen; only the
    MLP head (~190K params) is trained.
    """
    import torch
    torch.manual_seed(seed)

    print(f"\n--- Baseline 2: Frozen DistilRoBERTa + MLP [seed={seed}] ---")
    from sentence_transformers import SentenceTransformer
    from src.classifier import ARPDTrainer

    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

    def encode(df: pd.DataFrame) -> np.ndarray:
        return model.encode(
            _add_speaker_context(df), show_progress_bar=True, batch_size=batch_size
        )

    print("  Encoding train...")
    X_train = encode(train)
    print("  Encoding val...")
    X_val = encode(val)
    print("  Encoding test...")
    X_test = encode(test)

    trainer = ARPDTrainer(input_dim=X_train.shape[1])
    trainer.fit(
        X_train, train["label"].values,
        X_val,   val["label"].values,
        epochs=epochs, verbose=True,
    )

    result = {"model": "FrozenDistilRoBERTa+MLP", "seed": seed}
    for split_name, X, y in [
        ("val",  X_val,  val["label"].values),
        ("test", X_test, test["label"].values),
    ]:
        m = trainer.evaluate(X, y)
        result[f"{split_name}_acc"]      = m["accuracy"]
        result[f"{split_name}_f1_macro"] = m["f1_macro"]
        result[f"{split_name}_f1_fake"]  = m["f1_fake"]
        result[f"{split_name}_f1_real"]  = m["f1_real"]
        print(
            f"  {split_name}: acc={m['accuracy']:.4f}"
            f"  f1={m['f1_macro']:.4f}"
        )
    return result


def main(seed: int = 42) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    train, val, test = load_or_download()

    rows = [
        run_tfidf_lr(train, val, test, seed=seed),
        run_distilbert(train, val, test, seed=seed),
    ]

    out_path = RESULTS_DIR / "baseline_results.csv"
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}  ({len(new_df)} total rows)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)

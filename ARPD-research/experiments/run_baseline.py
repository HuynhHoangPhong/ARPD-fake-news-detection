"""
Baselines:
  1. TF-IDF + Logistic Regression (không dùng evidence)
  2. DistilBERT fine-tuned (không dùng evidence, không augmentation)

Kết quả lưu vào results/baseline_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_liar, load_processed, save_processed


RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_or_download() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load từ cache nếu có, download nếu chưa có."""
    paths = {s: DATA_DIR / f"liar_{s}.csv" for s in ["train", "validation", "test"]}
    if all(p.exists() for p in paths.values()):
        print("Loading cached data...")
        return tuple(pd.read_csv(p) for p in paths.values())

    print("Downloading LIAR dataset...")
    from src.data_loader import load_all_splits
    splits = load_all_splits()
    for name, df in splits.items():
        save_processed(df, DATA_DIR / f"liar_{name}.csv")
    return splits["train"], splits["validation"], splits["test"]


def run_tfidf_lr(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """TF-IDF + Logistic Regression baseline."""
    print("\n--- Baseline 1: TF-IDF + Logistic Regression ---")
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vec.fit_transform(train["claim"])
    X_val = vec.transform(val["claim"])
    X_test = vec.transform(test["claim"])

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, train["label"])

    results = {}
    for split_name, X, y in [("val", X_val, val["label"]), ("test", X_test, test["label"])]:
        preds = clf.predict(X)
        results[f"{split_name}_acc"] = accuracy_score(y, preds)
        results[f"{split_name}_f1"] = f1_score(y, preds, average="macro")
        print(f"  {split_name}: acc={results[f'{split_name}_acc']:.4f} f1={results[f'{split_name}_f1']:.4f}")

    results["model"] = "TF-IDF+LR"
    return results


def run_distilbert(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    epochs: int = 3, batch_size: int = 32,
) -> dict:
    """
    DistilBERT fine-tuned baseline.
    Dùng sentence-transformers encode rồi train MLP (để đơn giản và nhanh).
    DistilBERT: 66M params — trong giới hạn 125M.
    """
    print("\n--- Baseline 2: DistilBERT + MLP ---")
    from sentence_transformers import SentenceTransformer
    from src.classifier import ARPDTrainer

    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

    def encode(df: pd.DataFrame) -> np.ndarray:
        return model.encode(df["claim"].tolist(), show_progress_bar=True, batch_size=batch_size)

    print("  Encoding train...")
    X_train = encode(train)
    print("  Encoding val...")
    X_val = encode(val)
    print("  Encoding test...")
    X_test = encode(test)

    trainer = ARPDTrainer(input_dim=X_train.shape[1])
    trainer.fit(
        X_train, train["label"].values,
        X_val, val["label"].values,
        epochs=epochs, verbose=True,
    )

    results = {"model": "DistilBERT+MLP"}
    for split_name, X, y in [("val", X_val, val["label"].values), ("test", X_test, test["label"].values)]:
        metrics = trainer.evaluate(X, y)
        results[f"{split_name}_acc"] = metrics["accuracy"]
        results[f"{split_name}_f1"] = metrics["f1_macro"]
        print(f"  {split_name}: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    train, val, test = load_or_download()

    all_results = []
    all_results.append(run_tfidf_lr(train, val, test))
    all_results.append(run_distilbert(train, val, test))

    df_results = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "baseline_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()

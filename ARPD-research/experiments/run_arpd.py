"""
Chạy ARPD full pipeline trên LIAR dataset.
Kết quả lưu vào results/arpd_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import classification_report

from src.data_loader import load_processed, save_processed, load_all_splits
from src.pipeline import ARPDPipeline


RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_or_download() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {s: DATA_DIR / f"liar_{s}.csv" for s in ["train", "validation", "test"]}
    if all(p.exists() for p in paths.values()):
        return tuple(pd.read_csv(p) for p in paths.values())
    splits = load_all_splits()
    for name, df in splits.items():
        save_processed(df, DATA_DIR / f"liar_{name}.csv")
    return splits["train"], splits["validation"], splits["test"]


def main(
    augmentation: str = "synonym",
    retrieve: bool = True,
    epochs: int = 20,
    k_min: int = 1,
    k_max: int = 5,
):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    train, val, test = load_or_download()

    pipeline = ARPDPipeline(
        k_min=k_min,
        k_max=k_max,
        augmentation_method=augmentation,
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    print("\nFitting ARPD pipeline...")
    history = pipeline.fit(
        train["claim"].tolist(), train["label"].tolist(),
        val["claim"].tolist(), val["label"].tolist(),
        epochs=epochs,
        retrieve_evidence=retrieve,
        verbose=True,
    )

    # Evaluate on test
    print("\nEvaluating on test set...")
    preds = pipeline.predict(test["claim"].tolist(), retrieve_evidence=retrieve)
    y_test = test["label"].values

    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    from sklearn.metrics import accuracy_score, f1_score
    result = {
        "model": "ARPD",
        "augmentation": augmentation,
        "retrieve": retrieve,
        "k_min": k_min,
        "k_max": k_max,
        "test_acc": accuracy_score(y_test, preds),
        "test_f1_macro": f1_score(y_test, preds, average="macro"),
        "test_f1_fake": f1_score(y_test, preds, pos_label=0, average="binary"),
        "test_f1_real": f1_score(y_test, preds, pos_label=1, average="binary"),
    }

    df_result = pd.DataFrame([result])
    out_path = RESULTS_DIR / "arpd_results.csv"

    # Append nếu đã tồn tại (ablation study)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        df_result = pd.concat([existing, df_result], ignore_index=True)

    df_result.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    # Lưu training history
    history_path = RESULTS_DIR / f"history_{augmentation}_retrieve{retrieve}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"History saved → {history_path}")

    pipeline.save(RESULTS_DIR / "checkpoints")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", default="synonym",
                        choices=["none", "synonym", "backtranslate", "both"])
    parser.add_argument("--no-retrieve", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=5)
    args = parser.parse_args()

    main(
        augmentation=args.augmentation,
        retrieve=not args.no_retrieve,
        epochs=args.epochs,
        k_min=args.k_min,
        k_max=args.k_max,
    )

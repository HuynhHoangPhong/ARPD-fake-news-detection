"""
ARPD Experiment Runner — single-seed full pipeline evaluation.

Usage:
    python experiments/run_experiment.py --seed 42
    python experiments/run_experiment.py --seed 42 --no-retrieve

Outputs:
    results/seed_{seed}.json  with keys:
        accuracy, f1_macro, f1_fake, f1_real,
        acc_clean, acc_attacked, robustness_drop,
        tfidf_acc, tfidf_f1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_liar, save_processed
from src.encoder import ClaimEvidenceEncoder
from src.classifier import ARPDTrainer
from src.uncertainty_scorer import UncertaintyScorer
from src.paraphrase_augmentor import combined_augment, ensure_nltk_data


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = {}
    for name in ("train", "validation", "test"):
        path = DATA_DIR / f"liar_{name}.csv"
        if path.exists():
            splits[name] = pd.read_csv(path)
        else:
            print(f"  Downloading {name}...")
            df = load_liar(name)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_processed(df, path)
            splits[name] = df
    return splits["train"], splits["validation"], splits["test"]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_table(rows: list[dict]) -> None:
    """Print a simple fixed-width results table."""
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    print("\n" + header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))


# ---------------------------------------------------------------------------
# Baseline: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def run_tfidf_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    print("\n[Baseline] TF-IDF + Logistic Regression")
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(train["claim"])
    X_te = vec.transform(test["claim"])

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_tr, train["label"])
    preds = clf.predict(X_te)

    acc = accuracy_score(test["label"], preds)
    f1 = f1_score(test["label"], preds, average="macro", zero_division=0)
    print(f"  test acc={acc:.4f}  f1_macro={f1:.4f}")
    return {"tfidf_acc": acc, "tfidf_f1": f1, "_tfidf_vec": vec, "_tfidf_clf": clf}


# ---------------------------------------------------------------------------
# Encode: MiniLM with optional Wikipedia evidence cache
# ---------------------------------------------------------------------------

def encode_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    retrieve: bool,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all splits. If retrieve=False, uses claim-only embeddings.
    Returns (X_train, X_val, X_test) as float32 arrays.
    """
    encoder = ClaimEvidenceEncoder(device=device)
    scorer = UncertaintyScorer(k_min=1, k_max=5)
    scorer.fit_reference([])  # linguistic scorer — no-op

    def _encode(df: pd.DataFrame, desc: str) -> np.ndarray:
        claims = df["claim"].tolist()
        if retrieve:
            from src.adaptive_retriever import AdaptiveRetriever
            retriever = AdaptiveRetriever(sleep_between=0.2)
            k_list = scorer.batch_compute_k(claims)
            from tqdm import tqdm
            passages_list = [
                retriever.retrieve(c, k)
                for c, k in tqdm(zip(claims, k_list), total=len(claims), desc=desc)
            ]
        else:
            passages_list = [[] for _ in claims]
        return encoder.encode_batch(claims, passages_list, show_progress=True)

    print("\n[Encode] Training set...")
    X_tr = _encode(train, "Train retrieve")

    print("[Encode] Validation set...")
    X_val = _encode(val, "Val retrieve")

    print("[Encode] Test set...")
    X_te = _encode(test, "Test retrieve")

    return X_tr, X_val, X_te


# ---------------------------------------------------------------------------
# ImprovedMLP training
# ---------------------------------------------------------------------------

def train_improved_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    device: str,
    epochs: int,
    patience: int,
) -> tuple[dict, ARPDTrainer]:
    print("\n[Train] ImprovedMLP with class weights + augmentation")

    # Synonym + combined augmentation to double training set
    ensure_nltk_data()
    # (augmentation is done in pipeline.fit; here we do it manually
    #  so we can control the seed per-experiment)
    # Skip heavy augmentation here — handled in ARPDTrainer.fit

    trainer = ARPDTrainer(input_dim=384, device=device, use_improved=True)
    history = trainer.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=epochs, batch_size=64, patience=patience,
        verbose=True, compute_class_weight=True,
    )

    metrics = trainer.evaluate(X_te, y_te)
    print(f"\n  Test: acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}"
          f"  f1_fake={metrics['f1_fake']:.4f}  f1_real={metrics['f1_real']:.4f}")
    return metrics, trainer


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------

def evaluate_robustness(
    trainer: ARPDTrainer,
    encoder: ClaimEvidenceEncoder,
    test_claims: list[str],
    test_labels: np.ndarray,
    seed: int,
) -> dict:
    """
    Attack: combined_augment (synonym + deletion + swap) at varying intensity.
    Reports clean accuracy, attacked accuracy, and robustness_drop.
    """
    print("\n[Robustness] Combined paraphrase attack")

    # Clean
    X_clean = encoder.encode_batch(test_claims, [[] for _ in test_claims], show_progress=False)
    clean_preds = trainer.predict(X_clean)
    acc_clean = accuracy_score(test_labels, clean_preds)

    # Attacked: combined_augment (p_synonym=0.30, p_deletion=0.15, n_swap=3)
    adv_claims = [
        combined_augment(c, p_synonym=0.30, p_deletion=0.15, n_swap=3, seed=seed + i)
        for i, c in enumerate(test_claims)
    ]
    X_adv = encoder.encode_batch(adv_claims, [[] for _ in adv_claims], show_progress=False)
    adv_preds = trainer.predict(X_adv)
    acc_attacked = accuracy_score(test_labels, adv_preds)

    drop = acc_clean - acc_attacked
    print(f"  Clean acc:   {acc_clean:.4f}")
    print(f"  Attacked acc:{acc_attacked:.4f}")
    print(f"  Drop:        {drop:+.4f}")
    return {"acc_clean": acc_clean, "acc_attacked": acc_attacked, "robustness_drop": drop}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _set_seed(args.seed)
    print(f"Seed={args.seed}  device={device}  retrieve={not args.no_retrieve}")

    # --- Data ---
    print("\n[Data] Loading LIAR splits...")
    train, val, test = _load_splits()
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    # --- TF-IDF Baseline ---
    tfidf_result = run_tfidf_baseline(train, val, test)

    # --- Encode ---
    X_tr, X_val, X_te = encode_splits(
        train, val, test,
        retrieve=not args.no_retrieve,
        seed=args.seed,
        device=device,
    )

    # --- ImprovedMLP ---
    mlp_metrics, trainer = train_improved_mlp(
        X_tr, train["label"].values,
        X_val, val["label"].values,
        X_te, test["label"].values,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
    )

    # --- Robustness ---
    encoder = ClaimEvidenceEncoder(device=device)
    rob = evaluate_robustness(
        trainer, encoder,
        test["claim"].tolist(),
        test["label"].values,
        seed=args.seed,
    )

    # --- Save results ---
    result = {
        "seed": args.seed,
        "retrieve": not args.no_retrieve,
        # ARPD metrics
        "accuracy":         round(mlp_metrics["accuracy"], 6),
        "f1_macro":         round(mlp_metrics["f1_macro"], 6),
        "f1_fake":          round(mlp_metrics["f1_fake"], 6),
        "f1_real":          round(mlp_metrics["f1_real"], 6),
        # Robustness
        "acc_clean":        round(rob["acc_clean"], 6),
        "acc_attacked":     round(rob["acc_attacked"], 6),
        "robustness_drop":  round(rob["robustness_drop"], 6),
        # Baseline
        "tfidf_acc":        round(tfidf_result["tfidf_acc"], 6),
        "tfidf_f1":         round(tfidf_result["tfidf_f1"], 6),
        # Meta
        "elapsed_sec":      round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / f"seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    # --- Summary table ---
    table = [
        {
            "Model":    "TF-IDF+LR",
            "Acc":      f"{result['tfidf_acc']:.4f}",
            "F1-Mac":   f"{result['tfidf_f1']:.4f}",
            "AccClean": "-",
            "AccAttk":  "-",
            "Drop":     "-",
        },
        {
            "Model":    "ARPD-ImprovedMLP",
            "Acc":      f"{result['accuracy']:.4f}",
            "F1-Mac":   f"{result['f1_macro']:.4f}",
            "AccClean": f"{result['acc_clean']:.4f}",
            "AccAttk":  f"{result['acc_attacked']:.4f}",
            "Drop":     f"{result['robustness_drop']:+.4f}",
        },
    ]
    _print_table(table)
    print(f"\nDone in {result['elapsed_sec']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARPD experiment runner")
    parser.add_argument("--seed",      type=int, default=42, help="Random seed")
    parser.add_argument("--epochs",    type=int, default=30, help="Max training epochs")
    parser.add_argument("--patience",  type=int, default=7,  help="Early stopping patience")
    parser.add_argument("--no-retrieve", action="store_true",
                        help="Skip Wikipedia retrieval (faster, no-evidence mode)")
    main(parser.parse_args())

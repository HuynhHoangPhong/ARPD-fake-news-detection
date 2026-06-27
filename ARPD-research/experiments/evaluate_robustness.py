"""
Robustness Evaluation — đánh giá model dưới paraphrase attack.

Protocol:
  1. Load trained ARPD pipeline.
  2. Tạo adversarial test set bằng synonym substitution trên test claims.
  3. So sánh accuracy trên clean vs. adversarial.
  4. Tính Robustness Drop = clean_acc - adversarial_acc.
  5. Lưu kết quả vào results/robustness_results.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_processed
from src.paraphrase_augmentor import synonym_substitute, back_translate, ensure_nltk_data
from src.pipeline import ARPDPipeline


RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def create_adversarial(
    claims: list[str],
    method: str = "synonym",
    p_synonym: float = 0.3,
) -> list[str]:
    """
    Tạo adversarial versions của claims.
    p_synonym cao hơn training (0.15) để simulate mạnh hơn.
    """
    ensure_nltk_data()
    adv = []
    for i, c in enumerate(claims):
        if method == "synonym":
            adv.append(synonym_substitute(c, p=p_synonym, seed=i))
        elif method == "backtranslate":
            try:
                adv.append(back_translate(c))
            except Exception:
                adv.append(c)  # fallback nếu model chưa download
        else:
            adv.append(c)
    return adv


def evaluate_robustness(
    pipeline: ARPDPipeline,
    test_claims: list[str],
    test_labels: list[int],
    attack_methods: list[str] | None = None,
    cached_test_path=None,
) -> pd.DataFrame:
    """
    Đánh giá pipeline trên clean + adversarial test set.

    Returns:
        DataFrame với cột: attack_method, accuracy, f1_macro, robustness_drop
    """
    if attack_methods is None:
        attack_methods = ["synonym_p15", "synonym_p30", "synonym_p50"]

    # Clean accuracy
    clean_preds = pipeline.predict(test_claims, cached_test_path=cached_test_path)
    clean_acc = accuracy_score(test_labels, clean_preds)
    clean_f1 = f1_score(test_labels, clean_preds, average="macro")
    print(f"  Clean: acc={clean_acc:.4f} f1={clean_f1:.4f}")

    rows = [{"attack": "clean", "accuracy": clean_acc, "f1_macro": clean_f1, "robustness_drop": 0.0}]

    p_map = {"synonym_p15": 0.15, "synonym_p30": 0.30, "synonym_p50": 0.50}

    for method in attack_methods:
        if method.startswith("synonym"):
            p = p_map.get(method, 0.3)
            adv_claims = create_adversarial(test_claims, method="synonym", p_synonym=p)
        elif method == "backtranslate":
            adv_claims = create_adversarial(test_claims, method="backtranslate")
        else:
            continue

        adv_preds = pipeline.predict(adv_claims, cached_test_path=None)  # adversarial: zero-evidence
        adv_acc = accuracy_score(test_labels, adv_preds)
        adv_f1 = f1_score(test_labels, adv_preds, average="macro")
        drop = clean_acc - adv_acc

        print(f"  {method}: acc={adv_acc:.4f} f1={adv_f1:.4f} drop={drop:+.4f}")
        rows.append({"attack": method, "accuracy": adv_acc, "f1_macro": adv_f1, "robustness_drop": drop})

    return pd.DataFrame(rows)


CACHE_DIR = Path(__file__).parent.parent


def main(retrieve: bool = False):
    RESULTS_DIR.mkdir(exist_ok=True)

    test_path = DATA_DIR / "liar_test.csv"
    if not test_path.exists():
        print("Test data not found. Run run_arpd.py first.")
        return

    test = pd.read_csv(test_path)
    test_claims = test["claim"].tolist()
    test_labels = test["label"].tolist()

    test_cache = CACHE_DIR / "cached_test_evidence.csv" if retrieve else None

    # Load trained pipeline
    print("Loading ARPD pipeline...")
    pipeline = ARPDPipeline()

    ckpt_dir = RESULTS_DIR / "checkpoints"
    if (ckpt_dir / "arpd_classifier.pt").exists():
        pipeline.load(ckpt_dir)
    else:
        print("WARNING: No checkpoint found. Fit pipeline first via run_arpd.py")
        return

    # Cần fit_reference cho scorer
    train = pd.read_csv(DATA_DIR / "liar_train.csv")
    print("Fitting uncertainty scorer reference...")
    pipeline.scorer.fit_reference(train["claim"].tolist())

    print("\nEvaluating robustness...")
    df_rob = evaluate_robustness(
        pipeline, test_claims, test_labels,
        attack_methods=["synonym_p15", "synonym_p30", "synonym_p50"],
        cached_test_path=test_cache,
    )

    out_path = RESULTS_DIR / "robustness_results.csv"
    df_rob.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print(df_rob.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve", action="store_true",
                        help="Use Wikipedia retrieval during evaluation")
    args = parser.parse_args()
    main(retrieve=args.retrieve)

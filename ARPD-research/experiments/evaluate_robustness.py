"""
Robustness Evaluation — đánh giá model dưới paraphrase attack.

Protocol:
  1. Load trained ARPD pipeline (checkpoint + pipeline_state.pkl).
  2. Tạo adversarial test set bằng synonym substitution trên test claims.
  3. So sánh accuracy trên clean vs. adversarial.
  4. Tính Robustness Drop = clean_acc - adversarial_acc.
  5. Lưu kết quả vào results/robustness_results.csv.

Usage:
  python experiments/evaluate_robustness.py              # zero-evidence, full ARPD config
  python experiments/evaluate_robustness.py --retrieve   # use test cache
  python experiments/evaluate_robustness.py --no-speaker --no-ensemble  # ablation config
"""

import argparse
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
CACHE_DIR = Path(__file__).parent.parent


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
                adv.append(c)
        else:
            adv.append(c)
    return adv


def evaluate_robustness(
    pipeline: ARPDPipeline,
    test_claims: list[str],
    test_labels: list[int],
    test_speakers: list[str] | None = None,
    test_subjects: list[str] | None = None,
    attack_methods: list[str] | None = None,
    cached_test_path=None,
) -> pd.DataFrame:
    """
    Đánh giá pipeline trên clean + adversarial test set.

    Bug E fix: speakers/subjects are passed to every pipeline.predict() call.
    For adversarial claims, speaker/subject metadata stays the same — only the
    claim text is perturbed; the speaker identity does not change.

    Returns:
        DataFrame với cột: attack, accuracy, f1_macro, robustness_drop
    """
    if attack_methods is None:
        attack_methods = ["synonym_p15", "synonym_p30", "synonym_p50"]

    # Clean accuracy — pass speakers/subjects so encoder uses same context as training
    clean_preds = pipeline.predict(
        test_claims,
        speakers=test_speakers,
        subjects=test_subjects,
        cached_test_path=cached_test_path,
    )
    clean_acc = accuracy_score(test_labels, clean_preds)
    clean_f1  = f1_score(test_labels, clean_preds, average="macro")
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

        # Adversarial: zero-evidence (cache is for original claims only).
        # Speaker/subject metadata is unchanged — the identity of who said the
        # claim does not change when we paraphrase the claim text.
        adv_preds = pipeline.predict(
            adv_claims,
            speakers=test_speakers,
            subjects=test_subjects,
            cached_test_path=None,
        )
        adv_acc = accuracy_score(test_labels, adv_preds)
        adv_f1  = f1_score(test_labels, adv_preds, average="macro")
        drop    = clean_acc - adv_acc

        print(f"  {method}: acc={adv_acc:.4f} f1={adv_f1:.4f} drop={drop:+.4f}")
        rows.append({"attack": method, "accuracy": adv_acc, "f1_macro": adv_f1, "robustness_drop": drop})

    return pd.DataFrame(rows)


def main(
    retrieve: bool = False,
    use_speaker: bool = True,
    use_ensemble: bool = True,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    test_path = DATA_DIR / "liar_test.csv"
    if not test_path.exists():
        print("Test data not found. Run run_arpd.py first.")
        return

    test = pd.read_csv(test_path)
    test_claims = test["claim"].tolist()
    test_labels = test["label"].tolist()

    # Bug E fix: read speakers/subjects from test set so pipeline.predict() gets
    # the same context the model was trained with.
    test_speakers = test["speaker"].fillna("").tolist() if use_speaker else None
    test_subjects = test["subject"].fillna("").tolist() if use_speaker else None

    test_cache = CACHE_DIR / "cached_test_evidence.csv" if retrieve else None

    # Bug E fix: initialise pipeline with the flags that match the checkpoint.
    # If the checkpoint was trained with --no-speaker, pass use_speaker_context=False.
    print(
        f"Loading ARPD pipeline "
        f"[use_speaker={use_speaker}, use_ensemble={use_ensemble}]..."
    )
    pipeline = ARPDPipeline(
        use_speaker_context=use_speaker,
        use_ensemble=use_ensemble,
    )

    ckpt_dir = RESULTS_DIR / "checkpoints"
    if (ckpt_dir / "arpd_classifier.pt").exists():
        pipeline.load(ckpt_dir)
    else:
        print("WARNING: No checkpoint found. Fit pipeline first via run_arpd.py")
        return

    train = pd.read_csv(DATA_DIR / "liar_train.csv")
    print("Fitting uncertainty scorer reference...")
    pipeline.scorer.fit_reference(train["claim"].tolist())

    print("\nEvaluating robustness...")
    df_rob = evaluate_robustness(
        pipeline,
        test_claims,
        test_labels,
        test_speakers=test_speakers,
        test_subjects=test_subjects,
        attack_methods=["synonym_p15", "synonym_p30", "synonym_p50"],
        cached_test_path=test_cache,
    )

    out_path = RESULTS_DIR / "robustness_results.csv"
    df_rob.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")
    print(df_rob.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve",    action="store_true")
    parser.add_argument("--no-speaker",  action="store_true",
                        help="Match checkpoint trained with --no-speaker flag")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Match checkpoint trained with --no-ensemble flag")
    args = parser.parse_args()
    main(
        retrieve=args.retrieve,
        use_speaker=not args.no_speaker,
        use_ensemble=not args.no_ensemble,
    )

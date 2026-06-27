"""
ARPD Full Pipeline — Speaker-Aware Context + TF-IDF/LR Ensemble

Components (in order):
  1. UncertaintyScorer  -> k_adaptive per claim
  2. ParaphraseAugmentor -> training augmentation
  3. ClaimEvidenceEncoder -> 384-dim feature vectors (speaker-aware context)
  4. TF-IDF + LogisticRegression  (ensemble component)
  5. ImprovedMLP via ARPDTrainer  (ensemble component)
  6. Ensemble: (1-w)*prob_tfidf_lr + w*prob_mlp; w tuned on val set

Ablation flags:
  use_speaker_context=False -> disable speaker/subject prefix
  use_ensemble=False        -> MLP-only prediction
  augmentation_method="none" -> no paraphrase augmentation
  cached_*_path=None        -> zero-evidence mode
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .uncertainty_scorer import UncertaintyScorer
from .paraphrase_augmentor import augment_dataset, ensure_nltk_data
from .encoder import ClaimEvidenceEncoder
from .classifier import ARPDTrainer


_ENSEMBLE_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


class ARPDPipeline:
    """End-to-end ARPD pipeline with speaker-aware context and ensemble."""

    def __init__(
        self,
        k_min: int = 1,
        k_max: int = 5,
        augmentation_method: str = "synonym",
        p_synonym: float = 0.15,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        cached_train_path: str | Path | None = None,
        cached_val_path: str | Path | None = None,
        use_speaker_context: bool = True,
        use_ensemble: bool = True,
        ensemble_weight: float = 0.10,
        grid_search_weight: bool = True,
        tfidf_max_features: int = 15000,
        tfidf_ngram_range: tuple = (1, 3),
    ) -> None:
        """
        Args:
            use_speaker_context: Prepend "[speaker] [subject]" to claim text and
                encode as a single string. Ablation: set False for "No Speaker" variant.
            use_ensemble: Combine TF-IDF+LR and ImprovedMLP probabilities.
                Ablation: set False for "MLP-only" variant.
            ensemble_weight: MLP weight in ensemble (w in (1-w)*LR + w*MLP).
                Ignored when grid_search_weight=True (val-set optimised).
            grid_search_weight: If True, search _ENSEMBLE_WEIGHTS on val F1-macro
                during fit(). This sets ensemble_weight to the best found value.
                CRITICAL: search is done on VAL set only, never test set.
            tfidf_max_features: Vocabulary size for TF-IDF vectoriser.
            tfidf_ngram_range: N-gram range for TF-IDF.
        """
        self.augmentation_method = augmentation_method
        self.p_synonym = p_synonym
        self.cached_train_path = cached_train_path
        self.cached_val_path = cached_val_path
        self.use_speaker_context = use_speaker_context
        self.use_ensemble = use_ensemble
        self.ensemble_weight = ensemble_weight
        self.grid_search_weight = grid_search_weight

        self.scorer = UncertaintyScorer(model_name=encoder_model, k_min=k_min, k_max=k_max)
        self.encoder = ClaimEvidenceEncoder(
            model_name=encoder_model,
            device=device,
            use_speaker_context=use_speaker_context,
        )
        self.trainer = ARPDTrainer(input_dim=384, device=device)

        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
        )
        self.lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)

        ensure_nltk_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context_texts(
        self,
        claims: list[str],
        speakers: list[str] | None,
        subjects: list[str] | None,
    ) -> list[str]:
        """Format "[speaker] [subject] claim" strings for TF-IDF input."""
        if not self.use_speaker_context or speakers is None or subjects is None:
            return claims
        return [
            f"[{(spk or '').strip() or 'unknown'}] [{(sub or '').strip() or 'unknown'}] {c}"
            for c, spk, sub in zip(claims, speakers, subjects)
        ]

    def _load_cached_evidence(
        self,
        claims: list[str],
        cache_path: str | Path,
        desc: str,
    ) -> list[list[str]]:
        """Load pre-retrieved evidence from CSV; align by claim text."""
        if not cache_path or not Path(cache_path).exists():
            raise FileNotFoundError(f"Cache file not found at {cache_path}")

        print(f"[{desc}] Reading cache from {cache_path}...")
        df = pd.read_csv(cache_path)

        def parse_evidence(text):
            if not isinstance(text, str) or not text.strip():
                return []
            return [p for p in text.split(" [SEP] ") if p.strip()]

        df["retrieved_evidence"] = df["retrieved_evidence"].apply(parse_evidence)
        cache_dict = dict(zip(df["claim"], df["retrieved_evidence"]))
        return [cache_dict.get(claim, []) for claim in claims]

    def _featurize(
        self,
        claims: list[str],
        passages_list: list[list[str]],
        speakers: list[str] | None = None,
        subjects: list[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self.encoder.encode_batch(
            claims, passages_list,
            speakers=speakers, subjects=subjects,
            show_progress=show_progress,
        )

    def _ensemble_predict_proba(
        self, X_emb: np.ndarray, X_tfidf, w: float
    ) -> np.ndarray:
        """(1-w)*P_lr + w*P_mlp.  Returns (N, 2) probabilities."""
        prob_lr = self.lr.predict_proba(X_tfidf)          # (N, 2)
        prob_mlp = self.trainer.predict_proba(X_emb)      # (N, 2)
        return (1.0 - w) * prob_lr + w * prob_mlp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_claims: list[str],
        train_labels: list[int],
        val_claims: list[str],
        val_labels: list[int],
        train_speakers: list[str] | None = None,
        train_subjects: list[str] | None = None,
        val_speakers: list[str] | None = None,
        val_subjects: list[str] | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 5,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Train ARPD pipeline.

        Args:
            train_speakers / val_speakers: Speaker strings from LIAR dataset.
                Pass None to run in "No Speaker" ablation mode.
            train_subjects / val_subjects: Subject strings from LIAR dataset.
        Returns:
            Training history (list of per-epoch metric dicts).
        """
        if verbose:
            print("[1/5] Fitting uncertainty scorer...")
        self.scorer.fit_reference(train_claims)

        if verbose:
            print("[2/5] Augmenting training data...")
        aug_claims, aug_labels = train_claims, train_labels
        aug_speakers, aug_subjects = train_speakers, train_subjects
        if self.augmentation_method != "none":
            aug_claims, aug_labels = augment_dataset(
                train_claims, train_labels,
                method=self.augmentation_method,
                p_synonym=self.p_synonym,
            )
            # Mirror speaker/subject lists to match doubled claims
            if train_speakers is not None:
                aug_speakers = list(train_speakers) + list(train_speakers)
            if train_subjects is not None:
                aug_subjects = list(train_subjects) + list(train_subjects)

        if verbose:
            print("[3/5] Loading evidence caches...")
        if self.cached_train_path is not None:
            base_ev = self._load_cached_evidence(
                train_claims, self.cached_train_path, "Train"
            )
            passages_train = base_ev + base_ev if self.augmentation_method != "none" else base_ev
        else:
            if verbose:
                print("       [Ablation] No train cache -> zero-evidence mode.")
            passages_train = [[] for _ in aug_claims]

        if self.cached_val_path is not None:
            passages_val = self._load_cached_evidence(
                val_claims, self.cached_val_path, "Val"
            )
        else:
            passages_val = [[] for _ in val_claims]

        if verbose:
            print("[4/5] Encoding features (MLP path)...")
        X_train = self._featurize(
            aug_claims, passages_train, aug_speakers, aug_subjects
        )
        X_val = self._featurize(
            val_claims, passages_val, val_speakers, val_subjects
        )
        y_train = np.array(aug_labels)
        y_val = np.array(val_labels)

        # TF-IDF+LR (ensemble path) — fit on speaker-context claim texts
        if self.use_ensemble:
            ctx_train = self._build_context_texts(
                train_claims, train_speakers, train_subjects
            )
            ctx_val = self._build_context_texts(
                val_claims, val_speakers, val_subjects
            )
            X_tr_tfidf = self.tfidf.fit_transform(ctx_train)
            X_val_tfidf = self.tfidf.transform(ctx_val)
            self.lr.fit(X_tr_tfidf, train_labels)

        history = self.trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience, verbose=verbose,
        )

        if verbose:
            print("[5/5] Grid-searching ensemble weight on val set...")
        if self.use_ensemble and self.grid_search_weight:
            best_f1, best_w = -1.0, self.ensemble_weight
            for w in _ENSEMBLE_WEIGHTS:
                proba = self._ensemble_predict_proba(X_val, X_val_tfidf, w)
                preds = proba.argmax(axis=1)
                f1 = f1_score(y_val, preds, average="macro", zero_division=0)
                if verbose:
                    print(f"      w={w:.2f}: val F1-macro={f1:.4f}")
                if f1 > best_f1:
                    best_f1, best_w = f1, w
            self.ensemble_weight = best_w
            if verbose:
                print(f"  Best w={best_w:.2f} (val F1-macro={best_f1:.4f})")

        return history

    def predict(
        self,
        claims: list[str],
        speakers: list[str] | None = None,
        subjects: list[str] | None = None,
        cached_test_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Predict labels for claims.

        Args:
            speakers/subjects: Pass for speaker-aware encoding.
            cached_test_path: Path to pre-built test evidence CSV.
                              None -> zero-evidence (ablation) mode.
        Returns:
            Binary predictions (0=FAKE, 1=REAL), shape (N,).
        """
        if cached_test_path:
            passages_list = self._load_cached_evidence(
                claims, cached_test_path, "Test"
            )
        else:
            print("[Inference] No cache path -> zero-evidence ablation mode.")
            passages_list = [[] for _ in claims]

        X = self._featurize(claims, passages_list, speakers, subjects)

        if self.use_ensemble:
            ctx_texts = self._build_context_texts(claims, speakers, subjects)
            X_tfidf = self.tfidf.transform(ctx_texts)
            proba = self._ensemble_predict_proba(X, X_tfidf, self.ensemble_weight)
            return proba.argmax(axis=1)

        return self.trainer.predict(X)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save(save_dir / "arpd_classifier.pt")

    def load(self, save_dir: str | Path) -> None:
        self.trainer.load(Path(save_dir) / "arpd_classifier.pt")


if __name__ == "__main__":
    import pandas as pd

    mock_df = pd.DataFrame({
        "claim": ["Obama signed a bill.", "Vaccines cause autism."],
        "retrieved_evidence": ["Evidence 1", "Evidence 2"],
    })
    mock_df.to_csv("mock_train.csv", index=False)
    mock_df.to_csv("mock_val.csv", index=False)

    pipeline = ARPDPipeline(
        augmentation_method="synonym",
        cached_train_path="mock_train.csv",
        cached_val_path="mock_val.csv",
        use_speaker_context=True,
        use_ensemble=False,  # skip ensemble in smoke test (needs LR fitted)
    )
    history = pipeline.fit(
        ["Obama signed a bill.", "Vaccines cause autism."],
        [1, 0],
        ["Obama signed a bill.", "Vaccines cause autism."],
        [1, 0],
        train_speakers=["Barack Obama", "RFK Jr"],
        train_subjects=["health care", "vaccines"],
        val_speakers=["Barack Obama", "RFK Jr"],
        val_subjects=["health care", "vaccines"],
        epochs=1, verbose=True,
    )

    preds = pipeline.predict(
        ["The president raised taxes."],
        speakers=["unknown"],
        subjects=["economy"],
    )
    print(f"\nPrediction (zero-ev): {'REAL' if preds[0] == 1 else 'FAKE'}")

    import os
    os.remove("mock_train.csv")
    os.remove("mock_val.csv")

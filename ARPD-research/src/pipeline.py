"""
ARPD Full Pipeline — ghép 5 thành phần:
  1. UncertaintyScorer  → k_adaptive per claim
  2. AdaptiveRetriever  → evidence passages
  3. ParaphraseAugmentor → training augmentation
  4. ClaimEvidenceEncoder → feature vectors
  5. ARPDTrainer / ARPDClassifier → binary prediction

Hai chế độ:
  - fit():     Train ARPD từ đầu (bao gồm augmentation).
  - predict(): Inference cho claim mới.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .uncertainty_scorer import UncertaintyScorer
from .adaptive_retriever import AdaptiveRetriever
from .paraphrase_augmentor import augment_dataset, ensure_nltk_data, synonym_substitute
from .encoder import ClaimEvidenceEncoder
from .classifier import ARPDTrainer


class ARPDPipeline:
    """End-to-end ARPD pipeline."""

    def __init__(
        self,
        k_min: int = 1,
        k_max: int = 5,
        augmentation_method: str = "synonym",
        p_synonym: float = 0.15,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        retriever_sleep: float = 0.3,
    ) -> None:
        """
        Args:
            k_min: Số evidence tối thiểu.
            k_max: Số evidence tối đa.
            augmentation_method: "synonym" | "backtranslate" | "both" | "none".
            p_synonym: Xác suất synonym substitution.
            encoder_model: SentenceTransformer model name.
            device: "cuda" | "cpu" | None.
            retriever_sleep: Giây nghỉ giữa Wikipedia calls.
        """
        self.augmentation_method = augmentation_method
        self.p_synonym = p_synonym

        self.scorer = UncertaintyScorer(
            model_name=encoder_model, k_min=k_min, k_max=k_max
        )
        self.retriever = AdaptiveRetriever(sleep_between=retriever_sleep)
        self.encoder = ClaimEvidenceEncoder(model_name=encoder_model, device=device)
        self.trainer = ARPDTrainer(input_dim=384, device=device)

        ensure_nltk_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_all(
        self, claims: list[str], k_list: list[int], desc: str = "Retrieving"
    ) -> list[list[str]]:
        """Retrieve evidence cho toàn bộ claims với progress bar."""
        results = []
        for claim, k in tqdm(zip(claims, k_list), total=len(claims), desc=desc):
            results.append(self.retriever.retrieve(claim, k))
        return results

    def _featurize(
        self,
        claims: list[str],
        passages_list: list[list[str]],
        desc: str = "Encoding",
    ) -> np.ndarray:
        return self.encoder.encode_batch(claims, passages_list, show_progress=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_claims: list[str],
        train_labels: list[int],
        val_claims: list[str],
        val_labels: list[int],
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 5,
        retrieve_evidence: bool = True,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Train ARPD pipeline.

        Args:
            train_claims / train_labels: Training data.
            val_claims / val_labels: Validation data.
            retrieve_evidence: Có gọi Wikipedia không (tắt để debug nhanh).
            ...

        Returns:
            Training history (list of dicts).
        """
        # Bước 1: Fit uncertainty scorer trên training claims
        if verbose:
            print("[1/5] Fitting uncertainty scorer...")
        self.scorer.fit_reference(train_claims)

        # Bước 2: Tính k_adaptive cho training set
        if verbose:
            print("[2/5] Computing adaptive k values...")
        k_list_train = self.scorer.batch_compute_k(train_claims)

        # Bước 3: Augmentation
        aug_claims, aug_labels = train_claims, train_labels
        if self.augmentation_method != "none":
            if verbose:
                print(f"[3/5] Augmenting training data (method={self.augmentation_method})...")
            aug_claims, aug_labels = augment_dataset(
                train_claims, train_labels,
                method=self.augmentation_method,
                p_synonym=self.p_synonym,
            )
            # k_list cho phần augmented = same as original
            k_list_train = k_list_train + k_list_train
        else:
            if verbose:
                print("[3/5] Skipping augmentation.")

        # Bước 4: Retrieve evidence
        if retrieve_evidence:
            if verbose:
                print("[4/5] Retrieving Wikipedia evidence for training set...")
            passages_train = self._retrieve_all(aug_claims, k_list_train, "Train retrieve")

            if verbose:
                print("[4/5] Retrieving Wikipedia evidence for validation set...")
            k_list_val = self.scorer.batch_compute_k(val_claims)
            passages_val = self._retrieve_all(val_claims, k_list_val, "Val retrieve")
        else:
            passages_train = [[] for _ in aug_claims]
            passages_val = [[] for _ in val_claims]

        # Bước 5: Encode
        if verbose:
            print("[5/5] Encoding train features...")
        X_train = self._featurize(aug_claims, passages_train, "Encode train")
        if verbose:
            n_with_ev = sum(bool(p) for p in passages_train)
            print(f"       {n_with_ev}/{len(aug_claims)} train samples have evidence after filtering")
        if verbose:
            print("[5/5] Encoding val features...")
        X_val = self._featurize(val_claims, passages_val, "Encode val")

        y_train = np.array(aug_labels)
        y_val = np.array(val_labels)

        # Train classifier
        if verbose:
            print("\nTraining MLP classifier...")
        history = self.trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience, verbose=verbose,
        )
        return history

    def predict(self, claims: list[str], retrieve_evidence: bool = True) -> np.ndarray:
        """
        Inference trên list of claims.

        Returns:
            numpy array of binary predictions (0=FAKE, 1=REAL).
        """
        k_list = self.scorer.batch_compute_k(claims)

        if retrieve_evidence:
            passages_list = self._retrieve_all(claims, k_list, "Inference retrieve")
        else:
            passages_list = [[] for _ in claims]

        X = self._featurize(claims, passages_list, "Inference encode")
        return self.trainer.predict(X)

    def save(self, save_dir: str | Path) -> None:
        """Lưu classifier weights."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save(save_dir / "arpd_classifier.pt")

    def load(self, save_dir: str | Path) -> None:
        """Load classifier weights."""
        self.trainer.load(Path(save_dir) / "arpd_classifier.pt")


if __name__ == "__main__":
    # Smoke test với dummy data (không retrieve để nhanh)
    claims_train = ["Obama signed a bill.", "Vaccines cause autism."] * 50
    labels_train = [1, 0] * 50
    claims_val = ["New tax reform passed.", "Moon landing was faked."] * 10
    labels_val = [1, 0] * 10

    pipeline = ARPDPipeline(augmentation_method="synonym")
    history = pipeline.fit(
        claims_train, labels_train,
        claims_val, labels_val,
        epochs=3, retrieve_evidence=False, verbose=True,
    )
    preds = pipeline.predict(["The president raised taxes."], retrieve_evidence=False)
    print(f"\nPrediction: {'REAL' if preds[0] == 1 else 'FAKE'}")

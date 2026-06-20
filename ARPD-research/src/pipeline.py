"""
ARPD Full Pipeline — Offline Cached Version
  1. UncertaintyScorer  → k_adaptive per claim
  2. ParaphraseAugmentor → training augmentation
  3. ClaimEvidenceEncoder → feature vectors
  4. ARPDTrainer / ARPDClassifier → binary prediction

Chế độ:
  - fit():     Train ARPD sử dụng local cache và paraphrase augmentation.
  - predict(): Inference sử dụng local test cache.
"""

from __future__ import annotations

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .uncertainty_scorer import UncertaintyScorer
from .paraphrase_augmentor import augment_dataset, ensure_nltk_data
from .encoder import ClaimEvidenceEncoder
from .classifier import ARPDTrainer


class ARPDPipeline:
    """End-to-end ARPD pipeline (Strictly Offline Caches)."""

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
    ) -> None:
        self.augmentation_method = augmentation_method
        self.p_synonym = p_synonym
        self.cached_train_path = cached_train_path
        self.cached_val_path = cached_val_path

        self.scorer = UncertaintyScorer(model_name=encoder_model, k_min=k_min, k_max=k_max)
        self.encoder = ClaimEvidenceEncoder(model_name=encoder_model, device=device)
        self.trainer = ARPDTrainer(input_dim=384, device=device)

        ensure_nltk_data()

    def _load_cached_evidence(self, claims: list[str], cache_path: str | Path, desc: str) -> list[list[str]]:
        """Loads pre-retrieved evidence from CSV and aligns it with the requested claims."""
        if not cache_path or not Path(cache_path).exists():
            raise FileNotFoundError(f"Cache file not found at {cache_path}")
            
        print(f"[{desc}] Reading cache from {cache_path}...")
        df = pd.read_csv(cache_path)
        
        # Parse the custom ' [SEP] ' delimited format your extraction script used
        def parse_evidence(text):
            if not isinstance(text, str) or not text.strip():
                return []
            return text.split(" [SEP] ")
            
        df['retrieved_evidence'] = df['retrieved_evidence'].apply(parse_evidence)
        cache_dict = dict(zip(df['claim'], df['retrieved_evidence']))
        
        return [cache_dict.get(claim, []) for claim in claims]

    def _featurize(self, claims: list[str], passages_list: list[list[str]], desc: str = "Encoding") -> np.ndarray:
        return self.encoder.encode_batch(claims, passages_list, show_progress=True)

    def fit(
        self,
        train_claims: list[str],
        train_labels: list[int],
        val_claims: list[str],
        val_labels: list[int],
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 5,
        verbose: bool = True,
    ) -> list[dict]:
        """Train ARPD pipeline via local evidence files."""
        if verbose:
            print("[1/4] Fitting uncertainty scorer...")
        self.scorer.fit_reference(train_claims)

        if verbose:
            print("[2/4] Augmenting training data...")
        aug_claims, aug_labels = train_claims, train_labels
        if self.augmentation_method != "none":
            aug_claims, aug_labels = augment_dataset(
                train_claims, train_labels,
                method=self.augmentation_method,
                p_synonym=self.p_synonym,
            )

        if verbose:
            print("[3/4] Loading evidence from persistent cache...")
        base_passages_train = self._load_cached_evidence(train_claims, self.cached_train_path, "Train Cache")
        
        if self.augmentation_method != "none":
            passages_train = base_passages_train + base_passages_train
        else:
            passages_train = base_passages_train
            
        passages_val = self._load_cached_evidence(val_claims, self.cached_val_path, "Val Cache")

        if verbose:
            print("[4/4] Encoding features...")
        X_train = self._featurize(aug_claims, passages_train, "Encode train")
        X_val = self._featurize(val_claims, passages_val, "Encode val")

        y_train = np.array(aug_labels)
        y_val = np.array(val_labels)

        if verbose:
            print("\nTraining MLP classifier...")
        return self.trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience, verbose=verbose,
        )

    def predict(self, claims: list[str], cached_test_path: str | Path | None = None) -> np.ndarray:
        """Inference using local test cache."""
        if cached_test_path:
            passages_list = self._load_cached_evidence(claims, cached_test_path, "Test Cache")
        else:
            print("[Inference] WARNING: No cache path provided. Running zero-evidence ablation mode.")
            passages_list = [[] for _ in claims]

        X = self._featurize(claims, passages_list, "Inference encode")
        return self.trainer.predict(X)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save(save_dir / "arpd_classifier.pt")

    def load(self, save_dir: str | Path) -> None:
        self.trainer.load(Path(save_dir) / "arpd_classifier.pt")

if __name__ == "__main__":
    # Smoke test với dummy data sử dụng file tạm
    import pandas as pd
    
    # Tạo dummy cache files để không bị lỗi FileNotFoundError
    mock_df = pd.DataFrame({"claim": ["Obama signed a bill.", "Vaccines cause autism."], "evidence": ["['Evidence 1']", "['Evidence 2']"]})
    mock_df.to_csv("mock_train.csv", index=False)
    mock_df.to_csv("mock_val.csv", index=False)

    claims_train = ["Obama signed a bill.", "Vaccines cause autism."]
    labels_train = [1, 0]
    claims_val = ["Obama signed a bill.", "Vaccines cause autism."]
    labels_val = [1, 0]

    pipeline = ARPDPipeline(
        augmentation_method="synonym",
        cached_train_path="mock_train.csv",
        cached_val_path="mock_val.csv"
    )
    history = pipeline.fit(
        claims_train, labels_train,
        claims_val, labels_val,
        epochs=1, verbose=True
    )
    
    # Test zero-evidence mode prediction
    preds = pipeline.predict(["The president raised taxes."])
    print(f"\nPrediction (Zero-Ev): {'REAL' if preds[0] == 1 else 'FAKE'}")
    
    # Clean up mock files
    import os
    os.remove("mock_train.csv")
    os.remove("mock_val.csv")
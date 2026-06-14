"""
Unit tests for ARPD pipeline components.

Run with:  pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure src/ is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test 1: data_loader
# ---------------------------------------------------------------------------

class TestDataLoader:
    """load_liar returns well-formed DataFrame with binary labels."""

    @pytest.fixture(scope="class")
    def df(self):
        from src.data_loader import load_liar
        return load_liar("train")

    def test_required_columns(self, df):
        assert "claim" in df.columns, "Missing 'claim' column"
        assert "label" in df.columns, "Missing 'label' column"

    def test_binary_labels(self, df):
        unique = set(df["label"].unique())
        assert unique <= {0, 1}, f"Labels not binary: {unique}"

    def test_no_null_claims(self, df):
        assert df["claim"].isna().sum() == 0, "Null claims found"

    def test_minimum_size(self, df):
        assert len(df) >= 10_000, f"Expected >= 10000 rows, got {len(df)}"

    def test_both_classes_present(self, df):
        assert 0 in df["label"].values and 1 in df["label"].values


# ---------------------------------------------------------------------------
# Test 2: UncertaintyScorer
# ---------------------------------------------------------------------------

class TestUncertaintyScorer:
    """compute_k() in [1,5] and varies across claim types."""

    @pytest.fixture(scope="class")
    def scorer(self):
        from src.uncertainty_scorer import UncertaintyScorer
        s = UncertaintyScorer(k_min=1, k_max=5)
        s.fit_reference([])  # no-op, must not raise
        return s

    DIVERSE_CLAIMS = [
        "The unemployment rate fell to 3.5% in October 2023.",       # specific/numeric
        "Congress passed H.R.1234 by a 218-to-210 vote on Tuesday.", # specific
        "President Obama signed a healthcare bill.",                  # medium
        "Some people say the government might be hiding information.",# vague/hedge
        "Many experts suggest climate policy could possibly change.", # vague/hedge
    ]

    def test_k_in_range(self, scorer):
        for claim in self.DIVERSE_CLAIMS:
            k = scorer.compute_k(claim)
            assert 1 <= k <= 5, f"k={k} out of range for: {claim}"

    def test_k_varies(self, scorer):
        ks = [scorer.compute_k(c) for c in self.DIVERSE_CLAIMS]
        assert len(set(ks)) >= 3, f"k values not diverse enough: {ks}"

    def test_score_in_unit_interval(self, scorer):
        for claim in self.DIVERSE_CLAIMS:
            u = scorer.score(claim)
            assert 0.0 <= u <= 1.0, f"score={u} outside [0,1] for: {claim}"

    def test_fit_reference_noop(self):
        from src.uncertainty_scorer import UncertaintyScorer
        s = UncertaintyScorer()
        s.fit_reference(["some claim", "another claim"])  # should not raise


# ---------------------------------------------------------------------------
# Test 3: ClaimEvidenceEncoder
# ---------------------------------------------------------------------------

class TestEncoder:
    """encode_batch returns (N, 384) numpy array."""

    @pytest.fixture(scope="class")
    def encoder(self):
        from src.encoder import ClaimEvidenceEncoder
        return ClaimEvidenceEncoder()

    CLAIMS = [
        "The president signed a new bill.",
        "Vaccines cause autism.",
        "NASA confirmed water on Mars.",
    ]

    def test_output_shape_no_evidence(self, encoder):
        passages_list = [[] for _ in self.CLAIMS]
        X = encoder.encode_batch(self.CLAIMS, passages_list, show_progress=False)
        assert X.shape == (3, 384), f"Expected (3,384), got {X.shape}"

    def test_output_shape_with_evidence(self, encoder):
        passages_list = [
            ["Some relevant text about politics."],
            ["Studies show vaccines are safe.", "CDC recommends vaccination."],
            [],
        ]
        X = encoder.encode_batch(self.CLAIMS, passages_list, show_progress=False)
        assert X.shape == (3, 384), f"Expected (3,384), got {X.shape}"

    def test_output_dtype(self, encoder):
        X = encoder.encode_batch(self.CLAIMS, [[] for _ in self.CLAIMS], show_progress=False)
        assert X.dtype == np.float32

    def test_claim_only_equals_encode_claim(self, encoder):
        # No evidence -> output should equal encode_claim
        claim = "The president signed a new bill."
        X_batch = encoder.encode_batch([claim], [[]], show_progress=False)
        v_claim = encoder.encode_claim(claim)
        np.testing.assert_allclose(X_batch[0], v_claim, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 4: ImprovedMLP classifier
# ---------------------------------------------------------------------------

class TestImprovedMLP:
    """Forward pass: (batch, 384) -> (batch, 2)."""

    @pytest.fixture(scope="class")
    def trainer(self):
        from src.classifier import ARPDTrainer
        return ARPDTrainer(input_dim=384, use_improved=True, device="cpu")

    def test_forward_shape(self, trainer):
        x = torch.randn(16, 384)
        out = trainer.model(x)
        assert out.shape == (16, 2), f"Expected (16,2), got {out.shape}"

    def test_output_is_logits(self, trainer):
        # logits should not be clipped to [0,1]
        x = torch.randn(32, 384)
        out = trainer.model(x)
        # at least some logits outside (0,1)
        assert (out.abs() > 1.0).any(), "Outputs look like probabilities, not logits"

    def test_predict_binary(self, trainer):
        X = np.random.randn(20, 384).astype(np.float32)
        preds = trainer.predict(X)
        assert set(preds.tolist()) <= {0, 1}, f"Non-binary predictions: {set(preds)}"

    def test_fit_runs(self, trainer):
        np.random.seed(0)
        X_tr = np.random.randn(100, 384).astype(np.float32)
        y_tr = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 384).astype(np.float32)
        y_val = np.random.randint(0, 2, 20)
        history = trainer.fit(X_tr, y_tr, X_val, y_val, epochs=2, verbose=False)
        assert len(history) == 2
        assert "f1_macro" in history[0]

    def test_xavier_init(self, trainer):
        from src.classifier import ImprovedMLP
        model = ImprovedMLP(input_dim=384)
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # Xavier: std ~ sqrt(2/(fan_in + fan_out)), not too large
                assert param.abs().max().item() < 5.0, f"Suspicious weight magnitude in {name}"


# ---------------------------------------------------------------------------
# Test 5: Paraphrase augmentor
# ---------------------------------------------------------------------------

class TestAugmentor:
    """synonym_substitute changes >= 1 token in a 20-word sentence."""

    LONG_SENTENCE = (
        "The federal government announced major economic reforms affecting "
        "millions of workers across the entire national territory today."
    )

    def setup_method(self):
        from src.paraphrase_augmentor import ensure_nltk_data
        ensure_nltk_data()

    def test_synonym_changes_tokens(self):
        from src.paraphrase_augmentor import synonym_substitute
        # p=0.5 on a 20-word sentence: high probability of at least one change
        original_tokens = self.LONG_SENTENCE.split()
        changed = False
        for seed in range(10):  # try 10 seeds to avoid flakiness
            result = synonym_substitute(self.LONG_SENTENCE, p=0.5, seed=seed)
            if result.split() != original_tokens:
                changed = True
                break
        assert changed, "synonym_substitute never changed any token across 10 seeds"

    def test_random_deletion_reduces_length(self):
        from src.paraphrase_augmentor import random_deletion
        result = random_deletion(self.LONG_SENTENCE, p=0.5, seed=0)
        assert len(result.split()) < len(self.LONG_SENTENCE.split())

    def test_random_swap_changes_order(self):
        from src.paraphrase_augmentor import random_swap
        result = random_swap(self.LONG_SENTENCE, n=3, seed=42)
        assert result != self.LONG_SENTENCE

    def test_combined_augment_differs(self):
        from src.paraphrase_augmentor import combined_augment
        result = combined_augment(self.LONG_SENTENCE, seed=7)
        assert result != self.LONG_SENTENCE

    def test_short_sentence_deletion_safe(self):
        from src.paraphrase_augmentor import random_deletion
        short = "Yes."
        result = random_deletion(short, p=0.9, seed=0)
        assert len(result) > 0, "Deleted all tokens from short sentence"


# ---------------------------------------------------------------------------
# Test 6: End-to-end pipeline (5 claims, no retrieval)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Pipeline processes 5 claims without error."""

    CLAIMS = [
        "The president signed a new healthcare bill.",
        "Vaccines cause autism according to researchers.",
        "The unemployment rate fell to 3.5% last quarter.",
        "Some people believe the moon landing was faked.",
        "Congress passed a bipartisan infrastructure package.",
    ]
    LABELS = [1, 0, 1, 0, 1]

    def test_pipeline_fit_predict(self):
        from src.pipeline import ARPDPipeline

        pipeline = ARPDPipeline(
            augmentation_method="synonym",
            p_synonym=0.15,
            device="cpu",
        )
        # Use same 5 claims for train and val (tiny, just testing no crash)
        history = pipeline.fit(
            self.CLAIMS * 20, self.LABELS * 20,
            self.CLAIMS, self.LABELS,
            epochs=2,
            retrieve_evidence=False,
            verbose=False,
        )
        assert len(history) > 0

        preds = pipeline.predict(self.CLAIMS, retrieve_evidence=False)
        assert preds.shape == (5,), f"Expected (5,), got {preds.shape}"
        assert set(preds.tolist()) <= {0, 1}, f"Non-binary predictions: {preds}"

    def test_pipeline_save_load(self, tmp_path):
        from src.pipeline import ARPDPipeline

        pipeline = ARPDPipeline(device="cpu")
        pipeline.fit(
            self.CLAIMS * 10, self.LABELS * 10,
            self.CLAIMS, self.LABELS,
            epochs=1, retrieve_evidence=False, verbose=False,
        )

        pipeline.save(tmp_path / "ckpt")
        preds_before = pipeline.predict(self.CLAIMS, retrieve_evidence=False)

        pipeline2 = ARPDPipeline(device="cpu")
        pipeline2.scorer.fit_reference([])
        pipeline2.load(tmp_path / "ckpt")
        preds_after = pipeline2.predict(self.CLAIMS, retrieve_evidence=False)

        np.testing.assert_array_equal(preds_before, preds_after)

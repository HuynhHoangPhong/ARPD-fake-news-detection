"""
Unit tests for ARPD pipeline components.

Run with:  pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

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
        for col in ("claim", "label", "speaker", "subject"):
            assert col in df.columns, f"Missing '{col}' column"

    def test_binary_labels(self, df):
        unique = set(df["label"].unique())
        assert unique <= {0, 1}, f"Labels not binary: {unique}"

    def test_no_null_claims(self, df):
        assert df["claim"].isna().sum() == 0, "Null claims found"

    def test_no_null_speaker_subject(self, df):
        # NaN must be filled with empty string (not left as NaN)
        assert df["speaker"].isna().sum() == 0, "NaN in speaker after fillna"
        assert df["subject"].isna().sum() == 0, "NaN in subject after fillna"

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
        s.fit_reference([])
        return s

    DIVERSE_CLAIMS = [
        "The unemployment rate fell to 3.5% in October 2023.",
        "Congress passed H.R.1234 by a 218-to-210 vote on Tuesday.",
        "President Obama signed a healthcare bill.",
        "Some people say the government might be hiding information.",
        "Many experts suggest climate policy could possibly change.",
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
        s.fit_reference(["some claim", "another claim"])


# ---------------------------------------------------------------------------
# Test 3: ClaimEvidenceEncoder — including speaker-aware context
# ---------------------------------------------------------------------------

class TestEncoder:
    """encode_batch returns (N, 384); speaker-aware vs claim-only differ."""

    CLAIMS = [
        "The president signed a new bill.",
        "Vaccines cause autism.",
        "NASA confirmed water on Mars.",
    ]
    SPEAKERS = ["Barack Obama", "RFK Jr", "unknown"]
    SUBJECTS = ["health care", "vaccines", "space"]

    @pytest.fixture(scope="class")
    def enc_context(self):
        from src.encoder import ClaimEvidenceEncoder
        return ClaimEvidenceEncoder(use_speaker_context=True)

    @pytest.fixture(scope="class")
    def enc_legacy(self):
        from src.encoder import ClaimEvidenceEncoder
        return ClaimEvidenceEncoder(use_speaker_context=False)

    def test_output_shape_no_evidence(self, enc_context):
        X = enc_context.encode_batch(
            self.CLAIMS, [[] for _ in self.CLAIMS], show_progress=False
        )
        assert X.shape == (3, 384)

    def test_output_shape_with_evidence(self, enc_context):
        passages_list = [
            ["Some relevant political text."],
            ["CDC says vaccines are safe.", "Studies confirm no link."],
            [],
        ]
        X = enc_context.encode_batch(
            self.CLAIMS, passages_list, show_progress=False
        )
        assert X.shape == (3, 384)

    def test_speaker_context_changes_output(self, enc_context):
        """Encoding with speaker prefix must produce a DIFFERENT vector than without."""
        passages = [[] for _ in self.CLAIMS]
        X_no_ctx = enc_context.encode_batch(
            self.CLAIMS, passages, speakers=None, subjects=None, show_progress=False
        )
        X_ctx = enc_context.encode_batch(
            self.CLAIMS, passages,
            speakers=self.SPEAKERS, subjects=self.SUBJECTS,
            show_progress=False,
        )
        # At least one claim should differ meaningfully
        diffs = np.linalg.norm(X_ctx - X_no_ctx, axis=1)
        assert diffs.max() > 1e-3, (
            "Speaker-context encoding identical to no-context encoding; "
            "speaker prefix is not being applied"
        )

    def test_legacy_path_no_evidence_equals_claim(self, enc_legacy):
        claim = "The president signed a new bill."
        X_batch = enc_legacy.encode_batch([claim], [[]], show_progress=False)
        v_claim = enc_legacy.encode_claim(claim)
        np.testing.assert_allclose(X_batch[0], v_claim, rtol=1e-4)

    def test_output_dtype(self, enc_context):
        X = enc_context.encode_batch(
            self.CLAIMS, [[] for _ in self.CLAIMS], show_progress=False
        )
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 4: ImprovedMLP + predict_proba
# ---------------------------------------------------------------------------

class TestImprovedMLP:
    """Forward pass, predict_proba shape, binary predict."""

    @pytest.fixture(scope="class")
    def trainer(self):
        from src.classifier import ARPDTrainer
        return ARPDTrainer(input_dim=384, use_improved=True, device="cpu")

    def test_forward_shape(self, trainer):
        x = torch.randn(16, 384)
        out = trainer.model(x)
        assert out.shape == (16, 2)

    def test_predict_binary(self, trainer):
        X = np.random.randn(20, 384).astype(np.float32)
        preds = trainer.predict(X)
        assert set(preds.tolist()) <= {0, 1}

    def test_predict_proba_shape_and_sums(self, trainer):
        X = np.random.randn(20, 384).astype(np.float32)
        proba = trainer.predict_proba(X)
        assert proba.shape == (20, 2), f"Expected (20,2), got {proba.shape}"
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(20), atol=1e-5)

    def test_fit_runs(self, trainer):
        np.random.seed(0)
        X_tr = np.random.randn(100, 384).astype(np.float32)
        y_tr = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 384).astype(np.float32)
        y_val = np.random.randint(0, 2, 20)
        history = trainer.fit(X_tr, y_tr, X_val, y_val, epochs=2, verbose=False)
        assert len(history) == 2
        assert "f1_macro" in history[0]

    def test_xavier_init(self):
        from src.classifier import ImprovedMLP
        model = ImprovedMLP(input_dim=384)
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                assert param.abs().max().item() < 5.0, f"Suspicious weight in {name}"


# ---------------------------------------------------------------------------
# Test 5: Ensemble formula correctness
# ---------------------------------------------------------------------------

class TestEnsembleFormula:
    """(1-w)*P_lr + w*P_mlp gives correct convex combination."""

    def test_ensemble_formula_w0(self):
        """w=0.0 -> pure LR."""
        prob_lr  = np.array([[0.6, 0.4], [0.3, 0.7]])
        prob_mlp = np.array([[0.5, 0.5], [0.5, 0.5]])
        w = 0.0
        result = (1 - w) * prob_lr + w * prob_mlp
        np.testing.assert_allclose(result, prob_lr)

    def test_ensemble_formula_w1(self):
        """w=1.0 -> pure MLP."""
        prob_lr  = np.array([[0.6, 0.4]])
        prob_mlp = np.array([[0.2, 0.8]])
        result = (1 - 1.0) * prob_lr + 1.0 * prob_mlp
        np.testing.assert_allclose(result, prob_mlp)

    def test_ensemble_sums_to_one(self):
        """Output rows must sum to 1 for any w in [0, 1]."""
        rng = np.random.default_rng(42)
        N = 10
        for w in [0.05, 0.10, 0.20, 0.30]:
            p_lr  = rng.dirichlet([1, 1], size=N)
            p_mlp = rng.dirichlet([1, 1], size=N)
            combined = (1 - w) * p_lr + w * p_mlp
            np.testing.assert_allclose(
                combined.sum(axis=1), np.ones(N), atol=1e-6,
                err_msg=f"Ensemble probabilities don't sum to 1 at w={w}"
            )

    def test_ensemble_default_weight_in_valid_range(self):
        from src.pipeline import _ENSEMBLE_WEIGHTS
        assert 0.10 in _ENSEMBLE_WEIGHTS, "Default weight 0.10 should be in grid"
        for w in _ENSEMBLE_WEIGHTS:
            assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Test 6: AdaptiveRetriever — User-Agent header regression test
# ---------------------------------------------------------------------------

class TestRetrieverUserAgent:
    """AdaptiveRetriever must send User-Agent header on all requests.

    NOTE: bản tối ưu tốc độ (2026-06) dùng 1 requests.Session dùng chung với
    header set sẵn (session.headers.update), thay vì truyền headers= ở mỗi
    lệnh gọi requests.get() rời rạc như bản gốc. Test này kiểm tra đúng cơ
    chế mới: session phải có User-Agent, và mọi call thực tế đi qua session đó.
    """

    def test_session_has_user_agent_header(self):
        from src.adaptive_retriever import AdaptiveRetriever

        retriever = AdaptiveRetriever()
        assert "User-Agent" in retriever._session.headers, (
            "AdaptiveRetriever._session phải có User-Agent header sẵn. "
            "Wikipedia API trả về HTTP 403 nếu thiếu."
        )
        assert "ARPD-Research" in retriever._session.headers["User-Agent"]

    def test_fetch_passages_uses_session_not_bare_requests(self):
        from src.adaptive_retriever import AdaptiveRetriever

        retriever = AdaptiveRetriever(max_workers=1, sleep_between=0.0)

        with patch.object(retriever._session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "query": {"search": [{"title": "Obama"}]}
            }
            mock_get.return_value = mock_response

            try:
                retriever._fetch_passages("Obama healthcare bill", srlimit=3)
            except Exception:
                pass  # we only care the call went through the session

            assert mock_get.called, (
                "_fetch_passages() phải gọi qua self._session.get() "
                "(đã có User-Agent sẵn trong session headers), không tạo "
                "request trần requests.get() mới."
            )

# ---------------------------------------------------------------------------
# Test 7: Paraphrase augmentor
# ---------------------------------------------------------------------------

class TestAugmentor:
    """synonym_substitute changes >= 1 token; combined_augment differs."""

    LONG_SENTENCE = (
        "The federal government announced major economic reforms affecting "
        "millions of workers across the entire national territory today."
    )

    def setup_method(self):
        from src.paraphrase_augmentor import ensure_nltk_data
        ensure_nltk_data()

    def test_synonym_changes_tokens(self):
        from src.paraphrase_augmentor import synonym_substitute
        original_tokens = self.LONG_SENTENCE.split()
        changed = any(
            synonym_substitute(self.LONG_SENTENCE, p=0.5, seed=s).split() != original_tokens
            for s in range(10)
        )
        assert changed, "synonym_substitute never changed any token across 10 seeds"

    def test_random_deletion_reduces_length(self):
        from src.paraphrase_augmentor import random_deletion
        result = random_deletion(self.LONG_SENTENCE, p=0.5, seed=0)
        assert len(result.split()) < len(self.LONG_SENTENCE.split())

    def test_combined_augment_differs(self):
        from src.paraphrase_augmentor import combined_augment
        result = combined_augment(self.LONG_SENTENCE, seed=7)
        assert result != self.LONG_SENTENCE

    def test_short_sentence_deletion_safe(self):
        from src.paraphrase_augmentor import random_deletion
        result = random_deletion("Yes.", p=0.9, seed=0)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Test 8: End-to-end pipeline — MLP-only (no ensemble, no retrieval)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Pipeline fits and predicts without error on tiny synthetic data."""

    CLAIMS = [
        "The president signed a new healthcare bill.",
        "Vaccines cause autism according to researchers.",
        "The unemployment rate fell to 3.5% last quarter.",
        "Some people believe the moon landing was faked.",
        "Congress passed a bipartisan infrastructure package.",
    ]
    LABELS = [1, 0, 1, 0, 1]
    SPEAKERS = ["Obama", "RFK", "Biden", "unknown", "Congress"]
    SUBJECTS = ["health", "vaccines", "economy", "conspiracy", "infrastructure"]

    def test_pipeline_fit_predict_no_ensemble(self):
        from src.pipeline import ARPDPipeline

        pipeline = ARPDPipeline(
            augmentation_method="synonym",
            use_ensemble=False,
            device="cpu",
        )
        history = pipeline.fit(
            self.CLAIMS * 20, self.LABELS * 20,
            self.CLAIMS, self.LABELS,
            train_speakers=self.SPEAKERS * 20,
            train_subjects=self.SUBJECTS * 20,
            val_speakers=self.SPEAKERS,
            val_subjects=self.SUBJECTS,
            epochs=2, verbose=False,
        )
        assert len(history) > 0

        preds = pipeline.predict(
            self.CLAIMS,
            speakers=self.SPEAKERS,
            subjects=self.SUBJECTS,
        )
        assert preds.shape == (5,)
        assert set(preds.tolist()) <= {0, 1}

    def test_pipeline_save_load(self, tmp_path):
        from src.pipeline import ARPDPipeline

        pipeline = ARPDPipeline(device="cpu", use_ensemble=False)
        pipeline.fit(
            self.CLAIMS * 10, self.LABELS * 10,
            self.CLAIMS, self.LABELS,
            epochs=1, verbose=False,
        )
        pipeline.save(tmp_path / "ckpt")
        preds_before = pipeline.predict(self.CLAIMS)

        pipeline2 = ARPDPipeline(device="cpu", use_ensemble=False)
        pipeline2.scorer.fit_reference([])
        pipeline2.load(tmp_path / "ckpt")
        preds_after = pipeline2.predict(self.CLAIMS)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_pipeline_save_load_ensemble_state_restored(self, tmp_path):
        """
        Bug F regression: save() + load() must restore TF-IDF, LR, ensemble_weight,
        and config flags.  Loading into a pipeline with WRONG constructor flags
        must be overridden by the stored state.
        """
        from src.pipeline import ARPDPipeline

        # Fit with ensemble enabled and speaker context on (defaults)
        pipeline = ARPDPipeline(
            device="cpu",
            use_speaker_context=True,
            use_ensemble=True,
            grid_search_weight=True,
        )
        pipeline.fit(
            self.CLAIMS * 20, self.LABELS * 20,
            self.CLAIMS, self.LABELS,
            train_speakers=self.SPEAKERS * 20,
            train_subjects=self.SUBJECTS * 20,
            val_speakers=self.SPEAKERS,
            val_subjects=self.SUBJECTS,
            epochs=2, verbose=False,
        )
        saved_weight = pipeline.ensemble_weight
        preds_before = pipeline.predict(
            self.CLAIMS, speakers=self.SPEAKERS, subjects=self.SUBJECTS
        )
        pipeline.save(tmp_path / "ckpt")

        # Load into pipeline constructed with WRONG flags — load() must override them
        pipeline2 = ARPDPipeline(
            device="cpu",
            use_speaker_context=False,   # wrong — should be overridden to True
            use_ensemble=False,           # wrong — should be overridden to True
        )
        pipeline2.scorer.fit_reference([])
        pipeline2.load(tmp_path / "ckpt")

        # Flags restored from state
        assert pipeline2.use_speaker_context is True, (
            "use_speaker_context was not restored from pipeline_state.pkl"
        )
        assert pipeline2.use_ensemble is True, (
            "use_ensemble was not restored from pipeline_state.pkl"
        )
        assert pipeline2.ensemble_weight == saved_weight, (
            f"ensemble_weight mismatch: got {pipeline2.ensemble_weight}, "
            f"expected {saved_weight}"
        )
        # LR must be fitted (has coef_ attribute)
        assert hasattr(pipeline2.lr, "coef_"), (
            "LogisticRegression was not restored (NotFittedError would occur on predict)"
        )

        # Predictions must be identical
        preds_after = pipeline2.predict(
            self.CLAIMS, speakers=self.SPEAKERS, subjects=self.SUBJECTS
        )
        np.testing.assert_array_equal(
            preds_before, preds_after,
            err_msg="Predictions differ after save/load — ensemble state not fully restored",
        )

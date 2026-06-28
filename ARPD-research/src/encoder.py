"""
Lightweight Encoder — dùng MiniLM-L6-v2 (22M params).

Hai chế độ encoding (chọn qua use_speaker_context):

  use_speaker_context=True  (mặc định — khuyến nghị cho ARPD):
    Concatenate text rồi encode một lần:
      "[speaker] [subject] claim [SEP] evidence_text"  -> (N, 384)
    Đồng nhất với notebook Colab (add_full_context + encode_pairs).
    Ablation "No Speaker" -> truyền speakers=None, subjects=None.

  use_speaker_context=False  (legacy — chỉ dùng cho ablation so sánh):
    Weighted sum riêng: 0.7 * v_claim + 0.3 * mean(v_evidence) -> (N, 384)
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ENCODER_DIM = 384   # all-MiniLM-L6-v2 output dimension
COMBINED_DIM = ENCODER_DIM
CLAIM_WEIGHT = 0.7
EVIDENCE_WEIGHT = 0.3
_MAX_EVIDENCE_PASSAGES = 3  # cap to keep token count manageable


class ClaimEvidenceEncoder:
    """Encode (claim, evidence_list) -> 384-dim feature vector."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
        use_speaker_context: bool = True,
    ) -> None:
        """
        Args:
            model_name: SentenceTransformer model name.
            device: "cuda" | "cpu" | None (auto-detect).
            batch_size: Batch size when encoding.
            use_speaker_context: True -> encode "[speaker] [subject] claim [SEP] ev"
                                  as one string.  False -> weighted-sum of separate
                                  claim + evidence embeddings (legacy ablation mode).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.use_speaker_context = use_speaker_context
        self.model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Primary API: speaker-aware context encoding
    # ------------------------------------------------------------------

    def encode_batch_context(
        self,
        claims: list[str],
        passages_list: list[list[str]],
        speakers: list[str] | None = None,
        subjects: list[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Speaker-aware encoding: format and encode as a single string.

        Format: "[speaker] [subject] claim [SEP] evidence" (if evidence exists)
                "[speaker] [subject] claim"                 (if no evidence)
        Without speakers/subjects, reduces to "claim [SEP] evidence".

        Args:
            claims:        N claim strings.
            passages_list: N lists of evidence passages.
            speakers:      N speaker strings or None (disables speaker prefix).
            subjects:      N subject strings or None (disables subject prefix).
            show_progress: Show tqdm progress bar.

        Returns:
            numpy array (N, 384).
        """
        texts = []
        for i, (claim, passages) in enumerate(zip(claims, passages_list)):
            # Build speaker-subject prefix
            prefix = ""
            if speakers is not None and subjects is not None:
                spk = (speakers[i] or "").strip() or "unknown"
                sub = (subjects[i] or "").strip() or "unknown"
                prefix = f"[{spk}] [{sub}] "

            # Join up to _MAX_EVIDENCE_PASSAGES passages
            ev_text = " ".join(passages[:_MAX_EVIDENCE_PASSAGES]).strip() if passages else ""

            if ev_text:
                texts.append(f"{prefix}{claim} [SEP] {ev_text}")
            else:
                texts.append(f"{prefix}{claim}")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )  # (N, 384)

    # ------------------------------------------------------------------
    # Batch encoding (dispatches based on use_speaker_context)
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        claims: list[str],
        passages_list: list[list[str]],
        speakers: list[str] | None = None,
        subjects: list[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch. If use_speaker_context=True, calls encode_batch_context
        (single-string format). Otherwise uses legacy weighted-sum path.

        Returns: (N, 384) numpy array.
        """
        if self.use_speaker_context:
            return self.encode_batch_context(
                claims, passages_list,
                speakers=speakers, subjects=subjects,
                show_progress=show_progress,
            )
        return self._encode_batch_weighted_sum(claims, passages_list, show_progress)

    def _encode_batch_weighted_sum(
        self,
        claims: list[str],
        passages_list: list[list[str]],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Legacy: 0.7 * v_claim + 0.3 * mean(v_evidence). Used for ablation only."""
        claim_embs = self.model.encode(
            claims,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )  # (N, 384)

        out = np.empty_like(claim_embs)
        for i, passages in enumerate(passages_list):
            if passages:
                ev_embs = self.model.encode(
                    passages, convert_to_numpy=True, batch_size=self.batch_size
                )
                out[i] = CLAIM_WEIGHT * claim_embs[i] + EVIDENCE_WEIGHT * ev_embs.mean(axis=0)
            else:
                out[i] = claim_embs[i]
        return out  # (N, 384)

    # ------------------------------------------------------------------
    # Single-pair helpers (kept for backward compat / unit tests)
    # ------------------------------------------------------------------

    def encode_claim(self, claim: str) -> np.ndarray:
        return self.model.encode(claim, convert_to_numpy=True)

    def encode_evidence(self, passages: list[str]) -> np.ndarray | None:
        if not passages:
            return None
        embs = self.model.encode(passages, convert_to_numpy=True, batch_size=self.batch_size)
        return embs.mean(axis=0)

    def encode_pair(self, claim: str, passages: list[str]) -> np.ndarray:
        v_claim = self.encode_claim(claim)
        v_ev = self.encode_evidence(passages)
        if v_ev is None:
            return v_claim
        return CLAIM_WEIGHT * v_claim + EVIDENCE_WEIGHT * v_ev


if __name__ == "__main__":
    enc = ClaimEvidenceEncoder(use_speaker_context=True)
    claims = ["The president signed a new healthcare bill."]
    passages_list = [["The White House announced a major healthcare reform on Tuesday."]]
    speakers = ["Barack Obama"]
    subjects = ["health care"]

    out = enc.encode_batch_context(claims, passages_list, speakers, subjects, show_progress=False)
    assert out.shape == (1, 384), f"Expected (1,384), got {out.shape}"
    print(f"Context-aware: {out.shape}  OK")

    out2 = enc.encode_batch(claims, passages_list, speakers, subjects, show_progress=False)
    assert out2.shape == (1, 384)
    print(f"encode_batch (dispatcher): {out2.shape}  OK")

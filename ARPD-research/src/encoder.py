"""
Lightweight Encoder — dùng MiniLM-L6-v2 (22M params).

Input:  claim + list[evidence_passages]
Output: fixed-size vector (384-dim) để đưa vào MLP classifier.

Chiến lược kết hợp claim và evidence:
  - Nếu evidence rỗng (không có passage nào vượt similarity threshold):
      output = v_claim  (384-dim)
  - Nếu có evidence:
      output = 0.7 * v_claim + 0.3 * mean(v_evidence_passages)  (384-dim)

Lý do đổi từ concat 1152-dim sang weighted sum 384-dim:
  - Input dimension không đổi theo số evidence → MLP ổn định hơn.
  - Claim chiếm trọng số cao hơn vì với LIAR, evidence Wikipedia
    thường nhiễu; claim là signal chính.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ENCODER_DIM = 384   # all-MiniLM-L6-v2 output dimension
COMBINED_DIM = ENCODER_DIM  # output dim không đổi: 384
CLAIM_WEIGHT = 0.7
EVIDENCE_WEIGHT = 0.3


class ClaimEvidenceEncoder:
    """Encode (claim, evidence_list) → feature vector."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        """
        Args:
            model_name: SentenceTransformer model name.
            device: "cuda" | "cpu" | None (auto-detect).
            batch_size: Batch size khi encode.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode_claim(self, claim: str) -> np.ndarray:
        """Encode một claim → (384,) numpy array."""
        return self.model.encode(claim, convert_to_numpy=True)

    def encode_evidence(self, passages: list[str]) -> np.ndarray | None:
        """
        Encode list of passages → (384,) mean-pooled numpy array.
        Trả về None nếu passages rỗng (caller dùng claim-only path).
        """
        if not passages:
            return None
        embs = self.model.encode(passages, convert_to_numpy=True, batch_size=self.batch_size)
        return embs.mean(axis=0)  # (384,)

    def encode_pair(self, claim: str, passages: list[str]) -> np.ndarray:
        """
        Encode (claim, passages) → (384,) feature vector.

        - passages rỗng → v_claim
        - passages có nội dung → 0.7*v_claim + 0.3*mean(v_passages)
        """
        v_claim = self.encode_claim(claim)
        v_evidence = self.encode_evidence(passages)
        if v_evidence is None:
            return v_claim
        return CLAIM_WEIGHT * v_claim + EVIDENCE_WEIGHT * v_evidence  # (384,)

    def encode_batch(
        self,
        claims: list[str],
        passages_list: list[list[str]],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode nhiều (claim, passages) cùng lúc.

        Args:
            claims: List N claims.
            passages_list: List N lists of evidence passages.
            show_progress: In progress bar.

        Returns:
            numpy array (N, 1152).
        """
        assert len(claims) == len(passages_list)

        claim_embs = self.model.encode(
            claims,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )  # (N, 384)

        out = np.empty_like(claim_embs)  # (N, 384)
        for i, passages in enumerate(passages_list):
            if passages:
                ev_embs = self.model.encode(
                    passages, convert_to_numpy=True, batch_size=self.batch_size
                )
                out[i] = CLAIM_WEIGHT * claim_embs[i] + EVIDENCE_WEIGHT * ev_embs.mean(axis=0)
            else:
                out[i] = claim_embs[i]

        return out  # (N, 384)


if __name__ == "__main__":
    encoder = ClaimEvidenceEncoder()

    claim = "The president signed a new healthcare bill."
    passages = [
        "The White House announced a major healthcare reform on Tuesday.",
        "Congress voted on the Affordable Care Act amendment last week.",
    ]

    vec = encoder.encode_pair(claim, passages)
    print(f"Output vector shape: {vec.shape}")  # (1152,)
    print(f"Norm: {np.linalg.norm(vec):.4f}")

    # Batch encoding
    claims = [claim, "Vaccines cause autism according to researchers."]
    passages_list = [passages, ["Multiple studies show vaccines are safe."]]
    batch_vecs = encoder.encode_batch(claims, passages_list, show_progress=False)
    print(f"Batch output shape: {batch_vecs.shape}")  # (2, 1152)

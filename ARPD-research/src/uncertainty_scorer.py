"""
Uncertainty Scorer -- linguistic-based, no embedding needed.

Embedding-based entropy fails on short LIAR claims because all
political claims cluster tightly in MiniLM space, producing
near-uniform similarity distributions -> k=5 for 88% of inputs.

Replacement: five interpretable linguistic signals.

  length_score    = min(len(tokens) / 25.0, 1.0)
  specificity     = (has_number + has_percent + has_dollar) / 3.0
  hedge_score     = min(count(HEDGE_WORDS) / 2.0, 1.0)
  entity_score    = min(count(UPPER_WORDS) / 4.0, 1.0)
  vague_score     = min(count(VAGUE_WORDS) / 2.0, 1.0)
  short_penalty   = 1 if len(tokens) < 8 else 0

  uncertainty = (0.20 * length_score
               + 0.25 * (1 - specificity)
               + 0.15 * hedge_score
               + 0.15 * (1 - entity_score)
               + 0.15 * vague_score
               + 0.10 * short_penalty)

k buckets: <0.20->1, <0.35->2, <0.50->3, <0.65->4, else->5
"""

from __future__ import annotations

import re

import numpy as np


# Words that signal hedging / low confidence
_HEDGE_WORDS = {
    "allegedly", "apparently", "claims", "reportedly", "possibly",
    "perhaps", "maybe", "might", "could", "suggest", "suggests",
    "suggested", "seem", "seems", "likely", "unlikely", "rumored",
    "purportedly", "supposedly", "some", "say", "says",
}

# Vague quantifiers that make a claim hard to verify
_VAGUE_WORDS = {
    "many", "several", "few", "some", "various", "numerous",
    "most", "often", "sometimes", "frequently", "rarely",
    "occasionally", "generally", "largely", "broadly",
}

_NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*\b")
_PERCENT_RE = re.compile(r"\d+\s*%")
_DOLLAR_RE = re.compile(r"\$\s*\d")


class UncertaintyScorer:
    """
    Linguistic uncertainty scorer.

    fit_reference() is kept for interface compatibility but is a no-op;
    this scorer needs no training data.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # unused, kept for compat
        k_min: int = 1,
        k_max: int = 5,
        knn: int = 10,           # unused
        temperature: float = 0.1,  # unused
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        # Kept to avoid breaking callers that pass these args
        _ = model_name, knn, temperature

    def fit_reference(self, reference_claims: list[str]) -> None:
        """No-op -- linguistic scorer needs no reference set."""
        pass

    def score(self, claim: str) -> float:
        """
        Return linguistic uncertainty in [0, 1].
        Higher = more uncertain = needs more evidence passages.
        """
        tokens = claim.split()
        lower = claim.lower()
        lower_tokens = set(lower.split())

        # --- Feature computation ---
        length_score = min(len(tokens) / 25.0, 1.0)

        has_number  = 1.0 if _NUMBER_RE.search(claim) else 0.0
        has_percent = 1.0 if _PERCENT_RE.search(claim) else 0.0
        has_dollar  = 1.0 if _DOLLAR_RE.search(claim) else 0.0
        specificity = (has_number + has_percent + has_dollar) / 3.0

        hedge_count = sum(1 for w in lower_tokens if w in _HEDGE_WORDS)
        hedge_score = min(hedge_count / 2.0, 1.0)

        # Capitalized non-first words as proxy for named entities
        upper_count = sum(
            1 for i, tok in enumerate(tokens)
            if i > 0 and tok and tok[0].isupper() and tok.isalpha()
        )
        entity_score = min(upper_count / 4.0, 1.0)

        vague_count = sum(1 for w in lower_tokens if w in _VAGUE_WORDS)
        vague_score = min(vague_count / 2.0, 1.0)

        short_penalty = 1.0 if len(tokens) < 8 else 0.0

        # --- Weighted combination ---
        uncertainty = (
            0.20 * length_score
            + 0.25 * (1.0 - specificity)
            + 0.15 * hedge_score
            + 0.15 * (1.0 - entity_score)
            + 0.15 * vague_score
            + 0.10 * short_penalty
        )
        return float(np.clip(uncertainty, 0.0, 1.0))

    def compute_k(self, claim: str) -> int:
        """Return k_adaptive in [k_min, k_max] based on uncertainty score."""
        u = self.score(claim)
        if u < 0.20:
            k = 1
        elif u < 0.35:
            k = 2
        elif u < 0.50:
            k = 3
        elif u < 0.65:
            k = 4
        else:
            k = 5
        return int(np.clip(k, self.k_min, self.k_max))

    def batch_compute_k(self, claims: list[str]) -> list[int]:
        return [self.compute_k(c) for c in claims]


if __name__ == "__main__":
    test_claims = [
        # Should be low uncertainty (k=1 or 2): specific, numeric
        "The unemployment rate fell to 3.5% in October 2023.",
        "Congress passed H.R.1234 by a 218-to-210 vote on Tuesday.",
        # Should be medium (k=3): moderate specificity
        "President Obama signed a healthcare bill into law.",
        "Senator McCain voted against the tax reform measure.",
        # Should be high uncertainty (k=4 or 5): vague/hedging
        "Some people say the government might be hiding information.",
        "Many experts suggest climate policy could possibly change.",
        # Short/vague
        "Taxes bad.",
    ]

    scorer = UncertaintyScorer(k_min=1, k_max=5)
    scorer.fit_reference([])  # no-op, should not raise

    print(f"{'k':>3}  {'u':>5}  Claim")
    print("-" * 70)
    k_values = []
    for claim in test_claims:
        u = scorer.score(claim)
        k = scorer.compute_k(claim)
        k_values.append(k)
        print(f"  {k}  {u:.3f}  {claim[:65]}")

    unique_k = len(set(k_values))
    print(f"\nUnique k values: {unique_k} (need >= 3 across diverse claims)")
    assert unique_k >= 3, "Scorer not diverse enough!"
    print("PASSED")

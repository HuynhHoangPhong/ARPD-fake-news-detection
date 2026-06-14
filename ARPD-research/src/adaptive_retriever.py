"""
Adaptive Retriever — lấy Wikipedia evidence passages.

Chiến lược:
  1. Dùng TF-IDF keyword extraction để tạo query từ claim.
  2. Tìm kiếm Wikipedia, lấy top-k_adaptive articles.
  3. Chunking: chia summary thành passages ~100 từ.
  4. Nếu Wikipedia fail (timeout, không tìm thấy), trả về danh sách rỗng
     thay vì crash — pipeline vẫn chạy được.

Lưu ý: wikipedia-api có rate limit; dùng cache đơn giản để tránh gọi lại.
"""

from __future__ import annotations

import time

import numpy as np
import wikipediaapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _extract_keywords(claim: str, top_n: int = 5) -> str:
    """Dùng TF-IDF unigram để lấy top-n từ quan trọng nhất làm query."""
    # Fallback nếu claim quá ngắn
    words = claim.split()
    if len(words) <= top_n:
        return claim

    vec = TfidfVectorizer(stop_words="english", max_features=top_n)
    try:
        vec.fit([claim])
        keywords = list(vec.vocabulary_.keys())
    except Exception:
        keywords = words[:top_n]
    return " ".join(keywords)


def _chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    """Chia text thành các đoạn ~chunk_size từ."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


class AdaptiveRetriever:
    """Truy vấn Wikipedia và trả về evidence passages đã lọc theo similarity."""

    def __init__(
        self,
        language: str = "en",
        chunk_size: int = 100,
        sleep_between: float = 0.5,
        sim_threshold: float = 0.25,
        encoder=None,
    ) -> None:
        """
        Args:
            language: Ngôn ngữ Wikipedia.
            chunk_size: Số từ mỗi passage chunk.
            sleep_between: Giây nghỉ giữa các API call (tránh rate limit).
            sim_threshold: Cosine similarity tối thiểu để giữ passage.
            encoder: SentenceTransformer instance (hoặc None để lazy-load MiniLM).
        """
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="ARPD-Research/1.0 (phong.huynhhoang.work@gmail.com)",
        )
        self.chunk_size = chunk_size
        self.sleep_between = sleep_between
        self.sim_threshold = sim_threshold
        self._encoder = encoder
        self._cache: dict[str, list[str]] = {}

    def _get_encoder(self):
        """Lazy-load MiniLM nếu chưa có encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._encoder

    def _filter_by_similarity(
        self, claim: str, passages: list[str]
    ) -> list[str]:
        """
        Giữ lại passages có cosine similarity với claim >= sim_threshold.
        Trả về list rỗng nếu không có passage nào pass.
        """
        if not passages:
            return []
        enc = self._get_encoder()
        claim_emb = enc.encode([claim])                          # (1, D)
        passage_embs = enc.encode(passages)                      # (P, D)
        sims = cosine_similarity(claim_emb, passage_embs)[0]    # (P,)
        return [p for p, s in zip(passages, sims) if s >= self.sim_threshold]

    def _fetch_passages(self, query: str) -> list[str]:
        """Fetch Wikipedia summary và chunk thành passages."""
        if query in self._cache:
            return self._cache[query]

        try:
            page = self.wiki.page(query)
            if not page.exists():
                # Thử lại với query ngắn hơn (lấy từ đầu)
                short_query = " ".join(query.split()[:3])
                page = self.wiki.page(short_query)

            if page.exists():
                passages = _chunk_text(page.summary, self.chunk_size)
            else:
                passages = []
        except Exception:
            passages = []

        time.sleep(self.sleep_between)
        self._cache[query] = passages
        return passages

    def retrieve(self, claim: str, k: int) -> list[str]:
        """
        Lấy tối đa k evidence passages từ Wikipedia cho claim,
        sau đó lọc bỏ passages không liên quan (sim < threshold).

        Args:
            claim: Câu cần kiểm tra.
            k: Số passages cần lấy (từ k_adaptive).

        Returns:
            List passages đã lọc, tối đa k phần tử.
            Trả về [] nếu không có passage nào vượt ngưỡng similarity.
        """
        query = _extract_keywords(claim, top_n=5)
        passages = self._fetch_passages(query)

        if not passages:
            passages = self._fetch_passages(claim[:100])

        # Lọc theo cosine similarity trước khi cắt top-k
        filtered = self._filter_by_similarity(claim, passages)
        return filtered[:k]

    def retrieve_batch(
        self, claims: list[str], k_list: list[int]
    ) -> list[list[str]]:
        """
        Retrieve cho nhiều claims, mỗi claim có k riêng.

        Args:
            claims: List claims.
            k_list: List k tương ứng.

        Returns:
            List of list of passages.
        """
        assert len(claims) == len(k_list)
        return [self.retrieve(c, k) for c, k in zip(claims, k_list)]


if __name__ == "__main__":
    retriever = AdaptiveRetriever()

    test_cases = [
        ("The president of the United States signed a healthcare bill.", 2),
        ("Vaccines cause autism.", 3),
    ]

    for claim, k in test_cases:
        passages = retriever.retrieve(claim, k)
        print(f"Claim: {claim[:70]}")
        print(f"  k={k}, retrieved={len(passages)} passages")
        for i, p in enumerate(passages):
            print(f"  [{i+1}] {p[:120]}...")
        print()

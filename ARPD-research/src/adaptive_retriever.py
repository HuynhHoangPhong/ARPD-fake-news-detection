"""
Adaptive Retriever — lấy Wikipedia evidence passages.

Chiến lược:
  1. Dùng TF-IDF keyword extraction để tạo query từ claim.
  2. Tìm kiếm Wikipedia, lấy top-k_adaptive articles.
  3. Chunking: chia summary thành passages ~100 từ.
  4. Nếu Wikipedia fail (timeout, không tìm thấy), trả về danh sách rỗng
     thay vì crash — pipeline vẫn chạy được.

PERF NOTE (2026-06 speed fix):
  Bản gốc dùng `wikipediaapi.Wikipedia().page(title).summary` — mỗi page là
  1 lượt parse HTML/wikitext nặng qua thư viện wikipediaapi. Bản này gọi
  trực tiếp REST endpoint `/api/rest_v1/page/summary/{title}` (chỉ trả JSON
  nhẹ chứa plaintext summary, không cần parse) — giảm đáng kể thời gian
  mỗi request. Đồng thời các page-summary fetch của CÙNG một claim được
  chạy SONG SONG qua ThreadPoolExecutor (I/O-bound nên threading hiệu quả),
  thay vì tuần tự + sleep sau mỗi page. retrieve_batch() cũng song song hoá
  giữa nhiều claims với cùng 1 executor dùng chung.

  Logic khoa học (TF-IDF keyword extraction, similarity filter, k_adaptive,
  thứ tự ưu tiên evidence) GIỮ NGUYÊN — chỉ đổi cách gọi mạng.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
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
        max_workers: int = 16,
    ) -> None:
        """
        Args:
            language: Ngôn ngữ Wikipedia.
            chunk_size: Số từ mỗi passage chunk.
            sleep_between: Giây nghỉ giữa các API call khi chạy TUẦN TỰ
                (giữ để tương thích CLI cũ --sleep). Khi max_workers > 1,
                các page-summary fetch của 1 claim chạy song song nên
                sleep_between không áp dụng giữa chúng nữa; nó chỉ được
                dùng như backoff cơ bản khi gặp lỗi 429/503.
            sim_threshold: Cosine similarity tối thiểu để giữ passage.
            encoder: SentenceTransformer instance (hoặc None để lazy-load MiniLM).
            max_workers: Số thread song song tối đa khi fetch nhiều page
                summary cho 1 claim. 1 = tuần tự (tương đương bản cũ).
        """
        self.language = language
        self._ua_headers = {"User-Agent": "ARPD-Research/1.0 (phong.huynhhoang.work@gmail.com)"}
        self.chunk_size = chunk_size
        self.sleep_between = sleep_between
        self.sim_threshold = sim_threshold
        self._encoder = encoder
        self.max_workers = max(1, max_workers)
        self._cache: dict[str, list[str]] = {}
        # 1 Session dùng chung -> tận dụng connection pooling (HTTP keep-alive)
        self._session = requests.Session()
        self._session.headers.update(self._ua_headers)

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

    def _get_summary_text(self, title: str) -> str:
        """
        Lấy plaintext summary của 1 page qua REST API (nhẹ hơn nhiều so với
        wikipediaapi.page().summary, vì server chỉ trả JSON nhỏ thay vì
        parse toàn bộ page).

        Retry với backoff ngắn khi gặp 429 (rate limit) / 503; trả về ""
        nếu page không tồn tại (404) hoặc fail sau retry.
        """
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        backoff = max(self.sleep_between, 0.2)
        for attempt in range(3):
            try:
                resp = self._session.get(url, timeout=5)
            except requests.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    return ""
                return data.get("extract", "") or ""
            if resp.status_code == 404:
                return ""
            if resp.status_code in (429, 503):
                # Rate-limited or server busy — back off and retry.
                time.sleep(backoff)
                backoff *= 2
                continue
            # Other errors (4xx/5xx) — don't retry.
            return ""
        return ""

    def _fetch_passages(self, query: str, srlimit: int = 5) -> list[str]:
        """
        Uses Wikipedia's Search API to find valid page titles matching the query,
        then extracts and chunks summaries dynamically based on srlimit.

        Page-summary fetches for the discovered titles run in PARALLEL via a
        thread pool (I/O-bound network calls), instead of one-by-one with a
        sleep after each — this is the main speed fix vs. the original
        implementation.
        """
        if not query.strip():
            return []

        # 1. Search Wikipedia for the closest matching page titles
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": srlimit,  # Dynamically fetch exactly how many articles the scorer requested
        }

        try:
            response = self._session.get(search_url, params=search_params, timeout=5)
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
        except Exception as e:
            print(f"Search API request failed for query '{query}': {e}")
            return []

        if not search_results:
            return []

        titles = [item["title"] for item in search_results]

        # 2. Extract summaries from the valid page titles discovered — in parallel.
        all_passages: list[str] = []
        if self.max_workers <= 1:
            for title in titles:
                text = self._get_summary_text(title)
                if text:
                    all_passages.extend(_chunk_text(text, chunk_size=self.chunk_size))
                time.sleep(self.sleep_between)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(titles))) as ex:
                future_to_title = {ex.submit(self._get_summary_text, t): t for t in titles}
                for future in as_completed(future_to_title):
                    text = future.result()
                    if text:
                        all_passages.extend(_chunk_text(text, chunk_size=self.chunk_size))

        return all_passages

    # Over-fetch candidate articles before similarity filtering.
    # Diagnosis showed that fetching only k articles gives too few chunks to
    # filter down to k high-quality passages; 10 gives a good candidate pool.
    _CANDIDATE_SRLIMIT = 10

    def retrieve(self, claim: str, k: int) -> list[str]:
        """
        Main entry point for evidence retrieval.
        Extracts keywords, executes dynamic Wikipedia search, and filters by semantic similarity.
        """
        query = _extract_keywords(claim, top_n=5)

        # Over-fetch candidates; similarity filter then narrows to k
        passages = self._fetch_passages(query, srlimit=self._CANDIDATE_SRLIMIT)

        # Fallback to the raw claim if TF-IDF keyword extraction returned a dead end
        if not passages:
            passages = self._fetch_passages(claim[:100], srlimit=self._CANDIDATE_SRLIMIT)

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
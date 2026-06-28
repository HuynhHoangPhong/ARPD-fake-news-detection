"""
Adaptive Retriever — lấy Wikipedia evidence passages (Optimized Batch API).

Chiến lược:
  1. Dùng TF-IDF keyword extraction để tạo query từ claim.
  2. Tìm kiếm Wikipedia, lấy top-k_adaptive articles (1 API call).
  3. Lấy plaintext summary của TẤT CẢ articles trong 1 API call duy nhất 
     thông qua prop=extracts (loại bỏ hoàn toàn threading và rủi ro rate-limit).
  4. Chunking & lọc similarity.
"""

from __future__ import annotations

import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _extract_keywords(claim: str, top_n: int = 5) -> str:
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
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


class AdaptiveRetriever:
    def __init__(
        self,
        language: str = "en",
        chunk_size: int = 100,
        sim_threshold: float = 0.25,
        encoder=None,
    ) -> None:
        self.language = language
        self._ua_headers = {"User-Agent": "ARPD-Research/1.1 (phong.huynhhoang.work@gmail.com) BatchMode"}
        self.chunk_size = chunk_size
        self.sim_threshold = sim_threshold
        self._encoder = encoder
        
        # Connection pooling
        self._session = requests.Session()
        self._session.headers.update(self._ua_headers)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._encoder

    def _filter_by_similarity(self, claim: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        enc = self._get_encoder()
        claim_emb = enc.encode([claim])                          
        passage_embs = enc.encode(passages)                      
        sims = cosine_similarity(claim_emb, passage_embs)[0]    
        return [p for p, s in zip(passages, sims) if s >= self.sim_threshold]

    def _fetch_passages(self, query: str, srlimit: int = 10) -> list[str]:
        if not query.strip():
            return []

        # 1. Search API: Find titles
        api_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": srlimit,
        }

        try:
            resp = self._session.get(api_url, params=search_params, timeout=5)
            data = resp.json()
            search_results = data.get("query", {}).get("search", [])
        except Exception:
            return []

        if not search_results:
            return []

        titles = [item["title"] for item in search_results]
        titles_str = "|".join(titles) # Format for batch request

        # 2. Extract API: Fetch all summaries in ONE network request
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,        # Only the intro summary
            "explaintext": 1,    # Plain text, no HTML parsing needed
            "titles": titles_str,
            "format": "json"
        }

        all_passages = []
        try:
            resp = self._session.get(api_url, params=extract_params, timeout=10)
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page_info in pages.items():
                text = page_info.get("extract", "")
                if text:
                    all_passages.extend(_chunk_text(text, chunk_size=self.chunk_size))
        except Exception as e:
            pass # Failsafe for pipeline survival

        return all_passages

    _CANDIDATE_SRLIMIT = 10

    def retrieve(self, claim: str, k: int) -> list[str]:
        query = _extract_keywords(claim, top_n=5)
        passages = self._fetch_passages(query, srlimit=self._CANDIDATE_SRLIMIT)

        if not passages:
            passages = self._fetch_passages(claim[:100], srlimit=self._CANDIDATE_SRLIMIT)

        filtered = self._filter_by_similarity(claim, passages)
        return filtered[:k]

    def retrieve_batch(self, claims: list[str], k_list: list[int]) -> list[list[str]]:
        assert len(claims) == len(k_list)
        return [self.retrieve(c, k) for c, k in zip(claims, k_list)]

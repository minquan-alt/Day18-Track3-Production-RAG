"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

# ruff: noqa: E402

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    except Exception:
        return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in segment_vietnamese(text).split() if t.strip()]


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokens(chunk.get("text", "")) for chunk in chunks]
        if not chunks:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(doc.get("text", ""), float(scores[i]), doc.get("metadata", {}), "bm25"))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._memory_docs: list[dict] = []
        self._memory_vectors: list[Counter] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._memory_docs = chunks
        self._memory_vectors = [Counter(_tokens(c.get("text", ""))) for c in chunks]
        if not chunks or self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            vectors = self._get_encoder().encode(
                [c.get("text", "") for c in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i].get("text", "")},
                )
                for i, vector in enumerate(vectors)
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            self.client = None

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is not None:
            try:
                query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
                hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=hit.payload.get("text", ""),
                        score=float(hit.score),
                        metadata={k: v for k, v in hit.payload.items() if k != "text"},
                        method="dense",
                    )
                    for hit in hits
                ]
            except Exception:
                self.client = None

        query_vector = Counter(_tokens(query))
        scored = [
            (i, _cosine(query_vector, doc_vector))
            for i, doc_vector in enumerate(self._memory_vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        results = []
        for i, score in scored[:top_k]:
            doc = self._memory_docs[i]
            results.append(SearchResult(doc.get("text", ""), float(score), doc.get("metadata", {}), "dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            item = fused.setdefault(result.text, {"score": 0.0, "result": result})
            item["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

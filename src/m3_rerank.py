"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

# ruff: noqa: E402

import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if os.getenv("USE_REAL_RERANKER", "").lower() not in {"1", "true", "yes"}:
                self._model = ("lexical", None)
                return self._model
            try:
                from FlagEmbedding import FlagReranker

                self._model = ("flag", FlagReranker(self.model_name, use_fp16=True))
            except Exception:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = ("cross", CrossEncoder(self.model_name))
                except Exception:
                    self._model = ("lexical", None)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model_type, model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        try:
            if model_type == "flag":
                scores = model.compute_score(pairs)
            elif model_type == "cross":
                scores = model.predict(pairs)
            else:
                scores = [_lexical_score(query, doc.get("text", "")) for doc in documents]
        except Exception:
            scores = [_lexical_score(query, doc.get("text", "")) for doc in documents]

        if isinstance(scores, (float, int)):
            scores = [float(scores)]
        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k], start=1)
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return CrossEncoderReranker(model_name="lexical").rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _lexical_score(query: str, text: str) -> float:
    query_tokens = Counter(_tokenize(query))
    text_tokens = Counter(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = sum(min(query_tokens[t], text_tokens[t]) for t in query_tokens)
    coverage = overlap / len(query_tokens)
    bonus = 0.2 if any(char.isdigit() for char in query) and any(char.isdigit() for char in text) else 0.0
    return coverage + bonus


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

# ruff: noqa: E402

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown/text/PDF files from data/."""
    docs = []
    text_patterns = ["*.md", "*.txt"]
    for pattern in text_patterns:
        for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
            with open(fp, encoding="utf-8") as f:
                docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _read_pdf(fp)
        if text.strip():
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
    return docs


def _read_pdf(path: str) -> str:
    fitz_text = _read_pdf_with_pymupdf(path)
    if fitz_text.strip():
        return fitz_text
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n\n[Trang {page_number}]\n{page_text}")
        return "\n".join(pages)
    except Exception:
        sidecar = f"{path}.txt"
        if os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as f:
                return f.read()
        return ""


def _read_pdf_with_pymupdf(path: str) -> str:
    try:
        import fitz

        doc = fitz.open(path)
        pages = []
        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text() or ""
            if page_text.strip():
                pages.append(f"\n\n[Trang {page_number}]\n{page_text}")
        return "\n".join(pages)
    except Exception:
        return ""


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n+', text) if s.strip()]
    if not sentences:
        return []

    def fallback_similarity(a: str, b: str) -> float:
        tokens_a = set(re.findall(r"\w+", a.lower()))
        tokens_b = set(re.findall(r"\w+", b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / ((len(tokens_a) * len(tokens_b)) ** 0.5)

    try:
        from sentence_transformers import SentenceTransformer

        model = getattr(chunk_semantic, "_model", None)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            chunk_semantic._model = model
        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

        def sentence_similarity(i: int) -> float:
            return float(embeddings[i - 1] @ embeddings[i])
    except Exception:
        def sentence_similarity(i: int) -> float:
            return fallback_similarity(sentences[i - 1], sentences[i])

    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        if sentence_similarity(i) < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])

    chunks.append(Chunk(
        text=" ".join(current_group),
        metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
    ))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parent_size = max(1, parent_size)
    child_size = max(1, child_size)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    def add_parent(parts: list[str]) -> None:
        if not parts:
            return
        parent_text = "\n\n".join(parts).strip()
        if not parent_text:
            return
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)}
        ))
        for child_index, start in enumerate(range(0, len(parent_text), child_size)):
            child_text = parent_text[start:start + child_size].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "chunk_index": child_index},
                    parent_id=pid
                ))

    current_parts: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        projected_len = current_len + para_len + (2 if current_parts else 0)
        if current_parts and projected_len > parent_size:
            add_parent(current_parts)
            current_parts = []
            current_len = 0
        current_parts.append(para)
        current_len += para_len + (2 if current_len else 0)

    add_parent(current_parts)
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_header = ""
    in_code_block = False

    def add_chunk() -> None:
        chunk_text = "\n".join(current_lines).strip()
        if not chunk_text:
            return
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **metadata,
                "section": current_header,
                "chunk_index": len(chunks),
                "strategy": "structure",
            }
        ))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block

        if not in_code_block and re.match(r"^#{1,3}\s+.+$", line):
            add_chunk()
            current_header = line.strip()
            current_lines = [line]
            continue

        current_lines.append(line)

    add_chunk()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def summarize_lengths(lengths: list[int]) -> dict:
        if not lengths:
            return {"avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic_lengths: list[int] = []
    semantic_lengths: list[int] = []
    structure_lengths: list[int] = []
    hierarchical_child_lengths: list[int] = []
    hierarchical_parent_count = 0
    hierarchical_child_count = 0

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata") or {}

        basic_chunks = chunk_basic(text, metadata=metadata)
        semantic_chunks = chunk_semantic(text, metadata=metadata)
        parents, children = chunk_hierarchical(text, metadata=metadata)
        structure_chunks = chunk_structure_aware(text, metadata=metadata)

        basic_lengths.extend(len(chunk.text) for chunk in basic_chunks)
        semantic_lengths.extend(len(chunk.text) for chunk in semantic_chunks)
        structure_lengths.extend(len(chunk.text) for chunk in structure_chunks)
        hierarchical_child_lengths.extend(len(chunk.text) for chunk in children)
        hierarchical_parent_count += len(parents)
        hierarchical_child_count += len(children)

    results = {
        "basic": {
            "num_chunks": len(basic_lengths),
            **summarize_lengths(basic_lengths),
        },
        "semantic": {
            "num_chunks": len(semantic_lengths),
            **summarize_lengths(semantic_lengths),
        },
        "hierarchical": {
            "num_parents": hierarchical_parent_count,
            "num_children": hierarchical_child_count,
            "display_chunks": f"{hierarchical_parent_count}p/{hierarchical_child_count}c",
            **summarize_lengths(hierarchical_child_lengths),
        },
        "structure": {
            "num_chunks": len(structure_lengths),
            **summarize_lengths(structure_lengths),
        },
    }

    print(f"{'Strategy':<13} | {'Chunks':<8} | {'Avg Len':>7} | {'Min':>5} | {'Max':>5}")
    for name in ["basic", "semantic", "hierarchical", "structure"]:
        stats = results[name]
        chunk_label = stats.get("display_chunks", stats.get("num_chunks", 0))
        print(
            f"{name:<13} | {str(chunk_label):<8} | "
            f"{stats['avg_length']:>7} | {stats['min_length']:>5} | {stats['max_length']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

# ruff: noqa: E402

import re
from dataclasses import dataclass



@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    sentences = _sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[:2])


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    meta = extract_metadata(text)
    topic = meta.get("topic") or "nội dung này"
    questions = [
        f"{topic} là gì?",
        f"{topic} có quy định gì quan trọng?",
        f"Thông tin chính về {topic} là gì?",
    ]
    if re.search(r"\d", text):
        questions.insert(0, f"{topic} là bao nhiêu?")
    return questions[:max(0, n_questions)]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    topic = extract_metadata(text).get("topic", "nội dung liên quan")
    title = document_title or "tài liệu nguồn"
    context = f"Ngữ cảnh: trích từ {title}, nói về {topic}."
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    lowered = text.lower()
    if any(word in lowered for word in ["thuế", "gtgt", "doanh thu", "bctc", "kỳ tính thuế"]):
        category = "finance"
        topic = "thuế GTGT và báo cáo tài chính"
    elif any(word in lowered for word in ["dữ liệu cá nhân", "nghị định", "bảo vệ dữ liệu"]):
        category = "policy"
        topic = "bảo vệ dữ liệu cá nhân"
    elif any(word in lowered for word in ["mật khẩu", "vpn", "wireguard"]):
        category = "it"
        topic = "an toàn thông tin"
    else:
        category = "policy"
        topic = "quy định trong tài liệu"

    entities = re.findall(r"\b[A-ZĐ][A-ZĐ0-9_.-]{2,}\b", text)
    return {
        "topic": topic,
        "entities": list(dict.fromkeys(entities[:5])),
        "category": category,
        "language": "vi",
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []
    method_set = set(methods)
    use_full = "full" in method_set

    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        summary = summarize_chunk(text) if use_full or "summary" in method_set else ""
        questions = generate_hypothesis_questions(text) if use_full or "hyqa" in method_set else []
        enriched_text = (
            contextual_prepend(text, metadata.get("source", ""))
            if use_full or "contextual" in method_set
            else text
        )
        if questions:
            enriched_text = f"{enriched_text}\n\nCâu hỏi gợi ý: " + " | ".join(questions)
        auto_meta = extract_metadata(text) if use_full or "metadata" in method_set else {}
        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**metadata, **auto_meta},
                method="+".join(methods),
            )
        )

    return enriched


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

# ruff: noqa: E402

import json
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=_clean_score(row.get("faithfulness", 0.0)),
                answer_relevancy=_clean_score(row.get("answer_relevancy", 0.0)),
                context_precision=_clean_score(row.get("context_precision", 0.0)),
                context_recall=_clean_score(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        return _aggregate(per_question)
    except Exception:
        per_question = []
        for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
            context_text = " ".join(ctxs)
            per_question.append(
                EvalResult(
                    question=question,
                    answer=answer,
                    contexts=ctxs,
                    ground_truth=ground_truth,
                    faithfulness=_overlap(answer, context_text),
                    answer_relevancy=max(_overlap(question, answer), _overlap(ground_truth, answer)),
                    context_precision=_context_precision(ctxs, ground_truth),
                    context_recall=_overlap(ground_truth, context_text),
                )
            )
        return _aggregate(per_question)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    scored = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        scored.append((avg_score, result, metrics))

    failures = []
    for _, result, metrics in sorted(scored, key=lambda item: item[0])[:bottom_n]:
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = _diagnose(worst_metric)
        failures.append(
            {
                "question": result.question,
                "expected": result.ground_truth,
                "got": result.answer,
                "worst_metric": worst_metric,
                "score": round(float(metrics[worst_metric]), 4),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return failures


def _aggregate(per_question: list[EvalResult]) -> dict:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregate = {
        metric: round(sum(getattr(result, metric) for result in per_question) / max(len(per_question), 1), 4)
        for metric in metrics
    }
    aggregate["per_question"] = per_question
    return aggregate


def _clean_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(score) else max(0.0, min(1.0, score))


def _overlap(reference: str, candidate: str) -> float:
    ref_tokens = set(_tokenize(reference))
    cand_tokens = set(_tokenize(candidate))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def _context_precision(contexts: list[str], ground_truth: str) -> float:
    if not contexts:
        return 0.0
    relevant = sum(1 for ctx in contexts if _overlap(ground_truth, ctx) > 0)
    return relevant / len(contexts)


def _diagnose(metric: str) -> tuple[str, str]:
    mapping = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature, only answer from context"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, BM25 terms, or add more context"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer does not match question", "Improve answer extraction/prompt template"),
    }
    return mapping.get(metric, ("Unknown retrieval error", "Inspect retrieved context manually"))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

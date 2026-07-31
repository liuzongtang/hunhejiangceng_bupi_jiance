"""
Completeness scoring functions.

Evaluates whether a response fully addresses all aspects of the question.
A complete response should:
1. Address every sub-question or component of the prompt
2. Provide sufficient detail for each component
3. Not leave any part of the question unanswered

Scoring strategy:
1. Decompose the question into sub-questions / key points
2. Check each sub-question against the response
3. Score = fraction of sub-questions adequately covered
"""

import re
from typing import Dict, List, Optional


def _decompose_question(question: str) -> List[str]:
    """
    Decompose a complex question into sub-questions.

    Uses structural cues:
    - Numbered lists (1., 2., 3.)
    - Bullet points
    - Question marks
    - Semicolons separating distinct parts
    - Conjunctions suggesting multiple parts (and, also, additionally)
    """
    if not question:
        return []

    sub_questions: List[str] = []

    # Check for explicit numbered/comma-separated components
    # Pattern: "1. ... 2. ... 3. ..." or "(1) ... (2) ..."
    numbered = re.findall(
        r"(?:(?:^|\n)\s*(?:\d+[\.\)、]|[-•])\s*(.+?))(?=(?:\n\s*(?:\d+[\.\)、]|[-•]))|\Z)",
        question,
        re.DOTALL,
    )
    if len(numbered) >= 2:
        sub_questions.extend(s.strip() for s in numbered if s.strip())
        return sub_questions

    # Check for multiple question marks
    qmark_parts = re.split(r"\?\s*", question)
    qmark_parts = [p.strip() + "?" for p in qmark_parts if p.strip()]
    if len(qmark_parts) >= 2:
        sub_questions.extend(qmark_parts)
        return sub_questions

    # Split by common multi-part separators
    # Look for "also", "additionally", "furthermore", "moreover"
    conjunction_splits = re.split(
        r"(?:\s+(?:and|or)\s+also\b|\s*;\s*|\s+(?:additionally|furthermore|moreover|besides)\s*[,:]?\s*)",
        question,
        flags=re.IGNORECASE,
    )
    conjunction_splits = [s.strip() for s in conjunction_splits if len(s.strip()) > 10]
    if len(conjunction_splits) >= 2:
        sub_questions.extend(conjunction_splits)
        return sub_questions

    # If no decomposition found, treat the whole question as one component
    sub_questions.append(question)
    return sub_questions


def _extract_key_terms(text: str) -> List[str]:
    """Extract key content terms (nouns, named entities) from text."""
    # Simple extraction: words with 3+ chars, excluding stopwords
    stopwords = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "been",
        "some",
        "them",
        "then",
        "than",
        "this",
        "that",
        "with",
        "when",
        "will",
        "what",
        "which",
        "who",
        "how",
        "from",
        "your",
        "where",
        "there",
        "their",
        "about",
        "into",
        "would",
        "could",
        "should",
        "does",
        "each",
        "over",
        "under",
        "just",
        "very",
        "much",
        "such",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in stopwords]


def _compute_coverage(
    sub_question: str,
    response: str,
    key_terms: List[str],
) -> float:
    """
    Compute how well the response covers a sub-question.

    Uses keyword overlap between the sub-question's key terms
    and the response text.
    """
    q_terms = _extract_key_terms(sub_question)
    if not q_terms:
        return 0.5  # Cannot evaluate — give partial credit

    response_lower = response.lower()

    # Count how many question terms appear in the response
    covered = sum(1 for term in q_terms if term in response_lower)
    coverage_ratio = covered / len(q_terms)

    return coverage_ratio


def compute_completeness(
    response: str,
    question: Optional[str] = None,
    min_response_length: int = 20,
) -> float:
    """
    Compute completeness score for a response.

    Args:
        response: The model's generated response.
        question: The original prompt/question (can be None for format-only checks).
        min_response_length: Minimum character length for a response to be
                             considered "attempting to be complete."

    Returns:
        Score in [0.0, 1.0]:
        - 1.0 : fully addresses all question components
        - 0.0 : completely misses the question or too short
    """
    if not response:
        return 0.0

    # Very short responses are incomplete
    if len(response.strip()) < min_response_length:
        return max(0.0, len(response.strip()) / min_response_length)

    if question is None:
        # No question to compare — check if response has internal structure
        return _compute_self_contained_completeness(response)

    # Decompose the question
    sub_questions = _decompose_question(question)
    if not sub_questions:
        return 0.5

    # Score each sub-question
    scores: List[float] = []
    for sq in sub_questions:
        coverage = _compute_coverage(sq, response, _extract_key_terms(response))
        scores.append(coverage)

    if not scores:
        return 0.0

    # Weighted combination: mean coverage + bonus for all-covered
    mean_score = sum(scores) / len(scores)

    # Bonus if all sub-questions are at least partially covered
    if all(s > 0.3 for s in scores):
        mean_score = min(1.0, mean_score + 0.1)

    # Penalty if any sub-question is completely missed
    zeros = sum(1 for s in scores if s < 0.1)
    if zeros > 0:
        penalty = zeros / len(scores) * 0.2
        mean_score = max(0.0, mean_score - penalty)

    return mean_score


def _compute_self_contained_completeness(response: str) -> float:
    """
    Evaluate completeness of a response without a reference question.

    Checks for structural markers of a complete answer:
    - Has introduction/conclusion
    - Has sufficient length and detail
    - Contains multiple paragraphs or sections
    """
    score = 0.5  # Start neutral

    text = response.strip()

    # Length bonus
    if len(text) > 200:
        score += 0.1
    if len(text) > 500:
        score += 0.1

    # Structure bonus
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        score += 0.1
    if len(paragraphs) >= 4:
        score += 0.1

    # Has concluding language
    conclusion_patterns = [
        r"\b(in\s+conclusion|to\s+sum\s+up|in\s+summary|overall|finally)\b",
        r"\b(总结|综上|总而言之|最后)\b",
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in conclusion_patterns):
        score += 0.1

    return min(1.0, score)


def compute_completeness_report(response: str, question: str) -> Dict:
    """
    Generate a detailed completeness report.

    Returns:
        Dict with overall score, per-sub-question scores, and coverage details.
    """
    sub_questions = _decompose_question(question)
    per_question_scores: Dict[str, float] = {}

    for i, sq in enumerate(sub_questions):
        coverage = _compute_coverage(sq, response, _extract_key_terms(response))
        per_question_scores[f"q{i + 1}"] = coverage

    overall = compute_completeness(response, question)

    return {
        "overall_completeness": overall,
        "sub_question_count": len(sub_questions),
        "per_question_scores": per_question_scores,
        "fully_covered": all(s > 0.5 for s in per_question_scores.values()),
        "response_length": len(response),
    }

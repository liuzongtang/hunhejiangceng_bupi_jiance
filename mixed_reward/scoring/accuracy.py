"""
Accuracy scoring functions.

Evaluates how factually correct a model response is compared to
a ground truth answer. Supports multiple strategies:

1. Exact match — normalized string comparison
2. F1 score — token-level precision/recall
3. LLM-as-Judge — API-based semantic equivalence (optional)
"""

import math
import re
from typing import Callable, List, Optional, Set


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation for more lenient matching
    text = text.rstrip(".,;:!?。，；：！？")
    return text


def _extract_final_answer(text: str) -> str:
    """Extract the final answer from a chain-of-thought response."""
    # Try to find answer after common markers
    patterns = [
        r"(?:final\s+answer|answer|therefore|so|thus)[\s:]*[:\-]?\s*(.+?)$",
        r"(?:答案是|所以|因此|综上)[:：]?\s*(.+?)$",
        r"####\s*(.+?)$",  # GSM8K format
        r"\\boxed\{([^}]+)\}",  # LaTeX boxed
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenization."""
    return re.findall(r"\w+", text.lower())


def compute_exact_match(response: str, ground_truth: str) -> float:
    """
    Compute exact match score between response and ground truth.

    Returns 1.0 if the final answer matches, 0.0 otherwise.
    Handles chain-of-thought by extracting the final answer.
    """
    if not response or not ground_truth:
        return 0.0

    # Try extracting final answer from response
    final_answer = _extract_final_answer(response)
    normalized_response = _normalize(final_answer)
    normalized_truth = _normalize(ground_truth)

    # Exact match
    if normalized_response == normalized_truth:
        return 1.0

    # Check containment (truth is contained in response or vice versa)
    if (
        normalized_truth in normalized_response
        or normalized_response in normalized_truth
    ):
        return 0.8

    return 0.0


def compute_f1_score(response: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.

    Balances precision and recall for partial credit scoring.
    """
    if not response or not ground_truth:
        return 0.0

    final_answer = _extract_final_answer(response)
    response_tokens: Set[str] = set(_tokenize(final_answer))
    truth_tokens: Set[str] = set(_tokenize(ground_truth))

    if not truth_tokens:
        return 1.0 if not response_tokens else 0.0

    common = response_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(response_tokens) if response_tokens else 0.0
    recall = len(common) / len(truth_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_numeric_proximity(response: str, ground_truth: str) -> float:
    """
    For numeric answers: score based on relative error.

    Returns 1.0 for exact match, decreasing for larger errors.
    """
    try:
        # Extract numbers
        resp_nums = re.findall(r"-?\d+\.?\d*", response)
        truth_nums = re.findall(r"-?\d+\.?\d*", ground_truth)

        if not resp_nums or not truth_nums:
            return 0.0

        resp_val = float(resp_nums[-1])  # Last number is usually the answer
        truth_val = float(truth_nums[-1])

        if truth_val == 0:
            return 1.0 if resp_val == 0 else 0.0

        relative_error = abs(resp_val - truth_val) / abs(truth_val)
        # Score decays exponentially with relative error
        score = math.exp(-relative_error)
        return max(0.0, min(1.0, score))

    except (ValueError, IndexError):
        return 0.0


def compute_accuracy(
    response: str,
    ground_truth: Optional[str] = None,
    method: str = "auto",
    judge_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """
    Compute accuracy score for a response.

    Args:
        response: The model's generated response.
        ground_truth: The expected answer (can be None for LLM-as-Judge).
        method: Scoring method — "exact", "f1", "numeric", "auto", or "llm_judge".
        judge_fn: Optional external judge function(response, ground_truth) -> float.

    Returns:
        Score in [0.0, 1.0].
    """
    if judge_fn is not None:
        return judge_fn(response, ground_truth or "")

    if method == "llm_judge" and ground_truth is None:
        # Placeholder: return 0.5 when no judge available
        return 0.5

    if ground_truth is None:
        return 0.0

    if method == "exact":
        return compute_exact_match(response, ground_truth)
    elif method == "f1":
        return compute_f1_score(response, ground_truth)
    elif method == "numeric":
        return compute_numeric_proximity(response, ground_truth)
    elif method == "auto":
        # Auto-detect: try numeric first, fall back to F1
        numeric_score = compute_numeric_proximity(response, ground_truth)
        if numeric_score > 0.0:
            return numeric_score
        # Try exact match
        exact = compute_exact_match(response, ground_truth)
        if exact > 0.0:
            return exact
        # Fall back to F1 for partial credit
        return compute_f1_score(response, ground_truth)
    else:
        raise ValueError(f"Unknown accuracy method: {method}")

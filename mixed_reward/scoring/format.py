"""
Format adherence scoring functions.

Evaluates whether a response follows expected structural/formatting conventions.
Common format requirements include:

1. Proper markdown structure (headers, lists, code blocks)
2. Chain-of-thought format (reasoning before answer)
3. JSON output format
4. Specific tag wrapping (e.g., <think>...</think>, <answer>...</answer>)
5. No excessive repetition or truncation
"""

import json
import re
from typing import Dict, Optional, Set


def compute_format(
    response: str,
    expected_format: str = "general",
    format_spec: Optional[Dict] = None,
) -> float:
    """
    Compute format adherence score for a response.

    Args:
        response: The model's generated response.
        expected_format: Format type to check:
            - "general" : basic quality (no repetition, reasonable length)
            - "markdown" : proper markdown usage
            - "cot" : chain-of-thought (reasoning → answer)
            - "json" : valid JSON output
            - "code" : code blocks present and properly formatted
            - "custom" : use format_spec dict for custom rules
        format_spec: Custom format specification (used when expected_format="custom").

    Returns:
        Score in [0.0, 1.0].
    """
    if not response:
        return 0.0

    if expected_format == "general":
        return _compute_general_format(response)
    elif expected_format == "markdown":
        return _compute_markdown_format(response)
    elif expected_format == "cot":
        return _compute_cot_format(response)
    elif expected_format == "json":
        return _compute_json_format(response)
    elif expected_format == "code":
        return _compute_code_format(response)
    elif expected_format == "custom" and format_spec:
        return _compute_custom_format(response, format_spec)
    else:
        raise ValueError(f"Unknown format type: {expected_format}")


def _compute_general_format(response: str) -> float:
    """
    Check basic response quality.
    Penalizes: excessive repetition, truncation, gibberish.
    """
    score = 1.0
    text = response.strip()

    # Check for truncation (response ends mid-sentence)
    truncation_indicators = [
        r"\.{2,}$",  # trailing ellipsis
        r",\s*$",  # trailing comma
        r"\b(and|or|but|so|then)\s*$",  # trailing conjunction
    ]
    for pattern in truncation_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 0.1
            break

    # Check for excessive repetition (same sentence repeated)
    sentences = re.split(r"[.!?。！？]\s*", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 3:
        unique_sentences: Set[str] = {s.lower() for s in sentences}
        repetition_ratio = 1.0 - (len(unique_sentences) / len(sentences))
        if repetition_ratio > 0.3:
            score -= 0.2
        elif repetition_ratio > 0.5:
            score -= 0.4

    # Check for excessive capitalization (screaming)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars > 0:
        upper_ratio = sum(1 for c in text if c.isupper()) / alpha_chars
        if upper_ratio > 0.7 and len(text) > 50:
            score -= 0.2

    # Length penalty (too short is incomplete, not bad format per se)
    if len(text) < 20:
        score -= 0.1

    return max(0.0, score)


def _compute_markdown_format(response: str) -> float:
    """
    Check proper markdown formatting.
    Bonus for: headers, lists, code blocks, bold/italic.
    Penalty for: broken markdown (unclosed code fences, etc.).
    """
    score = 0.6  # Start with a baseline

    # Check for headers
    if re.search(r"^#{1,6}\s+\S", response, re.MULTILINE):
        score += 0.1

    # Check for lists (ordered or unordered)
    if re.search(r"^[\s]*[-*+]\s+\S", response, re.MULTILINE):
        score += 0.05
    if re.search(r"^[\s]*\d+[\.\)]\s+\S", response, re.MULTILINE):
        score += 0.05

    # Check for code blocks (properly fenced)
    code_fences = re.findall(r"```", response)
    if code_fences:
        if len(code_fences) % 2 == 0:
            score += 0.1  # Properly closed
        else:
            score -= 0.1  # Unclosed fence

    # Check for inline formatting
    if re.search(r"\*\*[^*]+\*\*|__[^_]+__", response):
        score += 0.05  # Bold
    if re.search(r"\*[^*]+\*|_[^_]+_", response):
        score += 0.05  # Italic

    return max(0.0, min(1.0, score))


def _compute_cot_format(response: str) -> float:
    """
    Check chain-of-thought format.
    Expects: reasoning/analysis section followed by a clear answer/conclusion.
    """
    score = 0.0

    # Check for reasoning markers
    reasoning_patterns = [
        r"\b(let(?:'s|\s+us)?\s+think)\b",
        r"\b(first|step\s*1|step\s*one)\b",
        r"\b(we\s+(?:need\s+to|can|should|must|have))\b",
        r"\b(?:首先|第一步|我们[需需]要)\b",
        r"<think>.*?</think>",
        r"<reasoning>.*?</reasoning>",
        r"\b(analysis|reasoning|thought\s+process)\b",
    ]
    has_reasoning = any(
        re.search(p, response, re.IGNORECASE | re.DOTALL) for p in reasoning_patterns
    )
    if has_reasoning:
        score += 0.4

    # Check for answer conclusion
    answer_patterns = [
        r"\b(therefore|thus|so|hence|accordingly)\b",
        r"\b(the\s+answer\s+is|final\s+answer|conclusion)\b",
        r"\b(答案是|所以|因此|综上)\b",
        r"<answer>.*?</answer>",
        r"\\boxed\{[^}]+\}",
        r"####\s+\S",
    ]
    has_answer = any(
        re.search(p, response, re.IGNORECASE | re.DOTALL) for p in answer_patterns
    )
    if has_answer:
        score += 0.3

    # Check structural separation (reasoning before answer)
    if has_reasoning and has_answer:
        # Ensure reasoning comes before answer
        reasoning_positions = []
        answer_positions = []
        for p in reasoning_patterns:
            for m in re.finditer(p, response, re.IGNORECASE | re.DOTALL):
                reasoning_positions.append(m.start())
        for p in answer_patterns:
            for m in re.finditer(p, response, re.IGNORECASE | re.DOTALL):
                answer_positions.append(m.start())

        if (
            reasoning_positions
            and answer_positions
            and min(reasoning_positions) < max(answer_positions)
        ):
            score += 0.3  # Reasoning before answer

    return min(1.0, score)


def _compute_json_format(response: str) -> float:
    """
    Check if response is valid JSON.
    Extracts JSON from markdown code blocks if present.
    """
    # Try to extract JSON from code blocks first
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
    else:
        candidate = response.strip()

    try:
        json.loads(candidate)
        return 1.0
    except json.JSONDecodeError:
        # Partial credit: looks like JSON but has errors
        if candidate.startswith("{") or candidate.startswith("["):
            return 0.5
        return 0.0


def _compute_code_format(response: str) -> float:
    """
    Check proper code block formatting.
    """
    score = 1.0

    # Check for code blocks
    code_blocks = re.findall(r"```(\w*)\n?(.*?)```", response, re.DOTALL)
    if not code_blocks:
        # No code blocks — check for inline code
        inline_code = len(re.findall(r"`[^`]+`", response))
        if inline_code > 0:
            return 0.8
        return 0.5

    # All code blocks should have a language tag
    tagged = sum(1 for lang, _ in code_blocks if lang.strip())
    if len(code_blocks) > 0:
        tag_ratio = tagged / len(code_blocks)
        score -= (1.0 - tag_ratio) * 0.2

    return max(0.0, score)


def _compute_custom_format(response: str, format_spec: Dict) -> float:
    """
    Check custom format requirements from format_spec.

    format_spec keys:
        - required_patterns: List[str] — regex patterns that MUST match
        - forbidden_patterns: List[str] — regex patterns that MUST NOT match
        - required_tags: List[str] — XML-like tags that must be present
        - min_length: int — minimum character length
        - max_length: int — maximum character length
    """
    score = 1.0
    checks = 0

    # Required patterns
    if "required_patterns" in format_spec:
        for pattern in format_spec["required_patterns"]:
            checks += 1
            if not re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                score -= 1.0 / max(1, len(format_spec.get("required_patterns", [])))

    # Forbidden patterns
    if "forbidden_patterns" in format_spec:
        for pattern in format_spec["forbidden_patterns"]:
            checks += 1
            if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                score -= 1.0 / max(1, len(format_spec.get("forbidden_patterns", [])))

    # Required tags
    if "required_tags" in format_spec:
        for tag in format_spec["required_tags"]:
            checks += 1
            if f"<{tag}>" not in response or f"</{tag}>" not in response:
                score -= 1.0 / max(1, len(format_spec["required_tags"]))

    # Length constraints
    if "min_length" in format_spec:
        checks += 1
        if len(response) < format_spec["min_length"]:
            ratio = len(response) / format_spec["min_length"]
            score -= (1.0 - ratio) * 0.3

    if "max_length" in format_spec:
        checks += 1
        if len(response) > format_spec["max_length"]:
            score -= 0.2

    return max(0.0, min(1.0, score))

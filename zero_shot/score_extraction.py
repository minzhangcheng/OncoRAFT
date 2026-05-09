"""
Score extraction utilities for zero-shot LLM outputs.
"""
import re


def extract_score(response):
    """Extract a numeric score in [0,1] from model-generated text.

    Tries multiple strategies in order of reliability:
    1. Explicit score patterns (**0.55**, Score: 0.55, etc.)
    2. Context-based patterns (likelihood: 0.55, probability: 0.55)
    3. Keyword sentiment fallback (returns 0.3-0.7 based on positive/negative
       keyword balance, or 0.5 if uncertain — never None once this branch is reached)

    Returns None only if `response` is empty or not a string.
    """
    if not response or not isinstance(response, str):
        return None

    text = response.strip()

    # --- Strategy 1: Explicit numeric patterns ---
    patterns = [
        r'\*\*(\d+\.\d+|\d+)\*\*',           # **0.55**
        r'Score[:\s]*(\d+\.\d+|\d+)',          # Score: 0.55
        r'Response[:\s]*(\d+\.\d+|\d+)',       # Response: 0.55
        r'(\d+\.\d+)\s*\(',                    # 0.55 (followed by paren)
        r'(?:^|\D)(\d+\.\d+)(?=\D|$)',         # standalone decimal
        r'(?:^|\D)([01])(?=\D|$)',             # standalone 0 or 1
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text):
            try:
                score = float(match)
                if 0 <= score <= 1:
                    return score
            except ValueError:
                continue

    # --- Strategy 2: Context-based patterns ---
    context_patterns = [
        r'likelihood[\s\w]*[:\s]*(\d+\.\d+|\d+)',
        r'probability[\s\w]*[:\s]*(\d+\.\d+|\d+)',
        r'response score[\s\w]*[:\s]*(\d+\.\d+|\d+)',
        r'(\d+\.\d+|\d+)[\s\w]*likelihood',
        r'(\d+\.\d+|\d+)[\s\w]*probability',
    ]

    for pattern in context_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                score = float(m.group(1))
                if 0 <= score <= 1:
                    return score
            except (ValueError, IndexError):
                continue

    # --- Strategy 3: Keyword sentiment fallback ---
    positive = [
        'high likelihood', 'likely to respond', 'positive response',
        'complete response', 'good prognosis', 'favorable outcome',
        'respond well', 'sensitive to', 'effective treatment',
    ]
    negative = [
        'low likelihood', 'unlikely to respond', 'negative response',
        'disease progression', 'poor prognosis', 'unfavorable outcome',
        'resistant to', 'refractory', 'aggressive disease', 'advanced stage',
    ]

    lower = text.lower()
    pos = sum(1 for p in positive if p in lower)
    neg = sum(1 for p in negative if p in lower)

    if pos > neg * 2:
        return 0.7
    elif neg > pos * 2:
        return 0.3
    elif pos > neg:
        return 0.6
    elif neg > pos:
        return 0.4
    else:
        return 0.5


QWEN3_END_THINK_TOKEN_ID = 151668  # </think>


def parse_qwen3_thinking(output_ids, tokenizer):
    """Parse Qwen3 output to separate thinking content from final answer.

    Qwen3 uses </think> (token id 151668) to mark the end of the thinking span.

    Returns:
        (thinking_content, final_content)
    """
    try:
        index = len(output_ids) - output_ids[::-1].index(QWEN3_END_THINK_TOKEN_ID)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking, content

"""Detect the token index where the final answer begins in a generation.

Returns:
    (token_idx, marker_name)  if a boundary is detected
    (None, None)              otherwise (drop the sample)

For non-reasoning models / thinkoff variants, call with force_zero=True and
this module returns (0, "none").
"""
from __future__ import annotations

import re
from typing import Optional, Tuple, Sequence

# Ordered by reliability. Each entry: (label, regex, where_to_land)
#   where_to_land = "end" -> answer starts after match
#                 = "start" -> answer starts at match.start() (for headers that are part of answer)
MARKERS: list[tuple[str, re.Pattern, str]] = [
    ("</think>",         re.compile(r"</think>\s*"),                             "end"),
    ("**Final Answer**", re.compile(r"\*\*Final Answer:?\*\*\s*"),               "end"),
    ("Final Answer:",    re.compile(r"(?:^|\n)\s*Final Answer:\s*"),             "end"),
    ("**Answer**",       re.compile(r"\*\*Answer:?\*\*\s*"),                     "end"),
    ("####",             re.compile(r"(?:^|\n)####\s*"),                         "end"),
    ("**Final Selection", re.compile(r"(?:^|\n)\s*[0-9]+\.\s*\*\*Final Selection[^\n]*\n+\s*"), "end"),
    ("**Final Response", re.compile(r"(?:^|\n)\s*[0-9]*\.?\s*\*\*Final Response[^\n]*\n+\s*"),  "end"),
    ("**Drafting final", re.compile(r"(?:^|\n)\s*[0-9]+\.\s*\*\*(?:Final )?Draft(?:ing)?[^\n]*\n+\s*"), "end"),
    ("Response:",        re.compile(r"(?:^|\n)\s*\*?\*?Response:?\*?\*?\s*\n"),  "end"),
]


def detect_char_boundary(text: str) -> Optional[Tuple[int, str]]:
    """Return (char_index_after_marker, marker_label) or None."""
    best: Optional[Tuple[int, str]] = None
    for label, pat, mode in MARKERS:
        m = pat.search(text)
        if not m:
            continue
        idx = m.end() if mode == "end" else m.start()
        # Prefer the earliest high-priority marker. Because we iterate in priority
        # order, we take the first hit we find.
        return (idx, label)
    return best


def char_to_token_index(text: str, token_ids: Sequence[int], tokenizer, char_idx: int) -> int:
    """Map a char index in the decoded text back to a token index.

    Strategy: decode progressive prefixes of token_ids until the decoded length
    reaches char_idx. O(log N) via binary search on cumulative decodes.
    """
    if char_idx <= 0:
        return 0
    if char_idx >= len(text):
        return len(token_ids)
    lo, hi = 0, len(token_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        decoded = tokenizer.decode(token_ids[:mid], skip_special_tokens=False)
        if len(decoded) < char_idx:
            lo = mid + 1
        else:
            hi = mid
    return lo


def detect_answer_start(
    text: str,
    token_ids: Optional[Sequence[int]] = None,
    tokenizer=None,
    force_zero: bool = False,
) -> Tuple[Optional[int], Optional[str]]:
    """Return (token_idx, marker) for answer start, or (None, None) if not found.

    If force_zero (for non-reasoning / thinkoff variants), always returns (0, "none").
    If tokenizer is None but token_ids is given, fall back to a char-ratio approximation.
    """
    if force_zero:
        return 0, "none"
    boundary = detect_char_boundary(text)
    if boundary is None:
        return None, None
    char_idx, label = boundary
    if token_ids is None:
        return None, label
    if tokenizer is not None:
        try:
            return char_to_token_index(text, token_ids, tokenizer, char_idx), label
        except Exception:
            pass
    # Fallback: proportional approximation
    if len(text) == 0:
        return 0, label
    approx = int(round((char_idx / len(text)) * len(token_ids)))
    return min(approx, len(token_ids)), label


# ---------------------------------------------------------------------------
# Self-test / sanity check
# ---------------------------------------------------------------------------

SAMPLES = [
    # (expected_has_marker, label_contains, text)
    (True, "Drafting", "Thinking Process:\n\n1. Analyze.\n\n5. **Drafting final answer:**\n\nWeather is chaotic..."),
    (True, "</think>", "<think>\nok let me think\n</think>\nRegression tests ensure..."),
    (True, "Final Answer", "Analysis...\n\n**Final Answer:** 42"),
    (True, "####",        "Reasoning step one...\n#### The answer\nIt is 42"),
    (False, None,         "Weather forecasts become unreliable because the atmosphere is chaotic."),
    (False, None,         "<think>We need to produce answer. We need to produce answer. We need to produce answer."),
    (True, "**Answer**",  "Working...\n\n**Answer:**\nYes, it works."),
    (True, "Response:",   "Steps:\n1. foo\n2. bar\n\nResponse:\nThe answer is 7."),
    (True, "</think>",    "<think></think>\nThis is the whole answer."),
    (True, "Drafting",    "1. Think.\n2. **Drafting - Attempt 1:**\n    Actual answer goes here."),
]


def _selftest():
    ok = 0
    for i, (has_marker, label_sub, text) in enumerate(SAMPLES):
        idx, label = detect_answer_start(text)
        got_marker = label is not None
        pass_ = (got_marker == has_marker) and (
            not has_marker or (label_sub and label_sub.lower() in (label or "").lower())
        )
        status = "OK " if pass_ else "FAIL"
        if pass_:
            ok += 1
        print(f"  [{status}] {i}: has_marker={has_marker} got={label!r}  text={text[:60]!r}")
    print(f"\n{ok}/{len(SAMPLES)} passed")


if __name__ == "__main__":
    _selftest()

#!/usr/bin/env python3
"""Generate high-N prompt pairs with tiny, mostly semantics-preserving edits."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


BASE_PROMPTS = [
    "Explain why weather forecasts become unreliable after enough time, in one concise paragraph.",
    "Describe when a cache should be invalidated in a web application.",
    "List the main risks of depending on an external API in production.",
    "Explain why JSON schemas are useful for service interfaces.",
    "Give three reasons production services need monitoring.",
    "Explain how rate limits should influence background job design.",
    "List the main reasons secrets should not be committed to a repository.",
    "Write a concise Python function that checks whether a string is a palindrome.",
    "Explain why regression tests are useful after refactoring code.",
    "Describe how a background job queue should handle retries.",
    "Explain why database migrations should be reviewed carefully.",
    "Describe how feature flags reduce deployment risk.",
    "Explain why retry logic should use exponential backoff.",
    "List the tradeoffs of using a monorepo for many services.",
    "Explain why observability matters in distributed systems.",
    "Describe how an idempotency key prevents duplicate work.",
    "Explain why user input should be validated at service boundaries.",
    "List the main benefits of type checking in a large codebase.",
    "Describe how a circuit breaker protects a downstream dependency.",
    "Explain why small changes can have large effects in complex systems.",
]

SMALL_WORDS = {"a", "an", "the", "to", "of", "in", "for", "and", "or", "is", "be", "as"}


def word_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\b[\w']+\b", text))


def space_positions(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r" ", text)]


def punctuation_positions(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r"[,.;:!?]", text)]


def insert_at(text: str, index: int, value: str) -> str:
    return text[:index] + value + text[index:]


def replace_at(text: str, index: int, length: int, value: str) -> str:
    return text[:index] + value + text[index + length :]


def mutate(text: str, kind: str, rng: random.Random) -> str:
    spaces = space_positions(text)
    punct = punctuation_positions(text)
    words = word_spans(text)

    if kind == "trailing_space":
        return text + (" " * rng.randint(1, 4))
    if kind == "leading_space":
        return (" " * rng.randint(1, 4)) + text
    if kind == "trailing_newline":
        return text + ("\n" * rng.randint(1, 3))
    if kind == "leading_newline":
        return ("\n" * rng.randint(1, 2)) + text
    if kind == "crlf_suffix":
        return text + "\r\n"
    if kind == "double_internal_space" and spaces:
        return insert_at(text, rng.choice(spaces), " ")
    if kind == "triple_internal_space" and spaces:
        return insert_at(text, rng.choice(spaces), "  ")
    if kind == "line_wrap" and spaces:
        return replace_at(text, rng.choice(spaces), 1, "\n")
    if kind == "blank_line_wrap" and spaces:
        return replace_at(text, rng.choice(spaces), 1, "\n\n")
    if kind == "tab_indent":
        return "\t" + text
    if kind == "tab_after_space" and spaces:
        return insert_at(text, rng.choice(spaces) + 1, "\t")
    if kind == "space_before_punctuation" and punct:
        return insert_at(text, rng.choice(punct), " ")
    if kind == "space_after_punctuation" and punct:
        idx = rng.choice(punct)
        return insert_at(text, idx + 1, " ")
    if kind == "duplicate_punctuation" and punct:
        idx = rng.choice(punct)
        return insert_at(text, idx, text[idx])
    if kind == "parenthesize_word" and words:
        w = rng.choice(words)
        return text[: w.start()] + "(" + text[w.start() : w.end()] + ")" + text[w.end() :]
    if kind == "duplicate_small_word":
        candidates = [w for w in words if w.group(0).lower() in SMALL_WORDS]
        if candidates:
            w = rng.choice(candidates)
            return text[: w.end()] + " " + w.group(0) + text[w.end() :]

    # Fallback stays semantics-preserving and guarantees a changed string.
    return text + " "


KINDS = [
    "trailing_space",
    "leading_space",
    "trailing_newline",
    "leading_newline",
    "crlf_suffix",
    "double_internal_space",
    "triple_internal_space",
    "line_wrap",
    "blank_line_wrap",
    "tab_indent",
    "tab_after_space",
    "space_before_punctuation",
    "space_after_punctuation",
    "duplicate_punctuation",
    "parenthesize_word",
    "duplicate_small_word",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("configs/prompt_pairs_micro_500.json"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--controls", type=int, default=25)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []

    for i in range(args.controls):
        prompt = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        rows.append(
            {
                "id": f"micro_control_identical_{i:03d}",
                "category": "micro_control_identical",
                "prompt_a": prompt,
                "prompt_b": prompt,
            }
        )

    for i in range(args.count):
        prompt = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        kind = KINDS[i % len(KINDS)]
        # Shuffle deterministically across prompts/kinds without losing coverage.
        if i % len(KINDS) == 0:
            rng.shuffle(KINDS)
        mutated = mutate(prompt, kind, rng)
        rows.append(
            {
                "id": f"micro_{kind}_{i:04d}",
                "category": f"micro_{kind}",
                "prompt_a": prompt,
                "prompt_b": mutated,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} pairs to {args.out}")
    print("Categories:")
    for kind in ["micro_control_identical", *[f"micro_{k}" for k in KINDS]]:
        print(f"  {kind}: {sum(1 for row in rows if row['category'] == kind)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a larger prompt-pair ladder for robustness runs."""

from __future__ import annotations

import json
from pathlib import Path


TOPICS = [
    ("weather", "Explain why weather forecasts become unreliable after enough time."),
    ("regression_tests", "Explain why regression tests are useful after refactoring code."),
    ("job_queue", "Describe how a background job queue should handle retries."),
    ("cache", "Describe when a cache should be invalidated in a web application."),
    ("palindrome", "Write a concise Python function that checks whether a string is a palindrome."),
    ("external_api", "List the main risks of depending on an external API in production."),
    ("json_schema", "Explain why JSON schemas are useful for service interfaces."),
    ("monitoring", "Give three reasons production services need monitoring."),
    ("rate_limits", "Explain how rate limits should influence background job design."),
    ("database_migrations", "Explain why database migrations should be reviewed carefully."),
    ("secrets", "List the main reasons secrets should not be committed to a repository."),
    ("incident_review", "Describe how an incident review should help a software team improve."),
    ("chaos_system", "Explain how small changes can affect a complex system over time."),
    ("feedback_loops", "Explain why fast feedback loops improve software development."),
    ("deployment", "Explain why deployment rollbacks should be tested before an incident."),
    ("observability", "Describe how logs, metrics, and traces help debug production systems."),
    ("idempotency", "Explain why idempotency matters for retryable API requests."),
    ("schema_migration", "Describe how to safely roll out a database schema migration."),
    ("load_shedding", "Explain when a service should shed load instead of accepting every request."),
    ("feature_flags", "Explain how feature flags reduce release risk."),
]

SYNONYMS = [
    ("fast", "quick", "Explain why fast feedback loops improve software development."),
    ("small", "tiny", "Explain how a small perturbation can grow in a chaotic system."),
    ("useful", "valuable", "Explain why regression tests are useful after refactoring code."),
    ("risks", "dangers", "List the main risks of depending on an external API in production."),
    ("unreliable", "inaccurate", "Explain why weather forecasts become unreliable after enough time."),
    ("carefully", "thoroughly", "Explain why database migrations should be reviewed carefully."),
    ("reduce", "lower", "Explain how feature flags reduce release risk."),
    ("debug", "diagnose", "Describe how logs, metrics, and traces help debug production systems."),
    ("accepting", "processing", "Explain when a service should shed load instead of accepting every request."),
    ("safe", "reliable", "Describe how to safely roll out a database schema migration."),
]

SEMANTIC_SMALL = [
    ("palindrome", "Write a concise Python function that checks whether a string is a palindrome.", "Write a concise Python function that checks whether a list is sorted."),
    ("cache_queue", "Describe when a cache should be invalidated in a web application.", "Describe when a background job should be retried in a web application."),
    ("logs_metrics", "Describe how logs help debug production systems.", "Describe how metrics help debug production systems."),
    ("rate_limits_backpressure", "Explain how rate limits should influence background job design.", "Explain how backpressure should influence background job design."),
    ("schema_api", "Explain why JSON schemas are useful for service interfaces.", "Explain why OpenAPI specs are useful for service interfaces."),
    ("secrets_tokens", "List the main reasons secrets should not be committed to a repository.", "List the main reasons API tokens should be rotated regularly."),
    ("rollback_flags", "Explain why deployment rollbacks should be tested before an incident.", "Explain why feature flags should be tested before a release."),
    ("monitoring_alerting", "Give three reasons production services need monitoring.", "Give three reasons production services need alerting."),
]

POSITIVE = [
    ("weather_palindrome", TOPICS[0][1], TOPICS[4][1]),
    ("cache_secrets", TOPICS[3][1], TOPICS[10][1]),
    ("json_incident", TOPICS[6][1], TOPICS[11][1]),
    ("queue_observability", TOPICS[2][1], TOPICS[15][1]),
    ("flags_weather", TOPICS[19][1], TOPICS[0][1]),
]


def add(rows: list[dict[str, str]], row_id: str, category: str, a: str, b: str) -> None:
    rows.append({"id": row_id, "category": category, "prompt_a": a, "prompt_b": b})


def main() -> None:
    rows: list[dict[str, str]] = []

    for key, prompt in TOPICS[:10]:
        add(rows, f"control_{key}", "control_identical", prompt, prompt)

    for key, prompt in TOPICS:
        add(rows, f"noop_{key}_trailing_space", "noop_format", prompt, prompt + " ")
        add(rows, f"noop_{key}_newline", "noop_format", prompt, prompt + "\n")

    for key, prompt in TOPICS:
        if "." in prompt:
            add(rows, f"punct_{key}_question", "punctuation", prompt, prompt.rstrip(".") + "?")
            add(rows, f"punct_{key}_colon", "punctuation", prompt, prompt.rstrip(".") + ":")

    for old, new, prompt in SYNONYMS:
        add(rows, f"synonym_{old}_{new}", "synonym", prompt, prompt.replace(old, new, 1))

    for key, a, b in SEMANTIC_SMALL:
        add(rows, f"semantic_{key}", "semantic_small", a, b)

    for key, a, b in POSITIVE:
        add(rows, f"positive_{key}", "positive_control", a, b)

    out = Path("configs/prompt_pairs_scaled.json")
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} prompt pairs to {out}")


if __name__ == "__main__":
    main()

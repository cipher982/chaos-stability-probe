"""BranchTrace — Branch Card format + renderer.

Read-only artifact tool. Consumes existing run directories under
`runs/` and produces self-contained Branch Cards (JSON + HTML) for
LLM prompt-perturbation branch events.

Schema version: branchcard/0.1
See docs/branchtrace_spec_20260506.md §4.
"""

from .schema import BranchCard

SCHEMA_VERSION = "branchcard/0.1"

__all__ = ["BranchCard", "SCHEMA_VERSION"]

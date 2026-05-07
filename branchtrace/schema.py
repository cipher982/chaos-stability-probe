"""Branch Card schema (branchcard/0.1).

Every top-level block carries a `source` tag so consumers can tell what
came from existing artifacts (`observed`), what was recomputed
(`derived`), and what was not available (`not_run`). See spec §4.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["observed", "derived", "not_run"]
Regime = Literal["edit_boundary", "trajectory_migration", "prompt_accumulation"]
EventKind = Literal["silent_logit_divergence", "immediate_visible_branch"]
PositionClass = Literal[
    "prompt_lcp",
    "aligned_prompt_control",
    "generated_prefix",
    "final_context",
]


class Runtime(BaseModel):
    source: Source = "observed"
    torch: str | None = None
    transformers: str | None = None
    tokenizer_revision: str | None = None
    model_revision: str | None = None
    dtype: str | None = None
    device: str | None = None
    batch_size: int | None = None
    attn_impl: str | None = None
    seed: int | None = None
    system_prompt: str | None = None


class RunSide(BaseModel):
    source: Source = "observed"
    model_name: str
    model_id: str
    model_family: str | None = None
    decode: dict = Field(default_factory=dict)  # do_sample, max_new_tokens, temperature, top_p
    runtime: Runtime
    prompt_text: str
    prompt_token_ids: list[int]
    prompt_token_count: int
    generated_text: str
    generated_token_ids: list[int]
    generated_token_count: int
    run_dir: str


class Edit(BaseModel):
    source: Source = "derived"
    kind: str
    prompt_token_edit_distance: int
    prompt_token_delta_kind: str | None = None
    prompt_token_lcp: int
    prompt_input_token_delta: int
    visible_diff_span: dict = Field(default_factory=dict)  # {"a": str, "b": str}


class BranchTokenSide(BaseModel):
    id: int
    text: str
    top1_prob: float | None = None
    top1_margin: float | None = None
    js_vs_other: float | None = None
    effective_branching_factor: float | None = None


class Branch(BaseModel):
    source: Source = "observed"
    branch_t: int
    event_kind: EventKind
    silent_logit_lead: float | None = None
    common_prefix_tokens: int
    branch_token_a: BranchTokenSide
    branch_token_b: BranchTokenSide
    topk_a: list[list] | None = None  # [[token_id, prob], ...]
    topk_b: list[list] | None = None


class Replay(BaseModel):
    deterministic_source: Source = "observed"
    deterministic_reproducible_a: bool | None = None
    deterministic_reproducible_b: bool | None = None
    forced_prefix_source: Source = "not_run"
    forced_prefix_a_flips_to_b_branch_token: bool | None = None
    forced_prefix_b_flips_to_a_branch_token: bool | None = None


class PatchEvidence(BaseModel):
    source: Source
    regime: Regime | None = None
    regime_basis: Literal["primary_82_case_panel", "local_aligned_borderline"] | None = None
    best_position_class: PositionClass | None = None
    best_position_label: str | None = None
    best_layer: int | None = None
    best_rescue_fraction: float | None = None
    prompt_lcp_rescue_fraction: float | None = None
    prompt_lcp_best_layer: int | None = None
    final_context_rescue_fraction: float | None = None
    generated_prefix_rescue_fraction: float | None = None
    aligned_prompt_control_max: float | None = None
    aligned_prompt_full_count: int | None = None
    wave: str | None = None
    heatmap_path: str | None = None


class SuspectedControllingSpan(BaseModel):
    source: Source = "derived"
    token_indices_a: list[int]
    token_indices_b: list[int]
    confidence: Literal["edit_boundary", "trajectory_state", "ambiguous"]
    method: str


class SelectionProvenance(BaseModel):
    source: Source = "observed"
    wave: str | None = None
    pool: Literal[
        "hand_selected_v1",
        "hand_selected_v2",
        "hand_selected_v3",
        "reverse_v4",
        "heldout_v5",
        "recommended_casebook",
        "local_aligned",
    ]
    archetype: str | None = None


class ArtifactRef(BaseModel):
    path: str
    sha256: str | None = None
    size_bytes: int | None = None


class BranchCard(BaseModel):
    schema_version: str = "branchcard/0.1"
    id: str
    generated_at: datetime

    run_a: RunSide
    run_b: RunSide
    edit: Edit
    branch: Branch
    replay: Replay
    patch_evidence: PatchEvidence
    suspected_controlling_span: SuspectedControllingSpan
    selection_provenance: SelectionProvenance
    caveats: list[str] = Field(default_factory=list)
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)

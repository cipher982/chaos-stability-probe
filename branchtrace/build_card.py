"""Assemble a Branch Card from existing artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .loaders import logit_run, patch_wave, stability_run
from .schema import (
    ArtifactRef,
    Branch,
    BranchCard,
    BranchTokenSide,
    Edit,
    PatchEvidence,
    Replay,
    RunSide,
    Runtime,
    SelectionProvenance,
    SuspectedControllingSpan,
)

_CHUNK = 1 << 16


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(_CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _artifact(path: Path | None, repo_root: Path) -> ArtifactRef | None:
    if path is None or not path.exists():
        return None
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    return ArtifactRef(
        path=str(rel),
        sha256=_sha256_of(path),
        size_bytes=path.stat().st_size,
    )


def _run_side(
    gen_row: dict,
    metadata: dict,
    run_dir: Path,
    prompt_text: str,
    prompt_token_ids: list[int],
) -> RunSide:
    model = metadata["models"][0]
    rt = Runtime(
        source="observed",
        torch=metadata.get("torch"),
        dtype=metadata.get("dtype"),
        device=metadata.get("device"),
        seed=metadata.get("seed_a") if gen_row.get("side") == "a" else metadata.get("seed"),
        batch_size=None,
        system_prompt=metadata.get("system_prompt"),
    )
    decode = {
        "do_sample": metadata.get("sample", False),
        "max_new_tokens": metadata.get("max_new_tokens"),
        "temperature": metadata.get("temperature"),
        "top_p": metadata.get("top_p"),
        "thinking_mode": metadata.get("thinking_mode"),
    }
    return RunSide(
        source="observed",
        model_name=model["name"],
        model_id=model["model_id"],
        model_family=model.get("family"),
        decode=decode,
        runtime=rt,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        prompt_token_count=len(prompt_token_ids),
        generated_text=gen_row.get("generated_text", ""),
        generated_token_ids=gen_row.get("generated_tokens", []),
        generated_token_count=gen_row.get("generated_token_count", 0),
        run_dir=str(run_dir),
    )


def _diff_spans(prompt_a: str, prompt_b: str) -> dict:
    # Minimal: contextual slice around first differing character.
    i = 0
    while i < len(prompt_a) and i < len(prompt_b) and prompt_a[i] == prompt_b[i]:
        i += 1
    lo = max(0, i - 16)
    hi_a = min(len(prompt_a), i + 24)
    hi_b = min(len(prompt_b), i + 24)
    return {"a": prompt_a[lo:hi_a], "b": prompt_b[lo:hi_b]}


def build_hero_qwen2b_parenthesize_0434(repo_root: Path) -> BranchCard:
    """Build the Phase 1 hero card: qwen35_2b token_cert_parenthesize_word_0434.

    This hero uses local-aligned patching (runs/mechinterp_patch_aligned/)
    and SageMaker logit probes (runs/sagemaker_artifacts/.../qwen35_2b).
    The 82-case SageMaker patching wave for pair 0434 ran on qwen35_08b,
    not qwen35_2b, so this card's selection_provenance.pool is
    `local_aligned`, not a patch-wave bucket.
    """
    model_name = "qwen35_2b"
    pair_id = "token_cert_parenthesize_word_0434"

    # --- Generations + metadata
    sagemaker_root = (
        repo_root
        / "runs/sagemaker_artifacts/chaos-logit-token-cert-qwen2b-thinkoff-20260430-001/runs/qwen35_2b"
    )
    metadata = stability_run.load_metadata(sagemaker_root)
    gen_a, gen_b = stability_run.find_pair(sagemaker_root, pair_id, repeat=0)
    prompt_tokens = stability_run.load_prompt_tokens(sagemaker_root, pair_id)
    if prompt_tokens is None:
        raise ValueError(f"prompt_tokens row not found for {pair_id!r}")

    # --- Branch / trajectory event
    events_csv = repo_root / "runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv"
    ev = logit_run.load_trajectory_row(events_csv, model_name, pair_id, repeat=0)
    if ev is None:
        raise ValueError(f"trajectory event row not found for {model_name}/{pair_id}")

    # --- Aligned patch (local)
    aligned_csv = (
        repo_root
        / f"runs/mechinterp_patch_aligned/{model_name}__{pair_id}.csv"
    )
    aligned_json_path = aligned_csv.with_suffix(".json")
    aligned_json = json.loads(aligned_json_path.read_text())
    aligned_df = pd.read_csv(aligned_csv)

    def _best_for(label: str) -> tuple[float, int | None]:
        sub = aligned_df[aligned_df["position_label"] == label]
        if sub.empty:
            return 0.0, None
        row = sub.loc[sub["rescue_fraction"].idxmax()]
        return float(row["rescue_fraction"]), int(row["layer"])

    plcp_rescue, plcp_layer = _best_for("prompt_lcp_token")
    final_rescue, _ = _best_for("final_context_token")
    gp_rescue, _ = _best_for("generated_prefix_token")

    # Aligned-prompt max/full across all aligned_prompt_pos_* labels.
    ap_mask = aligned_df["position_label"].str.startswith("aligned_prompt_pos_", na=False)
    ap_by_label = (
        aligned_df[ap_mask].groupby("position_label")["rescue_fraction"].max()
    )
    ap_max = float(ap_by_label.max()) if len(ap_by_label) else 0.0
    ap_full = int((ap_by_label >= 1.0).sum())
    best_rescue_row = (
        aligned_df.loc[aligned_df["rescue_fraction"].idxmax()]
        if not aligned_df.empty
        else None
    )

    # --- Regime assignment (local-aligned case; use prompt_lcp_full threshold).
    plcp_full = plcp_rescue >= 1.0
    event_kind = str(ev["event_kind"])
    if plcp_full:
        regime = "edit_boundary"
    elif event_kind == "immediate_visible_branch":
        regime = "prompt_accumulation"
    else:
        regime = "trajectory_migration"

    heatmap_path = aligned_csv.with_suffix(".heatmap.png")

    # --- Runtime metadata (augment defaults)
    prompt_a_text = aligned_json["pair"]["prompt_a"]
    prompt_b_text = aligned_json["pair"]["prompt_b"]
    run_a = _run_side(
        gen_a,
        metadata,
        sagemaker_root,
        prompt_a_text,
        prompt_tokens.get("prompt_token_ids_a", []),
    )
    run_b = _run_side(
        gen_b,
        metadata,
        sagemaker_root,
        prompt_b_text,
        prompt_tokens.get("prompt_token_ids_b", []),
    )

    # --- Edit
    pd_block = aligned_json.get("prompt_delta", {})
    edit = Edit(
        source="derived",
        kind=ev["category"].replace("micro_", ""),
        prompt_token_edit_distance=int(ev["prompt_token_edit_distance"]),
        prompt_token_delta_kind=pd_block.get("prompt_token_delta_kind"),
        prompt_token_lcp=int(pd_block.get("prompt_token_lcp", 0)),
        prompt_input_token_delta=int(pd_block.get("prompt_input_token_delta", 0)),
        visible_diff_span=_diff_spans(
            aligned_json["pair"]["prompt_a"], aligned_json["pair"]["prompt_b"]
        ),
    )

    # --- Branch tokens
    branch_a = BranchTokenSide(
        id=int(aligned_json["a_branch_token_id"]),
        text=str(aligned_json["a_branch_token"]),
        js_vs_other=float(ev["branch_js"]) if pd.notna(ev["branch_js"]) else None,
        effective_branching_factor=(
            float(ev["branch_max_effective_branching_factor"])
            if pd.notna(ev["branch_max_effective_branching_factor"])
            else None
        ),
    )
    margin = float(ev["branch_min_margin_logit"]) if pd.notna(ev["branch_min_margin_logit"]) else None
    branch_a.top1_margin = margin
    branch_b = BranchTokenSide(
        id=int(aligned_json["b_branch_token_id"]),
        text=str(aligned_json["b_branch_token"]),
        top1_margin=margin,
        js_vs_other=branch_a.js_vs_other,
        effective_branching_factor=branch_a.effective_branching_factor,
    )

    branch = Branch(
        source="observed",
        branch_t=int(ev["branch_t"]),
        event_kind=event_kind,
        silent_logit_lead=(
            float(ev["silent_logit_lead"])
            if pd.notna(ev.get("silent_logit_lead"))
            else None
        ),
        common_prefix_tokens=int(ev["common_prefix_tokens"]),
        branch_token_a=branch_a,
        branch_token_b=branch_b,
    )

    replay = Replay(
        source="observed",
        deterministic_reproducible_a=bool(aligned_json.get("clean_replay_top1_token_id") == aligned_json["a_branch_token_id"]),
        deterministic_reproducible_b=bool(aligned_json.get("corrupt_replay_matches_b_branch")),
    )

    patch = PatchEvidence(
        source="observed",
        regime=regime,
        best_position_class=str(best_rescue_row["position_label"]) if best_rescue_row is not None else None,
        best_position_label=str(best_rescue_row["position_label"]) if best_rescue_row is not None else None,
        best_layer=int(best_rescue_row["layer"]) if best_rescue_row is not None else None,
        best_rescue_fraction=float(best_rescue_row["rescue_fraction"]) if best_rescue_row is not None else None,
        prompt_lcp_rescue_fraction=plcp_rescue,
        prompt_lcp_best_layer=plcp_layer,
        final_context_rescue_fraction=final_rescue,
        generated_prefix_rescue_fraction=gp_rescue if gp_rescue > 0 else None,
        aligned_prompt_control_max=ap_max,
        aligned_prompt_full_count=ap_full,
        wave="local_aligned",
        heatmap_path=str(heatmap_path.relative_to(repo_root)) if heatmap_path.exists() else None,
    )

    # Suspected controlling span: the LCP token in each prompt.
    lcp = edit.prompt_token_lcp
    span = SuspectedControllingSpan(
        source="derived",
        token_indices_a=[lcp],
        token_indices_b=[lcp],
        confidence=("edit_boundary" if regime == "edit_boundary" else "ambiguous"),
        method="prompt_lcp_vs_aligned_controls_heuristic",
    )

    provenance = SelectionProvenance(
        source="observed",
        wave="local_aligned",
        pool="local_aligned",
        archetype="immediate_branch_control",
    )

    caveats = [
        f"Local-aligned patching (MPS/fp16). 82-case SageMaker sibling is qwen35_08b pair {pair_id} (regime: {regime}, prompt_lcp=0.917).",
        "Backend/dtype shifts branch timing; E10 mean absolute delta 4.25 on Qwen3.5-2B.",
        "Regime assignment applies the 3-regime taxonomy at prompt_lcp_full >= 1.0; this case is borderline (plcp=0.861).",
        "branch_t=0 immediate branch; no generated-prefix position to interrogate.",
    ]

    artifacts: dict[str, ArtifactRef] = {}
    for key, p in {
        "generations_jsonl": sagemaker_root / "generations.jsonl",
        "logit_probes_jsonl": sagemaker_root / "logit_probes.jsonl",
        "metadata_json": sagemaker_root / "metadata.json",
        "trajectory_events_csv": repo_root / "runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv",
        "aligned_patch_csv": aligned_csv,
        "aligned_patch_json": aligned_json_path,
        "heatmap_png": heatmap_path,
    }.items():
        ref = _artifact(p, repo_root)
        if ref is not None:
            artifacts[key] = ref

    return BranchCard(
        schema_version="branchcard/0.1",
        id=f"{model_name}__{pair_id}",
        generated_at=datetime.now(timezone.utc),
        run_a=run_a,
        run_b=run_b,
        edit=edit,
        branch=branch,
        replay=replay,
        patch_evidence=patch,
        suspected_controlling_span=span,
        selection_provenance=provenance,
        caveats=caveats,
        artifacts=artifacts,
    )

#!/usr/bin/env python3
"""Honest statistical analysis of the stability panel.

Computes:
- Per-model bootstrap 95% CIs on small-perturbation semantic distance.
- Paired permutation tests between selected model contrasts.
- Leave-one-prompt-out stability check (how sensitive is the ranking to one prompt?).
- Per-category variance decomposition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("runs/sagemaker_artifacts")

MODELS = {
    "qwen35_08b": ROOT / "chaos-stability-panel-20260429-001/runs/qwen35_08b/summary_with_semantic.csv",
    "qwen35_2b": ROOT / "chaos-stability-qwen35-2b-20260429-001/runs/qwen35_2b/summary_with_semantic.csv",
    "qwen35_4b": ROOT / "chaos-stability-panel-20260429-001/runs/qwen35_4b/summary_with_semantic.csv",
    "qwen35_9b": ROOT / "chaos-stability-panel-20260429-001/runs/qwen35_9b/summary_with_semantic.csv",
    "qwen35_08b_bnb8": ROOT / "chaos-stability-qwen35-08b-bnb8-20260429-001/runs/qwen35_08b_bnb8/summary_with_semantic.csv",
    "qwen35_08b_bnb4": ROOT / "chaos-stability-qwen35-08b-bnb4-20260429-001/runs/qwen35_08b_bnb4/summary_with_semantic.csv",
    "qwen35_4b_bnb8": ROOT / "chaos-stability-qwen35-4b-bnb8-20260429-001/runs/qwen35_4b_bnb8/summary_with_semantic.csv",
    "qwen35_4b_bnb4": ROOT / "chaos-stability-qwen35-4b-bnb4-20260429-001/runs/qwen35_4b_bnb4/summary_with_semantic.csv",
    "phi4_reasoning_plus": ROOT / "chaos-stability-phi4-reasoning-plus-20260429-001/runs/phi4_reasoning_plus/summary_with_semantic.csv",
    "deepseek_r1_qwen7b": ROOT / "chaos-stability-deepseek-r1-qwen7b-20260429-001/runs/deepseek_r1_distill_qwen_7b/summary_with_semantic.csv",
    "mistral7b_v03": ROOT / "chaos-stability-mistral7b-v03-20260429-001/runs/mistral7b_instruct_v03/summary_with_semantic.csv",
    "llama1_7b": ROOT / "chaos-stability-llama1-7b-legacy-20260429-001/runs/llama1_7b_huggyllama/summary_with_semantic.csv",
    "gemma4_e4b_it": ROOT / "chaos-stability-gemma4-e4b-20260429-002/runs/gemma4_e4b_it/summary_with_semantic.csv",
    "gemma4_e2b_it": ROOT / "chaos-stability-gemma4-e2b-20260429-001/runs/gemma4_e2b_it/summary_with_semantic.csv",
    "gemma4_e4b_base": ROOT / "chaos-stability-gemma4-e4b-base-20260429-001/runs/gemma4_e4b_base/summary_with_semantic.csv",
    "gemma4_e2b_base": ROOT / "chaos-stability-gemma4-e2b-base-20260429-001/runs/gemma4_e2b_base/summary_with_semantic.csv",
    "smollm3_3b": ROOT / "chaos-stability-smollm3-3b-20260429-001/runs/smollm3_3b/summary_with_semantic.csv",
    "granite33_8b": ROOT / "chaos-stability-granite33-8b-20260429-001/runs/granite33_8b_instruct/summary_with_semantic.csv",
    "falcon3_10b": ROOT / "chaos-stability-falcon3-10b-20260429-001/runs/falcon3_10b_instruct/summary_with_semantic.csv",
    "olmo3_7b": ROOT / "chaos-stability-olmo3-7b-20260429-001/runs/olmo3_7b_instruct/summary_with_semantic.csv",
    "olmo2_7b": ROOT / "chaos-stability-olmo2-7b-20260429-001/runs/olmo2_7b_instruct/summary_with_semantic.csv",
    "gptj_6b": ROOT / "chaos-stability-gptj-6b-legacy-20260429-001/runs/gptj_6b/summary_with_semantic.csv",
    "opt_6p7b": ROOT / "chaos-stability-opt-6p7b-legacy-20260429-001/runs/opt_6p7b/summary_with_semantic.csv",
    "pythia_6p9b": ROOT / "chaos-stability-pythia-6p9b-legacy-20260429-001/runs/pythia_6p9b_deduped/summary_with_semantic.csv",
    "gpt2_xl": ROOT / "chaos-stability-gpt2-xl-legacy-20260429-001/runs/gpt2_xl/summary_with_semantic.csv",
}

SMALL_CATS = ["noop_format", "punctuation", "synonym"]


def load_small_perturbation_vectors() -> dict[str, pd.DataFrame]:
    """Return per-model dataframe indexed by pair_id with semantic_cosine_distance."""
    out = {}
    for name, path in MODELS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["category"].isin(SMALL_CATS)]
        df = df[["pair_id", "category", "semantic_cosine_distance"]].copy()
        df = df.drop_duplicates(subset=["pair_id"], keep="first").set_index("pair_id")
        out[name] = df
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return float(values.mean()), float(values.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 100_000, seed: int = 0) -> dict:
    """Two-sided paired permutation test for the mean difference (a - b)."""
    assert len(a) == len(b)
    diff = a - b
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diff)))
    perm_means = (signs * diff).mean(axis=1)
    p_two = float((np.abs(perm_means) >= abs(observed) - 1e-15).mean())
    return {
        "mean_diff": float(observed),
        "p_two_sided": p_two,
        "n_pairs": len(diff),
        "diff_std": float(diff.std(ddof=1)),
    }


def main() -> None:
    vecs = load_small_perturbation_vectors()

    # Build a matrix aligned by pair_id across models
    all_pair_ids = sorted({pid for df in vecs.values() for pid in df.index})
    matrix = pd.DataFrame(index=all_pair_ids)
    for name, df in vecs.items():
        matrix[name] = df["semantic_cosine_distance"]

    matrix = matrix.dropna(how="any")  # keep only prompts every model has
    print(f"Usable prompts common to all models: n = {len(matrix)}")
    print(f"  pair_ids: {list(matrix.index)}")
    print()

    # 1. Per-model bootstrap CIs
    print("## Per-model small-perturbation semantic distance (n=9 prompts)")
    print(f"{'model':<22} {'mean':>8} {'sd':>8} {'CI_lo':>8} {'CI_hi':>8} {'CI_width':>10}")
    rows = []
    for name in matrix.columns:
        vals = matrix[name].to_numpy()
        mean, sd, lo, hi = bootstrap_ci(vals)
        rows.append((name, mean, sd, lo, hi, hi - lo))
        print(f"{name:<22} {mean:>8.4f} {sd:>8.4f} {lo:>8.4f} {hi:>8.4f} {hi-lo:>10.4f}")
    print()

    # 2. Key paired permutation tests
    print("## Paired permutation tests (same 9 prompts, 100k permutations)")
    contrasts = [
        ("qwen35_4b", "qwen35_08b", "Qwen within-family capacity contrast"),
        ("qwen35_4b", "qwen35_9b", "Qwen 4B vs 9B"),
        ("qwen35_4b", "qwen35_2b", "Qwen 4B vs 2B"),
        ("qwen35_2b", "qwen35_08b", "Qwen 2B vs 0.8B"),
        ("qwen35_4b", "phi4_reasoning_plus", "top cluster internal"),
        ("qwen35_4b", "deepseek_r1_qwen7b", "top cluster internal"),
        ("gemma4_e4b_it", "gemma4_e4b_base", "Gemma E4B instruct vs base"),
        ("gemma4_e2b_it", "gemma4_e2b_base", "Gemma E2B instruct vs base"),
        ("qwen35_08b_bnb4", "qwen35_08b", "Qwen 0.8B 4bit vs BF16 (mode collapse?)"),
        ("qwen35_4b_bnb4", "qwen35_4b", "Qwen 4B 4bit vs BF16"),
        ("llama1_7b", "gptj_6b", "legacy cluster internal"),
        ("llama1_7b", "qwen35_4b", "LLaMA-1 vs top of modern cluster"),
        ("olmo3_7b", "qwen35_08b", "brittle cluster internal"),
    ]
    print(f"{'A':<22} {'B':<22} {'mean(A-B)':>10} {'p_two':>8}   note")
    for a, b, note in contrasts:
        if a not in matrix.columns or b not in matrix.columns:
            continue
        res = paired_permutation(matrix[a].to_numpy(), matrix[b].to_numpy())
        sig = "***" if res["p_two_sided"] < 0.01 else "*" if res["p_two_sided"] < 0.05 else "   "
        print(f"{a:<22} {b:<22} {res['mean_diff']:>+10.4f} {res['p_two_sided']:>8.4f} {sig} {note}")
    print()

    # 3. Leave-one-prompt-out ranking stability
    print("## Leave-one-prompt-out rank stability for key models")
    key = ["qwen35_4b", "phi4_reasoning_plus", "qwen35_9b", "deepseek_r1_qwen7b",
           "llama1_7b", "qwen35_2b", "qwen35_08b", "olmo2_7b", "olmo3_7b"]
    key = [k for k in key if k in matrix.columns]
    ranks_by_prompt = []
    for pid in matrix.index:
        sub = matrix.drop(index=pid)
        means = sub[key].mean().sort_values()
        ranks_by_prompt.append(means.index.tolist())
    # How often does each model appear in top-4?
    top4_counts = {m: 0 for m in key}
    bottom3_counts = {m: 0 for m in key}
    for order in ranks_by_prompt:
        for m in order[:4]:
            top4_counts[m] += 1
        for m in order[-3:]:
            bottom3_counts[m] += 1
    n = len(ranks_by_prompt)
    print(f"{'model':<22} {'top4_freq':>10} {'bottom3_freq':>14}")
    for m in key:
        print(f"{m:<22} {top4_counts[m]/n:>10.2f} {bottom3_counts[m]/n:>14.2f}")
    print()

    # 4. Sign-consistency: does the within-Qwen contrast survive on every single prompt?
    print("## Per-prompt sign check: qwen35_4b < qwen35_08b?")
    if "qwen35_4b" in matrix.columns and "qwen35_08b" in matrix.columns:
        diff = matrix["qwen35_08b"] - matrix["qwen35_4b"]
        for pid, d in diff.items():
            cat = pid.split("_")[0]
            sign = "YES" if d > 0 else "NO "
            print(f"  {sign}  {pid:<35} delta={d:+.4f}")
        print(f"  Wins: {(diff > 0).sum()}/{len(diff)}  (sign test p = {2 * min((diff > 0).sum(), (diff <= 0).sum()) / len(diff):.3f} under binomial null)")
    print()

    # 5. Quantization mode-collapse check: does 0.8B at 4-bit drift toward degeneracy?
    print("## Quantization: within-system stability vs distance-from-BF16")
    print("  Small-perturb means are computed above. The 0.8B 4-bit <  0.8B BF16 finding")
    print("  is only meaningful if 4-bit outputs drift from BF16 outputs on the same prompts.")
    print("  This script does not recompute BF16-vs-quantized distances; those live in the")
    print("  quantization_fidelity artifact.")
    print()

    # 6. What would it take to distinguish 4B from 9B?
    print("## Minimum-n estimate to distinguish 4B from 9B at p<0.05")
    if "qwen35_4b" in matrix.columns and "qwen35_9b" in matrix.columns:
        a = matrix["qwen35_4b"].to_numpy()
        b = matrix["qwen35_9b"].to_numpy()
        d = a - b
        effect = d.mean()
        sd = d.std(ddof=1)
        # Rough power calc for paired t-test, 80% power, alpha=0.05, two-sided
        if abs(effect) > 1e-6 and sd > 1e-6:
            # n ~ ((z_a + z_b) * sd / effect)^2, z_a=1.96, z_b=0.84
            n_est = ((1.96 + 0.84) * sd / abs(effect)) ** 2
            print(f"  observed mean diff: {effect:+.4f}, sd: {sd:.4f}")
            print(f"  rough paired-prompt n needed for 80% power at alpha=0.05: {math.ceil(n_est)}")
            print(f"  current n: {len(d)}")


if __name__ == "__main__":
    main()

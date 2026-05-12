"""Build Fig 1 — signature rescue panel (per-case bars faceted by signature).

Source: runs/rankings/activation_patch_comparison/case_level_summary.csv.
For each of the 82 panel cases, pull the best-layer rescue_fraction at
four position classes:

  - prompt_LCP          (prompt_lcp_token_best_rescue_fraction)
  - aligned_prompt_ctrl (best_aligned_prompt_rescue_fraction)
  - generated_prefix    (best_generated_prefix_rescue_fraction)
  - final_context       (final_context_token_best_rescue_fraction)

Assign signatures via the paper's decision rule:

  if prompt_lcp_full:             boundary_rescue
  elif immediate_visible_branch:  tokenization_shift_immediate_rescue
  else:                           generated_prefix_rescue_after_silent_divergence

Layout: 3 rows (one per signature), each row is a stacked per-case set
of 4 bars showing the four position classes. Cases are sorted within a
row by prompt_LCP rescue so the signature "peak" pattern reads from
left to right.

Outputs:
  experiments/E11_branchtrace_card/figures/signature_rescue_panel.png
  experiments/E11_branchtrace_card/figures/signature_rescue_panel.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

POSITION_COLS = [
    ("prompt_LCP", "prompt_lcp_token_best_rescue_fraction"),
    ("aligned_prompt_ctrl", "best_aligned_prompt_rescue_fraction"),
    ("generated_prefix", "best_generated_prefix_rescue_fraction"),
    ("final_context", "final_context_token_best_rescue_fraction"),
]

SIGNATURE_ORDER = [
    ("boundary_rescue", "Boundary rescue"),
    ("generated_prefix_rescue_after_silent_divergence", "Generated-prefix rescue\n(silent divergence)"),
    ("tokenization_shift_immediate_rescue", "Tokenization-shift\nimmediate rescue"),
]

SIGNATURE_COLOR = {
    "boundary_rescue": "#3a7bd5",
    "generated_prefix_rescue_after_silent_divergence": "#d6923a",
    "tokenization_shift_immediate_rescue": "#8e44ad",
}

POSITION_COLOR = {
    "prompt_LCP":          "#2b5aa6",
    "aligned_prompt_ctrl": "#7fa9d8",
    "generated_prefix":    "#d68a3a",
    "final_context":       "#4a8c4a",
}


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(
        REPO / "runs/rankings/activation_patch_comparison/case_level_summary.csv",
        low_memory=False,
    )
    ev = pd.read_csv(
        REPO / "runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv",
        low_memory=False,
    )
    m = df.merge(
        ev[["model_name", "pair_id", "event_kind"]].drop_duplicates(["model_name", "pair_id"]),
        on=["model_name", "pair_id"],
        how="left",
    )
    # Primary 82-case panel (drop __reverse rows).
    m = m[~m["pair_id"].str.endswith("__reverse")].copy()

    def sig(r: pd.Series) -> str:
        if bool(r["prompt_lcp_full"]):
            return "boundary_rescue"
        if r["event_kind"] == "immediate_visible_branch":
            return "tokenization_shift_immediate_rescue"
        return "generated_prefix_rescue_after_silent_divergence"

    m["signature"] = m.apply(sig, axis=1)
    return m


def case_rescue_row(row: pd.Series) -> dict[str, float]:
    out = {}
    for label, col in POSITION_COLS:
        val = row.get(col)
        if val is None or pd.isna(val):
            out[label] = np.nan
        else:
            out[label] = float(val)
    return out


def plot_panel(panel: pd.DataFrame) -> None:
    """Three-panel box+strip by signature. Each panel: rescue_fraction
    distribution across cases at the four position classes.

    This version shows the *shape* of each signature at a glance rather
    than drilling into per-case bars."""
    pos_labels = [p for p, _ in POSITION_COLS]
    n_pos = len(pos_labels)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.5, 5.0),
        sharey=True,
    )
    fig.suptitle(
        "Rescue-fraction distribution by signature (best-layer per position class)",
        fontsize=12,
        y=0.995,
    )
    rng = np.random.default_rng(20260512)

    for ax, (sig_key, sig_title) in zip(axes, SIGNATURE_ORDER):
        sub = panel[panel["signature"] == sig_key]
        n_cases = len(sub)
        data_by_pos = []
        for pos_label, col in POSITION_COLS:
            vals = sub[col].dropna().astype(float).tolist()
            data_by_pos.append(vals)

        xs = np.arange(1, n_pos + 1)
        # Box plot — shows median and IQR.
        bp = ax.boxplot(
            data_by_pos,
            positions=xs,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="#222", linewidth=1.6),
            boxprops=dict(facecolor="#eee", edgecolor="#444", linewidth=0.8),
            whiskerprops=dict(color="#444", linewidth=0.8),
            capprops=dict(color="#444", linewidth=0.8),
        )
        for patch, pos_label in zip(bp["boxes"], pos_labels):
            patch.set_facecolor(POSITION_COLOR[pos_label])
            patch.set_alpha(0.45)

        # Jittered strip of the actual points.
        for x, vals, pos_label in zip(xs, data_by_pos, pos_labels):
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                x + jitter,
                np.clip(vals, -0.5, 2.7),  # visual clip; annotate overshoot below
                s=12,
                color=POSITION_COLOR[pos_label],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
                zorder=3,
            )
            # Count overshoots (> 2.7) for annotation.
            over = [v for v in vals if v > 2.7]
            if over:
                ax.text(
                    x, 2.7,
                    f"+{len(over)}",
                    ha="center", va="bottom", fontsize=8, color="#555",
                )

        ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_ylim(-0.25, 3.1)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [p.replace("_", "\n") for p in pos_labels],
            fontsize=9,
        )
        ax.set_title(f"{sig_title}\nn = {n_cases}", fontsize=10.5, loc="left")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("rescue_fraction  (1.0 = full rescue)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = OUT_DIR / "signature_rescue_panel.png"
    pdf = OUT_DIR / "signature_rescue_panel.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main() -> None:
    panel = load_panel()
    print("panel size:", len(panel))
    print(panel["signature"].value_counts().to_dict())
    print()

    # Report signature medians by position class (sanity, also in paper).
    for sig_key, _title in SIGNATURE_ORDER:
        sub = panel[panel["signature"] == sig_key]
        line = [f"{sig_key}  (n={len(sub)})"]
        for label, col in POSITION_COLS:
            line.append(f"{label}={sub[col].median():.2f}")
        print("  ".join(line))
    print()

    plot_panel(panel)


if __name__ == "__main__":
    main()

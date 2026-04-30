#!/usr/bin/env python3
"""Plot activation-patching rescue fractions as a layer/position heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ordered_positions(df: pd.DataFrame) -> list[str]:
    positions = (
        df[["position_label", "clean_pos", "corrupt_pos"]]
        .drop_duplicates()
        .sort_values(["clean_pos", "corrupt_pos", "position_label"])
    )
    return positions["position_label"].tolist()


def short_label(label: str) -> str:
    label = label.replace("aligned_prompt_pos_", "p")
    label = label.replace("aligned_generated_prefix_pos_", "g")
    label = label.replace("prompt_lcp_token", "lcp")
    label = label.replace("final_context_token", "final")
    label = label.replace("_to_", "->")
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch_csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.2)
    args = parser.parse_args()

    df = pd.read_csv(args.patch_csv)
    positions = ordered_positions(df)
    pivot = df.pivot_table(index="layer", columns="position_label", values="rescue_fraction", aggfunc="max")
    pivot = pivot.reindex(columns=positions)
    values = pivot.to_numpy(dtype=float)
    values = np.clip(values, args.clip_min, args.clip_max)

    fig_width = max(10, min(24, 0.32 * len(positions)))
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis", vmin=args.clip_min, vmax=args.clip_max)
    ax.set_xlabel("patched position")
    ax.set_ylabel("layer")
    ax.set_title(args.title or args.patch_csv.stem)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(layer)) for layer in pivot.index])

    stride = max(1, len(positions) // 30)
    xticks = list(range(0, len(positions), stride))
    if len(positions) - 1 not in xticks:
        xticks.append(len(positions) - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([short_label(positions[i]) for i in xticks], rotation=70, ha="right", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("rescue fraction, clipped")
    fig.tight_layout()

    out = args.out or args.patch_csv.with_suffix(".heatmap.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

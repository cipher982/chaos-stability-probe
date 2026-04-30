#!/usr/bin/env python3
"""Summarize clean/corrupt SAE feature deltas from branch feature extracts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_file(path: Path, top_n: int) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    group_cols = ["model_name", "pair_id", "layer", "position_label"]
    for (model, pair, layer, position), group in df.groupby(group_cols):
        clean = group[group["side"] == "clean"][["feature_id", "activation"]].rename(
            columns={"activation": "clean_activation"}
        )
        corrupt = group[group["side"] == "corrupt"][["feature_id", "activation"]].rename(
            columns={"activation": "corrupt_activation"}
        )
        merged = clean.merge(corrupt, on="feature_id", how="outer").fillna(0)
        merged["delta_clean_minus_corrupt"] = merged["clean_activation"] - merged["corrupt_activation"]
        overlap = len(set(clean["feature_id"]) & set(corrupt["feature_id"]))
        clean_token = group[group["side"] == "clean"]["token_text"].iloc[0]
        corrupt_token = group[group["side"] == "corrupt"]["token_text"].iloc[0]

        slices = [
            ("clean_over_corrupt", merged.sort_values("delta_clean_minus_corrupt", ascending=False).head(top_n)),
            ("corrupt_over_clean", merged.sort_values("delta_clean_minus_corrupt").head(top_n)),
        ]
        for side, items in slices:
            for rank, item in enumerate(items.itertuples(index=False), start=1):
                rows.append(
                    {
                        "source_csv": str(path),
                        "model_name": model,
                        "pair_id": pair,
                        "layer": int(layer),
                        "position_label": position,
                        "clean_token_text": clean_token,
                        "corrupt_token_text": corrupt_token,
                        "topk_overlap": overlap,
                        "delta_side": side,
                        "delta_rank": rank,
                        "feature_id": int(item.feature_id),
                        "clean_activation": float(item.clean_activation),
                        "corrupt_activation": float(item.corrupt_activation),
                        "delta_clean_minus_corrupt": float(item.delta_clean_minus_corrupt),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sae_dir", type=Path, nargs="?", default=Path("runs/mechinterp_sae"))
    parser.add_argument("--out", type=Path, default=Path("runs/mechinterp_sae/sae_feature_delta_summary.csv"))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    paths = sorted(args.sae_dir.glob("*__sae_features.csv"))
    if not paths:
        raise SystemExit(f"No SAE feature extracts found in {args.sae_dir}")

    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(summarize_file(path, args.top_n))

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    print(
        out.groupby(["pair_id", "layer", "position_label"])["topk_overlap"]
        .first()
        .to_string()
    )


if __name__ == "__main__":
    main()

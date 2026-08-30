#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v049_l2_fine_shrink_resweep.py
=====================================

v048에서 이미 생성한 temporal-transfer correction을 그대로 사용한다.
ExtraTrees 재학습은 전혀 하지 않는다.

목적:
    v048 coarse grid의 최솟값 0.25가 최상이라,
    0.25 아래/주변의 shrink를 더 촘촘히 확인.

입력:
    validation/cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep/
        transfer_predictions.csv.gz

출력:
    validation/cb_reg_v049_l2_fine_shrink_resweep/
        fine_by_year.csv
        fine_summary.csv
        selected_candidate.json

선택 우선순위:
    pass gate
    -> 2024 gain
    -> recent 2023-2024
    -> pooled 2022-2024

Gate:
    pooled > 0
    recent > 0
    2024 > 0
    temporal wins >= 2/3
    2024 seed wins >= 2/3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")

SOURCE_DIR = (
    ROOT
    / "validation"
    / "cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep"
)

SOURCE_PATH = (
    SOURCE_DIR
    / "transfer_predictions.csv.gz"
)

OUT_DIR = (
    ROOT
    / "validation"
    / "cb_reg_v049_l2_fine_shrink_resweep"
)

ANCHOR_TEAM_ID = 13

SEEDS = [
    42,
    2026,
    314,
]

ROUTES = [
    "R_CORE_ONLY",
    "ALL_R",
    "R_ANCHOR_ONLY",
]

# 0.25 was the lower boundary in v048.
# Explore below and around it without excessive micro-tuning.
SHRINK_GRID = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.35,
    0.40,
    0.45,
]


def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def bss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    mean_y = float(np.mean(y))
    ref = mean_y * (1.0 - mean_y)

    if ref <= 0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier(y, p) / ref
        )
    )


def domain_masks(df):
    gt = (
        df["game_type"]
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )

    is_f = gt == "F"

    anchor = (
        df["pitcher_team_id"]
        .eq(ANCHOR_TEAM_ID)
        .to_numpy()
        |
        df["batter_team_id"]
        .eq(ANCHOR_TEAM_ID)
        .to_numpy()
    )

    return {
        "F": is_f,
        "R": ~is_f,
        "R_CORE": (~is_f) & (~anchor),
        "R_ANCHOR": (~is_f) & anchor,
    }


def route_mask(df, route):
    m = domain_masks(df)

    if route == "R_CORE_ONLY":
        return m["R_CORE"]

    if route == "ALL_R":
        return m["R"]

    if route == "R_ANCHOR_ONLY":
        return m["R_ANCHOR"]

    raise ValueError(route)


def apply_correction(
    df,
    correction,
    shrink,
    route,
):
    base = (
        df["v42_pred"]
        .to_numpy(np.float64)
    )

    corr = np.asarray(
        correction,
        dtype=np.float64,
    )

    pred = base.copy()

    mask = route_mask(
        df,
        route,
    )

    pred[mask] = np.clip(
        base[mask]
        + float(shrink)
        * corr[mask],
        0.0,
        1.0,
    )

    # Non-applied rows must remain byte-identical numerically.
    if (~mask).any():
        if not np.array_equal(
            pred[~mask],
            base[~mask],
        ):
            raise RuntimeError(
                f"Protection failed: {route}"
            )

    return pred


def evaluate():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            SOURCE_PATH
        )

    if OUT_DIR.exists():
        raise FileExistsError(
            f"{OUT_DIR} already exists."
        )

    df = pd.read_csv(
        SOURCE_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "target_year",
        "y_true",
        "v42_pred",
        "corr_mean",
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
    }

    for seed in SEEDS:
        required.add(
            f"corr_seed_{seed}"
        )

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    by_year_rows = []
    summary_rows = []

    for route in ROUTES:
        for shrink in SHRINK_GRID:
            year_gains = []

            for year, g in df.groupby(
                "target_year",
                sort=True,
            ):
                y = (
                    g["y_true"]
                    .to_numpy(np.float64)
                )

                base = (
                    g["v42_pred"]
                    .to_numpy(np.float64)
                )

                pred = apply_correction(
                    g,
                    g["corr_mean"].to_numpy(
                        np.float64
                    ),
                    shrink,
                    route,
                )

                gain = float(
                    bss(y, pred)
                    - bss(y, base)
                )

                year_gains.append(
                    gain
                )

                by_year_rows.append({
                    "route": route,
                    "shrink": float(shrink),
                    "year": int(year),
                    "gain_vs_v42_bss": gain,
                })

            y_all = (
                df["y_true"]
                .to_numpy(np.float64)
            )

            base_all = (
                df["v42_pred"]
                .to_numpy(np.float64)
            )

            pred_all = apply_correction(
                df,
                df["corr_mean"]
                .to_numpy(np.float64),
                shrink,
                route,
            )

            years = (
                df["target_year"]
                .to_numpy(int)
            )

            recent = np.isin(
                years,
                [2023, 2024],
            )

            latest = years == 2024

            pooled_gain = float(
                bss(y_all, pred_all)
                - bss(y_all, base_all)
            )

            recent_gain = float(
                bss(
                    y_all[recent],
                    pred_all[recent],
                )
                - bss(
                    y_all[recent],
                    base_all[recent],
                )
            )

            latest_gain = float(
                bss(
                    y_all[latest],
                    pred_all[latest],
                )
                - bss(
                    y_all[latest],
                    base_all[latest],
                )
            )

            temporal_wins = int(
                np.sum(
                    np.asarray(
                        year_gains
                    ) > 0.0
                )
            )

            g24 = (
                df.loc[
                    df["target_year"]
                    .eq(2024)
                ]
                .reset_index(drop=True)
            )

            seed_gains = []

            for seed in SEEDS:
                pred_seed = apply_correction(
                    g24,
                    g24[
                        f"corr_seed_{seed}"
                    ].to_numpy(
                        np.float64
                    ),
                    shrink,
                    route,
                )

                y24 = (
                    g24["y_true"]
                    .to_numpy(np.float64)
                )

                b24 = (
                    g24["v42_pred"]
                    .to_numpy(np.float64)
                )

                seed_gain = float(
                    bss(y24, pred_seed)
                    - bss(y24, b24)
                )

                seed_gains.append(
                    seed_gain
                )

            seed_wins = int(
                np.sum(
                    np.asarray(
                        seed_gains
                    ) > 0.0
                )
            )

            passed = bool(
                pooled_gain > 0.0
                and recent_gain > 0.0
                and latest_gain > 0.0
                and temporal_wins >= 2
                and seed_wins >= 2
            )

            summary_rows.append({
                "route": route,
                "shrink": float(shrink),
                "pooled_2022_2024_gain_bss": pooled_gain,
                "recent_2023_2024_gain_bss": recent_gain,
                "gain_2024_bss": latest_gain,
                "temporal_wins": temporal_wins,
                "min_temporal_gain_bss": float(
                    np.min(year_gains)
                ),
                "mean_temporal_gain_bss": float(
                    np.mean(year_gains)
                ),
                "seed42_2024_gain_bss": float(
                    seed_gains[0]
                ),
                "seed2026_2024_gain_bss": float(
                    seed_gains[1]
                ),
                "seed314_2024_gain_bss": float(
                    seed_gains[2]
                ),
                "seed_wins_2024": seed_wins,
                "pass_gate": passed,
            })

    by_year = pd.DataFrame(
        by_year_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = (
        summary.sort_values(
            [
                "pass_gate",
                "gain_2024_bss",
                "recent_2023_2024_gain_bss",
                "pooled_2022_2024_gain_bss",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    passed = (
        summary.loc[
            summary["pass_gate"]
        ]
        .copy()
    )

    if len(passed) == 0:
        selected = {
            "selected": False,
            "reason": (
                "No candidate passed gate."
            ),
        }

    else:
        best = passed.iloc[0]

        selected = {
            "selected": True,
            "route": str(
                best["route"]
            ),
            "shrink": float(
                best["shrink"]
            ),
            "pooled_gain_bss": float(
                best[
                    "pooled_2022_2024_gain_bss"
                ]
            ),
            "recent_gain_bss": float(
                best[
                    "recent_2023_2024_gain_bss"
                ]
            ),
            "gain_2024_bss": float(
                best["gain_2024_bss"]
            ),
            "temporal_wins": int(
                best["temporal_wins"]
            ),
            "seed_wins_2024": int(
                best["seed_wins_2024"]
            ),
        }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    by_year.to_csv(
        OUT_DIR / "fine_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_DIR / "fine_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        OUT_DIR
        / "selected_candidate.json"
    ).write_text(
        json.dumps(
            selected,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 132
    )
    print(
        "FINE SHRINK — TOP CANDIDATES"
    )
    print(
        "=" * 132
    )

    print(
        summary.head(25)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "=" * 132
    )
    print(
        "SELECTED FINE L2"
    )
    print(
        "=" * 132
    )

    print(
        json.dumps(
            selected,
            ensure_ascii=False,
            indent=2,
        )
    )

    if selected.get(
        "selected",
        False,
    ):
        route = selected["route"]
        shrink = selected["shrink"]

        detail = (
            by_year.loc[
                by_year["route"]
                .eq(route)
                & np.isclose(
                    by_year["shrink"],
                    shrink,
                )
            ]
            .sort_values("year")
        )

        print()
        print(
            "SELECTED BY YEAR"
        )

        print(
            detail.to_string(
                index=False,
                float_format=lambda x: f"{x:+.5f}",
            )
        )


if __name__ == "__main__":
    evaluate()

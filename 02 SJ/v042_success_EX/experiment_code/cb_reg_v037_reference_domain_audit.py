#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v037_reference_domain_audit.py
=====================================

목적
----
reference V20에서 확인한 exact domain3 router를 현재 프로젝트 OOF에 적용해서
현재 L2 / (있다면) L3가 R_CORE, R_ANCHOR, F에서 어떻게 작동하는지 재학습 없이 확인한다.

Reference router
----------------
Router identity는 2022년까지의 F 구조만 사용해 anchor team을 찾는다.

    F:
        game_type == "F"

    R_ANCHOR:
        game_type == "R"
        AND (
            pitcher_team_id == ANCHOR_TEAM_ID
            OR batter_team_id == ANCHOR_TEAM_ID
        )

    R_CORE:
        나머지 R

anchor team:
    pre-2023 F rows에서 모든 경기에 사실상 항상 등장하는 단 하나의 team
    involvement > .999999

재학습
------
NONE

입력
----
data/train.csv

validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/oof_predictions.csv

Optional:
validation/cb_reg_v029_l3_temporal_ridge_on_l1pc_l2/oof_predictions.csv

출력
----
validation/cb_reg_v037_reference_domain_audit/
    domain_counts.csv
    l2_domain_by_seed_year.csv
    l2_domain_summary.csv
    l2_shrink1_domain_summary.csv
    l3_domain_by_seed_year.csv        (v029 있을 때)
    l3_routing_summary.csv            (v029 있을 때)
    metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v037_reference_domain_audit"

ROOT = Path(".")
TRAIN_PATH = ROOT / "data" / "train.csv"

V028_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
    / "oof_predictions.csv"
)

V029_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v029_l3_temporal_ridge_on_l1pc_l2"
    / "oof_predictions.csv"
)

OUT_DIR = ROOT / "validation" / EXPERIMENT_NAME

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"
PITCHER_TEAM = "pitcher_team_id"
BATTER_TEAM = "batter_team_id"

ANCHOR_DISCOVERY_MAX_SEASON = 2022

CURRENT_L2_SHRINK = 0.50
ALT_L2_SHRINK = 1.00

BOUNDARY_TOL = 1e-12


# =============================================================================
# Metrics
# =============================================================================

def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def bss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    score = brier(y, p)
    mu = float(np.mean(y))
    ref = max(mu * (1.0 - mu), 1e-12)

    return float(
        100000.0
        * (
            1.0
            - score / ref
        )
    )


# =============================================================================
# Exact reference router
# =============================================================================

def load_router_frame():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    cols = [
        ID,
        SEASON,
        GAME_TYPE,
        PITCHER_TEAM,
        BATTER_TEAM,
    ]

    df = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=cols,
        low_memory=False,
    )

    df[SEASON] = (
        pd.to_numeric(
            df[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    df[GAME_TYPE] = (
        df[GAME_TYPE]
        .astype("string")
        .fillna("<NA>")
        .astype(str)
    )

    if df[ID].duplicated().any():
        raise RuntimeError("train row_id is not unique")

    f_hist = df.loc[
        df[GAME_TYPE].eq("F")
        & df[SEASON].le(
            ANCHOR_DISCOVERY_MAX_SEASON
        )
    ].copy()

    if len(f_hist) == 0:
        raise RuntimeError(
            "No pre-2023 F rows for anchor discovery."
        )

    teams = sorted(
        set(
            f_hist[PITCHER_TEAM]
            .dropna()
            .tolist()
        )
        |
        set(
            f_hist[BATTER_TEAM]
            .dropna()
            .tolist()
        )
    )

    involvement = {}

    for team in teams:
        involvement[str(team)] = float(
            (
                f_hist[PITCHER_TEAM].eq(team)
                | f_hist[BATTER_TEAM].eq(team)
            )
            .mean()
        )

    anchors = [
        team
        for team in teams
        if float(
            (
                f_hist[PITCHER_TEAM].eq(team)
                | f_hist[BATTER_TEAM].eq(team)
            )
            .mean()
        )
        > 0.999999
    ]

    if len(anchors) != 1:
        top = sorted(
            involvement.items(),
            key=lambda x: -x[1],
        )[:10]

        raise RuntimeError(
            "Reference router expected exactly one persistent F anchor.\n"
            f"anchors={anchors}\n"
            f"top involvement={top}"
        )

    anchor = anchors[0]

    is_f = (
        df[GAME_TYPE]
        .eq("F")
        .to_numpy()
    )

    is_anchor = (
        (
            df[PITCHER_TEAM]
            .eq(anchor)
            | df[BATTER_TEAM]
            .eq(anchor)
        )
        .to_numpy()
    )

    df["domain3"] = np.where(
        is_f,
        "F",
        np.where(
            is_anchor,
            "R_ANCHOR",
            "R_CORE",
        ),
    )

    return (
        df[
            [
                ID,
                SEASON,
                GAME_TYPE,
                "domain3",
            ]
        ].copy(),
        anchor,
        involvement,
    )


# =============================================================================
# L2 audit
# =============================================================================

def load_v028():
    if not V028_OOF_PATH.exists():
        raise FileNotFoundError(V028_OOF_PATH)

    df = pd.read_csv(
        V028_OOF_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        ID,
        SEASON,
        "base_seed",
        "y_true",
        "B_L1_PC",
        "D_L1PC_L2",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"v028 OOF missing columns: {missing}"
        )

    df[SEASON] = (
        pd.to_numeric(
            df[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    df["base_seed"] = (
        pd.to_numeric(
            df["base_seed"],
            errors="raise",
        )
        .astype(int)
    )

    for c in [
        "y_true",
        "B_L1_PC",
        "D_L1PC_L2",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="raise",
        ).astype(np.float64)

    return df


def attach_router(oof, router):
    x = oof.merge(
        router[
            [
                ID,
                "domain3",
            ]
        ],
        on=ID,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if x["domain3"].isna().any():
        raise RuntimeError(
            "Some OOF rows did not match train router frame."
        )

    return x


def reconstruct_shrink1(df):
    """
    v028 D = clip(B + .5 * raw_corr, 0, 1).

    현재 .5 prediction에 boundary clipping이 없으면
    raw correction을 정확히 복원해서 shrink=1.0을 domain별로도 볼 수 있다.
    """

    d = df["D_L1PC_L2"].to_numpy(np.float64)

    boundary = (
        (d <= BOUNDARY_TOL)
        | (d >= 1.0 - BOUNDARY_TOL)
    )

    if int(boundary.sum()) > 0:
        return (
            None,
            int(boundary.sum()),
        )

    b = df["B_L1_PC"].to_numpy(np.float64)

    raw_corr = (
        d - b
    ) / CURRENT_L2_SHRINK

    alt = np.clip(
        b
        + ALT_L2_SHRINK
        * raw_corr,
        0.0,
        1.0,
    )

    return (
        alt,
        0,
    )


def build_domain_metrics(
    df,
    pred_col,
    baseline_col,
    model_label,
):
    rows = []

    group_cols = [
        "base_seed",
        SEASON,
        "domain3",
    ]

    for (
        seed,
        year,
        domain,
    ), g in df.groupby(
        group_cols,
        sort=True,
        observed=True,
    ):
        y = g["y_true"].to_numpy(np.float64)
        base = g[baseline_col].to_numpy(np.float64)
        pred = g[pred_col].to_numpy(np.float64)

        rows.append({
            "model": model_label,
            "base_seed": int(seed),
            "year": int(year),
            "domain3": str(domain),
            "n": int(len(g)),
            "baseline_bss": bss(y, base),
            "model_bss": bss(y, pred),
            "gain_vs_baseline_bss": float(
                bss(y, pred)
                - bss(y, base)
            ),
            "baseline_brier": brier(y, base),
            "model_brier": brier(y, pred),
        })

    return pd.DataFrame(rows)


def summarize_domain_metrics(metrics):
    rows = []

    for (
        model,
        domain,
    ), g in metrics.groupby(
        [
            "model",
            "domain3",
        ],
        sort=True,
    ):
        gains = g[
            "gain_vs_baseline_bss"
        ].to_numpy(np.float64)

        g24 = g.loc[
            g["year"].eq(2024),
            "gain_vs_baseline_bss",
        ].to_numpy(np.float64)

        grecent = g.loc[
            g["year"].isin(
                [
                    2023,
                    2024,
                ]
            ),
            "gain_vs_baseline_bss",
        ].to_numpy(np.float64)

        rows.append({
            "model": str(model),
            "domain3": str(domain),
            "mean_gain_bss": float(
                np.mean(gains)
            ),
            "std_gain_bss": float(
                np.std(
                    gains,
                    ddof=1,
                )
                if len(gains) > 1
                else 0.0
            ),
            "positive_seed_year_cases": int(
                np.sum(
                    gains > 0
                )
            ),
            "n_seed_year_cases": int(
                len(gains)
            ),
            "mean_recent_2023_2024_gain_bss": float(
                np.mean(grecent)
            )
            if len(grecent)
            else np.nan,
            "mean_2024_gain_bss": float(
                np.mean(g24)
            )
            if len(g24)
            else np.nan,
            "2024_seed_wins": int(
                np.sum(
                    g24 > 0
                )
            ),
            "2024_n_seeds": int(
                len(g24)
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Optional L3 routing audit
# =============================================================================

def load_v029_if_available():
    if not V029_OOF_PATH.exists():
        return None

    df = pd.read_csv(
        V029_OOF_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        ID,
        SEASON,
        "base_seed",
        "y_true",
        "D_L1PC_L2",
        "E_L1PC_L2_L3",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        print(
            "[WARN] v029 exists but expected columns are missing:",
            missing,
        )
        return None

    df[SEASON] = pd.to_numeric(
        df[SEASON],
        errors="raise",
    ).astype(int)

    df["base_seed"] = pd.to_numeric(
        df["base_seed"],
        errors="raise",
    ).astype(int)

    for c in [
        "y_true",
        "D_L1PC_L2",
        "E_L1PC_L2_L3",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="raise",
        ).astype(np.float64)

    return df


def audit_l3(l3, router):
    x = attach_router(
        l3,
        router,
    )

    x["L3_CORE_ONLY"] = (
        x["D_L1PC_L2"]
        .to_numpy(np.float64)
    )

    core = (
        x["domain3"]
        .eq("R_CORE")
        .to_numpy()
    )

    x.loc[
        core,
        "L3_CORE_ONLY",
    ] = (
        x.loc[
            core,
            "E_L1PC_L2_L3",
        ]
        .to_numpy(np.float64)
    )

    # Original all-R candidate.
    allr_metrics = build_domain_metrics(
        df=x,
        pred_col="E_L1PC_L2_L3",
        baseline_col="D_L1PC_L2",
        model_label="L3_ORIGINAL_ALL_R",
    )

    core_metrics = build_domain_metrics(
        df=x,
        pred_col="L3_CORE_ONLY",
        baseline_col="D_L1PC_L2",
        model_label="L3_R_CORE_ONLY",
    )

    domain_metrics = pd.concat(
        [
            allr_metrics,
            core_metrics,
        ],
        ignore_index=True,
    )

    # Overall routing comparison by seed-year.
    route_rows = []

    for (
        seed,
        year,
    ), g in x.groupby(
        [
            "base_seed",
            SEASON,
        ],
        sort=True,
    ):
        y = g["y_true"].to_numpy(np.float64)
        d = g["D_L1PC_L2"].to_numpy(np.float64)
        e = g["E_L1PC_L2_L3"].to_numpy(np.float64)
        c = g["L3_CORE_ONLY"].to_numpy(np.float64)

        route_rows.append({
            "base_seed": int(seed),
            "year": int(year),
            "n": int(len(g)),
            "original_all_r_gain_vs_l2_bss": float(
                bss(y, e)
                - bss(y, d)
            ),
            "r_core_only_gain_vs_l2_bss": float(
                bss(y, c)
                - bss(y, d)
            ),
            "core_only_minus_original_bss": float(
                bss(y, c)
                - bss(y, e)
            ),
        })

    route_df = pd.DataFrame(
        route_rows
    )

    return (
        domain_metrics,
        route_df,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"{OUT_DIR} already exists.\n"
            "기존 결과 보호를 위해 overwrite하지 않습니다."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        router,
        anchor,
        involvement,
    ) = load_router_frame()

    print(
        "=" * 126
    )
    print(EXPERIMENT_NAME)
    print(
        "=" * 126
    )
    print(
        f"ANCHOR_DISCOVERY_MAX_SEASON : {ANCHOR_DISCOVERY_MAX_SEASON}"
    )
    print(
        f"ANCHOR_TEAM_ID               : {anchor}"
    )
    print(
        "Retraining                   : NONE"
    )
    print(
        "=" * 126
    )

    counts = (
        router.groupby(
            [
                SEASON,
                "domain3",
            ],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    counts.to_csv(
        OUT_DIR
        / "domain_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # L2
    # -------------------------------------------------------------------------

    v028 = attach_router(
        load_v028(),
        router,
    )

    l2_metrics = build_domain_metrics(
        df=v028,
        pred_col="D_L1PC_L2",
        baseline_col="B_L1_PC",
        model_label="L2_SHRINK_0.50",
    )

    alt, boundary_n = reconstruct_shrink1(
        v028
    )

    all_l2_metrics = [
        l2_metrics
    ]

    if alt is not None:
        v028["L2_SHRINK_1"] = (
            alt
        )

        l2_alt_metrics = build_domain_metrics(
            df=v028,
            pred_col="L2_SHRINK_1",
            baseline_col="B_L1_PC",
            model_label="L2_SHRINK_1.00",
        )

        all_l2_metrics.append(
            l2_alt_metrics
        )

        l2_alt_metrics.to_csv(
            OUT_DIR
            / "l2_shrink1_domain_by_seed_year.csv",
            index=False,
            encoding="utf-8-sig",
        )

    l2_all = pd.concat(
        all_l2_metrics,
        ignore_index=True,
    )

    l2_summary = summarize_domain_metrics(
        l2_all
    )

    l2_all.to_csv(
        OUT_DIR
        / "l2_domain_by_seed_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    l2_summary.to_csv(
        OUT_DIR
        / "l2_domain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 126
    )
    print(
        "L2 EXACT DOMAIN SUMMARY"
    )
    print(
        "=" * 126
    )
    print(
        l2_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    # -------------------------------------------------------------------------
    # Optional L3
    # -------------------------------------------------------------------------

    l3 = load_v029_if_available()

    l3_available = (
        l3 is not None
    )

    if l3_available:
        (
            l3_domain,
            l3_route,
        ) = audit_l3(
            l3,
            router,
        )

        l3_domain.to_csv(
            OUT_DIR
            / "l3_domain_by_seed_year.csv",
            index=False,
            encoding="utf-8-sig",
        )

        l3_route.to_csv(
            OUT_DIR
            / "l3_routing_by_seed_year.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            "=" * 126
        )
        print(
            "L3 ORIGINAL ALL-R vs EXACT R_CORE-ONLY"
        )
        print(
            "=" * 126
        )

        summary_rows = []

        for label, col in [
            (
                "ORIGINAL_ALL_R",
                "original_all_r_gain_vs_l2_bss",
            ),
            (
                "R_CORE_ONLY",
                "r_core_only_gain_vs_l2_bss",
            ),
        ]:
            vals = l3_route[
                col
            ].to_numpy(np.float64)

            latest = l3_route.loc[
                l3_route["year"].eq(2024),
                col,
            ].to_numpy(np.float64)

            recent = l3_route.loc[
                l3_route["year"].isin(
                    [
                        2023,
                        2024,
                    ]
                ),
                col,
            ].to_numpy(np.float64)

            summary_rows.append({
                "routing": label,
                "mean_gain_vs_l2_bss": float(
                    np.mean(vals)
                ),
                "mean_recent_gain": float(
                    np.mean(recent)
                ),
                "mean_2024_gain": float(
                    np.mean(latest)
                ),
                "2024_seed_wins": int(
                    np.sum(
                        latest > 0
                    )
                ),
                "2024_n_seeds": int(
                    len(latest)
                ),
            })

        l3_summary = pd.DataFrame(
            summary_rows
        )

        l3_summary.to_csv(
            OUT_DIR
            / "l3_routing_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            l3_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:+.4f}",
            )
        )

    else:
        print()
        print(
            "[INFO] v029 OOF not found -> L3 routing audit skipped."
        )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),
        "retraining": False,
        "anchor_discovery_max_season": int(
            ANCHOR_DISCOVERY_MAX_SEASON
        ),
        "anchor_team_id": (
            int(anchor)
            if pd.notna(anchor)
            else str(anchor)
        ),
        "anchor_involvement": (
            involvement
        ),
        "router_definition": {
            "F": "game_type == F",
            "R_ANCHOR": (
                "game_type == R and "
                "(pitcher_team_id == anchor or batter_team_id == anchor)"
            ),
            "R_CORE": "remaining R",
        },
        "v028_oof": str(
            V028_OOF_PATH
        ),
        "l2_boundary_rows_at_shrink050": int(
            boundary_n
        ),
        "l2_shrink1_reconstruction_exact": bool(
            alt is not None
        ),
        "v029_oof": str(
            V029_OOF_PATH
        ),
        "v029_available": bool(
            l3_available
        ),
    }

    (
        OUT_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {OUT_DIR}"
    )
    print(
        "=" * 126
    )


if __name__ == "__main__":
    main()

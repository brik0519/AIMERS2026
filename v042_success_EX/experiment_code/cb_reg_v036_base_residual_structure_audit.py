#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v036_base_residual_structure_audit.py
=============================================

목적
----
현재 L1 champion:
    CatBoost + Exact-ASOF5 + R-only PC FULL

의 OOF residual 구조를 분석하여,
"모델을 하나 더 붙이는 것"이 아니라
"CatBoost가 반복해서 못 보는 정보"를 찾는다.

재학습
------
없음.
기존 v028 seed42 B_L1_PC OOF를 그대로 재사용한다.

핵심 질문
---------
1) 어떤 row-local subgroup에서 2021~2024 동안
   같은 방향의 residual bias가 반복되는가?

2) 그 bias가 과거 연도 -> 다음 연도로 strict temporal transfer 되는가?

3) 어떤 1-way / 2-way context가 다음 Base feature/view 후보인가?

Residual sign
-------------
residual = y - pred

positive residual:
    실제 y가 예측보다 큼
    -> CatBoost가 UNDER-predict

negative residual:
    실제 y가 예측보다 작음
    -> CatBoost가 OVER-predict

Strict temporal diagnostic
--------------------------
예:
target 2024 평가 시
2021~2023 OOF residual만 사용하여 subgroup별 mean residual을 추정한다.
2024 label/residual은 correction 생성에 절대 사용하지 않는다.

이것은 최종 deploy 보정 방식이 아니라,
"어떤 context의 residual bias가 다음 시즌에도 살아남는지"를 찾는 진단이다.

입력
----
data/train.csv

validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/oof_predictions.csv

출력
----
validation/cb_reg_v036_base_residual_structure_audit/
    one_way_value_summary.csv
    two_way_value_summary.csv
    one_way_feature_summary.csv
    two_way_feature_summary.csv
    temporal_transfer_by_target.csv
    temporal_transfer_summary.csv
    recommended_feature_views.csv
    metadata.json

실행
----
python cb_reg_v036_base_residual_structure_audit.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v036_base_residual_structure_audit"

ROOT = Path(".")

TRAIN_PATH = ROOT / "data" / "train.csv"

OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
    / "oof_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "validation"
    / EXPERIMENT_NAME
)

ID = "row_id"
SEASON = "season"
GAME_TYPE = "game_type"

CATBOOST_SEED = 42

YEARS = [
    2021,
    2022,
    2023,
    2024,
]

TRANSFER_TARGETS = [
    2022,
    2023,
    2024,
]

# 너무 작은 subgroup은 noise 가능성이 높으므로 제외.
MIN_VALUE_TOTAL_N_1WAY = 2000
MIN_VALUE_YEAR_N_1WAY = 400

MIN_VALUE_TOTAL_N_2WAY = 4000
MIN_VALUE_YEAR_N_2WAY = 800

# strict temporal residual mean shrinkage.
# subgroup residual mean을 그대로 쓰면 과적합 가능성이 커서
# 0으로 shrink한다.
RESIDUAL_SHRINK_STRENGTHS = [
    1000.0,
    3000.0,
    10000.0,
]

# transfer 시 history subgroup 최소 표본.
MIN_HISTORY_N = 1500

# 추천 후보 기준.
MIN_POSITIVE_TRANSFER_TARGETS = 2

PRED_BIN_EDGES = [
    -np.inf,
    0.35,
    0.45,
    0.55,
    0.65,
    np.inf,
]

PRED_BIN_LABELS = [
    "<.35",
    ".35-.45",
    ".45-.55",
    ".55-.65",
    ">.65",
]

N_BIN_EDGES = [
    -np.inf,
    25,
    75,
    150,
    300,
    600,
    np.inf,
]

N_BIN_LABELS = [
    "<=25",
    "26-75",
    "76-150",
    "151-300",
    "301-600",
    ">600",
]


# =============================================================================
# Metrics
# =============================================================================

def brier(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def bss(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    score = brier(
        y,
        p,
    )

    mu = float(
        np.mean(
            y
        )
    )

    ref = max(
        mu
        * (
            1.0
            - mu
        ),
        1e-12,
    )

    return float(
        100000.0
        * (
            1.0
            - score / ref
        )
    )


# =============================================================================
# Column resolution
# =============================================================================

def first_existing(
    columns,
    candidates,
):
    for c in candidates:
        if c in columns:
            return c

    return None


def require_col(
    columns,
    candidates,
    label,
):
    col = first_existing(
        columns,
        candidates,
    )

    if col is None:
        raise KeyError(
            f"{label} column not found. candidates={candidates}"
        )

    return col


# =============================================================================
# Load
# =============================================================================

def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            OOF_PATH
        )

    oof = pd.read_csv(
        OOF_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_oof = {
        ID,
        SEASON,
        GAME_TYPE,
        "base_seed",
        "y_true",
        "B_L1_PC",
    }

    missing = sorted(
        required_oof
        - set(
            oof.columns
        )
    )

    if missing:
        raise KeyError(
            f"OOF missing columns: {missing}"
        )

    oof = (
        oof.loc[
            pd.to_numeric(
                oof[
                    "base_seed"
                ],
                errors="raise",
            )
            .astype(int)
            .eq(
                CATBOOST_SEED
            )
        ]
        .copy()
    )

    oof[SEASON] = (
        pd.to_numeric(
            oof[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    oof = (
        oof.loc[
            oof[SEASON].isin(
                YEARS
            )
        ]
        .copy()
    )

    oof["y_true"] = (
        pd.to_numeric(
            oof["y_true"],
            errors="raise",
        )
        .astype(np.float64)
    )

    oof["pred"] = (
        pd.to_numeric(
            oof["B_L1_PC"],
            errors="raise",
        )
        .astype(np.float64)
    )

    oof["residual"] = (
        oof["y_true"]
        - oof["pred"]
    )

    oof["sq_error"] = (
        oof["residual"] ** 2
    )

    if oof[ID].duplicated().any():
        raise RuntimeError(
            "seed42 OOF row_id duplicate"
        )

    # train은 row-local context를 얻기 위해 사용.
    train_header = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        nrows=0,
    )

    train_cols = (
        train_header.columns
        .tolist()
    )

    # 필요한 후보 column만 읽어서 memory를 줄인다.
    candidate_cols = {
        ID,
        SEASON,
        GAME_TYPE,

        # hands
        "pitcher_hand",
        "pitcher_throws",
        "pitcher_hand_code",
        "batter_hand",
        "batter_stands",
        "batter_hand_code",

        # count
        "balls",
        "strikes",
        "count",
        "ball_count",
        "strike_count",

        # inning / outs / bases
        "inning",
        "inning_num",
        "inning_number",
        "outs_before",
        "outs",
        "base_state",

        # LI
        "li",
        "LI",
        "leverage_index",
        "leverage",

        # official sample-size/current-state signals
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_n",
        "asof_pitcher_pa",
        "exact_pitcher_season_n",

        # optional existing engineered row-local fields
        "pressure",
        "inning_phase",
        "runner_count",
        "runners",
        "hand_matchup",
    }

    usecols = [
        c
        for c in train_cols
        if c in candidate_cols
    ]

    if ID not in usecols:
        raise KeyError(
            f"{ID} not found in train.csv"
        )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=usecols,
        low_memory=False,
    )

    merged = oof.merge(
        train,
        on=ID,
        how="left",
        suffixes=(
            "_oof",
            "",
        ),
        validate="one_to_one",
        sort=False,
    )

    # season/game_type는 OOF를 canonical로 사용.
    if f"{SEASON}_oof" in merged.columns:
        merged[SEASON] = (
            pd.to_numeric(
                merged[
                    f"{SEASON}_oof"
                ],
                errors="raise",
            )
            .astype(int)
        )

    if f"{GAME_TYPE}_oof" in merged.columns:
        merged[GAME_TYPE] = (
            merged[
                f"{GAME_TYPE}_oof"
            ]
        )

    if merged[SEASON].isna().any():
        raise RuntimeError(
            "train merge failed for some OOF rows"
        )

    return (
        merged,
        train_cols,
    )


# =============================================================================
# Row-local analysis features
# =============================================================================

def normalize_str(s):
    return (
        s
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .replace(
            {
                "nan": "__MISSING__",
                "None": "__MISSING__",
                "<NA>": "__MISSING__",
            }
        )
    )


def derive_analysis_features(
    df,
):
    cols = set(
        df.columns
    )

    created = []

    # -------------------------------------------------------------------------
    # game type
    # -------------------------------------------------------------------------

    df["ana_game_type"] = normalize_str(
        df[
            GAME_TYPE
        ]
    )

    created.append(
        "ana_game_type"
    )

    # -------------------------------------------------------------------------
    # hands
    # -------------------------------------------------------------------------

    pitcher_hand_col = first_existing(
        cols,
        [
            "pitcher_hand",
            "pitcher_throws",
            "pitcher_hand_code",
        ],
    )

    batter_hand_col = first_existing(
        cols,
        [
            "batter_hand",
            "batter_stands",
            "batter_hand_code",
        ],
    )

    existing_matchup = first_existing(
        cols,
        [
            "hand_matchup",
        ],
    )

    if existing_matchup is not None:
        df["ana_hand_matchup"] = normalize_str(
            df[
                existing_matchup
            ]
        )

        created.append(
            "ana_hand_matchup"
        )

    elif (
        pitcher_hand_col is not None
        and batter_hand_col is not None
    ):
        df["ana_hand_matchup"] = (
            normalize_str(
                df[
                    pitcher_hand_col
                ]
            )
            + "_"
            + normalize_str(
                df[
                    batter_hand_col
                ]
            )
        )

        created.append(
            "ana_hand_matchup"
        )

    if batter_hand_col is not None:
        df["ana_batter_hand"] = normalize_str(
            df[
                batter_hand_col
            ]
        )

        created.append(
            "ana_batter_hand"
        )

    if pitcher_hand_col is not None:
        df["ana_pitcher_hand"] = normalize_str(
            df[
                pitcher_hand_col
            ]
        )

        created.append(
            "ana_pitcher_hand"
        )

    # -------------------------------------------------------------------------
    # count
    # -------------------------------------------------------------------------

    count_col = first_existing(
        cols,
        [
            "count",
        ],
    )

    balls_col = first_existing(
        cols,
        [
            "balls",
            "ball_count",
        ],
    )

    strikes_col = first_existing(
        cols,
        [
            "strikes",
            "strike_count",
        ],
    )

    if count_col is not None:
        df["ana_count"] = normalize_str(
            df[
                count_col
            ]
        )

        created.append(
            "ana_count"
        )

    elif (
        balls_col is not None
        and strikes_col is not None
    ):
        b = pd.to_numeric(
            df[
                balls_col
            ],
            errors="coerce",
        )

        s = pd.to_numeric(
            df[
                strikes_col
            ],
            errors="coerce",
        )

        df["ana_count"] = (
            b.fillna(-1)
            .astype(int)
            .astype(str)
            + "-"
            + s.fillna(-1)
            .astype(int)
            .astype(str)
        )

        created.append(
            "ana_count"
        )

    # pressure bucket based on count only.
    if (
        balls_col is not None
        and strikes_col is not None
    ):
        b = pd.to_numeric(
            df[
                balls_col
            ],
            errors="coerce",
        )

        s = pd.to_numeric(
            df[
                strikes_col
            ],
            errors="coerce",
        )

        pressure = np.full(
            len(
                df
            ),
            "neutral",
            dtype=object,
        )

        full = (
            b.eq(3)
            & s.eq(2)
        )

        three_ball = (
            b.eq(3)
            & ~s.eq(2)
        )

        two_strike = (
            s.eq(2)
            & ~b.eq(3)
        )

        pressure[
            three_ball.to_numpy()
        ] = "three_ball"

        pressure[
            two_strike.to_numpy()
        ] = "two_strike"

        pressure[
            full.to_numpy()
        ] = "full"

        missing = (
            b.isna()
            | s.isna()
        )

        pressure[
            missing.to_numpy()
        ] = "__MISSING__"

        df["ana_pressure"] = (
            pressure
        )

        created.append(
            "ana_pressure"
        )

    elif "pressure" in cols:
        df["ana_pressure"] = normalize_str(
            df[
                "pressure"
            ]
        )

        created.append(
            "ana_pressure"
        )

    # -------------------------------------------------------------------------
    # inning
    # -------------------------------------------------------------------------

    inning_col = first_existing(
        cols,
        [
            "inning",
            "inning_num",
            "inning_number",
        ],
    )

    if inning_col is not None:
        inning = pd.to_numeric(
            df[
                inning_col
            ],
            errors="coerce",
        )

        phase = np.full(
            len(
                df
            ),
            "__MISSING__",
            dtype=object,
        )

        phase[
            inning.le(3)
            .fillna(False)
            .to_numpy()
        ] = "early"

        phase[
            inning.between(
                4,
                6,
                inclusive="both",
            )
            .fillna(False)
            .to_numpy()
        ] = "middle"

        phase[
            inning.ge(7)
            .fillna(False)
            .to_numpy()
        ] = "late"

        df["ana_inning_phase"] = (
            phase
        )

        created.append(
            "ana_inning_phase"
        )

    elif "inning_phase" in cols:
        df["ana_inning_phase"] = normalize_str(
            df[
                "inning_phase"
            ]
        )

        created.append(
            "ana_inning_phase"
        )

    # -------------------------------------------------------------------------
    # base / runners
    # -------------------------------------------------------------------------

    if "base_state" in cols:
        df["ana_base_state"] = normalize_str(
            df[
                "base_state"
            ]
        )

        created.append(
            "ana_base_state"
        )

        def runner_count_from_state(x):
            if pd.isna(x):
                return "__MISSING__"

            s = str(x)

            if s in {
                "",
                "___",
                "0",
                "none",
                "None",
                "empty",
                "__MISSING__",
            }:
                return "0"

            # common encodings: 1__, _2_, __3, 12_, 1_3, _23, 123
            cnt = sum(
                ch in {
                    "1",
                    "2",
                    "3",
                }
                for ch in s
            )

            if cnt <= 0:
                return "0"

            if cnt >= 3:
                return "3"

            return str(
                cnt
            )

        df["ana_runner_count"] = (
            df[
                "base_state"
            ]
            .map(
                runner_count_from_state
            )
        )

        df["ana_runners"] = (
            df[
                "ana_runner_count"
            ]
            .map(
                lambda x:
                "__MISSING__"
                if x == "__MISSING__"
                else (
                    "empty"
                    if x == "0"
                    else "on"
                )
            )
        )

        created.extend(
            [
                "ana_runner_count",
                "ana_runners",
            ]
        )

    elif "runner_count" in cols:
        df["ana_runner_count"] = normalize_str(
            df[
                "runner_count"
            ]
        )

        created.append(
            "ana_runner_count"
        )

    # -------------------------------------------------------------------------
    # outs
    # -------------------------------------------------------------------------

    outs_col = first_existing(
        cols,
        [
            "outs_before",
            "outs",
        ],
    )

    if outs_col is not None:
        df["ana_outs"] = normalize_str(
            df[
                outs_col
            ]
        )

        created.append(
            "ana_outs"
        )

    # -------------------------------------------------------------------------
    # LI
    # -------------------------------------------------------------------------

    li_col = first_existing(
        cols,
        [
            "li",
            "LI",
            "leverage_index",
            "leverage",
        ],
    )

    if li_col is not None:
        li = pd.to_numeric(
            df[
                li_col
            ],
            errors="coerce",
        )

        # coarse bins, row-local only.
        df["ana_li_bucket"] = pd.cut(
            li,
            bins=[
                -np.inf,
                0.5,
                1.0,
                1.5,
                2.5,
                np.inf,
            ],
            labels=[
                "very_low",
                "low",
                "medium",
                "high",
                "very_high",
            ],
        ).astype(
            "object"
        )

        df["ana_li_bucket"] = (
            df[
                "ana_li_bucket"
            ]
            .where(
                pd.notna(
                    df[
                        "ana_li_bucket"
                    ]
                ),
                "__MISSING__",
            )
            .astype(str)
        )

        created.append(
            "ana_li_bucket"
        )

    # -------------------------------------------------------------------------
    # pitcher current sample size
    # -------------------------------------------------------------------------

    n_col = first_existing(
        cols,
        [
            "asof_pitcher_pitchmix_n",
            "asof_pitcher_n",
            "asof_pitcher_pa",
            "exact_pitcher_season_n",
        ],
    )

    if n_col is not None:
        n = pd.to_numeric(
            df[
                n_col
            ],
            errors="coerce",
        )

        df["ana_pitcher_n_bin"] = pd.cut(
            n,
            bins=N_BIN_EDGES,
            labels=N_BIN_LABELS,
        ).astype(
            "object"
        )

        df["ana_pitcher_n_bin"] = (
            df[
                "ana_pitcher_n_bin"
            ]
            .where(
                pd.notna(
                    df[
                        "ana_pitcher_n_bin"
                    ]
                ),
                "__MISSING__",
            )
            .astype(str)
        )

        created.append(
            "ana_pitcher_n_bin"
        )

    # -------------------------------------------------------------------------
    # current L1 prediction confidence
    # -------------------------------------------------------------------------

    df["ana_l1_pred_bin"] = pd.cut(
        df[
            "pred"
        ],
        bins=PRED_BIN_EDGES,
        labels=PRED_BIN_LABELS,
    ).astype(str)

    created.append(
        "ana_l1_pred_bin"
    )

    # unique preserve order
    created = list(
        dict.fromkeys(
            created
        )
    )

    return (
        df,
        created,
    )


# =============================================================================
# Static residual summaries
# =============================================================================

def summarize_one_feature_values(
    df,
    feature,
    min_total_n,
    min_year_n,
):
    rows = []

    grouped = (
        df.groupby(
            feature,
            dropna=False,
            observed=True,
        )
    )

    for value, g in grouped:
        n_total = int(
            len(
                g
            )
        )

        if n_total < min_total_n:
            continue

        year_means = {}
        year_ns = {}

        valid_year_signs = []

        for year in YEARS:
            gy = (
                g.loc[
                    g[
                        SEASON
                    ]
                    .eq(
                        year
                    )
                ]
            )

            n_y = int(
                len(
                    gy
                )
            )

            year_ns[
                year
            ] = (
                n_y
            )

            if n_y >= min_year_n:
                mean_res = float(
                    gy[
                        "residual"
                    ]
                    .mean()
                )

                year_means[
                    year
                ] = (
                    mean_res
                )

                if mean_res > 0:
                    valid_year_signs.append(
                        1
                    )
                elif mean_res < 0:
                    valid_year_signs.append(
                        -1
                    )

            else:
                year_means[
                    year
                ] = np.nan

        n_valid_years = int(
            len(
                valid_year_signs
            )
        )

        all_same_sign = (
            n_valid_years >= 2
            and abs(
                sum(
                    valid_year_signs
                )
            )
            == n_valid_years
        )

        positive_years = int(
            sum(
                x > 0
                for x in valid_year_signs
            )
        )

        negative_years = int(
            sum(
                x < 0
                for x in valid_year_signs
            )
        )

        recent_vals = [
            year_means[
                year
            ]
            for year in [
                2023,
                2024,
            ]
            if np.isfinite(
                year_means[
                    year
                ]
            )
        ]

        recent_same_sign = (
            len(
                recent_vals
            )
            == 2
            and np.sign(
                recent_vals[0]
            )
            == np.sign(
                recent_vals[1]
            )
            and np.sign(
                recent_vals[0]
            )
            != 0
        )

        row = {
            "feature": feature,
            "value": str(
                value
            ),
            "n_total": (
                n_total
            ),
            "n_valid_years": (
                n_valid_years
            ),
            "positive_years": (
                positive_years
            ),
            "negative_years": (
                negative_years
            ),
            "all_same_sign": bool(
                all_same_sign
            ),
            "recent_same_sign": bool(
                recent_same_sign
            ),
            "mean_residual": float(
                g[
                    "residual"
                ]
                .mean()
            ),
            "mean_abs_residual": float(
                np.abs(
                    g[
                        "residual"
                    ]
                )
                .mean()
            ),
            "mean_sq_error": float(
                g[
                    "sq_error"
                ]
                .mean()
            ),
        }

        for year in YEARS:
            row[
                f"n_{year}"
            ] = int(
                year_ns[
                    year
                ]
            )

            row[
                f"residual_mean_{year}"
            ] = float(
                year_means[
                    year
                ]
            ) if np.isfinite(
                year_means[
                    year
                ]
            ) else np.nan

        finite_year_means = [
            v
            for v in year_means.values()
            if np.isfinite(
                v
            )
        ]

        if finite_year_means:
            row[
                "mean_abs_year_residual_bias"
            ] = float(
                np.mean(
                    np.abs(
                        finite_year_means
                    )
                )
            )

            row[
                "min_abs_year_residual_bias"
            ] = float(
                np.min(
                    np.abs(
                        finite_year_means
                    )
                )
            )
        else:
            row[
                "mean_abs_year_residual_bias"
            ] = np.nan

            row[
                "min_abs_year_residual_bias"
            ] = np.nan

        rows.append(
            row
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    )


def build_one_way_summary(
    df,
    features,
):
    frames = []

    for feature in features:
        print(
            f"[1-way] {feature}"
        )

        s = summarize_one_feature_values(
            df=df,
            feature=feature,
            min_total_n=MIN_VALUE_TOTAL_N_1WAY,
            min_year_n=MIN_VALUE_YEAR_N_1WAY,
        )

        if len(
            s
        ):
            frames.append(
                s
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_two_way_summary(
    df,
    features,
):
    frames = []

    # 2-way는 지나친 조합 폭증을 막기 위해
    # 의미 있는 low-cardinality context만 사용.
    blocked = {
        "ana_l1_pred_bin",
    }

    pair_features = [
        f
        for f in features
        if f not in blocked
    ]

    for f1, f2 in itertools.combinations(
        pair_features,
        2,
    ):
        print(
            f"[2-way] {f1} × {f2}"
        )

        temp_name = (
            "__PAIR_VALUE__"
        )

        temp = (
            df[
                [
                    SEASON,
                    "residual",
                    "sq_error",
                    f1,
                    f2,
                ]
            ]
            .copy()
        )

        temp[
            temp_name
        ] = (
            normalize_str(
                temp[
                    f1
                ]
            )
            + " || "
            + normalize_str(
                temp[
                    f2
                ]
            )
        )

        s = summarize_one_feature_values(
            df=temp,
            feature=temp_name,
            min_total_n=MIN_VALUE_TOTAL_N_2WAY,
            min_year_n=MIN_VALUE_YEAR_N_2WAY,
        )

        if len(
            s
        ):
            s[
                "feature_1"
            ] = (
                f1
            )

            s[
                "feature_2"
            ] = (
                f2
            )

            s[
                "feature"
            ] = (
                f1
                + " × "
                + f2
            )

            frames.append(
                s
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def aggregate_feature_summary(
    value_summary,
    is_pair,
):
    if value_summary.empty:
        return pd.DataFrame()

    key = (
        [
            "feature",
            "feature_1",
            "feature_2",
        ]
        if is_pair
        else [
            "feature",
        ]
    )

    available_key = [
        c
        for c in key
        if c in value_summary.columns
    ]

    rows = []

    for group_key, g in value_summary.groupby(
        available_key,
        dropna=False,
        observed=True,
    ):
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (
                group_key,
            )

        row = {
            col: val
            for col, val in zip(
                available_key,
                group_key,
            )
        }

        row.update({
            "n_values": int(
                len(
                    g
                )
            ),
            "n_all_same_sign_values": int(
                g[
                    "all_same_sign"
                ]
                .sum()
            ),
            "n_recent_same_sign_values": int(
                g[
                    "recent_same_sign"
                ]
                .sum()
            ),
            "weighted_abs_bias": float(
                np.average(
                    g[
                        "mean_abs_year_residual_bias"
                    ]
                    .fillna(0.0),
                    weights=g[
                        "n_total"
                    ],
                )
            ),
            "max_abs_bias": float(
                g[
                    "mean_abs_year_residual_bias"
                ]
                .max()
            ),
        })

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Strict temporal transfer
# =============================================================================

def apply_group_residual_correction(
    history,
    target,
    feature_cols,
    strength,
):
    group_cols = list(
        feature_cols
    )

    hist_stats = (
        history.groupby(
            group_cols,
            dropna=False,
            observed=True,
        )
        .agg(
            hist_n=(
                "residual",
                "size",
            ),
            hist_residual_mean=(
                "residual",
                "mean",
            ),
        )
        .reset_index()
    )

    hist_stats = (
        hist_stats.loc[
            hist_stats[
                "hist_n"
            ]
            >= MIN_HISTORY_N
        ]
        .copy()
    )

    if hist_stats.empty:
        corrected = (
            target[
                "pred"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        return (
            corrected,
            0.0,
            0,
        )

    hist_stats[
        "shrunk_residual"
    ] = (
        (
            hist_stats[
                "hist_n"
            ]
            / (
                hist_stats[
                    "hist_n"
                ]
                + float(
                    strength
                )
            )
        )
        * hist_stats[
            "hist_residual_mean"
        ]
    )

    apply = (
        target[
            group_cols
            + [
                "pred",
            ]
        ]
        .copy()
        .merge(
            hist_stats[
                group_cols
                + [
                    "hist_n",
                    "shrunk_residual",
                ]
            ],
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    correction = (
        apply[
            "shrunk_residual"
        ]
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    pred = (
        apply[
            "pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
        + correction
    )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    coverage = float(
        np.mean(
            np.abs(
                correction
            )
            > 0
        )
    )

    n_groups = int(
        len(
            hist_stats
        )
    )

    return (
        pred,
        coverage,
        n_groups,
    )


def run_temporal_transfer(
    df,
    features,
):
    rows = []

    # 1-way
    feature_sets = [
        {
            "kind": "1WAY",
            "name": f,
            "cols": [
                f,
            ],
        }
        for f in features
    ]

    # 2-way
    blocked = {
        "ana_l1_pred_bin",
    }

    pair_features = [
        f
        for f in features
        if f not in blocked
    ]

    for f1, f2 in itertools.combinations(
        pair_features,
        2,
    ):
        feature_sets.append({
            "kind": "2WAY",
            "name": (
                f1
                + " × "
                + f2
            ),
            "cols": [
                f1,
                f2,
            ],
        })

    for spec in feature_sets:
        print(
            f"[transfer] {spec['name']}"
        )

        for strength in RESIDUAL_SHRINK_STRENGTHS:
            for target_year in TRANSFER_TARGETS:
                history = (
                    df.loc[
                        df[
                            SEASON
                        ]
                        < target_year
                    ]
                    .copy()
                )

                target = (
                    df.loc[
                        df[
                            SEASON
                        ]
                        .eq(
                            target_year
                        )
                    ]
                    .copy()
                )

                if (
                    history.empty
                    or target.empty
                ):
                    continue

                corrected, coverage, n_groups = (
                    apply_group_residual_correction(
                        history=history,
                        target=target,
                        feature_cols=spec[
                            "cols"
                        ],
                        strength=strength,
                    )
                )

                y = (
                    target[
                        "y_true"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                base_pred = (
                    target[
                        "pred"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                gain = float(
                    bss(
                        y,
                        corrected,
                    )
                    - bss(
                        y,
                        base_pred,
                    )
                )

                rows.append({
                    "kind": (
                        spec[
                            "kind"
                        ]
                    ),
                    "feature_view": (
                        spec[
                            "name"
                        ]
                    ),
                    "strength": float(
                        strength
                    ),
                    "target_year": int(
                        target_year
                    ),
                    "gain_vs_l1_bss": (
                        gain
                    ),
                    "coverage": (
                        coverage
                    ),
                    "n_history_groups": (
                        n_groups
                    ),
                })

    return pd.DataFrame(
        rows
    )


def summarize_transfer(
    transfer,
):
    rows = []

    for (
        kind,
        feature_view,
        strength,
    ), g in transfer.groupby(
        [
            "kind",
            "feature_view",
            "strength",
        ],
        sort=True,
    ):
        gains = {
            int(
                row[
                    "target_year"
                ]
            ): float(
                row[
                    "gain_vs_l1_bss"
                ]
            )
            for _, row in g.iterrows()
        }

        vals = np.asarray(
            list(
                gains.values()
            ),
            dtype=np.float64,
        )

        recent_vals = [
            gains[
                y
            ]
            for y in [
                2023,
                2024,
            ]
            if y in gains
        ]

        rows.append({
            "kind": (
                kind
            ),
            "feature_view": (
                feature_view
            ),
            "strength": float(
                strength
            ),

            "mean_gain_bss": float(
                np.mean(
                    vals
                )
            ),

            "mean_recent_2023_2024_gain_bss": float(
                np.mean(
                    recent_vals
                )
            ) if recent_vals else np.nan,

            "gain_2022_bss": float(
                gains.get(
                    2022,
                    np.nan,
                )
            ),

            "gain_2023_bss": float(
                gains.get(
                    2023,
                    np.nan,
                )
            ),

            "gain_2024_bss": float(
                gains.get(
                    2024,
                    np.nan,
                )
            ),

            "positive_target_years": int(
                np.sum(
                    vals > 0
                )
            ),

            "n_target_years": int(
                len(
                    vals
                )
            ),

            "mean_coverage": float(
                g[
                    "coverage"
                ]
                .mean()
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_recommendations(
    transfer_summary,
):
    if transfer_summary.empty:
        return pd.DataFrame()

    cand = (
        transfer_summary.loc[
            (
                transfer_summary[
                    "positive_target_years"
                ]
                >= MIN_POSITIVE_TRANSFER_TARGETS
            )
            & (
                transfer_summary[
                    "gain_2024_bss"
                ]
                > 0
            )
            & (
                transfer_summary[
                    "mean_recent_2023_2024_gain_bss"
                ]
                > 0
            )
            & (
                transfer_summary[
                    "mean_gain_bss"
                ]
                > 0
            )
        ]
        .copy()
    )

    if cand.empty:
        return cand

    cand[
        "strong_candidate"
    ] = (
        cand[
            "positive_target_years"
        ]
        .eq(
            cand[
                "n_target_years"
            ]
        )
        & (
            cand[
                "gain_2024_bss"
            ]
            > 0
        )
        & (
            cand[
                "mean_recent_2023_2024_gain_bss"
            ]
            > 0
        )
    )

    cand = (
        cand.sort_values(
            [
                "strong_candidate",
                "gain_2024_bss",
                "mean_recent_2023_2024_gain_bss",
                "mean_gain_bss",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return cand


# =============================================================================
# Print helpers
# =============================================================================

def print_static_top(
    summary,
    title,
    top_n=20,
):
    print()
    print(
        "=" * 132
    )

    print(
        title
    )

    print(
        "=" * 132
    )

    if summary.empty:
        print(
            "None"
        )
        return

    s = (
        summary.loc[
            (
                summary[
                    "all_same_sign"
                ]
            )
            & (
                summary[
                    "n_valid_years"
                ]
                >= 3
            )
        ]
        .copy()
    )

    if s.empty:
        print(
            "None"
        )
        return

    s = (
        s.sort_values(
            [
                "n_valid_years",
                "mean_abs_year_residual_bias",
                "n_total",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(
            top_n
        )
    )

    show_cols = [
        "feature",
        "value",
        "n_total",
        "n_valid_years",
        "positive_years",
        "negative_years",
        "mean_abs_year_residual_bias",
        "residual_mean_2021",
        "residual_mean_2022",
        "residual_mean_2023",
        "residual_mean_2024",
    ]

    show_cols = [
        c
        for c in show_cols
        if c in s.columns
    ]

    print(
        s[
            show_cols
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}",
        )
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
        df,
        train_cols,
    ) = load_data()

    (
        df,
        features,
    ) = derive_analysis_features(
        df
    )

    print(
        "=" * 132
    )

    print(
        EXPERIMENT_NAME
    )

    print(
        "=" * 132
    )

    print(
        f"OOF     : {OOF_PATH}"
    )

    print(
        f"Seed    : {CATBOOST_SEED}"
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Years   : {sorted(df[SEASON].unique().tolist())}"
    )

    print(
        "Retrain : NONE"
    )

    print(
        "Analysis features:"
    )

    for f in features:
        print(
            f"  - {f}"
        )

    print(
        "=" * 132
    )

    # -------------------------------------------------------------------------
    # Static residual sign consistency
    # -------------------------------------------------------------------------

    one_way_values = build_one_way_summary(
        df=df,
        features=features,
    )

    two_way_values = build_two_way_summary(
        df=df,
        features=features,
    )

    one_way_features = aggregate_feature_summary(
        value_summary=one_way_values,
        is_pair=False,
    )

    two_way_features = aggregate_feature_summary(
        value_summary=two_way_values,
        is_pair=True,
    )

    # -------------------------------------------------------------------------
    # Strict temporal transfer
    # -------------------------------------------------------------------------

    transfer = run_temporal_transfer(
        df=df,
        features=features,
    )

    transfer_summary = summarize_transfer(
        transfer
    )

    recommendations = build_recommendations(
        transfer_summary
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    one_way_values.to_csv(
        OUT_DIR
        / "one_way_value_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    two_way_values.to_csv(
        OUT_DIR
        / "two_way_value_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    one_way_features.to_csv(
        OUT_DIR
        / "one_way_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    two_way_features.to_csv(
        OUT_DIR
        / "two_way_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transfer.to_csv(
        OUT_DIR
        / "temporal_transfer_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transfer_summary.to_csv(
        OUT_DIR
        / "temporal_transfer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recommendations.to_csv(
        OUT_DIR
        / "recommended_feature_views.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    print_static_top(
        summary=one_way_values,
        title=(
            "TOP SYSTEMATIC 1-WAY RESIDUAL PATTERNS "
            "(same sign across >=3 years)"
        ),
        top_n=20,
    )

    print_static_top(
        summary=two_way_values,
        title=(
            "TOP SYSTEMATIC 2-WAY RESIDUAL PATTERNS "
            "(same sign across >=3 years)"
        ),
        top_n=20,
    )

    print()
    print(
        "=" * 132
    )

    print(
        "TOP STRICT TEMPORAL TRANSFER CANDIDATES"
    )

    print(
        "=" * 132
    )

    if recommendations.empty:
        print(
            "None"
        )

    else:
        show = (
            recommendations.head(
                30
            )
        )

        print(
            show[
                [
                    "kind",
                    "feature_view",
                    "strength",
                    "mean_gain_bss",
                    "mean_recent_2023_2024_gain_bss",
                    "gain_2022_bss",
                    "gain_2023_bss",
                    "gain_2024_bss",
                    "positive_target_years",
                    "n_target_years",
                    "mean_coverage",
                    "strong_candidate",
                ]
            ]
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
        "TOP 1-WAY FEATURE VIEWS BY 2024 TRANSFER"
    )

    print(
        "=" * 132
    )

    top_1 = (
        transfer_summary.loc[
            transfer_summary[
                "kind"
            ]
            .eq(
                "1WAY"
            )
        ]
        .sort_values(
            [
                "gain_2024_bss",
                "mean_recent_2023_2024_gain_bss",
                "mean_gain_bss",
            ],
            ascending=False,
        )
        .head(
            15
        )
    )

    print(
        top_1.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
        if len(
            top_1
        )
        else "None"
    )

    print()
    print(
        "=" * 132
    )

    print(
        "TOP 2-WAY FEATURE VIEWS BY 2024 TRANSFER"
    )

    print(
        "=" * 132
    )

    top_2 = (
        transfer_summary.loc[
            transfer_summary[
                "kind"
            ]
            .eq(
                "2WAY"
            )
        ]
        .sort_values(
            [
                "gain_2024_bss",
                "mean_recent_2023_2024_gain_bss",
                "mean_gain_bss",
            ],
            ascending=False,
        )
        .head(
            20
        )
    )

    print(
        top_2.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
        if len(
            top_2
        )
        else "None"
    )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),

        "input_oof": str(
            OOF_PATH
        ),

        "catboost_seed": (
            CATBOOST_SEED
        ),

        "retraining": (
            False
        ),

        "years": (
            YEARS
        ),

        "transfer_targets": (
            TRANSFER_TARGETS
        ),

        "analysis_features": (
            features
        ),

        "min_value_total_n_1way": (
            MIN_VALUE_TOTAL_N_1WAY
        ),

        "min_value_year_n_1way": (
            MIN_VALUE_YEAR_N_1WAY
        ),

        "min_value_total_n_2way": (
            MIN_VALUE_TOTAL_N_2WAY
        ),

        "min_value_year_n_2way": (
            MIN_VALUE_YEAR_N_2WAY
        ),

        "residual_shrink_strengths": (
            RESIDUAL_SHRINK_STRENGTHS
        ),

        "min_history_n": (
            MIN_HISTORY_N
        ),

        "notes": [
            (
                "Residual = y - current L1 prediction. "
                "Positive means L1 under-predicts."
            ),
            (
                "Static same-sign patterns are descriptive only."
            ),
            (
                "Strict temporal transfer uses only earlier OOF years "
                "to estimate subgroup residual bias for the target year."
            ),
            (
                "Temporal residual correction is a diagnostic for "
                "feature/view discovery, not the proposed final inference rule."
            ),
            (
                "All subgroup definitions are row-local/deployable contexts; "
                "no other validation/test row is used to construct a row's "
                "analysis feature."
            ),
        ],
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
        "=" * 132
    )


if __name__ == "__main__":
    main()

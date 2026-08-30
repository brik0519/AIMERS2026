# train.py
# =============================================================================
# fanchor_blend_v001
#
# Confirmatory experiment
#
# 이전 fanchor_dyn_v001 탐색 결과를 바탕으로 설정을 사전 고정한다.
#
# Dynamic Anchor:
#
#   A_t = (1 - alpha_t) * A_prev + alpha_t * R_t
#
#   alpha_t = lambda * n_t / (n_t + kappa)
#
# Frozen configuration:
#
#   lambda = 1.0
#   kappa  = 100
#
# F prediction:
#
#   p_F =
#       (1 - BLEND_WEIGHT) * p_control
#       + BLEND_WEIGHT * p_dynamic
#
#   BLEND_WEIGHT = 0.25
#
# Validation:
#
#   2019~2020 -> 2021
#   2019~2021 -> 2022
#   2019~2022 -> 2023
#   2019~2023 -> 2024
#
# Seeds:
#
#   42 / 2026 / 314
#
# Primary confirmation gate:
#
#   2024 F gain > 0 for all 3 seeds
#
# Rule compliance:
#
#   - test.csv는 header만 읽는다.
#   - test의 다른 행을 Feature/통계 생성에 사용하지 않는다.
#   - Dynamic state는 해당 row의 official ASOF만 사용한다.
#   - historical Anchor는 fold-train official label만 사용한다.
# =============================================================================

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =============================================================================
# 경로
# =============================================================================

EXPERIMENT_NAME = "fanchor_blend_v001"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =============================================================================
# 기본 설정
# =============================================================================

ID = "row_id"
TARGET = "control_success"

SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

VALIDATION_YEARS = [
    2021,
    2022,
    2023,
    2024,
]

MODEL_SEEDS = [
    42,
    2026,
    314,
]


# =============================================================================
# Confirmatory configuration
# =============================================================================

ANCHOR_STRENGTH = 100.0
ANCHOR_FALLBACK_RATE = 0.5

DYNAMIC_LAMBDA = 1.0
DYNAMIC_KAPPA = 100.0

F_BLEND_WEIGHT = 0.25


# =============================================================================
# CatBoost
# =============================================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
]

BASE_MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# =============================================================================
# Exact-ASOF5
# =============================================================================

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


# =============================================================================
# HB-EB
# =============================================================================

HBEB_CONFIGS = [
    {
        "group_cols": ["pitcher_hand"],
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "new_col": "hbeb_pitcher_success_rate",
    },
    {
        "group_cols": ["batter_hand"],
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "new_col": "hbeb_batter_success_rate",
    },
]

EXPECTED_CONTROL_FEATURES = 52


# =============================================================================
# Utility
# =============================================================================

def prepare_features(df, cat_cols):
    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


def brier_score(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def calculate_bss(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    brier = brier_score(
        y_true,
        y_pred,
    )

    p = float(
        np.mean(y_true)
    )

    baseline = (
        p * (1.0 - p)
    )

    if baseline <= 0.0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier / baseline
        )
    )


def error_decomposition(y_true, y_pred):
    """
    Brier decomposition:

        total = level + shape

    level:
        mean(pred - y)^2

    shape:
        mean((pred-y)^2) - level
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    error = (
        y_pred - y_true
    )

    bias = float(
        np.mean(error)
    )

    total = float(
        np.mean(
            error ** 2
        )
    )

    level = float(
        bias ** 2
    )

    shape = float(
        max(
            total - level,
            0.0,
        )
    )

    p = float(
        np.mean(y_true)
    )

    baseline = float(
        p * (1.0 - p)
    )

    return {
        "brier": total,
        "bias": bias,
        "level_brier": level,
        "shape_brier": shape,
        "baseline_brier": baseline,
    }


def compare_to_control(
    y_true,
    control_pred,
    candidate_pred,
):
    control = error_decomposition(
        y_true,
        control_pred,
    )

    candidate = error_decomposition(
        y_true,
        candidate_pred,
    )

    baseline = control[
        "baseline_brier"
    ]

    if baseline <= 0.0:
        total_gain = 0.0
        level_gain = 0.0
        shape_gain = 0.0

    else:
        total_gain = (
            100000.0
            * (
                control["brier"]
                - candidate["brier"]
            )
            / baseline
        )

        level_gain = (
            100000.0
            * (
                control["level_brier"]
                - candidate["level_brier"]
            )
            / baseline
        )

        shape_gain = (
            100000.0
            * (
                control["shape_brier"]
                - candidate["shape_brier"]
            )
            / baseline
        )

    return {
        "control_brier": (
            control["brier"]
        ),

        "candidate_brier": (
            candidate["brier"]
        ),

        "control_bias": (
            control["bias"]
        ),

        "candidate_bias": (
            candidate["bias"]
        ),

        "total_gain_pt": float(
            total_gain
        ),

        "level_gain_pt": float(
            level_gain
        ),

        "shape_gain_pt": float(
            shape_gain
        ),
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            to_jsonable(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)

        return (
            None
            if not np.isfinite(value)
            else value
        )

    if isinstance(obj, float):
        return (
            None
            if not np.isfinite(obj)
            else obj
        )

    if isinstance(obj, Path):
        return str(obj)

    return obj


# =============================================================================
# Pitcher season stats
# =============================================================================

def build_pitcher_season_stats(df_train):
    required = {
        "pitcher_id",
        SEASON_COL,
        TARGET,
    }

    missing = sorted(
        required
        - set(df_train.columns)
    )

    if missing:
        raise KeyError(
            f"pitcher season stats 필수 컬럼 없음: "
            f"{missing}"
        )

    stats = (
        df_train
        .groupby(
            [
                "pitcher_id",
                SEASON_COL,
            ],
            dropna=False,
            sort=False,
        )[TARGET]
        .agg(
            n="size",
            success="sum",
        )
        .reset_index()
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats["n"] = (
        stats["n"]
        .astype(np.float64)
    )

    stats["success"] = (
        stats["success"]
        .astype(np.float64)
    )

    return stats


# =============================================================================
# Exact-ASOF5
# =============================================================================

def add_pitcher_exact_asof_features(
    df,
    pitcher_season_stats,
):
    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 컬럼 없음: "
            f"{missing}"
        )

    out = df.copy()

    for col in EXACT_FEATURES:
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = pitcher_season_stats.copy()

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        row_mask = (
            seasons
            == int(target_year)
        )

        rows = out.loc[
            row_mask
        ]

        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist) > 0:
            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[
                    [
                        "n",
                        "success",
                    ]
                ]
                .sum()
            )

            hist_n = (
                rows["pitcher_id"]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                rows["pitcher_id"]
                .map(
                    hist_agg[
                        "success"
                    ]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

        else:
            hist_n = np.zeros(
                len(rows),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(rows),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_n"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_rate = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_success = (
            asof_n
            * asof_rate
        )

        raw_current_n = (
            asof_n
            - hist_n
        )

        current_n = np.maximum(
            raw_current_n,
            0.0,
        )

        raw_current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                raw_current_success
            ),
            np.clip(
                raw_current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_current = (
            (current_n > 0.0)
            & np.isfinite(
                current_success
            )
        )

        current_rate[
            valid_current
        ] = (
            current_success[
                valid_current
            ]
            / current_n[
                valid_current
            ]
        )

        past_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_past = (
            hist_n > 0.0
        )

        past_rate[
            valid_past
        ] = (
            hist_success[
                valid_past
            ]
            / hist_n[
                valid_past
            ]
        )

        idx = rows.index

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = current_n

        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = np.log1p(
            current_n
        )

        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = current_rate

        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = past_rate

        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = (
            current_rate
            - past_rate
        )

    return out


# =============================================================================
# HB-EB
# =============================================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    g_mu = float(
        df_train[
            rate_col
        ].mean()
    )

    g_var = float(
        df_train[
            rate_col
        ].var()
    )

    max_bernoulli_var = (
        g_mu * (1.0 - g_mu)
    )

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = (
            max_bernoulli_var
            * 0.5
        )

    g_m = (
        max_bernoulli_var
        / g_var
    ) - 1.0

    g_alpha = (
        g_mu * g_m
    )

    g_beta = (
        (1.0 - g_mu)
        * g_m
    )

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            [
                "mean",
                "var",
                "count",
            ]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0
            - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(
            stats["var"]
        )
        & (
            stats["var"] > 0.0
        )
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"] >= 30
        )
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

    if valid_mask.any():
        mu = stats.loc[
            valid_mask,
            "mean",
        ]

        var = stats.loc[
            valid_mask,
            "var",
        ]

        m = (
            mu
            * (1.0 - mu)
            / var
        ) - 1.0

        stats.loc[
            valid_mask,
            "alpha",
        ] = (
            mu * m
        )

        stats.loc[
            valid_mask,
            "beta",
        ] = (
            (1.0 - mu)
            * m
        )

    prior_df = stats[
        group_cols
        + [
            "alpha",
            "beta",
        ]
    ].copy()

    return (
        prior_df,
        (
            float(g_alpha),
            float(g_beta),
        ),
    )


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    lookup = (
        df[group_cols]
        .reset_index(
            drop=True
        )
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n * rate
    )

    denominator = (
        n
        + alpha
        + beta
    )

    return np.divide(
        success_count
        + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=(
            denominator > 0.0
        ),
    )


def apply_hbeb_pair(
    df_train,
    df_apply,
    base_features,
):
    transformed = (
        df_apply.copy()
    )

    features = list(
        base_features
    )

    for config in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config[
                "group_cols"
            ],
            rate_col=config[
                "rate_col"
            ],
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=config[
                "group_cols"
            ],
            n_col=config[
                "n_col"
            ],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            config["rate_col"]
            in features
        ):
            features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in features
        ):
            features.append(
                config["new_col"]
            )

    return (
        transformed,
        features,
    )


# =============================================================================
# Control view
# =============================================================================

def make_control_views(
    fold_train,
    fold_val,
    base_features,
):
    pitcher_stats = (
        build_pitcher_season_stats(
            fold_train
        )
    )

    train_exact = (
        add_pitcher_exact_asof_features(
            fold_train,
            pitcher_stats,
        )
    )

    val_exact = (
        add_pitcher_exact_asof_features(
            fold_val,
            pitcher_stats,
        )
    )

    exact_features = list(
        base_features
    )

    for col in EXACT_FEATURES:
        if col not in exact_features:
            exact_features.append(
                col
            )

    train_control, features = (
        apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )
    )

    val_control, val_features = (
        apply_hbeb_pair(
            df_train=train_exact,
            df_apply=val_exact,
            base_features=exact_features,
        )
    )

    if (
        list(features)
        != list(val_features)
    ):
        raise RuntimeError(
            "HB-EB feature order mismatch"
        )

    if (
        len(features)
        != EXPECTED_CONTROL_FEATURES
    ):
        raise RuntimeError(
            f"control feature count="
            f"{len(features)} "
            f"!= {EXPECTED_CONTROL_FEATURES}"
        )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in features
    ]

    return {
        "train": train_control,
        "val": val_control,
        "features": features,
        "cat_cols": cat_cols,
        "pitcher_stats": pitcher_stats,
    }


# =============================================================================
# Previous Anchor
# =============================================================================

def build_season_league_means(
    df_train,
):
    means = (
        df_train
        .groupby(
            SEASON_COL,
            sort=True,
        )[TARGET]
        .mean()
        .astype(
            np.float64
        )
    )

    return {
        int(k): float(v)
        for k, v in means.items()
    }


def league_prev_for(
    target_year,
    season_league_means,
):
    prev = (
        int(target_year) - 1
    )

    if prev in season_league_means:
        return float(
            season_league_means[
                prev
            ]
        )

    older = [
        value
        for year, value
        in season_league_means.items()
        if year < int(target_year)
    ]

    if older:
        return float(
            np.mean(
                older
            )
        )

    return float(
        ANCHOR_FALLBACK_RATE
    )


def compute_prev_anchor(
    rows,
    pitcher_season_stats,
    season_league_means,
):
    seasons = (
        pd.to_numeric(
            rows[
                SEASON_COL
            ],
            errors="raise",
        )
        .astype(int)
    )

    stats = (
        pitcher_season_stats
        .copy()
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    anchor = np.full(
        len(rows),
        np.nan,
        dtype=np.float64,
    )

    positions = np.arange(
        len(rows)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        mask = (
            seasons
            .eq(
                int(target_year)
            )
            .to_numpy()
        )

        pos = positions[
            mask
        ]

        sub = rows.loc[
            mask
        ]

        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist) > 0:
            agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[
                    [
                        "n",
                        "success",
                    ]
                ]
                .sum()
            )

            hist_n = (
                sub[
                    "pitcher_id"
                ]
                .map(
                    agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                sub[
                    "pitcher_id"
                ]
                .map(
                    agg[
                        "success"
                    ]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

        else:
            hist_n = np.zeros(
                len(sub),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(sub),
                dtype=np.float64,
            )

        league_prev = (
            league_prev_for(
                int(target_year),
                season_league_means,
            )
        )

        anchor[
            pos
        ] = (
            hist_success
            + ANCHOR_STRENGTH
            * league_prev
        ) / (
            hist_n
            + ANCHOR_STRENGTH
        )

    if not np.isfinite(
        anchor
    ).all():
        raise RuntimeError(
            "Previous anchor contains NaN/Inf"
        )

    return np.clip(
        anchor,
        0.0,
        1.0,
    )


# =============================================================================
# Dynamic ASOF Anchor
# =============================================================================

def compute_dynamic_anchor(
    exact_view,
    prev_anchor,
):
    prev_anchor = np.asarray(
        prev_anchor,
        dtype=np.float64,
    )

    n_t = (
        pd.to_numeric(
            exact_view[
                "exact_pitcher_season_n"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    r_t = (
        pd.to_numeric(
            exact_view[
                "exact_pitcher_season_success_rate"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    valid = (
        (n_t > 0.0)
        & np.isfinite(
            r_t
        )
    )

    reliability = np.zeros(
        len(exact_view),
        dtype=np.float64,
    )

    reliability[
        valid
    ] = (
        n_t[
            valid
        ]
        / (
            n_t[
                valid
            ]
            + DYNAMIC_KAPPA
        )
    )

    alpha = (
        DYNAMIC_LAMBDA
        * reliability
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )

    current_state = (
        prev_anchor.copy()
    )

    current_state[
        valid
    ] = np.clip(
        r_t[
            valid
        ],
        0.0,
        1.0,
    )

    dynamic_anchor = (
        (1.0 - alpha)
        * prev_anchor
        + alpha
        * current_state
    )

    if not np.isfinite(
        dynamic_anchor
    ).all():
        raise RuntimeError(
            "Dynamic anchor contains NaN/Inf"
        )

    diagnostics = {
        "asof_available_rate": float(
            np.mean(
                valid
            )
        ),

        "mean_current_n": float(
            np.mean(
                n_t
            )
        ),

        "median_current_n": float(
            np.median(
                n_t
            )
        ),

        "mean_alpha": float(
            np.mean(
                alpha
            )
        ),

        "p90_alpha": float(
            np.quantile(
                alpha,
                0.90,
            )
        ),

        "mean_abs_anchor_shift": float(
            np.mean(
                np.abs(
                    dynamic_anchor
                    - prev_anchor
                )
            )
        ),
    }

    return (
        np.clip(
            dynamic_anchor,
            0.0,
            1.0,
        ),
        diagnostics,
    )


# =============================================================================
# CatBoost
# =============================================================================

def fit_control_model(
    train_view,
    val_view,
    features,
    cat_cols,
    y_train,
    y_val,
    seed,
    validation_year,
):
    params = {
        **BASE_MODEL_PARAMS,
        "random_seed": int(seed),
    }

    model = (
        CatBoostRegressor(
            **params
        )
    )

    X_train = prepare_features(
        train_view[
            features
        ],
        cat_cols,
    )

    X_val = prepare_features(
        val_view[
            features
        ],
        cat_cols,
    )

    print()
    print(
        f"[CONTROL]"
        f" year={validation_year}"
        f" seed={seed}"
        f" | rows={len(X_train):,}"
        f" -> {len(X_val):,}"
    )

    t0 = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_val,
            y_val,
        ),
        cat_features=cat_cols,
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
        verbose=100,
    )

    elapsed = (
        time.time() - t0
    )

    pred = np.clip(
        np.asarray(
            model.predict(
                X_val
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    best_zero = int(
        model.get_best_iteration()
    )

    if best_zero < 0:
        raise RuntimeError(
            "Control best iteration invalid"
        )

    return (
        pred,
        best_zero,
        elapsed,
    )


def fit_dynamic_residual_model(
    train_view,
    val_view,
    features,
    cat_cols,
    y_train,
    y_val,
    dynamic_anchor_train,
    dynamic_anchor_val,
    seed,
    validation_year,
):
    params = {
        **BASE_MODEL_PARAMS,
        "random_seed": int(seed),
    }

    model = (
        CatBoostRegressor(
            **params
        )
    )

    X_train = prepare_features(
        train_view[
            features
        ],
        cat_cols,
    )

    X_val = prepare_features(
        val_view[
            features
        ],
        cat_cols,
    )

    print()
    print(
        f"[DYNAMIC]"
        f" year={validation_year}"
        f" seed={seed}"
        f" | lambda={DYNAMIC_LAMBDA}"
        f" kappa={DYNAMIC_KAPPA}"
    )

    t0 = time.time()

    model.fit(
        X_train,
        y_train
        - dynamic_anchor_train,
        eval_set=(
            X_val,
            y_val
            - dynamic_anchor_val,
        ),
        cat_features=cat_cols,
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
        verbose=100,
    )

    elapsed = (
        time.time() - t0
    )

    correction = np.asarray(
        model.predict(
            X_val
        ),
        dtype=np.float64,
    )

    pred = np.clip(
        dynamic_anchor_val
        + correction,
        0.0,
        1.0,
    )

    best_zero = int(
        model.get_best_iteration()
    )

    if best_zero < 0:
        raise RuntimeError(
            "Dynamic best iteration invalid"
        )

    return (
        pred,
        correction,
        best_zero,
        elapsed,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()

    print("=" * 112)
    print(EXPERIMENT_NAME)
    print("=" * 112)

    print(
        "Purpose      : confirm Control + Dynamic ASOF F blend"
    )

    print(
        f"Dynamic λ    : {DYNAMIC_LAMBDA}"
    )

    print(
        f"Dynamic κ    : {DYNAMIC_KAPPA}"
    )

    print(
        f"F blend w    : {F_BLEND_WEIGHT}"
    )

    print(
        f"Seeds        : {MODEL_SEEDS}"
    )

    print(
        "Validation   : expanding 2021~2024"
    )

    print(
        "Primary gate : 2024 F gain > 0 for 3/3 seeds"
    )

    print(
        "Test usage   : HEADER ONLY"
    )

    print(
        "b_F          : OFF"
    )

    print(
        "L2           : OFF"
    )

    print("=" * 112)

    # -----------------------------------------------------------------
    # File check
    # -----------------------------------------------------------------

    for path in [
        TRAIN_PATH,
        TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보호를 위해 중단: "
            f"{VALIDATION_DIR}"
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------------------
    # test header ONLY
    # -----------------------------------------------------------------

    test_cols = (
        pd.read_csv(
            TEST_PATH,
            encoding="utf-8-sig",
            nrows=0,
        )
        .columns
        .tolist()
    )

    base_features = [
        c
        for c in test_cols
        if c != ID
    ]

    if (
        "asof_pitcher_pitchmix_n"
        not in base_features
    ):
        raise RuntimeError(
            "canonical feature "
            "asof_pitcher_pitchmix_n missing"
        )

    # -----------------------------------------------------------------
    # train
    # -----------------------------------------------------------------

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    train[SEASON_COL] = (
        pd.to_numeric(
            train[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    required = {
        ID,
        TARGET,
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    }

    missing = sorted(
        required
        - set(train.columns)
    )

    if missing:
        raise KeyError(
            f"train required columns missing: "
            f"{missing}"
        )

    metrics_rows = []
    prediction_frames = []
    anchor_diag_rows = []

    # =================================================================
    # Expanding folds
    # =================================================================

    for validation_year in VALIDATION_YEARS:
        print()
        print("#" * 112)
        print(
            f"VALIDATION YEAR = "
            f"{validation_year}"
        )
        print("#" * 112)

        fold_train = (
            train.loc[
                train[SEASON_COL]
                < int(validation_year)
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        fold_val = (
            train.loc[
                train[SEASON_COL]
                == int(validation_year)
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if (
            len(fold_train) == 0
            or len(fold_val) == 0
        ):
            raise RuntimeError(
                f"Fold rows missing: "
                f"{validation_year}"
            )

        views = make_control_views(
            fold_train=fold_train,
            fold_val=fold_val,
            base_features=base_features,
        )

        y_train = (
            fold_train[
                TARGET
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_val = (
            fold_val[
                TARGET
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        game_type = (
            fold_val[
                GAME_TYPE_COL
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .to_numpy()
        )

        f_mask = (
            game_type == "F"
        )

        if not f_mask.any():
            raise RuntimeError(
                f"No F rows in "
                f"{validation_year}"
            )

        # -------------------------------------------------------------
        # Previous anchor
        # -------------------------------------------------------------

        season_league_means = (
            build_season_league_means(
                fold_train
            )
        )

        prev_anchor_train = (
            compute_prev_anchor(
                fold_train,
                views[
                    "pitcher_stats"
                ],
                season_league_means,
            )
        )

        prev_anchor_val = (
            compute_prev_anchor(
                fold_val,
                views[
                    "pitcher_stats"
                ],
                season_league_means,
            )
        )

        # -------------------------------------------------------------
        # Dynamic Anchor
        # -------------------------------------------------------------

        (
            dynamic_anchor_train,
            dynamic_diag_train,
        ) = compute_dynamic_anchor(
            exact_view=views[
                "train"
            ],
            prev_anchor=(
                prev_anchor_train
            ),
        )

        (
            dynamic_anchor_val,
            dynamic_diag_val,
        ) = compute_dynamic_anchor(
            exact_view=views[
                "val"
            ],
            prev_anchor=(
                prev_anchor_val
            ),
        )

        anchor_diag_rows.append(
            {
                "validation_year": int(
                    validation_year
                ),

                "n_validation": int(
                    len(fold_val)
                ),

                "n_F": int(
                    f_mask.sum()
                ),

                "lambda": float(
                    DYNAMIC_LAMBDA
                ),

                "kappa": float(
                    DYNAMIC_KAPPA
                ),

                "train_mean_alpha": (
                    dynamic_diag_train[
                        "mean_alpha"
                    ]
                ),

                "val_mean_alpha": (
                    dynamic_diag_val[
                        "mean_alpha"
                    ]
                ),

                "val_p90_alpha": (
                    dynamic_diag_val[
                        "p90_alpha"
                    ]
                ),

                "val_mean_current_n": (
                    dynamic_diag_val[
                        "mean_current_n"
                    ]
                ),

                "val_median_current_n": (
                    dynamic_diag_val[
                        "median_current_n"
                    ]
                ),

                "val_asof_available_rate": (
                    dynamic_diag_val[
                        "asof_available_rate"
                    ]
                ),

                "val_mean_abs_anchor_shift": (
                    dynamic_diag_val[
                        "mean_abs_anchor_shift"
                    ]
                ),
            }
        )

        # =============================================================
        # Seed
        # =============================================================

        for seed in MODEL_SEEDS:
            print()
            print("=" * 112)

            print(
                f"YEAR={validation_year}"
                f" | SEED={seed}"
            )

            print("=" * 112)

            # ---------------------------------------------------------
            # Control
            # ---------------------------------------------------------

            (
                control_pred,
                control_best_zero,
                control_elapsed,
            ) = fit_control_model(
                train_view=views[
                    "train"
                ],
                val_view=views[
                    "val"
                ],
                features=views[
                    "features"
                ],
                cat_cols=views[
                    "cat_cols"
                ],
                y_train=y_train,
                y_val=y_val,
                seed=seed,
                validation_year=(
                    validation_year
                ),
            )

            # ---------------------------------------------------------
            # Dynamic residual
            # ---------------------------------------------------------

            (
                dynamic_pred,
                correction,
                dynamic_best_zero,
                dynamic_elapsed,
            ) = fit_dynamic_residual_model(
                train_view=views[
                    "train"
                ],
                val_view=views[
                    "val"
                ],
                features=views[
                    "features"
                ],
                cat_cols=views[
                    "cat_cols"
                ],
                y_train=y_train,
                y_val=y_val,
                dynamic_anchor_train=(
                    dynamic_anchor_train
                ),
                dynamic_anchor_val=(
                    dynamic_anchor_val
                ),
                seed=seed,
                validation_year=(
                    validation_year
                ),
            )

            # ---------------------------------------------------------
            # F only
            # ---------------------------------------------------------

            y_f = (
                y_val[
                    f_mask
                ]
            )

            control_f = (
                control_pred[
                    f_mask
                ]
            )

            dynamic_f = (
                dynamic_pred[
                    f_mask
                ]
            )

            # ---------------------------------------------------------
            # CONFIRMATORY BLEND
            # ---------------------------------------------------------

            blend_f = np.clip(
                (
                    1.0
                    - F_BLEND_WEIGHT
                )
                * control_f
                + F_BLEND_WEIGHT
                * dynamic_f,
                0.0,
                1.0,
            )

            # ---------------------------------------------------------
            # Dynamic standalone diagnostic
            # ---------------------------------------------------------

            dynamic_cmp = (
                compare_to_control(
                    y_true=y_f,
                    control_pred=control_f,
                    candidate_pred=dynamic_f,
                )
            )

            # ---------------------------------------------------------
            # Blend = PRIMARY candidate
            # ---------------------------------------------------------

            blend_cmp = (
                compare_to_control(
                    y_true=y_f,
                    control_pred=control_f,
                    candidate_pred=blend_f,
                )
            )

            # ---------------------------------------------------------
            # Whole-fold BSS impact
            #
            # R rows remain identical.
            #
            # Therefore this is exactly the contribution caused by
            # replacing only the F branch.
            # ---------------------------------------------------------

            hybrid_blend = (
                control_pred.copy()
            )

            hybrid_blend[
                f_mask
            ] = blend_f

            control_bss_all = (
                calculate_bss(
                    y_val,
                    control_pred,
                )
            )

            blend_bss_all = (
                calculate_bss(
                    y_val,
                    hybrid_blend,
                )
            )

            delta_bss_all = (
                blend_bss_all
                - control_bss_all
            )

            print()
            print(
                f"[CONFIRM RESULT]"
                f" year={validation_year}"
                f" seed={seed}"
            )

            print(
                f"CONTROL F Brier : "
                f"{blend_cmp['control_brier']:.10f}"
            )

            print(
                f"DYNAMIC F Brier : "
                f"{dynamic_cmp['candidate_brier']:.10f}"
            )

            print(
                f"BLEND F Brier   : "
                f"{blend_cmp['candidate_brier']:.10f}"
            )

            print(
                f"DYNAMIC F gain  : "
                f"{dynamic_cmp['total_gain_pt']:+.3f} pt"
            )

            print(
                f"BLEND F gain    : "
                f"{blend_cmp['total_gain_pt']:+.3f} pt"
            )

            print(
                f"  level         : "
                f"{blend_cmp['level_gain_pt']:+.3f}"
            )

            print(
                f"  shape         : "
                f"{blend_cmp['shape_gain_pt']:+.3f}"
            )

            print(
                f"Pooled ΔBSS     : "
                f"{delta_bss_all:+.3f}"
            )

            print(
                f"PASS            : "
                f"{blend_cmp['total_gain_pt'] > 0.0}"
            )

            metrics_rows.append(
                {
                    "validation_year": int(
                        validation_year
                    ),

                    "seed": int(
                        seed
                    ),

                    "n_validation": int(
                        len(fold_val)
                    ),

                    "n_F": int(
                        f_mask.sum()
                    ),

                    "lambda": float(
                        DYNAMIC_LAMBDA
                    ),

                    "kappa": float(
                        DYNAMIC_KAPPA
                    ),

                    "blend_weight": float(
                        F_BLEND_WEIGHT
                    ),

                    "control_tree_count": int(
                        control_best_zero
                        + 1
                    ),

                    "dynamic_tree_count": int(
                        dynamic_best_zero
                        + 1
                    ),

                    "control_fit_seconds": float(
                        control_elapsed
                    ),

                    "dynamic_fit_seconds": float(
                        dynamic_elapsed
                    ),

                    # Dynamic standalone
                    "dynamic_brier_F": (
                        dynamic_cmp[
                            "candidate_brier"
                        ]
                    ),

                    "dynamic_gain_F_pt": (
                        dynamic_cmp[
                            "total_gain_pt"
                        ]
                    ),

                    "dynamic_level_gain_F_pt": (
                        dynamic_cmp[
                            "level_gain_pt"
                        ]
                    ),

                    "dynamic_shape_gain_F_pt": (
                        dynamic_cmp[
                            "shape_gain_pt"
                        ]
                    ),

                    # Confirmatory blend
                    "control_brier_F": (
                        blend_cmp[
                            "control_brier"
                        ]
                    ),

                    "blend_brier_F": (
                        blend_cmp[
                            "candidate_brier"
                        ]
                    ),

                    "blend_bias_F": (
                        blend_cmp[
                            "candidate_bias"
                        ]
                    ),

                    "blend_gain_F_pt": (
                        blend_cmp[
                            "total_gain_pt"
                        ]
                    ),

                    "blend_level_gain_F_pt": (
                        blend_cmp[
                            "level_gain_pt"
                        ]
                    ),

                    "blend_shape_gain_F_pt": (
                        blend_cmp[
                            "shape_gain_pt"
                        ]
                    ),

                    "blend_win": bool(
                        blend_cmp[
                            "total_gain_pt"
                        ]
                        > 0.0
                    ),

                    # Whole fold
                    "control_bss_all": float(
                        control_bss_all
                    ),

                    "blend_bss_all": float(
                        blend_bss_all
                    ),

                    "delta_bss_all": float(
                        delta_bss_all
                    ),

                    # residual diagnostics
                    "correction_mean": float(
                        np.mean(
                            correction
                        )
                    ),

                    "correction_std": float(
                        np.std(
                            correction
                        )
                    ),

                    "correction_abs_mean": float(
                        np.mean(
                            np.abs(
                                correction
                            )
                        )
                    ),
                }
            )

            # ---------------------------------------------------------
            # Paired F prediction
            # ---------------------------------------------------------

            prediction_frames.append(
                pd.DataFrame(
                    {
                        ID: (
                            fold_val.loc[
                                f_mask,
                                ID,
                            ]
                            .to_numpy()
                        ),

                        "validation_year": int(
                            validation_year
                        ),

                        "seed": int(
                            seed
                        ),

                        "y_true": y_f,

                        "control_pred": (
                            control_f
                        ),

                        "prev_anchor": (
                            prev_anchor_val[
                                f_mask
                            ]
                        ),

                        "dynamic_anchor": (
                            dynamic_anchor_val[
                                f_mask
                            ]
                        ),

                        "dynamic_pred": (
                            dynamic_f
                        ),

                        "blend_pred": (
                            blend_f
                        ),
                    }
                )
            )

    # =================================================================
    # Save raw metrics
    # =================================================================

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    anchor_diag_df = pd.DataFrame(
        anchor_diag_rows
    )

    metrics_df.to_csv(
        VALIDATION_DIR
        / "blend_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        VALIDATION_DIR
        / "f_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anchor_diag_df.to_csv(
        VALIDATION_DIR
        / "anchor_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =================================================================
    # Year summary
    # =================================================================

    year_summary = (
        metrics_df
        .groupby(
            "validation_year",
            as_index=False,
        )
        .agg(
            seed_count=(
                "seed",
                "size",
            ),

            seed_win_count=(
                "blend_win",
                "sum",
            ),

            mean_control_brier_F=(
                "control_brier_F",
                "mean",
            ),

            mean_dynamic_brier_F=(
                "dynamic_brier_F",
                "mean",
            ),

            mean_blend_brier_F=(
                "blend_brier_F",
                "mean",
            ),

            mean_dynamic_gain_F_pt=(
                "dynamic_gain_F_pt",
                "mean",
            ),

            mean_blend_gain_F_pt=(
                "blend_gain_F_pt",
                "mean",
            ),

            std_blend_gain_F_pt=(
                "blend_gain_F_pt",
                "std",
            ),

            min_blend_gain_F_pt=(
                "blend_gain_F_pt",
                "min",
            ),

            max_blend_gain_F_pt=(
                "blend_gain_F_pt",
                "max",
            ),

            mean_blend_level_gain_F_pt=(
                "blend_level_gain_F_pt",
                "mean",
            ),

            mean_blend_shape_gain_F_pt=(
                "blend_shape_gain_F_pt",
                "mean",
            ),

            mean_delta_bss_all=(
                "delta_bss_all",
                "mean",
            ),
        )
    )

    year_summary.to_csv(
        VALIDATION_DIR
        / "year_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =================================================================
    # Seed summary
    # =================================================================

    seed_summary = (
        metrics_df
        .groupby(
            "seed",
            as_index=False,
        )
        .agg(
            fold_count=(
                "validation_year",
                "size",
            ),

            fold_win_count=(
                "blend_win",
                "sum",
            ),

            mean_blend_gain_F_pt=(
                "blend_gain_F_pt",
                "mean",
            ),

            min_blend_gain_F_pt=(
                "blend_gain_F_pt",
                "min",
            ),

            mean_delta_bss_all=(
                "delta_bss_all",
                "mean",
            ),
        )
    )

    latest_by_seed = (
        metrics_df.loc[
            metrics_df[
                "validation_year"
            ]
            == 2024,
            [
                "seed",
                "blend_gain_F_pt",
                "blend_level_gain_F_pt",
                "blend_shape_gain_F_pt",
                "delta_bss_all",
                "blend_win",
            ],
        ]
        .rename(
            columns={
                "blend_gain_F_pt": (
                    "gain_2024_F_pt"
                ),

                "blend_level_gain_F_pt": (
                    "level_gain_2024_F_pt"
                ),

                "blend_shape_gain_F_pt": (
                    "shape_gain_2024_F_pt"
                ),

                "delta_bss_all": (
                    "delta_bss_2024_all"
                ),

                "blend_win": (
                    "win_2024"
                ),
            }
        )
    )

    seed_summary = (
        seed_summary
        .merge(
            latest_by_seed,
            on="seed",
            how="left",
        )
    )

    seed_summary.to_csv(
        VALIDATION_DIR
        / "seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =================================================================
    # Confirmation gate
    # =================================================================

    latest = (
        metrics_df.loc[
            metrics_df[
                "validation_year"
            ]
            == 2024
        ]
        .copy()
    )

    if len(latest) != len(
        MODEL_SEEDS
    ):
        raise RuntimeError(
            "2024 seed result count mismatch"
        )

    primary_seed_wins = int(
        latest[
            "blend_win"
        ].sum()
    )

    primary_pass = bool(
        primary_seed_wins
        == len(MODEL_SEEDS)
    )

    year_mean_positive = {
        int(row["validation_year"]):
        bool(
            row[
                "mean_blend_gain_F_pt"
            ]
            > 0.0
        )
        for _, row
        in year_summary.iterrows()
    }

    temporal_mean_win_count = int(
        sum(
            year_mean_positive.values()
        )
    )

    # =================================================================
    # Metadata
    # =================================================================

    confirmation = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "experiment_type": (
            "confirmatory"
        ),

        "frozen_before_run": {
            "dynamic_lambda": float(
                DYNAMIC_LAMBDA
            ),

            "dynamic_kappa": float(
                DYNAMIC_KAPPA
            ),

            "f_blend_weight": float(
                F_BLEND_WEIGHT
            ),

            "model_seeds": list(
                MODEL_SEEDS
            ),
        },

        "prediction_formula": (
            "F=(1-w)*Control + "
            "w*DynamicASOF"
        ),

        "primary_gate": (
            "2024 blend F gain > 0 "
            "for all 3 model seeds"
        ),

        "primary_2024_seed_wins": (
            primary_seed_wins
        ),

        "primary_2024_seed_total": int(
            len(MODEL_SEEDS)
        ),

        "primary_pass": (
            primary_pass
        ),

        "temporal_mean_positive": (
            year_mean_positive
        ),

        "temporal_mean_win_count": (
            temporal_mean_win_count
        ),

        "temporal_fold_count": int(
            len(VALIDATION_YEARS)
        ),

        "interpretation_note": (
            "seed replication tests model "
            "randomness robustness; "
            "it is not temporal replication"
        ),

        "rule_compliance": {
            "test_rows_used": False,

            "test_usage": (
                "header only"
            ),

            "test_row_aggregation": False,

            "historical_anchor_source": (
                "official labeled fold-train only"
            ),

            "dynamic_state_source": (
                "current row official ASOF only"
            ),

            "other_test_rows_used": False,
        },

        "elapsed_seconds": float(
            time.time()
            - total_start
        ),
    }

    (
        VALIDATION_DIR
        / "confirmation.json"
    ).write_text(
        json.dumps(
            to_jsonable(
                confirmation
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =================================================================
    # Print
    # =================================================================

    print()
    print("=" * 112)
    print("YEAR SUMMARY")
    print("=" * 112)

    print(
        year_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print("SEED SUMMARY")
    print("=" * 112)

    print(
        seed_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print("CONFIRMATION")
    print("=" * 112)

    print(
        f"2024 seed wins : "
        f"{primary_seed_wins}"
        f"/{len(MODEL_SEEDS)}"
    )

    print(
        f"PRIMARY PASS   : "
        f"{primary_pass}"
    )

    print(
        f"Temporal mean wins : "
        f"{temporal_mean_win_count}"
        f"/{len(VALIDATION_YEARS)}"
    )

    print()
    print("Saved:")

    for file_name in [
        "blend_metrics.csv",
        "f_predictions.csv",
        "anchor_diagnostics.csv",
        "year_summary.csv",
        "seed_summary.csv",
        "confirmation.json",
    ]:
        print(
            VALIDATION_DIR
            / file_name
        )

    print(
        f"\nTotal elapsed: "
        f"{time.time() - total_start:.1f}s"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
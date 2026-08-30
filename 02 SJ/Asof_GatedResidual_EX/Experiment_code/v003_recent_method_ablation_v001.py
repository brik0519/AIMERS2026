# ============================================================
# v003_recent_method_ablation_v001.py
#
# 목적
# ------------------------------------------------------------
# "최근 데이터를 더 중요하게 반영하는 가장 효과적인 방법"을 비교한다.
#
# canonical cb_reg_v003를 기준으로:
#
# A. Base
# B. Base + Exact-ASOF
# C. Base + Recent Weight
# D. Base + Exact-ASOF + Recent Weight
#
# 를 동일한 Expanding Window 환경에서 비교한다.
#
# ------------------------------------------------------------
# Exact-ASOF
# ------------------------------------------------------------
# 기존 exact-asof branch에서 사용했던 5개 pitcher feature:
#
# 1. exact_pitcher_season_n
# 2. exact_pitcher_season_logn
# 3. exact_pitcher_season_success_rate
# 4. exact_pitcher_past_success_rate
# 5. exact_pitcher_season_minus_past
#
# 현재 row의 공식:
#   asof_pitcher_n
#   asof_pitcher_success_rate
#
# 와 과거 시즌 train 통계만 이용해,
# 현재 시즌의 "그 시점 직전 상태"를 복원한다.
#
# ------------------------------------------------------------
# Recent Weight
# ------------------------------------------------------------
# old_050 아이디어를 Expanding Window에 맞게 적용:
#
# 각 Fold의 가장 최근 train season = 1.0
# 그 이전 season                  = 0.5
#
# 이후 평균 weight=1이 되도록 normalize.
#
# 예:
# 2019~2023 -> 2024
#
# 2019  0.5
# 2020  0.5
# 2021  0.5
# 2022  0.5
# 2023  1.0
#
# ------------------------------------------------------------
# 중요한 규칙
# ------------------------------------------------------------
# - asof_pitcher_pitchmix_n 반드시 유지
# - HB-EB는 canonical v003와 동일
# - CatBoost 파라미터 동일
# - Expanding Window 동일
# - Early Stopping 동일
# - seed 동일
# - V18 correction 없음
# - test.csv는 header만 사용
# - test 다른 row 사용 없음
#
# ------------------------------------------------------------
# 판정
# ------------------------------------------------------------
# Base가 기존 canonical v003 결과를 먼저 재현해야 한다.
#
# 이후:
# Exact - Base
# Weight - Base
# Both - Base
#
# 뿐 아니라
#
# Both - Exact  : Exact가 있어도 Weight가 추가 효과?
# Both - Weight : Weight가 있어도 Exact가 추가 효과?
#
# 까지 확인한다.
#
# Brier는 낮을수록 좋다.
# 출력 Delta는:
#
#   Base Brier - Candidate Brier
#
# 따라서 Delta > 0 이면 개선이다.
# ============================================================


import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = (
    "v003_recent_method_ablation_v001"
)

DATA_DIR = Path("./data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = (
    VALIDATION_ROOT / EXPERIMENT_NAME
)


# ============================================================
# 기본 설정
# ============================================================

ID = "row_id"
TARGET = "control_success"

SEASON_COL = "season"
GAME_TYPE_COL = "game_type"


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


EXPANDING_FOLDS = [
    {
        "fold": 1,
        "train_seasons": [
            2019,
            2020,
        ],
        "validation_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [
            2019,
            2020,
            2021,
        ],
        "validation_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [
            2019,
            2020,
            2021,
            2022,
        ],
        "validation_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [
            2019,
            2020,
            2021,
            2022,
            2023,
        ],
        "validation_season": 2024,
    },
]


# ============================================================
# 실험 조건
# ============================================================

VARIANTS = {
    "base": {
        "exact_asof": False,
        "recent_weight": False,
    },
    "exact_asof": {
        "exact_asof": True,
        "recent_weight": False,
    },
    "recent_weight": {
        "exact_asof": False,
        "recent_weight": True,
    },
    "exact_asof_weight": {
        "exact_asof": True,
        "recent_weight": True,
    },
}


# ============================================================
# Exact-ASOF Feature
# ============================================================

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


# ============================================================
# HB-EB
# ============================================================

HBEB_CONFIGS = [
    {
        "group_cols": [
            "pitcher_hand"
        ],
        "n_col": (
            "asof_pitcher_n"
        ),
        "rate_col": (
            "asof_pitcher_success_rate"
        ),
        "new_col": (
            "hbeb_pitcher_success_rate"
        ),
    },
    {
        "group_cols": [
            "batter_hand"
        ],
        "n_col": (
            "asof_batter_n"
        ),
        "rate_col": (
            "asof_batter_success_rate"
        ),
        "new_col": (
            "hbeb_batter_success_rate"
        ),
    },
]


# ============================================================
# CatBoost
# ============================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# ============================================================
# canonical v003 재현값
# ============================================================

EXPECTED_BASE_FOLD_BRIER = {
    2021: 0.2457566023316621,
    2022: 0.24337582123208099,
    2023: 0.2514010454095717,
    2024: 0.2480912722504766,
}

EXPECTED_BASE_POOLED_BRIER = (
    0.24715408781215828
)

# 머신/라이브러리 미세 차이는 허용하되,
# 이보다 크게 다르면 환경이 달라졌다고 판단.
BASE_REPRO_TOL = 5e-6


# ============================================================
# Metric
# ============================================================

def brier_score(
    y_true,
    y_pred,
):
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
            (
                y_pred
                - y_true
            ) ** 2
        )
    )


def calculate_bss(
    y_true,
    y_pred,
):
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

    positive_rate = float(
        np.mean(y_true)
    )

    baseline_brier = (
        positive_rate
        * (
            1.0
            - positive_rate
        )
    )

    if baseline_brier <= 0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier
            / baseline_brier
        )
    )


def domain_brier(
    y_true,
    y_pred,
    game_type,
    domain,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    game_type = (
        pd.Series(game_type)
        .astype(str)
        .to_numpy()
    )

    mask = (
        game_type
        == str(domain)
    )

    if mask.sum() == 0:
        return np.nan

    return brier_score(
        y_true[mask],
        y_pred[mask],
    )


# ============================================================
# 범주형 전처리
# ============================================================

def prepare_features(
    df,
    cat_cols,
):
    df = df.copy()

    for col in cat_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )

    return df


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    canonical v003와 동일.

    현재 Fold의 train subset만 이용해
    Beta prior를 추정한다.
    """

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
        g_mu
        * (
            1.0
            - g_mu
        )
    )

    if (
        not np.isfinite(
            g_var
        )
        or g_var <= 0.0
        or g_var
        >= max_bernoulli_var
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
        (
            1.0
            - g_mu
        )
        * g_m
    )

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[
            rate_col
        ]
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
        stats[
            "var"
        ].notna()
        & np.isfinite(
            stats["var"]
        )
        & (
            stats["var"]
            > 0.0
        )
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"]
            >= 30
        )
    )

    stats["alpha"] = (
        g_alpha
    )

    stats["beta"] = (
        g_beta
    )

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
            * (
                1.0
                - mu
            )
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
            (
                1.0
                - mu
            )
            * m
        )

    prior_df = stats[
        group_cols
        + [
            "alpha",
            "beta",
        ]
    ].copy()

    global_prior = (
        float(g_alpha),
        float(g_beta),
    )

    return (
        prior_df,
        global_prior,
    )


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """
    train-only prior를 각 row에 lookup.
    """

    original_index = (
        df.index
    )

    lookup = (
        df[
            group_cols
        ]
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
        lookup[
            "alpha"
        ]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup[
            "beta"
        ]
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

    eb_rate = (
        success_count
        + alpha
    ) / denominator

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def apply_hbeb_pair(
    df_train,
    df_val,
    base_features,
):
    """
    Fold train만으로 prior를 만들고
    train / validation에 동일하게 적용.
    """

    train_out = (
        df_train.copy()
    )

    val_out = (
        df_val.copy()
    )

    features = list(
        base_features
    )

    for config in (
        HBEB_CONFIGS
    ):

        (
            prior_df,
            global_prior,
        ) = (
            compute_hbeb_priors(
                df_train=(
                    df_train
                ),
                group_cols=(
                    config[
                        "group_cols"
                    ]
                ),
                rate_col=(
                    config[
                        "rate_col"
                    ]
                ),
            )
        )

        train_out[
            config[
                "new_col"
            ]
        ] = apply_hbeb(
            df=train_out,
            group_cols=(
                config[
                    "group_cols"
                ]
            ),
            n_col=(
                config[
                    "n_col"
                ]
            ),
            rate_col=(
                config[
                    "rate_col"
                ]
            ),
            prior_df=(
                prior_df
            ),
            global_prior=(
                global_prior
            ),
        )

        val_out[
            config[
                "new_col"
            ]
        ] = apply_hbeb(
            df=val_out,
            group_cols=(
                config[
                    "group_cols"
                ]
            ),
            n_col=(
                config[
                    "n_col"
                ]
            ),
            rate_col=(
                config[
                    "rate_col"
                ]
            ),
            prior_df=(
                prior_df
            ),
            global_prior=(
                global_prior
            ),
        )

        # raw rate를 HB-EB로 교체
        if (
            config[
                "rate_col"
            ]
            in features
        ):
            features.remove(
                config[
                    "rate_col"
                ]
            )

        if (
            config[
                "new_col"
            ]
            not in features
        ):
            features.append(
                config[
                    "new_col"
                ]
            )

    return (
        train_out,
        val_out,
        features,
    )


# ============================================================
# Exact-ASOF용 pitcher-season 과거 통계
# ============================================================

def build_pitcher_season_stats(
    df_train,
):
    """
    공식 train label만 이용해
    pitcher × season의 완결 시즌 통계를 생성.

    이 통계는 target row의 season보다
    과거 season에 대해서만 사용된다.
    """

    required = {
        "pitcher_id",
        "season",
        TARGET,
    }

    missing = (
        required
        - set(
            df_train.columns
        )
    )

    if missing:
        raise KeyError(
            "pitcher season stats "
            "필수 컬럼 없음: "
            f"{sorted(missing)}"
        )

    stats = (
        df_train
        .groupby(
            [
                "pitcher_id",
                "season",
            ],
            dropna=False,
            sort=False,
        )[
            TARGET
        ]
        .agg(
            n="size",
            success="sum",
        )
        .reset_index()
    )

    stats["season"] = (
        pd.to_numeric(
            stats["season"],
            errors="raise",
        )
        .astype(int)
    )

    stats["n"] = (
        stats["n"]
        .astype(
            np.float64
        )
    )

    stats["success"] = (
        stats[
            "success"
        ]
        .astype(
            np.float64
        )
    )

    return stats


# ============================================================
# Exact-ASOF 생성
# ============================================================

def add_pitcher_exact_asof_features(
    df,
    pitcher_season_stats,
):
    """
    기존 Exact-ASOF branch와 같은 핵심 방식.

    각 row에서:
      career/cumulative ASOF
      - 과거 완결 시즌 통계
      =
      현재 시즌 ASOF 상태

    다른 validation/test row는 사용하지 않는다.
    """

    required = {
        "season",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            "Exact-ASOF 필수 컬럼 없음: "
            f"{sorted(missing)}"
        )

    out = df.copy()

    for col in (
        EXACT_FEATURES
    ):
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[
                "season"
            ],
            errors="raise",
        )
        .astype(int)
    )

    stats = (
        pitcher_season_stats
        .copy()
    )

    stats["season"] = (
        pd.to_numeric(
            stats[
                "season"
            ],
            errors="raise",
        )
        .astype(int)
    )

    diagnostic_rows = []

    for target_year in sorted(
        seasons.unique()
    ):

        row_mask = (
            seasons
            == int(
                target_year
            )
        )

        rows = (
            out.loc[
                row_mask
            ]
        )

        # target year보다 과거인
        # 공식 train season만 사용
        hist = (
            stats.loc[
                stats[
                    "season"
                ]
                < int(
                    target_year
                )
            ]
        )

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
                rows[
                    "pitcher_id"
                ]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                rows[
                    "pitcher_id"
                ]
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

            hist_n = (
                np.zeros(
                    len(rows),
                    dtype=np.float64,
                )
            )

            hist_success = (
                np.zeros(
                    len(rows),
                    dtype=np.float64,
                )
            )

        asof_n = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_n"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
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

        # -----------------------------------------------
        # 현재 시즌 sample 수
        # -----------------------------------------------

        raw_current_n = (
            asof_n
            - hist_n
        )

        negative_n = (
            raw_current_n
            < -1e-8
        )

        current_n = (
            np.maximum(
                raw_current_n,
                0.0,
            )
        )

        # -----------------------------------------------
        # 현재 시즌 success 수
        # -----------------------------------------------

        raw_current_success = (
            asof_success
            - hist_success
        )

        invalid_success = (
            np.isfinite(
                raw_current_success
            )
            & (
                (
                    raw_current_success
                    < -1e-8
                )
                |
                (
                    raw_current_success
                    > current_n
                    + 1e-8
                )
            )
        )

        current_success = (
            np.where(
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
        )

        # -----------------------------------------------
        # 현재 시즌 rate
        # -----------------------------------------------

        current_rate = (
            np.full(
                len(rows),
                np.nan,
                dtype=np.float64,
            )
        )

        valid_current = (
            (
                current_n
                > 0.0
            )
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

        # -----------------------------------------------
        # 과거 시즌 rate
        # -----------------------------------------------

        past_rate = (
            np.full(
                len(rows),
                np.nan,
                dtype=np.float64,
            )
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

        idx = (
            rows.index
        )

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = (
            current_n
        )

        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = (
            np.log1p(
                current_n
            )
        )

        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = (
            current_rate
        )

        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = (
            past_rate
        )

        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = (
            current_rate
            - past_rate
        )

        diagnostic_rows.append(
            {
                "season": int(
                    target_year
                ),
                "n_rows": int(
                    len(rows)
                ),
                "negative_current_n": int(
                    negative_n.sum()
                ),
                "invalid_current_success": int(
                    invalid_success.sum()
                ),
                "current_rate_available": int(
                    np.isfinite(
                        current_rate
                    ).sum()
                ),
                "past_rate_available": int(
                    np.isfinite(
                        past_rate
                    ).sum()
                ),
            }
        )

    diagnostics = (
        pd.DataFrame(
            diagnostic_rows
        )
    )

    return (
        out,
        diagnostics,
    )


# ============================================================
# Recent Weight
# ============================================================

def make_recent_sample_weight(
    season_series,
):
    """
    Expanding Window-safe old_050 style.

    가장 최근 train season = 1.0
    과거 train season       = 0.5

    이후 평균 1로 normalize.
    """

    seasons = (
        pd.to_numeric(
            season_series,
            errors="raise",
        )
        .astype(int)
    )

    latest_season = int(
        seasons.max()
    )

    raw_weight = (
        np.where(
            seasons.to_numpy()
            == latest_season,
            1.0,
            0.5,
        )
        .astype(
            np.float64
        )
    )

    weight = (
        raw_weight
        / raw_weight.mean()
    )

    return (
        weight,
        {
            "latest_train_season": (
                latest_season
            ),
            "old_raw_weight": 0.5,
            "latest_raw_weight": 1.0,
            "normalized_mean": float(
                weight.mean()
            ),
        },
    )


# ============================================================
# 단일 Fold 학습
# ============================================================

def run_single_fold(
    train,
    base_features,
    fold_config,
    variant_name,
    variant_config,
):
    fold = int(
        fold_config["fold"]
    )

    train_seasons = (
        fold_config[
            "train_seasons"
        ]
    )

    validation_season = int(
        fold_config[
            "validation_season"
        ]
    )

    train_mask = (
        train[
            SEASON_COL
        ].isin(
            train_seasons
        )
    )

    val_mask = (
        train[
            SEASON_COL
        ]
        == validation_season
    )

    df_train = (
        train.loc[
            train_mask
        ]
        .copy()
    )

    df_val = (
        train.loc[
            val_mask
        ]
        .copy()
    )

    variant_features = list(
        base_features
    )

    exact_diagnostics = None

    # ====================================================
    # Exact-ASOF
    # ====================================================

    if (
        variant_config[
            "exact_asof"
        ]
    ):

        pitcher_season_stats = (
            build_pitcher_season_stats(
                df_train
            )
        )

        (
            df_train,
            train_exact_diag,
        ) = (
            add_pitcher_exact_asof_features(
                df=df_train,
                pitcher_season_stats=(
                    pitcher_season_stats
                ),
            )
        )

        (
            df_val,
            val_exact_diag,
        ) = (
            add_pitcher_exact_asof_features(
                df=df_val,
                pitcher_season_stats=(
                    pitcher_season_stats
                ),
            )
        )

        for col in (
            EXACT_FEATURES
        ):

            if (
                col
                not in variant_features
            ):
                variant_features.append(
                    col
                )

        train_exact_diag[
            "scope"
        ] = "train"

        val_exact_diag[
            "scope"
        ] = "validation"

        exact_diagnostics = (
            pd.concat(
                [
                    train_exact_diag,
                    val_exact_diag,
                ],
                ignore_index=True,
            )
        )

    # ====================================================
    # HB-EB
    # ====================================================

    (
        df_train,
        df_val,
        final_features,
    ) = apply_hbeb_pair(
        df_train=df_train,
        df_val=df_val,
        base_features=(
            variant_features
        ),
    )

    # ====================================================
    # Feature
    # ====================================================

    cat_cols = [
        c
        for c in CAT_COLS
        if c in final_features
    ]

    X_train = (
        prepare_features(
            df_train[
                final_features
            ],
            cat_cols,
        )
    )

    X_val = (
        prepare_features(
            df_val[
                final_features
            ],
            cat_cols,
        )
    )

    y_train = (
        df_train[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    y_val = (
        df_val[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    # ====================================================
    # Weight
    # ====================================================

    sample_weight = None
    weight_info = None

    if (
        variant_config[
            "recent_weight"
        ]
    ):

        (
            sample_weight,
            weight_info,
        ) = (
            make_recent_sample_weight(
                df_train[
                    SEASON_COL
                ]
            )
        )

    # ====================================================
    # Model
    # ====================================================

    print()
    print("=" * 90)

    print(
        f"{variant_name}"
        f" | Fold {fold}"
        f" | {train_seasons[0]}"
        f"~{train_seasons[-1]}"
        f" -> {validation_season}"
    )

    print(
        f"Exact-ASOF    : "
        f"{variant_config['exact_asof']}"
    )

    print(
        f"Recent Weight : "
        f"{variant_config['recent_weight']}"
    )

    print(
        f"Features      : "
        f"{len(final_features)}"
    )

    if weight_info is not None:

        print(
            "Weight policy : "
            f"latest {weight_info['latest_train_season']}=1.0, "
            "older=0.5"
        )

    print("=" * 90)

    model = (
        CatBoostRegressor(
            **MODEL_PARAMS
        )
    )

    start = (
        time.time()
    )

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "cat_features": (
            cat_cols
        ),
        "eval_set": (
            X_val,
            y_val,
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "use_best_model": True,
    }

    if (
        sample_weight
        is not None
    ):
        fit_kwargs[
            "sample_weight"
        ] = sample_weight

    model.fit(
        **fit_kwargs
    )

    elapsed = (
        time.time()
        - start
    )

    pred_raw = (
        np.asarray(
            model.predict(
                X_val
            ),
            dtype=np.float64,
        )
    )

    pred = (
        np.clip(
            pred_raw,
            0.0,
            1.0,
        )
    )

    brier = (
        brier_score(
            y_val,
            pred,
        )
    )

    bss = (
        calculate_bss(
            y_val,
            pred,
        )
    )

    r_brier = (
        domain_brier(
            y_true=y_val,
            y_pred=pred,
            game_type=(
                df_val[
                    GAME_TYPE_COL
                ]
            ),
            domain="R",
        )
    )

    f_brier = (
        domain_brier(
            y_true=y_val,
            y_pred=pred,
            game_type=(
                df_val[
                    GAME_TYPE_COL
                ]
            ),
            domain="F",
        )
    )

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        final_iterations = (
            MODEL_PARAMS[
                "iterations"
            ]
        )
    else:
        final_iterations = (
            int(
                best_iteration
            )
            + 1
        )

    print()
    print(
        f"Brier        : "
        f"{brier:.10f}"
    )

    print(
        f"BSS          : "
        f"{bss:.2f}"
    )

    print(
        f"R Brier      : "
        f"{r_brier:.10f}"
        if np.isfinite(
            r_brier
        )
        else "R Brier      : N/A"
    )

    print(
        f"F Brier      : "
        f"{f_brier:.10f}"
        if np.isfinite(
            f_brier
        )
        else "F Brier      : N/A"
    )

    print(
        f"Best iter    : "
        f"{best_iteration}"
    )

    print(
        f"Elapsed      : "
        f"{elapsed:.1f}s"
    )

    result = {
        "variant": (
            variant_name
        ),
        "fold": fold,
        "train_seasons": (
            ",".join(
                map(
                    str,
                    train_seasons,
                )
            )
        ),
        "validation_season": (
            validation_season
        ),
        "exact_asof": bool(
            variant_config[
                "exact_asof"
            ]
        ),
        "recent_weight": bool(
            variant_config[
                "recent_weight"
            ]
        ),
        "n_train": int(
            len(df_train)
        ),
        "n_validation": int(
            len(df_val)
        ),
        "n_features": int(
            len(
                final_features
            )
        ),
        "brier": float(
            brier
        ),
        "bss": float(
            bss
        ),
        "r_brier": (
            float(
                r_brier
            )
            if np.isfinite(
                r_brier
            )
            else np.nan
        ),
        "f_brier": (
            float(
                f_brier
            )
            if np.isfinite(
                f_brier
            )
            else np.nan
        ),
        "best_iteration": int(
            best_iteration
        ),
        "final_iterations": int(
            final_iterations
        ),
        "elapsed_seconds": float(
            elapsed
        ),
    }

    oof = pd.DataFrame(
        {
            ID: (
                df_val[
                    ID
                ].to_numpy()
            ),
            "target": (
                y_val.to_numpy()
            ),
            "prediction_raw": (
                pred_raw
            ),
            "prediction": (
                pred
            ),
            "game_type": (
                df_val[
                    GAME_TYPE_COL
                ]
                .astype(str)
                .to_numpy()
            ),
            "validation_season": (
                validation_season
            ),
            "variant": (
                variant_name
            ),
        }
    )

    return (
        result,
        oof,
        exact_diagnostics,
    )


# ============================================================
# Variant 전체 Fold 실행
# ============================================================

def run_variant(
    train,
    base_features,
    variant_name,
):
    config = (
        VARIANTS[
            variant_name
        ]
    )

    results = []
    oof_list = []
    diag_list = []

    print()
    print()
    print("#" * 100)

    print(
        f"VARIANT: "
        f"{variant_name}"
    )

    print("#" * 100)

    for fold_config in (
        EXPANDING_FOLDS
    ):

        (
            result,
            oof,
            exact_diag,
        ) = run_single_fold(
            train=train,
            base_features=(
                base_features
            ),
            fold_config=(
                fold_config
            ),
            variant_name=(
                variant_name
            ),
            variant_config=(
                config
            ),
        )

        results.append(
            result
        )

        oof_list.append(
            oof
        )

        if (
            exact_diag
            is not None
        ):

            exact_diag = (
                exact_diag.copy()
            )

            exact_diag[
                "variant"
            ] = (
                variant_name
            )

            exact_diag[
                "validation_season"
            ] = (
                fold_config[
                    "validation_season"
                ]
            )

            diag_list.append(
                exact_diag
            )

    results_df = (
        pd.DataFrame(
            results
        )
    )

    oof_df = (
        pd.concat(
            oof_list,
            ignore_index=True,
        )
    )

    if diag_list:

        diag_df = (
            pd.concat(
                diag_list,
                ignore_index=True,
            )
        )

    else:

        diag_df = None

    return (
        results_df,
        oof_df,
        diag_df,
    )


# ============================================================
# Base 재현 검증
# ============================================================

def verify_base_reproduction(
    base_results,
    base_oof,
):
    print()
    print("=" * 100)
    print("CANONICAL v003 REPRODUCTION CHECK")
    print("=" * 100)

    failed = False

    for _, row in (
        base_results
        .sort_values(
            "validation_season"
        )
        .iterrows()
    ):

        season = int(
            row[
                "validation_season"
            ]
        )

        actual = float(
            row["brier"]
        )

        expected = (
            EXPECTED_BASE_FOLD_BRIER[
                season
            ]
        )

        diff = (
            actual
            - expected
        )

        status = (
            "PASS"
            if abs(diff)
            <= BASE_REPRO_TOL
            else "FAIL"
        )

        if (
            status == "FAIL"
        ):
            failed = True

        print(
            f"{season}: "
            f"actual={actual:.10f} | "
            f"expected={expected:.10f} | "
            f"diff={diff:+.10f} | "
            f"{status}"
        )

    pooled = (
        brier_score(
            base_oof[
                "target"
            ],
            base_oof[
                "prediction"
            ],
        )
    )

    pooled_diff = (
        pooled
        - EXPECTED_BASE_POOLED_BRIER
    )

    pooled_status = (
        "PASS"
        if abs(
            pooled_diff
        )
        <= BASE_REPRO_TOL
        else "FAIL"
    )

    if (
        pooled_status
        == "FAIL"
    ):
        failed = True

    print()

    print(
        f"Pooled: "
        f"actual={pooled:.10f} | "
        f"expected="
        f"{EXPECTED_BASE_POOLED_BRIER:.10f} | "
        f"diff={pooled_diff:+.10f} | "
        f"{pooled_status}"
    )

    print("=" * 100)

    if failed:

        raise RuntimeError(
            "\n"
            "Base가 canonical cb_reg_v003를 "
            "재현하지 못했습니다.\n"
            "이 상태에서 Exact-ASOF/Weight를 비교하면 "
            "실험 환경이 달라질 수 있으므로 중단합니다."
        )


# ============================================================
# 전체 요약
# ============================================================

def make_summary(
    all_results,
    all_oof,
):
    summary_rows = []

    for variant in (
        VARIANTS.keys()
    ):

        oof = (
            all_oof.loc[
                all_oof[
                    "variant"
                ]
                == variant
            ]
            .copy()
        )

        folds = (
            all_results.loc[
                all_results[
                    "variant"
                ]
                == variant
            ]
            .copy()
        )

        pooled_brier = (
            brier_score(
                oof[
                    "target"
                ],
                oof[
                    "prediction"
                ],
            )
        )

        pooled_bss = (
            calculate_bss(
                oof[
                    "target"
                ],
                oof[
                    "prediction"
                ],
            )
        )

        pooled_r = (
            domain_brier(
                y_true=(
                    oof[
                        "target"
                    ]
                ),
                y_pred=(
                    oof[
                        "prediction"
                    ]
                ),
                game_type=(
                    oof[
                        "game_type"
                    ]
                ),
                domain="R",
            )
        )

        pooled_f = (
            domain_brier(
                y_true=(
                    oof[
                        "target"
                    ]
                ),
                y_pred=(
                    oof[
                        "prediction"
                    ]
                ),
                game_type=(
                    oof[
                        "game_type"
                    ]
                ),
                domain="F",
            )
        )

        recent = (
            oof.loc[
                oof[
                    "validation_season"
                ].isin(
                    [
                        2023,
                        2024,
                    ]
                )
            ]
        )

        recent_brier = (
            brier_score(
                recent[
                    "target"
                ],
                recent[
                    "prediction"
                ],
            )
        )

        summary_rows.append(
            {
                "variant": (
                    variant
                ),
                "pooled_oof_brier": (
                    pooled_brier
                ),
                "pooled_oof_bss": (
                    pooled_bss
                ),
                "pooled_r_brier": (
                    pooled_r
                ),
                "pooled_f_brier": (
                    pooled_f
                ),
                "recent_2023_2024_brier": (
                    recent_brier
                ),
                "mean_fold_brier": float(
                    folds[
                        "brier"
                    ].mean()
                ),
                "std_fold_brier": float(
                    folds[
                        "brier"
                    ].std(
                        ddof=0
                    )
                ),
                "mean_best_iteration": float(
                    folds[
                        "final_iterations"
                    ].mean()
                ),
            }
        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    base = (
        summary.loc[
            summary[
                "variant"
            ]
            == "base"
        ]
        .iloc[0]
    )

    summary[
        "pooled_delta_vs_base"
    ] = (
        base[
            "pooled_oof_brier"
        ]
        - summary[
            "pooled_oof_brier"
        ]
    )

    summary[
        "recent_delta_vs_base"
    ] = (
        base[
            "recent_2023_2024_brier"
        ]
        - summary[
            "recent_2023_2024_brier"
        ]
    )

    summary[
        "r_delta_vs_base"
    ] = (
        base[
            "pooled_r_brier"
        ]
        - summary[
            "pooled_r_brier"
        ]
    )

    summary[
        "f_delta_vs_base"
    ] = (
        base[
            "pooled_f_brier"
        ]
        - summary[
            "pooled_f_brier"
        ]
    )

    return summary


# ============================================================
# Fold별 비교표
# ============================================================

def make_fold_comparison(
    all_results,
):
    brier_wide = (
        all_results
        .pivot(
            index=(
                "validation_season"
            ),
            columns="variant",
            values="brier",
        )
        .reset_index()
    )

    for variant in [
        "exact_asof",
        "recent_weight",
        "exact_asof_weight",
    ]:

        brier_wide[
            f"delta_base_vs_{variant}"
        ] = (
            brier_wide[
                "base"
            ]
            - brier_wide[
                variant
            ]
        )

    # Exact가 있는 상태에서
    # Weight를 추가했을 때 개선
    brier_wide[
        "extra_weight_given_exact"
    ] = (
        brier_wide[
            "exact_asof"
        ]
        - brier_wide[
            "exact_asof_weight"
        ]
    )

    # Weight가 있는 상태에서
    # Exact를 추가했을 때 개선
    brier_wide[
        "extra_exact_given_weight"
    ] = (
        brier_wide[
            "recent_weight"
        ]
        - brier_wide[
            "exact_asof_weight"
        ]
    )

    return brier_wide


# ============================================================
# Main
# ============================================================

def main():

    if (
        VALIDATION_DIR.exists()
    ):

        raise FileExistsError(
            f"{VALIDATION_DIR} already exists.\n"
            "기존 결과를 보존하려면 "
            "EXPERIMENT_NAME을 바꾸세요."
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # Feature schema
    # ========================================================

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

    # canonical v003 확인
    required_pitchmix = (
        "asof_pitcher_pitchmix_n"
    )

    if (
        required_pitchmix
        not in base_features
    ):

        raise ValueError(
            "canonical cb_reg_v003에서 "
            "유지해야 하는 "
            "'asof_pitcher_pitchmix_n'이 없습니다."
        )

    required_exact_sources = [
        "season",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    ]

    missing_exact_sources = [
        c
        for c in required_exact_sources
        if c not in base_features
    ]

    if (
        missing_exact_sources
    ):

        raise ValueError(
            "Exact-ASOF source feature 누락: "
            f"{missing_exact_sources}"
        )

    # ========================================================
    # Train
    # ========================================================

    usecols = list(
        dict.fromkeys(
            [
                ID,
                *base_features,
                TARGET,
            ]
        )
    )

    train = (
        pd.read_csv(
            TRAIN_PATH,
            encoding="utf-8-sig",
            usecols=usecols,
        )
    )

    if (
        not train[
            ID
        ].is_unique
    ):

        raise ValueError(
            "train row_id가 unique하지 않습니다."
        )

    if (
        train[
            TARGET
        ].isna().any()
    ):

        raise ValueError(
            "target에 NaN이 있습니다."
        )

    print()
    print("=" * 100)

    print(
        f"Experiment : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        "Base       : "
        "canonical cb_reg_v003"
    )

    print(
        f"Features   : "
        f"{len(base_features)}"
    )

    print(
        "pitchmix_n retained : "
        f"{required_pitchmix in base_features}"
    )

    print(
        "Variants   : "
        "Base / Exact-ASOF / "
        "Recent Weight / Both"
    )

    print(
        "Weight     : "
        "latest train season=1.0, older=0.5"
    )

    print(
        "V18        : OFF"
    )

    print("=" * 100)

    # ========================================================
    # 1. BASE 먼저 실행
    # ========================================================

    (
        base_results,
        base_oof,
        _,
    ) = run_variant(
        train=train,
        base_features=(
            base_features
        ),
        variant_name="base",
    )

    # Base가 canonical v003와 맞는지 확인.
    # 안 맞으면 여기서 중단.
    verify_base_reproduction(
        base_results=(
            base_results
        ),
        base_oof=(
            base_oof
        ),
    )

    all_results = [
        base_results
    ]

    all_oof = [
        base_oof
    ]

    all_diags = []

    # ========================================================
    # 2. Treatment 3개
    # ========================================================

    for variant_name in [
        "exact_asof",
        "recent_weight",
        "exact_asof_weight",
    ]:

        (
            results,
            oof,
            diag,
        ) = run_variant(
            train=train,
            base_features=(
                base_features
            ),
            variant_name=(
                variant_name
            ),
        )

        all_results.append(
            results
        )

        all_oof.append(
            oof
        )

        if (
            diag is not None
        ):
            all_diags.append(
                diag
            )

    all_results = (
        pd.concat(
            all_results,
            ignore_index=True,
        )
    )

    all_oof = (
        pd.concat(
            all_oof,
            ignore_index=True,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = (
        make_summary(
            all_results=(
                all_results
            ),
            all_oof=(
                all_oof
            ),
        )
    )

    fold_comparison = (
        make_fold_comparison(
            all_results
        )
    )

    # ========================================================
    # 저장
    # ========================================================

    all_results.to_csv(
        VALIDATION_DIR
        / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_oof.to_csv(
        VALIDATION_DIR
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        VALIDATION_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_comparison.to_csv(
        VALIDATION_DIR
        / "fold_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if (
        all_diags
    ):

        exact_diag = (
            pd.concat(
                all_diags,
                ignore_index=True,
            )
        )

        exact_diag.to_csv(
            VALIDATION_DIR
            / "exact_asof_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base_model": (
            "cb_reg_v003"
        ),
        "canonical_pitchmix_retained": (
            True
        ),
        "variants": (
            VARIANTS
        ),
        "exact_features": (
            EXACT_FEATURES
        ),
        "weight_policy": {
            "method": (
                "latest_one_train_season_1.0_older_0.5"
            ),
            "normalized_mean": 1.0,
        },
        "model_params": (
            MODEL_PARAMS
        ),
        "folds": (
            EXPANDING_FOLDS
        ),
        "v18_correction": (
            False
        ),
    }

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # 출력
    # ========================================================

    print()
    print()
    print("=" * 120)
    print("FOLD COMPARISON")
    print("모든 Delta > 0 = 개선")
    print("=" * 120)

    print(
        fold_comparison
        .to_string(
            index=False
        )
    )

    print()
    print()
    print("=" * 120)
    print("TOTAL SUMMARY")
    print("pooled_delta_vs_base > 0 = Base보다 개선")
    print("=" * 120)

    print(
        summary
        .sort_values(
            "pooled_oof_brier"
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)

    print(
        f"Saved to: "
        f"{VALIDATION_DIR}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
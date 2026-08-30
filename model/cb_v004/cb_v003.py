# cb_v003.py
#
# CatBoostClassifier version of cb_reg_v003_exact
#
# Experiment 1:
#
#   1) Iteration selection
#      2019~2022 -> 2023
#
#   2) Weight-model training
#      2019~2023 + 2024 H1
#
#   3) Pure holdout prediction
#      2024 H2
#
# 목적:
#   cb_reg_v003_exact와 함께 2024 H2 ensemble weight를
#   추정하기 위한 CatBoostClassifier prediction 생성.
#
# Feature:
#   canonical cb_reg_v003
#   + Exact-ASOF 5
#   + canonical HB-EB
#
# 규칙:
# - test.csv는 schema(header)만 사용한다.
# - test 값/분포/다른 row를 사용하지 않는다.
# - Exact-ASOF는 공식 train 과거 완료 시즌 통계와
#   현재 row의 공식 asof_* 값만 사용한다.
# - HB-EB prior는 해당 training 구간에서만 fit한다.
# - 2024 H2 target은 모델 학습, iteration 선택,
#   Exact-ASOF bank, HB-EB prior에 사용하지 않는다.
# - 2024 H2는 최종 holdout 평가 및 ensemble weight 산출에만 사용한다.


from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# Config
# ============================================================

EXPERIMENT_NAME = "cb_v004"

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


ID = "row_id"
TARGET = "control_success"

SEASON_COL = "season"
TIME_COL = "game_month"

WEIGHT_HOLDOUT_YEAR = 2024
ITERATION_VALIDATION_YEAR = 2023

WEIGHT_SPLIT_MODE = "balanced_month"


# ============================================================
# Canonical categorical policy
# ============================================================

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


# ============================================================
# Exact-ASOF
# ============================================================

EXACT_COLS = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


# ============================================================
# Canonical HB-EB
# ============================================================

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


# ============================================================
# CatBoostClassifier
# ============================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "eval_metric": "BrierScore",
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
# Common
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> pd.DataFrame:

    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


def calculate_metrics(
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

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    positive_rate = float(
        np.mean(y_true)
    )

    baseline_brier = float(
        positive_rate
        * (1.0 - positive_rate)
    )

    if baseline_brier <= 0.0:
        bss_raw = 0.0
    else:
        bss_raw = float(
            100000.0
            * (
                1.0
                - brier / baseline_brier
            )
        )

    return {
        "brier": brier,
        "positive_rate": positive_rate,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": max(
            0.0,
            bss_raw,
        ),
    }


# ============================================================
# 2024 H1 / H2
# ============================================================

def select_balanced_month_cutoff(
    df: pd.DataFrame,
    year: int,
) -> dict:
    """
    game_month block을 유지하면서
    2024 row 수가 가능한 한 50:50에 가까워지는
    cutoff month를 선택한다.

    month <= cutoff:
        H1

    month > cutoff:
        H2

    target은 split 선택에 사용하지 않는다.
    """

    required = {
        SEASON_COL,
        TIME_COL,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"H1/H2 split 필수 컬럼 없음: {missing}"
        )

    season = pd.to_numeric(
        df[SEASON_COL],
        errors="raise",
    ).astype(int)

    year_df = df.loc[
        season == int(year),
        [TIME_COL],
    ].copy()

    if year_df.empty:
        raise ValueError(
            f"{year} 데이터가 없습니다."
        )

    month = pd.to_numeric(
        year_df[TIME_COL],
        errors="raise",
    ).astype(int)

    months = sorted(
        int(x)
        for x in month.unique()
    )

    if len(months) < 2:
        raise ValueError(
            "H1/H2 split 가능한 month가 부족합니다."
        )

    candidates = []

    for cutoff in months[:-1]:

        h1_mask = (
            month <= cutoff
        )

        h2_mask = (
            month > cutoff
        )

        n_h1 = int(
            h1_mask.sum()
        )

        n_h2 = int(
            h2_mask.sum()
        )

        if (
            n_h1 == 0
            or n_h2 == 0
        ):
            continue

        total = (
            n_h1 + n_h2
        )

        candidates.append({
            "cutoff_month": int(cutoff),
            "n_h1": n_h1,
            "n_h2": n_h2,
            "h1_share": float(
                n_h1 / total
            ),
            "h2_share": float(
                n_h2 / total
            ),
            "imbalance": int(
                abs(
                    n_h1 - n_h2
                )
            ),
        })

    if not candidates:
        raise RuntimeError(
            "유효한 H1/H2 split을 찾지 못했습니다."
        )

    best = min(
        candidates,
        key=lambda x: (
            x["imbalance"],
            x["cutoff_month"],
        ),
    )

    return {
        "year": int(year),
        "split_mode": WEIGHT_SPLIT_MODE,
        "time_col": TIME_COL,
        "available_months": months,
        "selected_cutoff_month": int(
            best["cutoff_month"]
        ),
        "n_h1": int(
            best["n_h1"]
        ),
        "n_h2": int(
            best["n_h2"]
        ),
        "h1_share": float(
            best["h1_share"]
        ),
        "h2_share": float(
            best["h2_share"]
        ),
        "candidates": candidates,
    }


def build_holdout_masks(
    df: pd.DataFrame,
    split_state: dict,
):

    year = int(
        split_state["year"]
    )

    cutoff = int(
        split_state[
            "selected_cutoff_month"
        ]
    )

    season = pd.to_numeric(
        df[SEASON_COL],
        errors="raise",
    ).astype(int)

    month = pd.to_numeric(
        df[TIME_COL],
        errors="raise",
    ).astype(int)

    train_mask = (
        (season < year)
        |
        (
            (season == year)
            & (month <= cutoff)
        )
    )

    holdout_mask = (
        (season == year)
        & (month > cutoff)
    )

    if not train_mask.any():
        raise RuntimeError(
            "Weight training row가 없습니다."
        )

    if not holdout_mask.any():
        raise RuntimeError(
            "2024 H2 row가 없습니다."
        )

    if np.any(
        train_mask
        & holdout_mask
    ):
        raise RuntimeError(
            "H1/H2 overlap detected."
        )

    return (
        train_mask,
        holdout_mask,
    )


# ============================================================
# Exact-ASOF
# ============================================================

def build_pitcher_season_stats(
    history: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "pitcher_id",
        SEASON_COL,
        TARGET,
    }

    missing = sorted(
        required
        - set(history.columns)
    )

    if missing:
        raise KeyError(
            "pitcher season stats "
            f"필수 컬럼 없음: {missing}"
        )

    stats = (
        history
        .groupby(
            [
                "pitcher_id",
                SEASON_COL,
            ],
            sort=False,
            dropna=False,
        )[TARGET]
        .agg(
            ["size", "sum"]
        )
        .reset_index()
        .rename(
            columns={
                "size": "n",
                "sum": "success",
            }
        )
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


def add_pitcher_exact_asof_features(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required
        - set(rows.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 컬럼 없음: {missing}"
        )

    out = rows.copy()

    for col in EXACT_COLS:
        out[col] = np.nan

    row_season = (
        pd.to_numeric(
            out[SEASON_COL],
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

    for target_year in sorted(
        row_season.unique()
    ):

        mask = (
            row_season
            == int(target_year)
        )

        part = out.loc[
            mask
        ]

        # 현재 row season보다 엄격히 과거 시즌만 사용.
        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist):

            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[
                    ["n", "success"]
                ]
                .sum()
            )

            hist_n = (
                part["pitcher_id"]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                part["pitcher_id"]
                .map(
                    hist_agg["success"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

        else:

            hist_n = np.zeros(
                len(part),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(part),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                part[
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
                part[
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

        current_n = np.maximum(
            asof_n
            - hist_n,
            0.0,
        )

        current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                current_success
            ),
            np.clip(
                current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(part),
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
            len(part),
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

        idx = part.index

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


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train: pd.DataFrame,
    group_cols: list[str],
    rate_col: str,
):

    rate = pd.to_numeric(
        df_train[rate_col],
        errors="coerce",
    )

    g_mu = float(
        rate.mean()
    )

    g_var = float(
        rate.var()
    )

    max_bernoulli_var = (
        g_mu
        * (1.0 - g_mu)
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
        g_mu
        * g_m
    )

    g_beta = (
        (1.0 - g_mu)
        * g_m
    )

    tmp = (
        df_train[
            group_cols
        ]
        .copy()
    )

    tmp["_rate_"] = rate

    stats = (
        tmp
        .groupby(
            group_cols,
            dropna=False,
        )["_rate_"]
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

    valid = (
        stats["var"].notna()
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

    if valid.any():

        mu = stats.loc[
            valid,
            "mean",
        ]

        var = stats.loc[
            valid,
            "var",
        ]

        m = (
            mu
            * (1.0 - mu)
            / var
        ) - 1.0

        stats.loc[
            valid,
            "alpha",
        ] = (
            mu * m
        )

        stats.loc[
            valid,
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
    df: pd.DataFrame,
    group_cols: list[str],
    n_col: str,
    rate_col: str,
    prior_df: pd.DataFrame,
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


def add_hbeb_features(
    df_train_for_prior: pd.DataFrame,
    df_apply: pd.DataFrame,
    features: list[str],
):

    out = df_apply.copy()

    final_features = list(
        features
    )

    priors = {}

    for cfg in HBEB_CONFIGS:

        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train_for_prior,
            group_cols=cfg[
                "group_cols"
            ],
            rate_col=cfg[
                "rate_col"
            ],
        )

        out[
            cfg["new_col"]
        ] = apply_hbeb(
            df=out,
            group_cols=cfg[
                "group_cols"
            ],
            n_col=cfg[
                "n_col"
            ],
            rate_col=cfg[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            cfg["rate_col"]
            in final_features
        ):
            final_features.remove(
                cfg["rate_col"]
            )

        if (
            cfg["new_col"]
            not in final_features
        ):
            final_features.append(
                cfg["new_col"]
            )

        priors[
            cfg["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": global_prior,
            "group_cols": list(
                cfg["group_cols"]
            ),
            "n_col": cfg[
                "n_col"
            ],
            "rate_col": cfg[
                "rate_col"
            ],
        }

    return (
        out,
        final_features,
        priors,
    )


def apply_existing_hbeb_priors(
    df_apply: pd.DataFrame,
    features: list[str],
    priors: dict,
):

    out = df_apply.copy()

    final_features = list(
        features
    )

    for cfg in HBEB_CONFIGS:

        info = priors[
            cfg["new_col"]
        ]

        out[
            cfg["new_col"]
        ] = apply_hbeb(
            df=out,
            group_cols=info[
                "group_cols"
            ],
            n_col=info[
                "n_col"
            ],
            rate_col=info[
                "rate_col"
            ],
            prior_df=info[
                "prior_df"
            ],
            global_prior=info[
                "global_prior"
            ],
        )

        if (
            cfg["rate_col"]
            in final_features
        ):
            final_features.remove(
                cfg["rate_col"]
            )

        if (
            cfg["new_col"]
            not in final_features
        ):
            final_features.append(
                cfg["new_col"]
            )

    return (
        out,
        final_features,
    )


# ============================================================
# Shared Feature Builder
# ============================================================

def build_fold_features(
    df_train_raw: pd.DataFrame,
    df_apply_raw: pd.DataFrame,
    base_features: list[str],
):
    """
    df_train_raw:
        Exact-ASOF historical bank와 HB-EB prior의
        유일한 train source.

    df_apply_raw:
        transform 대상.

    apply 대상의 target은 feature 생성에 사용하지 않는다.
    """

    pitcher_stats = (
        build_pitcher_season_stats(
            df_train_raw
        )
    )

    df_train = (
        add_pitcher_exact_asof_features(
            df_train_raw,
            pitcher_stats,
        )
    )

    df_apply = (
        add_pitcher_exact_asof_features(
            df_apply_raw,
            pitcher_stats,
        )
    )

    base_plus_exact = (
        list(base_features)
        + [
            col
            for col in EXACT_COLS
            if col not in base_features
        ]
    )

    (
        df_train,
        final_features,
        hbeb_priors,
    ) = add_hbeb_features(
        df_train_for_prior=df_train_raw,
        df_apply=df_train,
        features=base_plus_exact,
    )

    (
        df_apply,
        final_features_check,
    ) = apply_existing_hbeb_priors(
        df_apply=df_apply,
        features=base_plus_exact,
        priors=hbeb_priors,
    )

    if (
        final_features
        != final_features_check
    ):
        raise RuntimeError(
            "train/apply feature 목록 불일치"
        )

    if (
        "asof_pitcher_pitchmix_n"
        not in final_features
    ):
        raise RuntimeError(
            "canonical v003 feature "
            "'asof_pitcher_pitchmix_n'이 사라졌습니다."
        )

    return (
        df_train,
        df_apply,
        final_features,
        pitcher_stats,
        hbeb_priors,
    )


# ============================================================
# Step 1
# 2019~2022 -> 2023 iteration selection
# ============================================================

def select_classifier_iterations(
    train: pd.DataFrame,
    base_features: list[str],
    cat_cols: list[str],
):
    """
    2024 H2 target을 early stopping에 사용하지 않기 위해
    iteration 수는 2023 transition에서 선택한다.

    Train:
        2019~2022

    Validation:
        2023
    """

    season = pd.to_numeric(
        train[SEASON_COL],
        errors="raise",
    ).astype(int)

    train_mask = (
        season < ITERATION_VALIDATION_YEAR
    )

    val_mask = (
        season == ITERATION_VALIDATION_YEAR
    )

    df_tr_raw = train.loc[
        train_mask
    ].copy()

    df_va_raw = train.loc[
        val_mask
    ].copy()

    if df_tr_raw.empty:
        raise RuntimeError(
            "Iteration selection train이 없습니다."
        )

    if df_va_raw.empty:
        raise RuntimeError(
            "2023 iteration validation이 없습니다."
        )

    (
        df_tr,
        df_va,
        final_features,
        _,
        _,
    ) = build_fold_features(
        df_train_raw=df_tr_raw,
        df_apply_raw=df_va_raw,
        base_features=base_features,
    )

    final_cat_cols = [
        col
        for col in cat_cols
        if col in final_features
    ]

    X_tr = prepare_features(
        df_tr[final_features],
        final_cat_cols,
    )

    X_va = prepare_features(
        df_va[final_features],
        final_cat_cols,
    )

    y_tr = (
        df_tr[TARGET]
        .astype(np.float64)
    )

    y_va = (
        df_va[TARGET]
        .astype(np.float64)
    )

    model = CatBoostClassifier(
        **MODEL_PARAMS
    )

    started = time.time()

    model.fit(
        X_tr,
        y_tr,
        cat_features=final_cat_cols,
        eval_set=(
            X_va,
            y_va,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = (
        time.time()
        - started
    )

    pred = np.asarray(
        model.predict_proba(
            X_va
        )[:, 1],
        dtype=np.float64,
    )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_va,
        pred,
    )

    best_iteration_zero = int(
        model.get_best_iteration()
    )

    if best_iteration_zero >= 0:
        final_iterations = (
            best_iteration_zero
            + 1
        )
    else:
        final_iterations = int(
            model.tree_count_
        )

    result = {
        "train_seasons": sorted(
            int(x)
            for x in
            df_tr_raw[
                SEASON_COL
            ].unique()
        ),

        "validation_season": int(
            ITERATION_VALIDATION_YEAR
        ),

        "n_train": int(
            len(df_tr_raw)
        ),

        "n_validation": int(
            len(df_va_raw)
        ),

        "brier": float(
            metrics["brier"]
        ),

        "bss_raw": float(
            metrics["bss_raw"]
        ),

        "best_iteration_zero_based": int(
            best_iteration_zero
        ),

        "final_iterations": int(
            final_iterations
        ),

        "elapsed_seconds": float(
            elapsed
        ),
    }

    return (
        final_iterations,
        result,
    )


# ============================================================
# Step 2 / 3
# H1 training -> H2 pure holdout
# ============================================================

def train_h1_validate_h2(
    train: pd.DataFrame,
    base_features: list[str],
    cat_cols: list[str],
    final_iterations: int,
):

    split_state = (
        select_balanced_month_cutoff(
            train,
            WEIGHT_HOLDOUT_YEAR,
        )
    )

    (
        train_mask,
        holdout_mask,
    ) = build_holdout_masks(
        train,
        split_state,
    )

    df_tr_raw = train.loc[
        train_mask
    ].copy()

    df_ho_raw = train.loc[
        holdout_mask
    ].copy()

    cutoff = int(
        split_state[
            "selected_cutoff_month"
        ]
    )

    # --------------------------------------------------------
    # Split sanity check
    # --------------------------------------------------------

    train_2024 = df_tr_raw.loc[
        df_tr_raw[SEASON_COL]
        == WEIGHT_HOLDOUT_YEAR
    ]

    holdout_2024 = df_ho_raw.loc[
        df_ho_raw[SEASON_COL]
        == WEIGHT_HOLDOUT_YEAR
    ]

    if (
        pd.to_numeric(
            train_2024[TIME_COL],
            errors="raise",
        )
        > cutoff
    ).any():
        raise RuntimeError(
            "2024 H2가 training에 들어갔습니다."
        )

    if (
        pd.to_numeric(
            holdout_2024[TIME_COL],
            errors="raise",
        )
        <= cutoff
    ).any():
        raise RuntimeError(
            "2024 H1이 holdout에 들어갔습니다."
        )

    # --------------------------------------------------------
    # Feature
    # --------------------------------------------------------

    (
        df_tr,
        df_ho,
        final_features,
        pitcher_stats,
        hbeb_priors,
    ) = build_fold_features(
        df_train_raw=df_tr_raw,
        df_apply_raw=df_ho_raw,
        base_features=base_features,
    )

    final_cat_cols = [
        col
        for col in cat_cols
        if col in final_features
    ]

    X_tr = prepare_features(
        df_tr[final_features],
        final_cat_cols,
    )

    X_ho = prepare_features(
        df_ho[final_features],
        final_cat_cols,
    )

    y_tr = (
        df_tr[TARGET]
        .astype(np.float64)
    )

    y_ho = (
        df_ho[TARGET]
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # Fixed-iteration training
    #
    # 중요:
    # H2 target을 eval_set으로 전달하지 않는다.
    # --------------------------------------------------------

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    model = CatBoostClassifier(
        **final_params
    )

    started = time.time()

    model.fit(
        X_tr,
        y_tr,
        cat_features=final_cat_cols,
    )

    elapsed = (
        time.time()
        - started
    )

    # --------------------------------------------------------
    # Pure H2 prediction
    # --------------------------------------------------------

    pred = np.asarray(
        model.predict_proba(
            X_ho
        )[:, 1],
        dtype=np.float64,
    )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_ho,
        pred,
    )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            "CatBoostClassifier"
        ),

        "role": (
            "2024 H1 -> H2 "
            "ensemble weight estimation"
        ),

        "source_feature_recipe": (
            "cb_reg_v003_exact"
        ),

        "model": model,

        "features": (
            final_features
        ),

        "cat_cols": (
            final_cat_cols
        ),

        "hbeb_priors": (
            hbeb_priors
        ),

        "pitcher_season_stats": (
            pitcher_stats
        ),

        "exact_cols": list(
            EXACT_COLS
        ),

        "split_state": (
            split_state
        ),

        "final_iterations": int(
            final_iterations
        ),

        "model_params": (
            final_params
        ),

        "prediction_policy": (
            "model.predict_proba(X)[:, 1]"
        ),

        "baseline_definition": (
            "canonical cb_reg_v003 feature recipe "
            "+ Exact-ASOF5 "
            "+ CatBoostClassifier(Logloss)"
        ),

        "training_policy": (
            "2019~2023 + 2024 H1"
        ),

        "validation_policy": (
            "2024 H2 pure holdout"
        ),
    }

    pred_df = pd.DataFrame({
        ID: (
            df_ho_raw[ID]
            .to_numpy()
        ),

        "target": (
            y_ho.to_numpy(
                dtype=np.float64
            )
        ),

        "prediction": (
            pred
        ),

        "validation_season": (
            WEIGHT_HOLDOUT_YEAR
        ),

        "validation_segment": (
            "2024_H2"
        ),
    })

    result = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            "CatBoostClassifier"
        ),

        "source_feature_recipe": (
            "cb_reg_v003_exact"
        ),

        "split_state": (
            split_state
        ),

        "n_train": int(
            len(df_tr_raw)
        ),

        "n_validation": int(
            len(df_ho_raw)
        ),

        "training_seasons": sorted(
            int(x)
            for x in
            df_tr_raw[
                SEASON_COL
            ].unique()
        ),

        "validation_season": int(
            WEIGHT_HOLDOUT_YEAR
        ),

        "validation_segment": (
            "2024_H2"
        ),

        "n_features": int(
            len(final_features)
        ),

        "features": (
            final_features
        ),

        "cat_cols": (
            final_cat_cols
        ),

        "exact_cols": (
            EXACT_COLS
        ),

        "hbeb_configs": (
            HBEB_CONFIGS
        ),

        "asof_pitcher_pitchmix_n_kept": (
            "asof_pitcher_pitchmix_n"
            in final_features
        ),

        "brier": float(
            metrics["brier"]
        ),

        "positive_rate": float(
            metrics[
                "positive_rate"
            ]
        ),

        "baseline_brier": float(
            metrics[
                "baseline_brier"
            ]
        ),

        "bss_raw": float(
            metrics["bss_raw"]
        ),

        "bss": float(
            metrics["bss"]
        ),

        "final_iterations": int(
            final_iterations
        ),

        "elapsed_seconds": float(
            elapsed
        ),

        "model_params": (
            final_params
        ),

        "rule_compliance": {
            "test_values_used": False,
            "cross_test_aggregation": False,
            "h2_target_used_for_training": False,
            "h2_target_used_for_iteration_selection": False,
            "h2_target_used_for_exact_asof_bank": False,
            "h2_target_used_for_hbeb_prior": False,
            "row_independent_test_transform": True,
        },
    }

    return (
        artifact,
        pred_df,
        result,
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            "기존 model 보존을 위해 중단: "
            f"{MODEL_DIR}"
        )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            "기존 validation 보존을 위해 중단: "
            f"{VALIDATION_DIR}"
        )

    # --------------------------------------------------------
    # test schema only
    # --------------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise KeyError(
            f"test schema에 '{ID}'가 없습니다."
        )

    base_features = [
        col
        for col in test_cols
        if col != ID
    ]

    if (
        "asof_pitcher_pitchmix_n"
        not in base_features
    ):
        raise RuntimeError(
            "test schema에 "
            "asof_pitcher_pitchmix_n이 없습니다. "
            "canonical cb_reg_v003_exact와 "
            "동일한 schema가 아닙니다."
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *base_features,
            TARGET,
        ],
    )

    required = {
        ID,
        TARGET,
        SEASON_COL,
        TIME_COL,
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
            f"train 필수 컬럼 없음: {missing}"
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"{TARGET}에 결측값이 있습니다."
        )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in base_features
    ]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 88)
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )
    print(
        "Model      : CatBoostClassifier"
    )
    print(
        "Loss       : Logloss"
    )
    print(
        "Eval metric: BrierScore"
    )
    print(
        "Feature    : canonical cb_reg_v003 + Exact-ASOF5"
    )
    print(
        "HB-EB      : ON"
    )
    print(
        "Residual   : NO"
    )
    print(
        "Conditional: NO"
    )
    print(
        "Routing    : NO"
    )
    print(
        "pitchmix_n : KEEP"
    )
    print(
        "H2 early stop: NO"
    )
    print(
        f"Train rows : {len(train):,}"
    )
    print(
        f"Base feats : {len(base_features)}"
    )
    print("=" * 88)

    # --------------------------------------------------------
    # Directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # Stage 1
    # 2019~2022 -> 2023
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Stage 1: iteration selection"
    )
    print(
        "2019~2022 -> 2023"
    )
    print("=" * 88)

    (
        final_iterations,
        iteration_result,
    ) = select_classifier_iterations(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    print(
        "2023 Brier       :",
        f"{iteration_result['brier']:.10f}",
    )

    print(
        "2023 BSS raw     :",
        f"{iteration_result['bss_raw']:.2f}",
    )

    print(
        "Selected trees   :",
        final_iterations,
    )

    # ========================================================
    # Stage 2
    # H1 train -> H2 pure holdout
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Stage 2: 2024 H1 -> H2"
    )
    print("=" * 88)

    (
        artifact,
        pred_df,
        holdout_result,
    ) = train_h1_validate_h2(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
        final_iterations=final_iterations,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    pred_df.to_csv(
        VALIDATION_DIR
        / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            "CatBoostClassifier"
        ),

        "iteration_selection": (
            iteration_result
        ),

        "weight_holdout": (
            holdout_result
        ),

        "final_iterations": int(
            final_iterations
        ),

        "model_path": str(
            MODEL_PATH
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
            ensure_ascii=False,
            indent=2,
        )

    with open(
        MODEL_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR / "cb_v003.py",
        )
    except Exception:
        pass

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    split_state = (
        holdout_result[
            "split_state"
        ]
    )

    print()
    print("=" * 88)
    print(
        "cb_v003 Result"
    )
    print("=" * 88)

    print(
        "2024 months      :",
        split_state[
            "available_months"
        ],
    )

    print(
        "Cutoff month     :",
        split_state[
            "selected_cutoff_month"
        ],
    )

    print(
        "2024 H1 rows     :",
        f"{split_state['n_h1']:,}",
        f"({split_state['h1_share']:.4f})",
    )

    print(
        "2024 H2 rows     :",
        f"{split_state['n_h2']:,}",
        f"({split_state['h2_share']:.4f})",
    )

    print("-" * 88)

    print(
        "Final iterations :",
        final_iterations,
    )

    print(
        "H2 Brier         :",
        f"{holdout_result['brier']:.10f}",
    )

    print(
        "H2 BSS raw       :",
        f"{holdout_result['bss_raw']:.2f}",
    )

    print(
        "Features         :",
        holdout_result[
            "n_features"
        ],
    )

    print("-" * 88)

    print(
        "MODEL_PATH       :",
        MODEL_PATH,
    )

    print(
        "H2 prediction    :",
        VALIDATION_DIR
        / "oof_pred.csv",
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
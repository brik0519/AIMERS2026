# train.py: CatBoost Regressor cb_reg_v006
#
# cb_reg_v003의 HB-EB + CatBoostRegressor(RMSE) + Expanding Window 구조를 유지하고,
# target에 더 가까운 조건부 확률 신호를 추가한다.
#
# 핵심 비교:
#   Control   : cb_reg_v003 = 기존 HB-EB
#   Treatment : cb_reg_v006 = Control + Pitcher × Count-State Hierarchical Conditional EB
#
# 추가 Feature:
#   - hc_pitcher_count_n
#   - hc_pitcher_count_eb_rate
#
# Leakage 방지:
#   - Validation row는 fold train까지만 사용한 static lookup으로 변환
#   - Fold train row도 자기 season보다 과거 season만 사용하여 HC-EB 생성
#   - Final train도 각 row season보다 이전 season만 사용
#   - 2025 inference lookup은 공식 train 2019~2024만으로 고정
#
# 평가 데이터 독립 추론 규칙:
# - test.csv는 Feature schema 확인을 위해 header만 읽는다.
# - Feature/Prior는 공식 train.csv만으로 생성한다.
# - 다른 test 행 또는 test 전체 분포를 사용하지 않는다.
#
# Validation 저장:
# - fold별 pred.csv / validation.npz / metadata.json
# - pooled OOF pred.csv / validation.npz / metadata.json
# - Regressor raw prediction과 clipped prediction을 모두 저장
# - [0, 1] 범위 이탈 진단 저장

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v006"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"

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
        "train_seasons": [2019, 2020],
        "validation_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "validation_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "validation_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "validation_season": 2024,
    },
]


# =========================================================
# HB-EB 설정
# =========================================================

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


# =========================================================
# Hierarchical Conditional EB 설정
# =========================================================

HC_COUNT_STATE_COL = "__hc_count_state"

HC_FEATURES = [
    "hc_pitcher_count_n",
    "hc_pitcher_count_eb_rate",
]

# Empirical-Bayes strength 추정의 극단값 방지.
HC_MIN_PRIOR_STRENGTH = 5.0
HC_MAX_PRIOR_STRENGTH = 500.0
HC_MIN_GROUPS_FOR_STRENGTH = 20



# =========================================================
# CatBoost 설정
# =========================================================

# cb_v003와 capacity 관련 설정을 동일하게 유지하고
# objective만 Logloss -> RMSE로 변경한다.
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
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# HB-EB
# =========================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    train subset만으로 HB-EB prior를 추정한다.

    Validation/Test 정보를 prior 추정에 사용하지 않는다.
    """

    g_mu = float(
        df_train[rate_col].mean()
    )

    g_var = float(
        df_train[rate_col].var()
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
            max_bernoulli_var * 0.5
        )

    g_m = (
        max_bernoulli_var / g_var
    ) - 1.0

    g_alpha = g_mu * g_m
    g_beta = (
        1.0 - g_mu
    ) * g_m

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            ["mean", "var", "count"]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0 - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(stats["var"])
        & (stats["var"] > 0.0)
        & (stats["var"] < group_max_var)
        & (stats["count"] >= 30)
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
            mu * (1.0 - mu) / var
        ) - 1.0

        stats.loc[
            valid_mask,
            "alpha",
        ] = mu * m

        stats.loc[
            valid_mask,
            "beta",
        ] = (
            1.0 - mu
        ) * m

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
    """
    train-only prior를 lookup하여 각 행을 독립적으로 변환한다.
    """

    original_index = df.index

    lookup_df = (
        df[group_cols]
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup_df["alpha"]
        .fillna(global_prior[0])
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup_df["beta"]
        .fillna(global_prior[1])
        .to_numpy(dtype=np.float64)
    )

    n = (
        df[n_col]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        df[rate_col]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    success_count = n * rate

    denominator = (
        n + alpha + beta
    )

    eb_rate = (
        success_count + alpha
    ) / denominator

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def apply_hbeb_configs(
    df_train,
    df_apply,
    base_features,
):
    """
    df_train만으로 prior를 추정하고 df_apply에 HB-EB feature를 적용한다.

    반환:
    - 변환된 df_apply
    - 최종 feature 목록
    - 추론용 prior artifact
    """

    transformed = (
        df_apply.copy()
    )

    current_features = (
        list(base_features)
    )

    priors = {}

    for config in HBEB_CONFIGS:
        prior_df, global_prior = (
            compute_hbeb_priors(
                df_train=df_train,
                group_cols=config[
                    "group_cols"
                ],
                rate_col=config[
                    "rate_col"
                ],
            )
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=config[
                "group_cols"
            ],
            n_col=config["n_col"],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            config["rate_col"]
            in current_features
        ):
            current_features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in current_features
        ):
            current_features.append(
                config["new_col"]
            )

        priors[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config["group_cols"]
            ),
            "n_col": config["n_col"],
            "rate_col": (
                config["rate_col"]
            ),
        }

    return (
        transformed,
        current_features,
        priors,
    )



# =========================================================
# Hierarchical Conditional EB
# =========================================================

def add_count_state(df):
    """
    balls_before × strikes_before를 하나의 discrete state로 만든다.
    모델 Feature에는 직접 넣지 않고 HC lookup key로만 사용한다.
    """
    df = df.copy()

    if (
        "balls_before" not in df.columns
        or "strikes_before" not in df.columns
    ):
        raise ValueError(
            "HC-EB 생성에 balls_before / strikes_before가 필요합니다."
        )

    df[HC_COUNT_STATE_COL] = (
        df["balls_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
        + "_"
        + df["strikes_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    return df


def estimate_hc_global_prior(df_ref):
    """
    과거 reference의 pitcher별 target rate 분산으로
    global Beta prior strength를 empirical-Bayes 방식으로 추정한다.

    표본이 부족하거나 분산 추정이 불안정하면 보수적인 strength로 fallback한다.
    """
    if len(df_ref) == 0:
        return {
            "global_mean": 0.5,
            "prior_strength": 30.0,
            "alpha": 15.0,
            "beta": 15.0,
            "n_reference": 0,
            "n_pitchers": 0,
        }

    global_mean = float(
        df_ref[TARGET].mean()
    )

    pitcher_stats = (
        df_ref
        .groupby(
            "pitcher_id",
            dropna=False,
        )[TARGET]
        .agg(
            success="sum",
            n="count",
        )
        .reset_index()
    )

    pitcher_stats["rate"] = (
        pitcher_stats["success"]
        / pitcher_stats["n"]
    )

    valid = (
        pitcher_stats["n"] >= 20
    )

    rates = (
        pitcher_stats.loc[
            valid,
            "rate",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    ns = (
        pitcher_stats.loc[
            valid,
            "n",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    fallback_strength = 30.0

    if (
        len(rates)
        < HC_MIN_GROUPS_FOR_STRENGTH
    ):
        strength = fallback_strength
    else:
        weights = ns / ns.sum()

        observed_var = float(
            np.sum(
                weights
                * (
                    rates - global_mean
                ) ** 2
            )
        )

        binomial_noise = float(
            np.sum(
                weights
                * (
                    global_mean
                    * (
                        1.0 - global_mean
                    )
                    / np.maximum(
                        ns,
                        1.0,
                    )
                )
            )
        )

        latent_var = max(
            observed_var - binomial_noise,
            1e-8,
        )

        raw_strength = (
            global_mean
            * (
                1.0 - global_mean
            )
            / latent_var
        ) - 1.0

        if not np.isfinite(
            raw_strength
        ):
            strength = fallback_strength
        else:
            strength = float(
                np.clip(
                    raw_strength,
                    HC_MIN_PRIOR_STRENGTH,
                    HC_MAX_PRIOR_STRENGTH,
                )
            )

    alpha = (
        global_mean
        * strength
    )

    beta = (
        (
            1.0 - global_mean
        )
        * strength
    )

    return {
        "global_mean": float(
            global_mean
        ),
        "prior_strength": float(
            strength
        ),
        "alpha": float(alpha),
        "beta": float(beta),
        "n_reference": int(
            len(df_ref)
        ),
        "n_pitchers": int(
            len(pitcher_stats)
        ),
    }


def build_hc_pitcher_count_lookup(df_ref):
    """
    Global -> Pitcher -> Pitcher × Count-State hierarchy를 만든다.

    child posterior:
        (S_pc + strength * parent_rate) / (N_pc + strength)

    parent_rate:
        pitcher target count를 global Beta prior로 shrink한 posterior.
    """
    ref = add_count_state(
        df_ref
    )

    prior = (
        estimate_hc_global_prior(
            ref
        )
    )

    if len(ref) == 0:
        return {
            "prior": prior,
            "pitcher_lookup": pd.DataFrame(
                columns=[
                    "pitcher_id",
                    "hc_pitcher_parent_rate",
                    "hc_pitcher_parent_n",
                ]
            ),
            "child_lookup": pd.DataFrame(
                columns=[
                    "pitcher_id",
                    HC_COUNT_STATE_COL,
                    "hc_pitcher_count_n",
                    "hc_pitcher_count_eb_rate",
                ]
            ),
        }

    pitcher = (
        ref
        .groupby(
            "pitcher_id",
            dropna=False,
        )[TARGET]
        .agg(
            hc_pitcher_parent_success="sum",
            hc_pitcher_parent_n="count",
        )
        .reset_index()
    )

    pitcher[
        "hc_pitcher_parent_rate"
    ] = (
        pitcher[
            "hc_pitcher_parent_success"
        ]
        + prior["alpha"]
    ) / (
        pitcher[
            "hc_pitcher_parent_n"
        ]
        + prior["alpha"]
        + prior["beta"]
    )

    child = (
        ref
        .groupby(
            [
                "pitcher_id",
                HC_COUNT_STATE_COL,
            ],
            dropna=False,
        )[TARGET]
        .agg(
            hc_pitcher_count_success="sum",
            hc_pitcher_count_n="count",
        )
        .reset_index()
    )

    child = child.merge(
        pitcher[
            [
                "pitcher_id",
                "hc_pitcher_parent_rate",
            ]
        ],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )

    strength = float(
        prior["prior_strength"]
    )

    child[
        "hc_pitcher_count_eb_rate"
    ] = (
        child[
            "hc_pitcher_count_success"
        ]
        + (
            strength
            * child[
                "hc_pitcher_parent_rate"
            ]
        )
    ) / (
        child[
            "hc_pitcher_count_n"
        ]
        + strength
    )

    pitcher_lookup = (
        pitcher[
            [
                "pitcher_id",
                "hc_pitcher_parent_rate",
                "hc_pitcher_parent_n",
            ]
        ]
        .copy()
    )

    child_lookup = (
        child[
            [
                "pitcher_id",
                HC_COUNT_STATE_COL,
                "hc_pitcher_count_n",
                "hc_pitcher_count_eb_rate",
            ]
        ]
        .copy()
    )

    if pitcher_lookup[
        "pitcher_id"
    ].duplicated().any():
        raise ValueError(
            "HC pitcher lookup key 중복"
        )

    if child_lookup[
        [
            "pitcher_id",
            HC_COUNT_STATE_COL,
        ]
    ].duplicated().any():
        raise ValueError(
            "HC pitcher-count lookup key 중복"
        )

    return {
        "prior": prior,
        "pitcher_lookup": (
            pitcher_lookup
        ),
        "child_lookup": (
            child_lookup
        ),
    }


def apply_hc_pitcher_count_lookup(
    df,
    lookup,
    fallback_col="hbeb_pitcher_success_rate",
):
    """
    static train-only HC lookup을 각 행에 독립 적용한다.

    fallback:
    exact pitcher×count
      -> pitcher parent
      -> row의 기존 safe HB-EB pitcher rate
      -> global prior
    """
    out = add_count_state(
        df
    )

    child = (
        lookup[
            "child_lookup"
        ]
    )

    pitcher = (
        lookup[
            "pitcher_lookup"
        ]
    )

    out = out.merge(
        child,
        on=[
            "pitcher_id",
            HC_COUNT_STATE_COL,
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    out = out.merge(
        pitcher,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    exact_hit = (
        out[
            "hc_pitcher_count_eb_rate"
        ].notna()
    )

    parent_hit = (
        (~exact_hit)
        & out[
            "hc_pitcher_parent_rate"
        ].notna()
    )

    fallback = (
        out[
            fallback_col
        ]
        if fallback_col in out.columns
        else pd.Series(
            np.nan,
            index=out.index,
        )
    )

    rate = (
        out[
            "hc_pitcher_count_eb_rate"
        ]
        .fillna(
            out[
                "hc_pitcher_parent_rate"
            ]
        )
        .fillna(
            fallback
        )
        .fillna(
            lookup[
                "prior"
            ][
                "global_mean"
            ]
        )
        .astype(
            np.float64
        )
    )

    n = (
        out[
            "hc_pitcher_count_n"
        ]
        .fillna(0)
        .astype(
            np.float64
        )
    )

    out[
        "hc_pitcher_count_eb_rate"
    ] = rate

    out[
        "hc_pitcher_count_n"
    ] = n

    diagnostics = {
        "n_rows": int(
            len(out)
        ),
        "exact_hit_count": int(
            exact_hit.sum()
        ),
        "exact_hit_ratio": float(
            exact_hit.mean()
        ),
        "parent_fallback_count": int(
            parent_hit.sum()
        ),
        "parent_fallback_ratio": float(
            parent_hit.mean()
        ),
        "safe_row_fallback_count": int(
            (
                (~exact_hit)
                & (~parent_hit)
            ).sum()
        ),
        "safe_row_fallback_ratio": float(
            (
                (~exact_hit)
                & (~parent_hit)
            ).mean()
        ),
        "mean_support": float(
            n.mean()
        ),
        "median_support": float(
            np.median(n)
        ),
        "prior_strength": float(
            lookup[
                "prior"
            ][
                "prior_strength"
            ]
        ),
        "global_mean": float(
            lookup[
                "prior"
            ][
                "global_mean"
            ]
        ),
    }

    out = out.drop(
        columns=[
            HC_COUNT_STATE_COL,
            "hc_pitcher_parent_rate",
            "hc_pitcher_parent_n",
        ],
        errors="ignore",
    )

    return (
        out,
        diagnostics,
    )


def apply_hc_to_training_season_frozen(
    df_train,
):
    """
    Training row의 HC feature는 반드시 자기 season보다 과거 season만 참조한다.

    2019 row처럼 과거 season이 없는 경우:
    - exact support = 0
    - rate는 기존 safe HB-EB pitcher feature로 fallback
    """
    frames = []
    diagnostics = {}

    seasons = sorted(
        int(x)
        for x in df_train[
            "season"
        ].unique()
    )

    for season in seasons:
        current_mask = (
            df_train["season"]
            == season
        )

        ref = (
            df_train.loc[
                df_train[
                    "season"
                ]
                < season
            ]
            .copy()
        )

        current = (
            df_train.loc[
                current_mask
            ]
            .copy()
        )

        lookup = (
            build_hc_pitcher_count_lookup(
                ref
            )
        )

        (
            transformed,
            diag,
        ) = apply_hc_pitcher_count_lookup(
            current,
            lookup,
        )

        diagnostics[
            str(season)
        ] = {
            **diag,
            "reference_seasons": sorted(
                int(x)
                for x in ref[
                    "season"
                ].unique()
            ),
            "n_reference": int(
                len(ref)
            ),
        }

        frames.append(
            transformed
        )

    result = pd.concat(
        frames,
        axis=0,
    ).sort_index()

    if not result.index.equals(
        df_train.index
    ):
        # merge가 RangeIndex를 재구성할 수 있어 row_id로 최종 정렬 검증.
        if (
            ID in result.columns
            and ID in df_train.columns
        ):
            result = (
                result
                .set_index(ID)
                .loc[
                    df_train[ID]
                ]
                .reset_index()
            )
            result.index = (
                df_train.index
            )
        else:
            raise ValueError(
                "HC season-frozen training row 정렬 복원 실패"
            )

    return (
        result,
        diagnostics,
    )


def add_hc_features_to_feature_list(
    features,
):
    out = list(features)

    for feature in HC_FEATURES:
        if feature not in out:
            out.append(
                feature
            )

    return out


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    """
    cb_v003와 동일한 범주형 전처리.
    """

    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Metric
# =========================================================

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

    positive_rate = float(
        np.mean(y_true)
    )

    brier = float(
        np.mean(
            (
                y_pred - y_true
            ) ** 2
        )
    )

    baseline_brier = float(
        positive_rate
        * (
            1.0 - positive_rate
        )
    )

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0
            * (
                1.0
                - brier
                / baseline_brier
            )
        )

        bss = max(
            0.0,
            bss_raw,
        )

    return {
        "positive_rate": (
            positive_rate
        ),
        "brier": brier,
        "baseline_brier": (
            baseline_brier
        ),
        "bss_raw": bss_raw,
        "bss": bss,
    }


def calculate_range_diagnostics(
    y_pred_raw,
):
    y_pred_raw = np.asarray(
        y_pred_raw,
        dtype=np.float64,
    )

    below_zero = (
        y_pred_raw < 0.0
    )

    above_one = (
        y_pred_raw > 1.0
    )

    outside = (
        below_zero
        | above_one
    )

    return {
        "n_prediction_below_0": int(
            np.sum(below_zero)
        ),
        "ratio_prediction_below_0": float(
            np.mean(below_zero)
        ),
        "n_prediction_above_1": int(
            np.sum(above_one)
        ),
        "ratio_prediction_above_1": float(
            np.mean(above_one)
        ),
        "n_prediction_outside_0_1": int(
            np.sum(outside)
        ),
        "ratio_prediction_outside_0_1": float(
            np.mean(outside)
        ),
    }


# =========================================================
# Validation 저장
# =========================================================

def save_fold_validation(
    fold,
    train_seasons,
    validation_season,
    estimator,
    row_ids,
    y_true,
    y_pred_raw,
    y_pred_clip,
    raw_metrics,
    clipped_metrics,
    elapsed_seconds,
    best_iteration,
    final_iterations,
    n_train,
    hc_train_diagnostics=None,
    hc_validation_diagnostics=None,
):
    fold_dir = (
        VALIDATION_DIR
        / f"fold_{validation_season}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    row_ids = np.asarray(
        row_ids
    )

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred_raw = np.asarray(
        y_pred_raw,
        dtype=np.float64,
    )

    y_pred_clip = np.asarray(
        y_pred_clip,
        dtype=np.float64,
    )

    if not (
        len(row_ids)
        == len(y_true)
        == len(y_pred_raw)
        == len(y_pred_clip)
    ):
        raise ValueError(
            "Validation 저장 배열 길이가 일치하지 않습니다."
        )

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction_raw": (
            y_pred_raw
        ),
        "prediction_clip": (
            y_pred_clip
        ),
        # 기존 OOF/ensemble 분석 코드와 호환
        "prediction": (
            y_pred_clip
        ),
        "validation_season": (
            validation_season
        ),
    })

    pred_df.to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        fold_dir
        / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction_raw=(
            y_pred_raw
        ),
        prediction_clip=(
            y_pred_clip
        ),
        prediction=(
            y_pred_clip
        ),
        validation_season=(
            np.full(
                len(y_true),
                validation_season,
            )
        ),
    )

    range_diag = (
        calculate_range_diagnostics(
            y_pred_raw
        )
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "fold": int(fold),
        "train_seasons": [
            int(s)
            for s in train_seasons
        ],
        "validation_season": int(
            validation_season
        ),
        "n_train": int(n_train),
        "n_validation": int(
            len(y_true)
        ),
        "model_type": (
            type(estimator).__name__
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "positive_rate": float(
            clipped_metrics[
                "positive_rate"
            ]
        ),

        "raw_brier": float(
            raw_metrics["brier"]
        ),
        "raw_bss_raw": float(
            raw_metrics["bss_raw"]
        ),
        "raw_bss": float(
            raw_metrics["bss"]
        ),

        "clipped_brier": float(
            clipped_metrics[
                "brier"
            ]
        ),
        "clipped_bss_raw": float(
            clipped_metrics[
                "bss_raw"
            ]
        ),
        "clipped_bss": float(
            clipped_metrics[
                "bss"
            ]
        ),

        # 기존 분석 코드 호환용
        "brier": float(
            clipped_metrics[
                "brier"
            ]
        ),
        "baseline_brier": float(
            clipped_metrics[
                "baseline_brier"
            ]
        ),
        "bss_raw": float(
            clipped_metrics[
                "bss_raw"
            ]
        ),
        "bss": float(
            clipped_metrics[
                "bss"
            ]
        ),

        "prediction_raw_mean": float(
            np.mean(y_pred_raw)
        ),
        "prediction_raw_std": float(
            np.std(y_pred_raw)
        ),
        "prediction_raw_min": float(
            np.min(y_pred_raw)
        ),
        "prediction_raw_max": float(
            np.max(y_pred_raw)
        ),

        "prediction_clip_mean": float(
            np.mean(y_pred_clip)
        ),
        "prediction_clip_std": float(
            np.std(y_pred_clip)
        ),
        "prediction_clip_min": float(
            np.min(y_pred_clip)
        ),
        "prediction_clip_max": float(
            np.max(y_pred_clip)
        ),

        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "best_iteration": int(
            best_iteration
        ),
        "final_iterations": int(
            final_iterations
        ),
        **range_diag,
        "hc_features": HC_FEATURES,
        "hc_train_diagnostics": (
            hc_train_diagnostics
        ),
        "hc_validation_diagnostics": (
            hc_validation_diagnostics
        ),
    }

    with open(
        fold_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return (
        pred_df,
        metadata,
    )


def save_validation_summary(
    fold_results,
    oof,
):
    oof.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR
        / "oof_validation.npz",
        row_id=oof[
            "row_id"
        ].to_numpy(),
        target=oof[
            "target"
        ].to_numpy(),
        prediction_raw=oof[
            "prediction_raw"
        ].to_numpy(),
        prediction_clip=oof[
            "prediction_clip"
        ].to_numpy(),
        prediction=oof[
            "prediction"
        ].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    raw_oof_metrics = (
        calculate_metrics(
            oof["target"],
            oof[
                "prediction_raw"
            ],
        )
    )

    clipped_oof_metrics = (
        calculate_metrics(
            oof["target"],
            oof[
                "prediction_clip"
            ],
        )
    )

    fold_briers = np.asarray(
        [
            x["brier"]
            for x in fold_results
        ],
        dtype=np.float64,
    )

    fold_bss = np.asarray(
        [
            x["bss"]
            for x in fold_results
        ],
        dtype=np.float64,
    )

    fold_sizes = np.asarray(
        [
            x["n_validation"]
            for x in fold_results
        ],
        dtype=np.float64,
    )

    recent_briers = [
        x["brier"]
        for x in fold_results
        if x["validation_season"]
        in (2023, 2024)
    ]

    range_diag = (
        calculate_range_diagnostics(
            oof[
                "prediction_raw"
            ]
        )
    )

    summary = {
        "n_oof": int(len(oof)),
        "mean_fold_brier": float(
            np.mean(fold_briers)
        ),
        "std_fold_brier": float(
            np.std(fold_briers)
        ),
        "weighted_fold_brier": float(
            np.average(
                fold_briers,
                weights=fold_sizes,
            )
        ),
        "best_fold_brier": float(
            np.min(fold_briers)
        ),
        "worst_fold_brier": float(
            np.max(fold_briers)
        ),
        "fold_brier_range": float(
            np.max(fold_briers)
            - np.min(fold_briers)
        ),
        "recent_2023_2024_mean_brier": float(
            np.mean(
                recent_briers
            )
        ),
        "mean_fold_bss": float(
            np.mean(fold_bss)
        ),
        "weighted_fold_bss": float(
            np.average(
                fold_bss,
                weights=fold_sizes,
            )
        ),

        "pooled_oof_raw_brier": float(
            raw_oof_metrics[
                "brier"
            ]
        ),
        "pooled_oof_raw_bss_raw": float(
            raw_oof_metrics[
                "bss_raw"
            ]
        ),
        "pooled_oof_raw_bss": float(
            raw_oof_metrics[
                "bss"
            ]
        ),

        "pooled_oof_clipped_brier": float(
            clipped_oof_metrics[
                "brier"
            ]
        ),
        "pooled_oof_clipped_baseline_brier": float(
            clipped_oof_metrics[
                "baseline_brier"
            ]
        ),
        "pooled_oof_clipped_bss_raw": float(
            clipped_oof_metrics[
                "bss_raw"
            ]
        ),
        "pooled_oof_clipped_bss": float(
            clipped_oof_metrics[
                "bss"
            ]
        ),

        # 기존 분석 코드 호환
        "pooled_oof_brier": float(
            clipped_oof_metrics[
                "brier"
            ]
        ),
        "pooled_oof_baseline_brier": float(
            clipped_oof_metrics[
                "baseline_brier"
            ]
        ),
        "pooled_oof_bss_raw": float(
            clipped_oof_metrics[
                "bss_raw"
            ]
        ),
        "pooled_oof_bss": float(
            clipped_oof_metrics[
                "bss"
            ]
        ),
        **range_diag,
    }

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "validation_method": (
            "expanding_window"
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "prediction_for_evaluation": (
            "clipped"
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "hierarchical_conditional_eb": {
            "enabled": True,
            "hierarchy": [
                "global",
                "pitcher",
                "pitcher_x_count_state",
            ],
            "features": HC_FEATURES,
            "count_state_source": [
                "balls_before",
                "strikes_before",
            ],
            "training_policy": (
                "row_season_frozen_using_only_prior_seasons"
            ),
            "validation_policy": (
                "fold_train_static_lookup"
            ),
        },
        "folds": fold_results,
        "summary": summary,
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

    return summary


# =========================================================
# Expanding Window Validation
# =========================================================

def run_expanding_validation(
    train,
    base_features,
    cat_cols,
):
    fold_results = []
    oof_frames = []
    final_features = None

    for fold_config in (
        EXPANDING_FOLDS
    ):
        fold = (
            fold_config["fold"]
        )

        train_seasons = (
            fold_config[
                "train_seasons"
            ]
        )

        validation_season = (
            fold_config[
                "validation_season"
            ]
        )

        train_mask = (
            train["season"].isin(
                train_seasons
            )
        )

        val_mask = (
            train["season"]
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

        # ---------------------------------------------
        # Fold train-only HB-EB prior
        # ---------------------------------------------

        current_features = (
            list(base_features)
        )

        for config in (
            HBEB_CONFIGS
        ):
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

            df_train[
                config["new_col"]
            ] = apply_hbeb(
                df=df_train,
                group_cols=config[
                    "group_cols"
                ],
                n_col=config["n_col"],
                rate_col=config[
                    "rate_col"
                ],
                prior_df=prior_df,
                global_prior=(
                    global_prior
                ),
            )

            df_val[
                config["new_col"]
            ] = apply_hbeb(
                df=df_val,
                group_cols=config[
                    "group_cols"
                ],
                n_col=config["n_col"],
                rate_col=config[
                    "rate_col"
                ],
                prior_df=prior_df,
                global_prior=(
                    global_prior
                ),
            )

            if (
                config["rate_col"]
                in current_features
            ):
                current_features.remove(
                    config["rate_col"]
                )

            if (
                config["new_col"]
                not in current_features
            ):
                current_features.append(
                    config["new_col"]
                )

        # ---------------------------------------------
        # Leakage-safe Hierarchical Conditional EB
        # ---------------------------------------------

        (
            df_train,
            hc_train_diagnostics,
        ) = (
            apply_hc_to_training_season_frozen(
                df_train
            )
        )

        validation_lookup = (
            build_hc_pitcher_count_lookup(
                df_train.loc[
                    :,
                    [
                        c
                        for c in df_train.columns
                        if c not in HC_FEATURES
                    ]
                ]
            )
        )

        (
            df_val,
            hc_validation_diagnostics,
        ) = (
            apply_hc_pitcher_count_lookup(
                df_val,
                validation_lookup,
            )
        )

        current_features = (
            add_hc_features_to_feature_list(
                current_features
            )
        )

        final_features = (
            current_features
        )

        X_train = (
            prepare_features(
                df_train[
                    current_features
                ],
                cat_cols,
            )
        )

        y_train = (
            df_train[TARGET]
            .astype(np.float64)
        )

        X_val = (
            prepare_features(
                df_val[
                    current_features
                ],
                cat_cols,
            )
        )

        y_val = (
            df_val[TARGET]
            .astype(np.float64)
        )

        val_row_id = (
            df_val[ID]
        )

        print()
        print("=" * 72)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print("=" * 72)
        print(
            "학습 데이터:",
            X_train.shape,
        )
        print(
            "검증 데이터:",
            X_val.shape,
        )
        print(
            "Training objective    : RMSE"
        )
        print(
            "Evaluation prediction : raw + clipped"
        )

        estimator = (
            CatBoostRegressor(
                **MODEL_PARAMS
            )
        )

        start_time = time.time()

        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(
                X_val,
                y_val,
            ),
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            use_best_model=True,
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        val_pred_raw = (
            estimator.predict(
                X_val
            )
        )

        val_pred_clip = np.clip(
            val_pred_raw,
            0.0,
            1.0,
        )

        raw_metrics = (
            calculate_metrics(
                y_val,
                val_pred_raw,
            )
        )

        clipped_metrics = (
            calculate_metrics(
                y_val,
                val_pred_clip,
            )
        )

        # ---------------------------------------------
        # Best iteration
        # ---------------------------------------------

        best_iteration = (
            estimator
            .get_best_iteration()
        )

        if (
            best_iteration
            is not None
            and best_iteration >= 0
        ):
            final_iterations = (
                best_iteration + 1
            )
        else:
            final_iterations = (
                MODEL_PARAMS[
                    "iterations"
                ]
            )

        pred_df, metadata = (
            save_fold_validation(
                fold=fold,
                train_seasons=(
                    train_seasons
                ),
                validation_season=(
                    validation_season
                ),
                estimator=estimator,
                row_ids=val_row_id,
                y_true=y_val,
                y_pred_raw=(
                    val_pred_raw
                ),
                y_pred_clip=(
                    val_pred_clip
                ),
                raw_metrics=(
                    raw_metrics
                ),
                clipped_metrics=(
                    clipped_metrics
                ),
                elapsed_seconds=(
                    elapsed
                ),
                best_iteration=(
                    best_iteration
                ),
                final_iterations=(
                    final_iterations
                ),
                n_train=len(y_train),
                hc_train_diagnostics=(
                    hc_train_diagnostics
                ),
                hc_validation_diagnostics=(
                    hc_validation_diagnostics
                ),
            )
        )

        fold_results.append(
            metadata
        )

        oof_frames.append(
            pred_df
        )

        range_diag = (
            calculate_range_diagnostics(
                val_pred_raw
            )
        )

        print()
        print("[Raw prediction]")
        print(
            f"Brier Score      : "
            f"{raw_metrics['brier']:.8f}"
        )
        print(
            f"BSS              : "
            f"{raw_metrics['bss']:.2f}"
        )

        print()
        print("[Clipped prediction]")
        print(
            f"Brier Score      : "
            f"{clipped_metrics['brier']:.8f}"
        )
        print(
            f"BSS              : "
            f"{clipped_metrics['bss']:.2f}"
        )

        print()
        print("[Range diagnostics]")
        print(
            f"Below 0          : "
            f"{range_diag['n_prediction_below_0']} "
            f"("
            f"{range_diag['ratio_prediction_below_0']:.6%}"
            f")"
        )
        print(
            f"Above 1          : "
            f"{range_diag['n_prediction_above_1']} "
            f"("
            f"{range_diag['ratio_prediction_above_1']:.6%}"
            f")"
        )
        print(
            f"Outside [0,1]    : "
            f"{range_diag['n_prediction_outside_0_1']} "
            f"("
            f"{range_diag['ratio_prediction_outside_0_1']:.6%}"
            f")"
        )

        print()
        print(
            f"Best iteration   : "
            f"{best_iteration}"
        )
        print(
            f"Final iterations : "
            f"{final_iterations}"
        )
        print(
            f"Elapsed          : "
            f"{elapsed:.1f}s"
        )
        print(
            "HC exact hit      : "
            f"{hc_validation_diagnostics['exact_hit_ratio']:.2%}"
        )
        print(
            "HC parent fallback: "
            f"{hc_validation_diagnostics['parent_fallback_ratio']:.2%}"
        )
        print(
            "HC row fallback   : "
            f"{hc_validation_diagnostics['safe_row_fallback_ratio']:.2%}"
        )
        print(
            "HC prior strength : "
            f"{hc_validation_diagnostics['prior_strength']:.4f}"
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    return (
        fold_results,
        oof,
        final_features,
    )


# =========================================================
# Final iteration 선택
# =========================================================

def select_final_iterations(
    fold_results,
):
    if (
        FINAL_ITERATION_STRATEGY
        == "latest_fold"
    ):
        latest_fold = max(
            fold_results,
            key=lambda x: x[
                "validation_season"
            ],
        )

        return int(
            latest_fold[
                "final_iterations"
            ]
        )

    raise ValueError(
        "Unsupported "
        "FINAL_ITERATION_STRATEGY: "
        f"{FINAL_ITERATION_STRATEGY}"
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )

    # -----------------------------------------------------
    # Feature schema
    # -----------------------------------------------------

    # test.csv의 다른 행이나 분포는 사용하지 않는다.
    # header만 읽어 train/test 공통 input schema를 확정한다.
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        c
        for c in test_cols
        if c != ID
    ]

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + base_features
            + [TARGET]
        ),
    )

    print("=" * 72)
    print(
        f"Experiment           : "
        f"{EXPERIMENT_NAME}"
    )
    print(
        "Model                : "
        "CatBoostRegressor"
    )
    print(
        "Objective            : RMSE"
    )
    print(
        "Validation           : "
        "Expanding Window"
    )
    print(
        "Feature Engineering  : "
        "HB-EB + Pitcher×Count HC-EB"
    )
    print(
        f"Train shape          : "
        f"{train.shape}"
    )
    print(
        f"Base Feature 수      : "
        f"{len(base_features)}"
    )
    print(
        f"Categorical Feature  : "
        f"{len(cat_cols)}"
    )
    print(
        "시즌                 : "
        f"{train['season'].min()} "
        "~ "
        f"{train['season'].max()}"
    )
    print(
        f"Target mean          : "
        f"{train[TARGET].mean():.6f}"
    )
    print("=" * 72)

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Expanding Window Validation
    # -----------------------------------------------------

    (
        fold_results,
        oof,
        final_features,
    ) = run_expanding_validation(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    summary = (
        save_validation_summary(
            fold_results=fold_results,
            oof=oof,
        )
    )

    print()
    print("=" * 72)
    print(
        "Expanding Window Summary"
    )
    print("=" * 72)

    for fold in fold_results:
        print(
            f"Valid "
            f"{fold['validation_season']} | "
            f"Brier="
            f"{fold['brier']:.8f} | "
            f"BSS="
            f"{fold['bss']:.2f} | "
            f"iter="
            f"{fold['final_iterations']} | "
            f"outside="
            f"{fold['ratio_prediction_outside_0_1']:.6%}"
        )

    print()
    print(
        f"Pooled OOF Brier     : "
        f"{summary['pooled_oof_brier']:.8f}"
    )
    print(
        f"Pooled OOF BSS       : "
        f"{summary['pooled_oof_bss']:.2f}"
    )
    print(
        f"Recent 23~24 Brier   : "
        f"{summary['recent_2023_2024_mean_brier']:.8f}"
    )
    print(
        f"OOF outside [0,1]    : "
        f"{summary['ratio_prediction_outside_0_1']:.6%}"
    )
    print(
        f"Validation 저장      : "
        f"{VALIDATION_DIR}"
    )
    print("=" * 72)

    final_iterations = (
        select_final_iterations(
            fold_results
        )
    )

    print()
    print(
        "Final iteration strategy : "
        f"{FINAL_ITERATION_STRATEGY}"
    )
    print(
        "Final iterations         : "
        f"{final_iterations}"
    )

    # -----------------------------------------------------
    # 전체 train으로 최종 HB-EB prior 생성
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "최종 HB-EB prior 및 "
        "CatBoostRegressor 학습"
    )
    print("=" * 72)

    df_full = (
        train.copy()
    )

    final_priors_dict = {}
    final_features_check = (
        list(base_features)
    )

    for config in (
        HBEB_CONFIGS
    ):
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_full,
            group_cols=config[
                "group_cols"
            ],
            rate_col=config[
                "rate_col"
            ],
        )

        df_full[
            config["new_col"]
        ] = apply_hbeb(
            df=df_full,
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
            global_prior=(
                global_prior
            ),
        )

        if (
            config["rate_col"]
            in final_features_check
        ):
            final_features_check.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in final_features_check
        ):
            final_features_check.append(
                config["new_col"]
            )

        final_priors_dict[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config["group_cols"]
            ),
            "n_col": config[
                "n_col"
            ],
            "rate_col": config[
                "rate_col"
            ],
        }

    # -----------------------------------------------------
    # Final train row도 자기 season 이전 target만 사용한 HC-EB
    # -----------------------------------------------------

    (
        df_full,
        final_hc_training_diagnostics,
    ) = (
        apply_hc_to_training_season_frozen(
            df_full
        )
    )

    final_features_check = (
        add_hc_features_to_feature_list(
            final_features_check
        )
    )

    if (
        final_features_check
        != final_features
    ):
        raise ValueError(
            "Validation과 Final Training의 "
            "Feature 구성이 일치하지 않습니다."
        )

    # 2025 inference용 static HC lookup은 2019~2024 전체 공식 train으로 고정.
    final_hc_lookup = (
        build_hc_pitcher_count_lookup(
            train.copy()
        )
    )

    X_full = prepare_features(
        df_full[
            final_features
        ],
        cat_cols,
    )

    y_full = (
        df_full[TARGET]
        .astype(np.float64)
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    final_model = (
        CatBoostRegressor(
            **final_params
        )
    )

    start_time = time.time()

    final_model.fit(
        X_full,
        y_full,
        cat_features=cat_cols,
    )

    final_elapsed = (
        time.time()
        - start_time
    )

    # -----------------------------------------------------
    # 단일 Artifact 저장
    # -----------------------------------------------------

    final_artifact = {
        "model": final_model,
        "hbeb_priors": (
            final_priors_dict
        ),
        "features": (
            final_features
        ),
        "cat_cols": cat_cols,
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "final_iterations": int(
            final_iterations
        ),
        "prediction_policy": (
            "np.clip(raw_prediction, 0.0, 1.0)"
        ),
        "hc_features": (
            HC_FEATURES
        ),
        "hc_pitcher_count_lookup": (
            final_hc_lookup
        ),
        "hc_policy": (
            "global_to_pitcher_to_pitcher_count_static_train_only_lookup"
        ),
        "hc_count_state_source": [
            "balls_before",
            "strikes_before",
        ],
    }

    joblib.dump(
        final_artifact,
        MODEL_PATH,
        compress=3,
    )

    final_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "validation_method": (
            "expanding_window"
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "final_iterations": int(
            final_iterations
        ),
        "training_seasons": sorted(
            [
                int(x)
                for x in train[
                    "season"
                ].unique()
            ]
        ),
        "n_train": int(
            len(train)
        ),
        "base_features": (
            base_features
        ),
        "features": (
            final_features
        ),
        "categorical_features": (
            cat_cols
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "hierarchical_conditional_eb": {
            "enabled": True,
            "features": HC_FEATURES,
            "hierarchy": [
                "global",
                "pitcher",
                "pitcher_x_count_state",
            ],
            "training_policy": (
                "row_season_frozen_using_only_prior_seasons"
            ),
            "inference_policy": (
                "static_lookup_from_official_train_2019_2024"
            ),
            "final_training_diagnostics": (
                final_hc_training_diagnostics
            ),
            "final_lookup_prior": (
                final_hc_lookup[
                    "prior"
                ]
            ),
            "final_child_lookup_rows": int(
                len(
                    final_hc_lookup[
                        "child_lookup"
                    ]
                )
            ),
        },
        "model_params": (
            final_params
        ),
        "elapsed_seconds": float(
            final_elapsed
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "prediction_policy": (
            "clip_to_0_1"
        ),
    }

    with open(
        MODEL_DIR
        / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    source_path = (
        Path(__file__)
        .resolve()
    )

    shutil.copy2(
        source_path,
        MODEL_DIR
        / "train.py",
    )

    print()
    print("=" * 72)
    print(
        "최종 Artifact 저장 완료"
    )
    print(
        f"- Model   : "
        f"{MODEL_PATH}"
    )
    print(
        f"- Metadata: "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        f"- Train.py: "
        f"{MODEL_DIR / 'train.py'}"
    )
    print(
        f"- elapsed : "
        f"{final_elapsed:.1f}s"
    )
    print("=" * 72)

    print()
    print("[추론 시 필수]")
    print(
        "CatBoostRegressor는 predict()를 사용하고, "
        "최종 제출 확률은 "
        "np.clip(pred_raw, 0.0, 1.0)을 적용합니다."
    )


if __name__ == "__main__":
    main()

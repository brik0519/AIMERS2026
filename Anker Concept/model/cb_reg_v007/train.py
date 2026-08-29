# train.py: CatBoost Regressor cb_reg_v007 — 2023 R-only Early Stopping
#
# cb_reg_v003의 HB-EB Feature Engineering + Expanding Window Validation 구조를 그대로 유지하면서
# 2023 validation fold에서만 game_type != "F"를 early stopping eval set으로 사용한다.
#
# 핵심 비교:
#   Control   : cb_v003     = HB-EB + CatBoostClassifier + Logloss
#   Treatment : cb_reg_v003 = HB-EB + CatBoostRegressor  + RMSE
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

EXPERIMENT_NAME = "cb_reg_v007"

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
# Early Stopping 정책
# =========================================================
#
# Experiment B:
# - 2023 validation fold에서만 game_type != "F" 행(R/non-F)을
#   early stopping용 eval_set으로 사용한다.
# - 학습 데이터는 기존과 동일하게 R/F 전체를 사용한다.
# - 2023 fold의 최종 성능 평가는 R/F 전체 validation에서 수행한다.
# - 2021, 2022, 2024 fold는 기존처럼 전체 validation을 eval_set으로 사용한다.
#
# 목적:
# 2023 F domain의 급격한 level shift가 전체 Brier를 오염시켜
# CatBoost가 R에서 학습 가능한 signal까지 조기에 포기하는지 진단한다.
#
# 주의:
# 이 정책 자체는 최종 deployment objective가 아니다.
# 최종 채택 여부는 R-only 성능이 아니라 전체 R+F 성능과
# 2024 temporal generalization을 함께 보고 판단한다.

R_ONLY_EARLY_STOPPING_SEASON = 2023
R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE = "F"


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


def calculate_domain_metrics(
    game_type,
    y_true,
    y_pred,
):
    """
    Validation 전체 예측을 R(non-F) / F로 분리해 진단한다.

    - R은 game_type != "F"로 정의한다.
    - 이 함수는 평가/진단용이며 학습 Feature를 생성하지 않는다.
    """

    game_type = pd.Series(
        game_type
    ).astype(str).to_numpy()

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if not (
        len(game_type)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "Domain metric 입력 길이가 일치하지 않습니다."
        )

    masks = {
        "R": (
            game_type
            != R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE
        ),
        "F": (
            game_type
            == R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE
        ),
    }

    result = {}

    for domain, mask in masks.items():
        n = int(
            np.sum(mask)
        )

        if n == 0:
            result[domain] = {
                "n": 0,
                "share": 0.0,
                "metrics": None,
                "prediction_mean": None,
                "prediction_std": None,
            }
            continue

        metrics = calculate_metrics(
            y_true[mask],
            y_pred[mask],
        )

        result[domain] = {
            "n": n,
            "share": float(
                n / len(y_true)
            ),
            "metrics": metrics,
            "prediction_mean": float(
                np.mean(
                    y_pred[mask]
                )
            ),
            "prediction_std": float(
                np.std(
                    y_pred[mask]
                )
            ),
        }

    return result


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
    game_type,
    y_true,
    y_pred_raw,
    y_pred_clip,
    raw_metrics,
    clipped_metrics,
    elapsed_seconds,
    best_iteration,
    final_iterations,
    n_train,
    early_stopping_policy,
    n_early_stopping_eval,
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

    game_type = np.asarray(
        game_type
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
        == len(game_type)
        == len(y_true)
        == len(y_pred_raw)
        == len(y_pred_clip)
    ):
        raise ValueError(
            "Validation 저장 배열 길이가 일치하지 않습니다."
        )

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "game_type": game_type,
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

    domain_metrics = (
        calculate_domain_metrics(
            game_type=game_type,
            y_true=y_true,
            y_pred=y_pred_clip,
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
        "early_stopping_policy": (
            early_stopping_policy
        ),
        "n_early_stopping_eval": int(
            n_early_stopping_eval
        ),
        "early_stopping_eval_share": float(
            n_early_stopping_eval
            / len(y_true)
        ),
        "domain_metrics": (
            domain_metrics
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
        "early_stopping_policy": {
            "default": "full_validation",
            "special_validation_season": int(
                R_ONLY_EARLY_STOPPING_SEASON
            ),
            "special_policy": "R_only_non_F",
            "excluded_game_type": (
                R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE
            ),
            "training_rows": "R_and_F_all",
            "fold_evaluation_rows": "R_and_F_all",
        },
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
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

        val_game_type = (
            df_val["game_type"]
            .astype(str)
        )

        # ---------------------------------------------
        # Early stopping eval set 정책
        # ---------------------------------------------
        #
        # 2023 fold에서만 F를 early stopping metric에서 제외한다.
        # 학습 자체는 R/F 전체 training rows로 수행한다.
        # 이후 성능 평가는 전체 2023 validation에서 그대로 수행한다.

        if (
            validation_season
            == R_ONLY_EARLY_STOPPING_SEASON
        ):
            es_mask = (
                val_game_type
                != R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE
            ).to_numpy()

            if not np.any(
                es_mask
            ):
                raise ValueError(
                    "2023 R-only early stopping eval set이 비어 있습니다."
                )

            X_es = (
                X_val.loc[
                    es_mask
                ]
            )

            y_es = (
                y_val.loc[
                    es_mask
                ]
            )

            early_stopping_policy = (
                "R_only_non_F"
            )
        else:
            X_es = X_val
            y_es = y_val

            early_stopping_policy = (
                "full_validation"
            )

        n_early_stopping_eval = (
            len(y_es)
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
        print(
            "Early stopping policy : "
            f"{early_stopping_policy}"
        )
        print(
            "Early stopping rows   : "
            f"{n_early_stopping_eval:,} "
            f"/ {len(y_val):,}"
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
                X_es,
                y_es,
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
                game_type=(
                    val_game_type
                ),
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
                early_stopping_policy=(
                    early_stopping_policy
                ),
                n_early_stopping_eval=(
                    n_early_stopping_eval
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

        domain_metrics = (
            metadata[
                "domain_metrics"
            ]
        )

        print()
        print("[Domain diagnostics]")
        for domain in (
            "R",
            "F",
        ):
            d = domain_metrics[
                domain
            ]

            if d[
                "metrics"
            ] is None:
                print(
                    f"{domain}: no rows"
                )
                continue

            print(
                f"{domain} | "
                f"n={d['n']:,} | "
                f"share={d['share']:.2%} | "
                f"Brier={d['metrics']['brier']:.8f} | "
                f"BSS_raw={d['metrics']['bss_raw']:.2f} | "
                f"pred_mean={d['prediction_mean']:.6f} | "
                f"actual={d['metrics']['positive_rate']:.6f}"
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
        "HB-EB"
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

    if (
        final_features_check
        != final_features
    ):
        raise ValueError(
            "Validation과 Final Training의 "
            "Feature 구성이 일치하지 않습니다."
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
        "early_stopping_policy": {
            "default": "full_validation",
            "special_validation_season": int(
                R_ONLY_EARLY_STOPPING_SEASON
            ),
            "special_policy": "R_only_non_F",
            "excluded_game_type": (
                R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE
            ),
        },
        "final_iterations": int(
            final_iterations
        ),
        "prediction_policy": (
            "np.clip(raw_prediction, 0.0, 1.0)"
        ),
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
        "early_stopping_policy": {
            "default": "full_validation",
            "special_validation_season": int(
                R_ONLY_EARLY_STOPPING_SEASON
            ),
            "special_policy": "R_only_non_F",
            "excluded_game_type": (
                R_ONLY_EARLY_STOPPING_EXCLUDED_GAME_TYPE
            ),
            "training_rows": "R_and_F_all",
            "fold_evaluation_rows": "R_and_F_all",
        },
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

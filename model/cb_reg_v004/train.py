# train.py: CatBoost Regressor cb_reg_v004
#
# cb_reg_v003의 HB-EB + RMSE + Expanding Window 구조를 그대로 유지하고,
# 고신뢰 Tier A Trackman season-frozen physical profile의 incremental effect를 검증한다.
#
# 핵심 비교:
#   Control   : cb_reg_v003 = HB-EB + CatBoostRegressor + RMSE
#   Treatment : cb_reg_v004 = Control + Tier A Trackman physical profile
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

EXPERIMENT_NAME = "cb_reg_v004"

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
# Trackman 경로 / 실험 정책
# =========================================================

TRACKMAN_DIR = DATA_DIR / "trackman"
TRACKMAN_PROFILE_PATH = (
    TRACKMAN_DIR / "trackman_pitcher_profiles.csv"
)

# 첫 Trackman paired ablation은 가장 보수적인 Tier A만 사용한다.
TRACKMAN_ALLOWED_TIERS = ["A"]

TRACKMAN_FEATURES = [
    "tm_history_available",
    "tm_n",
    "tm_rel_speed_mean",
    "tm_rel_speed_std",
    "tm_spin_rate_mean",
    "tm_spin_rate_std",
    "tm_induced_vert_break_mean",
    "tm_induced_vert_break_std",
    "tm_rel_height_mean",
    "tm_rel_height_std",
    "tm_rel_side_mean",
    "tm_rel_side_std",
]

TRACKMAN_AUDIT_COLS = [
    "pitcher_trackman_id",
    "confidence_tier",
    "target_season",
    "tm_history_max_season",
]


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
# Trackman profile
# =========================================================

def get_fold_trackman_profile_path(validation_season):
    """
    Expanding Window fold용 Trackman profile.

    preprocess_trackman.py가 각 cutoff에서 독립적으로 생성한
    fold-frozen mapping + row-season-frozen profile만 사용한다.
    """
    return (
        TRACKMAN_DIR
        / f"trackman_pitcher_profiles_fold_{validation_season}.csv"
    )


def load_trackman_profile(path):
    """
    preprocess_trackman.py의 단일 profile artifact를 읽고
    schema / uniqueness / future leakage invariant를 검증한다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trackman profile을 찾을 수 없습니다: {path}\n"
            "먼저 data/trackman/preprocess_trackman.py를 실행하세요."
        )

    required_cols = [
        "pitcher_id",
        "pitcher_trackman_id",
        "confidence_tier",
        "target_season",
        *TRACKMAN_FEATURES,
        "tm_history_max_season",
    ]

    profile = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    missing = [
        c
        for c in required_cols
        if c not in profile.columns
    ]

    if missing:
        raise ValueError(
            f"Trackman profile 필수 컬럼 누락: {missing}\n"
            f"path={path}"
        )

    if profile[
        ["pitcher_id", "target_season"]
    ].duplicated().any():
        raise ValueError(
            "Trackman profile의 "
            "(pitcher_id, target_season)가 unique하지 않습니다."
        )

    available = (
        profile["tm_history_available"] == 1
    )

    violation = (
        available
        & (
            profile["tm_history_max_season"]
            >= profile["target_season"]
        )
    )

    if violation.any():
        raise ValueError(
            "Trackman future leakage 발견: "
            "tm_history_max_season >= target_season"
        )

    inf_cols = [
        c for c in TRACKMAN_FEATURES
        if c in profile.columns
    ]

    numeric = (
        profile[inf_cols]
        .select_dtypes(include=[np.number])
    )

    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(
            f"Trackman profile에 Inf가 존재합니다: {path}"
        )

    return profile


def apply_trackman_profile(
    df,
    profile,
    allowed_tiers,
):
    """
    main row의 (pitcher_id, season)에
    이미 고정된 Trackman profile을 many-to-one lookup한다.

    중요:
    - test/train의 다른 행에서 통계를 만들지 않는다.
    - allowed tier 외의 mapping은 unavailable로 처리한다.
    - audit 컬럼은 검증용으로만 유지하고 모델 Feature에는 넣지 않는다.
    """
    df = df.copy()

    allowed_profile = (
        profile.loc[
            profile["confidence_tier"].isin(
                allowed_tiers
            )
        ]
        .copy()
    )

    if allowed_profile[
        ["pitcher_id", "target_season"]
    ].duplicated().any():
        raise ValueError(
            "Tier filtering 후 Trackman profile key 중복"
        )

    merged = df.merge(
        allowed_profile,
        left_on=[
            "pitcher_id",
            "season",
        ],
        right_on=[
            "pitcher_id",
            "target_season",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Tier A가 아니거나 mapping/history가 없는 row는 unavailable.
    merged["tm_history_available"] = (
        merged["tm_history_available"]
        .fillna(0)
        .astype(np.int8)
    )

    merged["tm_n"] = (
        merged["tm_n"]
        .fillna(0)
        .astype(np.float64)
    )

    unavailable = (
        merged["tm_history_available"] == 0
    )

    physical_cols = [
        c
        for c in TRACKMAN_FEATURES
        if c not in {
            "tm_history_available",
            "tm_n",
        }
    ]

    for col in physical_cols:
        merged.loc[
            unavailable,
            col,
        ] = np.nan

    # future leakage audit
    available = (
        merged["tm_history_available"] == 1
    )

    future_violation = (
        available
        & (
            merged["tm_history_max_season"]
            >= merged["season"]
        )
    )

    if future_violation.any():
        bad = merged.loc[
            future_violation,
            [
                "pitcher_id",
                "season",
                "pitcher_trackman_id",
                "tm_history_max_season",
            ],
        ].head(20)

        raise ValueError(
            "Trackman future leakage 발견\n"
            f"{bad}"
        )

    # Tier A only 실험에서 활성 profile이 모두 A인지 확인
    active_tiers = set(
        merged.loc[
            available,
            "confidence_tier",
        ]
        .dropna()
        .unique()
    )

    if not active_tiers.issubset(
        set(allowed_tiers)
    ):
        raise ValueError(
            "허용되지 않은 Trackman tier가 활성화되었습니다: "
            f"{active_tiers}"
        )

    return merged


def calculate_trackman_coverage(df):
    """
    진단/metadata용 Trackman availability 통계.
    예측값 또는 Feature 생성에 다시 사용하지 않는다.
    """
    available = (
        df["tm_history_available"]
        .fillna(0)
        .astype(int)
        == 1
    )

    return {
        "tm_available_count": int(
            available.sum()
        ),
        "tm_available_ratio": float(
            available.mean()
        ),
    }


def get_2025_trackman_lookup(profile):
    """
    최종 artifact에 넣을 2025 Tier A frozen lookup.
    test inference에서는 이 lookup을 행별로 merge하기만 한다.
    """
    lookup = (
        profile.loc[
            (profile["target_season"] == 2025)
            & profile["confidence_tier"].isin(
                TRACKMAN_ALLOWED_TIERS
            ),
            [
                "pitcher_id",
                "confidence_tier",
                "target_season",
                *TRACKMAN_FEATURES,
                "tm_history_max_season",
            ],
        ]
        .copy()
    )

    if lookup["pitcher_id"].duplicated().any():
        raise ValueError(
            "2025 Trackman lookup pitcher_id가 unique하지 않습니다."
        )

    available = (
        lookup["tm_history_available"] == 1
    )

    if (
        lookup.loc[
            available,
            "tm_history_max_season",
        ]
        >= 2025
    ).any():
        raise ValueError(
            "2025 Trackman lookup에 2025 이후 정보가 포함되었습니다."
        )

    return lookup

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
    trackman_train_stats=None,
    trackman_validation_stats=None,
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
    }

    if trackman_train_stats is not None:
        metadata[
            "trackman_train"
        ] = trackman_train_stats

    if trackman_validation_stats is not None:
        metadata[
            "trackman_validation"
        ] = trackman_validation_stats

    metadata[
        "trackman_allowed_tiers"
    ] = TRACKMAN_ALLOWED_TIERS

    metadata[
        "trackman_features"
    ] = TRACKMAN_FEATURES

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
        "trackman_enabled": True,
        "trackman_allowed_tiers": (
            TRACKMAN_ALLOWED_TIERS
        ),
        "trackman_features": (
            TRACKMAN_FEATURES
        ),
        "trackman_policy": (
            "fold_frozen_mapping_and_target_season_frozen_profile"
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
        # Fold-frozen Trackman profile
        # ---------------------------------------------

        fold_profile_path = (
            get_fold_trackman_profile_path(
                validation_season
            )
        )

        fold_profile = load_trackman_profile(
            fold_profile_path
        )

        df_train = apply_trackman_profile(
            df=df_train,
            profile=fold_profile,
            allowed_tiers=TRACKMAN_ALLOWED_TIERS,
        )

        df_val = apply_trackman_profile(
            df=df_val,
            profile=fold_profile,
            allowed_tiers=TRACKMAN_ALLOWED_TIERS,
        )

        trackman_train_stats = (
            calculate_trackman_coverage(
                df_train
            )
        )

        trackman_validation_stats = (
            calculate_trackman_coverage(
                df_val
            )
        )

        # ---------------------------------------------
        # Fold train-only HB-EB prior
        # ---------------------------------------------

        current_features = (
            list(base_features)
            + list(TRACKMAN_FEATURES)
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
                trackman_train_stats=(
                    trackman_train_stats
                ),
                trackman_validation_stats=(
                    trackman_validation_stats
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
            f"Trackman train   : "
            f"{trackman_train_stats['tm_available_ratio']:.2%}"
        )
        print(
            f"Trackman valid   : "
            f"{trackman_validation_stats['tm_available_ratio']:.2%}"
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

    required_trackman_paths = [
        TRACKMAN_PROFILE_PATH,
        *[
            get_fold_trackman_profile_path(
                x["validation_season"]
            )
            for x in EXPANDING_FOLDS
        ],
    ]

    missing_trackman_paths = [
        p
        for p in required_trackman_paths
        if not Path(p).exists()
    ]

    if missing_trackman_paths:
        raise FileNotFoundError(
            "필수 Trackman preprocessing artifact가 없습니다:\n"
            + "\n".join(
                str(p)
                for p in missing_trackman_paths
            )
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
        "HB-EB + Trackman Tier A"
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

    final_trackman_profile = (
        load_trackman_profile(
            TRACKMAN_PROFILE_PATH
        )
    )

    # 최종 2019~2024 재학습용 row-season-frozen profile.
    # final mapping은 <=2024 evidence로 고정되어 있지만,
    # 각 row의 physical statistics는 반드시 Trackman season < row season.
    df_full = apply_trackman_profile(
        df=df_full,
        profile=final_trackman_profile,
        allowed_tiers=TRACKMAN_ALLOWED_TIERS,
    )

    final_trackman_stats = (
        calculate_trackman_coverage(
            df_full
        )
    )

    print(
        "Final Trackman coverage : "
        f"{final_trackman_stats['tm_available_ratio']:.2%}"
    )

    final_priors_dict = {}
    final_features_check = (
        list(base_features)
        + list(TRACKMAN_FEATURES)
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
        "final_iterations": int(
            final_iterations
        ),
        "prediction_policy": (
            "np.clip(raw_prediction, 0.0, 1.0)"
        ),
        "trackman_allowed_tiers": (
            TRACKMAN_ALLOWED_TIERS
        ),
        "trackman_features": (
            TRACKMAN_FEATURES
        ),
        "trackman_policy": (
            "tier_a_only_fold_frozen_mapping_and_target_season_frozen_profile"
        ),
        "trackman_profile_2025": (
            get_2025_trackman_lookup(
                final_trackman_profile
            )
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
        "trackman_enabled": True,
        "trackman_allowed_tiers": (
            TRACKMAN_ALLOWED_TIERS
        ),
        "trackman_features": (
            TRACKMAN_FEATURES
        ),
        "trackman_profile_path": str(
            TRACKMAN_PROFILE_PATH
        ),
        "trackman_policy": (
            "tier_a_only_fold_frozen_mapping_and_target_season_frozen_profile"
        ),
        "trackman_final_train_coverage": (
            final_trackman_stats
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

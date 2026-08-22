# train.py: CatBoost Regressor cb_reg_v005
#
# Paired ablation:
#   Control   : old_050 + HB-EB + CatBoostRegressor(RMSE)
#   Treatment : Control + Tier A Trackman physical profile
#
# Validation:
#   Train      : 2019~2023
#   Validation : 2024
#
# 저장 정책:
# - Validation 결과는 validation/cb_reg_v005/metadata.json 하나에 통합
# - Validation용 Control/Treatment pkl은 저장하지 않음
# - 최종 Treatment 모델만 model/cb_reg_v005/cb_reg_v005.pkl 로 저장
#
# 규칙:
# - test.csv를 학습/Feature 생성에 사용하지 않는다.
# - Trackman mapping/profile은 preprocess_trackman.py가 만든 정적 artifact만 사용한다.
# - 2024 Validation은 fold-frozen 2024 profile만 사용한다.
# - Physical profile은 항상 Trackman season < target season을 만족해야 한다.
# - 최종 2025 inference용 Trackman lookup은 artifact 내부에 저장한다.

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

EXPERIMENT_NAME = "cb_reg_v005"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
METADATA_PATH = VALIDATION_DIR / "metadata.json"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

TRACKMAN_DIR = DATA_DIR / "trackman"
TRACKMAN_PROFILE_PATH = (
    TRACKMAN_DIR / "trackman_pitcher_profiles.csv"
)
TRACKMAN_VALIDATION_PROFILE_PATH = (
    TRACKMAN_DIR
    / "trackman_pitcher_profiles_fold_2024.csv"
)


# =========================================================
# 기본 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

TRAIN_SEASONS = [
    2019,
    2020,
    2021,
    2022,
    2023,
]
VALIDATION_SEASON = 2024

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
# CatBoost
# =========================================================

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


# =========================================================
# Recency Weight
# =========================================================

# recency_weight_2024_v001.py의 old_050 정책을 그대로 사용.
VALIDATION_WEIGHT_MAP = {
    2019: 0.50,
    2020: 0.50,
    2021: 0.50,
    2022: 0.50,
    2023: 1.00,
}

# 2025 final training에서는 2023~2024를 recent regime으로 취급.
FINAL_WEIGHT_MAP = {
    2019: 0.50,
    2020: 0.50,
    2021: 0.50,
    2022: 0.50,
    2023: 1.00,
    2024: 1.00,
}


# =========================================================
# 공통 유틸
# =========================================================

def to_python(obj):
    """json.dump가 처리할 수 있도록 numpy scalar/container를 변환한다."""
    if isinstance(obj, dict):
        return {
            str(k): to_python(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            to_python(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value):
            return None
        return value

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj


def make_sample_weight(
    season_series,
    weight_map,
):
    """
    recency_weight_2024_v001.py와 동일:
    season별 상대 weight를 만든 뒤 평균 1로 normalize한다.
    """
    weights = (
        season_series
        .map(weight_map)
        .astype(float)
        .to_numpy()
    )

    if np.isnan(weights).any():
        missing_seasons = sorted(
            season_series[
                season_series.map(
                    weight_map
                ).isna()
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Weight가 정의되지 않은 season이 있습니다: "
            f"{missing_seasons}"
        )

    weights = (
        weights
        / weights.mean()
    )

    return weights


def summarize_weight_by_season(
    season_series,
    sample_weight,
    raw_weight_map,
):
    tmp = pd.DataFrame({
        "season": (
            season_series
            .to_numpy()
        ),
        "weight": sample_weight,
    })

    result = {}

    for season, group in (
        tmp.groupby("season")
    ):
        result[str(int(season))] = {
            "raw_relative_weight": float(
                raw_weight_map[
                    int(season)
                ]
            ),
            "normalized_weight_mean": float(
                group["weight"].mean()
            ),
            "n": int(len(group)),
        }

    return result


# =========================================================
# Trackman profile
# =========================================================

def load_trackman_profile(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trackman profile을 찾을 수 없습니다: {path}\n"
            "먼저 preprocess_trackman.py를 실행하세요."
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
            "Trackman profile 필수 컬럼 누락: "
            f"{missing}"
        )

    if profile[
        [
            "pitcher_id",
            "target_season",
        ]
    ].duplicated().any():
        raise ValueError(
            "Trackman profile의 "
            "(pitcher_id, target_season)가 unique하지 않습니다."
        )

    available = (
        profile["tm_history_available"]
        == 1
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

    numeric = (
        profile[
            TRACKMAN_FEATURES
        ]
        .select_dtypes(
            include=[np.number]
        )
    )

    if np.isinf(
        numeric.to_numpy(
            dtype=np.float64
        )
    ).any():
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
    (pitcher_id, season) 기준 정적 lookup.
    다른 main/test 행을 사용해 통계를 만들지 않는다.
    """
    df = df.copy()

    allowed_profile = (
        profile.loc[
            profile[
                "confidence_tier"
            ].isin(
                allowed_tiers
            )
        ]
        .copy()
    )

    if allowed_profile[
        [
            "pitcher_id",
            "target_season",
        ]
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

    merged[
        "tm_history_available"
    ] = (
        merged[
            "tm_history_available"
        ]
        .fillna(0)
        .astype(np.int8)
    )

    merged["tm_n"] = (
        merged["tm_n"]
        .fillna(0)
        .astype(np.float64)
    )

    unavailable = (
        merged[
            "tm_history_available"
        ]
        == 0
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

    available = (
        merged[
            "tm_history_available"
        ]
        == 1
    )

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

    future_violation = (
        available
        & (
            merged[
                "tm_history_max_season"
            ]
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

    return merged


def calculate_trackman_coverage(df):
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
        "tm_unavailable_count": int(
            (~available).sum()
        ),
        "tm_unavailable_ratio": float(
            (~available).mean()
        ),
    }


def get_2025_trackman_lookup(profile):
    lookup = (
        profile.loc[
            (
                profile[
                    "target_season"
                ]
                == 2025
            )
            & profile[
                "confidence_tier"
            ].isin(
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

    if lookup[
        "pitcher_id"
    ].duplicated().any():
        raise ValueError(
            "2025 Trackman lookup pitcher_id가 unique하지 않습니다."
        )

    available = (
        lookup[
            "tm_history_available"
        ]
        == 1
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
    학습 subset만으로 prior를 추정한다.
    Validation/Test 정보는 prior 추정에 사용하지 않는다.
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
            1.0 - g_mu
        )
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
                1.0 - mu
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
                1.0 - mu
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

    return (
        prior_df,
        (
            float(
                g_alpha
            ),
            float(
                g_beta
            ),
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
    original_index = (
        df.index
    )

    lookup_df = (
        df[
            group_cols
        ]
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup_df["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup_df["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        df[n_col]
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        df[rate_col]
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n * rate
    )

    eb_rate = (
        success_count
        + alpha
    ) / (
        n
        + alpha
        + beta
    )

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def fit_hbeb_priors(
    df_train,
):
    priors = {}

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

        priors[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config[
                    "group_cols"
                ]
            ),
            "n_col": (
                config["n_col"]
            ),
            "rate_col": (
                config[
                    "rate_col"
                ]
            ),
        }

    return priors


def apply_hbeb_with_priors(
    df,
    base_features,
    priors,
):
    transformed = (
        df.copy()
    )

    current_features = (
        list(base_features)
    )

    for config in (
        HBEB_CONFIGS
    ):
        prior_info = (
            priors[
                config["new_col"]
            ]
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
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
                prior_info[
                    "prior_df"
                ]
            ),
            global_prior=(
                prior_info[
                    "global_prior"
                ]
            ),
        )

        if (
            config["rate_col"]
            in current_features
        ):
            current_features.remove(
                config[
                    "rate_col"
                ]
            )

        if (
            config["new_col"]
            not in current_features
        ):
            current_features.append(
                config[
                    "new_col"
                ]
            )

    return (
        transformed,
        current_features,
    )


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    df = df.copy()

    for col in (
        cat_cols
    ):
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(
                    "__MISSING__"
                )
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
                y_pred
                - y_true
            ) ** 2
        )
    )

    baseline_brier = float(
        positive_rate
        * (
            1.0
            - positive_rate
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

        bss = float(
            max(
                0.0,
                bss_raw,
            )
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
    pred_raw,
):
    pred_raw = np.asarray(
        pred_raw,
        dtype=np.float64,
    )

    below_zero = (
        pred_raw < 0.0
    )

    above_one = (
        pred_raw > 1.0
    )

    outside = (
        below_zero
        | above_one
    )

    return {
        "n_prediction_below_0": int(
            below_zero.sum()
        ),
        "ratio_prediction_below_0": float(
            below_zero.mean()
        ),
        "n_prediction_above_1": int(
            above_one.sum()
        ),
        "ratio_prediction_above_1": float(
            above_one.mean()
        ),
        "n_prediction_outside_0_1": int(
            outside.sum()
        ),
        "ratio_prediction_outside_0_1": float(
            outside.mean()
        ),
    }


def prediction_stats(
    pred_raw,
    pred_clip,
):
    return {
        "prediction_raw_mean": float(
            np.mean(
                pred_raw
            )
        ),
        "prediction_raw_std": float(
            np.std(
                pred_raw
            )
        ),
        "prediction_raw_min": float(
            np.min(
                pred_raw
            )
        ),
        "prediction_raw_max": float(
            np.max(
                pred_raw
            )
        ),
        "prediction_clip_mean": float(
            np.mean(
                pred_clip
            )
        ),
        "prediction_clip_std": float(
            np.std(
                pred_clip
            )
        ),
        "prediction_clip_min": float(
            np.min(
                pred_clip
            )
        ),
        "prediction_clip_max": float(
            np.max(
                pred_clip
            )
        ),
    }


def train_validation_model(
    name,
    train_df,
    valid_df,
    features,
    cat_cols,
    sample_weight,
):
    X_train = (
        prepare_features(
            train_df[
                features
            ],
            cat_cols,
        )
    )

    y_train = (
        train_df[TARGET]
        .astype(
            np.float64
        )
    )

    X_valid = (
        prepare_features(
            valid_df[
                features
            ],
            cat_cols,
        )
    )

    y_valid = (
        valid_df[TARGET]
        .astype(
            np.float64
        )
    )

    model = (
        CatBoostRegressor(
            **MODEL_PARAMS
        )
    )

    print()
    print("=" * 72)
    print(
        f"{name} Validation 학습"
    )
    print("=" * 72)

    start_time = (
        time.time()
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=(
            sample_weight
        ),
        cat_features=(
            cat_cols
        ),
        eval_set=(
            X_valid,
            y_valid,
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

    pred_raw = (
        model.predict(
            X_valid
        )
    )

    pred_clip = (
        np.clip(
            pred_raw,
            0.0,
            1.0,
        )
    )

    metrics = (
        calculate_metrics(
            y_valid,
            pred_clip,
        )
    )

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        final_iterations = int(
            MODEL_PARAMS[
                "iterations"
            ]
        )
    else:
        final_iterations = int(
            best_iteration
            + 1
        )

    result = {
        "name": name,
        "n_train": int(
            len(train_df)
        ),
        "n_validation": int(
            len(valid_df)
        ),
        **metrics,
        "best_iteration": int(
            best_iteration
        ),
        "final_iterations": (
            final_iterations
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **prediction_stats(
            pred_raw,
            pred_clip,
        ),
        **calculate_range_diagnostics(
            pred_raw
        ),
    }

    return (
        model,
        np.asarray(
            pred_clip,
            dtype=np.float64,
        ),
        result,
    )


# =========================================================
# Paired 분석
# =========================================================

def safe_corr(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        np.std(x) == 0.0
        or np.std(y) == 0.0
    ):
        return None

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def subset_comparison(
    y_true,
    control_pred,
    treatment_pred,
    mask,
):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    n = int(
        mask.sum()
    )

    if n == 0:
        return {
            "n": 0,
            "control_brier": None,
            "treatment_brier": None,
            "delta_brier": None,
            "prediction_correlation": None,
            "mean_abs_delta_prediction": None,
        }

    y = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )[mask]
    )

    pc = (
        np.asarray(
            control_pred,
            dtype=np.float64,
        )[mask]
    )

    pt = (
        np.asarray(
            treatment_pred,
            dtype=np.float64,
        )[mask]
    )

    control_brier = float(
        np.mean(
            (
                pc - y
            ) ** 2
        )
    )

    treatment_brier = float(
        np.mean(
            (
                pt - y
            ) ** 2
        )
    )

    return {
        "n": n,
        "control_brier": (
            control_brier
        ),
        "treatment_brier": (
            treatment_brier
        ),
        "delta_brier": float(
            control_brier
            - treatment_brier
        ),
        "prediction_correlation": (
            safe_corr(
                pc,
                pt,
            )
        ),
        "mean_abs_delta_prediction": float(
            np.mean(
                np.abs(
                    pt - pc
                )
            )
        ),
    }


def build_prediction_change_analysis(
    y_true,
    control_pred,
    treatment_pred,
):
    abs_delta = (
        np.abs(
            treatment_pred
            - control_pred
        )
    )

    # qcut의 duplicate edge 가능성을 피하기 위해 rank(method="first") 사용.
    ranks = (
        pd.Series(
            abs_delta
        )
        .rank(
            method="first"
        )
    )

    quantiles = (
        pd.qcut(
            ranks,
            q=4,
            labels=[
                "q1",
                "q2",
                "q3",
                "q4",
            ],
        )
    )

    result = {}

    for label in [
        "q1",
        "q2",
        "q3",
        "q4",
    ]:
        mask = (
            np.asarray(
                quantiles
                == label
            )
        )

        block = (
            subset_comparison(
                y_true,
                control_pred,
                treatment_pred,
                mask,
            )
        )

        block[
            "mean_abs_delta_prediction"
        ] = float(
            np.mean(
                abs_delta[
                    mask
                ]
            )
        )

        result[
            label
        ] = block

    return result


def build_paired_analysis(
    y_true,
    control_pred,
    treatment_pred,
    tm_available,
    control_metrics,
    treatment_metrics,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    control_pred = np.asarray(
        control_pred,
        dtype=np.float64,
    )

    treatment_pred = np.asarray(
        treatment_pred,
        dtype=np.float64,
    )

    delta_pred = (
        treatment_pred
        - control_pred
    )

    control_residual = (
        y_true
        - control_pred
    )

    treatment_residual = (
        y_true
        - treatment_pred
    )

    paired = {
        "control_brier": float(
            control_metrics[
                "brier"
            ]
        ),
        "treatment_brier": float(
            treatment_metrics[
                "brier"
            ]
        ),
        "delta_brier": float(
            control_metrics[
                "brier"
            ]
            - treatment_metrics[
                "brier"
            ]
        ),
        "control_bss": float(
            control_metrics[
                "bss"
            ]
        ),
        "treatment_bss": float(
            treatment_metrics[
                "bss"
            ]
        ),
        "delta_bss": float(
            treatment_metrics[
                "bss"
            ]
            - control_metrics[
                "bss"
            ]
        ),
        "prediction_correlation": (
            safe_corr(
                control_pred,
                treatment_pred,
            )
        ),
        "residual_correlation": (
            safe_corr(
                control_residual,
                treatment_residual,
            )
        ),
        "prediction_mae_difference": float(
            np.mean(
                np.abs(
                    delta_pred
                )
            )
        ),
        "prediction_rmse_difference": float(
            np.sqrt(
                np.mean(
                    delta_pred ** 2
                )
            )
        ),
        "prediction_delta_mean": float(
            np.mean(
                delta_pred
            )
        ),
        "prediction_delta_std": float(
            np.std(
                delta_pred
            )
        ),
    }

    subset_analysis = {
        "available": (
            subset_comparison(
                y_true,
                control_pred,
                treatment_pred,
                tm_available,
            )
        ),
        "unavailable": (
            subset_comparison(
                y_true,
                control_pred,
                treatment_pred,
                ~tm_available,
            )
        ),
    }

    prediction_change = (
        build_prediction_change_analysis(
            y_true,
            control_pred,
            treatment_pred,
        )
    )

    return (
        paired,
        subset_analysis,
        prediction_change,
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

    for path in [
        TRACKMAN_PROFILE_PATH,
        TRACKMAN_VALIDATION_PROFILE_PATH,
    ]:
        if not Path(
            path
        ).exists():
            raise FileNotFoundError(
                f"필수 Trackman artifact가 없습니다: {path}"
            )

    # -----------------------------------------------------
    # Data / Feature schema
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
    )

    required = {
        ID,
        TARGET,
        SEASON_COL,
    }

    missing_required = (
        required
        - set(
            train.columns
        )
    )

    if missing_required:
        raise ValueError(
            "train.csv 필수 컬럼 누락: "
            f"{sorted(missing_required)}"
        )

    base_features = [
        c
        for c in train.columns
        if c not in {
            ID,
            TARGET,
        }
    ]

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    print("=" * 72)
    print(
        f"Experiment           : {EXPERIMENT_NAME}"
    )
    print(
        "Experiment type      : "
        "old_050 + Trackman Tier A paired ablation"
    )
    print(
        "Model                : CatBoostRegressor"
    )
    print(
        "Objective            : RMSE"
    )
    print(
        "Validation           : 2019~2023 -> 2024"
    )
    print(
        f"Train shape          : {train.shape}"
    )
    print(
        f"Base Feature 수      : {len(base_features)}"
    )
    print(
        f"Categorical Feature  : {len(cat_cols)}"
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
    # Holdout split
    # -----------------------------------------------------

    train_mask = (
        train[
            SEASON_COL
        ].isin(
            TRAIN_SEASONS
        )
    )

    valid_mask = (
        train[
            SEASON_COL
        ]
        == VALIDATION_SEASON
    )

    df_train_raw = (
        train.loc[
            train_mask
        ]
        .copy()
    )

    df_valid_raw = (
        train.loc[
            valid_mask
        ]
        .copy()
    )

    if (
        len(
            df_train_raw
        )
        == 0
        or len(
            df_valid_raw
        )
        == 0
    ):
        raise ValueError(
            "Validation split이 비어 있습니다."
        )

    sample_weight = (
        make_sample_weight(
            df_train_raw[
                SEASON_COL
            ],
            VALIDATION_WEIGHT_MAP,
        )
    )

    validation_weight_summary = (
        summarize_weight_by_season(
            df_train_raw[
                SEASON_COL
            ],
            sample_weight,
            VALIDATION_WEIGHT_MAP,
        )
    )

    print()
    print("[Validation Weight]")
    for season in (
        TRAIN_SEASONS
    ):
        info = (
            validation_weight_summary[
                str(season)
            ]
        )
        print(
            f"- {season}: "
            f"raw={info['raw_relative_weight']:.2f}, "
            f"normalized={info['normalized_weight_mean']:.6f}"
        )
    print(
        f"- normalized mean: {sample_weight.mean():.6f}"
    )

    # -----------------------------------------------------
    # Shared HB-EB prior
    # -----------------------------------------------------

    validation_priors = (
        fit_hbeb_priors(
            df_train_raw
        )
    )

    # -----------------------------------------------------
    # Control = old_050 + HB-EB
    # -----------------------------------------------------

    (
        control_train,
        control_features,
    ) = apply_hbeb_with_priors(
        df=df_train_raw,
        base_features=base_features,
        priors=validation_priors,
    )

    (
        control_valid,
        control_features_check,
    ) = apply_hbeb_with_priors(
        df=df_valid_raw,
        base_features=base_features,
        priors=validation_priors,
    )

    if (
        control_features
        != control_features_check
    ):
        raise ValueError(
            "Control train/validation Feature 불일치"
        )

    # -----------------------------------------------------
    # Treatment = Control + Tier A Trackman
    # -----------------------------------------------------

    validation_profile = (
        load_trackman_profile(
            TRACKMAN_VALIDATION_PROFILE_PATH
        )
    )

    treatment_train_raw = (
        apply_trackman_profile(
            df_train_raw,
            validation_profile,
            TRACKMAN_ALLOWED_TIERS,
        )
    )

    treatment_valid_raw = (
        apply_trackman_profile(
            df_valid_raw,
            validation_profile,
            TRACKMAN_ALLOWED_TIERS,
        )
    )

    tm_train_coverage = (
        calculate_trackman_coverage(
            treatment_train_raw
        )
    )

    tm_validation_coverage = (
        calculate_trackman_coverage(
            treatment_valid_raw
        )
    )

    treatment_base_features = (
        list(
            base_features
        )
        + list(
            TRACKMAN_FEATURES
        )
    )

    (
        treatment_train,
        treatment_features,
    ) = apply_hbeb_with_priors(
        df=treatment_train_raw,
        base_features=(
            treatment_base_features
        ),
        priors=validation_priors,
    )

    (
        treatment_valid,
        treatment_features_check,
    ) = apply_hbeb_with_priors(
        df=treatment_valid_raw,
        base_features=(
            treatment_base_features
        ),
        priors=validation_priors,
    )

    if (
        treatment_features
        != treatment_features_check
    ):
        raise ValueError(
            "Treatment train/validation Feature 불일치"
        )

    # Treatment가 Trackman 외에는 Control과 동일한지 확인
    expected_treatment_features = (
        list(
            control_features
        )
    )

    for feature in (
        TRACKMAN_FEATURES
    ):
        if (
            feature
            not in expected_treatment_features
        ):
            expected_treatment_features.insert(
                -2,
                feature,
            )

    if (
        set(
            treatment_features
        )
        != set(
            control_features
        ).union(
            TRACKMAN_FEATURES
        )
    ):
        raise ValueError(
            "Treatment Feature가 Control + Trackman 구조가 아닙니다."
        )

    # -----------------------------------------------------
    # Control / Treatment paired training
    # -----------------------------------------------------

    (
        control_model,
        control_pred,
        control_result,
    ) = train_validation_model(
        name="control_old_050",
        train_df=control_train,
        valid_df=control_valid,
        features=control_features,
        cat_cols=cat_cols,
        sample_weight=sample_weight,
    )

    (
        treatment_model,
        treatment_pred,
        treatment_result,
    ) = train_validation_model(
        name=(
            "treatment_old_050_trackman_tier_a"
        ),
        train_df=treatment_train,
        valid_df=treatment_valid,
        features=treatment_features,
        cat_cols=cat_cols,
        sample_weight=sample_weight,
    )

    y_valid = (
        df_valid_raw[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    tm_available = (
        treatment_valid_raw[
            "tm_history_available"
        ]
        .to_numpy(
            dtype=np.int8
        )
        == 1
    )

    (
        paired_comparison,
        subset_analysis,
        prediction_change_analysis,
    ) = build_paired_analysis(
        y_true=y_valid,
        control_pred=control_pred,
        treatment_pred=treatment_pred,
        tm_available=tm_available,
        control_metrics=control_result,
        treatment_metrics=treatment_result,
    )

    # Validation 모델은 저장하지 않는다.
    del control_model
    del treatment_model

    # -----------------------------------------------------
    # Final Treatment training: 2019~2024
    # -----------------------------------------------------

    final_iterations = int(
        treatment_result[
            "final_iterations"
        ]
    )

    final_sample_weight = (
        make_sample_weight(
            train[
                SEASON_COL
            ],
            FINAL_WEIGHT_MAP,
        )
    )

    final_weight_summary = (
        summarize_weight_by_season(
            train[
                SEASON_COL
            ],
            final_sample_weight,
            FINAL_WEIGHT_MAP,
        )
    )

    final_profile = (
        load_trackman_profile(
            TRACKMAN_PROFILE_PATH
        )
    )

    final_train_raw = (
        apply_trackman_profile(
            train.copy(),
            final_profile,
            TRACKMAN_ALLOWED_TIERS,
        )
    )

    final_trackman_coverage = (
        calculate_trackman_coverage(
            final_train_raw
        )
    )

    final_priors = (
        fit_hbeb_priors(
            final_train_raw
        )
    )

    final_base_features = (
        list(
            base_features
        )
        + list(
            TRACKMAN_FEATURES
        )
    )

    (
        final_train,
        final_features,
    ) = apply_hbeb_with_priors(
        df=final_train_raw,
        base_features=(
            final_base_features
        ),
        priors=final_priors,
    )

    if (
        final_features
        != treatment_features
    ):
        raise ValueError(
            "Validation Treatment와 Final Training Feature 순서가 다릅니다."
        )

    X_full = (
        prepare_features(
            final_train[
                final_features
            ],
            cat_cols,
        )
    )

    y_full = (
        final_train[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": (
            final_iterations
        ),
    }

    final_model = (
        CatBoostRegressor(
            **final_params
        )
    )

    print()
    print("=" * 72)
    print(
        "Final Treatment 학습 "
        "(2019~2024, old_050 extended)"
    )
    print("=" * 72)

    final_start = (
        time.time()
    )

    final_model.fit(
        X_full,
        y_full,
        sample_weight=(
            final_sample_weight
        ),
        cat_features=(
            cat_cols
        ),
    )

    final_elapsed = (
        time.time()
        - final_start
    )

    # -----------------------------------------------------
    # Final artifact 저장
    # -----------------------------------------------------

    final_artifact = {
        "model": final_model,
        "hbeb_priors": final_priors,
        "features": final_features,
        "cat_cols": cat_cols,
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "final_iterations": (
            final_iterations
        ),
        "prediction_policy": (
            "np.clip(raw_prediction, 0.0, 1.0)"
        ),
        "weight_policy": (
            "old_050_extended"
        ),
        "final_weight_map": (
            FINAL_WEIGHT_MAP
        ),
        "trackman_allowed_tiers": (
            TRACKMAN_ALLOWED_TIERS
        ),
        "trackman_features": (
            TRACKMAN_FEATURES
        ),
        "trackman_policy": (
            "tier_a_only_fold_frozen_validation_and_target_season_frozen_profile"
        ),
        "trackman_profile_2025": (
            get_2025_trackman_lookup(
                final_profile
            )
        ),
    }

    joblib.dump(
        final_artifact,
        MODEL_PATH,
        compress=3,
    )

    # -----------------------------------------------------
    # metadata.json 하나에 결과 통합
    # -----------------------------------------------------

    metadata = {
        "experiment": {
            "experiment_name": (
                EXPERIMENT_NAME
            ),
            "experiment_type": (
                "old_050_trackman_tier_a_paired_ablation"
            ),
            "model_type": (
                "CatBoostRegressor"
            ),
            "objective": "RMSE",
            "random_seed": int(
                MODEL_PARAMS[
                    "random_seed"
                ]
            ),
        },
        "data": {
            "train_seasons": (
                TRAIN_SEASONS
            ),
            "validation_season": (
                VALIDATION_SEASON
            ),
            "n_train": int(
                len(
                    df_train_raw
                )
            ),
            "n_validation": int(
                len(
                    df_valid_raw
                )
            ),
            "target_mean_train": float(
                df_train_raw[
                    TARGET
                ].mean()
            ),
            "target_mean_validation": float(
                df_valid_raw[
                    TARGET
                ].mean()
            ),
        },
        "weighting": {
            "policy": (
                "old_050"
            ),
            "validation_weight_map": (
                VALIDATION_WEIGHT_MAP
            ),
            "validation_normalized_weights": (
                validation_weight_summary
            ),
            "final_policy": (
                "old_050_extended_2023_2024_recent"
            ),
            "final_weight_map": (
                FINAL_WEIGHT_MAP
            ),
            "final_normalized_weights": (
                final_weight_summary
            ),
        },
        "features": {
            "base_features": (
                base_features
            ),
            "control_features": (
                control_features
            ),
            "treatment_features": (
                treatment_features
            ),
            "categorical_features": (
                cat_cols
            ),
            "hbeb_configs": (
                HBEB_CONFIGS
            ),
            "trackman_features": (
                TRACKMAN_FEATURES
            ),
        },
        "trackman": {
            "enabled_in_control": False,
            "enabled_in_treatment": True,
            "allowed_tiers": (
                TRACKMAN_ALLOWED_TIERS
            ),
            "validation_profile_path": str(
                TRACKMAN_VALIDATION_PROFILE_PATH
            ),
            "final_profile_path": str(
                TRACKMAN_PROFILE_PATH
            ),
            "policy": (
                "tier_a_only_fold_frozen_validation_and_target_season_frozen_profile"
            ),
            "train_coverage_2024_holdout": (
                tm_train_coverage
            ),
            "validation_coverage_2024": (
                tm_validation_coverage
            ),
            "final_train_coverage": (
                final_trackman_coverage
            ),
        },
        "model_params": (
            MODEL_PARAMS
        ),
        "control": {
            **control_result,
            "trackman_enabled": False,
            "features_count": int(
                len(
                    control_features
                )
            ),
        },
        "treatment": {
            **treatment_result,
            "trackman_enabled": True,
            "trackman_allowed_tiers": (
                TRACKMAN_ALLOWED_TIERS
            ),
            "features_count": int(
                len(
                    treatment_features
                )
            ),
        },
        "paired_comparison": (
            paired_comparison
        ),
        "trackman_subset_analysis": (
            subset_analysis
        ),
        "prediction_change_analysis": (
            prediction_change_analysis
        ),
        "selection": {
            "artifact_model": (
                "treatment"
            ),
            "decision_status": (
                "analysis_required"
            ),
            "selection_note": (
                "Validation에서는 Control/Treatment를 비교하지만 "
                "cb_reg_v005 artifact는 Treatment 정의로 저장한다."
            ),
        },
        "final_training": {
            "training_seasons": [
                2019,
                2020,
                2021,
                2022,
                2023,
                2024,
            ],
            "selected_model": (
                "treatment_old_050_trackman_tier_a"
            ),
            "final_iterations": int(
                final_iterations
            ),
            "weight_policy": (
                "old_050_extended"
            ),
            "model_path": str(
                MODEL_PATH
            ),
            "elapsed_seconds": float(
                final_elapsed
            ),
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(
                metadata
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # train.py만 모델 폴더에 보존
    source_path = (
        Path(
            __file__
        )
        .resolve()
    )

    shutil.copy2(
        source_path,
        MODEL_DIR / "train.py",
    )

    # -----------------------------------------------------
    # 출력
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Paired Ablation 결과")
    print("=" * 72)
    print(
        "Control Brier        : "
        f"{control_result['brier']:.10f}"
    )
    print(
        "Treatment Brier      : "
        f"{treatment_result['brier']:.10f}"
    )
    print(
        "Delta Brier          : "
        f"{paired_comparison['delta_brier']:+.10f}"
    )
    print(
        "Control BSS          : "
        f"{control_result['bss']:.4f}"
    )
    print(
        "Treatment BSS        : "
        f"{treatment_result['bss']:.4f}"
    )
    print(
        "Delta BSS            : "
        f"{paired_comparison['delta_bss']:+.4f}"
    )
    print()
    print(
        "TM available Delta   : "
        f"{subset_analysis['available']['delta_brier']}"
    )
    print(
        "TM unavailable Delta : "
        f"{subset_analysis['unavailable']['delta_brier']}"
    )
    print(
        "Prediction Corr      : "
        f"{paired_comparison['prediction_correlation']}"
    )
    print(
        "Residual Corr        : "
        f"{paired_comparison['residual_correlation']}"
    )
    print("=" * 72)

    print()
    print(
        f"Validation metadata : {METADATA_PATH}"
    )
    print(
        f"Final model         : {MODEL_PATH}"
    )
    print(
        f"Final train.py      : {MODEL_DIR / 'train.py'}"
    )

    print()
    print("[저장 정책]")
    print("- Validation Control/Treatment pkl 저장 안 함")
    print("- Validation 결과는 metadata.json 하나에 통합")
    print("- 최종 Treatment pkl 하나만 저장")


if __name__ == "__main__":
    main()

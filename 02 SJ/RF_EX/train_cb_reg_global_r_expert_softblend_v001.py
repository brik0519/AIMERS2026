# train_cb_reg_global_r_expert_softblend_v001.py
# ============================================================
# Global + R-only Expert Soft Blend
#
# 목적
# ----
# 기존 cb_reg_v003_test를 기준으로,
#
#   Global model:
#       R + F 전체 train 학습
#
#   R expert:
#       R train만 학습
#
# 을 만든 뒤 R row에서만 두 예측을 soft blend한다.
#
#   R final = (1 - w) * Global + w * R expert
#   F final = Global 그대로
#
# Blend weight w는 2021~2023 discovery folds에서만 선택하고,
# 2024에서는 그 weight를 고정해서 평가한다.
#
# 중요
# ----
# - HB-EB prior는 각 fold의 전체 train(R+F)에서만 추정한다.
#   → Global과 R expert가 동일한 HB-EB 변환을 공유하므로
#     실험 차이는 "CatBoost 학습 row를 R-only로 제한"하는 것에 집중된다.
#
# - 2024는 blend weight 선택에는 사용하지 않는다.
# - 다만 각 fold의 CatBoost Early Stopping은 기존 Expanding Window
#   validation 방식과 동일하게 해당 fold validation을 사용한다.
#   즉 이 코드는 "blend weight의 temporal transfer"를 검증한다.
#
# 대회 독립행 규칙
# ----------------
# - test.csv는 schema 확인용 header만 읽는다.
# - HB-EB prior는 train에서만 생성한다.
# - test 다른 row / 분포 / rolling / lag / group aggregate를 사용하지 않는다.
# - routing/blend는 각 row 자신의 game_type과 사전에 고정된 weight만 사용한다.
# ============================================================

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Feature 제외
# ============================================================

EXCLUDE_FEATURES = [
    "asof_pitcher_pitchmix_n",
]


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "cb_reg_global_r_expert_softblend_v001"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME


# ============================================================
# 기본 설정
# ============================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"
ALLOWED_GROUPS = {
    R_GROUP,
    F_GROUP,
}

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

DISCOVERY_SEASONS = [
    2021,
    2022,
    2023,
]

HOLDOUT_SEASON = 2024

R_BLEND_GRID = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# HB-EB
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
# JSON helper
# ============================================================

def to_python(obj):
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

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    각 fold의 train subset만으로 HB-EB prior를 추정한다.
    """
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

    g_alpha = (
        g_mu * g_m
    )

    g_beta = (
        (1.0 - g_mu) * g_m
    )

    temp = (
        df_train[
            group_cols + [rate_col]
        ]
        .copy()
    )

    temp[rate_col] = pd.to_numeric(
        temp[rate_col],
        errors="coerce",
    )

    stats = (
        temp.groupby(
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
            1.0 - stats["mean"]
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
            (1.0 - mu) * m
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
    """
    train-only prior를 lookup하여 각 row를 독립적으로 변환한다.
    """
    original_index = (
        df.index
    )

    lookup_df = (
        df[group_cols]
        .reset_index(drop=True)
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

    eb_rate = np.divide(
        success_count + alpha,
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

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def fit_hbeb_priors(
    df_train,
):
    priors = {}

    for config in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=(
                config["group_cols"]
            ),
            rate_col=(
                config["rate_col"]
            ),
        )

        priors[
            config["new_col"]
        ] = {
            "prior_df": (
                prior_df
            ),
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config["group_cols"]
            ),
            "n_col": (
                config["n_col"]
            ),
            "rate_col": (
                config["rate_col"]
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

    for config in HBEB_CONFIGS:
        info = (
            priors[
                config["new_col"]
            ]
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=(
                config["group_cols"]
            ),
            n_col=(
                config["n_col"]
            ),
            rate_col=(
                config["rate_col"]
            ),
            prior_df=(
                info["prior_df"]
            ),
            global_prior=(
                info["global_prior"]
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

    return (
        transformed,
        current_features,
    )


# ============================================================
# Feature
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
# Metric
# ============================================================

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

        bss = float(
            max(
                0.0,
                bss_raw,
            )
        )

    return {
        "n": int(
            len(y_true)
        ),
        "positive_rate": (
            positive_rate
        ),
        "brier": (
            brier
        ),
        "baseline_brier": (
            baseline_brier
        ),
        "bss_raw": (
            bss_raw
        ),
        "bss": (
            bss
        ),
    }


# ============================================================
# Game-type check
# ============================================================

def validate_game_types(
    df,
    context,
):
    values = set(
        df[GROUP_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    unexpected = (
        values - ALLOWED_GROUPS
    )

    if unexpected:
        raise ValueError(
            f"{context}: 예상하지 못한 game_type: "
            f"{sorted(unexpected)}"
        )

    if R_GROUP not in values:
        raise ValueError(
            f"{context}: R row가 없습니다."
        )


# ============================================================
# Model fit
# ============================================================

def fit_global_model(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_cols,
):
    model = CatBoostRegressor(
        **MODEL_PARAMS
    )

    start = (
        time.time()
    )

    model.fit(
        X_train,
        y_train,
        cat_features=(
            cat_cols
        ),
        eval_set=(
            X_val,
            y_val,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = float(
        time.time()
        - start
    )

    best_iteration = (
        model.get_best_iteration()
    )

    final_iterations = int(
        best_iteration + 1
        if (
            best_iteration is not None
            and best_iteration >= 0
        )
        else MODEL_PARAMS[
            "iterations"
        ]
    )

    return (
        model,
        final_iterations,
        elapsed,
    )


def fit_r_expert_model(
    X_train,
    y_train,
    train_groups,
    X_val,
    y_val,
    val_groups,
    cat_cols,
):
    """
    CatBoost 자체 학습 row를 R-only로 제한한다.
    Early Stopping도 R validation만 사용한다.
    """
    train_r_mask = (
        train_groups
        .astype(str)
        .eq(R_GROUP)
        .to_numpy()
    )

    val_r_mask = (
        val_groups
        .astype(str)
        .eq(R_GROUP)
        .to_numpy()
    )

    if (
        train_r_mask.sum() == 0
        or val_r_mask.sum() == 0
    ):
        raise ValueError(
            "R expert train/validation subset이 비었습니다."
        )

    X_train_r = (
        X_train.loc[
            train_r_mask
        ]
    )

    y_train_r = (
        y_train.loc[
            train_r_mask
        ]
    )

    X_val_r = (
        X_val.loc[
            val_r_mask
        ]
    )

    y_val_r = (
        y_val.loc[
            val_r_mask
        ]
    )

    model = CatBoostRegressor(
        **MODEL_PARAMS
    )

    start = (
        time.time()
    )

    model.fit(
        X_train_r,
        y_train_r,
        cat_features=(
            cat_cols
        ),
        eval_set=(
            X_val_r,
            y_val_r,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = float(
        time.time()
        - start
    )

    best_iteration = (
        model.get_best_iteration()
    )

    final_iterations = int(
        best_iteration + 1
        if (
            best_iteration is not None
            and best_iteration >= 0
        )
        else MODEL_PARAMS[
            "iterations"
        ]
    )

    return (
        model,
        final_iterations,
        elapsed,
        int(train_r_mask.sum()),
        int(val_r_mask.sum()),
    )


# ============================================================
# Fold
# ============================================================

def run_fold(
    fold_config,
    train,
    base_features,
    cat_cols,
):
    fold = int(
        fold_config["fold"]
    )

    train_seasons = list(
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

    df_train_raw = (
        train.loc[
            train_mask
        ]
        .copy()
    )

    df_val_raw = (
        train.loc[
            val_mask
        ]
        .copy()
    )

    validate_game_types(
        df_train_raw,
        f"Fold {fold} train",
    )

    validate_game_types(
        df_val_raw,
        f"Fold {fold} validation",
    )

    # --------------------------------------------------------
    # Shared train-only HB-EB prior
    # --------------------------------------------------------

    priors = (
        fit_hbeb_priors(
            df_train_raw
        )
    )

    (
        df_train,
        current_features,
    ) = apply_hbeb_with_priors(
        df=df_train_raw,
        base_features=base_features,
        priors=priors,
    )

    (
        df_val,
        valid_features,
    ) = apply_hbeb_with_priors(
        df=df_val_raw,
        base_features=base_features,
        priors=priors,
    )

    if (
        current_features
        != valid_features
    ):
        raise ValueError(
            "Train/Validation Feature 불일치"
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
        df_train[
            TARGET
        ]
        .astype(
            np.float64
        )
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
        df_val[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    train_groups = (
        df_train[
            GROUP_COL
        ]
    )

    val_groups = (
        df_val[
            GROUP_COL
        ]
    )

    print()
    print("=" * 90)
    print(
        f"Fold {fold}: "
        f"{train_seasons[0]}~{train_seasons[-1]} "
        f"-> {validation_season}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    (
        global_model,
        global_iterations,
        global_elapsed,
    ) = fit_global_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_cols=cat_cols,
    )

    global_raw = np.asarray(
        global_model.predict(
            X_val
        ),
        dtype=np.float64,
    )

    global_pred = np.clip(
        global_raw,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # R-only expert
    # --------------------------------------------------------

    (
        r_model,
        r_iterations,
        r_elapsed,
        n_train_r,
        n_val_r,
    ) = fit_r_expert_model(
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        X_val=X_val,
        y_val=y_val,
        val_groups=val_groups,
        cat_cols=cat_cols,
    )

    r_mask = (
        val_groups
        .astype(str)
        .eq(R_GROUP)
        .to_numpy()
    )

    r_expert_pred = np.full(
        len(X_val),
        np.nan,
        dtype=np.float64,
    )

    r_raw = np.asarray(
        r_model.predict(
            X_val.loc[
                r_mask
            ]
        ),
        dtype=np.float64,
    )

    r_expert_pred[
        r_mask
    ] = np.clip(
        r_raw,
        0.0,
        1.0,
    )

    pred_df = pd.DataFrame({
        ID: (
            df_val[
                ID
            ].to_numpy()
        ),
        "target": (
            y_val.to_numpy()
        ),
        "game_type": (
            val_groups
            .astype(str)
            .to_numpy()
        ),
        "global_prediction": (
            global_pred
        ),
        "r_expert_prediction": (
            r_expert_pred
        ),
        "validation_season": (
            validation_season
        ),
    })

    result = {
        "fold": (
            fold
        ),
        "train_seasons": (
            train_seasons
        ),
        "validation_season": (
            validation_season
        ),
        "n_train": int(
            len(df_train)
        ),
        "n_validation": int(
            len(df_val)
        ),
        "n_train_R": (
            n_train_r
        ),
        "n_validation_R": (
            n_val_r
        ),
        "global_iterations": (
            global_iterations
        ),
        "R_expert_iterations": (
            r_iterations
        ),
        "global_elapsed_seconds": (
            global_elapsed
        ),
        "R_expert_elapsed_seconds": (
            r_elapsed
        ),
    }

    print(
        f"Global iterations   : {global_iterations}"
    )
    print(
        f"R expert iterations : {r_iterations}"
    )
    print(
        f"R train / valid     : "
        f"{n_train_r:,} / {n_val_r:,}"
    )

    return (
        pred_df,
        result,
        current_features,
    )


# ============================================================
# Blend
# ============================================================

def make_blended_prediction(
    df,
    weight,
):
    """
    F는 Global 그대로.
    R만 Global + R expert blend.
    """
    pred = (
        df[
            "global_prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
        .copy()
    )

    r_mask = (
        df[
            "game_type"
        ]
        .astype(str)
        .eq(R_GROUP)
        .to_numpy()
    )

    global_r = (
        df.loc[
            r_mask,
            "global_prediction",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    expert_r = (
        df.loc[
            r_mask,
            "r_expert_prediction",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        expert_r
    ).all():
        raise ValueError(
            "R expert prediction에 NaN/Inf가 있습니다."
        )

    pred[
        r_mask
    ] = (
        (1.0 - weight)
        * global_r
        + weight
        * expert_r
    )

    return np.clip(
        pred,
        0.0,
        1.0,
    )


def evaluate_prediction(
    df,
    pred,
):
    y = (
        df[
            "target"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    overall = (
        calculate_metrics(
            y,
            pred,
        )
    )

    r_mask = (
        df[
            "game_type"
        ]
        .astype(str)
        .eq(R_GROUP)
        .to_numpy()
    )

    f_mask = (
        df[
            "game_type"
        ]
        .astype(str)
        .eq(F_GROUP)
        .to_numpy()
    )

    r_metrics = (
        calculate_metrics(
            y[
                r_mask
            ],
            np.asarray(
                pred
            )[
                r_mask
            ],
        )
    )

    f_metrics = (
        calculate_metrics(
            y[
                f_mask
            ],
            np.asarray(
                pred
            )[
                f_mask
            ],
        )
    )

    return {
        "overall": (
            overall
        ),
        "R": (
            r_metrics
        ),
        "F": (
            f_metrics
        ),
    }


def select_weight(
    discovery_df,
):
    """
    2021~2023 discovery folds에서만 weight를 선택한다.
    기준 = pooled overall Brier 최소.
    F prediction은 모든 weight에서 동일하므로,
    실질적으로 R 개선을 선택하는 것과 같은 방향이다.
    """
    rows = []

    for weight in (
        R_BLEND_GRID
    ):
        pred = (
            make_blended_prediction(
                discovery_df,
                weight,
            )
        )

        metrics = (
            evaluate_prediction(
                discovery_df,
                pred,
            )
        )

        rows.append({
            "weight": float(
                weight
            ),
            "overall_brier": float(
                metrics[
                    "overall"
                ]["brier"]
            ),
            "R_brier": float(
                metrics[
                    "R"
                ]["brier"]
            ),
            "F_brier": float(
                metrics[
                    "F"
                ]["brier"]
            ),
        })

    curve = pd.DataFrame(
        rows
    ).sort_values(
        [
            "overall_brier",
            "weight",
        ],
        ascending=[
            True,
            True,
        ],
    )

    best_weight = float(
        curve.iloc[0][
            "weight"
        ]
    )

    return (
        best_weight,
        curve,
    )


# ============================================================
# Main
# ============================================================

def main():
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

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        c
        for c in test_cols
        if (
            c != ID
            and c not in EXCLUDE_FEATURES
        )
    ]

    if GROUP_COL not in base_features:
        raise ValueError(
            f"{GROUP_COL}이 feature schema에 없습니다."
        )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        DATA_DIR
        / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + base_features
            + [TARGET]
        ),
    )

    validate_game_types(
        train,
        "Full train",
    )

    print("=" * 90)
    print(
        f"Experiment          : {EXPERIMENT_NAME}"
    )
    print(
        "Global              : R+F 전체 train"
    )
    print(
        "R expert            : R-only train"
    )
    print(
        "F                   : Global 그대로"
    )
    print(
        f"Blend grid          : {R_BLEND_GRID}"
    )
    print(
        f"Weight discovery    : {DISCOVERY_SEASONS}"
    )
    print(
        f"Weight holdout      : {HOLDOUT_SEASON}"
    )
    print(
        "Feature Engineering : HB-EB"
    )
    print(
        f"Excluded            : {EXCLUDE_FEATURES}"
    )
    print("=" * 90)

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # 4-fold predictions 생성
    # --------------------------------------------------------

    fold_frames = []
    fold_results = []
    final_features = None

    for fold_config in (
        EXPANDING_FOLDS
    ):
        (
            pred_df,
            fold_result,
            features,
        ) = run_fold(
            fold_config=(
                fold_config
            ),
            train=train,
            base_features=(
                base_features
            ),
            cat_cols=(
                cat_cols
            ),
        )

        if final_features is None:
            final_features = list(
                features
            )
        elif (
            final_features
            != features
        ):
            raise ValueError(
                "Fold별 Feature 구성이 다릅니다."
            )

        fold_frames.append(
            pred_df
        )

        fold_results.append(
            fold_result
        )

    oof = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Discovery / Holdout 분리
    # --------------------------------------------------------

    discovery_df = (
        oof.loc[
            oof[
                "validation_season"
            ].isin(
                DISCOVERY_SEASONS
            )
        ]
        .copy()
    )

    holdout_df = (
        oof.loc[
            oof[
                "validation_season"
            ]
            == HOLDOUT_SEASON
        ]
        .copy()
    )

    if (
        len(discovery_df) == 0
        or len(holdout_df) == 0
    ):
        raise ValueError(
            "Discovery/Holdout split이 비었습니다."
        )

    # --------------------------------------------------------
    # 2021~2023에서 weight 선택
    # --------------------------------------------------------

    (
        selected_weight,
        weight_curve,
    ) = select_weight(
        discovery_df
    )

    weight_curve.to_csv(
        VALIDATION_DIR
        / "weight_search_2021_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Discovery 결과
    # --------------------------------------------------------

    discovery_global_pred = (
        discovery_df[
            "global_prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    discovery_treatment_pred = (
        make_blended_prediction(
            discovery_df,
            selected_weight,
        )
    )

    discovery_control = (
        evaluate_prediction(
            discovery_df,
            discovery_global_pred,
        )
    )

    discovery_treatment = (
        evaluate_prediction(
            discovery_df,
            discovery_treatment_pred,
        )
    )

    # --------------------------------------------------------
    # 2024: selected weight 고정
    # --------------------------------------------------------

    holdout_global_pred = (
        holdout_df[
            "global_prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    holdout_treatment_pred = (
        make_blended_prediction(
            holdout_df,
            selected_weight,
        )
    )

    holdout_control = (
        evaluate_prediction(
            holdout_df,
            holdout_global_pred,
        )
    )

    holdout_treatment = (
        evaluate_prediction(
            holdout_df,
            holdout_treatment_pred,
        )
    )

    # F는 반드시 완전히 동일해야 한다.
    f_mask = (
        holdout_df[
            "game_type"
        ]
        .astype(str)
        .eq(F_GROUP)
        .to_numpy()
    )

    if not np.allclose(
        holdout_global_pred[
            f_mask
        ],
        holdout_treatment_pred[
            f_mask
        ],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "2024 F prediction이 변경되었습니다."
        )

    # --------------------------------------------------------
    # Fold별 selected-weight 결과
    # --------------------------------------------------------

    fold_selected_results = []

    for season in [
        2021,
        2022,
        2023,
        2024,
    ]:
        fold_df = (
            oof.loc[
                oof[
                    "validation_season"
                ]
                == season
            ]
            .copy()
        )

        global_pred = (
            fold_df[
                "global_prediction"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        blend_pred = (
            make_blended_prediction(
                fold_df,
                selected_weight,
            )
        )

        control = (
            evaluate_prediction(
                fold_df,
                global_pred,
            )
        )

        treatment = (
            evaluate_prediction(
                fold_df,
                blend_pred,
            )
        )

        fold_selected_results.append({
            "validation_season": int(
                season
            ),
            "control_brier": float(
                control[
                    "overall"
                ]["brier"]
            ),
            "treatment_brier": float(
                treatment[
                    "overall"
                ]["brier"]
            ),
            "brier_gain": float(
                control[
                    "overall"
                ]["brier"]
                - treatment[
                    "overall"
                ]["brier"]
            ),
            "R_control_brier": float(
                control[
                    "R"
                ]["brier"]
            ),
            "R_treatment_brier": float(
                treatment[
                    "R"
                ]["brier"]
            ),
            "R_brier_gain": float(
                control[
                    "R"
                ]["brier"]
                - treatment[
                    "R"
                ]["brier"]
            ),
        })

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    oof_out = (
        oof.copy()
    )

    oof_out[
        "selected_blend_prediction"
    ] = (
        make_blended_prediction(
            oof_out,
            selected_weight,
        )
    )

    oof_out.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "architecture": {
            "global": (
                "CatBoostRegressor trained on all R+F rows"
            ),
            "R_expert": (
                "CatBoostRegressor trained only on R rows"
            ),
            "R_prediction": (
                "(1-w)*global + w*R_expert"
            ),
            "F_prediction": (
                "global only"
            ),
        },
        "blend_selection": {
            "grid": (
                R_BLEND_GRID
            ),
            "discovery_seasons": (
                DISCOVERY_SEASONS
            ),
            "holdout_season": (
                HOLDOUT_SEASON
            ),
            "selected_weight": (
                selected_weight
            ),
            "selection_metric": (
                "pooled overall Brier on discovery folds"
            ),
        },
        "discovery": {
            "control": (
                discovery_control
            ),
            "treatment": (
                discovery_treatment
            ),
            "overall_brier_gain": float(
                discovery_control[
                    "overall"
                ]["brier"]
                - discovery_treatment[
                    "overall"
                ]["brier"]
            ),
            "R_brier_gain": float(
                discovery_control[
                    "R"
                ]["brier"]
                - discovery_treatment[
                    "R"
                ]["brier"]
            ),
        },
        "holdout_2024": {
            "control": (
                holdout_control
            ),
            "treatment": (
                holdout_treatment
            ),
            "overall_brier_gain": float(
                holdout_control[
                    "overall"
                ]["brier"]
                - holdout_treatment[
                    "overall"
                ]["brier"]
            ),
            "R_brier_gain": float(
                holdout_control[
                    "R"
                ]["brier"]
                - holdout_treatment[
                    "R"
                ]["brier"]
            ),
            "F_brier_gain": float(
                holdout_control[
                    "F"
                ]["brier"]
                - holdout_treatment[
                    "F"
                ]["brier"]
            ),
            "overall_bss_gain": float(
                holdout_treatment[
                    "overall"
                ]["bss"]
                - holdout_control[
                    "overall"
                ]["bss"]
            ),
        },
        "fold_training": (
            fold_results
        ),
        "fold_selected_weight_results": (
            fold_selected_results
        ),
        "features": (
            final_features
        ),
        "categorical_features": (
            cat_cols
        ),
        "exclude_features": (
            EXCLUDE_FEATURES
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "validation_note": (
            "2024 is not used for blend-weight selection. "
            "Per-fold early stopping still uses each fold validation, "
            "matching the existing expanding-window model training procedure."
        ),
    }

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(
                summary
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    shutil.copy2(
        Path(
            __file__
        ).resolve(),
        MODEL_DIR
        / "train.py",
    )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "Global + R Expert Soft Blend Summary"
    )
    print("=" * 90)

    print()
    print("[Discovery weight search: 2021~2023]")
    for _, row in (
        weight_curve
        .sort_values(
            "weight"
        )
        .iterrows()
    ):
        print(
            f"w={row['weight']:.2f} | "
            f"Overall={row['overall_brier']:.10f} | "
            f"R={row['R_brier']:.10f}"
        )

    print()
    print(
        f"Selected R expert weight : "
        f"{selected_weight:.2f}"
    )

    print()
    print("[Discovery 2021~2023]")
    print(
        "Control Brier           : "
        f"{discovery_control['overall']['brier']:.10f}"
    )
    print(
        "Treatment Brier         : "
        f"{discovery_treatment['overall']['brier']:.10f}"
    )
    print(
        "Overall Brier Gain      : "
        f"{summary['discovery']['overall_brier_gain']:+.10f}"
    )
    print(
        "R Brier Gain            : "
        f"{summary['discovery']['R_brier_gain']:+.10f}"
    )

    print()
    print("[Locked-weight evaluation: 2024]")
    print(
        "Control Brier           : "
        f"{holdout_control['overall']['brier']:.10f}"
    )
    print(
        "Treatment Brier         : "
        f"{holdout_treatment['overall']['brier']:.10f}"
    )
    print(
        "Overall Brier Gain      : "
        f"{summary['holdout_2024']['overall_brier_gain']:+.10f}"
    )
    print(
        "Overall BSS Gain        : "
        f"{summary['holdout_2024']['overall_bss_gain']:+.4f}"
    )
    print(
        "R Brier Gain            : "
        f"{summary['holdout_2024']['R_brier_gain']:+.10f}"
    )
    print(
        "F Brier Gain            : "
        f"{summary['holdout_2024']['F_brier_gain']:+.10f}"
    )

    print()
    print("[Selected weight fold results]")
    for row in (
        fold_selected_results
    ):
        print(
            f"{row['validation_season']} | "
            f"Overall gain="
            f"{row['brier_gain']:+.10f} | "
            f"R gain="
            f"{row['R_brier_gain']:+.10f}"
        )

    print("=" * 90)

    print()
    print(
        f"Validation output : {VALIDATION_DIR}"
    )


if __name__ == "__main__":
    main()

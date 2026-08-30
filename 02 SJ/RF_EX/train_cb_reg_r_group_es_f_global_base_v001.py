# train_cb_reg_r_group_es_f_global_base_v001.py
# ============================================================
# BASE CANDIDATE
# R Group-ES + F Global
#
# 기준 구조
# ---------
# Global model
#   - Train: 해당 fold의 R + F 전체
#   - Early Stopping: 해당 fold validation 전체
#
# R Group-ES model
#   - Train: 해당 fold의 R + F 전체  (R-only 학습 아님)
#   - Early Stopping: 해당 fold validation의 R row만
#
# 최종 validation prediction
#   - game_type == "R" -> R Group-ES model
#   - game_type == "F" -> Global model
#
# 목적
# ----
# 1) 4개 Expanding Window fold에서 Base 성능을 최종 확인
# 2) OOF prediction / OOF residual을 저장
# 3) 이후 residual correction 분석의 기준 Base로 사용
#
# Expanding Window
# ----------------
# Fold 1: 2019~2020 -> 2021
# Fold 2: 2019~2021 -> 2022
# Fold 3: 2019~2022 -> 2023
# Fold 4: 2019~2023 -> 2024
#
# OOF residual
# ------------
# residual = target - base_prediction
#
# 양수 residual:
#   모델이 실제보다 낮게 예측
#
# 음수 residual:
#   모델이 실제보다 높게 예측
#
# 대회 규칙
# ---------
# - test.csv는 feature schema 확인용 header만 읽는다.
# - HB-EB prior는 fold train에서만 생성한다.
# - routing은 각 row 자신의 game_type만 사용한다.
# - test 다른 row / rolling / lag / test 집계 / test 분포 사용 없음.
# ============================================================

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Experiment
# ============================================================

EXPERIMENT_NAME = "cb_reg_r_group_es_f_global_base_v001"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME


# ============================================================
# Columns
# ============================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"
ALLOWED_GROUPS = {R_GROUP, F_GROUP}

EXCLUDE_FEATURES = [
    "asof_pitcher_pitchmix_n",
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


# ============================================================
# Expanding Window
# ============================================================

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
# Helpers
# ============================================================

def to_python(obj):
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]

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


def validate_game_types(df, context):
    values = set(
        df[GROUP_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    unexpected = values - ALLOWED_GROUPS

    if unexpected:
        raise ValueError(
            f"{context}: 예상하지 못한 game_type: {sorted(unexpected)}"
        )

    missing = ALLOWED_GROUPS - values

    if missing:
        raise ValueError(
            f"{context}: game_type 누락: {sorted(missing)}"
        )


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    fold train에서만 HB-EB prior 추정.
    """
    rate = pd.to_numeric(
        df_train[rate_col],
        errors="coerce",
    )

    g_mu = float(rate.mean())
    g_var = float(rate.var())

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
        & np.isfinite(stats["var"])
        & (stats["var"] > 0.0)
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
    train-only prior를 lookup하여 row별 EB rate 생성.
    """
    original_index = df.index

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
        .fillna(global_prior[0])
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup_df["beta"]
        .fillna(global_prior[1])
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

    success_count = n * rate

    denominator = (
        n + alpha + beta
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


def fit_hbeb_priors(df_train):
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
            "prior_df": prior_df,
            "global_prior": global_prior,
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
    transformed = df.copy()
    current_features = list(
        base_features
    )

    for config in HBEB_CONFIGS:
        info = priors[
            config["new_col"]
        ]

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

        # 기존 cumulative success rate 대신 HB-EB rate 사용
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
# Feature preparation
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
        "n": int(len(y_true)),
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


def get_group_metrics(
    df,
    pred,
    group_value,
):
    mask = (
        df[GROUP_COL]
        .astype(str)
        .eq(group_value)
        .to_numpy()
    )

    return calculate_metrics(
        df.loc[
            mask,
            TARGET,
        ].to_numpy(
            dtype=np.float64
        ),
        np.asarray(
            pred,
            dtype=np.float64,
        )[mask],
    )


# ============================================================
# CatBoost fit
# ============================================================

def fit_with_es(
    X_train,
    y_train,
    X_eval,
    y_eval,
    cat_cols,
    model_name,
):
    model = CatBoostRegressor(
        **MODEL_PARAMS
    )

    print()
    print(
        f"[{model_name}]"
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(
            X_eval,
            y_eval,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = float(
        time.time() - start
    )

    best_iteration = (
        model.get_best_iteration()
    )

    n_iterations = int(
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
        n_iterations,
        elapsed,
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
        train[SEASON_COL]
        .isin(train_seasons)
    )

    valid_mask = (
        train[SEASON_COL]
        == validation_season
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
        len(df_train_raw) == 0
        or len(df_valid_raw) == 0
    ):
        raise ValueError(
            f"Fold {fold}: train/validation split이 비었습니다."
        )

    validate_game_types(
        df_train_raw,
        f"Fold {fold} train",
    )

    validate_game_types(
        df_valid_raw,
        f"Fold {fold} validation",
    )

    # --------------------------------------------------------
    # HB-EB: fold train only
    # --------------------------------------------------------

    priors = fit_hbeb_priors(
        df_train_raw
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
        df_valid,
        valid_features,
    ) = apply_hbeb_with_priors(
        df=df_valid_raw,
        base_features=base_features,
        priors=priors,
    )

    if (
        current_features
        != valid_features
    ):
        raise ValueError(
            f"Fold {fold}: feature mismatch"
        )

    X_train = prepare_features(
        df_train[
            current_features
        ],
        cat_cols,
    )

    y_train = (
        df_train[TARGET]
        .astype(np.float64)
    )

    X_valid = prepare_features(
        df_valid[
            current_features
        ],
        cat_cols,
    )

    y_valid = (
        df_valid[TARGET]
        .astype(np.float64)
    )

    valid_groups = (
        df_valid[GROUP_COL]
        .astype(str)
    )

    r_mask = (
        valid_groups
        .eq(R_GROUP)
        .to_numpy()
    )

    f_mask = (
        valid_groups
        .eq(F_GROUP)
        .to_numpy()
    )

    print()
    print("=" * 90)
    print(
        f"Fold {fold} | "
        f"Train {train_seasons[0]}~{train_seasons[-1]} "
        f"-> Valid {validation_season}"
    )
    print(
        f"Train rows={len(df_train):,} | "
        f"Valid rows={len(df_valid):,} | "
        f"R={r_mask.sum():,} | "
        f"F={f_mask.sum():,}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Global
    # train = all R+F
    # eval  = all validation
    # --------------------------------------------------------

    (
        global_model,
        global_iterations,
        global_elapsed,
    ) = fit_with_es(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_valid,
        y_eval=y_valid,
        cat_cols=cat_cols,
        model_name="Global",
    )

    global_pred = np.clip(
        np.asarray(
            global_model.predict(
                X_valid
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 2. R Group-ES
    # train = all R+F
    # eval  = R validation only
    # --------------------------------------------------------

    (
        r_model,
        r_iterations,
        r_elapsed,
    ) = fit_with_es(
        X_train=X_train,
        y_train=y_train,
        X_eval=(
            X_valid.loc[
                r_mask
            ]
        ),
        y_eval=(
            y_valid.loc[
                r_mask
            ]
        ),
        cat_cols=cat_cols,
        model_name="R Group-ES",
    )

    r_group_pred = np.clip(
        np.asarray(
            r_model.predict(
                X_valid.loc[
                    r_mask
                ]
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 3. Base routing
    #
    # R -> R Group-ES
    # F -> Global
    # --------------------------------------------------------

    base_pred = (
        global_pred.copy()
    )

    base_pred[
        r_mask
    ] = r_group_pred

    # F가 global과 완전히 동일한지 확인
    if not np.allclose(
        base_pred[f_mask],
        global_pred[f_mask],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"Fold {fold}: F prediction이 Global과 달라졌습니다."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    global_all = calculate_metrics(
        y_valid,
        global_pred,
    )

    base_all = calculate_metrics(
        y_valid,
        base_pred,
    )

    global_r = get_group_metrics(
        df_valid,
        global_pred,
        R_GROUP,
    )

    base_r = get_group_metrics(
        df_valid,
        base_pred,
        R_GROUP,
    )

    global_f = get_group_metrics(
        df_valid,
        global_pred,
        F_GROUP,
    )

    base_f = get_group_metrics(
        df_valid,
        base_pred,
        F_GROUP,
    )

    # --------------------------------------------------------
    # OOF residual 저장
    # --------------------------------------------------------

    residual = (
        y_valid.to_numpy(
            dtype=np.float64
        )
        - base_pred
    )

    global_residual = (
        y_valid.to_numpy(
            dtype=np.float64
        )
        - global_pred
    )

    pred_df = pd.DataFrame({
        ID: (
            df_valid[ID]
            .to_numpy()
        ),
        "validation_season": (
            validation_season
        ),
        "game_type": (
            valid_groups.to_numpy()
        ),
        "target": (
            y_valid.to_numpy()
        ),
        "global_prediction": (
            global_pred
        ),
        "base_prediction": (
            base_pred
        ),
        "global_residual": (
            global_residual
        ),
        "oof_residual": (
            residual
        ),
    })

    # residual 분석에서 row-local feature를 바로 쓸 수 있도록
    # validation의 원본 feature도 함께 저장
    for col in base_features:
        if (
            col not in pred_df.columns
            and col in df_valid_raw.columns
        ):
            pred_df[col] = (
                df_valid_raw[col]
                .to_numpy()
            )

    fold_result = {
        "fold": fold,
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
            len(df_valid)
        ),
        "n_validation_R": int(
            r_mask.sum()
        ),
        "n_validation_F": int(
            f_mask.sum()
        ),
        "iterations": {
            "global": (
                global_iterations
            ),
            "R_group_es": (
                r_iterations
            ),
        },
        "elapsed_seconds": {
            "global": (
                global_elapsed
            ),
            "R_group_es": (
                r_elapsed
            ),
        },
        "metrics": {
            "global": {
                "overall": (
                    global_all
                ),
                "R": global_r,
                "F": global_f,
            },
            "base": {
                "overall": (
                    base_all
                ),
                "R": base_r,
                "F": base_f,
            },
        },
        "gain": {
            # 양수 = Base 개선
            "overall_brier": float(
                global_all["brier"]
                - base_all["brier"]
            ),
            "R_brier": float(
                global_r["brier"]
                - base_r["brier"]
            ),
            "F_brier": float(
                global_f["brier"]
                - base_f["brier"]
            ),
            "overall_bss": float(
                base_all["bss"]
                - global_all["bss"]
            ),
        },
    }

    print()
    print(
        f"Valid {validation_season} | "
        f"Global={global_all['brier']:.10f} | "
        f"Base={base_all['brier']:.10f} | "
        f"Gain={fold_result['gain']['overall_brier']:+.10f} | "
        f"iters G={global_iterations}, R={r_iterations}"
    )

    print(
        f"R Gain={fold_result['gain']['R_brier']:+.10f} | "
        f"F Gain={fold_result['gain']['F_brier']:+.10f}"
    )

    return (
        pred_df,
        fold_result,
        current_features,
    )


# ============================================================
# Final fixed-iteration model training
# ============================================================

def fit_fixed_model(
    X_train,
    y_train,
    cat_cols,
    iterations,
    model_name,
):
    params = {
        **MODEL_PARAMS,
        "iterations": int(
            iterations
        ),
        # fixed iteration이므로 training log만 유지
        "verbose": 100,
    }

    model = CatBoostRegressor(
        **params
    )

    print()
    print(
        f"[Final {model_name}] "
        f"iterations={iterations}"
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
    )

    elapsed = float(
        time.time() - start
    )

    return (
        model,
        elapsed,
    )


def save_hbeb_priors(
    priors,
    output_dir,
):
    """
    inference에서 재사용할 수 있도록 train-only HB-EB prior 저장.
    """
    prior_meta = {}

    for new_col, info in priors.items():
        filename = (
            f"{new_col}_prior.csv"
        )

        info[
            "prior_df"
        ].to_csv(
            output_dir
            / filename,
            index=False,
            encoding="utf-8-sig",
        )

        prior_meta[
            new_col
        ] = {
            "file": filename,
            "global_prior": [
                float(
                    info[
                        "global_prior"
                    ][0]
                ),
                float(
                    info[
                        "global_prior"
                    ][1]
                ),
            ],
            "group_cols": (
                info[
                    "group_cols"
                ]
            ),
            "n_col": (
                info["n_col"]
            ),
            "rate_col": (
                info["rate_col"]
            ),
        }

    return prior_meta


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
    # Feature schema
    # --------------------------------------------------------

    # test 데이터 자체는 읽지 않고 header만 읽는다.
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        col
        for col in test_cols
        if (
            col != ID
            and col
            not in EXCLUDE_FEATURES
        )
    ]

    if GROUP_COL not in base_features:
        raise ValueError(
            f"{GROUP_COL}이 feature schema에 없습니다."
        )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in base_features
    ]

    train = pd.read_csv(
        TRAIN_PATH,
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
        f"Experiment       : {EXPERIMENT_NAME}"
    )
    print(
        "Base policy      : "
        "R -> R Group-ES / F -> Global"
    )
    print(
        "R Group-ES train : "
        "ALL R+F rows"
    )
    print(
        "F prediction     : "
        "Global unchanged"
    )
    print(
        "Feature Eng.     : HB-EB"
    )
    print(
        f"Excluded         : {EXCLUDE_FEATURES}"
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
    # Expanding Window OOF
    # --------------------------------------------------------

    oof_frames = []
    fold_results = []
    final_features = None

    for fold_config in EXPANDING_FOLDS:
        (
            pred_df,
            fold_result,
            fold_features,
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
                fold_features
            )
        elif (
            final_features
            != fold_features
        ):
            raise ValueError(
                "Fold별 feature 구성이 다릅니다."
            )

        oof_frames.append(
            pred_df
        )

        fold_results.append(
            fold_result
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Pooled metrics
    # --------------------------------------------------------

    pooled_global = calculate_metrics(
        oof["target"],
        oof[
            "global_prediction"
        ],
    )

    pooled_base = calculate_metrics(
        oof["target"],
        oof[
            "base_prediction"
        ],
    )

    pooled_r_mask = (
        oof[GROUP_COL]
        .astype(str)
        .eq(R_GROUP)
        .to_numpy()
    )

    pooled_f_mask = (
        oof[GROUP_COL]
        .astype(str)
        .eq(F_GROUP)
        .to_numpy()
    )

    pooled_global_r = (
        calculate_metrics(
            oof.loc[
                pooled_r_mask,
                "target",
            ],
            oof.loc[
                pooled_r_mask,
                "global_prediction",
            ],
        )
    )

    pooled_base_r = (
        calculate_metrics(
            oof.loc[
                pooled_r_mask,
                "target",
            ],
            oof.loc[
                pooled_r_mask,
                "base_prediction",
            ],
        )
    )

    pooled_global_f = (
        calculate_metrics(
            oof.loc[
                pooled_f_mask,
                "target",
            ],
            oof.loc[
                pooled_f_mask,
                "global_prediction",
            ],
        )
    )

    pooled_base_f = (
        calculate_metrics(
            oof.loc[
                pooled_f_mask,
                "target",
            ],
            oof.loc[
                pooled_f_mask,
                "base_prediction",
            ],
        )
    )

    fold_wins = sum(
        1
        for row in fold_results
        if (
            row["gain"][
                "overall_brier"
            ] > 0.0
        )
    )

    fold_losses = (
        len(fold_results)
        - fold_wins
    )

    # --------------------------------------------------------
    # Save OOF + residual
    # --------------------------------------------------------

    oof.to_csv(
        VALIDATION_DIR
        / "oof_predictions_with_residual.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # R residual만 별도 저장
    oof.loc[
        pooled_r_mask
    ].to_csv(
        VALIDATION_DIR
        / "oof_R_residual.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Final model
    #
    # 기존 cb_reg_v003_test와 동일하게 latest fold의
    # best iteration을 final iteration으로 사용한다.
    # --------------------------------------------------------

    latest_fold = max(
        fold_results,
        key=lambda x: x[
            "validation_season"
        ],
    )

    final_global_iterations = int(
        latest_fold[
            "iterations"
        ]["global"]
    )

    final_r_iterations = int(
        latest_fold[
            "iterations"
        ]["R_group_es"]
    )

    # Full train HB-EB prior
    full_priors = fit_hbeb_priors(
        train
    )

    (
        full_train,
        full_features,
    ) = apply_hbeb_with_priors(
        df=train,
        base_features=(
            base_features
        ),
        priors=(
            full_priors
        ),
    )

    if (
        full_features
        != final_features
    ):
        raise ValueError(
            "OOF / Final feature 불일치"
        )

    X_full = prepare_features(
        full_train[
            final_features
        ],
        cat_cols,
    )

    y_full = (
        full_train[TARGET]
        .astype(np.float64)
    )

    (
        final_global_model,
        final_global_elapsed,
    ) = fit_fixed_model(
        X_train=X_full,
        y_train=y_full,
        cat_cols=cat_cols,
        iterations=(
            final_global_iterations
        ),
        model_name="Global",
    )

    (
        final_r_model,
        final_r_elapsed,
    ) = fit_fixed_model(
        X_train=X_full,
        y_train=y_full,
        cat_cols=cat_cols,
        iterations=(
            final_r_iterations
        ),
        model_name="R Group-ES",
    )

    final_global_model.save_model(
        str(
            MODEL_DIR
            / "global_model.cbm"
        )
    )

    final_r_model.save_model(
        str(
            MODEL_DIR
            / "r_group_es_model.cbm"
        )
    )

    prior_meta = save_hbeb_priors(
        full_priors,
        MODEL_DIR,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    summary = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base_policy": {
            "R": (
                "model trained on all R+F, "
                "early-stopped using R validation rows"
            ),
            "F": (
                "global model trained on all R+F, "
                "early-stopped using full validation"
            ),
        },
        "fold_results": (
            fold_results
        ),
        "pooled": {
            "global": {
                "overall": (
                    pooled_global
                ),
                "R": (
                    pooled_global_r
                ),
                "F": (
                    pooled_global_f
                ),
            },
            "base": {
                "overall": (
                    pooled_base
                ),
                "R": (
                    pooled_base_r
                ),
                "F": (
                    pooled_base_f
                ),
            },
            "gain": {
                "overall_brier": float(
                    pooled_global[
                        "brier"
                    ]
                    - pooled_base[
                        "brier"
                    ]
                ),
                "R_brier": float(
                    pooled_global_r[
                        "brier"
                    ]
                    - pooled_base_r[
                        "brier"
                    ]
                ),
                "F_brier": float(
                    pooled_global_f[
                        "brier"
                    ]
                    - pooled_base_f[
                        "brier"
                    ]
                ),
                "overall_bss": float(
                    pooled_base["bss"]
                    - pooled_global["bss"]
                ),
            },
            "fold_wins": (
                fold_wins
            ),
            "fold_losses": (
                fold_losses
            ),
        },
        "final_iteration_strategy": (
            "latest_fold"
        ),
        "final_iterations": {
            "global": (
                final_global_iterations
            ),
            "R_group_es": (
                final_r_iterations
            ),
        },
        "final_training_seconds": {
            "global": (
                final_global_elapsed
            ),
            "R_group_es": (
                final_r_elapsed
            ),
        },
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
        "hbeb_artifacts": (
            prior_meta
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "residual_definition": (
            "oof_residual = target - base_prediction"
        ),
        "competition_rule_note": (
            "Routing uses only the current row's game_type. "
            "HB-EB priors are fitted on training data only."
        ),
    }

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        MODEL_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    shutil.copy2(
        Path(__file__).resolve(),
        MODEL_DIR
        / "train.py",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "R Group-ES + F Global Base Summary"
    )
    print("=" * 90)

    for row in fold_results:
        print(
            f"Valid {row['validation_season']} | "
            f"Global="
            f"{row['metrics']['global']['overall']['brier']:.10f} | "
            f"Base="
            f"{row['metrics']['base']['overall']['brier']:.10f} | "
            f"Gain="
            f"{row['gain']['overall_brier']:+.10f} | "
            f"R Gain="
            f"{row['gain']['R_brier']:+.10f} | "
            f"iters G="
            f"{row['iterations']['global']}, "
            f"R="
            f"{row['iterations']['R_group_es']}"
        )

    print()
    print(
        "Pooled Global Brier : "
        f"{pooled_global['brier']:.10f}"
    )

    print(
        "Pooled Base Brier   : "
        f"{pooled_base['brier']:.10f}"
    )

    print(
        "Pooled Brier Gain   : "
        f"{summary['pooled']['gain']['overall_brier']:+.10f}"
    )

    print(
        "Pooled BSS Gain     : "
        f"{summary['pooled']['gain']['overall_bss']:+.4f}"
    )

    print(
        "Pooled R Gain       : "
        f"{summary['pooled']['gain']['R_brier']:+.10f}"
    )

    print(
        "Pooled F Gain       : "
        f"{summary['pooled']['gain']['F_brier']:+.10f}"
    )

    print(
        "Fold wins/losses    : "
        f"{fold_wins}/{fold_losses}"
    )

    print()
    print(
        "Final iterations    : "
        f"Global={final_global_iterations}, "
        f"R={final_r_iterations}"
    )

    if (
        final_global_iterations
        == final_r_iterations
    ):
        print(
            "NOTE: latest-fold Global/R iterations가 동일하므로 "
            "현재 final Global/R 모델은 같은 seed/feature/data 기준 "
            "사실상 동일 모델이 됩니다."
        )

    print()
    print(
        "Residual files:"
    )

    print(
        f"- {VALIDATION_DIR / 'oof_predictions_with_residual.csv'}"
    )

    print(
        f"- {VALIDATION_DIR / 'oof_R_residual.csv'}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()

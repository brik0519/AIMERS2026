# train.py
#
# xgb_v001_h1
#
# XGBoostRegressor H1 model for
# CatBoostRegressor + XGBoostRegressor ensemble experiment.
#
# Temporal design
# ------------------------------------------------------------
# Stage 1:
#   2019~2022 -> 2023
#   early stopping으로 XGBoost iteration 결정
#
# Stage 2:
#   2019~2023 + 2024 early 70%
#   고정 iteration으로 XGBoost 재학습
#
# Stage 3:
#   2024 late 30%
#   pure holdout prediction
#
# 2024 H2 target은:
# - preprocessing fit에 사용하지 않음
# - model training에 사용하지 않음
# - iteration selection에 사용하지 않음
# - early stopping에 사용하지 않음
#
# 이후 ensemble_v009/v011 등에서
# CatBoost H1 prediction과 row_id 기준으로 결합하여
# ensemble weight 선택에만 사용한다.


from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBRegressor


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "xgb_v001_h1"

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


# ============================================================
# Columns
# ============================================================

ID = "row_id"
TARGET = "control_success"

SEASON_COL = "season"
TIME_COL = "game_month"


# ============================================================
# XGBoost categorical policy
#
# 현재 70:30 Cat/XGB 실험에서 사용한 정책과 동일:
# low-cardinality categorical만 ordinal encoding.
#
# 나머지 pitcher_id / batter_id 등은 원본 numeric 표현 유지.
# ============================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
]


# ============================================================
# Temporal split
# ============================================================

HOLDOUT_YEAR = 2024

H1_TARGET_SHARE = 0.70

ITERATION_VALIDATION_YEAR = 2023


# ============================================================
# XGBoost
#
# 주의:
# 이 파라미터는 현재 70:30 실험용 설정.
# 별도 tuning 결과가 아니라면 canonical parameter로
# 간주하지 않는다.
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 10.0,

    "subsample": 0.90,
    "colsample_bytree": 0.90,

    "reg_alpha": 0.0,
    "reg_lambda": 10.0,

    "objective": "reg:squarederror",
    "eval_metric": "rmse",

    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 80


# ============================================================
# Metrics
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

    if len(y_true) != len(y_pred):
        raise ValueError(
            "target/prediction length mismatch"
        )

    if not np.all(
        np.isfinite(y_true)
    ):
        raise ValueError(
            "target contains non-finite values"
        )

    if not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError(
            "prediction contains non-finite values"
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
                - brier
                / baseline_brier
            )
        )

    prediction_mean = float(
        np.mean(y_pred)
    )

    prediction_std = float(
        np.std(y_pred)
    )

    bias = float(
        prediction_mean
        - positive_rate
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
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "bias": bias,
    }


# ============================================================
# 2024 70:30 month-block split
# ============================================================

def select_month_cutoff(
    df: pd.DataFrame,
    year: int,
    target_share: float,
):
    """
    동일 game_month 내 row를 쪼개지 않는다.

    month <= cutoff:
        H1

    month > cutoff:
        H2

    H1 row 비율이 target_share=0.70에
    가장 가까워지는 month cutoff를 선택한다.

    target은 split 선택에 사용하지 않는다.
    """

    required = {
        SEASON_COL,
        TIME_COL,
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Temporal split 필수 컬럼 없음: {missing}"
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
            "Temporal split 가능한 month가 부족합니다."
        )

    n_total = int(
        len(year_df)
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

        h1_share = float(
            n_h1
            / n_total
        )

        h2_share = float(
            n_h2
            / n_total
        )

        candidates.append({
            "cutoff_month": int(
                cutoff
            ),
            "n_h1": n_h1,
            "n_h2": n_h2,
            "h1_share": h1_share,
            "h2_share": h2_share,
            "distance_from_target": float(
                abs(
                    h1_share
                    - target_share
                )
            ),
        })

    if not candidates:
        raise RuntimeError(
            "유효한 70:30 month cutoff를 찾지 못했습니다."
        )

    best = min(
        candidates,
        key=lambda x: (
            x[
                "distance_from_target"
            ],
            x[
                "cutoff_month"
            ],
        ),
    )

    return {
        "year": int(year),
        "split_mode": (
            "temporal_month_block"
        ),
        "target_h1_share": float(
            target_share
        ),
        "available_months": months,
        "selected_cutoff_month": int(
            best[
                "cutoff_month"
            ]
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


def build_h1_h2_masks(
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

    h1_mask = (
        (season < year)
        |
        (
            (season == year)
            & (month <= cutoff)
        )
    )

    h2_mask = (
        (season == year)
        & (month > cutoff)
    )

    if not h1_mask.any():
        raise RuntimeError(
            "H1 training set이 비어 있습니다."
        )

    if not h2_mask.any():
        raise RuntimeError(
            "H2 holdout set이 비어 있습니다."
        )

    if np.any(
        h1_mask & h2_mask
    ):
        raise RuntimeError(
            "H1/H2 overlap detected."
        )

    return (
        h1_mask,
        h2_mask,
    )


# ============================================================
# XGBoost preprocessing
# ============================================================

def build_preprocessor(
    features: list[str],
):
    """
    Preprocessor 자체의 모든 통계는 training subset에만 fit.

    categorical:
        missing fill
        -> ordinal encoding

    numeric:
        median imputation

    test/H2 전체 분포는 fit에 사용하지 않는다.
    """

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]

    cat_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown=(
                        "use_encoded_value"
                    ),
                    unknown_value=-1,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                cat_pipeline,
                cat_cols,
            ),
            (
                "num",
                num_pipeline,
                num_cols,
            ),
        ],
        remainder="drop",
    )

    return (
        preprocessor,
        cat_cols,
        num_cols,
    )


def fit_transform_train_apply(
    preprocessor,
    train_features: pd.DataFrame,
    apply_features: pd.DataFrame,
):
    """
    fit:
        training subset만.

    transform:
        training / validation 각각.

    apply dataset의 다른 row를 이용한 통계 생성 없음.
    """

    X_train = (
        preprocessor
        .fit_transform(
            train_features
        )
    )

    X_apply = (
        preprocessor
        .transform(
            apply_features
        )
    )

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )

    X_apply = np.asarray(
        X_apply,
        dtype=np.float32,
    )

    return (
        X_train,
        X_apply,
    )


# ============================================================
# Stage 1
# 2019~2022 -> 2023
# XGBoost iteration selection
# ============================================================

def select_xgb_iterations(
    train: pd.DataFrame,
    features: list[str],
):
    """
    H2를 iteration selection에 사용하지 않는다.

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
        season
        < ITERATION_VALIDATION_YEAR
    )

    validation_mask = (
        season
        == ITERATION_VALIDATION_YEAR
    )

    df_train = train.loc[
        train_mask
    ].copy()

    df_validation = train.loc[
        validation_mask
    ].copy()

    if df_train.empty:
        raise RuntimeError(
            "Iteration selection train이 없습니다."
        )

    if df_validation.empty:
        raise RuntimeError(
            "Iteration selection validation이 없습니다."
        )

    (
        preprocessor,
        cat_cols,
        num_cols,
    ) = build_preprocessor(
        features
    )

    (
        X_train,
        X_validation,
    ) = fit_transform_train_apply(
        preprocessor,
        df_train[features],
        df_validation[features],
    )

    y_train = (
        df_train[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    y_validation = (
        df_validation[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    stage1_params = {
        **MODEL_PARAMS,
        "early_stopping_rounds": int(
            EARLY_STOPPING_ROUNDS
        ),
    }

    model = XGBRegressor(
        **stage1_params
    )

    started = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    elapsed = float(
        time.time()
        - started
    )

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    if best_iteration is None:

        final_iterations = int(
            MODEL_PARAMS[
                "n_estimators"
            ]
        )

        best_iteration_zero = int(
            final_iterations
            - 1
        )

    else:

        best_iteration_zero = int(
            best_iteration
        )

        final_iterations = int(
            best_iteration_zero
            + 1
        )

    pred_raw = np.asarray(
        model.predict(
            X_validation
        ),
        dtype=np.float64,
    )

    pred = np.clip(
        pred_raw,
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_validation,
        pred,
    )

    result = {
        "model_type": (
            "XGBRegressor"
        ),

        "train_seasons": sorted(
            int(x)
            for x in
            df_train[
                SEASON_COL
            ].unique()
        ),

        "validation_season": int(
            ITERATION_VALIDATION_YEAR
        ),

        "n_train": int(
            len(df_train)
        ),

        "n_validation": int(
            len(df_validation)
        ),

        "n_features": int(
            len(features)
        ),

        "categorical_columns": (
            cat_cols
        ),

        "numeric_columns": (
            num_cols
        ),

        "best_iteration_zero_based": int(
            best_iteration_zero
        ),

        "final_iterations": int(
            final_iterations
        ),

        "brier": float(
            metrics[
                "brier"
            ]
        ),

        "bss_raw": float(
            metrics[
                "bss_raw"
            ]
        ),

        "prediction_mean": float(
            metrics[
                "prediction_mean"
            ]
        ),

        "bias": float(
            metrics[
                "bias"
            ]
        ),

        "elapsed_seconds": (
            elapsed
        ),
    }

    return (
        final_iterations,
        result,
    )


# ============================================================
# Stage 2 / 3
# H1 fixed-fit -> H2 pure holdout
# ============================================================

def train_h1_validate_h2(
    train: pd.DataFrame,
    features: list[str],
    final_iterations: int,
):

    # --------------------------------------------------------
    # 70:30 temporal split
    # --------------------------------------------------------

    split_state = (
        select_month_cutoff(
            train,
            year=HOLDOUT_YEAR,
            target_share=H1_TARGET_SHARE,
        )
    )

    (
        h1_mask,
        h2_mask,
    ) = build_h1_h2_masks(
        train,
        split_state,
    )

    df_h1 = train.loc[
        h1_mask
    ].copy()

    df_h2 = train.loc[
        h2_mask
    ].copy()

    cutoff = int(
        split_state[
            "selected_cutoff_month"
        ]
    )

    # --------------------------------------------------------
    # Explicit leakage checks
    # --------------------------------------------------------

    h1_2024 = df_h1.loc[
        pd.to_numeric(
            df_h1[
                SEASON_COL
            ],
            errors="raise",
        ).astype(int)
        == HOLDOUT_YEAR
    ]

    h2_2024 = df_h2.loc[
        pd.to_numeric(
            df_h2[
                SEASON_COL
            ],
            errors="raise",
        ).astype(int)
        == HOLDOUT_YEAR
    ]

    if (
        pd.to_numeric(
            h1_2024[
                TIME_COL
            ],
            errors="raise",
        )
        > cutoff
    ).any():
        raise RuntimeError(
            "2024 H2 row가 H1 training에 유입되었습니다."
        )

    if (
        pd.to_numeric(
            h2_2024[
                TIME_COL
            ],
            errors="raise",
        )
        <= cutoff
    ).any():
        raise RuntimeError(
            "2024 H1 row가 H2에 유입되었습니다."
        )

    # --------------------------------------------------------
    # Preprocessor
    #
    # H1에만 fit.
    # --------------------------------------------------------

    (
        preprocessor,
        cat_cols,
        num_cols,
    ) = build_preprocessor(
        features
    )

    (
        X_h1,
        X_h2,
    ) = fit_transform_train_apply(
        preprocessor,
        df_h1[features],
        df_h2[features],
    )

    y_h1 = (
        df_h1[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    y_h2 = (
        df_h2[TARGET]
        .astype(np.float64)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Fixed-iteration final H1 model
    #
    # H2 target을 eval_set으로 사용하지 않는다.
    # --------------------------------------------------------

    final_params = {
        **MODEL_PARAMS,
        "n_estimators": int(
            final_iterations
        ),
    }

    model = XGBRegressor(
        **final_params
    )

    started = time.time()

    model.fit(
        X_h1,
        y_h1,
        verbose=False,
    )

    elapsed = float(
        time.time()
        - started
    )

    # --------------------------------------------------------
    # Pure H2 prediction
    # --------------------------------------------------------

    pred_raw = np.asarray(
        model.predict(
            X_h2
        ),
        dtype=np.float64,
    )

    pred = np.clip(
        pred_raw,
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_h2,
        pred,
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    pred_df = pd.DataFrame({
        ID: (
            df_h2[ID]
            .to_numpy()
        ),

        "target": (
            y_h2
        ),

        "prediction_raw": (
            pred_raw
        ),

        "prediction": (
            pred
        ),

        "validation_season": (
            HOLDOUT_YEAR
        ),

        "validation_segment": (
            "2024_H2_30"
        ),
    })

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            "XGBRegressor"
        ),

        "role": (
            "2024 70:30 H1/H2 "
            "ensemble weight estimation"
        ),

        "model": (
            model
        ),

        "preprocessor": (
            preprocessor
        ),

        "features": list(
            features
        ),

        "cat_cols": (
            cat_cols
        ),

        "num_cols": (
            num_cols
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
            "np.clip("
            "model.predict(preprocessor.transform(X)), "
            "0.0, 1.0)"
        ),

        "training_policy": (
            "2019~2023 + 2024 H1(70%)"
        ),

        "validation_policy": (
            "2024 H2(30%) pure holdout"
        ),

        "rule_compliance": {
            "test_values_used": False,
            "test_distribution_used": False,
            "cross_test_aggregation": False,
            "h2_target_used_for_training": False,
            "h2_target_used_for_preprocessing_fit": False,
            "h2_target_used_for_iteration_selection": False,
            "h2_target_used_for_early_stopping": False,
            "row_independent_test_transform": True,
        },
    }

    # --------------------------------------------------------
    # Result metadata
    # --------------------------------------------------------

    result = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            "XGBRegressor"
        ),

        "split_state": (
            split_state
        ),

        "n_train": int(
            len(df_h1)
        ),

        "n_validation": int(
            len(df_h2)
        ),

        "training_seasons": sorted(
            int(x)
            for x in
            pd.to_numeric(
                df_h1[
                    SEASON_COL
                ],
                errors="raise",
            ).unique()
        ),

        "validation_season": (
            HOLDOUT_YEAR
        ),

        "validation_segment": (
            "2024_H2_30"
        ),

        "n_features": int(
            len(features)
        ),

        "categorical_columns": (
            cat_cols
        ),

        "numeric_columns": (
            num_cols
        ),

        "final_iterations": int(
            final_iterations
        ),

        "brier": float(
            metrics[
                "brier"
            ]
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
            metrics[
                "bss_raw"
            ]
        ),

        "bss": float(
            metrics[
                "bss"
            ]
        ),

        "prediction_mean": float(
            metrics[
                "prediction_mean"
            ]
        ),

        "prediction_std": float(
            metrics[
                "prediction_std"
            ]
        ),

        "bias": float(
            metrics[
                "bias"
            ]
        ),

        "elapsed_seconds": (
            elapsed
        ),

        "model_params": (
            final_params
        ),

        "rule_compliance": (
            artifact[
                "rule_compliance"
            ]
        ),
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
    # File safety
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
    # Feature schema
    #
    # test 값은 읽지 않고 header만 확인.
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

    features = [
        col
        for col in test_cols
        if col != ID
    ]

    # --------------------------------------------------------
    # Train load
    # --------------------------------------------------------

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *features,
            TARGET,
        ],
    )

    required = {
        ID,
        TARGET,
        SEASON_COL,
        TIME_COL,
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

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 88)
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )
    print(
        "Model      : XGBRegressor"
    )
    print(
        "Objective  : reg:squarederror"
    )
    print(
        "Validation : 2024 temporal 70:30"
    )
    print(
        "Iteration  : 2019~2022 -> 2023"
    )
    print(
        "H2 early stop: OFF"
    )
    print(
        f"Features   : {len(features)}"
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
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Stage 1: XGBoost iteration selection"
    )
    print(
        "2019~2022 -> 2023"
    )
    print("=" * 88)

    (
        final_iterations,
        iteration_result,
    ) = select_xgb_iterations(
        train=train,
        features=features,
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
        "2023 bias        :",
        f"{iteration_result['bias']:+.8f}",
    )

    print(
        "Selected trees   :",
        final_iterations,
    )

    # ========================================================
    # Stage 2 / 3
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Stage 2: 2024 H1(70%) training"
    )
    print(
        "Stage 3: 2024 H2(30%) pure holdout"
    )
    print("=" * 88)

    (
        artifact,
        pred_df,
        holdout_result,
    ) = train_h1_validate_h2(
        train=train,
        features=features,
        final_iterations=(
            final_iterations
        ),
    )

    # --------------------------------------------------------
    # Save artifact
    # --------------------------------------------------------

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            "XGBRegressor"
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

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR
            / "train.py",
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
        f"{EXPERIMENT_NAME} Result"
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
        f"({split_state['h1_share']:.4%})",
    )

    print(
        "2024 H2 rows     :",
        f"{split_state['n_h2']:,}",
        f"({split_state['h2_share']:.4%})",
    )

    print("-" * 88)

    print(
        "Final iterations :",
        final_iterations,
    )

    print(
        "H2 Target mean   :",
        f"{holdout_result['positive_rate']:.8f}",
    )

    print(
        "H2 Pred mean     :",
        f"{holdout_result['prediction_mean']:.8f}",
    )

    print(
        "H2 Bias          :",
        f"{holdout_result['bias']:+.8f}",
    )

    print("-" * 88)

    print(
        "H2 Brier         :",
        f"{holdout_result['brier']:.10f}",
    )

    print(
        "H2 BSS raw       :",
        f"{holdout_result['bss_raw']:.2f}",
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
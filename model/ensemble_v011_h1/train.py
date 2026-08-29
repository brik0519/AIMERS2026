# train.py
#
# ensemble_v011_h1
#
# CatBoostRegressor + XGBRegressor
# 2024 temporal month-block 70:30 experiment
#
# 핵심 변경:
# - 2023을 iteration selector에서 제거
# - 2023은 training data에는 그대로 포함
#
# Outer:
#   H1 = 2019~2023 + 2024 early 70%
#   H2 = 2024 late 30% (pure holdout)
#
# Inner iteration selection:
#   Train = 2019~2023 + 2024 H1의 앞부분
#   Valid = 2024 H1의 마지막 2개 month block
#
# 현재 데이터에서는:
#   Inner Train = 2019~2023 + 2024 Mar~May
#   Inner Valid = 2024 Jun~Jul
#   Outer H2    = 2024 Aug~Oct
#
# H2 target은
# - preprocessing fit
# - base model training
# - iteration selection
# - early stopping
# 에 사용하지 않는다.
#
# H2 target은 ensemble weight 선택에만 사용한다.


from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBRegressor


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "ensemble_v011_h1"

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
# 기존 baseline categorical 정책 유지
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

# Outer H1 목표 비율
H1_TARGET_SHARE = 0.70

# H1의 마지막 몇 개 month block을
# iteration-selection validation으로 사용할 것인가
INNER_VALIDATION_MONTH_COUNT = 2


# ============================================================
# CatBoost
# ============================================================

CAT_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

CAT_EARLY_STOPPING_ROUNDS = 80


# ============================================================
# XGBoost
# ============================================================

XGB_PARAMS = {
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

XGB_EARLY_STOPPING_ROUNDS = 80


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

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "target contains non-finite values"
        )

    if not np.all(np.isfinite(y_pred)):
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


def safe_corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if len(a) < 2:
        return float("nan")

    if (
        np.std(a) <= 1e-15
        or np.std(b) <= 1e-15
    ):
        return float("nan")

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


# ============================================================
# Outer 70:30 split
# ============================================================

def select_outer_month_cutoff(
    df: pd.DataFrame,
    year: int,
    target_share: float,
):
    """
    game_month block을 유지하면서
    H1 row 비율이 70%에 가장 가까워지는
    cutoff를 선택한다.

    H1:
        month <= cutoff

    H2:
        month > cutoff
    """

    season = pd.to_numeric(
        df[SEASON_COL],
        errors="raise",
    ).astype(int)

    year_df = df.loc[
        season == year,
        [TIME_COL],
    ].copy()

    if year_df.empty:
        raise ValueError(
            f"{year} data not found"
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
        raise RuntimeError(
            "not enough month blocks"
        )

    total = int(
        len(year_df)
    )

    candidates = []

    for cutoff in months[:-1]:

        n_h1 = int(
            (month <= cutoff).sum()
        )

        n_h2 = int(
            (month > cutoff).sum()
        )

        if n_h1 == 0 or n_h2 == 0:
            continue

        h1_share = float(
            n_h1 / total
        )

        candidates.append({
            "cutoff_month": int(
                cutoff
            ),
            "n_h1": n_h1,
            "n_h2": n_h2,
            "h1_share": h1_share,
            "h2_share": float(
                n_h2 / total
            ),
            "distance_from_target": float(
                abs(
                    h1_share
                    - target_share
                )
            ),
        })

    if not candidates:
        raise RuntimeError(
            "valid outer cutoff not found"
        )

    best = min(
        candidates,
        key=lambda x: (
            x["distance_from_target"],
            x["cutoff_month"],
        ),
    )

    return {
        "year": int(year),
        "target_h1_share": float(
            target_share
        ),
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


def build_outer_masks(
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

    if np.any(
        h1_mask & h2_mask
    ):
        raise RuntimeError(
            "outer H1/H2 overlap"
        )

    return (
        h1_mask,
        h2_mask,
    )


# ============================================================
# Inner iteration-selection split
# ============================================================

def build_inner_split_state(
    df: pd.DataFrame,
    outer_split_state: dict,
):
    """
    Outer H1의 마지막 N개 month block을
    iteration-selection validation으로 사용.

    현재 데이터에서는:

        Outer H1:
            Mar Apr May Jun Jul

        Inner validation:
            Jun Jul

        Inner training:
            2019~2023
            + Mar Apr May

    2023은 validation이 아니라 training에 포함된다.
    """

    year = int(
        outer_split_state["year"]
    )

    outer_cutoff = int(
        outer_split_state[
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

    h1_months = sorted(
        int(x)
        for x in
        month.loc[
            (season == year)
            & (month <= outer_cutoff)
        ].unique()
    )

    if (
        len(h1_months)
        <= INNER_VALIDATION_MONTH_COUNT
    ):
        raise RuntimeError(
            "Outer H1 month가 inner split하기에 부족합니다."
        )

    validation_months = (
        h1_months[
            -INNER_VALIDATION_MONTH_COUNT:
        ]
    )

    training_months = (
        h1_months[
            :-INNER_VALIDATION_MONTH_COUNT
        ]
    )

    return {
        "year": year,
        "outer_cutoff_month": (
            outer_cutoff
        ),
        "h1_months": h1_months,
        "inner_training_months": (
            training_months
        ),
        "inner_validation_months": (
            validation_months
        ),
    }


def build_inner_masks(
    df: pd.DataFrame,
    inner_state: dict,
):
    """
    2023은 모두 inner training에 포함된다.

    Validation은 2024 H1 내부의
    마지막 month block들뿐이다.
    """

    year = int(
        inner_state["year"]
    )

    train_months = set(
        inner_state[
            "inner_training_months"
        ]
    )

    validation_months = set(
        inner_state[
            "inner_validation_months"
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
            & month.isin(
                train_months
            )
        )
    )

    validation_mask = (
        (season == year)
        & month.isin(
            validation_months
        )
    )

    if np.any(
        train_mask
        & validation_mask
    ):
        raise RuntimeError(
            "inner train/validation overlap"
        )

    if not train_mask.any():
        raise RuntimeError(
            "inner train empty"
        )

    if not validation_mask.any():
        raise RuntimeError(
            "inner validation empty"
        )

    return (
        train_mask,
        validation_mask,
    )


# ============================================================
# CatBoost preprocessing
# ============================================================

def prepare_catboost(
    df: pd.DataFrame,
):

    out = df.copy()

    for col in CAT_COLS:

        if col in out.columns:

            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


# ============================================================
# XGBoost preprocessing
# ============================================================

def build_xgb_preprocessor(
    features: list[str],
):

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


def fit_transform_xgb(
    preprocessor,
    X_train,
    X_apply,
):

    X_train_t = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    X_apply_t = (
        preprocessor
        .transform(
            X_apply
        )
    )

    return (
        np.asarray(
            X_train_t,
            dtype=np.float32,
        ),
        np.asarray(
            X_apply_t,
            dtype=np.float32,
        ),
    )


# ============================================================
# CatBoost iteration selection
# ============================================================

def select_cat_iterations(
    train: pd.DataFrame,
    features: list[str],
    inner_state: dict,
):

    (
        train_mask,
        validation_mask,
    ) = build_inner_masks(
        train,
        inner_state,
    )

    df_train = train.loc[
        train_mask
    ].copy()

    df_validation = train.loc[
        validation_mask
    ].copy()

    X_train = prepare_catboost(
        df_train[features]
    )

    X_validation = prepare_catboost(
        df_validation[features]
    )

    y_train = (
        df_train[TARGET]
        .astype(np.float64)
    )

    y_validation = (
        df_validation[TARGET]
        .astype(np.float64)
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    model = CatBoostRegressor(
        **CAT_PARAMS
    )

    started = time.time()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(
            X_validation,
            y_validation,
        ),
        early_stopping_rounds=(
            CAT_EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = float(
        time.time()
        - started
    )

    best_iteration = int(
        model.get_best_iteration()
    )

    if best_iteration >= 0:

        final_iterations = int(
            best_iteration
            + 1
        )

    else:

        final_iterations = int(
            model.tree_count_
        )

    pred = np.clip(
        np.asarray(
            model.predict(
                X_validation
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_validation,
        pred,
    )

    result = {
        "model": (
            "CatBoostRegressor"
        ),

        "selection_method": (
            "2024_H1_inner_temporal"
        ),

        "inner_training_months": (
            inner_state[
                "inner_training_months"
            ]
        ),

        "inner_validation_months": (
            inner_state[
                "inner_validation_months"
            ]
        ),

        "n_train": int(
            len(df_train)
        ),

        "n_validation": int(
            len(df_validation)
        ),

        "final_iterations": int(
            final_iterations
        ),

        "best_iteration_zero_based": int(
            best_iteration
        ),

        "brier": float(
            metrics["brier"]
        ),

        "bss_raw": float(
            metrics["bss_raw"]
        ),

        "bias": float(
            metrics["bias"]
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
# XGBoost iteration selection
# ============================================================

def select_xgb_iterations(
    train: pd.DataFrame,
    features: list[str],
    inner_state: dict,
):

    (
        train_mask,
        validation_mask,
    ) = build_inner_masks(
        train,
        inner_state,
    )

    df_train = train.loc[
        train_mask
    ].copy()

    df_validation = train.loc[
        validation_mask
    ].copy()

    (
        preprocessor,
        _,
        _,
    ) = build_xgb_preprocessor(
        features
    )

    (
        X_train,
        X_validation,
    ) = fit_transform_xgb(
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

    params = {
        **XGB_PARAMS,
        "early_stopping_rounds": int(
            XGB_EARLY_STOPPING_ROUNDS
        ),
    }

    model = XGBRegressor(
        **params
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
            XGB_PARAMS[
                "n_estimators"
            ]
        )

        best_iteration_zero = (
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

    pred = np.clip(
        np.asarray(
            model.predict(
                X_validation
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_validation,
        pred,
    )

    result = {
        "model": (
            "XGBRegressor"
        ),

        "selection_method": (
            "2024_H1_inner_temporal"
        ),

        "inner_training_months": (
            inner_state[
                "inner_training_months"
            ]
        ),

        "inner_validation_months": (
            inner_state[
                "inner_validation_months"
            ]
        ),

        "n_train": int(
            len(df_train)
        ),

        "n_validation": int(
            len(df_validation)
        ),

        "final_iterations": int(
            final_iterations
        ),

        "best_iteration_zero_based": int(
            best_iteration_zero
        ),

        "brier": float(
            metrics["brier"]
        ),

        "bss_raw": float(
            metrics["bss_raw"]
        ),

        "bias": float(
            metrics["bias"]
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
# Final H1 CatBoost fit
# ============================================================

def train_cat_h1(
    df_h1,
    df_h2,
    features,
    final_iterations,
):

    X_h1 = prepare_catboost(
        df_h1[features]
    )

    X_h2 = prepare_catboost(
        df_h2[features]
    )

    y_h1 = (
        df_h1[TARGET]
        .astype(np.float64)
    )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in features
    ]

    params = {
        **CAT_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    model = CatBoostRegressor(
        **params
    )

    started = time.time()

    # H2를 eval_set으로 사용하지 않는다.
    model.fit(
        X_h1,
        y_h1,
        cat_features=cat_cols,
    )

    elapsed = float(
        time.time()
        - started
    )

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

    return (
        model,
        pred_raw,
        pred,
        elapsed,
    )


# ============================================================
# Final H1 XGBoost fit
# ============================================================

def train_xgb_h1(
    df_h1,
    df_h2,
    features,
    final_iterations,
):

    (
        preprocessor,
        cat_cols,
        num_cols,
    ) = build_xgb_preprocessor(
        features
    )

    (
        X_h1,
        X_h2,
    ) = fit_transform_xgb(
        preprocessor,
        df_h1[features],
        df_h2[features],
    )

    y_h1 = (
        df_h1[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    params = {
        **XGB_PARAMS,
        "n_estimators": int(
            final_iterations
        ),
    }

    model = XGBRegressor(
        **params
    )

    started = time.time()

    # H2 target을 전달하지 않는다.
    model.fit(
        X_h1,
        y_h1,
        verbose=False,
    )

    elapsed = float(
        time.time()
        - started
    )

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

    preprocessing_artifact = {
        "preprocessor": (
            preprocessor
        ),
        "cat_cols": (
            cat_cols
        ),
        "num_cols": (
            num_cols
        ),
    }

    return (
        model,
        preprocessing_artifact,
        pred_raw,
        pred,
        elapsed,
    )


# ============================================================
# Ensemble
# ============================================================

def optimal_cat_weight(
    y_true,
    cat_pred,
    xgb_pred,
):
    """
    p =
        w * Cat
        +
        (1-w) * XGB

    Brier-optimal closed-form convex weight.
    """

    y = np.asarray(
        y_true,
        dtype=np.float64,
    )

    cat = np.asarray(
        cat_pred,
        dtype=np.float64,
    )

    xgb = np.asarray(
        xgb_pred,
        dtype=np.float64,
    )

    d = (
        cat - xgb
    )

    denominator = float(
        np.dot(d, d)
    )

    if denominator <= 1e-15:
        return 0.5

    numerator = float(
        np.dot(
            d,
            y - xgb,
        )
    )

    return float(
        np.clip(
            numerator
            / denominator,
            0.0,
            1.0,
        )
    )


def blend_predictions(
    cat_pred,
    xgb_pred,
    cat_weight,
):

    return (
        cat_weight
        * np.asarray(
            cat_pred,
            dtype=np.float64,
        )
        +
        (
            1.0
            - cat_weight
        )
        * np.asarray(
            xgb_pred,
            dtype=np.float64,
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Safety
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
            f"기존 model 보존: {MODEL_DIR}"
        )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보존: {VALIDATION_DIR}"
        )

    # --------------------------------------------------------
    # Test schema only
    # --------------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise KeyError(
            f"test schema에 {ID} 없음"
        )

    features = [
        c
        for c in test_cols
        if c != ID
    ]

    # --------------------------------------------------------
    # Train
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
            f"필수 컬럼 없음: {missing}"
        )

    # --------------------------------------------------------
    # Outer split
    # --------------------------------------------------------

    outer_state = (
        select_outer_month_cutoff(
            train,
            HOLDOUT_YEAR,
            H1_TARGET_SHARE,
        )
    )

    (
        h1_mask,
        h2_mask,
    ) = build_outer_masks(
        train,
        outer_state,
    )

    df_h1 = train.loc[
        h1_mask
    ].copy()

    df_h2 = train.loc[
        h2_mask
    ].copy()

    # --------------------------------------------------------
    # Inner split
    # --------------------------------------------------------

    inner_state = (
        build_inner_split_state(
            train,
            outer_state,
        )
    )

    # --------------------------------------------------------
    # Explicit assertions
    # --------------------------------------------------------

    inner_train_mask, inner_val_mask = (
        build_inner_masks(
            train,
            inner_state,
        )
    )

    # Inner iteration selector에
    # Outer H2 row가 절대 들어가면 안 된다.
    if np.any(
        (
            inner_train_mask
            | inner_val_mask
        )
        & h2_mask
    ):
        raise RuntimeError(
            "Outer H2 leaked into iteration selection."
        )

    # 2023은 validation에 존재하면 안 된다.
    inner_val_seasons = set(
        pd.to_numeric(
            train.loc[
                inner_val_mask,
                SEASON_COL,
            ],
            errors="raise",
        ).astype(int)
    )

    if 2023 in inner_val_seasons:
        raise RuntimeError(
            "2023 is still being used as iteration selector."
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 88)

    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        "Models     : CatBoostRegressor + XGBRegressor"
    )

    print(
        "Outer      : 2024 temporal 70:30"
    )

    print(
        "Iteration  : 2024 H1 internal temporal validation"
    )

    print(
        "2023 selector: OFF"
    )

    print(
        "H2 early stopping: OFF"
    )

    print(
        f"Features   : {len(features)}"
    )

    print("=" * 88)

    print(
        "Outer months       :",
        outer_state[
            "available_months"
        ],
    )

    print(
        "Outer cutoff       :",
        outer_state[
            "selected_cutoff_month"
        ],
    )

    print(
        "H1 rows            :",
        f"{outer_state['n_h1']:,}",
        f"({outer_state['h1_share']:.4%})",
    )

    print(
        "H2 rows            :",
        f"{outer_state['n_h2']:,}",
        f"({outer_state['h2_share']:.4%})",
    )

    print(
        "Inner train months :",
        inner_state[
            "inner_training_months"
        ],
    )

    print(
        "Inner valid months :",
        inner_state[
            "inner_validation_months"
        ],
    )

    # --------------------------------------------------------
    # Directories
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
        "Stage 1: iteration selection"
    )
    print(
        "2023 selector = OFF"
    )
    print("=" * 88)

    (
        cat_iterations,
        cat_iteration_meta,
    ) = select_cat_iterations(
        train,
        features,
        inner_state,
    )

    (
        xgb_iterations,
        xgb_iteration_meta,
    ) = select_xgb_iterations(
        train,
        features,
        inner_state,
    )

    print(
        "Cat trees      :",
        cat_iterations,
    )

    print(
        "Cat Inner BSS  :",
        f"{cat_iteration_meta['bss_raw']:.2f}",
    )

    print(
        "Cat Inner bias :",
        f"{cat_iteration_meta['bias']:+.8f}",
    )

    print(
        "XGB trees      :",
        xgb_iterations,
    )

    print(
        "XGB Inner BSS  :",
        f"{xgb_iteration_meta['bss_raw']:.2f}",
    )

    print(
        "XGB Inner bias :",
        f"{xgb_iteration_meta['bias']:+.8f}",
    )

    # ========================================================
    # Stage 2
    # Full H1 fixed-fit
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Stage 2: Full H1 fixed-iteration training"
    )
    print("=" * 88)

    (
        cat_model,
        cat_pred_raw,
        cat_pred,
        cat_elapsed,
    ) = train_cat_h1(
        df_h1,
        df_h2,
        features,
        cat_iterations,
    )

    (
        xgb_model,
        xgb_preprocessing,
        xgb_pred_raw,
        xgb_pred,
        xgb_elapsed,
    ) = train_xgb_h1(
        df_h1,
        df_h2,
        features,
        xgb_iterations,
    )

    # ========================================================
    # Stage 3
    # H2 pure holdout
    # ========================================================

    y_h2 = (
        df_h2[TARGET]
        .astype(np.float64)
        .to_numpy()
    )

    cat_metrics = (
        calculate_metrics(
            y_h2,
            cat_pred,
        )
    )

    xgb_metrics = (
        calculate_metrics(
            y_h2,
            xgb_pred,
        )
    )

    # ========================================================
    # Stage 4
    # Ensemble weight
    # ========================================================

    cat_weight = (
        optimal_cat_weight(
            y_h2,
            cat_pred,
            xgb_pred,
        )
    )

    xgb_weight = (
        1.0
        - cat_weight
    )

    ensemble_pred = (
        blend_predictions(
            cat_pred,
            xgb_pred,
            cat_weight,
        )
    )

    ensemble_metrics = (
        calculate_metrics(
            y_h2,
            ensemble_pred,
        )
    )

    equal_pred = (
        blend_predictions(
            cat_pred,
            xgb_pred,
            0.5,
        )
    )

    equal_metrics = (
        calculate_metrics(
            y_h2,
            equal_pred,
        )
    )

    prediction_corr = safe_corr(
        cat_pred,
        xgb_pred,
    )

    residual_corr = safe_corr(
        y_h2 - cat_pred,
        y_h2 - xgb_pred,
    )

    best_base_brier = min(
        cat_metrics["brier"],
        xgb_metrics["brier"],
    )

    blend_gain = float(
        best_base_brier
        - ensemble_metrics[
            "brier"
        ]
    )

    # ========================================================
    # OOF
    # ========================================================

    pred_df = pd.DataFrame({
        ID: (
            df_h2[ID]
            .to_numpy()
        ),

        "target": (
            y_h2
        ),

        "cat_prediction_raw": (
            cat_pred_raw
        ),

        "cat_prediction": (
            cat_pred
        ),

        "xgb_prediction_raw": (
            xgb_pred_raw
        ),

        "xgb_prediction": (
            xgb_pred
        ),

        "ensemble_prediction": (
            ensemble_pred
        ),

        "validation_season": (
            HOLDOUT_YEAR
        ),

        "validation_segment": (
            "2024_H2_30"
        ),
    })

    pred_df.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Artifact
    # ========================================================

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "features": (
            features
        ),

        "outer_split": (
            outer_state
        ),

        "inner_iteration_split": (
            inner_state
        ),

        "catboost": {
            "model": cat_model,
            "iterations": int(
                cat_iterations
            ),
        },

        "xgboost": {
            "model": xgb_model,
            "preprocessing": (
                xgb_preprocessing
            ),
            "iterations": int(
                xgb_iterations
            ),
        },

        "ensemble": {
            "cat_weight": float(
                cat_weight
            ),
            "xgb_weight": float(
                xgb_weight
            ),
        },

        "rule_compliance": {
            "test_values_used": False,
            "cross_test_aggregation": False,
            "2023_used_as_iteration_selector": False,
            "2023_used_as_training_data": True,
            "h2_target_used_for_training": False,
            "h2_target_used_for_iteration_selection": False,
            "h2_target_used_for_early_stopping": False,
            "h2_target_used_for_ensemble_weight": True,
        },
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "validation_method": (
            "2024 outer 70:30 + "
            "H1 internal temporal iteration selection"
        ),

        "outer_split": (
            outer_state
        ),

        "inner_iteration_split": (
            inner_state
        ),

        "iteration_selection": {
            "catboost": (
                cat_iteration_meta
            ),

            "xgboost": (
                xgb_iteration_meta
            ),
        },

        "h2": {
            "target_mean": float(
                np.mean(y_h2)
            ),

            "catboost": (
                cat_metrics
            ),

            "xgboost": (
                xgb_metrics
            ),

            "equal_ensemble": (
                equal_metrics
            ),

            "optimal_ensemble": (
                ensemble_metrics
            ),

            "cat_weight": float(
                cat_weight
            ),

            "xgb_weight": float(
                xgb_weight
            ),

            "prediction_corr": float(
                prediction_corr
            ),

            "residual_corr": float(
                residual_corr
            ),

            "blend_gain_brier": (
                blend_gain
            ),
        },

        "rule_compliance": (
            artifact[
                "rule_compliance"
            ]
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
            MODEL_DIR / "train.py",
        )

    except Exception:
        pass

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 88)

    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print("=" * 88)

    print(
        "Inner train months:",
        inner_state[
            "inner_training_months"
        ],
    )

    print(
        "Inner valid months:",
        inner_state[
            "inner_validation_months"
        ],
    )

    print(
        "Outer H2 months  :",
        [
            m
            for m in outer_state[
                "available_months"
            ]
            if m
            > outer_state[
                "selected_cutoff_month"
            ]
        ],
    )

    print("-" * 88)

    print(
        "Cat trees  :",
        cat_iterations,
    )

    print(
        "XGB trees  :",
        xgb_iterations,
    )

    print("-" * 88)

    print(
        "Target mean:",
        f"{np.mean(y_h2):.8f}",
    )

    print()

    print(
        "Cat Brier  :",
        f"{cat_metrics['brier']:.10f}",
    )

    print(
        "Cat BSS    :",
        f"{cat_metrics['bss_raw']:.2f}",
    )

    print(
        "Cat bias   :",
        f"{cat_metrics['bias']:+.8f}",
    )

    print()

    print(
        "XGB Brier  :",
        f"{xgb_metrics['brier']:.10f}",
    )

    print(
        "XGB BSS    :",
        f"{xgb_metrics['bss_raw']:.2f}",
    )

    print(
        "XGB bias   :",
        f"{xgb_metrics['bias']:+.8f}",
    )

    print("-" * 88)

    print(
        "Equal Brier:",
        f"{equal_metrics['brier']:.10f}",
    )

    print(
        "Equal BSS  :",
        f"{equal_metrics['bss_raw']:.2f}",
    )

    print("-" * 88)

    print(
        "Opt Brier  :",
        f"{ensemble_metrics['brier']:.10f}",
    )

    print(
        "Opt BSS    :",
        f"{ensemble_metrics['bss_raw']:.2f}",
    )

    print(
        "Blend Gain :",
        f"{blend_gain:+.10f}",
    )

    print("-" * 88)

    print("Final weight:")

    print(
        "  CatBoost :",
        f"{cat_weight:.8f}",
    )

    print(
        "  XGBoost  :",
        f"{xgb_weight:.8f}",
    )

    print("-" * 88)

    print(
        "Prediction Corr:",
        f"{prediction_corr:.8f}",
    )

    print(
        "Residual Corr  :",
        f"{residual_corr:.8f}",
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
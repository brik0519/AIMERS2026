# train.py
#
# ensemble_v011_h1
#
# CatBoostRegressor + XGBoostRegressor
# 2024 temporal 70:30 H1/H2 experiment
#
# Provided baseline train.py의 47-feature schema를 유지한다.
#
# Stage 1
#   2019~2022 -> 2023
#   CatBoost / XGBoost iteration 선택
#
# Stage 2
#   2019~2023 + 2024 early 70%
#   fixed iteration으로 두 base model 재학습
#
# Stage 3
#   2024 late 30%
#   pure holdout prediction
#
# Stage 4
#   2024 late 30% target으로
#   Cat/XGB Brier-optimal convex weight 선택
#
# 주의:
# - 이 artifact는 2025 최종 추론 모델이 아니라
#   ensemble weight를 결정하기 위한 H1/H2 실험용이다.
# - 최종 제출 시에는 weight를 freeze하고
#   두 base model을 2019~2024 전체로 다시 학습해야 한다.
#
# Rule compliance:
# - test.csv는 schema(header)만 읽는다.
# - test 값/분포/다른 test row를 사용하지 않는다.
# - 2024 H2 target은 base model training / preprocessing /
#   iteration selection / early stopping에 사용하지 않는다.
# - H2 target은 최종 ensemble weight 선택에만 사용한다.


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
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "ensemble_v010"

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


# 제공된 baseline train.py와 동일
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
#
# 주의:
# 제공된 원본 train.py에는 XGBoost 설정이 없으므로
# 아래 설정은 이번 실험용 신규 설정이다.
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
                - brier / baseline_brier
            )
        )

    return {
        "brier": brier,
        "positive_rate": positive_rate,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": max(0.0, bss_raw),
        "prediction_mean": float(
            np.mean(y_pred)
        ),
        "prediction_std": float(
            np.std(y_pred)
        ),
        "bias": float(
            np.mean(y_pred)
            - positive_rate
        ),
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
        np.corrcoef(a, b)[0, 1]
    )


# ============================================================
# 2024 70:30 month-block split
# ============================================================

def select_month_cutoff(
    df: pd.DataFrame,
    year: int,
    target_share: float,
):
    """
    month block을 깨지 않으면서
    H1 row share가 target_share에 가장 가까운
    cutoff month를 선택한다.

    H1:
        month <= cutoff

    H2:
        month > cutoff

    target은 cutoff 선택에 사용하지 않는다.
    """

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
            f"{year} season not found"
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
            "not enough months for temporal split"
        )

    total = int(len(year_df))

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
            "cutoff_month": int(cutoff),
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
            "valid month cutoff not found"
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
            "H1 training set empty"
        )

    if not h2_mask.any():
        raise RuntimeError(
            "H2 validation set empty"
        )

    if np.any(
        h1_mask & h2_mask
    ):
        raise RuntimeError(
            "H1/H2 overlap detected"
        )

    return h1_mask, h2_mask


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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                ),
                cat_cols,
            ),
            (
                "num",
                SimpleImputer(
                    strategy="median",
                ),
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
    X_train: pd.DataFrame,
    X_apply: pd.DataFrame,
):
    X_train_t = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_apply_t = (
        preprocessor.transform(
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
# Stage 1-A
# CatBoost iteration selection
# ============================================================

def select_cat_iterations(
    train: pd.DataFrame,
    features: list[str],
):
    season = pd.to_numeric(
        train[SEASON_COL],
        errors="raise",
    ).astype(int)

    fit_mask = (
        season
        < ITERATION_VALIDATION_YEAR
    )

    val_mask = (
        season
        == ITERATION_VALIDATION_YEAR
    )

    df_fit = train.loc[
        fit_mask
    ].copy()

    df_val = train.loc[
        val_mask
    ].copy()

    X_fit = prepare_catboost(
        df_fit[features]
    )

    X_val = prepare_catboost(
        df_val[features]
    )

    y_fit = (
        df_fit[TARGET]
        .astype(np.float64)
    )

    y_val = (
        df_val[TARGET]
        .astype(np.float64)
    )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in features
    ]

    model = CatBoostRegressor(
        **CAT_PARAMS
    )

    started = time.time()

    model.fit(
        X_fit,
        y_fit,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        early_stopping_rounds=(
            CAT_EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = (
        time.time() - started
    )

    best_iteration = int(
        model.get_best_iteration()
    )

    if best_iteration >= 0:
        final_iterations = (
            best_iteration + 1
        )
    else:
        final_iterations = int(
            model.tree_count_
        )

    pred = np.clip(
        np.asarray(
            model.predict(X_val),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_val,
        pred,
    )

    return (
        final_iterations,
        {
            "model": "CatBoostRegressor",
            "train_seasons": sorted(
                int(x)
                for x in
                df_fit[
                    SEASON_COL
                ].unique()
            ),
            "validation_season": int(
                ITERATION_VALIDATION_YEAR
            ),
            "n_train": int(
                len(df_fit)
            ),
            "n_validation": int(
                len(df_val)
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
            "elapsed_seconds": float(
                elapsed
            ),
        },
    )


# ============================================================
# Stage 1-B
# XGBoost iteration selection
# ============================================================

def select_xgb_iterations(
    train: pd.DataFrame,
    features: list[str],
):
    season = pd.to_numeric(
        train[SEASON_COL],
        errors="raise",
    ).astype(int)

    fit_mask = (
        season
        < ITERATION_VALIDATION_YEAR
    )

    val_mask = (
        season
        == ITERATION_VALIDATION_YEAR
    )

    df_fit = train.loc[
        fit_mask
    ].copy()

    df_val = train.loc[
        val_mask
    ].copy()

    preprocessor, _, _ = (
        build_xgb_preprocessor(
            features
        )
    )

    X_fit, X_val = (
        fit_transform_xgb(
            preprocessor,
            df_fit[features],
            df_val[features],
        )
    )

    y_fit = (
        df_fit[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    y_val = (
        df_val[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    params = {
        **XGB_PARAMS,
        "early_stopping_rounds": (
            XGB_EARLY_STOPPING_ROUNDS
        ),
    }

    model = XGBRegressor(
        **params
    )

    started = time.time()

    model.fit(
        X_fit,
        y_fit,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False,
    )

    elapsed = (
        time.time() - started
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
            final_iterations - 1
        )
    else:
        best_iteration_zero = int(
            best_iteration
        )
        final_iterations = (
            best_iteration_zero + 1
        )

    pred = np.clip(
        np.asarray(
            model.predict(X_val),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_val,
        pred,
    )

    return (
        final_iterations,
        {
            "model": "XGBRegressor",
            "train_seasons": sorted(
                int(x)
                for x in
                df_fit[
                    SEASON_COL
                ].unique()
            ),
            "validation_season": int(
                ITERATION_VALIDATION_YEAR
            ),
            "n_train": int(
                len(df_fit)
            ),
            "n_validation": int(
                len(df_val)
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
            "elapsed_seconds": float(
                elapsed
            ),
        },
    )


# ============================================================
# Stage 2
# H1 CatBoost
# ============================================================

def train_cat_h1(
    df_h1: pd.DataFrame,
    df_h2: pd.DataFrame,
    features: list[str],
    final_iterations: int,
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

    # H2를 eval_set으로 넘기지 않는다.
    model.fit(
        X_h1,
        y_h1,
        cat_features=cat_cols,
    )

    elapsed = (
        time.time() - started
    )

    pred_raw = np.asarray(
        model.predict(X_h2),
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
# Stage 2
# H1 XGBoost
# ============================================================

def train_xgb_h1(
    df_h1: pd.DataFrame,
    df_h2: pd.DataFrame,
    features: list[str],
    final_iterations: int,
):
    (
        preprocessor,
        xgb_cat_cols,
        xgb_num_cols,
    ) = build_xgb_preprocessor(
        features
    )

    X_h1, X_h2 = (
        fit_transform_xgb(
            preprocessor,
            df_h1[features],
            df_h2[features],
        )
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

    # final fixed-fit에서는 early stopping 제거
    params.pop(
        "early_stopping_rounds",
        None,
    )

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

    elapsed = (
        time.time() - started
    )

    pred_raw = np.asarray(
        model.predict(X_h2),
        dtype=np.float64,
    )

    pred = np.clip(
        pred_raw,
        0.0,
        1.0,
    )

    preprocessing_artifact = {
        "preprocessor": preprocessor,
        "cat_cols": xgb_cat_cols,
        "num_cols": xgb_num_cols,
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
        w * cat_pred
        +
        (1-w) * xgb_pred

    Brier loss의 closed-form optimum.
    convex combination으로 [0,1] 제한.
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

    d = cat - xgb

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
            numerator / denominator,
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
        float(cat_weight)
        * np.asarray(
            cat_pred,
            dtype=np.float64,
        )
        +
        (
            1.0
            - float(cat_weight)
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
            f"기존 model 보존을 위해 중단: {MODEL_DIR}"
        )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보존을 위해 중단: {VALIDATION_DIR}"
        )

    # --------------------------------------------------------
    # Feature schema
    #
    # 원본 train.py와 동일하게 test header만 사용.
    # test row 값은 읽지 않는다.
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

    FEATURES = [
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
            *FEATURES,
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
        required - set(train.columns)
    )

    if missing:
        raise KeyError(
            f"필수 컬럼 없음: {missing}"
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"{TARGET} contains NaN"
        )

    # --------------------------------------------------------
    # 2024 70:30
    # --------------------------------------------------------

    split_state = (
        select_month_cutoff(
            train,
            year=HOLDOUT_YEAR,
            target_share=H1_TARGET_SHARE,
        )
    )

    h1_mask, h2_mask = (
        build_h1_h2_masks(
            train,
            split_state,
        )
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
        df_h1[SEASON_COL]
        == HOLDOUT_YEAR
    ]

    h2_2024 = df_h2.loc[
        df_h2[SEASON_COL]
        == HOLDOUT_YEAR
    ]

    if (
        h1_2024[TIME_COL]
        > cutoff
    ).any():
        raise RuntimeError(
            "H2 row leaked into H1"
        )

    if (
        h2_2024[TIME_COL]
        <= cutoff
    ).any():
        raise RuntimeError(
            "H1 row leaked into H2"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 88)
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )
    print(
        "Models     : CatBoostRegressor + XGBRegressor"
    )
    print(
        "Feature    : provided baseline 47-feature schema"
    )
    print(
        "Split      : 2024 temporal month-block 70:30"
    )
    print(
        "H2 early stopping: OFF"
    )
    print(
        f"Features   : {len(FEATURES)}"
    )
    print("=" * 88)

    print(
        "2024 months:",
        split_state[
            "available_months"
        ],
    )

    print(
        "Cutoff     :",
        cutoff,
    )

    print(
        "H1 rows    :",
        f"{split_state['n_h1']:,}",
        f"({split_state['h1_share']:.4%})",
    )

    print(
        "H2 rows    :",
        f"{split_state['n_h2']:,}",
        f"({split_state['h2_share']:.4%})",
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
    # Iteration selection on 2023
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
        cat_iterations,
        cat_iteration_meta,
    ) = select_cat_iterations(
        train,
        FEATURES,
    )

    (
        xgb_iterations,
        xgb_iteration_meta,
    ) = select_xgb_iterations(
        train,
        FEATURES,
    )

    print(
        "Cat trees  :",
        cat_iterations,
    )

    print(
        "Cat 2023 BSS:",
        f"{cat_iteration_meta['bss_raw']:.2f}",
    )

    print(
        "XGB trees  :",
        xgb_iterations,
    )

    print(
        "XGB 2023 BSS:",
        f"{xgb_iteration_meta['bss_raw']:.2f}",
    )

    # ========================================================
    # Stage 2
    # H1 fixed-fit
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Stage 2: H1 training"
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
        FEATURES,
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
        FEATURES,
        xgb_iterations,
    )

    # ========================================================
    # Stage 3
    # Pure H2 evaluation
    # ========================================================

    y_h2 = (
        df_h2[TARGET]
        .astype(np.float64)
        .to_numpy()
    )

    cat_metrics = calculate_metrics(
        y_h2,
        cat_pred,
    )

    xgb_metrics = calculate_metrics(
        y_h2,
        xgb_pred,
    )

    # ========================================================
    # Stage 4
    # Weight selection
    # ========================================================

    cat_weight = optimal_cat_weight(
        y_h2,
        cat_pred,
        xgb_pred,
    )

    xgb_weight = (
        1.0 - cat_weight
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

    equal_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        0.5,
    )

    equal_metrics = (
        calculate_metrics(
            y_h2,
            equal_pred,
        )
    )

    cat_residual = (
        y_h2 - cat_pred
    )

    xgb_residual = (
        y_h2 - xgb_pred
    )

    prediction_corr = safe_corr(
        cat_pred,
        xgb_pred,
    )

    residual_corr = safe_corr(
        cat_residual,
        xgb_residual,
    )

    best_base_brier = min(
        cat_metrics["brier"],
        xgb_metrics["brier"],
    )

    blend_gain_brier = float(
        best_base_brier
        - ensemble_metrics["brier"]
    )

    # ========================================================
    # Save H2 predictions
    # ========================================================

    pred_df = pd.DataFrame({
        ID: (
            df_h2[ID]
            .to_numpy()
        ),
        "target": y_h2,
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

        "purpose": (
            "2024 70:30 H1/H2 "
            "CatBoostRegressor + XGBRegressor "
            "ensemble weight selection"
        ),

        "not_for_final_2025_inference": True,

        "features": FEATURES,

        "cat_cols": CAT_COLS,

        "split_state": split_state,

        "catboost": {
            "model": cat_model,
            "iterations": int(
                cat_iterations
            ),
            "params": {
                **CAT_PARAMS,
                "iterations": int(
                    cat_iterations
                ),
            },
        },

        "xgboost": {
            "model": xgb_model,
            "preprocessing": (
                xgb_preprocessing
            ),
            "iterations": int(
                xgb_iterations
            ),
            "params": {
                **XGB_PARAMS,
                "n_estimators": int(
                    xgb_iterations
                ),
            },
        },

        "ensemble": {
            "cat_weight": float(
                cat_weight
            ),
            "xgb_weight": float(
                xgb_weight
            ),
            "formula": (
                "cat_weight * cat_prediction "
                "+ xgb_weight * xgb_prediction"
            ),
        },

        "rule_compliance": {
            "test_values_used": False,
            "test_distribution_used": False,
            "cross_test_aggregation": False,
            "h2_target_used_for_base_training": False,
            "h2_target_used_for_preprocessing_fit": False,
            "h2_target_used_for_iteration_selection": False,
            "h2_target_used_for_early_stopping": False,
            "h2_target_used_for_ensemble_weight": True,
            "row_independent_final_design": True,
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

        "models": [
            "CatBoostRegressor",
            "XGBRegressor",
        ],

        "validation_method": (
            "2024 temporal month-block 70:30"
        ),

        "split_state": (
            split_state
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
            "n_rows": int(
                len(df_h2)
            ),

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

            "blend_gain_brier": float(
                blend_gain_brier
            ),
        },

        "training_time_seconds": {
            "catboost": float(
                cat_elapsed
            ),
            "xgboost": float(
                xgb_elapsed
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

    # ========================================================
    # Reproducibility
    # ========================================================

    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR / "train.py",
        )
    except Exception:
        pass

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 88)
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )
    print("=" * 88)

    print(
        "Validation : 2024_H2_30"
    )

    print(
        "Cutoff     :",
        cutoff,
    )

    print(
        "Rows       :",
        f"{len(df_h2):,}",
    )

    print("-" * 88)

    print(
        "Target mean:",
        f"{np.mean(y_h2):.8f}",
    )

    print("-" * 88)

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

    print("-" * 88)

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
        f"{blend_gain_brier:+.10f}",
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

    print("-" * 88)

    print(
        "MODEL_PATH:",
        MODEL_PATH,
    )

    print(
        "OOF       :",
        VALIDATION_DIR
        / "oof_pred.csv",
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
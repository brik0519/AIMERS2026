from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "ensemble_v006"

ID = "row_id"
TARGET = "control_success"

BASE_CAT_WEIGHT = 0.5655685
BASE_XGB_WEIGHT = 1.0 - BASE_CAT_WEIGHT

# =========================================================
# Feature 실험 Boolean Mode
# =========================================================
# Research_v003 권고에 따라 모델의 강한 공통 Feature를 유지한 채
# Temporal extension과 XGBoost raw player ID 유지/제거를 독립적으로
# ablation한다. 하나의 변경이 여러 가설을 동시에 섞지 않도록 한다.
#
# 대표 실험:
# A) Raw / Raw
#    CAT_USE_TEMPORAL = False
#    XGB_USE_TEMPORAL = False
#    XGB_USE_RAW_PLAYER_IDS = True
#
# B) Cat Temporal / XGB Raw
#    CAT_USE_TEMPORAL = True
#    XGB_USE_TEMPORAL = False
#    XGB_USE_RAW_PLAYER_IDS = True
#
# C) Cat Raw / XGB Temporal  <- 우선 검증 권장
#    CAT_USE_TEMPORAL = False
#    XGB_USE_TEMPORAL = True
#    XGB_USE_RAW_PLAYER_IDS = True
#
# D) Temporal / Temporal
#    CAT_USE_TEMPORAL = True
#    XGB_USE_TEMPORAL = True
#    XGB_USE_RAW_PLAYER_IDS = True
#
# C-ID-DROP) XGB specialist에서 raw pitcher/batter ID 제거 효과 분리
#    CAT_USE_TEMPORAL = False
#    XGB_USE_TEMPORAL = True
#    XGB_USE_RAW_PLAYER_IDS = False
CAT_USE_TEMPORAL = False
XGB_USE_TEMPORAL = True
XGB_USE_RAW_PLAYER_IDS = True

# 최종 blend weight:
# - pooled_oof : 모든 expanding OOF prediction으로 최적화
# - fixed      : 기존 weight 사용
FINAL_WEIGHT_MODE = "pooled_oof"

# Hidden future와 가장 가까운 2024 Fold의 best iteration을
# Final model iteration으로 사용한다.
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# 경로
# =========================================================
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


# =========================================================
# Expanding Window Validation
# =========================================================
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
# CatBoost Feature 정책
# =========================================================
# train_cb.py의 categorical 정책을 그대로 유지한다.
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
# XGBoost Feature 정책
# =========================================================
# train_xgb.py의 low-cardinality categorical 정책을 유지한다.
XGB_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

# 기존 XGBoost와의 비교 가능성을 위해 ID-like dtype 정책은 유지한다.
XGB_ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]

# Specialist 실험:
# player raw identity는 CatBoost가 담당하고,
# XGBoost는 pitcher/batter raw ID 대신 temporal representation을 강조한다.
# 팀 ID는 기존 XGB처럼 numeric 유지한다.
XGB_DROP_RAW_ID_COLS = [
    "pitcher_id",
    "batter_id",
]

TEMPORAL_FEATURES = [
    "tf_success_prev1_vs_career",
    "tf_success_prev3_vs_career",
    "tf_success_prev5_vs_career",
    "tf_success_prev1_vs_prev3",
    "tf_success_prev3_vs_prev5",
    "tf_middle_prev1_vs_career",
    "tf_middle_prev3_vs_career",
    "tf_middle_prev5_vs_career",
    "tf_middle_prev1_vs_prev3",
    "tf_middle_prev3_vs_prev5",
]


# =========================================================
# CatBoost 설정 — train_cb.py 적용
# =========================================================
CATBOOST_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

CAT_EARLY_STOPPING_ROUNDS = 80


# =========================================================
# XGBoost 설정 — train_xgb.py 적용
# =========================================================
XGB_MAX_ITERATIONS = 5000
XGB_EARLY_STOPPING_ROUNDS = 200
XGB_RANDOM_STATE = 42

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": XGB_RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# Metric
# =========================================================
def brier_metric(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0 * (1.0 - brier / baseline_brier)
        )
        bss = max(0.0, bss_raw)

    return {
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": bss,
    }


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return float("nan")
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def residual_covariance(y_true, cat_pred, xgb_pred):
    y_true = np.asarray(y_true, dtype=float)
    e_cat = y_true - np.asarray(cat_pred, dtype=float)
    e_xgb = y_true - np.asarray(xgb_pred, dtype=float)

    if len(y_true) < 2:
        return float("nan")

    return float(np.cov(e_cat, e_xgb, ddof=1)[0, 1])


def optimal_cat_weight(y_true, cat_pred, xgb_pred):
    """
    Brier loss를 최소화하는 2-model blend의 CatBoost weight.
    0~1 범위로 clip한다.
    """
    y_true = np.asarray(y_true, dtype=float)
    cat_pred = np.asarray(cat_pred, dtype=float)
    xgb_pred = np.asarray(xgb_pred, dtype=float)

    d = cat_pred - xgb_pred
    denom = float(np.dot(d, d))

    if denom <= 1e-15:
        return 0.5

    weight = float(np.dot(d, y_true - xgb_pred) / denom)
    return float(np.clip(weight, 0.0, 1.0))


def blend_predictions(cat_pred, xgb_pred, cat_weight):
    return (
        float(cat_weight) * np.asarray(cat_pred, dtype=float)
        + (1.0 - float(cat_weight))
        * np.asarray(xgb_pred, dtype=float)
    )


# =========================================================
# Temporal Feature
# =========================================================
def add_temporal_features(df):
    """
    현재 행의 공식 asof_* 값끼리만 연산한다.

    다른 validation/test row에 대한 groupby / rolling / lag /
    expanding / frequency / aggregate를 사용하지 않는다.
    """
    df = df.copy()

    required = [
        "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    career_s = df["asof_pitcher_success_rate"]
    prev1_s = df["asof_pitcher_prev1_game_success_rate"]
    prev3_s = df["asof_pitcher_prev3_game_success_rate"]
    prev5_s = df["asof_pitcher_prev5_game_success_rate"]

    career_m = df["asof_pitcher_middle_rate"]
    prev1_m = df["asof_pitcher_prev1_game_middle_rate"]
    prev3_m = df["asof_pitcher_prev3_game_middle_rate"]
    prev5_m = df["asof_pitcher_prev5_game_middle_rate"]

    df["tf_success_prev1_vs_career"] = prev1_s - career_s
    df["tf_success_prev3_vs_career"] = prev3_s - career_s
    df["tf_success_prev5_vs_career"] = prev5_s - career_s
    df["tf_success_prev1_vs_prev3"] = prev1_s - prev3_s
    df["tf_success_prev3_vs_prev5"] = prev3_s - prev5_s

    df["tf_middle_prev1_vs_career"] = prev1_m - career_m
    df["tf_middle_prev3_vs_career"] = prev3_m - career_m
    df["tf_middle_prev5_vs_career"] = prev5_m - career_m
    df["tf_middle_prev1_vs_prev3"] = prev1_m - prev3_m
    df["tf_middle_prev3_vs_prev5"] = prev3_m - prev5_m

    return df


# =========================================================
# CatBoost 전처리 — train_cb.py 적용
# =========================================================
def prepare_cat_features(df, features, cat_cols):
    X = df[features].copy()

    for col in cat_cols:
        if col in X.columns:
            X[col] = (
                X[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return X


# =========================================================
# XGBoost 전처리 — train_xgb.py 적용
# =========================================================
def prepare_xgb_raw_features(
    df,
    features,
    id_like_cols,
):
    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "입력 데이터에 XGBoost Feature가 없습니다: "
            + ", ".join(missing_features)
        )

    X = df[features].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


def create_xgb_preprocessor(
    cat_cols,
    num_cols,
):
    """
    train_xgb.py와 동일:
    - low-cardinality categorical -> OrdinalEncoder
    - unseen/missing category -> -1
    - numeric -> passthrough
    - numeric NaN -> XGBoost가 직접 처리
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# 모델 생성
# =========================================================
def create_cat_validation_model():
    return CatBoostClassifier(
        **CATBOOST_PARAMS
    )


def create_cat_final_model(n_iterations):
    params = {
        **CATBOOST_PARAMS,
        "iterations": int(n_iterations),
    }
    return CatBoostClassifier(**params), params


def create_xgb_validation_model():
    return XGBClassifier(
        n_estimators=XGB_MAX_ITERATIONS,
        eval_metric=brier_metric,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        **XGBOOST_PARAMS,
    )


def create_xgb_final_model(n_estimators):
    return XGBClassifier(
        n_estimators=int(n_estimators),
        **XGBOOST_PARAMS,
    )


# =========================================================
# Fold 평가
# =========================================================
def evaluate_pair(
    y_true,
    cat_pred,
    xgb_pred,
    walk_forward_cat_weight,
):
    y_true = np.asarray(y_true, dtype=float)
    cat_pred = np.asarray(cat_pred, dtype=float)
    xgb_pred = np.asarray(xgb_pred, dtype=float)

    e_cat = y_true - cat_pred
    e_xgb = y_true - xgb_pred

    diagnostic_weight = optimal_cat_weight(
        y_true,
        cat_pred,
        xgb_pred,
    )

    fixed_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        BASE_CAT_WEIGHT,
    )
    walk_forward_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        walk_forward_cat_weight,
    )
    diagnostic_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        diagnostic_weight,
    )

    return {
        "cat_brier": calculate_metrics(
            y_true,
            cat_pred,
        )["brier"],
        "xgb_brier": calculate_metrics(
            y_true,
            xgb_pred,
        )["brier"],
        "prediction_corr": safe_corr(
            cat_pred,
            xgb_pred,
        ),
        "residual_corr": safe_corr(
            e_cat,
            e_xgb,
        ),
        "residual_covariance": residual_covariance(
            y_true,
            cat_pred,
            xgb_pred,
        ),
        "fixed_cat_weight": float(BASE_CAT_WEIGHT),
        "fixed_ensemble_brier": calculate_metrics(
            y_true,
            fixed_pred,
        )["brier"],
        "walk_forward_cat_weight": float(
            walk_forward_cat_weight
        ),
        "walk_forward_ensemble_brier": calculate_metrics(
            y_true,
            walk_forward_pred,
        )["brier"],
        # 같은 fold target으로 찾은 weight라 진단용이다.
        "diagnostic_opt_cat_weight": float(
            diagnostic_weight
        ),
        "diagnostic_opt_ensemble_brier": calculate_metrics(
            y_true,
            diagnostic_pred,
        )["brier"],
        "best_base_brier": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ),
        "fixed_blend_gain": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ) - calculate_metrics(y_true, fixed_pred)["brier"],
        "walk_forward_blend_gain": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ) - calculate_metrics(y_true, walk_forward_pred)["brier"],
        "diagnostic_opt_blend_gain": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ) - calculate_metrics(y_true, diagnostic_pred)["brier"],
    }


# =========================================================
# Final iteration 선택
# =========================================================
def select_final_iterations(
    fold_results,
    key,
):
    if FINAL_ITERATION_STRATEGY != "latest_fold":
        raise ValueError(
            "지원하지 않는 FINAL_ITERATION_STRATEGY: "
            f"{FINAL_ITERATION_STRATEGY}"
        )

    latest_fold = max(
        fold_results,
        key=lambda x: x["validation_season"],
    )

    return int(latest_fold[key])


# =========================================================
# Main
# =========================================================
def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            "Validation experiment already exists: "
            f"{VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            "Model experiment already exists: "
            f"{MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # 파일 검증
    # -----------------------------------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.csv를 찾을 수 없습니다: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv를 찾을 수 없습니다: {TEST_PATH}"
        )

    # -----------------------------------------------------
    # test schema만 이용해 공식 Raw Feature 결정
    # test의 값/분포/다른 row 정보는 사용하지 않는다.
    # -----------------------------------------------------
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise ValueError(
            f"test.csv에 ID 컬럼 '{ID}'가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID
    ]

    # -----------------------------------------------------
    # Boolean Feature Mode 해석
    # -----------------------------------------------------
    # Temporal Feature는 현재 행의 공식 asof_* 값만으로 생성되므로
    # 평가 데이터의 다른 row에 의존하지 않는다.
    cat_features = raw_features.copy()
    if CAT_USE_TEMPORAL:
        cat_features += TEMPORAL_FEATURES

    if XGB_USE_RAW_PLAYER_IDS:
        xgb_dropped_raw_id_cols = []
        xgb_features = raw_features.copy()
    else:
        xgb_dropped_raw_id_cols = XGB_DROP_RAW_ID_COLS.copy()
        xgb_features = [
            col
            for col in raw_features
            if col not in xgb_dropped_raw_id_cols
        ]

    if XGB_USE_TEMPORAL:
        xgb_features += TEMPORAL_FEATURES

    cat_cols = [
        col
        for col in CAT_COLS
        if col in cat_features
    ]

    xgb_cat_cols = [
        col
        for col in XGB_CAT_COLS
        if col in xgb_features
    ]

    xgb_id_like_cols = [
        col
        for col in XGB_ID_LIKE_COLS
        if col in xgb_features
    ]

    xgb_num_cols = [
        col
        for col in xgb_features
        if col not in xgb_cat_cols
    ]

    # -----------------------------------------------------
    # Train load
    # -----------------------------------------------------
    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *raw_features,
            TARGET,
        ],
    )

    required_cols = {
        ID,
        TARGET,
        "season",
    }
    missing_cols = required_cols - set(train.columns)

    if missing_cols:
        raise ValueError(
            "train.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_cols))
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"타깃 컬럼 '{TARGET}'에 결측값이 있습니다."
        )

    if CAT_USE_TEMPORAL or XGB_USE_TEMPORAL:
        train = add_temporal_features(train)

    print("=" * 72)
    print("Dual Model Feature-Diverse Ensemble")
    print("=" * 72)
    print("Experiment              :", EXPERIMENT_NAME)
    print("XGBoost version         :", xgb.__version__)
    print("Train shape             :", train.shape)
    print("Raw Feature 수          :", len(raw_features))
    print("CatBoost Feature 수     :", len(cat_features))
    print("CatBoost categorical 수 :", len(cat_cols))
    print("XGBoost Feature 수      :", len(xgb_features))
    print("XGBoost categorical 수  :", len(xgb_cat_cols))
    print("CAT_USE_TEMPORAL        :", CAT_USE_TEMPORAL)
    print("XGB_USE_TEMPORAL        :", XGB_USE_TEMPORAL)
    print("XGB_USE_RAW_PLAYER_IDS  :", XGB_USE_RAW_PLAYER_IDS)
    print(
        "Cat Temporal Feature 수 :",
        len(TEMPORAL_FEATURES) if CAT_USE_TEMPORAL else 0,
    )
    print(
        "XGB Temporal Feature 수 :",
        len(TEMPORAL_FEATURES) if XGB_USE_TEMPORAL else 0,
    )
    print("XGB 제거 Raw ID         :", xgb_dropped_raw_id_cols)
    print(
        "Season                  :",
        train["season"].min(),
        "~",
        train["season"].max(),
    )
    print(
        "Target mean             :",
        f"{train[TARGET].mean():.6f}",
    )
    print("=" * 72)

    # -----------------------------------------------------
    # 디렉터리 생성
    # -----------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )
    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Expanding Window
    # -----------------------------------------------------
    fold_results = []
    oof_frames = []

    previous_y = []
    previous_cat_pred = []
    previous_xgb_pred = []

    for fold_config in EXPANDING_FOLDS:

        fold = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        validation_season = (
            fold_config["validation_season"]
        )

        train_mask = train["season"].isin(
            train_seasons
        )
        val_mask = (
            train["season"]
            == validation_season
        )

        if not train_mask.any():
            raise ValueError(
                f"Fold {fold} 학습 데이터가 없습니다."
            )

        if not val_mask.any():
            raise ValueError(
                f"Fold {fold} Validation 데이터가 없습니다."
            )

        train_df = train.loc[
            train_mask
        ].copy()

        val_df = train.loc[
            val_mask
        ].copy()

        y_train = train_df[
            TARGET
        ].copy()

        y_val = val_df[
            TARGET
        ].copy()

        val_row_id = val_df[
            ID
        ].copy()

        print()
        print("=" * 72)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print(
            f"Train={len(train_df):,}, "
            f"Valid={len(val_df):,}"
        )
        print("=" * 72)

        # =================================================
        # CatBoost
        # =================================================
        X_cat_train = prepare_cat_features(
            train_df,
            cat_features,
            cat_cols,
        )

        X_cat_val = prepare_cat_features(
            val_df,
            cat_features,
            cat_cols,
        )

        cat_model = create_cat_validation_model()

        cat_start = time.time()

        cat_model.fit(
            X_cat_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(
                X_cat_val,
                y_val,
            ),
            early_stopping_rounds=(
                CAT_EARLY_STOPPING_ROUNDS
            ),
            use_best_model=True,
        )

        cat_elapsed = (
            time.time() - cat_start
        )

        cat_pred = (
            cat_model.predict_proba(
                X_cat_val
            )[:, 1]
        )

        cat_best_iteration = (
            cat_model.get_best_iteration()
        )

        if (
            cat_best_iteration is None
            or cat_best_iteration < 0
        ):
            cat_final_iterations = (
                CATBOOST_PARAMS["iterations"]
            )
        else:
            cat_final_iterations = (
                int(cat_best_iteration) + 1
            )

        # =================================================
        # XGBoost
        # =================================================
        X_xgb_train_raw = prepare_xgb_raw_features(
            train_df,
            xgb_features,
            xgb_id_like_cols,
        )

        X_xgb_val_raw = prepare_xgb_raw_features(
            val_df,
            xgb_features,
            xgb_id_like_cols,
        )

        # Fold마다 train 구간에만 fit되는 독립 Encoder.
        xgb_preprocessor = create_xgb_preprocessor(
            xgb_cat_cols,
            xgb_num_cols,
        )

        X_xgb_train = (
            xgb_preprocessor
            .fit_transform(
                X_xgb_train_raw
            )
        )

        X_xgb_val = (
            xgb_preprocessor
            .transform(
                X_xgb_val_raw
            )
        )

        xgb_model = create_xgb_validation_model()

        xgb_start = time.time()

        xgb_model.fit(
            X_xgb_train,
            y_train,
            eval_set=[
                (
                    X_xgb_val,
                    y_val,
                ),
            ],
            verbose=100,
        )

        xgb_elapsed = (
            time.time() - xgb_start
        )

        xgb_pred = (
            xgb_model
            .predict_proba(
                X_xgb_val
            )[:, 1]
        )

        xgb_best_iteration = getattr(
            xgb_model,
            "best_iteration",
            None,
        )

        if (
            xgb_best_iteration is None
            or xgb_best_iteration < 0
        ):
            xgb_best_iteration = (
                XGB_MAX_ITERATIONS - 1
            )

        xgb_final_iterations = (
            int(xgb_best_iteration) + 1
        )

        # =================================================
        # Ensemble / Diversity
        # =================================================
        if previous_y:
            previous_y_all = np.concatenate(
                previous_y
            )
            previous_cat_all = np.concatenate(
                previous_cat_pred
            )
            previous_xgb_all = np.concatenate(
                previous_xgb_pred
            )

            walk_forward_cat_weight = (
                optimal_cat_weight(
                    previous_y_all,
                    previous_cat_all,
                    previous_xgb_all,
                )
            )
        else:
            walk_forward_cat_weight = (
                BASE_CAT_WEIGHT
            )

        pair_metrics = evaluate_pair(
            y_val,
            cat_pred,
            xgb_pred,
            walk_forward_cat_weight,
        )

        cat_metrics = calculate_metrics(
            y_val,
            cat_pred,
        )
        xgb_metrics = calculate_metrics(
            y_val,
            xgb_pred,
        )

        fixed_pred = blend_predictions(
            cat_pred,
            xgb_pred,
            BASE_CAT_WEIGHT,
        )

        walk_forward_pred = blend_predictions(
            cat_pred,
            xgb_pred,
            walk_forward_cat_weight,
        )

        fold_result = {
            "fold": int(fold),
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "validation_season": int(
                validation_season
            ),
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "cat_brier": float(
                cat_metrics["brier"]
            ),
            "cat_bss": float(
                cat_metrics["bss"]
            ),
            "cat_best_iteration": int(
                cat_best_iteration
            ),
            "cat_final_iterations": int(
                cat_final_iterations
            ),
            "cat_elapsed_seconds": float(
                cat_elapsed
            ),
            "xgb_brier": float(
                xgb_metrics["brier"]
            ),
            "xgb_bss": float(
                xgb_metrics["bss"]
            ),
            "xgb_best_iteration": int(
                xgb_best_iteration
            ),
            "xgb_final_iterations": int(
                xgb_final_iterations
            ),
            "xgb_elapsed_seconds": float(
                xgb_elapsed
            ),
            **pair_metrics,
        }

        fold_results.append(
            fold_result
        )

        pred_df = pd.DataFrame({
            ID: val_row_id.to_numpy(),
            "target": y_val.to_numpy(),
            "validation_season": (
                validation_season
            ),
            "cat_prediction": cat_pred,
            "xgb_prediction": xgb_pred,
            "fixed_ensemble_prediction": (
                fixed_pred
            ),
            "walk_forward_ensemble_prediction": (
                walk_forward_pred
            ),
        })

        oof_frames.append(
            pred_df
        )

        fold_dir = (
            VALIDATION_DIR
            / f"fold_{validation_season}"
        )
        fold_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        pred_df.to_csv(
            fold_dir / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            fold_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                fold_result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Cat Brier              : "
            f"{pair_metrics['cat_brier']:.8f}"
        )
        print(
            f"XGB Brier              : "
            f"{pair_metrics['xgb_brier']:.8f}"
        )
        print(
            f"Prediction Corr        : "
            f"{pair_metrics['prediction_corr']:.8f}"
        )
        print(
            f"Residual Corr          : "
            f"{pair_metrics['residual_corr']:.8f}"
        )
        print(
            f"Residual Covariance    : "
            f"{pair_metrics['residual_covariance']:.10f}"
        )
        print(
            f"Fixed Ensemble Brier   : "
            f"{pair_metrics['fixed_ensemble_brier']:.8f}"
        )
        print(
            f"Walk-forward Brier     : "
            f"{pair_metrics['walk_forward_ensemble_brier']:.8f} "
            f"(Cat={walk_forward_cat_weight:.6f})"
        )
        print(
            f"Diagnostic Opt Brier   : "
            f"{pair_metrics['diagnostic_opt_ensemble_brier']:.8f} "
            f"(Cat={pair_metrics['diagnostic_opt_cat_weight']:.6f})"
        )
        print(
            f"Walk-forward Blend Gain: "
            f"{pair_metrics['walk_forward_blend_gain']:+.8f}"
        )
        print(
            f"Cat best/final iter    : "
            f"{cat_best_iteration} / {cat_final_iterations}"
        )
        print(
            f"XGB best/final iter    : "
            f"{xgb_best_iteration} / {xgb_final_iterations}"
        )

        previous_y.append(
            y_val.to_numpy(dtype=float)
        )
        previous_cat_pred.append(
            np.asarray(cat_pred, dtype=float)
        )
        previous_xgb_pred.append(
            np.asarray(xgb_pred, dtype=float)
        )

    # -----------------------------------------------------
    # Pooled OOF
    # -----------------------------------------------------
    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    y_oof = oof[
        "target"
    ].to_numpy(dtype=float)

    cat_oof = oof[
        "cat_prediction"
    ].to_numpy(dtype=float)

    xgb_oof = oof[
        "xgb_prediction"
    ].to_numpy(dtype=float)

    pooled_cat_weight = optimal_cat_weight(
        y_oof,
        cat_oof,
        xgb_oof,
    )

    if FINAL_WEIGHT_MODE == "pooled_oof":
        final_cat_weight = pooled_cat_weight
    elif FINAL_WEIGHT_MODE == "fixed":
        final_cat_weight = BASE_CAT_WEIGHT
    else:
        raise ValueError(
            "지원하지 않는 FINAL_WEIGHT_MODE: "
            f"{FINAL_WEIGHT_MODE}"
        )

    final_xgb_weight = (
        1.0 - final_cat_weight
    )

    pooled_pred = blend_predictions(
        cat_oof,
        xgb_oof,
        pooled_cat_weight,
    )

    pooled_fixed_pred = blend_predictions(
        cat_oof,
        xgb_oof,
        BASE_CAT_WEIGHT,
    )

    pooled_metrics = {
        "n_oof": int(len(oof)),
        "cat_brier": float(
            calculate_metrics(
                y_oof,
                cat_oof,
            )["brier"]
        ),
        "xgb_brier": float(
            calculate_metrics(
                y_oof,
                xgb_oof,
            )["brier"]
        ),
        "prediction_corr": float(
            safe_corr(
                cat_oof,
                xgb_oof,
            )
        ),
        "residual_corr": float(
            safe_corr(
                y_oof - cat_oof,
                y_oof - xgb_oof,
            )
        ),
        "residual_covariance": float(
            residual_covariance(
                y_oof,
                cat_oof,
                xgb_oof,
            )
        ),
        "fixed_cat_weight": float(
            BASE_CAT_WEIGHT
        ),
        "fixed_ensemble_brier": float(
            calculate_metrics(
                y_oof,
                pooled_fixed_pred,
            )["brier"]
        ),
        "pooled_opt_cat_weight": float(
            pooled_cat_weight
        ),
        "pooled_opt_xgb_weight": float(
            1.0 - pooled_cat_weight
        ),
        # 전체 OOF를 이용해 weight를 찾았으므로
        # 모델 선택용 지표이며 독립 test score가 아니다.
        "pooled_opt_ensemble_brier": float(
            calculate_metrics(
                y_oof,
                pooled_pred,
            )["brier"]
        ),
        "walk_forward_ensemble_brier": float(
            calculate_metrics(
                y_oof,
                oof[
                    "walk_forward_ensemble_prediction"
                ].to_numpy(dtype=float),
            )["brier"]
        ),
        "best_base_brier": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
        ),
        "fixed_blend_gain": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
            - calculate_metrics(y_oof, pooled_fixed_pred)["brier"]
        ),
        "pooled_opt_blend_gain": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
            - calculate_metrics(y_oof, pooled_pred)["brier"]
        ),
        "walk_forward_blend_gain": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
            - calculate_metrics(
                y_oof,
                oof[
                    "walk_forward_ensemble_prediction"
                ].to_numpy(dtype=float),
            )["brier"]
        ),
    }

    diagnostic_weights = np.asarray(
        [
            fold[
                "diagnostic_opt_cat_weight"
            ]
            for fold in fold_results
        ],
        dtype=float,
    )

    pooled_metrics[
        "diagnostic_weight_mean"
    ] = float(
        np.mean(diagnostic_weights)
    )

    pooled_metrics[
        "diagnostic_weight_std"
    ] = float(
        np.std(
            diagnostic_weights,
            ddof=1,
        )
    )

    oof["pooled_opt_ensemble_prediction"] = (
        pooled_pred
    )

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        fold_results
    ).to_csv(
        VALIDATION_DIR / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Final iterations
    # -----------------------------------------------------
    cat_final_iterations = (
        select_final_iterations(
            fold_results,
            "cat_final_iterations",
        )
    )

    xgb_final_iterations = (
        select_final_iterations(
            fold_results,
            "xgb_final_iterations",
        )
    )

    print()
    print("=" * 72)
    print("Pooled OOF")
    print("=" * 72)
    print(
        "Cat Brier              :",
        f"{pooled_metrics['cat_brier']:.8f}",
    )
    print(
        "XGB Brier              :",
        f"{pooled_metrics['xgb_brier']:.8f}",
    )
    print(
        "Prediction Corr        :",
        f"{pooled_metrics['prediction_corr']:.8f}",
    )
    print(
        "Residual Corr          :",
        f"{pooled_metrics['residual_corr']:.8f}",
    )
    print(
        "Residual Covariance    :",
        f"{pooled_metrics['residual_covariance']:.10f}",
    )
    print(
        "Fixed Ensemble Brier   :",
        f"{pooled_metrics['fixed_ensemble_brier']:.8f}",
    )
    print(
        "Walk-forward Brier     :",
        f"{pooled_metrics['walk_forward_ensemble_brier']:.8f}",
    )
    print(
        "Pooled Opt Brier       :",
        f"{pooled_metrics['pooled_opt_ensemble_brier']:.8f}",
    )
    print(
        "Pooled Opt Blend Gain  :",
        f"{pooled_metrics['pooled_opt_blend_gain']:+.8f}",
    )
    print(
        "Final weight           :",
        f"Cat={final_cat_weight:.8f}, "
        f"XGB={final_xgb_weight:.8f}",
    )
    print(
        "Diagnostic weight std  :",
        f"{pooled_metrics['diagnostic_weight_std']:.8f}",
    )
    print(
        "Final Cat iterations   :",
        cat_final_iterations,
    )
    print(
        "Final XGB iterations   :",
        xgb_final_iterations,
    )
    print("=" * 72)

    # -----------------------------------------------------
    # 전체 데이터 Final CatBoost
    # -----------------------------------------------------
    X_cat_full = prepare_cat_features(
        train,
        cat_features,
        cat_cols,
    )

    y_full = train[
        TARGET
    ].copy()

    (
        final_cat_model,
        final_cat_params,
    ) = create_cat_final_model(
        cat_final_iterations
    )

    cat_final_start = time.time()

    final_cat_model.fit(
        X_cat_full,
        y_full,
        cat_features=cat_cols,
    )

    cat_final_elapsed = (
        time.time() - cat_final_start
    )

    # -----------------------------------------------------
    # 전체 데이터 Final XGBoost
    # -----------------------------------------------------
    X_xgb_full_raw = prepare_xgb_raw_features(
        train,
        xgb_features,
        xgb_id_like_cols,
    )

    final_xgb_preprocessor = (
        create_xgb_preprocessor(
            xgb_cat_cols,
            xgb_num_cols,
        )
    )

    X_xgb_full = (
        final_xgb_preprocessor
        .fit_transform(
            X_xgb_full_raw
        )
    )

    final_xgb_model = create_xgb_final_model(
        xgb_final_iterations
    )

    xgb_final_start = time.time()

    final_xgb_model.fit(
        X_xgb_full,
        y_full,
        verbose=100,
    )

    xgb_final_elapsed = (
        time.time() - xgb_final_start
    )

    # -----------------------------------------------------
    # Ensemble Artifact
    # -----------------------------------------------------
    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": (
            "CatBoostClassifier + "
            "XGBClassifier boolean-ablation ensemble"
        ),
        "catboost_model": final_cat_model,
        "xgboost_model": final_xgb_model,
        "xgboost_preprocessor": (
            final_xgb_preprocessor
        ),
        "cat_features": cat_features,
        "cat_categorical_features": cat_cols,
        "xgb_features": xgb_features,
        "xgb_categorical_features": (
            xgb_cat_cols
        ),
        "xgb_numeric_features": (
            xgb_num_cols
        ),
        "xgb_id_like_features": (
            xgb_id_like_cols
        ),
        "xgb_dropped_raw_id_features": (
            xgb_dropped_raw_id_cols
        ),
        "feature_flags": {
            "cat_use_temporal": bool(CAT_USE_TEMPORAL),
            "xgb_use_temporal": bool(XGB_USE_TEMPORAL),
            "xgb_use_raw_player_ids": bool(XGB_USE_RAW_PLAYER_IDS),
        },
        "cat_temporal_features": (
            TEMPORAL_FEATURES if CAT_USE_TEMPORAL else []
        ),
        "xgb_temporal_features": (
            TEMPORAL_FEATURES if XGB_USE_TEMPORAL else []
        ),
        "temporal_features": (
            TEMPORAL_FEATURES
        ),
        "cat_weight": float(
            final_cat_weight
        ),
        "xgb_weight": float(
            final_xgb_weight
        ),
        "weight_mode": FINAL_WEIGHT_MODE,
        "catboost_params": (
            final_cat_params
        ),
        "xgboost_params": {
            **XGBOOST_PARAMS,
            "n_estimators": int(
                xgb_final_iterations
            ),
        },
        "cat_final_iterations": int(
            cat_final_iterations
        ),
        "xgb_final_iterations": int(
            xgb_final_iterations
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "training_seasons": sorted(
            [
                int(x)
                for x in train[
                    "season"
                ].unique()
            ]
        ),
        "n_train": int(len(train)),
        "raw_features": raw_features,
        "cat_features": cat_features,
        "cat_categorical_features": cat_cols,
        "xgb_features": xgb_features,
        "xgb_categorical_features": (
            xgb_cat_cols
        ),
        "xgb_numeric_features": (
            xgb_num_cols
        ),
        "xgb_id_like_features": (
            xgb_id_like_cols
        ),
        "xgb_dropped_raw_id_features": (
            xgb_dropped_raw_id_cols
        ),
        "feature_flags": {
            "cat_use_temporal": bool(CAT_USE_TEMPORAL),
            "xgb_use_temporal": bool(XGB_USE_TEMPORAL),
            "xgb_use_raw_player_ids": bool(XGB_USE_RAW_PLAYER_IDS),
        },
        "cat_temporal_features": (
            TEMPORAL_FEATURES if CAT_USE_TEMPORAL else []
        ),
        "xgb_temporal_features": (
            TEMPORAL_FEATURES if XGB_USE_TEMPORAL else []
        ),
        "temporal_features": (
            TEMPORAL_FEATURES
        ),
        "catboost_validation_params": (
            CATBOOST_PARAMS
        ),
        "xgboost_validation_params": (
            XGBOOST_PARAMS
        ),
        "cat_early_stopping_rounds": (
            CAT_EARLY_STOPPING_ROUNDS
        ),
        "xgb_early_stopping_rounds": (
            XGB_EARLY_STOPPING_ROUNDS
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "final_weight_mode": (
            FINAL_WEIGHT_MODE
        ),
        "final_cat_weight": float(
            final_cat_weight
        ),
        "final_xgb_weight": float(
            final_xgb_weight
        ),
        "cat_final_iterations": int(
            cat_final_iterations
        ),
        "xgb_final_iterations": int(
            xgb_final_iterations
        ),
        "cat_final_elapsed_seconds": float(
            cat_final_elapsed
        ),
        "xgb_final_elapsed_seconds": float(
            xgb_final_elapsed
        ),
        "folds": fold_results,
        "pooled_oof": pooled_metrics,
        "model_path": str(MODEL_PATH),
    }

    with open(
        VALIDATION_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    feature_manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "feature_flags": {
            "cat_use_temporal": bool(CAT_USE_TEMPORAL),
            "xgb_use_temporal": bool(XGB_USE_TEMPORAL),
            "xgb_use_raw_player_ids": bool(XGB_USE_RAW_PLAYER_IDS),
        },
        "raw_features": raw_features,
        "cat_features": cat_features,
        "cat_categorical_features": cat_cols,
        "xgb_features": xgb_features,
        "xgb_categorical_features": xgb_cat_cols,
        "xgb_numeric_features": xgb_num_cols,
        "xgb_dropped_raw_id_features": xgb_dropped_raw_id_cols,
        "cat_temporal_features": (
            TEMPORAL_FEATURES if CAT_USE_TEMPORAL else []
        ),
        "xgb_temporal_features": (
            TEMPORAL_FEATURES if XGB_USE_TEMPORAL else []
        ),
        "independent_test_row_rule": (
            "Temporal features use only values from the current row's "
            "official asof_* columns; no cross-test aggregation, rolling, "
            "lag, frequency, rank, or distribution is used."
        ),
    }

    with open(
        VALIDATION_DIR / "feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    source_path = Path(
        __file__
    ).resolve()

    shutil.copy2(
        source_path,
        MODEL_DIR / "train.py",
    )

    print()
    print("=" * 72)
    print("Final Ensemble 저장 완료")
    print("- MODEL_PATH :", MODEL_PATH)
    print("- Validation :", VALIDATION_DIR)
    print(
        "- Weight     : "
        f"Cat={final_cat_weight:.8f}, "
        f"XGB={final_xgb_weight:.8f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

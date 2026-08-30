# train.py
#
# ensemble_v011
#
# Final full-train model for LB submission
#
# Source experiment:
#   ensemble_v011_h1
#
# Frozen decisions:
#   CatBoost iterations = 207
#   XGBoost iterations  = 115
#   Cat weight          = 0.70
#   XGB weight          = 0.30
#
# Training:
#   2019~2024 official train.csv ALL rows
#
# Important:
# - 2024 H2 is no longer validation data here.
# - It is now legitimate official training data.
# - Ensemble weight and iterations are already frozen BEFORE this fit.
# - test.csv values/distribution are never used for fitting.


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

EXPERIMENT_NAME = "ensemble_v011"

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


# ============================================================
# Frozen decisions from ensemble_v011_h1
# ============================================================

SOURCE_EXPERIMENT_NAME = "ensemble_v011_h1"

CAT_FINAL_ITERATIONS = 207
XGB_FINAL_ITERATIONS = 115

CAT_WEIGHT = 0.70
XGB_WEIGHT = 0.30


if not np.isclose(
    CAT_WEIGHT + XGB_WEIGHT,
    1.0,
):
    raise ValueError(
        "ensemble weights must sum to 1"
    )


# ============================================================
# Categorical policy
#
# 반드시 v011_h1과 동일하게 유지
# ============================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
]


# ============================================================
# CatBoost
#
# v011_h1과 동일한 parameter.
# iterations만 frozen value로 교체.
# ============================================================

CAT_PARAMS = {
    "iterations": CAT_FINAL_ITERATIONS,
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


# ============================================================
# XGBoost
#
# v011_h1과 동일한 parameter.
# n_estimators만 frozen value로 교체.
# ============================================================

XGB_PARAMS = {
    "n_estimators": XGB_FINAL_ITERATIONS,
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


# ============================================================
# CatBoost preprocessing
# ============================================================

def prepare_catboost(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> pd.DataFrame:

    out = df.copy()

    for col in cat_cols:

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
                    handle_unknown="use_encoded_value",
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
            "기존 model 보존을 위해 중단: "
            f"{MODEL_DIR}"
        )

    # --------------------------------------------------------
    # Feature schema
    #
    # v011_h1과 동일하게 test header만 사용.
    # test 값은 읽지 않는다.
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
    # Official full train
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

    if train[TARGET].isna().any():
        raise ValueError(
            f"{TARGET}에 결측값이 있습니다."
        )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

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
        "Training   : 2019~2024 full official train"
    )

    print(
        "Features   :",
        len(features),
    )

    print(
        "Cat trees  :",
        CAT_FINAL_ITERATIONS,
    )

    print(
        "XGB trees  :",
        XGB_FINAL_ITERATIONS,
    )

    print(
        "Cat weight :",
        CAT_WEIGHT,
    )

    print(
        "XGB weight :",
        XGB_WEIGHT,
    )

    print("=" * 88)

    # --------------------------------------------------------
    # Directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # CatBoost Full Train
    # ========================================================

    print()
    print("=" * 88)
    print("CatBoost full training")
    print("=" * 88)

    X_cat = prepare_catboost(
        train[features],
        cat_cols,
    )

    y = (
        train[TARGET]
        .astype(np.float64)
    )

    cat_model = CatBoostRegressor(
        **CAT_PARAMS
    )

    cat_started = time.time()

    cat_model.fit(
        X_cat,
        y,
        cat_features=cat_cols,
    )

    cat_elapsed = float(
        time.time()
        - cat_started
    )

    # ========================================================
    # XGBoost Full Train
    # ========================================================

    print()
    print("=" * 88)
    print("XGBoost full training")
    print("=" * 88)

    (
        xgb_preprocessor,
        xgb_cat_cols,
        xgb_num_cols,
    ) = build_xgb_preprocessor(
        features
    )

    # 중요:
    # preprocessing fit도 공식 train 전체에서만 수행.
    X_xgb = (
        xgb_preprocessor
        .fit_transform(
            train[features]
        )
    )

    X_xgb = np.asarray(
        X_xgb,
        dtype=np.float32,
    )

    y_xgb = (
        train[TARGET]
        .astype(np.float32)
        .to_numpy()
    )

    xgb_model = XGBRegressor(
        **XGB_PARAMS
    )

    xgb_started = time.time()

    xgb_model.fit(
        X_xgb,
        y_xgb,
        verbose=False,
    )

    xgb_elapsed = float(
        time.time()
        - xgb_started
    )

    # ========================================================
    # Artifact
    # ========================================================

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "source_experiment_name": (
            SOURCE_EXPERIMENT_NAME
        ),

        "model_type": (
            "CatBoostRegressor + XGBRegressor"
        ),

        "features": (
            features
        ),

        "catboost": {
            "model": (
                cat_model
            ),
            "cat_cols": (
                cat_cols
            ),
            "iterations": int(
                CAT_FINAL_ITERATIONS
            ),
            "params": (
                CAT_PARAMS
            ),
        },

        "xgboost": {
            "model": (
                xgb_model
            ),
            "preprocessor": (
                xgb_preprocessor
            ),
            "cat_cols": (
                xgb_cat_cols
            ),
            "num_cols": (
                xgb_num_cols
            ),
            "iterations": int(
                XGB_FINAL_ITERATIONS
            ),
            "params": (
                XGB_PARAMS
            ),
        },

        "ensemble": {
            "cat_weight": float(
                CAT_WEIGHT
            ),
            "xgb_weight": float(
                XGB_WEIGHT
            ),
            "formula": (
                "0.70 * cat_prediction "
                "+ 0.30 * xgb_prediction"
            ),
        },

        "training": {
            "scope": (
                "full official train"
            ),
            "n_rows": int(
                len(train)
            ),
            "target_mean": float(
                train[TARGET].mean()
            ),
            "cat_elapsed_seconds": (
                cat_elapsed
            ),
            "xgb_elapsed_seconds": (
                xgb_elapsed
            ),
        },

        "prediction_policy": {
            "catboost": (
                "np.clip(model.predict(X), 0, 1)"
            ),
            "xgboost": (
                "np.clip(model.predict(preprocessor.transform(X)), 0, 1)"
            ),
            "ensemble": (
                "0.70 * cat + 0.30 * xgb"
            ),
        },

        "rule_compliance": {
            "test_values_used_for_training": False,
            "test_distribution_used": False,
            "cross_test_aggregation": False,
            "weights_frozen_before_test_inference": True,
            "model_parameters_frozen_before_test_inference": True,
            "row_independent_test_prediction": True,
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "source_experiment_name": (
            SOURCE_EXPERIMENT_NAME
        ),

        "n_train_rows": int(
            len(train)
        ),

        "n_features": int(
            len(features)
        ),

        "cat_iterations": int(
            CAT_FINAL_ITERATIONS
        ),

        "xgb_iterations": int(
            XGB_FINAL_ITERATIONS
        ),

        "cat_weight": float(
            CAT_WEIGHT
        ),

        "xgb_weight": float(
            XGB_WEIGHT
        ),

        "cat_elapsed_seconds": (
            cat_elapsed
        ),

        "xgb_elapsed_seconds": (
            xgb_elapsed
        ),

        "model_path": str(
            MODEL_PATH
        ),

        "rule_compliance": (
            artifact[
                "rule_compliance"
            ]
        ),
    }

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
            MODEL_DIR
            / "train.py",
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    print()
    print("=" * 88)

    print("Full training complete")

    print(
        "MODEL_PATH:",
        MODEL_PATH,
    )

    print(
        "Cat elapsed:",
        f"{cat_elapsed:.2f}s",
    )

    print(
        "XGB elapsed:",
        f"{xgb_elapsed:.2f}s",
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
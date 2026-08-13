# train.py
#
# XGBoost 단독 + EXP3 Temporal Feature
# Dacon 제출 모델 학습용
#
# 기존 XGBoost Basic 코드와 동일한:
# - OrdinalEncoder 전처리
# - categorical / numeric 정책
# - XGBoost 하이퍼파라미터
# - 2024 Validation + Brier early stopping
#
# 변경점:
# - EXP3 Temporal Feature 10개만 추가
# - 최종 모델을 ./model/rf.pkl 로 저장
#
# 실행:
#   python train.py
#
# 최종 생성:
#   ./model/rf.pkl

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 기본 설정
# =========================================================

DATA_DIR = Path("./data")

ID = "row_id"
TARGET = "control_success"

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "rf.pkl"

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42


# =========================================================
# 기존 XGBoost Feature 정책 그대로 유지
# =========================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]


# =========================================================
# EXP3 Temporal Feature 10개
# =========================================================

TEMPORAL_FEATURES = [
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
    "success_trend_1_3",
    "success_trend_3_5",
    "middle_trend_1_3",
    "middle_trend_3_5",
]


def add_temporal_features(df):
    """
    EXP3 Temporal Feature 10개 생성.

    원본 asof_* Feature는 제거하지 않는다.
    test row끼리 새로 집계하지 않고,
    각 row에 이미 제공된 asof_* 값만 사용한다.
    """

    df = df.copy()

    required_cols = [
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "EXP3 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    # Recent vs cumulative level
    df["recent1_success_vs_career"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["recent3_success_vs_career"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["recent5_success_vs_career"] = (
        df["asof_pitcher_prev5_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    # Trend
    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )

    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )

    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


# =========================================================
# 기존 XGBoost 파라미터 그대로 유지
# =========================================================

MODEL_PARAMS = {
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
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# Brier Score
# =========================================================

def brier_metric(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


# =========================================================
# XGBoost 입력 준비
# =========================================================

def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "입력 데이터에 Feature가 없습니다: "
            + ", ".join(missing)
        )

    X = df[features].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


def create_preprocessor(
    cat_cols,
    num_cols,
):
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


def create_validation_model():
    return XGBClassifier(
        n_estimators=MAX_ITERATIONS,
        eval_metric=brier_metric,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **MODEL_PARAMS,
    )


def create_final_model(n_estimators):
    return XGBClassifier(
        n_estimators=n_estimators,
        **MODEL_PARAMS,
    )


# =========================================================
# Main
# =========================================================

def main():

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv 없음: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"test.csv 없음: {test_path}"
        )

    # -----------------------------------------------------
    # test.csv 기준 Raw Feature 47개 결정
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise ValueError(
            f"test.csv에 {ID}가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID
    ]

    # -----------------------------------------------------
    # train.csv 로드
    # -----------------------------------------------------

    train = pd.read_csv(
        train_path,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *raw_features,
            TARGET,
        ],
    )

    if "season" not in train.columns:
        raise ValueError(
            "train.csv에 season 컬럼이 없습니다."
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"{TARGET}에 결측값이 있습니다."
        )

    # -----------------------------------------------------
    # EXP3 추가
    # -----------------------------------------------------

    train = add_temporal_features(
        train
    )

    features = (
        raw_features
        + TEMPORAL_FEATURES
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    id_like_cols = [
        col
        for col in ID_LIKE_COLS
        if col in features
    ]

    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]

    print("=" * 70)
    print("XGBoost + EXP3")
    print("=" * 70)
    print("XGBoost version     :", xgb.__version__)
    print("Raw Feature         :", len(raw_features))
    print("Temporal Feature    :", len(TEMPORAL_FEATURES))
    print("Total Feature       :", len(features))
    print("Categorical Feature :", len(cat_cols))
    print("Season              :", train["season"].min(), "~", train["season"].max())
    print("=" * 70)

    # =====================================================
    # 2024 Validation
    # =====================================================

    is_val = (
        train["season"]
        == 2024
    )

    train_df = train.loc[
        ~is_val
    ].copy()

    val_df = train.loc[
        is_val
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    y_val = val_df[
        TARGET
    ].copy()

    X_train_raw = prepare_raw_features(
        train_df,
        features,
        id_like_cols,
    )

    X_val_raw = prepare_raw_features(
        val_df,
        features,
        id_like_cols,
    )

    validation_preprocessor = (
        create_preprocessor(
            cat_cols,
            num_cols,
        )
    )

    X_train = (
        validation_preprocessor
        .fit_transform(
            X_train_raw
        )
    )

    X_val = (
        validation_preprocessor
        .transform(
            X_val_raw
        )
    )

    validation_model = (
        create_validation_model()
    )

    print()
    print("=" * 70)
    print("2024 Validation 학습")
    print("=" * 70)

    start = time.time()

    validation_model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_val,
                y_val,
            ),
        ],
        verbose=100,
    )

    print(
        f"Validation 학습 시간: "
        f"{time.time() - start:.1f}s"
    )

    best_iteration = getattr(
        validation_model,
        "best_iteration",
        None,
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        best_iteration = (
            MAX_ITERATIONS - 1
        )

    final_iterations = (
        best_iteration + 1
    )

    val_pred = (
        validation_model
        .predict_proba(
            X_val
        )[:, 1]
    )

    y_val_array = (
        y_val.to_numpy()
    )

    brier = float(
        np.mean(
            (
                val_pred
                - y_val_array
            ) ** 2
        )
    )

    validation_rate = float(
        y_val.mean()
    )

    baseline_brier = (
        validation_rate
        * (
            1.0
            - validation_rate
        )
    )

    validation_score = max(
        0.0,
        100000.0
        * (
            1.0
            - brier
            / baseline_brier
        ),
    )

    print()
    print("=" * 70)
    print("2024 Validation 결과")
    print("=" * 70)
    print(f"Brier Score      : {brier:.10f}")
    print(f"Baseline Brier   : {baseline_brier:.10f}")
    print(f"Validation Score : {validation_score:.4f}")
    print(f"Best iteration   : {best_iteration}")
    print(f"Final iterations : {final_iterations}")
    print(f"Prediction mean  : {val_pred.mean():.6f}")
    print(f"Prediction std   : {val_pred.std():.6f}")
    print(f"Prediction min   : {val_pred.min():.6f}")
    print(f"Prediction max   : {val_pred.max():.6f}")
    print("=" * 70)

    # =====================================================
    # 2019~2024 전체 재학습
    # =====================================================

    X_full_raw = prepare_raw_features(
        train,
        features,
        id_like_cols,
    )

    y_full = train[
        TARGET
    ].copy()

    final_preprocessor = (
        create_preprocessor(
            cat_cols,
            num_cols,
        )
    )

    X_full = (
        final_preprocessor
        .fit_transform(
            X_full_raw
        )
    )

    final_model = (
        create_final_model(
            final_iterations
        )
    )

    print()
    print("=" * 70)
    print("2019~2024 전체 재학습")
    print("=" * 70)

    start = time.time()

    final_model.fit(
        X_full,
        y_full,
        verbose=False,
    )

    print(
        f"전체 학습 시간: "
        f"{time.time() - start:.1f}s"
    )

    # =====================================================
    # 제출용 model/rf.pkl 저장
    # =====================================================

    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,

        "raw_features": raw_features,
        "temporal_features": TEMPORAL_FEATURES,
        "features": features,

        "cat_cols": cat_cols,
        "id_like_cols": id_like_cols,
        "num_cols": num_cols,

        "best_iteration": int(
            best_iteration
        ),

        "final_iterations": int(
            final_iterations
        ),

        "validation_brier": float(
            brier
        ),

        "validation_score": float(
            validation_score
        ),

        "model_name": "xgb_exp3",
        "feature_mode": "EXP3_TEMPORAL",
        "xgboost_version": xgb.__version__,
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    print()
    print("=" * 70)
    print("제출용 모델 저장 완료")
    print("=" * 70)
    print("Model:", MODEL_PATH)
    print("Features:", len(features))
    print("Final iterations:", final_iterations)
    print("=" * 70)


if __name__ == "__main__":
    main()

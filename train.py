# train.py

import json
import shutil
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
# 실험 설정
# =========================================================

DATA_DIR = Path("./data")

ID = "row_id"
TARGET = "control_success"

# 실험마다 이 값만 변경
EXPERIMENT_NAME = "xgb_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# 학습 설정
# =========================================================

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42


# =========================================================
# Feature 정책
# =========================================================

# 저카디널리티 명목형만 categorical encoding
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

# 고카디널리티 ID는 categorical encoding하지 않고
# numeric 값으로 유지
ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]


# =========================================================
# XGBoost 모델 파라미터
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

def brier_metric(
    y_true,
    y_pred,
):
    """
    XGBoost validation용 Brier Score.

    callable eval_metric은 cost function으로 사용되며
    값이 작을수록 좋다.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


# =========================================================
# Raw Feature 준비
# =========================================================

def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    """
    XGBoost 전처리 직전 DataFrame을 생성한다.

    1. 학습 Feature 순서 고정
    2. ID 계열은 numeric으로 강제 변환
    3. 나머지 numeric NaN은 그대로 유지
       → XGBoost가 직접 처리
    """

    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "입력 데이터에 학습 Feature가 없습니다: "
            + ", ".join(
                missing_features
            )
        )

    X = df[
        features
    ].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


# =========================================================
# 전처리 생성
# =========================================================

def create_preprocessor(
    cat_cols,
    num_cols,
):
    """
    새로운 OrdinalEncoder를 포함하는
    XGBoost 전처리 객체를 매번 생성한다.

    Validation과 Final model이 서로 독립된
    fitted encoder를 갖도록 한다.

    categorical:
        "__MISSING__" 사용
        unseen category → -1

    numeric:
        그대로 통과
        NaN도 그대로 XGBoost에 전달
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
# Validation 모델 생성
# =========================================================

def create_validation_model():
    """
    Brier Score 기반 early stopping을 사용하는
    Validation용 XGBClassifier를 생성한다.
    """

    return XGBClassifier(
        n_estimators=MAX_ITERATIONS,

        eval_metric=brier_metric,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,

        **MODEL_PARAMS,
    )


# =========================================================
# Final 모델 생성
# =========================================================

def create_final_model(
    n_estimators,
):
    """
    Validation에서 찾은 best iteration만큼
    전체 데이터로 다시 학습할 Final 모델.

    Final 모델에는 eval_metric callable과
    early stopping 설정을 넣지 않는다.

    이렇게 해야 저장된 model.pkl이
    script.py의 함수 정의에 의존하지 않는다.
    """

    return XGBClassifier(
        n_estimators=n_estimators,
        **MODEL_PARAMS,
    )


# =========================================================
# Validation 결과 저장
# =========================================================

def save_validation(
    experiment_name,
    estimator,
    model_params,
    row_ids,
    y_true,
    y_pred,
    brier,
    baseline_brier,
    bss,
    output_root,
    best_iteration,
):
    row_ids = np.asarray(
        row_ids
    )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    assert (
        len(row_ids)
        == len(y_true)
        == len(y_pred)
    )

    output_dir = (
        Path(output_root)
        / experiment_name
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Validation experiment already exists: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # pred.csv
    # -----------------------------------------------------

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction": y_pred,
    })

    pred_df.to_csv(
        output_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # validation.npz
    # -----------------------------------------------------

    np.savez_compressed(
        output_dir / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction=y_pred,
    )

    # -----------------------------------------------------
    # metadata.json
    # -----------------------------------------------------

    metadata = {
        "experiment_name": experiment_name,
        "model_type": type(estimator).__name__,
        "model_params": model_params,

        "max_iterations": MAX_ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(best_iteration),

        "validation_season": 2024,
        "n_validation": int(len(y_true)),

        "positive_rate": float(
            np.mean(y_true)
        ),

        "brier": float(brier),
        "baseline_brier": float(
            baseline_brier
        ),
        "bss": float(bss),

        "prediction_mean": float(
            np.mean(y_pred)
        ),

        "prediction_std": float(
            np.std(y_pred)
        ),

        "prediction_min": float(
            np.min(y_pred)
        ),

        "prediction_max": float(
            np.max(y_pred)
        ),

        "xgboost_version": xgb.__version__,
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("[Validation 저장 완료]")
    print(
        f"- {output_dir / 'pred.csv'}"
    )
    print(
        f"- {output_dir / 'validation.npz'}"
    )
    print(
        f"- {output_dir / 'metadata.json'}"
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
            f"Validation experiment already exists: "
            f"{VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"Model experiment already exists: "
            f"{MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # 경로
    # -----------------------------------------------------

    TEST_PATH = (
        DATA_DIR
        / "test.csv"
    )

    TRAIN_PATH = (
        DATA_DIR
        / "train.csv"
    )

    # -----------------------------------------------------
    # 데이터 파일 검증
    # -----------------------------------------------------

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv를 찾을 수 없습니다: "
            f"{TEST_PATH}"
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.csv를 찾을 수 없습니다: "
            f"{TRAIN_PATH}"
        )

    # -----------------------------------------------------
    # test.csv 기준 Feature 결정
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise ValueError(
            f"test.csv에 ID 컬럼 "
            f"'{ID}'가 없습니다."
        )

    features = [
        col
        for col in test_cols
        if col != ID
    ]

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

    # -----------------------------------------------------
    # 학습 데이터 로드
    # -----------------------------------------------------

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *features,
            TARGET,
        ],
    )

    required_columns = {
        ID,
        "season",
        TARGET,
    }

    missing_columns = (
        required_columns
        - set(train.columns)
    )

    if missing_columns:
        raise ValueError(
            "train.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"타깃 컬럼 '{TARGET}'에 "
            "결측값이 있습니다."
        )

    print("=" * 70)
    print("XGBoost 데이터 로드 완료")
    print("=" * 70)

    print(
        "XGBoost version      :",
        xgb.__version__,
    )

    print(
        "Train shape          :",
        train.shape,
    )

    print(
        "Feature 수           :",
        len(features),
    )

    print(
        "Categorical 수       :",
        len(cat_cols),
    )

    print(
        "Numeric 수           :",
        len(num_cols),
    )

    print(
        "Categorical Feature  :",
        cat_cols,
    )

    print(
        "Numeric ID Feature   :",
        id_like_cols,
    )

    print(
        "시즌 범위            :",
        train["season"].min(),
        "~",
        train["season"].max(),
    )

    print(
        f"전체 제구 성공률      : "
        f"{train[TARGET].mean():.6f}"
    )

    print(
        "Device               : cpu"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # 2024 Validation 분할
    # -----------------------------------------------------

    is_val = (
        train["season"]
        == 2024
    )

    if not is_val.any():
        raise ValueError(
            "2024 시즌 Validation 데이터가 없습니다."
        )

    if is_val.all():
        raise ValueError(
            "2024년 이전 학습 데이터가 없습니다."
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

    val_row_id = val_df[
        ID
    ].copy()

    # -----------------------------------------------------
    # Validation raw features
    # -----------------------------------------------------

    X_train_raw = prepare_raw_features(
        df=train_df,
        features=features,
        id_like_cols=id_like_cols,
    )

    X_val_raw = prepare_raw_features(
        df=val_df,
        features=features,
        id_like_cols=id_like_cols,
    )

    # -----------------------------------------------------
    # Validation encoder
    # -----------------------------------------------------

    validation_preprocessor = (
        create_preprocessor(
            cat_cols=cat_cols,
            num_cols=num_cols,
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

    print(
        "학습 데이터 shape     :",
        X_train.shape,
    )

    print(
        "검증 데이터 shape     :",
        X_val.shape,
    )

    print(
        f"학습 Target 평균      : "
        f"{y_train.mean():.6f}"
    )

    print(
        f"검증 Target 평균      : "
        f"{y_val.mean():.6f}"
    )

    # -----------------------------------------------------
    # Validation 모델
    # -----------------------------------------------------

    validation_model = (
        create_validation_model()
    )

    print("=" * 70)
    print("XGBoost Validation 학습 시작")

    print(
        f"최대 iteration        : "
        f"{MAX_ITERATIONS}"
    )

    print(
        f"Early stopping rounds : "
        f"{EARLY_STOPPING_ROUNDS}"
    )

    print(
        "Early stopping metric : "
        "Brier Score"
    )

    print(
        "Device                : "
        "CPU"
    )

    print("=" * 70)

    validation_start = (
        time.time()
    )

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

    validation_elapsed = (
        time.time()
        - validation_start
    )

    print(
        f"Validation 학습 완료 : "
        f"{validation_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # Best iteration
    # -----------------------------------------------------

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

    # XGBoost best_iteration은 0-based
    final_iterations = (
        best_iteration + 1
    )

    print(
        f"Best iteration       : "
        f"{best_iteration}"
    )

    print(
        f"Final iterations     : "
        f"{final_iterations}"
    )

    # -----------------------------------------------------
    # Validation prediction
    # -----------------------------------------------------

    val_pred = (
        validation_model
        .predict_proba(
            X_val
        )[:, 1]
    )

    # -----------------------------------------------------
    # Validation prediction 검증
    # -----------------------------------------------------

    if len(val_pred) != len(y_val):
        raise RuntimeError(
            "Validation 예측 수와 "
            "정답 수가 다릅니다."
        )

    if not np.isfinite(
        val_pred
    ).all():
        raise RuntimeError(
            "Validation 예측에 "
            "NaN 또는 무한대가 있습니다."
        )

    if not (
        (val_pred >= 0.0)
        & (val_pred <= 1.0)
    ).all():
        raise RuntimeError(
            "Validation 예측이 "
            "0~1 범위를 벗어났습니다."
        )

    # -----------------------------------------------------
    # Brier Skill Score
    # -----------------------------------------------------

    y_val_array = (
        y_val.to_numpy()
    )

    brier = np.mean(
        (
            val_pred
            - y_val_array
        ) ** 2
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

    if baseline_brier <= 0:
        raise RuntimeError(
            "Baseline Brier Score가 "
            "0 이하입니다."
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

    print("=" * 70)
    print("XGBoost Validation 결과")
    print("=" * 70)

    print(
        f"Brier Score          : "
        f"{brier:.10f}"
    )

    print(
        f"Baseline Brier       : "
        f"{baseline_brier:.10f}"
    )

    print(
        f"Validation Score     : "
        f"{validation_score:.4f}"
    )

    print(
        f"Best iteration       : "
        f"{best_iteration}"
    )

    print(
        f"Final iterations     : "
        f"{final_iterations}"
    )

    print(
        f"Target mean          : "
        f"{y_val_array.mean():.6f}"
    )

    print(
        f"Prediction mean      : "
        f"{val_pred.mean():.6f}"
    )

    print(
        f"Prediction std       : "
        f"{val_pred.std():.6f}"
    )

    print(
        f"Prediction min       : "
        f"{val_pred.min():.6f}"
    )

    print(
        f"Prediction max       : "
        f"{val_pred.max():.6f}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Validation artifact 저장
    # -----------------------------------------------------

    save_validation(
        experiment_name=EXPERIMENT_NAME,
        estimator=validation_model,
        model_params=MODEL_PARAMS,
        row_ids=val_row_id,
        y_true=y_val,
        y_pred=val_pred,
        brier=brier,
        baseline_brier=baseline_brier,
        bss=validation_score,
        output_root=VALIDATION_ROOT,
        best_iteration=best_iteration,
    )

    # -----------------------------------------------------
    # 전체 데이터용 Raw Feature
    # -----------------------------------------------------

    X_full_raw = prepare_raw_features(
        df=train,
        features=features,
        id_like_cols=id_like_cols,
    )

    y_full = train[
        TARGET
    ].copy()

    # -----------------------------------------------------
    # Final encoder
    #
    # Validation encoder를 재사용하지 않는다.
    # -----------------------------------------------------

    final_preprocessor = (
        create_preprocessor(
            cat_cols=cat_cols,
            num_cols=num_cols,
        )
    )

    X_full = (
        final_preprocessor
        .fit_transform(
            X_full_raw
        )
    )

    # -----------------------------------------------------
    # Final model
    #
    # Validation model을 재사용하지 않는다.
    # -----------------------------------------------------

    final_model = (
        create_final_model(
            n_estimators=final_iterations,
        )
    )

    print("=" * 70)
    print("XGBoost 전체 데이터 재학습 시작")

    print(
        f"사용 iteration       : "
        f"{final_iterations}"
    )

    print(
        "Device               : CPU"
    )

    print("=" * 70)

    final_start = (
        time.time()
    )

    final_model.fit(
        X_full,
        y_full,
        verbose=False,
    )

    final_elapsed = (
        time.time()
        - final_start
    )

    print(
        f"전체 데이터 재학습 완료: "
        f"{final_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # 최종 Model artifact
    # -----------------------------------------------------

    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,

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

        "model_name": EXPERIMENT_NAME,

        "xgboost_version": (
            xgb.__version__
        ),
    }

    # -----------------------------------------------------
    # 모델 저장
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    shutil.copy2(
        "./train.py",
        MODEL_DIR / "train.py",
    )

    print("=" * 70)
    print("XGBoost 모델 저장 완료")

    print(
        f"저장 경로            : "
        f"{MODEL_PATH}"
    )

    print(
        f"Final iterations     : "
        f"{final_iterations}"
    )

    print(
        f"Validation Brier     : "
        f"{brier:.10f}"
    )

    print(
        f"Validation Score     : "
        f"{validation_score:.4f}"
    )

    print(
        f"train.py snapshot    : "
        f"{MODEL_DIR / 'train.py'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
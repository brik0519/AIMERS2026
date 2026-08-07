# train.py

import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


# =========================================================
# 실험 설정
# =========================================================

DATA_DIR = Path("./data")

ID = "row_id"
TARGET = "control_success"

# 실험마다 이 값만 변경
EXPERIMENT_NAME = "lgb_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42


# =========================================================
# 범주형 컬럼 정의
# =========================================================

# 저카디널리티 명목형 변수만 categorical로 처리
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

# ID 계열은 numeric으로 유지
ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]


# =========================================================
# LightGBM 모델 파라미터
# =========================================================

MODEL_PARAMS = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 200,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# =========================================================
# 사용자 정의 Brier Score
# =========================================================

def brier_metric(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    brier = np.mean(
        (y_pred - y_true) ** 2
    )

    return (
        "brier",
        float(brier),
        False,
    )


# =========================================================
# 범주형 level 생성
# =========================================================

def fit_category_levels(
    df,
    cat_cols,
):
    category_levels = {}

    for col in cat_cols:
        values = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        category_levels[col] = sorted(
            values.unique().tolist()
        )

    return category_levels


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    features,
    cat_cols,
    category_levels,
    id_like_cols,
):
    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "입력 데이터에 학습 Feature가 없습니다: "
            + ", ".join(missing_features)
        )

    X = df[
        features
    ].copy()

    # 저카디널리티 categorical 처리
    for col in cat_cols:
        values = (
            X[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        X[col] = pd.Categorical(
            values,
            categories=category_levels[col],
        )

    # 고카디널리티 ID는 numeric 유지
    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


# =========================================================
# LightGBM 모델 생성
# =========================================================

def create_model(
    n_estimators,
):
    return LGBMClassifier(
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
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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
        "positive_rate": float(np.mean(y_true)),
        "brier": float(brier),
        "baseline_brier": float(baseline_brier),
        "bss": float(bss),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
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
    # 파일 경로
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
        "season",
        TARGET,
        ID,
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
    print("LightGBM V2 데이터 로드 완료")
    print("=" * 70)
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
    # Validation category level
    # -----------------------------------------------------

    category_levels = (
        fit_category_levels(
            train_df,
            cat_cols,
        )
    )

    X_train = prepare_features(
        df=train_df,
        features=features,
        cat_cols=cat_cols,
        category_levels=category_levels,
        id_like_cols=id_like_cols,
    )

    X_val = prepare_features(
        df=val_df,
        features=features,
        cat_cols=cat_cols,
        category_levels=category_levels,
        id_like_cols=id_like_cols,
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

    estimator = create_model(
        n_estimators=MAX_ITERATIONS,
    )

    print("=" * 70)
    print(
        "LightGBM V2 Validation 학습 시작"
    )
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
    print("=" * 70)

    validation_start = time.time()

    estimator.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        eval_metric=brier_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=100,
            ),
        ],
    )

    validation_elapsed = (
        time.time()
        - validation_start
    )

    print(
        "Validation 학습 완료 : "
        f"{validation_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # Best iteration
    # -----------------------------------------------------

    best_iteration = (
        estimator.best_iteration_
    )

    if (
        best_iteration is None
        or best_iteration <= 0
    ):
        best_iteration = (
            MAX_ITERATIONS
        )

    print(
        "Best iteration       :",
        best_iteration,
    )

    # -----------------------------------------------------
    # Validation prediction
    # -----------------------------------------------------

    val_pred = estimator.predict_proba(
        X_val,
        num_iteration=best_iteration,
    )[:, 1]

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
    print(
        "LightGBM V2 Validation 결과"
    )
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
        estimator=estimator,
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
    # Validation 관련 메모리 정리
    # -----------------------------------------------------

    del X_train
    del X_val
    del train_df
    del val_df
    del estimator

    gc.collect()

    # -----------------------------------------------------
    # 전체 데이터 기준 category level 재생성
    # -----------------------------------------------------

    final_category_levels = (
        fit_category_levels(
            train,
            cat_cols,
        )
    )

    X_full = prepare_features(
        df=train,
        features=features,
        cat_cols=cat_cols,
        category_levels=final_category_levels,
        id_like_cols=id_like_cols,
    )

    y_full = train[
        TARGET
    ].copy()

    # -----------------------------------------------------
    # 최종 모델
    # -----------------------------------------------------

    final_model = create_model(
        n_estimators=best_iteration,
    )

    print("=" * 70)
    print(
        "LightGBM V2 전체 데이터 재학습 시작"
    )
    print(
        f"사용 iteration       : "
        f"{best_iteration}"
    )
    print("=" * 70)

    final_start = time.time()

    final_model.fit(
        X_full,
        y_full,
        categorical_feature=cat_cols,
    )

    final_elapsed = (
        time.time()
        - final_start
    )

    print(
        "전체 데이터 재학습 완료: "
        f"{final_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # Model artifact
    # -----------------------------------------------------

    artifact = {
        "model": final_model,
        "features": features,
        "cat_cols": cat_cols,
        "id_like_cols": id_like_cols,
        "category_levels": final_category_levels,
        "best_iteration": int(
            best_iteration
        ),
        "validation_brier": float(
            brier
        ),
        "validation_score": float(
            validation_score
        ),
        "model_name": EXPERIMENT_NAME,
    }

    # -----------------------------------------------------
    # Model 저장
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
    print(
        "LightGBM V2 모델 저장 완료"
    )
    print(
        f"저장 경로            : "
        f"{MODEL_PATH}"
    )
    print(
        f"Best iteration       : "
        f"{best_iteration}"
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
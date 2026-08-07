# train.py

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# =========================================================
# 실험 설정
# =========================================================

DATA_DIR = Path("./data")

ID = "row_id"
TARGET = "control_success"

# 실험마다 이 값만 변경
EXPERIMENT_NAME = "et_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# Feature 정책
# =========================================================

# 저카디널리티 명목형 변수만 categorical encoding
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
# ExtraTrees 설정
# =========================================================

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 50,
    "max_features": "sqrt",
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": 42,
}


# =========================================================
# 전처리 생성
# =========================================================

def create_preprocessor(
    cat_cols,
    num_cols,
):
    """
    ExtraTrees 입력 전처리 객체를 새로 생성한다.

    Validation 모델과 Final 모델이 서로 독립된
    fitted preprocessing state를 갖도록 매번 새 객체를 반환한다.
    """

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
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
                        ),
                    ),
                ]),
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


# =========================================================
# ExtraTrees 모델 생성
# =========================================================

def create_model(
    cat_cols,
    num_cols,
):
    """
    새로운 전처리 객체와 새로운 ExtraTreesClassifier를 생성하여
    독립적인 Pipeline을 반환한다.
    """

    preprocessor = create_preprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

    estimator = ExtraTreesClassifier(
        **MODEL_PARAMS
    )

    model = Pipeline([
        (
            "pre",
            preprocessor,
        ),
        (
            "clf",
            estimator,
        ),
    ])

    return model, estimator


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
    print(f"- {output_dir / 'pred.csv'}")
    print(f"- {output_dir / 'validation.npz'}")
    print(f"- {output_dir / 'metadata.json'}")


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

    TEST_PATH = DATA_DIR / "test.csv"
    TRAIN_PATH = DATA_DIR / "train.csv"

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
    print("ExtraTrees 데이터 로드 완료")
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

    X_train = train.loc[
        ~is_val,
        features,
    ]

    y_train = train.loc[
        ~is_val,
        TARGET,
    ]

    X_val = train.loc[
        is_val,
        features,
    ]

    y_val = train.loc[
        is_val,
        TARGET,
    ]

    val_row_id = train.loc[
        is_val,
        ID,
    ]

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
    # Validation 모델 생성
    # -----------------------------------------------------

    model, estimator = create_model(
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

    # -----------------------------------------------------
    # Validation 학습
    # -----------------------------------------------------

    print("=" * 70)
    print("ExtraTrees Validation 학습 시작")
    print("=" * 70)

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    validation_elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Validation 학습 완료 : "
        f"{validation_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # Validation prediction
    # -----------------------------------------------------

    val_pred = model.predict_proba(
        X_val
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
    print("ExtraTrees Validation 결과")
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
    )

    # -----------------------------------------------------
    # 전체 데이터 재학습용 모델 생성
    # -----------------------------------------------------

    final_model, _ = create_model(
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

    # -----------------------------------------------------
    # 전체 데이터 재학습
    # -----------------------------------------------------

    print("=" * 70)
    print("ExtraTrees 전체 데이터 재학습 시작")
    print("=" * 70)

    start_time = time.time()

    final_model.fit(
        train[features],
        train[TARGET],
    )

    final_elapsed = (
        time.time()
        - start_time
    )

    print(
        f"전체 데이터 재학습 완료: "
        f"{final_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # 최종 모델 저장
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
        compress=3,
    )

    # 해당 모델을 생성한 train.py snapshot
    shutil.copy2(
        "./train.py",
        MODEL_DIR / "train.py",
    )

    print("=" * 70)
    print("ExtraTrees 모델 저장 완료")

    print(
        f"저장 경로            : "
        f"{MODEL_PATH}"
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
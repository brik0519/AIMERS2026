# script.py: XGBoost xgb_v002 inference
# train.py에서 저장한 최종 artifact(model + preprocessor)를 로드하여
# test.csv 각 행을 독립적으로 추론하고 submission.csv를 생성한다.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 실험 설정
# =========================================================

EXPERIMENT_NAME = "xgb_v002"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_PATH = MODEL_ROOT / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"


# =========================================================
# Raw Feature 준비
# =========================================================

def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    """
    train.py와 동일한 XGBoost 전처리 직전 DataFrame을 생성한다.

    1. 학습 Feature 순서 고정
    2. ID 계열은 numeric으로 강제 변환
    3. 나머지 numeric NaN은 그대로 유지
       → 저장된 preprocessor와 XGBoost가 처리

    test의 다른 행 또는 test 전체 통계는 사용하지 않는다.
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
# Prediction 검증
# =========================================================

def validate_predictions(
    predictions,
    n_expected,
):
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(predictions) != n_expected:
        raise RuntimeError(
            "예측 수와 test 행 수가 다릅니다."
        )

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "예측값에 NaN 또는 무한대가 있습니다."
        )

    if not (
        (predictions >= 0.0)
        & (predictions <= 1.0)
    ).all():
        raise RuntimeError(
            "예측값이 0~1 범위를 벗어났습니다."
        )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 파일 검증
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: "
            f"{MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv를 찾을 수 없습니다: "
            f"{TEST_PATH}"
        )

    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(
            "sample_submission.csv를 "
            "찾을 수 없습니다: "
            f"{SAMPLE_SUB_PATH}"
        )

    # -----------------------------------------------------
    # Final artifact 로드
    # -----------------------------------------------------

    print("=" * 70)
    print("XGBoost xgb_v002 Inference")
    print("=" * 70)

    artifact = joblib.load(
        MODEL_PATH
    )

    required_keys = {
        "model",
        "preprocessor",
        "features",
        "cat_cols",
        "id_like_cols",
        "num_cols",
    }

    missing_keys = (
        required_keys
        - set(artifact.keys())
    )

    if missing_keys:
        raise ValueError(
            "모델 artifact에 필요한 key가 없습니다: "
            + ", ".join(
                sorted(
                    missing_keys
                )
            )
        )

    model = artifact[
        "model"
    ]

    preprocessor = artifact[
        "preprocessor"
    ]

    features = artifact[
        "features"
    ]

    cat_cols = artifact[
        "cat_cols"
    ]

    id_like_cols = artifact[
        "id_like_cols"
    ]

    num_cols = artifact[
        "num_cols"
    ]

    print(
        f"Model path           : "
        f"{MODEL_PATH}"
    )

    print(
        f"Model name           : "
        f"{artifact.get('model_name', EXPERIMENT_NAME)}"
    )

    print(
        f"Final iterations     : "
        f"{artifact.get('final_iterations', '?')}"
    )

    print(
        f"Validation method    : "
        f"{artifact.get('validation_method', '?')}"
    )

    print(
        f"Feature 수           : "
        f"{len(features)}"
    )

    print(
        f"Categorical 수       : "
        f"{len(cat_cols)}"
    )

    print(
        f"Numeric 수           : "
        f"{len(num_cols)}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # test / sample_submission 로드
    # -----------------------------------------------------

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    sample_submission = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    if ID not in test.columns:
        raise ValueError(
            f"test.csv에 ID 컬럼 "
            f"'{ID}'가 없습니다."
        )

    if ID not in sample_submission.columns:
        raise ValueError(
            "sample_submission.csv에 ID 컬럼 "
            f"'{ID}'가 없습니다."
        )

    if TARGET not in sample_submission.columns:
        raise ValueError(
            "sample_submission.csv에 Target 컬럼 "
            f"'{TARGET}'가 없습니다."
        )

    if len(test) != len(
        sample_submission
    ):
        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "행 수가 일치하지 않습니다."
        )

    if not test[
        ID
    ].reset_index(
        drop=True
    ).equals(
        sample_submission[
            ID
        ].reset_index(
            drop=True
        )
    ):
        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "row_id 순서가 일치하지 않습니다."
        )

    print(
        f"Test shape           : "
        f"{test.shape}"
    )

    # -----------------------------------------------------
    # Raw Feature 준비
    # -----------------------------------------------------

    X_test_raw = prepare_raw_features(
        df=test,
        features=features,
        id_like_cols=id_like_cols,
    )

    # -----------------------------------------------------
    # 저장된 Final preprocessor 적용
    #
    # fit / fit_transform 금지.
    # 공식 train 전체로 이미 fit된 객체의 transform만 사용한다.
    # -----------------------------------------------------

    X_test = (
        preprocessor
        .transform(
            X_test_raw
        )
    )

    print(
        f"Transformed shape    : "
        f"{X_test.shape}"
    )

    # -----------------------------------------------------
    # 추론
    #
    # 각 행의 예측에는 해당 행의 입력값과
    # 공식 train으로 학습된 artifact만 사용한다.
    # 다른 test 행의 값/통계/순서/분포를 사용하지 않는다.
    # -----------------------------------------------------

    predictions = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    validate_predictions(
        predictions=predictions,
        n_expected=len(test),
    )

    print()
    print("=" * 70)
    print("Inference 결과")
    print("=" * 70)

    print(
        f"Prediction mean      : "
        f"{predictions.mean():.6f}"
    )

    print(
        f"Prediction std       : "
        f"{predictions.std():.6f}"
    )

    print(
        f"Prediction min       : "
        f"{predictions.min():.6f}"
    )

    print(
        f"Prediction max       : "
        f"{predictions.max():.6f}"
    )

    # -----------------------------------------------------
    # submission.csv 생성
    # -----------------------------------------------------

    submission = (
        sample_submission
        .copy()
    )

    submission[
        TARGET
    ] = predictions

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Submission 저장 완료")
    print("=" * 70)

    print(
        f"저장 경로            : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"행 수                : "
        f"{len(submission)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

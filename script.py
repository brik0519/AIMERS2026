# script.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 모델 탐색
# =========================================================

def find_model():
    """
    로컬 개발 환경:
        project root의 train.py에서 MODEL_PATH를 가져온다.

    제출 환경:
        train.py가 없으므로 ./model/*.pkl 단일 파일을 자동 탐색한다.
    """

    try:
        import train

        model_path = Path(
            train.MODEL_PATH
        )

        if model_path.exists():
            return model_path

    except ImportError:
        pass

    model_files = sorted(
        MODEL_DIR.glob("*.pkl")
    )

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: "
            f"{MODEL_DIR}"
        )

    if len(model_files) > 1:
        raise RuntimeError(
            f"model/에 .pkl 파일이 여러 개 있습니다: "
            f"{model_files}"
        )

    return model_files[0]


# =========================================================
# 데이터 로드
# =========================================================

def load_test(path):
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    if ID_COL not in df.columns:
        raise ValueError(
            f"test 데이터에 {ID_COL} 컬럼이 없습니다."
        )

    return df


def load_sample_submission(path):
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    expected_cols = [
        ID_COL,
        TARGET_COL,
    ]

    if list(df.columns) != expected_cols:
        raise ValueError(
            f"sample_submission 컬럼이 "
            f"{expected_cols}가 아닙니다: "
            f"{list(df.columns)}"
        )

    return df


# =========================================================
# Raw Feature 준비
# =========================================================

def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    """
    train.py와 동일하게 XGBoost 전처리 직전
    Raw Feature DataFrame을 생성한다.

    1. 학습 당시 Feature 순서 유지
    2. ID 계열은 numeric으로 강제 변환
    3. numeric NaN은 그대로 유지
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
# Submission 생성
# =========================================================

def build_submission(
    sample_submission,
    ids,
    predictions,
):
    pred_df = pd.DataFrame({
        ID_COL: ids,
        TARGET_COL: predictions,
    })

    submission = sample_submission[
        [ID_COL]
    ].merge(
        pred_df,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    n_missing = submission[
        TARGET_COL
    ].isna().sum()

    if n_missing > 0:
        raise ValueError(
            f"예측값이 없는 row_id가 "
            f"{n_missing}건 존재합니다."
        )

    return submission


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 모델 artifact 로드
    # -----------------------------------------------------

    model_path = find_model()

    print(
        f"[Model artifact] {model_path}"
    )

    artifact = joblib.load(
        model_path
    )

    # -----------------------------------------------------
    # Artifact 검증
    # -----------------------------------------------------

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "XGBoost 모델 파일은 "
            "artifact dictionary여야 합니다."
        )

    required_keys = {
        "model",
        "preprocessor",
        "features",
        "cat_cols",
        "id_like_cols",
        "num_cols",
        "best_iteration",
        "final_iterations",
    }

    missing_keys = (
        required_keys
        - set(artifact.keys())
    )

    if missing_keys:
        raise ValueError(
            "모델 artifact에 필요한 항목이 없습니다: "
            + ", ".join(
                sorted(missing_keys)
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

    best_iteration = artifact[
        "best_iteration"
    ]

    final_iterations = artifact[
        "final_iterations"
    ]

    print(
        f"model_name="
        f"{artifact.get('model_name', '?')}"
    )

    print(
        f"xgboost_version="
        f"{artifact.get('xgboost_version', '?')}"
    )

    print(
        f"best_iteration="
        f"{best_iteration}"
    )

    print(
        f"final_iterations="
        f"{final_iterations}"
    )

    print(
        f"features={len(features)}"
    )

    print(
        f"categorical={cat_cols}"
    )

    print(
        f"numeric_id={id_like_cols}"
    )

    print(
        f"numeric={len(num_cols)}"
    )

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------

    test = load_test(
        TEST_PATH
    )

    sample_submission = (
        load_sample_submission(
            SAMPLE_SUB_PATH
        )
    )

    print(
        f"test={len(test)}"
        f" | submission="
        f"{len(sample_submission)}"
    )

    # -----------------------------------------------------
    # row_id 분리
    # -----------------------------------------------------

    ids = test[
        ID_COL
    ].to_numpy()

    # -----------------------------------------------------
    # Raw Feature 생성
    # -----------------------------------------------------

    X_raw = prepare_raw_features(
        df=test,
        features=features,
        id_like_cols=id_like_cols,
    )

    # -----------------------------------------------------
    # 학습 당시 fitted preprocessor 적용
    # -----------------------------------------------------

    X = preprocessor.transform(
        X_raw
    )

    print(
        f"prepared features="
        f"{X.shape[1]}"
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    predictions = model.predict_proba(
        X
    )[:, 1]

    # -----------------------------------------------------
    # Prediction 검증
    # -----------------------------------------------------

    if len(predictions) != len(test):
        raise RuntimeError(
            "Test 예측 수와 입력 row 수가 다릅니다."
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

    print(
        f"predictions="
        f"{len(predictions)}"
    )

    print(
        f"prediction mean="
        f"{predictions.mean():.6f}"
        f" | std="
        f"{predictions.std():.6f}"
    )

    print(
        f"prediction min="
        f"{predictions.min():.6f}"
        f" | max="
        f"{predictions.max():.6f}"
    )

    # -----------------------------------------------------
    # Submission 생성
    # -----------------------------------------------------

    submission = build_submission(
        sample_submission=sample_submission,
        ids=ids,
        predictions=predictions,
    )

    # -----------------------------------------------------
    # 저장
    # -----------------------------------------------------

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
    print("[Submission 저장 완료]")
    print(f"- {OUTPUT_PATH}")
    print(f"- rows: {len(submission)}")


if __name__ == "__main__":
    main()
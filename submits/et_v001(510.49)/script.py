# script.py - ExtraTrees

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

        model_path = Path(train.MODEL_PATH)

        if model_path.exists():
            return model_path

    except ImportError:
        pass

    model_files = sorted(
        MODEL_DIR.glob("*.pkl")
    )

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_DIR}"
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
# Feature 생성
# =========================================================

def build_features(df):
    """
    train.py와 동일하게 row_id를 모델 입력에서 제외한다.

    ExtraTrees 모델은 sklearn Pipeline으로 저장되어 있으므로
    categorical encoding과 missing-value imputation은
    저장된 모델 내부에서 자동 수행된다.
    """

    return df.drop(
        columns=[ID_COL]
    )


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
    # 모델 탐색 / 로드
    # -----------------------------------------------------

    model_path = find_model()

    print(
        f"[Model] {model_path}"
    )

    model = joblib.load(
        model_path
    )

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------

    test = load_test(
        TEST_PATH
    )

    sample_submission = load_sample_submission(
        SAMPLE_SUB_PATH
    )

    print(
        f"test={len(test)}"
        f" | submission={len(sample_submission)}"
    )

    # -----------------------------------------------------
    # Feature 생성
    # -----------------------------------------------------

    ids = test[
        ID_COL
    ].to_numpy()

    X = build_features(
        test
    )

    print(
        f"features={X.shape[1]}"
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
        f"predictions={len(predictions)}"
    )

    print(
        f"prediction mean={predictions.mean():.6f}"
        f" | std={predictions.std():.6f}"
    )

    print(
        f"prediction min={predictions.min():.6f}"
        f" | max={predictions.max():.6f}"
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
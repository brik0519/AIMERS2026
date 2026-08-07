# script.py

from pathlib import Path

import joblib
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
    개발 환경:
        train.py의 MODEL_PATH 사용

    제출 환경:
        train.py가 없으므로 model/*.pkl 단일 파일 자동 탐색
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
            "model/에 .pkl 파일이 여러 개 있습니다: "
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
# Feature
# =========================================================

def build_features(df):
    return df.drop(
        columns=[ID_COL]
    )


# =========================================================
# Submission
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
    # 모델
    # -----------------------------------------------------

    model_path = find_model()

    print(
        f"[Model] {model_path}"
    )

    model = joblib.load(
        model_path
    )

    # -----------------------------------------------------
    # 데이터
    # -----------------------------------------------------

    test = load_test(
        TEST_PATH
    )

    sample_submission = load_sample_submission(
        SAMPLE_SUB_PATH
    )

    ids = test[
        ID_COL
    ].to_numpy()

    X = build_features(
        test
    )

    print(
        f"test: {len(test)}"
        f" | features: {X.shape[1]}"
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    predictions = model.predict_proba(
        X
    )[:, 1]

    print(
        f"prediction mean: {predictions.mean():.6f}"
        f" | std: {predictions.std():.6f}"
    )

    # -----------------------------------------------------
    # Submission
    # -----------------------------------------------------

    submission = build_submission(
        sample_submission=sample_submission,
        ids=ids,
        predictions=predictions,
    )

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


if __name__ == "__main__":
    main()
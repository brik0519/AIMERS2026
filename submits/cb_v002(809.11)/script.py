# script.py: CatBoost cb_v002 inference
# train.py에서 저장한 Final Model을 로드하여 test.csv를 추론하고
# sample_submission.csv 형식의 submission.csv를 생성한다.

from pathlib import Path

import joblib
import pandas as pd


# =========================================================
# 실험 설정
# =========================================================

EXPERIMENT_NAME = "cb_v002"

DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"

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
# Feature 전처리
# =========================================================

def prepare_features(df, cat_cols):
    """
    train.py와 동일한 전처리.

    - 수치형 결측값은 CatBoost가 직접 처리
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 문자열 dtype으로 통일
    """

    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 경로 확인
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_PATH}"
        )

    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(
            "Sample submission file not found: "
            f"{SAMPLE_SUB_PATH}"
        )

    # -----------------------------------------------------
    # 모델 로드
    # -----------------------------------------------------

    model = joblib.load(MODEL_PATH)

    print("=" * 70)
    print("모델 로드 완료")
    print(f"- {MODEL_PATH}")
    print("=" * 70)

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    sample_submission = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    features = [
        c for c in test.columns
        if c != ID
    ]

    cat_cols = [
        c for c in CAT_COLS
        if c in features
    ]

    X_test = prepare_features(
        test[features],
        cat_cols,
    )

    print("test shape:", test.shape)
    print("Feature 수:", len(features))
    print("범주형 Feature 수:", len(cat_cols))
    print("범주형 Feature:", cat_cols)

    # -----------------------------------------------------
    # 추론
    # -----------------------------------------------------

    prediction = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print("=" * 70)
    print("Inference 결과")
    print("=" * 70)
    print(
        f"Prediction mean : "
        f"{prediction.mean():.6f}"
    )
    print(
        f"Prediction std  : "
        f"{prediction.std():.6f}"
    )
    print(
        f"Prediction min  : "
        f"{prediction.min():.6f}"
    )
    print(
        f"Prediction max  : "
        f"{prediction.max():.6f}"
    )

    # -----------------------------------------------------
    # Submission 생성
    # -----------------------------------------------------

    if len(sample_submission) != len(test):
        raise ValueError(
            "sample_submission.csv와 test.csv의 "
            "행 수가 일치하지 않습니다."
        )

    if ID in sample_submission.columns:
        if not sample_submission[ID].equals(
            test[ID]
        ):
            raise ValueError(
                "sample_submission.csv와 test.csv의 "
                "row_id 순서가 일치하지 않습니다."
            )

    target_columns = [
        c for c in sample_submission.columns
        if c != ID
    ]

    if len(target_columns) != 1:
        raise ValueError(
            "sample_submission.csv에서 "
            "예측 대상 컬럼을 하나로 식별할 수 없습니다."
        )

    target_column = target_columns[0]

    submission = sample_submission.copy()
    submission[target_column] = prediction

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
    print(f"- {OUTPUT_PATH}")
    print(
        f"- target column: {target_column}"
    )
    print(
        f"- rows: {len(submission)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

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
            f"model/에 .pkl 파일이 여러 개 있습니다: {model_files}"
        )

    return model_files[0]


# =========================================================
# 데이터 로드
# =========================================================

def load_test(path):
    """
    평가 데이터 로드.
    """

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
    """
    sample_submission.csv 로드.
    """

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
# CatBoost Feature 전처리
# =========================================================

def build_features(df):
    """
    CatBoost train.py의 prepare_features()와 동일한 전처리를 수행한다.

    1. row_id 제거
    2. 실제 존재하는 categorical column만 사용
    3. categorical NaN -> "__MISSING__"
    4. categorical dtype -> str

    수치형 결측값은 CatBoost가 직접 처리한다.
    """

    X = df.drop(
        columns=[ID_COL]
    ).copy()

    cat_cols = [
        c for c in CAT_COLS
        if c in X.columns
    ]

    for col in cat_cols:
        X[col] = (
            X[col]
            .fillna("__MISSING__")
            .astype(str)
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
    """
    sample_submission의 row_id 순서를 기준으로
    prediction을 one-to-one merge한다.
    """

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
            f"예측값이 없는 row_id가 {n_missing}건 존재합니다."
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

    print(
        f"predictions={len(predictions)}"
    )

    print(
        f"prediction mean={predictions.mean():.6f}"
        f" | std={predictions.std():.6f}"
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
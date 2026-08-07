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
# LightGBM Feature 전처리
# =========================================================

def prepare_features(
    df,
    features,
    cat_cols,
    category_levels,
    id_like_cols,
):
    """
    train.py와 동일한 LightGBM V2 전처리를 수행한다.

    1. 학습 시점 Feature 순서로 정렬
    2. 저카디널리티 범주형만 pd.Categorical 처리
    3. 학습 시 저장한 category level을 그대로 적용
    4. ID 계열은 numeric 유지
    5. 수치형 결측값은 LightGBM이 직접 처리
    """

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

    # -----------------------------------------------------
    # 저카디널리티 categorical
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ID 계열 numeric 유지
    # -----------------------------------------------------

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
    # artifact 구조 검증
    # -----------------------------------------------------

    required_keys = {
        "model",
        "features",
        "cat_cols",
        "id_like_cols",
        "category_levels",
        "best_iteration",
    }

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "LightGBM 모델 파일은 "
            "artifact dictionary여야 합니다."
        )

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

    features = artifact[
        "features"
    ]

    cat_cols = artifact[
        "cat_cols"
    ]

    id_like_cols = artifact[
        "id_like_cols"
    ]

    category_levels = artifact[
        "category_levels"
    ]

    best_iteration = artifact[
        "best_iteration"
    ]

    print(
        f"model_name={artifact.get('model_name', '?')}"
    )

    print(
        f"best_iteration={best_iteration}"
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

    X = prepare_features(
        df=test,
        features=features,
        cat_cols=cat_cols,
        category_levels=category_levels,
        id_like_cols=id_like_cols,
    )

    print(
        f"prepared features={X.shape[1]}"
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    predictions = model.predict_proba(
        X,
        num_iteration=best_iteration,
    )[:, 1]

    # -----------------------------------------------------
    # Prediction 검증
    # -----------------------------------------------------

    if len(predictions) != len(test):
        raise RuntimeError(
            "Test 예측 수와 입력 row 수가 다릅니다."
        )

    if not pd.Series(
        predictions
    ).map(
        lambda x: pd.notna(x)
    ).all():
        raise RuntimeError(
            "예측값에 NaN이 존재합니다."
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
    print(
        f"- {OUTPUT_PATH}"
    )
    print(
        f"- rows: {len(submission)}"
    )


if __name__ == "__main__":
    main()
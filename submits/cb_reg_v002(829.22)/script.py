# script.py: CatBoost Regressor cb_reg_v002 추론 / 제출 생성
#
# cb_reg_v002/train.py와 동일한 Feature 전처리를 사용한다.
# CatBoostRegressor의 raw prediction은 [0, 1] 범위를 벗어날 수 있으므로
# Validation 정책과 동일하게 np.clip(pred, 0.0, 1.0)을 적용한다.
#
# 평가 데이터 독립 추론 규칙:
# - 각 test 행의 예측은 해당 행의 입력값만 사용한다.
# - 다른 test 행의 통계, 집계, rolling, lag, rank 등을 사용하지 않는다.
# - test 전체 분포를 이용한 보정도 수행하지 않는다.

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v002"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 데이터 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"

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
# 데이터 로드
# =========================================================

def load_test(path):
    """
    평가 데이터를 그대로 로드한다.

    주의:
    - test 전체 통계/분포를 Feature로 만들지 않는다.
    - 행 간 정보 공유를 하지 않는다.
    """

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    if ID not in df.columns:
        raise ValueError(
            f"test 데이터에 {ID} 컬럼이 없습니다."
        )

    if df[ID].duplicated().any():
        raise ValueError(
            f"test 데이터의 {ID}에 중복이 있습니다."
        )

    return df


def load_sample_submission(path):
    """
    sample_submission.csv를 로드한다.
    제출 컬럼 및 row_id 순서를 검증하는 용도로만 사용한다.
    """

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    expected_cols = [
        ID,
        TARGET,
    ]

    if list(df.columns) != expected_cols:
        raise ValueError(
            "sample_submission 컬럼이 예상과 다릅니다.\n"
            f"expected={expected_cols}\n"
            f"actual={list(df.columns)}"
        )

    if df[ID].duplicated().any():
        raise ValueError(
            f"sample_submission의 {ID}에 중복이 있습니다."
        )

    return df


# =========================================================
# 학습과 동일한 Feature 전처리
# =========================================================

def prepare_features(df, cat_cols):
    """
    cb_reg_v002/train.py와 동일한 전처리.

    - 수치형 결측값은 CatBoost가 직접 처리
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 문자열 dtype으로 통일

    모든 처리는 각 행의 값만 사용하므로
    평가 데이터 독립 추론 규칙을 위반하지 않는다.
    """

    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


def build_features(test):
    """
    cb_reg_v002/train.py와 동일하게
    test.csv의 row_id를 제외한 모든 컬럼을 Feature로 사용한다.

    test 행 간 집계/통계는 생성하지 않는다.
    """

    features = [
        col
        for col in test.columns
        if col != ID
    ]

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    X = prepare_features(
        test[features],
        cat_cols,
    )

    return X, features, cat_cols


# =========================================================
# Prediction 진단
# =========================================================

def print_prediction_diagnostics(
    pred_raw,
    pred_clip,
):
    """
    CatBoostRegressor 출력의 범위 및 분포를 확인한다.

    이 통계는 출력 진단용일 뿐,
    개별 test prediction을 변경하는 데 사용하지 않는다.
    """

    pred_raw = np.asarray(
        pred_raw,
        dtype=np.float64,
    )

    pred_clip = np.asarray(
        pred_clip,
        dtype=np.float64,
    )

    below_zero = (
        pred_raw < 0.0
    )

    above_one = (
        pred_raw > 1.0
    )

    outside_range = (
        below_zero
        | above_one
    )

    print("=" * 70)
    print("Prediction diagnostics")
    print("=" * 70)

    print("[Raw prediction]")
    print(
        f"mean                 : {pred_raw.mean():.6f}"
    )
    print(
        f"std                  : {pred_raw.std():.6f}"
    )
    print(
        f"min                  : {pred_raw.min():.6f}"
    )
    print(
        f"max                  : {pred_raw.max():.6f}"
    )

    print()

    print("[Range diagnostics]")
    print(
        f"prediction < 0       : "
        f"{below_zero.sum()} "
        f"({below_zero.mean():.6%})"
    )
    print(
        f"prediction > 1       : "
        f"{above_one.sum()} "
        f"({above_one.mean():.6%})"
    )
    print(
        f"outside [0, 1]       : "
        f"{outside_range.sum()} "
        f"({outside_range.mean():.6%})"
    )

    print()

    print("[Clipped prediction]")
    print(
        f"mean                 : {pred_clip.mean():.6f}"
    )
    print(
        f"std                  : {pred_clip.std():.6f}"
    )
    print(
        f"min                  : {pred_clip.min():.6f}"
    )
    print(
        f"max                  : {pred_clip.max():.6f}"
    )

    print("=" * 70)


# =========================================================
# Submission 생성
# =========================================================

def build_submission(
    sample_submission,
    test_ids,
    predictions,
):
    """
    test row_id별 예측을 sample_submission 순서에 맞춰 병합한다.

    row_id는 정렬/제출 대응에만 사용하며,
    다른 test 행의 정보를 Feature나 예측값 계산에 사용하지 않는다.
    """

    pred_df = pd.DataFrame({
        ID: test_ids,
        TARGET: predictions,
    })

    if pred_df[ID].duplicated().any():
        raise ValueError(
            "예측 row_id에 중복이 있습니다."
        )

    sample_ids = set(
        sample_submission[ID]
    )

    pred_ids = set(
        pred_df[ID]
    )

    missing_ids = (
        sample_ids
        - pred_ids
    )

    extra_ids = (
        pred_ids
        - sample_ids
    )

    if missing_ids:
        raise ValueError(
            "sample_submission에 있으나 "
            "test prediction에 없는 row_id가 있습니다. "
            f"count={len(missing_ids)}"
        )

    if extra_ids:
        raise ValueError(
            "test prediction에는 있으나 "
            "sample_submission에 없는 row_id가 있습니다. "
            f"count={len(extra_ids)}"
        )

    submission = (
        sample_submission[
            [ID]
        ]
        .merge(
            pred_df,
            on=ID,
            how="left",
            validate="one_to_one",
        )
    )

    if submission[TARGET].isna().any():
        raise ValueError(
            "제출 prediction에 NaN이 있습니다."
        )

    if not np.isfinite(
        submission[TARGET].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "제출 prediction에 Inf 또는 NaN이 있습니다."
        )

    if not (
        submission[TARGET]
        .between(0.0, 1.0)
        .all()
    ):
        raise ValueError(
            "제출 prediction에 [0, 1] 범위를 "
            "벗어난 값이 있습니다."
        )

    return submission


def save_submission(
    path,
    submission,
):
    """
    제출 파일을 submission.csv로 저장한다.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 모델 로드
    # -----------------------------------------------------

    print("=" * 70)
    print("CatBoostRegressor inference")
    print("=" * 70)

    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Model path : {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    print()
    print("모델 로드...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"모델 타입: {type(model).__name__}"
    )

    if type(model).__name__ != "CatBoostRegressor":
        raise TypeError(
            "cb_reg_v002는 CatBoostRegressor 모델을 "
            "사용해야 합니다.\n"
            f"actual={type(model).__name__}"
        )

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------

    print()
    print("Test 데이터 로드...")

    test = load_test(
        TEST_PATH
    )

    sample_submission = (
        load_sample_submission(
            SAMPLE_SUB_PATH
        )
    )

    print(
        f"test rows             : {len(test)}"
    )

    print(
        f"submission rows       : {len(sample_submission)}"
    )

    # -----------------------------------------------------
    # Feature 구성
    # -----------------------------------------------------

    print()
    print("Feature 구성...")

    X_test, features, cat_cols = (
        build_features(
            test
        )
    )

    test_ids = (
        test[ID]
        .to_numpy()
    )

    print(
        f"Feature 수            : {len(features)}"
    )

    print(
        f"범주형 Feature 수     : {len(cat_cols)}"
    )

    print(
        f"범주형 Feature        : {cat_cols}"
    )

    # -----------------------------------------------------
    # 모델 Feature 검증
    # -----------------------------------------------------

    model_feature_names = getattr(
        model,
        "feature_names_",
        None,
    )

    if model_feature_names:
        model_feature_names = list(
            model_feature_names
        )

        if model_feature_names != features:
            raise ValueError(
                "학습 모델의 Feature 순서와 "
                "test Feature 순서가 다릅니다.\n"
                f"model={model_feature_names}\n"
                f"test={features}"
            )

    # -----------------------------------------------------
    # 추론
    # -----------------------------------------------------

    print()
    print("Inference 시작...")

    if len(X_test) == 0:
        raise ValueError(
            "test 데이터가 비어 있습니다."
        )

    # CatBoostRegressor이므로 predict_proba가 아니라 predict 사용
    pred_raw = model.predict(
        X_test
    )

    pred_raw = np.asarray(
        pred_raw,
        dtype=np.float64,
    )

    if len(pred_raw) != len(test):
        raise ValueError(
            "예측 개수와 test 행 수가 다릅니다.\n"
            f"prediction={len(pred_raw)}, "
            f"test={len(test)}"
        )

    if not np.isfinite(
        pred_raw
    ).all():
        raise ValueError(
            "모델 예측에 NaN 또는 Inf가 있습니다."
        )

    # cb_reg_v002 Validation 정책과 동일한 후처리.
    #
    # 각 행의 raw prediction에 고정된 deterministic transform만
    # 적용하므로 다른 test 행의 정보는 사용하지 않는다.
    pred_clip = np.clip(
        pred_raw,
        0.0,
        1.0,
    )

    print(
        f"predictions           : {len(pred_clip)}"
    )

    # -----------------------------------------------------
    # Prediction 진단
    # -----------------------------------------------------

    print()
    print_prediction_diagnostics(
        pred_raw,
        pred_clip,
    )

    # -----------------------------------------------------
    # Submission 생성
    # -----------------------------------------------------

    print()
    print("Submission 생성...")

    submission = build_submission(
        sample_submission=sample_submission,
        test_ids=test_ids,
        predictions=pred_clip,
    )

    save_submission(
        OUTPUT_PATH,
        submission,
    )

    print()
    print("=" * 70)
    print("제출 파일 저장 완료")
    print(
        f"- {OUTPUT_PATH}"
    )
    print(
        f"- rows: {len(submission)}"
    )
    print(
        f"- prediction mean: "
        f"{submission[TARGET].mean():.6f}"
    )
    print(
        f"- prediction min : "
        f"{submission[TARGET].min():.6f}"
    )
    print(
        f"- prediction max : "
        f"{submission[TARGET].max():.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

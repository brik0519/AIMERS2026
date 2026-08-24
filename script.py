# script.py: cb_v004 추론 및 제출
#
# train.py(cb_v004)에서 생성된 단일 Artifact(model/cb_v004/cb_v004.pkl)를
# 로드하여 test.csv 각 행을 독립적으로 추론하고 output/submission.csv를
# 생성한다.
#
# cb_v004 train.py는 categorical 표현(CAT_FULL9/CAT_NOID7/CAT_MIN3 중 하나)과
# Temporal10 채택 여부를 실행마다 데이터로부터 탐색해 결정하므로, 이 파일은
# 그 결과를 하드코딩하지 않고 Artifact에 저장된 값(features/cat_cols/
# use_temporal_features)을 그대로 읽어 동일하게 재현한다.
#
# 규칙 준수:
# - test.csv는 이 파일에서만 읽으며, 각 행은 독립적으로 추론된다.
# - Temporal10(채택된 경우)은 해당 행의 asof_* 컬럼만으로 계산되는 결정론적
#   함수이며, test 내 다른 행이나 분포 통계를 전혀 참조하지 않는다.
# - test 행 간 groupby/rolling/rank/평균 등 어떤 형태의 집계도 사용하지 않는다.

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 경로 (표준 규칙 준수)
# =========================================================
EXPERIMENT_NAME = "cb_v004_claude"

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
# 기본 설정
# =========================================================
ID = "row_id"
TARGET = "control_success"

# Temporal10 계산에 필요한 원본 asof_* 소스 컬럼.
# train.py의 add_temporal_features()와 완전히 동일한 정의를 유지해야 한다.
TEMPORAL_SOURCE_COLUMNS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]

EXPECTED_TEMPORAL_FEATURES = [
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
    "success_trend_1_3",
    "success_trend_3_5",
    "middle_trend_1_3",
    "middle_trend_3_5",
]


# =========================================================
# Feature Engineering (train.py와 완전히 동일한 정의)
# =========================================================
def add_temporal_features(df):
    """
    Temporal10: 해당 행의 asof_* 컬럼만으로 계산되는 결정론적 차이 Feature.
    train.py의 동일 함수와 로직이 반드시 일치해야 한다. 다른 행을 참조하지
    않으므로 test 행 독립성 규칙을 만족한다.
    """
    df = df.copy()

    df["recent1_success_vs_career"] = (
        df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    )
    df["recent3_success_vs_career"] = (
        df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    )
    df["recent5_success_vs_career"] = (
        df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]
    )
    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    )
    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    )
    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    )
    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )
    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )
    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )
    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


def prepare_matrix(df, feature_cols, cat_cols):
    """
    feature_cols만 선택하고, cat_cols에 속한 컬럼만 결측치를 채운 뒤
    문자열로 변환한다 (train.py와 완전히 동일한 규약). cat_cols에 속하지
    않는 수치형 컬럼은 원본 그대로 두어 CatBoost가 NaN을 native하게
    처리하도록 한다.
    """
    X = df[feature_cols].copy()
    for col in cat_cols:
        X[col] = X[col].fillna("__MISSING__").astype(str)
    return X


# =========================================================
# 제출 파일 생성
# =========================================================
def build_submission(sample_submission, test_ids, predictions):
    """
    row_id 기준으로 예측값을 매핑하고 제출 형식을 철저히 검증한다.
    """
    pred_df = pd.DataFrame({ID: test_ids, TARGET: predictions})

    if pred_df[ID].duplicated().any():
        raise ValueError("예측 row_id에 중복이 있습니다.")

    submission = sample_submission[[ID]].merge(
        pred_df,
        on=ID,
        how="left",
        validate="one_to_one",
    )

    if submission[TARGET].isna().any():
        raise ValueError("제출 prediction에 NaN이 있습니다.")

    if not np.isfinite(submission[TARGET].to_numpy(dtype=np.float64)).all():
        raise ValueError("제출 prediction에 NaN 또는 Inf가 있습니다.")

    if not submission[TARGET].between(0.0, 1.0).all():
        raise ValueError("제출 prediction에 [0, 1] 범위를 벗어난 값이 있습니다.")

    return submission


# =========================================================
# Main
# =========================================================
def main():
    start_time = time.time()

    print("=" * 60)
    print(f"CatBoost Classifier Inference : {EXPERIMENT_NAME}")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. 단일 Artifact 로드
    # -----------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 Artifact 파일을 찾을 수 없습니다: {MODEL_PATH}")

    print("모델 Artifact 로드 중...")
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    final_features = artifact["features"]
    cat_cols = artifact["cat_cols"]
    use_temporal = bool(artifact.get("use_temporal_features", False))
    cat_set_name = artifact.get("cat_set_name", "UNKNOWN")

    print(f"- 모델 타입          : {type(model).__name__}")
    print(f"- Feature 수         : {len(final_features)}")
    print(f"- Categorical Feature : {len(cat_cols)} ({cat_set_name})")
    print(f"- Temporal10 사용    : {use_temporal}")
    print(f"- best_iteration     : {artifact.get('best_iteration')}")

    if type(model).__name__ != "CatBoostClassifier":
        raise TypeError(
            "cb_v004 Artifact의 model은 CatBoostClassifier여야 합니다. "
            f"actual={type(model).__name__}"
        )

    artifact_experiment = artifact.get("experiment_name")
    if artifact_experiment is not None and artifact_experiment != EXPERIMENT_NAME:
        raise ValueError(
            "Artifact experiment_name이 script 설정과 다릅니다. "
            f"artifact={artifact_experiment}, script={EXPERIMENT_NAME}"
        )

    artifact_objective = artifact.get("objective")
    if artifact_objective is not None and artifact_objective != "Logloss":
        raise ValueError(
            "cb_v004 Artifact objective는 Logloss여야 합니다. "
            f"actual={artifact_objective}"
        )

    if use_temporal:
        expected_features = set(EXPECTED_TEMPORAL_FEATURES)
        missing_expected = expected_features - set(final_features)
        if missing_expected:
            raise ValueError(
                "Artifact가 use_temporal_features=True이지만 Temporal10 "
                f"Feature가 features 목록에 없습니다: {sorted(missing_expected)}"
            )

    # -----------------------------------------------------
    # 2. 데이터 로드
    # -----------------------------------------------------
    print("\nTest 데이터 로드 중...")
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")

    test_ids = test_df[ID].values
    print(f"- Test 샘플 수: {len(test_df)}")

    # -----------------------------------------------------
    # 3. Temporal10 Feature Engineering (Artifact가 채택한 경우에만)
    # -----------------------------------------------------
    if use_temporal:
        print("\nTemporal10 Feature 생성 중 (해당 행의 asof_* 컬럼만 사용)...")
        missing_source = [
            c for c in TEMPORAL_SOURCE_COLUMNS if c not in test_df.columns
        ]
        if missing_source:
            raise ValueError(
                "Temporal10 계산에 필요한 원본 asof_* 컬럼이 test.csv에 "
                f"없습니다: {missing_source}"
            )
        test_df = add_temporal_features(test_df)

    # -----------------------------------------------------
    # 4. Feature 준비 및 정렬
    # -----------------------------------------------------
    print("\n최종 Feature 정렬 중...")

    missing_final_features = [
        c for c in final_features if c not in test_df.columns
    ]
    if missing_final_features:
        raise ValueError(
            "추론에 필요한 Feature가 test_df에 없습니다: "
            f"{missing_final_features}"
        )

    X_test = prepare_matrix(test_df, final_features, cat_cols)

    # 모델 학습 시의 Feature 순서와 완벽히 동일한지 검증.
    if list(model.feature_names_) != list(X_test.columns):
        raise ValueError("추론 Feature 순서가 모델 학습 시와 다릅니다.")

    # -----------------------------------------------------
    # 5. 모델 추론
    # -----------------------------------------------------
    print("\n추론(Inference) 시작...")
    pred_raw = model.predict_proba(X_test)[:, 1]
    pred_raw = np.asarray(pred_raw, dtype=np.float64)

    if not np.isfinite(pred_raw).all():
        raise ValueError("Raw prediction에 NaN 또는 Inf가 있습니다.")

    # 각 행의 prediction에 고정된 deterministic clip만 적용하므로
    # 다른 test 행의 정보를 사용하지 않는다.
    pred_prob = np.clip(pred_raw, 0.0, 1.0)

    below_zero = pred_raw < 0.0
    above_one = pred_raw > 1.0
    outside_range = below_zero | above_one

    print("\n[Raw prediction diagnostics]")
    print(f"- Mean       : {pred_raw.mean():.6f}")
    print(f"- Std        : {pred_raw.std():.6f}")
    print(f"- Min        : {pred_raw.min():.6f}")
    print(f"- Max        : {pred_raw.max():.6f}")
    print(
        f"- Outside    : {outside_range.sum()} "
        f"({outside_range.mean():.6%})"
    )

    # -----------------------------------------------------
    # 6. Submission 생성 및 저장
    # -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    submission = build_submission(
        sample_submission=sample_sub,
        test_ids=test_ids,
        predictions=pred_prob,
    )

    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("제출 파일 저장 완료")
    print(f"- 저장 경로       : {OUTPUT_PATH}")
    print(f"- Prediction Mean : {submission[TARGET].mean():.6f}")
    print(f"- Prediction Min  : {submission[TARGET].min():.6f}")
    print(f"- Prediction Max  : {submission[TARGET].max():.6f}")
    print(f"- 총 실행 시간     : {elapsed_time:.2f} 초")
    print("=" * 60)


if __name__ == "__main__":
    main()
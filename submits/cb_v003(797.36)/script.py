# script.py: CatBoost Classifier cb_v003 추론 및 제출
# cb_v003 train.py에서 생성된 단일 Artifact(.pkl)를 로드하여 
# HB-EB Rate 파생 변수를 적용하고 예측을 수행한다.
# 모든 경로 상수명은 프로젝트 표준 규칙을 엄격히 준수한다.

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# =========================================================
# 경로 (표준 규칙 준수)
# =========================================================
EXPERIMENT_NAME = "cb_v003"

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================
ID = "row_id"
TARGET = "control_success"


# =========================================================
# HB-EB Rate 적용 함수 (Train과 동일 로직)
# =========================================================
def apply_hbeb(df, group_cols, n_col, rate_col, prior_df, global_prior):
    """
    Train 단계에서 추정된 사전 분포(prior_df)를 단순 병합(Lookup)하여
    Test 데이터의 각 행을 철저히 독립적으로 변환합니다.
    """
    # 평가 데이터의 독립 예측 원칙 준수: 
    # prior_df는 오직 Train에서만 만들어진 정적 데이터임
    df_merged = df.merge(prior_df, on=group_cols, how='left')
    
    alpha = df_merged['alpha'].fillna(global_prior[0]).values
    beta = df_merged['beta'].fillna(global_prior[1]).values
    
    S = df[n_col].values * df[rate_col].fillna(0).values
    N = df[n_col].values
    
    eb_rate = (S + alpha) / (N + alpha + beta)
    return eb_rate


# =========================================================
# Feature 전처리
# =========================================================
def prepare_features(df, cat_cols):
    """
    범주형 결측값을 치환하고 문자열로 고정합니다.
    """
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna("__MISSING__").astype(str)
    return df


# =========================================================
# 제출 파일 생성
# =========================================================
def build_submission(sample_submission, test_ids, predictions):
    """
    row_id 기준으로 예측값을 매핑하고 제출 형식을 철저히 검증합니다.
    """
    pred_df = pd.DataFrame({
        ID: test_ids,
        TARGET: predictions,
    })

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

    if not submission[TARGET].between(0.0, 1.0).all():
        raise ValueError("제출 prediction에 [0, 1] 범위를 벗어난 값이 있습니다.")

    return submission


# =========================================================
# Main
# =========================================================
def main():
    start_time = time.time()
    
    # -----------------------------------------------------
    # 1. 단일 Artifact 로드
    # -----------------------------------------------------
    print("=" * 60)
    print(f"CatBoost Classifier Inference : {EXPERIMENT_NAME}")
    print("=" * 60)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 Artifact 파일을 찾을 수 없습니다: {MODEL_PATH}")

    print("모델 Artifact 로드 중...")
    artifact = joblib.load(MODEL_PATH)
    
    model = artifact["model"]
    hbeb_priors = artifact["hbeb_priors"]
    final_features = artifact["features"]
    cat_cols = artifact["cat_cols"]
    
    print(f"- 모델 타입: {type(model).__name__}")
    print(f"- 사용 Feature 수: {len(final_features)}")

    # -----------------------------------------------------
    # 2. 데이터 로드
    # -----------------------------------------------------
    print("\nTest 데이터 로드 중...")
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    
    test_ids = test_df[ID].values
    print(f"- Test 샘플 수: {len(test_df)}")

    # -----------------------------------------------------
    # 3. HB-EB Feature Engineering
    # -----------------------------------------------------
    print("\nHB-EB Rate 파생 변수 생성 중...")
    for new_col, config in hbeb_priors.items():
        test_df[new_col] = apply_hbeb(
            df=test_df,
            group_cols=config["group_cols"],
            n_col=config["n_col"],
            rate_col=config["rate_col"],
            prior_df=config["prior_df"],
            global_prior=config["global_prior"]
        )
    
    # -----------------------------------------------------
    # 4. Feature 준비 및 정렬
    # -----------------------------------------------------
    print("\n최종 Feature 정렬 중...")
    X_test = prepare_features(test_df[final_features], cat_cols)
    
    # 모델 학습 시의 Feature 순서와 완벽히 동일한지 검증
    if list(model.feature_names_) != list(X_test.columns):
        raise ValueError("추론 Feature 순서가 모델 학습 시와 다릅니다.")

    # -----------------------------------------------------
    # 5. 모델 추론
    # -----------------------------------------------------
    print("\n추론(Inference) 시작...")
    # CatBoostClassifier는 확률값을 예측하므로 predict_proba 사용
    pred_prob = model.predict_proba(X_test)[:, 1]
    
    # -----------------------------------------------------
    # 6. Submission 생성 및 저장
    # -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    submission = build_submission(
        sample_submission=sample_sub,
        test_ids=test_ids,
        predictions=pred_prob
    )
    
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("제출 파일 저장 완료")
    print(f"- 저장 경로: {OUTPUT_PATH}")
    print(f"- Prediction Mean: {submission[TARGET].mean():.6f}")
    print(f"- Prediction Min : {submission[TARGET].min():.6f}")
    print(f"- Prediction Max : {submission[TARGET].max():.6f}")
    print(f"- 총 실행 시간: {elapsed_time:.2f} 초")
    print("=" * 60)

if __name__ == "__main__":
    main()
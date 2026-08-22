# script.py: CatBoost Regressor cb_reg_v005 추론 및 제출
# cb_reg_v005 train.py에서 생성된 단일 Artifact(.pkl)를 로드하여
# HB-EB Rate + Tier A 2025 Trackman frozen profile을 적용하고 예측한다.
# 모든 경로 상수명은 프로젝트 표준 규칙을 엄격히 준수한다.

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# =========================================================
# 경로 (표준 규칙 준수)
# =========================================================
EXPERIMENT_NAME = "cb_reg_v005"

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
# Trackman Tier A 2025 Frozen Profile
# =========================================================
def apply_trackman_profile_2025(
    df,
    profile_2025,
    trackman_features,
    allowed_tiers,
):
    """
    train.py artifact에 저장된 2025 frozen Trackman lookup을
    pitcher_id 기준으로 각 Test 행에 독립적으로 적용한다.

    규칙:
    - test 전체 통계/groupby/rolling을 사용하지 않는다.
    - profile_2025는 train + 공식 Trackman history만으로 사전 생성된 정적 lookup이다.
    - 허용 Tier 외 선수 또는 history가 없는 선수는 unavailable 처리한다.
    """
    df = df.copy()
    profile = profile_2025.copy()

    required_cols = [
        "pitcher_id",
        "confidence_tier",
        "target_season",
        *trackman_features,
        "tm_history_max_season",
    ]

    missing = [
        c for c in required_cols
        if c not in profile.columns
    ]

    if missing:
        raise ValueError(
            "Artifact의 Trackman 2025 profile에 필수 컬럼이 없습니다: "
            f"{missing}"
        )

    if profile["pitcher_id"].duplicated().any():
        raise ValueError(
            "Artifact의 Trackman 2025 profile에서 pitcher_id가 중복됩니다."
        )

    # 2025 frozen lookup인지 확인
    if not (profile["target_season"] == 2025).all():
        raise ValueError(
            "Artifact Trackman profile에 target_season != 2025 행이 있습니다."
        )

    active_profile = profile.loc[
        profile["confidence_tier"].isin(
            allowed_tiers
        )
    ].copy()

    if active_profile["pitcher_id"].duplicated().any():
        raise ValueError(
            "Tier filtering 후 Trackman 2025 lookup pitcher_id가 중복됩니다."
        )

    merged = df.merge(
        active_profile,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # test 자체의 season이 존재한다면 2025인지 확인한다.
    if "season" in merged.columns:
        non_2025 = (
            merged["season"].notna()
            & (merged["season"] != 2025)
        )
        if non_2025.any():
            raise ValueError(
                "cb_reg_v004의 2025 frozen Trackman lookup과 "
                "test season이 일치하지 않습니다."
            )

    merged["tm_history_available"] = (
        merged["tm_history_available"]
        .fillna(0)
        .astype(np.int8)
    )

    merged["tm_n"] = (
        merged["tm_n"]
        .fillna(0)
        .astype(np.float64)
    )

    physical_cols = [
        c
        for c in trackman_features
        if c not in {
            "tm_history_available",
            "tm_n",
        }
    ]

    unavailable = (
        merged["tm_history_available"] == 0
    )

    for col in physical_cols:
        merged.loc[
            unavailable,
            col,
        ] = np.nan

    # 활성 profile은 Tier A only여야 한다.
    available = (
        merged["tm_history_available"] == 1
    )

    active_tiers = set(
        merged.loc[
            available,
            "confidence_tier",
        ]
        .dropna()
        .unique()
    )

    if not active_tiers.issubset(
        set(allowed_tiers)
    ):
        raise ValueError(
            "허용되지 않은 Trackman confidence tier가 활성화되었습니다: "
            f"{active_tiers}"
        )

    # Future leakage audit:
    # 2025 prediction에는 Trackman season <= 2024만 허용.
    violation = (
        available
        & (
            merged["tm_history_max_season"]
            >= 2025
        )
    )

    if violation.any():
        raise ValueError(
            "Trackman future leakage 발견: "
            "tm_history_max_season >= 2025"
        )

    # 모델 Feature 값의 Inf 방지.
    numeric = merged[
        [
            c for c in trackman_features
            if c in merged.columns
        ]
    ].select_dtypes(
        include=[np.number]
    )

    if np.isinf(
        numeric.to_numpy(
            dtype=np.float64
        )
    ).any():
        raise ValueError(
            "Trackman Feature에 Inf가 있습니다."
        )

    return merged


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
    
    # -----------------------------------------------------
    # 1. 단일 Artifact 로드
    # -----------------------------------------------------
    print("=" * 60)
    print(f"CatBoost Regressor Inference : {EXPERIMENT_NAME}")
    print("=" * 60)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 Artifact 파일을 찾을 수 없습니다: {MODEL_PATH}")

    print("모델 Artifact 로드 중...")
    artifact = joblib.load(MODEL_PATH)
    
    model = artifact["model"]
    hbeb_priors = artifact["hbeb_priors"]
    final_features = artifact["features"]
    cat_cols = artifact["cat_cols"]

    trackman_profile_2025 = artifact[
        "trackman_profile_2025"
    ]

    trackman_features = artifact[
        "trackman_features"
    ]

    trackman_allowed_tiers = artifact[
        "trackman_allowed_tiers"
    ]
    
    print(f"- 모델 타입: {type(model).__name__}")
    print(f"- 사용 Feature 수: {len(final_features)}")
    print(
        f"- Weight policy: "
        f"{artifact.get('weight_policy', 'N/A')}"
    )
    print(
        f"- Trackman tiers: "
        f"{trackman_allowed_tiers}"
    )

    if type(model).__name__ != "CatBoostRegressor":
        raise TypeError(
            "cb_reg_v004 Artifact의 model은 CatBoostRegressor여야 합니다. "
            f"actual={type(model).__name__}"
        )

    artifact_experiment = artifact.get("experiment_name")
    if artifact_experiment is not None and artifact_experiment != EXPERIMENT_NAME:
        raise ValueError(
            "Artifact experiment_name이 script 설정과 다릅니다. "
            f"artifact={artifact_experiment}, script={EXPERIMENT_NAME}"
        )

    artifact_objective = artifact.get("objective")
    if artifact_objective is not None and artifact_objective != "RMSE":
        raise ValueError(
            "cb_reg_v005 Artifact objective는 RMSE여야 합니다. "
            f"actual={artifact_objective}"
        )


    # cb_reg_v005는 old_050 extended 정책으로 최종 학습된 Treatment artifact다.
    # Weight는 학습 시에만 사용하며 추론 Test 행에는 적용하지 않는다.
    artifact_weight_policy = artifact.get(
        "weight_policy"
    )

    if (
        artifact_weight_policy is not None
        and artifact_weight_policy != "old_050_extended"
    ):
        raise ValueError(
            "cb_reg_v005 Artifact weight_policy가 예상과 다릅니다. "
            f"actual={artifact_weight_policy}"
        )

    expected_final_weight_map = {
        2019: 0.5,
        2020: 0.5,
        2021: 0.5,
        2022: 0.5,
        2023: 1.0,
        2024: 1.0,
    }

    artifact_final_weight_map = artifact.get(
        "final_weight_map"
    )

    if artifact_final_weight_map is not None:
        normalized_artifact_weight_map = {
            int(k): float(v)
            for k, v in artifact_final_weight_map.items()
        }

        if (
            normalized_artifact_weight_map
            != expected_final_weight_map
        ):
            raise ValueError(
                "cb_reg_v005 Artifact final_weight_map이 예상과 다릅니다. "
                f"actual={normalized_artifact_weight_map}"
            )


    if trackman_allowed_tiers != ["A"]:
        raise ValueError(
            "cb_reg_v005는 Tier A only 실험이어야 합니다. "
            f"artifact tiers={trackman_allowed_tiers}"
        )

    expected_trackman_features = [
        "tm_history_available",
        "tm_n",
        "tm_rel_speed_mean",
        "tm_rel_speed_std",
        "tm_spin_rate_mean",
        "tm_spin_rate_std",
        "tm_induced_vert_break_mean",
        "tm_induced_vert_break_std",
        "tm_rel_height_mean",
        "tm_rel_height_std",
        "tm_rel_side_mean",
        "tm_rel_side_std",
    ]

    if list(trackman_features) != expected_trackman_features:
        raise ValueError(
            "Artifact Trackman Feature 구성이 cb_reg_v005 정의와 다릅니다.\n"
            f"artifact={trackman_features}"
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
    # 3. Tier A Trackman 2025 Frozen Profile
    #    (old_050는 학습 weight 정책이므로 Test에서는 별도 적용하지 않음)
    # -----------------------------------------------------
    print("\nTier A Trackman 2025 frozen profile 적용 중...")

    test_df = apply_trackman_profile_2025(
        df=test_df,
        profile_2025=trackman_profile_2025,
        trackman_features=trackman_features,
        allowed_tiers=trackman_allowed_tiers,
    )

    tm_available = (
        test_df["tm_history_available"] == 1
    )

    print(
        f"- Trackman available: "
        f"{int(tm_available.sum()):,} / {len(test_df):,} "
        f"({tm_available.mean():.2%})"
    )

    # -----------------------------------------------------
    # 4. HB-EB Feature Engineering
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
    # 5. Feature 준비 및 정렬
    # -----------------------------------------------------
    print("\n최종 Feature 정렬 중...")

    missing_final_features = [
        c
        for c in final_features
        if c not in test_df.columns
    ]

    if missing_final_features:
        raise ValueError(
            "추론에 필요한 Feature가 test_df에 없습니다: "
            f"{missing_final_features}"
        )

    missing_trackman_model_features = [
        c
        for c in trackman_features
        if c not in final_features
    ]

    if missing_trackman_model_features:
        raise ValueError(
            "Model final_features에 Trackman Feature가 누락되었습니다: "
            f"{missing_trackman_model_features}"
        )

    X_test = prepare_features(
        test_df[final_features],
        cat_cols,
    )
    
    # 모델 학습 시의 Feature 순서와 완벽히 동일한지 검증
    if list(model.feature_names_) != list(X_test.columns):
        raise ValueError("추론 Feature 순서가 모델 학습 시와 다릅니다.")

    # -----------------------------------------------------
    # 6. 모델 추론
    # -----------------------------------------------------
    print("\n추론(Inference) 시작...")
    # CatBoostRegressor는 predict()로 raw prediction을 생성한다.
    pred_raw = model.predict(X_test)
    pred_raw = np.asarray(pred_raw, dtype=np.float64)

    if not np.isfinite(pred_raw).all():
        raise ValueError("Raw prediction에 NaN 또는 Inf가 있습니다.")

    # train.py의 Validation 정책과 동일하게 [0, 1] 범위로 clipping한다.
    # 각 행의 prediction에 고정된 deterministic transform만 적용하므로
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
    # 7. Submission 생성 및 저장
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
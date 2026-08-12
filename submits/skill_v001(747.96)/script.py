# script.py: skill_v001
#
# 역할:
# - ./model/skill_v001.pkl bundle 로드
# - test.csv 로드
# - train.py와 동일한 deterministic skill feature 생성
# - 저장된 feature 순서 / categorical schema 적용
# - CatBoost predict_proba
# - submission.csv 생성
#
# 핵심 원칙:
# - 공식 test row의 asof_* feature만 사용
# - target 사용 없음
# - test row 간 rolling / expanding / state update 없음
# - test 전체 distribution을 이용한 보정 없음
# - train.py에서 선택된 ablation의 engineered feature만 사용
# - submission.csv 파일명은 변경하지 않음
#
# package.py 호환:
# package.py가 MODEL_PATH를 submit/model/skill_v001.pkl로 복사하므로
# 제출 환경에서는 ./model/skill_v001.pkl을 읽는다.


from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 경로
# =========================================================

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")

MODEL_PATH = MODEL_DIR / "skill_v001.pkl"

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = (
    DATA_DIR / "sample_submission.csv"
)

# 대회 제출 파일명은 고정
SUBMISSION_PATH = Path("./output/submission.csv")


# =========================================================
# 기본 컬럼
# =========================================================

ID = "row_id"
TARGET = "control_success"

PITCHER_N = "asof_pitcher_n"
PITCHER_RATE = "asof_pitcher_success_rate"

PREV1 = "asof_pitcher_prev1_game_success_rate"
PREV3 = "asof_pitcher_prev3_game_success_rate"
PREV5 = "asof_pitcher_prev5_game_success_rate"

EXPECTED_EXPERIMENT = "skill_v001"


# =========================================================
# Engineered feature names
# train.py와 동일
# =========================================================

F_POST_MEAN = "pitcher_skill_post_mean"

F_POST_SD = "pitcher_skill_post_sd"
F_SHRINK_GAP = "pitcher_skill_shrink_gap"

F_PREV1_LONG = "pitcher_form_prev1_minus_long"
F_PREV3_LONG = "pitcher_form_prev3_minus_long"
F_PREV5_LONG = "pitcher_form_prev5_minus_long"

F_PREV1_PREV3 = "pitcher_form_prev1_minus_prev3"
F_PREV3_PREV5 = "pitcher_form_prev3_minus_prev5"
F_RECENT_RANGE = "pitcher_recent_window_range"


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    """
    train.py와 동일한 CatBoost 전처리.

    - 수치형 결측값은 그대로 유지
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 문자열 dtype으로 통일
    """

    out = df.copy()

    for col in cat_cols:

        if col not in out.columns:
            raise KeyError(
                f"범주형 feature가 없습니다: {col}"
            )

        out[col] = (
            out[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return out


# =========================================================
# Skill feature 생성
# train.py와 반드시 동일해야 함
# =========================================================

def add_pitcher_skill_features(
    df,
    prior_mean,
    prior_strength,
):
    """
    공식 row-local asof feature만 이용하여
    deterministic skill representation을 생성한다.

    TARGET은 사용하지 않는다.

    A. long-term
       count-aware shrinkage mean

    B. uncertainty
       posterior-style sd
       shrink gap

    C. recent form
       prev1/3/5 - long skill

    D. multi-scale
       prev1-prev3
       prev3-prev5
       recent window range

    train.py와 구현을 동일하게 유지해야 한다.
    """

    out = df.copy()

    required = [
        PITCHER_N,
        PITCHER_RATE,
        PREV1,
        PREV3,
        PREV5,
    ]

    missing = [
        col
        for col in required
        if col not in out.columns
    ]

    if missing:
        raise KeyError(
            "Skill feature 생성에 필요한 "
            f"공식 컬럼이 없습니다: {missing}"
        )

    n_raw = pd.to_numeric(
        out[PITCHER_N],
        errors="coerce",
    )

    p_raw = pd.to_numeric(
        out[PITCHER_RATE],
        errors="coerce",
    )

    n = (
        n_raw
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(np.float64)
    )

    p = (
        p_raw
        .fillna(prior_mean)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .astype(np.float64)
    )

    m = float(
        prior_strength
    )

    mu = float(
        prior_mean
    )

    # -----------------------------------------------------
    # A. Long-term skill
    # -----------------------------------------------------

    k = (
        n * p
    )

    post_mean = (
        k + m * mu
    ) / (
        n + m
    )

    out[
        F_POST_MEAN
    ] = (
        post_mean
        .astype(np.float32)
    )

    # -----------------------------------------------------
    # B. Uncertainty
    # -----------------------------------------------------

    post_var = (
        post_mean
        * (
            1.0 - post_mean
        )
        / (
            n + m + 1.0
        )
    )

    post_sd = np.sqrt(
        np.maximum(
            post_var,
            0.0,
        )
    )

    shrink_gap = (
        post_mean
        - p
    )

    out[
        F_POST_SD
    ] = (
        post_sd
        .astype(np.float32)
    )

    out[
        F_SHRINK_GAP
    ] = (
        shrink_gap
        .astype(np.float32)
    )

    # -----------------------------------------------------
    # C / D. Recent form
    # -----------------------------------------------------

    prev1 = pd.to_numeric(
        out[PREV1],
        errors="coerce",
    ).astype(np.float64)

    prev3 = pd.to_numeric(
        out[PREV3],
        errors="coerce",
    ).astype(np.float64)

    prev5 = pd.to_numeric(
        out[PREV5],
        errors="coerce",
    ).astype(np.float64)

    out[
        F_PREV1_LONG
    ] = (
        prev1 - post_mean
    ).astype(np.float32)

    out[
        F_PREV3_LONG
    ] = (
        prev3 - post_mean
    ).astype(np.float32)

    out[
        F_PREV5_LONG
    ] = (
        prev5 - post_mean
    ).astype(np.float32)

    out[
        F_PREV1_PREV3
    ] = (
        prev1 - prev3
    ).astype(np.float32)

    out[
        F_PREV3_PREV5
    ] = (
        prev3 - prev5
    ).astype(np.float32)

    recent = pd.concat(
        [
            prev1.rename("prev1"),
            prev3.rename("prev3"),
            prev5.rename("prev5"),
        ],
        axis=1,
    )

    out[
        F_RECENT_RANGE
    ] = (
        recent.max(
            axis=1,
            skipna=True,
        )
        - recent.min(
            axis=1,
            skipna=True,
        )
    ).astype(np.float32)

    return out


# =========================================================
# Bundle 검증
# =========================================================

def validate_bundle(
    bundle,
):
    """
    skill_v001.pkl의 구조를 검증한다.
    """

    required_keys = {
        "model",
        "features",
        "base_features",
        "cat_cols",
        "feature_config",
        "metadata",
    }

    missing = (
        required_keys
        - set(bundle.keys())
    )

    if missing:
        raise KeyError(
            "Model bundle에 필요한 key가 없습니다: "
            f"{sorted(missing)}"
        )

    metadata = (
        bundle[
            "metadata"
        ]
    )

    experiment_name = (
        metadata.get(
            "experiment_name"
        )
    )

    if (
        experiment_name
        != EXPECTED_EXPERIMENT
    ):
        raise ValueError(
            "잘못된 experiment bundle입니다.\n"
            f"expected: {EXPECTED_EXPERIMENT}\n"
            f"actual  : {experiment_name}"
        )

    feature_config = (
        bundle[
            "feature_config"
        ]
    )

    for key in [
        "prior_mean",
        "prior_strength",
        "selected_ablation",
        "selected_engineered_features",
    ]:

        if key not in feature_config:
            raise KeyError(
                "feature_config에 필요한 key가 없습니다: "
                f"{key}"
            )


# =========================================================
# Test schema 검증
# =========================================================

def validate_test_schema(
    test,
    base_features,
):
    """
    train.py에서 사용한 모든 raw feature가
    test.csv에 존재하는지 확인한다.
    """

    missing = [
        col
        for col in base_features
        if col not in test.columns
    ]

    if missing:
        raise KeyError(
            "test.csv에 학습 raw feature가 없습니다: "
            f"{missing}"
        )

    if ID not in test.columns:
        raise KeyError(
            f"test.csv에 {ID}가 없습니다."
        )


# =========================================================
# Engineered feature 검증
# =========================================================

def validate_engineered_features(
    test_fe,
    selected_engineered_features,
):
    """
    저장된 ablation이 요구하는 engineered feature가
    모두 생성되었는지 확인한다.
    """

    missing = [
        col
        for col in selected_engineered_features
        if col not in test_fe.columns
    ]

    if missing:
        raise KeyError(
            "Engineered feature 생성 실패: "
            f"{missing}"
        )


# =========================================================
# Diagnostic
# =========================================================

def print_feature_diagnostics(
    test_fe,
    selected_engineered_features,
):
    """
    inference 전에 engineered feature의
    결측률과 간단한 분포를 출력한다.
    """

    print()
    print("=" * 60)
    print("Engineered Feature Diagnostics")
    print("=" * 60)

    for col in selected_engineered_features:

        s = pd.to_numeric(
            test_fe[col],
            errors="coerce",
        )

        missing_rate = float(
            s.isna().mean()
        )

        valid = s.dropna()

        print()
        print(
            f"[{col}]"
        )

        print(
            f"missing : "
            f"{missing_rate:.4%}"
        )

        if len(valid):

            print(
                f"mean    : "
                f"{valid.mean():.6f}"
            )

            print(
                f"std     : "
                f"{valid.std():.6f}"
            )

            print(
                f"min     : "
                f"{valid.min():.6f}"
            )

            print(
                f"max     : "
                f"{valid.max():.6f}"
            )

    print()
    print("=" * 60)


# =========================================================
# Submission 생성
# =========================================================

def build_submission(
    test,
    pred,
):
    """
    sample_submission.csv 구조를 보존하여
    submission.csv를 생성한다.
    """

    submission = pd.read_csv(
        SAMPLE_SUBMISSION_PATH,
        encoding="utf-8-sig",
    )

    if len(submission) != len(test):
        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "행 수가 다릅니다.\n"
            f"test rows       : {len(test)}\n"
            f"submission rows : {len(submission)}"
        )

    if TARGET not in submission.columns:
        raise KeyError(
            "sample_submission.csv에 "
            f"{TARGET} 컬럼이 없습니다."
        )

    # row_id가 submission에도 존재할 경우 순서까지 검증
    if (
        ID in submission.columns
        and ID in test.columns
    ):

        same_order = (
            submission[ID]
            .astype(str)
            .to_numpy()
            ==
            test[ID]
            .astype(str)
            .to_numpy()
        )

        if not same_order.all():
            raise ValueError(
                "test.csv와 sample_submission.csv의 "
                "row_id 순서가 일치하지 않습니다."
            )

    submission[
        TARGET
    ] = np.asarray(
        pred,
        dtype=np.float64,
    )

    return submission


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 파일 확인
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다: {MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv를 찾을 수 없습니다: {TEST_PATH}"
        )

    if not SAMPLE_SUBMISSION_PATH.exists():
        raise FileNotFoundError(
            "sample_submission.csv를 찾을 수 없습니다: "
            f"{SAMPLE_SUBMISSION_PATH}"
        )

    # -----------------------------------------------------
    # Bundle load
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("skill_v001 Inference")
    print("=" * 60)

    bundle = joblib.load(
        MODEL_PATH
    )

    validate_bundle(
        bundle
    )

    model = (
        bundle[
            "model"
        ]
    )

    features = list(
        bundle[
            "features"
        ]
    )

    base_features = list(
        bundle[
            "base_features"
        ]
    )

    cat_cols = list(
        bundle[
            "cat_cols"
        ]
    )

    feature_config = (
        bundle[
            "feature_config"
        ]
    )

    metadata = (
        bundle[
            "metadata"
        ]
    )

    prior_mean = float(
        feature_config[
            "prior_mean"
        ]
    )

    prior_strength = float(
        feature_config[
            "prior_strength"
        ]
    )

    selected_ablation = (
        feature_config[
            "selected_ablation"
        ]
    )

    selected_engineered_features = list(
        feature_config[
            "selected_engineered_features"
        ]
    )

    print(
        f"Model              : {MODEL_PATH}"
    )

    print(
        "Experiment         :",
        metadata.get(
            "experiment_name"
        ),
    )

    print(
        "Selected ablation  :",
        selected_ablation,
    )

    print(
        "Validation decision:",
        metadata.get(
            "decision"
        ),
    )

    print(
        "Prior mean         :",
        prior_mean,
    )

    print(
        "Prior strength     :",
        prior_strength,
    )

    print(
        "Base features      :",
        len(base_features),
    )

    print(
        "Engineered features:",
        len(
            selected_engineered_features
        ),
    )

    print(
        "Final features     :",
        len(features),
    )

    print(
        "Categorical cols   :",
        len(cat_cols),
    )

    print("=" * 60)

    print()
    print("Selected engineered features:")

    for feature in selected_engineered_features:
        print(
            f"- {feature}"
        )

    # -----------------------------------------------------
    # Test load
    # -----------------------------------------------------

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 60)
    print("Test Data")
    print("=" * 60)

    print(
        "test shape:",
        test.shape,
    )

    validate_test_schema(
        test,
        base_features,
    )

    # -----------------------------------------------------
    # Skill feature 생성
    # -----------------------------------------------------

    test_fe = (
        add_pitcher_skill_features(
            test,
            prior_mean=(
                prior_mean
            ),
            prior_strength=(
                prior_strength
            ),
        )
    )

    validate_engineered_features(
        test_fe,
        selected_engineered_features,
    )

    print_feature_diagnostics(
        test_fe,
        selected_engineered_features,
    )

    # -----------------------------------------------------
    # 최종 feature schema
    # -----------------------------------------------------

    missing_final = [
        col
        for col in features
        if col not in test_fe.columns
    ]

    if missing_final:
        raise KeyError(
            "최종 model feature를 구성할 수 없습니다: "
            f"{missing_final}"
        )

    X_test = prepare_features(
        test_fe[
            features
        ],
        cat_cols,
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("Prediction 시작")
    print("=" * 60)

    pred = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "Prediction에 NaN 또는 inf가 존재합니다."
        )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    print(
        f"Prediction mean : {pred.mean():.6f}"
    )

    print(
        f"Prediction std  : {pred.std():.6f}"
    )

    print(
        f"Prediction min  : {pred.min():.6f}"
    )

    print(
        f"Prediction max  : {pred.max():.6f}"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Submission
    # -----------------------------------------------------

    submission = (
        build_submission(
            test,
            pred,
        )
    )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 60)
    print("Submission 생성 완료")
    print("=" * 60)

    print(
        f"Output : {SUBMISSION_PATH}"
    )

    print(
        f"Rows   : {len(submission):,}"
    )

    print(
        f"Target : {TARGET}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()

# script.py: CatBoost Regressor cb_reg_v009 추론 및 제출
#
# train.py(cb_reg_v009)에서 생성된 단일 Artifact
#   model/cb_reg_v009/cb_reg_v009.pkl
# 를 로드하여 test.csv 각 행을 독립적으로 추론하고
# output/submission.csv를 생성한다.
#
# 추론 파이프라인
# ----------------
# 1. Artifact에 저장된 HB-EB prior를 각 행에 lookup하여 Feature 변환
# 2. CatBoostRegressor.predict()로 raw prediction 생성
# 3. game_type == "F"인 행에만 Artifact의 frozen logit delta 적용
# 4. np.clip(prediction, 0.0, 1.0)
#
# 평가 데이터 독립 추론 규칙
# -------------------------
# - test.csv의 다른 행이나 전체 분포를 Feature/Prediction 생성에 사용하지 않는다.
# - test 기반 groupby / rolling / lag / rank / 평균 / 빈도 / 통계를 사용하지 않는다.
# - HB-EB prior는 train.py에서 공식 train.csv만으로 계산되어 Artifact에 저장된 값만 사용한다.
# - F correction anchor와 delta도 train.py에서 공식 train.csv만으로 확정된 값을 사용한다.
# - 추론 시 test prediction 평균으로 delta를 다시 계산하지 않는다.
# - 한 행만 입력했을 때와 전체 test.csv에 포함했을 때 해당 행의 예측값이 동일하다.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v009"

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

CORRECTION_EPS = 1e-6


# =========================================================
# 유틸
# =========================================================

def validate_paths():
    required = [
        MODEL_PATH,
        TEST_PATH,
        SAMPLE_SUB_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "필수 파일이 없습니다:\n- "
            + "\n- ".join(missing)
        )


def load_artifact():
    artifact = joblib.load(
        MODEL_PATH
    )

    required_keys = [
        "model",
        "hbeb_priors",
        "features",
        "cat_cols",
        "experiment_name",
        "model_type",
        "objective",
        "final_f_correction",
        "prediction_policy",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing:
        raise KeyError(
            "Artifact 필수 key 누락: "
            + ", ".join(missing)
        )

    if (
        artifact["experiment_name"]
        != EXPERIMENT_NAME
    ):
        raise ValueError(
            "Artifact experiment_name 불일치: "
            f"{artifact['experiment_name']} "
            f"!= {EXPERIMENT_NAME}"
        )

    if (
        artifact["model_type"]
        != "CatBoostRegressor"
    ):
        raise ValueError(
            "지원하지 않는 model_type: "
            f"{artifact['model_type']}"
        )

    if artifact["objective"] != "RMSE":
        raise ValueError(
            "지원하지 않는 objective: "
            f"{artifact['objective']}"
        )

    return artifact


def validate_input_schema(
    test_df,
    sample_sub,
    artifact,
):
    if ID not in test_df.columns:
        raise ValueError(
            f"test.csv에 {ID} 컬럼이 없습니다."
        )

    if list(sample_sub.columns) != [
        ID,
        TARGET,
    ]:
        raise ValueError(
            "sample_submission.csv 컬럼이 "
            f"[{ID}, {TARGET}]와 일치하지 않습니다: "
            f"{list(sample_sub.columns)}"
        )

    if test_df[ID].duplicated().any():
        raise ValueError(
            "test.csv의 row_id가 중복되어 있습니다."
        )

    if sample_sub[ID].duplicated().any():
        raise ValueError(
            "sample_submission.csv의 row_id가 중복되어 있습니다."
        )

    test_ids = set(
        test_df[ID].tolist()
    )
    sub_ids = set(
        sample_sub[ID].tolist()
    )

    if test_ids != sub_ids:
        only_test = len(
            test_ids - sub_ids
        )
        only_sub = len(
            sub_ids - test_ids
        )

        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "row_id 집합이 일치하지 않습니다. "
            f"only_test={only_test}, "
            f"only_submission={only_sub}"
        )

    hbeb_priors = artifact[
        "hbeb_priors"
    ]

    required_raw_cols = set()

    for prior_info in (
        hbeb_priors.values()
    ):
        required_raw_cols.update(
            prior_info["group_cols"]
        )
        required_raw_cols.add(
            prior_info["n_col"]
        )
        required_raw_cols.add(
            prior_info["rate_col"]
        )

    required_raw_cols.update(
        artifact["cat_cols"]
    )

    # final F correction에 game_type이 필요하다.
    required_raw_cols.add(
        "game_type"
    )

    missing_cols = sorted(
        col
        for col in required_raw_cols
        if col not in test_df.columns
    )

    if missing_cols:
        raise ValueError(
            "test.csv에 추론 필수 컬럼이 없습니다: "
            + ", ".join(missing_cols)
        )


# =========================================================
# HB-EB
# =========================================================

def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """
    train.py에서 저장한 train-only prior를 lookup하여
    각 행을 독립적으로 HB-EB 변환한다.

    다른 test 행의 값은 사용하지 않는다.
    """

    original_index = df.index

    lookup_df = (
        df[group_cols]
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup_df["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup_df["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n * rate
    )

    denominator = (
        n + alpha + beta
    )

    if np.any(
        denominator <= 0.0
    ):
        raise ValueError(
            f"HB-EB denominator가 0 이하입니다: "
            f"{rate_col}"
        )

    eb_rate = (
        success_count + alpha
    ) / denominator

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def apply_saved_hbeb_priors(
    df_apply,
    artifact,
):
    """
    Artifact에 저장된 HB-EB prior만 사용하여
    test Feature를 재현한다.

    prior 재추정은 하지 않는다.
    """

    transformed = (
        df_apply.copy()
    )

    final_features = list(
        artifact["features"]
    )

    hbeb_priors = artifact[
        "hbeb_priors"
    ]

    for (
        new_col,
        prior_info,
    ) in hbeb_priors.items():
        transformed[
            new_col
        ] = apply_hbeb(
            df=transformed,
            group_cols=(
                prior_info[
                    "group_cols"
                ]
            ),
            n_col=(
                prior_info[
                    "n_col"
                ]
            ),
            rate_col=(
                prior_info[
                    "rate_col"
                ]
            ),
            prior_df=(
                prior_info[
                    "prior_df"
                ]
            ),
            global_prior=(
                prior_info[
                    "global_prior"
                ]
            ),
        )

    missing_features = [
        col
        for col in final_features
        if col
        not in transformed.columns
    ]

    if missing_features:
        raise ValueError(
            "HB-EB 변환 후 모델 Feature가 누락되었습니다: "
            + ", ".join(
                missing_features
            )
        )

    return transformed


# =========================================================
# CatBoost Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    """
    train.py와 동일한 categorical 전처리.
    """

    df = df.copy()

    for col in cat_cols:
        if col not in df.columns:
            raise ValueError(
                f"범주형 Feature 누락: {col}"
            )

        df[col] = (
            df[col]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

    return df


# =========================================================
# Frozen F correction
# =========================================================

def _safe_logit(
    p,
    eps=CORRECTION_EPS,
):
    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        eps,
        1.0 - eps,
    )

    return np.log(
        p / (1.0 - p)
    )


def _sigmoid(z):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    out = np.empty_like(
        z,
        dtype=np.float64,
    )

    positive = z >= 0.0

    out[positive] = (
        1.0
        / (
            1.0
            + np.exp(
                -z[positive]
            )
        )
    )

    exp_z = np.exp(
        z[~positive]
    )

    out[~positive] = (
        exp_z
        / (
            1.0
            + exp_z
        )
    )

    return out


def apply_frozen_f_correction(
    pred_raw,
    game_type,
    correction,
):
    """
    train.py가 공식 train.csv만으로 확정한 frozen delta를
    F 행에만 row-wise 적용한다.

    금지:
    - pred_raw.mean()으로 delta 재계산
    - test F 평균/비율 계산
    - test 전체 통계를 이용한 calibration
    """

    pred_raw = np.asarray(
        pred_raw,
        dtype=np.float64,
    )

    game_type = (
        pd.Series(
            game_type
        )
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    if (
        len(pred_raw)
        != len(game_type)
    ):
        raise ValueError(
            "prediction과 game_type 길이가 "
            "일치하지 않습니다."
        )

    if not correction.get(
        "enabled",
        False,
    ):
        return pred_raw.copy()

    method = correction.get(
        "method"
    )

    if method != "logit_shift":
        raise ValueError(
            "지원하지 않는 F correction method: "
            f"{method}"
        )

    if correction.get(
        "uses_test_distribution",
        False,
    ):
        raise ValueError(
            "Artifact가 test distribution 사용으로 "
            "표시되어 있어 추론을 중단합니다."
        )

    if not correction.get(
        "row_independent_application",
        False,
    ):
        raise ValueError(
            "Artifact의 F correction이 row-independent로 "
            "표시되어 있지 않습니다."
        )

    domain = str(
        correction.get(
            "domain",
            "F",
        )
    )

    delta = float(
        correction["delta"]
    )

    corrected = (
        pred_raw.copy()
    )

    mask = (
        game_type
        == domain
    )

    if np.any(mask):
        corrected[mask] = (
            _sigmoid(
                _safe_logit(
                    pred_raw[mask]
                )
                + delta
            )
        )

    return corrected


# =========================================================
# 행 독립 추론 파이프라인
# =========================================================

def predict_rows(
    df,
    artifact,
):
    """
    입력된 각 행을 독립적으로 예측한다.

    동일한 한 행을 단독 입력하거나 더 큰 DataFrame에 포함해 입력해도
    해당 행의 Feature 및 prediction은 동일하다.
    """

    if len(df) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    model = artifact[
        "model"
    ]
    final_features = list(
        artifact[
            "features"
        ]
    )
    cat_cols = list(
        artifact[
            "cat_cols"
        ]
    )

    transformed = (
        apply_saved_hbeb_priors(
            df_apply=df,
            artifact=artifact,
        )
    )

    X = prepare_features(
        transformed[
            final_features
        ],
        cat_cols,
    )

    pred_model_raw = (
        model.predict(X)
    )

    pred_corrected = (
        apply_frozen_f_correction(
            pred_raw=pred_model_raw,
            game_type=df[
                "game_type"
            ],
            correction=artifact[
                "final_f_correction"
            ],
        )
    )

    pred_prob = np.clip(
        np.asarray(
            pred_corrected,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    if not np.all(
        np.isfinite(
            pred_prob
        )
    ):
        raise ValueError(
            "최종 prediction에 NaN/Inf가 존재합니다."
        )

    return pred_prob


# =========================================================
# Submission
# =========================================================

def build_submission(
    test_df,
    sample_sub,
    predictions,
):
    if (
        len(test_df)
        != len(predictions)
    ):
        raise ValueError(
            "test row 수와 prediction 수가 "
            "일치하지 않습니다."
        )

    pred_df = pd.DataFrame(
        {
            ID: (
                test_df[
                    ID
                ].to_numpy()
            ),
            TARGET: (
                np.asarray(
                    predictions,
                    dtype=np.float64,
                )
            ),
        }
    )

    submission = (
        sample_sub[
            [ID]
        ]
        .merge(
            pred_df,
            on=ID,
            how="left",
            sort=False,
            validate="one_to_one",
        )
    )

    if submission[
        TARGET
    ].isna().any():
        n_missing = int(
            submission[
                TARGET
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "submission prediction 누락: "
            f"{n_missing}건"
        )

    if not np.all(
        np.isfinite(
            submission[
                TARGET
            ].to_numpy(
                dtype=np.float64
            )
        )
    ):
        raise ValueError(
            "submission에 NaN/Inf prediction이 있습니다."
        )

    return submission


# =========================================================
# main
# =========================================================

def main():
    validate_paths()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )
    print(
        f"Model      : {MODEL_PATH}"
    )
    print(
        f"Test       : {TEST_PATH}"
    )
    print(
        f"Output     : {OUTPUT_PATH}"
    )
    print("=" * 72)

    print()
    print("[1/5] Artifact 로드")

    artifact = (
        load_artifact()
    )

    correction = artifact[
        "final_f_correction"
    ]

    print(
        "Model type        : "
        f"{artifact['model_type']}"
    )
    print(
        "Objective         : "
        f"{artifact['objective']}"
    )
    print(
        "Final iterations  : "
        f"{artifact.get('final_iterations', '?')}"
    )
    print(
        "Feature 수        : "
        f"{len(artifact['features'])}"
    )
    print(
        "HB-EB Feature 수  : "
        f"{len(artifact['hbeb_priors'])}"
    )
    print(
        "F correction      : "
        f"{correction.get('enabled', False)}"
    )

    if correction.get(
        "enabled",
        False,
    ):
        print(
            "F anchor          : "
            f"{correction['anchor']:.6f}"
        )
        print(
            "F anchor policy   : "
            f"{correction['anchor_policy']}"
        )
        print(
            "Anchor seasons    : "
            f"{correction['anchor_source_seasons']}"
        )
        print(
            "Reference season  : "
            f"{correction['reference_season']}"
        )
        print(
            "Frozen delta      : "
            f"{correction['delta']:+.8f}"
        )

    print()
    print("[2/5] Test / Sample Submission 로드")

    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    sample_sub = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    validate_input_schema(
        test_df=test_df,
        sample_sub=sample_sub,
        artifact=artifact,
    )

    print(
        f"Test rows          : {len(test_df):,}"
    )
    print(
        f"Submission rows    : {len(sample_sub):,}"
    )

    print()
    print("[3/5] Row-wise inference")

    predictions = predict_rows(
        df=test_df,
        artifact=artifact,
    )

    print(
        f"Prediction rows    : {len(predictions):,}"
    )

    print()
    print("[4/5] Submission 생성")

    submission = (
        build_submission(
            test_df=test_df,
            sample_sub=sample_sub,
            predictions=predictions,
        )
    )

    print()
    print("[5/5] 저장")

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 72)
    print("완료")
    print(
        f"Submission : {OUTPUT_PATH}"
    )
    print(
        f"Rows       : {len(submission):,}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

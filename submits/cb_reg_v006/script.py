# script.py: CatBoost Regressor cb_reg_v006 추론 및 제출
#
# cb_reg_v006 train.py에서 생성된 단일 Artifact(.pkl)를 로드하여
# HB-EB + Pitcher×Count-State Hierarchical Conditional EB(HC-EB)를 적용하고 예측한다.
#
# 평가 데이터 독립 추론 규칙:
# - test.csv의 각 행은 독립적으로 변환한다.
# - test 전체 groupby / rolling / frequency / rank / aggregate를 사용하지 않는다.
# - HB-EB prior와 HC lookup은 공식 train 2019~2024로 미리 고정된 artifact만 사용한다.
# - HC lookup key는 현재 행의 pitcher_id, balls_before, strikes_before만 사용한다.

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v006"

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

HC_COUNT_STATE_COL = "__hc_count_state"

EXPECTED_HC_FEATURES = [
    "hc_pitcher_count_n",
    "hc_pitcher_count_eb_rate",
]


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
    train.py artifact에 저장된 static HB-EB prior를 현재 행에 lookup한다.
    다른 test 행은 사용하지 않는다.
    """
    lookup = (
        df[group_cols]
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
            validate="many_to_one",
        )
    )

    alpha = (
        lookup["alpha"]
        .fillna(global_prior[0])
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup["beta"]
        .fillna(global_prior[1])
        .to_numpy(dtype=np.float64)
    )

    n = (
        df[n_col]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        df[rate_col]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    success_count = n * rate

    eb_rate = (
        success_count + alpha
    ) / (
        n + alpha + beta
    )

    return pd.Series(
        eb_rate,
        index=df.index,
        dtype=np.float64,
    )


# =========================================================
# HC-EB: Pitcher × Count-State
# =========================================================

def add_count_state(df):
    """
    balls_before × strikes_before를 HC lookup key로 만든다.
    """
    df = df.copy()

    required = {
        "balls_before",
        "strikes_before",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "HC-EB 추론 필수 컬럼 누락: "
            f"{sorted(missing)}"
        )

    df[HC_COUNT_STATE_COL] = (
        df["balls_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
        + "_"
        + df["strikes_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    return df


def validate_hc_lookup(lookup):
    required_keys = {
        "prior",
        "pitcher_lookup",
        "child_lookup",
    }

    if not required_keys.issubset(
        set(lookup.keys())
    ):
        raise ValueError(
            "Artifact HC lookup 구조가 예상과 다릅니다."
        )

    prior = lookup["prior"]
    pitcher = lookup["pitcher_lookup"]
    child = lookup["child_lookup"]

    required_prior = {
        "global_mean",
        "prior_strength",
    }

    if not required_prior.issubset(
        set(prior.keys())
    ):
        raise ValueError(
            "Artifact HC prior 구조가 예상과 다릅니다."
        )

    required_pitcher_cols = {
        "pitcher_id",
        "hc_pitcher_parent_rate",
        "hc_pitcher_parent_n",
    }

    required_child_cols = {
        "pitcher_id",
        HC_COUNT_STATE_COL,
        "hc_pitcher_count_n",
        "hc_pitcher_count_eb_rate",
    }

    if not required_pitcher_cols.issubset(
        set(pitcher.columns)
    ):
        raise ValueError(
            "Artifact HC pitcher_lookup 컬럼 누락"
        )

    if not required_child_cols.issubset(
        set(child.columns)
    ):
        raise ValueError(
            "Artifact HC child_lookup 컬럼 누락"
        )

    if pitcher["pitcher_id"].duplicated().any():
        raise ValueError(
            "HC pitcher_lookup의 pitcher_id가 unique하지 않습니다."
        )

    if child[
        [
            "pitcher_id",
            HC_COUNT_STATE_COL,
        ]
    ].duplicated().any():
        raise ValueError(
            "HC child_lookup key가 unique하지 않습니다."
        )


def apply_hc_pitcher_count_lookup(
    df,
    lookup,
    fallback_col="hbeb_pitcher_success_rate",
):
    """
    official train 2019~2024만으로 만든 static lookup을 현재 test 행에 적용한다.

    fallback:
      exact pitcher×count
      -> pitcher parent
      -> 현재 행의 기존 HB-EB pitcher rate
      -> global prior

    test의 다른 행이나 test 전체 분포는 전혀 사용하지 않는다.
    """
    validate_hc_lookup(
        lookup
    )

    out = add_count_state(
        df
    )

    child = (
        lookup[
            "child_lookup"
        ]
        .copy()
    )

    pitcher = (
        lookup[
            "pitcher_lookup"
        ]
        .copy()
    )

    prior = (
        lookup["prior"]
    )

    out = out.merge(
        child,
        on=[
            "pitcher_id",
            HC_COUNT_STATE_COL,
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    out = out.merge(
        pitcher,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    exact_hit = (
        out[
            "hc_pitcher_count_eb_rate"
        ].notna()
    )

    parent_hit = (
        (~exact_hit)
        & out[
            "hc_pitcher_parent_rate"
        ].notna()
    )

    if fallback_col in out.columns:
        row_fallback = (
            out[
                fallback_col
            ]
        )
    else:
        row_fallback = pd.Series(
            np.nan,
            index=out.index,
        )

    global_mean = float(
        prior[
            "global_mean"
        ]
    )

    out[
        "hc_pitcher_count_eb_rate"
    ] = (
        out[
            "hc_pitcher_count_eb_rate"
        ]
        .fillna(
            out[
                "hc_pitcher_parent_rate"
            ]
        )
        .fillna(
            row_fallback
        )
        .fillna(
            global_mean
        )
        .astype(np.float64)
    )

    out[
        "hc_pitcher_count_n"
    ] = (
        out[
            "hc_pitcher_count_n"
        ]
        .fillna(0.0)
        .astype(np.float64)
    )

    safe_row_fallback = (
        (~exact_hit)
        & (~parent_hit)
    )

    diagnostics = {
        "n_rows": int(
            len(out)
        ),
        "exact_hit_count": int(
            exact_hit.sum()
        ),
        "exact_hit_ratio": float(
            exact_hit.mean()
        ),
        "parent_fallback_count": int(
            parent_hit.sum()
        ),
        "parent_fallback_ratio": float(
            parent_hit.mean()
        ),
        "safe_row_fallback_count": int(
            safe_row_fallback.sum()
        ),
        "safe_row_fallback_ratio": float(
            safe_row_fallback.mean()
        ),
        "mean_support": float(
            out[
                "hc_pitcher_count_n"
            ].mean()
        ),
        "median_support": float(
            np.median(
                out[
                    "hc_pitcher_count_n"
                ]
            )
        ),
        "global_mean": float(
            global_mean
        ),
        "prior_strength": float(
            prior[
                "prior_strength"
            ]
        ),
    }

    out = out.drop(
        columns=[
            HC_COUNT_STATE_COL,
            "hc_pitcher_parent_rate",
            "hc_pitcher_parent_n",
        ],
        errors="ignore",
    )

    return (
        out,
        diagnostics,
    )


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )

    return df


# =========================================================
# Submission
# =========================================================

def build_submission(
    sample_submission,
    test_ids,
    predictions,
):
    pred_df = pd.DataFrame({
        ID: test_ids,
        TARGET: predictions,
    })

    if pred_df[
        ID
    ].duplicated().any():
        raise ValueError(
            "예측 row_id에 중복이 있습니다."
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

    if submission[
        TARGET
    ].isna().any():
        raise ValueError(
            "제출 prediction에 NaN이 있습니다."
        )

    pred = (
        submission[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "제출 prediction에 NaN 또는 Inf가 있습니다."
        )

    if not submission[
        TARGET
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "제출 prediction에 [0, 1] 범위를 벗어난 값이 있습니다."
        )

    return submission


# =========================================================
# Main
# =========================================================

def main():
    start_time = (
        time.time()
    )

    # -----------------------------------------------------
    # 1. Artifact
    # -----------------------------------------------------

    print("=" * 72)
    print(
        f"CatBoost Regressor Inference : {EXPERIMENT_NAME}"
    )
    print("=" * 72)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 Artifact를 찾을 수 없습니다: {MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    required_artifact_keys = {
        "model",
        "hbeb_priors",
        "features",
        "cat_cols",
        "hc_features",
        "hc_pitcher_count_lookup",
    }

    missing_artifact = (
        required_artifact_keys
        - set(
            artifact.keys()
        )
    )

    if missing_artifact:
        raise ValueError(
            "Artifact 필수 key 누락: "
            f"{sorted(missing_artifact)}"
        )

    model = artifact[
        "model"
    ]

    hbeb_priors = artifact[
        "hbeb_priors"
    ]

    final_features = artifact[
        "features"
    ]

    cat_cols = artifact[
        "cat_cols"
    ]

    hc_features = artifact[
        "hc_features"
    ]

    hc_lookup = artifact[
        "hc_pitcher_count_lookup"
    ]

    if type(
        model
    ).__name__ != "CatBoostRegressor":
        raise TypeError(
            "cb_reg_v006 Artifact의 model은 "
            "CatBoostRegressor여야 합니다. "
            f"actual={type(model).__name__}"
        )

    artifact_experiment = artifact.get(
        "experiment_name"
    )

    if (
        artifact_experiment is not None
        and artifact_experiment
        != EXPERIMENT_NAME
    ):
        raise ValueError(
            "Artifact experiment_name과 script 설정이 다릅니다. "
            f"artifact={artifact_experiment}, "
            f"script={EXPERIMENT_NAME}"
        )

    objective = artifact.get(
        "objective"
    )

    if (
        objective is not None
        and objective != "RMSE"
    ):
        raise ValueError(
            "cb_reg_v006 Artifact objective는 RMSE여야 합니다. "
            f"actual={objective}"
        )

    if list(
        hc_features
    ) != EXPECTED_HC_FEATURES:
        raise ValueError(
            "Artifact HC Feature 구성이 예상과 다릅니다. "
            f"actual={hc_features}"
        )

    expected_hc_policy = (
        "global_to_pitcher_to_pitcher_count_static_train_only_lookup"
    )

    artifact_hc_policy = artifact.get(
        "hc_policy"
    )

    if (
        artifact_hc_policy is not None
        and artifact_hc_policy
        != expected_hc_policy
    ):
        raise ValueError(
            "Artifact HC policy가 예상과 다릅니다. "
            f"actual={artifact_hc_policy}"
        )

    validate_hc_lookup(
        hc_lookup
    )

    print(
        f"- 모델 타입      : {type(model).__name__}"
    )
    print(
        f"- Feature 수      : {len(final_features)}"
    )
    print(
        f"- HC Feature      : {hc_features}"
    )
    print(
        f"- HC prior mean   : "
        f"{hc_lookup['prior']['global_mean']:.6f}"
    )
    print(
        f"- HC strength     : "
        f"{hc_lookup['prior']['prior_strength']:.4f}"
    )

    # -----------------------------------------------------
    # 2. Data
    # -----------------------------------------------------

    print()
    print("Test 데이터 로드")

    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    sample_sub = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    if ID not in test_df.columns:
        raise ValueError(
            f"test.csv에 {ID}가 없습니다."
        )

    test_ids = (
        test_df[
            ID
        ]
        .to_numpy()
    )

    print(
        f"- Test rows       : {len(test_df):,}"
    )

    # -----------------------------------------------------
    # 3. HB-EB
    # -----------------------------------------------------

    print()
    print("HB-EB 적용")

    for (
        new_col,
        config,
    ) in hbeb_priors.items():
        test_df[
            new_col
        ] = apply_hbeb(
            df=test_df,
            group_cols=(
                config[
                    "group_cols"
                ]
            ),
            n_col=(
                config[
                    "n_col"
                ]
            ),
            rate_col=(
                config[
                    "rate_col"
                ]
            ),
            prior_df=(
                config[
                    "prior_df"
                ]
            ),
            global_prior=(
                config[
                    "global_prior"
                ]
            ),
        )

    # -----------------------------------------------------
    # 4. HC-EB
    # -----------------------------------------------------

    print()
    print(
        "Pitcher × Count-State HC-EB 적용"
    )

    (
        test_df,
        hc_diag,
    ) = (
        apply_hc_pitcher_count_lookup(
            df=test_df,
            lookup=hc_lookup,
            fallback_col=(
                "hbeb_pitcher_success_rate"
            ),
        )
    )

    print(
        f"- Exact hit        : "
        f"{hc_diag['exact_hit_count']:,} "
        f"({hc_diag['exact_hit_ratio']:.2%})"
    )

    print(
        f"- Parent fallback  : "
        f"{hc_diag['parent_fallback_count']:,} "
        f"({hc_diag['parent_fallback_ratio']:.2%})"
    )

    print(
        f"- Row fallback     : "
        f"{hc_diag['safe_row_fallback_count']:,} "
        f"({hc_diag['safe_row_fallback_ratio']:.2%})"
    )

    print(
        f"- Mean support     : "
        f"{hc_diag['mean_support']:.2f}"
    )

    print(
        f"- Median support   : "
        f"{hc_diag['median_support']:.2f}"
    )

    # -----------------------------------------------------
    # 5. Feature
    # -----------------------------------------------------

    print()
    print(
        "최종 Feature 정렬"
    )

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

    missing_hc_features = [
        c
        for c in hc_features
        if c not in final_features
    ]

    if missing_hc_features:
        raise ValueError(
            "Model final_features에 HC Feature가 없습니다: "
            f"{missing_hc_features}"
        )

    X_test = (
        prepare_features(
            test_df[
                final_features
            ],
            cat_cols,
        )
    )

    model_feature_names = list(
        model.feature_names_
    )

    if (
        model_feature_names
        != list(
            X_test.columns
        )
    ):
        raise ValueError(
            "추론 Feature 순서가 학습 시 Feature 순서와 다릅니다."
        )

    # -----------------------------------------------------
    # 6. Inference
    # -----------------------------------------------------

    print()
    print(
        "Inference 시작"
    )

    pred_raw = (
        model.predict(
            X_test
        )
    )

    pred_raw = np.asarray(
        pred_raw,
        dtype=np.float64,
    )

    if not np.isfinite(
        pred_raw
    ).all():
        raise ValueError(
            "Raw prediction에 NaN 또는 Inf가 있습니다."
        )

    pred_prob = np.clip(
        pred_raw,
        0.0,
        1.0,
    )

    outside = (
        (pred_raw < 0.0)
        | (pred_raw > 1.0)
    )

    print()
    print(
        "[Prediction diagnostics]"
    )
    print(
        f"- Mean             : {pred_raw.mean():.6f}"
    )
    print(
        f"- Std              : {pred_raw.std():.6f}"
    )
    print(
        f"- Min              : {pred_raw.min():.6f}"
    )
    print(
        f"- Max              : {pred_raw.max():.6f}"
    )
    print(
        f"- Outside [0,1]    : "
        f"{int(outside.sum()):,} "
        f"({outside.mean():.6%})"
    )

    # -----------------------------------------------------
    # 7. Submission
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission = (
        build_submission(
            sample_submission=(
                sample_sub
            ),
            test_ids=(
                test_ids
            ),
            predictions=(
                pred_prob
            ),
        )
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 72)
    print(
        "제출 파일 저장 완료"
    )
    print(
        f"- 저장 경로       : {OUTPUT_PATH}"
    )
    print(
        f"- Prediction mean : "
        f"{submission[TARGET].mean():.6f}"
    )
    print(
        f"- Prediction min  : "
        f"{submission[TARGET].min():.6f}"
    )
    print(
        f"- Prediction max  : "
        f"{submission[TARGET].max():.6f}"
    )
    print(
        f"- 총 실행 시간    : "
        f"{elapsed:.2f} sec"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

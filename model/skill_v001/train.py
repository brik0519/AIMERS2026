# train.py: skill_v001
#
# 목적:
# CatBoost baseline에 pitcher의 historical information을
# "long-term skill + uncertainty + recent form + multi-scale trend"로
# 구조화하여 추가하고, Feature family 단위 ablation으로 검증한다.
#
# Ablation:
#
#   BASELINE
#       기존 47개 feature
#
#   A_LONG_TERM
#       + pitcher_skill_post_mean
#
#   B_UNCERTAINTY
#       + pitcher_skill_post_sd
#       + pitcher_skill_shrink_gap
#
#   C_RECENT_FORM
#       + prev1 - long_skill
#       + prev3 - long_skill
#       + prev5 - long_skill
#
#   D_MULTI_SCALE
#       + prev1 - prev3
#       + prev3 - prev5
#       + recent window range
#
# 원칙:
# - 모든 engineered feature는 공식 asof_* 컬럼만 사용한다.
# - 현재/미래 target을 feature 생성에 사용하지 않는다.
# - train / validation / test에서 동일한 deterministic transform을 사용한다.
# - raw asof feature는 제거하지 않는다.
# - CatBoost parameter는 cb_v001과 동일하게 유지한다.
# - Primary metric은 Brier Score.
# - 2019~2023 Train / 2024 Validation.
#
# 참고:
# asof_pitcher_n과 asof_pitcher_success_rate가 정확히 같은 denominator 관계인지
# 별도 audit를 출력한다. 일치하지 않더라도 본 코드는 이들을
# "count-aware shrinkage transform"으로만 사용하므로 실행은 계속한다.
#
# package.py 호환:
# EXPERIMENT_NAME / MODEL_PATH를 제공하고
# 단일 pkl bundle을 ./model/<experiment>/ 에 저장한다.


import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 실험 설정
# =========================================================

DATA_DIR = Path("./data")

ID = "row_id"
TARGET = "control_success"

SEASON_COL = "season"

PITCHER_N = "asof_pitcher_n"
PITCHER_RATE = "asof_pitcher_success_rate"

PREV1 = "asof_pitcher_prev1_game_success_rate"
PREV3 = "asof_pitcher_prev3_game_success_rate"
PREV5 = "asof_pitcher_prev5_game_success_rate"

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
# Experiment
# =========================================================

EXPERIMENT_NAME = "skill_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# CatBoost 설정
# cb_v001과 동일
# =========================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80
VALIDATION_SEASON = 2024


# =========================================================
# Shrinkage 설정
# =========================================================

# 첫 screening에서는 하나의 prior만 고정한다.
# baseline을 이긴 뒤에만 strength sweep을 권장한다.
PRIOR_MEAN = 0.5
PRIOR_STRENGTH = 50.0


# =========================================================
# Engineered feature names
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
# Ablation 정의
# =========================================================

ABLATIONS = [
    {
        "name": "A_LONG_TERM",
        "new_features": [
            F_POST_MEAN,
        ],
        "description": (
            "Count-aware long-term pitcher skill only"
        ),
    },
    {
        "name": "B_UNCERTAINTY",
        "new_features": [
            F_POST_MEAN,
            F_POST_SD,
            F_SHRINK_GAP,
        ],
        "description": (
            "Long-term skill + posterior-style uncertainty"
        ),
    },
    {
        "name": "C_RECENT_FORM",
        "new_features": [
            F_POST_MEAN,
            F_POST_SD,
            F_SHRINK_GAP,
            F_PREV1_LONG,
            F_PREV3_LONG,
            F_PREV5_LONG,
        ],
        "description": (
            "Long-term + uncertainty + recent-vs-long form"
        ),
    },
    {
        "name": "D_MULTI_SCALE",
        "new_features": [
            F_POST_MEAN,
            F_POST_SD,
            F_SHRINK_GAP,
            F_PREV1_LONG,
            F_PREV3_LONG,
            F_PREV5_LONG,
            F_PREV1_PREV3,
            F_PREV3_PREV5,
            F_RECENT_RANGE,
        ],
        "description": (
            "Long-term + uncertainty + recent form "
            "+ multi-scale recent trend"
        ),
    },
]


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    """
    cb_v001과 동일한 전처리.

    - 수치형 결측값은 CatBoost가 직접 처리
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 문자열 dtype
    """

    out = df.copy()

    for col in cat_cols:

        out[col] = (
            out[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return out


# =========================================================
# Metric
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    y_pred = np.clip(
        y_pred,
        0.0,
        1.0,
    )

    r = float(
        np.mean(y_true)
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    baseline_brier = float(
        r * (1.0 - r)
    )

    if baseline_brier <= 0:

        bss = 0.0

    else:

        bss = max(
            0.0,
            100000.0
            * (
                1.0
                - brier / baseline_brier
            ),
        )

    return {
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": float(bss),
        "positive_rate": r,
    }


# =========================================================
# Official-asof denominator audit
# =========================================================

def audit_pitcher_rate_denominator(
    df,
):
    """
    asof_pitcher_n * asof_pitcher_success_rate가
    integer success count와 얼마나 일치하는지 확인한다.

    공식 문서상 두 값 모두 투구 직전 누적 통계이지만,
    정확한 denominator identity를 이름만으로 단정하지 않는다.

    결과가 rate 저장 정밀도 수준에서 거의 integer라면
    Beta-Binomial 해석의 근거가 강해진다.
    그렇지 않아도 아래 transform은 count-aware shrinkage로 유효하다.
    """

    print()
    print("=" * 60)
    print("Pitcher rate denominator audit")
    print("=" * 60)

    n = pd.to_numeric(
        df[PITCHER_N],
        errors="coerce",
    )

    p = pd.to_numeric(
        df[PITCHER_RATE],
        errors="coerce",
    )

    valid = (
        n.notna()
        & p.notna()
        & (n > 0)
    )

    n = (
        n.loc[valid]
        .astype(np.float64)
    )

    p = (
        p.loc[valid]
        .astype(np.float64)
    )

    pseudo_success = (
        n * p
    )

    integer_gap = np.abs(
        pseudo_success
        - np.rint(
            pseudo_success
        )
    )

    if len(integer_gap) == 0:

        print(
            "검증 가능한 row가 없습니다."
        )

        return

    print(
        f"Comparable rows : {len(integer_gap):,}"
    )

    print(
        f"Mean abs gap    : "
        f"{integer_gap.mean():.10f}"
    )

    print(
        f"Median abs gap  : "
        f"{np.median(integer_gap):.10f}"
    )

    print(
        f"Max abs gap     : "
        f"{integer_gap.max():.10f}"
    )

    for tol in [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
    ]:

        ratio = float(
            np.mean(
                integer_gap <= tol
            )
        )

        print(
            f"Integer-like <= {tol:<7g}: "
            f"{ratio:.6%}"
        )

    print("=" * 60)


# =========================================================
# Skill feature 생성
# =========================================================

def add_pitcher_skill_features(
    df,
    prior_mean=PRIOR_MEAN,
    prior_strength=PRIOR_STRENGTH,
):
    """
    공식 row-local asof feature만 이용하여
    deterministic skill representation을 만든다.

    TARGET은 사용하지 않는다.

    A. long-term:
       post_mean = (n*p + m*mu) / (n+m)

    B. uncertainty:
       post_sd
       shrink_gap

    C. recent form:
       prev1/3/5 - long skill

    D. multi-scale trend:
       prev1-prev3
       prev3-prev5
       max(prev1,3,5)-min(prev1,3,5)

    recent NaN은 억지로 long-term으로 채우지 않는다.
    CatBoost가 결측 자체를 처리하도록 그대로 유지한다.
    """

    out = df.copy()

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
    #
    # exact Beta posterior interpretation은
    # denominator audit 결과에 따라 달라질 수 있으므로
    # posterior-style uncertainty transform으로 사용한다.
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
# CatBoost 학습
# =========================================================

def train_validation_model(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_cols,
):

    estimator = (
        CatBoostClassifier(
            **MODEL_PARAMS
        )
    )

    start_time = time.time()

    estimator.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(
            X_val,
            y_val,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    elapsed = (
        time.time()
        - start_time
    )

    val_pred = (
        estimator.predict_proba(
            X_val
        )[:, 1]
    )

    metrics = (
        calculate_metrics(
            y_val,
            val_pred,
        )
    )

    best_iteration = (
        estimator.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):

        final_iterations = int(
            MODEL_PARAMS[
                "iterations"
            ]
        )

    else:

        final_iterations = int(
            best_iteration
            + 1
        )

    return {
        "estimator": estimator,
        "prediction": val_pred,
        "metrics": metrics,
        "best_iteration": int(
            best_iteration
        ),
        "final_iterations": (
            final_iterations
        ),
        "elapsed": float(
            elapsed
        ),
    }


# =========================================================
# Subgroup Brier
# =========================================================

def calculate_subgroup_brier(
    val_df,
    y_true,
    baseline_pred,
    candidate_pred,
):
    """
    history support별 Brier를 비교한다.

    목적:
    uncertainty/shrinkage가 low-history pitcher에서
    실제로 도움을 주는지 확인.
    """

    n = pd.to_numeric(
        val_df[PITCHER_N],
        errors="coerce",
    ).fillna(0.0)

    bins = [
        -np.inf,
        0,
        50,
        200,
        1000,
        np.inf,
    ]

    labels = [
        "n=0",
        "1-50",
        "51-200",
        "201-1000",
        "1001+",
    ]

    bucket = pd.cut(
        n,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    diag = pd.DataFrame({
        "bucket": bucket,
        "target": np.asarray(
            y_true
        ),
        "baseline_pred": np.asarray(
            baseline_pred
        ),
        "candidate_pred": np.asarray(
            candidate_pred
        ),
    })

    rows = []

    for label in labels:

        block = diag.loc[
            diag["bucket"]
            == label
        ]

        if len(block) == 0:
            continue

        y = (
            block["target"]
            .to_numpy(
                dtype=np.float64
            )
        )

        p_base = (
            block["baseline_pred"]
            .to_numpy(
                dtype=np.float64
            )
        )

        p_new = (
            block["candidate_pred"]
            .to_numpy(
                dtype=np.float64
            )
        )

        base_brier = float(
            np.mean(
                (p_base - y) ** 2
            )
        )

        new_brier = float(
            np.mean(
                (p_new - y) ** 2
            )
        )

        rows.append({
            "bucket": label,
            "n_rows": int(
                len(block)
            ),
            "baseline_brier": (
                base_brier
            ),
            "candidate_brier": (
                new_brier
            ),
            "delta_brier": float(
                new_brier
                - base_brier
            ),
        })

    return rows


# =========================================================
# Validation artifact 저장
# =========================================================

def save_ablation_validation(
    output_dir,
    row_ids,
    y_true,
    y_pred,
    metadata,
    subgroup_rows=None,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    pred_df = pd.DataFrame({
        "row_id": np.asarray(
            row_ids
        ),
        "target": np.asarray(
            y_true
        ),
        "prediction": np.asarray(
            y_pred
        ),
    })

    pred_df.to_csv(
        output_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        output_dir / "validation.npz",
        row_id=np.asarray(
            row_ids
        ),
        target=np.asarray(
            y_true
        ),
        prediction=np.asarray(
            y_pred
        ),
    )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if subgroup_rows is not None:

        pd.DataFrame(
            subgroup_rows
        ).to_csv(
            output_dir
            / "subgroup_brier.csv",
            index=False,
            encoding="utf-8-sig",
        )


# =========================================================
# Console output
# =========================================================

def print_result(
    title,
    result,
    baseline_metrics=None,
):

    metrics = result[
        "metrics"
    ]

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        "Brier Score          : "
        f"{metrics['brier']:.8f}"
    )

    print(
        "Baseline Brier       : "
        f"{metrics['baseline_brier']:.8f}"
    )

    print(
        "Validation Score     : "
        f"{metrics['bss']:.2f}"
    )

    if baseline_metrics is not None:

        print(
            "ΔBrier vs baseline   : "
            f"{metrics['brier'] - baseline_metrics['brier']:+.8f}"
        )

        print(
            "ΔScore vs baseline   : "
            f"{metrics['bss'] - baseline_metrics['bss']:+.2f}"
        )

    print(
        "Best iteration       : "
        f"{result['best_iteration']}"
    )

    print(
        "Final iterations     : "
        f"{result['final_iterations']}"
    )

    print(
        "Training seconds     : "
        f"{result['elapsed']:.1f}"
    )

    print(
        "Prediction mean      : "
        f"{np.mean(result['prediction']):.6f}"
    )

    print(
        "Prediction std       : "
        f"{np.std(result['prediction']):.6f}"
    )

    print(
        "Prediction min       : "
        f"{np.min(result['prediction']):.6f}"
    )

    print(
        "Prediction max       : "
        f"{np.max(result['prediction']):.6f}"
    )

    print("=" * 60)


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():

        raise FileExistsError(
            "Validation experiment already exists: "
            f"{VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_DIR.exists():

        raise FileExistsError(
            "Model experiment already exists: "
            f"{MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # 데이터 schema
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        DATA_DIR / "test.csv",
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        c
        for c in test_cols
        if c != ID
    ]

    required = [
        PITCHER_N,
        PITCHER_RATE,
        PREV1,
        PREV3,
        PREV5,
        SEASON_COL,
    ]

    missing = [
        c
        for c in required
        if c not in base_features
    ]

    if missing:

        raise KeyError(
            "skill_v001에 필요한 공식 feature가 없습니다: "
            f"{missing}"
        )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[
            ID,
            *base_features,
            TARGET,
        ],
    )

    print("=" * 60)
    print("데이터 로드 완료")
    print("train shape:", train.shape)
    print(
        "기존 Feature 수:",
        len(base_features),
    )
    print(
        "범주형 Feature 수:",
        len(cat_cols),
    )
    print(
        "범주형 Feature:",
        cat_cols,
    )
    print(
        "시즌:",
        train[SEASON_COL].min(),
        "~",
        train[SEASON_COL].max(),
    )
    print(
        f"제구 성공률: "
        f"{train[TARGET].mean():.6f}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # denominator audit
    # -----------------------------------------------------

    audit_pitcher_rate_denominator(
        train
    )

    # -----------------------------------------------------
    # Engineered feature 전체 한 번 생성
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("Pitcher skill feature 생성")
    print("=" * 60)

    train_fe = (
        add_pitcher_skill_features(
            train,
            prior_mean=(
                PRIOR_MEAN
            ),
            prior_strength=(
                PRIOR_STRENGTH
            ),
        )
    )

    all_engineered = (
        ABLATIONS[-1][
            "new_features"
        ]
    )

    print(
        "Engineered Feature 수:",
        len(all_engineered),
    )

    for feature in all_engineered:
        print(
            f"- {feature}"
        )

    print("=" * 60)

    # -----------------------------------------------------
    # Temporal split
    # -----------------------------------------------------

    is_val = (
        train_fe[SEASON_COL]
        == VALIDATION_SEASON
    )

    fit_df = (
        train_fe.loc[
            ~is_val
        ]
        .reset_index(
            drop=True
        )
    )

    val_df = (
        train_fe.loc[
            is_val
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = (
        fit_df[TARGET]
        .copy()
    )

    y_val = (
        val_df[TARGET]
        .copy()
    )

    val_row_id = (
        val_df[ID]
        .copy()
    )

    print()
    print("=" * 60)
    print("Validation Split")
    print("=" * 60)

    print(
        "학습 데이터:",
        fit_df.shape,
    )

    print(
        "검증 데이터:",
        val_df.shape,
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Validation root 생성
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # =====================================================
    # 1. Baseline
    # =====================================================

    print()
    print("=" * 60)
    print("BASELINE CatBoost 학습 시작")
    print("=" * 60)

    X_train_base = prepare_features(
        fit_df[
            base_features
        ],
        cat_cols,
    )

    X_val_base = prepare_features(
        val_df[
            base_features
        ],
        cat_cols,
    )

    baseline_result = (
        train_validation_model(
            X_train_base,
            y_train,
            X_val_base,
            y_val,
            cat_cols,
        )
    )

    baseline_metrics = (
        baseline_result[
            "metrics"
        ]
    )

    baseline_pred = (
        baseline_result[
            "prediction"
        ]
    )

    print_result(
        "BASELINE Validation",
        baseline_result,
    )

    save_ablation_validation(
        output_dir=(
            VALIDATION_DIR
            / "BASELINE"
        ),
        row_ids=(
            val_row_id
        ),
        y_true=(
            y_val
        ),
        y_pred=(
            baseline_pred
        ),
        metadata={
            "experiment_name": (
                EXPERIMENT_NAME
            ),
            "ablation": (
                "BASELINE"
            ),
            "features_added": [],
            "model_params": (
                MODEL_PARAMS
            ),
            "validation_season": (
                VALIDATION_SEASON
            ),
            "brier": float(
                baseline_metrics[
                    "brier"
                ]
            ),
            "baseline_brier": float(
                baseline_metrics[
                    "baseline_brier"
                ]
            ),
            "bss": float(
                baseline_metrics[
                    "bss"
                ]
            ),
            "best_iteration": int(
                baseline_result[
                    "best_iteration"
                ]
            ),
            "final_iterations": int(
                baseline_result[
                    "final_iterations"
                ]
            ),
        },
    )

    # baseline estimator는 final에 필요 없음
    del baseline_result[
        "estimator"
    ]

    # =====================================================
    # 2. Ablation sequence
    # =====================================================

    ablation_results = []

    best_candidate = None

    for spec in ABLATIONS:

        ablation_name = (
            spec["name"]
        )

        new_features = list(
            spec["new_features"]
        )

        description = (
            spec["description"]
        )

        features = (
            base_features
            + new_features
        )

        experiment_cat_cols = [
            c
            for c in cat_cols
            if c in features
        ]

        print()
        print("=" * 60)
        print(
            f"{ablation_name} 학습 시작"
        )
        print("=" * 60)

        print(
            f"Description: {description}"
        )

        print(
            "추가 Feature:"
        )

        for feature in new_features:
            print(
                f"- {feature}"
            )

        X_train = prepare_features(
            fit_df[
                features
            ],
            experiment_cat_cols,
        )

        X_val = prepare_features(
            val_df[
                features
            ],
            experiment_cat_cols,
        )

        result = (
            train_validation_model(
                X_train,
                y_train,
                X_val,
                y_val,
                experiment_cat_cols,
            )
        )

        print_result(
            f"{ablation_name} Validation",
            result,
            baseline_metrics=(
                baseline_metrics
            ),
        )

        metrics = (
            result[
                "metrics"
            ]
        )

        delta_brier = float(
            metrics["brier"]
            - baseline_metrics[
                "brier"
            ]
        )

        delta_bss = float(
            metrics["bss"]
            - baseline_metrics[
                "bss"
            ]
        )

        subgroup_rows = (
            calculate_subgroup_brier(
                val_df,
                y_val,
                baseline_pred,
                result[
                    "prediction"
                ],
            )
        )

        print()
        print(
            "[History-count subgroup Brier]"
        )

        for row in subgroup_rows:

            print(
                f"{row['bucket']:<10} "
                f"n={row['n_rows']:>7,} "
                f"base={row['baseline_brier']:.8f} "
                f"new={row['candidate_brier']:.8f} "
                f"Δ={row['delta_brier']:+.8f}"
            )

        summary = {
            "name": (
                ablation_name
            ),
            "description": (
                description
            ),
            "features_added": (
                new_features
            ),
            "n_features_added": int(
                len(
                    new_features
                )
            ),
            "brier": float(
                metrics[
                    "brier"
                ]
            ),
            "baseline_brier": float(
                metrics[
                    "baseline_brier"
                ]
            ),
            "bss": float(
                metrics[
                    "bss"
                ]
            ),
            "delta_brier": float(
                delta_brier
            ),
            "delta_bss": float(
                delta_bss
            ),
            "best_iteration": int(
                result[
                    "best_iteration"
                ]
            ),
            "final_iterations": int(
                result[
                    "final_iterations"
                ]
            ),
            "training_seconds": float(
                result[
                    "elapsed"
                ]
            ),
            "subgroup_brier": (
                subgroup_rows
            ),
        }

        ablation_results.append(
            summary
        )

        save_ablation_validation(
            output_dir=(
                VALIDATION_DIR
                / ablation_name
            ),
            row_ids=(
                val_row_id
            ),
            y_true=(
                y_val
            ),
            y_pred=(
                result[
                    "prediction"
                ]
            ),
            metadata={
                "experiment_name": (
                    EXPERIMENT_NAME
                ),
                "ablation": (
                    ablation_name
                ),
                "description": (
                    description
                ),
                "features_added": (
                    new_features
                ),
                "prior_mean": float(
                    PRIOR_MEAN
                ),
                "prior_strength": float(
                    PRIOR_STRENGTH
                ),
                "model_params": (
                    MODEL_PARAMS
                ),
                "validation_season": (
                    VALIDATION_SEASON
                ),
                **summary,
            },
            subgroup_rows=(
                subgroup_rows
            ),
        )

        # best engineered candidate 선택
        if (
            best_candidate is None
            or metrics["brier"]
            < best_candidate[
                "summary"
            ]["brier"]
        ):

            if (
                best_candidate
                is not None
            ):
                del best_candidate[
                    "model"
                ]

            best_candidate = {
                "model": (
                    result[
                        "estimator"
                    ]
                ),
                "prediction": (
                    result[
                        "prediction"
                    ]
                ),
                "summary": (
                    summary
                ),
                "features": (
                    features
                ),
                "cat_cols": (
                    experiment_cat_cols
                ),
            }

        else:

            del result[
                "estimator"
            ]

    # =====================================================
    # 3. Ablation summary
    # =====================================================

    print()
    print("=" * 70)
    print("skill_v001 Ablation 결과")
    print("=" * 70)

    print(
        f"{'Model':<18} "
        f"{'Brier':>12} "
        f"{'ΔBrier':>12} "
        f"{'BSS':>10}"
    )

    print(
        f"{'BASELINE':<18} "
        f"{baseline_metrics['brier']:>12.8f} "
        f"{0.0:>+12.8f} "
        f"{baseline_metrics['bss']:>10.2f}"
    )

    for row in ablation_results:

        print(
            f"{row['name']:<18} "
            f"{row['brier']:>12.8f} "
            f"{row['delta_brier']:>+12.8f} "
            f"{row['bss']:>10.2f}"
        )

    best_summary = (
        best_candidate[
            "summary"
        ]
    )

    decision = (
        "GO"
        if best_summary[
            "brier"
        ] < baseline_metrics[
            "brier"
        ]
        else "NO-GO"
    )

    print()
    print(
        f"Best engineered model : "
        f"{best_summary['name']}"
    )

    print(
        f"Best Brier            : "
        f"{best_summary['brier']:.8f}"
    )

    print(
        f"Best ΔBrier           : "
        f"{best_summary['delta_brier']:+.8f}"
    )

    print(
        f"Best BSS              : "
        f"{best_summary['bss']:.2f}"
    )

    print(
        f"Decision              : "
        f"{decision}"
    )

    print("=" * 70)

    # =====================================================
    # 4. Experiment summary 저장
    # =====================================================

    summary_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "validation_season": (
            VALIDATION_SEASON
        ),
        "prior_mean": float(
            PRIOR_MEAN
        ),
        "prior_strength": float(
            PRIOR_STRENGTH
        ),
        "baseline": {
            "brier": float(
                baseline_metrics[
                    "brier"
                ]
            ),
            "bss": float(
                baseline_metrics[
                    "bss"
                ]
            ),
            "best_iteration": int(
                baseline_result[
                    "best_iteration"
                ]
            ),
            "final_iterations": int(
                baseline_result[
                    "final_iterations"
                ]
            ),
        },
        "ablations": (
            ablation_results
        ),
        "best_ablation": (
            best_summary[
                "name"
            ]
        ),
        "decision": (
            decision
        ),
    }

    with open(
        VALIDATION_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =====================================================
    # 5. 전체 데이터 재학습
    # =====================================================

    selected_features = (
        best_candidate[
            "features"
        ]
    )

    selected_cat_cols = (
        best_candidate[
            "cat_cols"
        ]
    )

    final_iterations = int(
        best_summary[
            "final_iterations"
        ]
    )

    print()
    print("=" * 60)
    print("전체 데이터 재학습")
    print("=" * 60)

    print(
        "Selected ablation :",
        best_summary[
            "name"
        ],
    )

    print(
        "Feature 수        :",
        len(
            selected_features
        ),
    )

    print(
        "Final iterations  :",
        final_iterations,
    )

    X_full = prepare_features(
        train_fe[
            selected_features
        ],
        selected_cat_cols,
    )

    y_full = (
        train_fe[TARGET]
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": (
            final_iterations
        ),
    }

    final_model = (
        CatBoostClassifier(
            **final_params
        )
    )

    start_time = time.time()

    final_model.fit(
        X_full,
        y_full,
        cat_features=(
            selected_cat_cols
        ),
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "전체 데이터 재학습 완료: "
        f"{elapsed:.1f}초"
    )

    # =====================================================
    # 6. Bundle 저장
    # =====================================================

    bundle = {
        "model": (
            final_model
        ),
        "features": (
            selected_features
        ),
        "base_features": (
            base_features
        ),
        "cat_cols": (
            selected_cat_cols
        ),
        "feature_config": {
            "prior_mean": float(
                PRIOR_MEAN
            ),
            "prior_strength": float(
                PRIOR_STRENGTH
            ),
            "selected_ablation": (
                best_summary[
                    "name"
                ]
            ),
            "selected_engineered_features": (
                best_summary[
                    "features_added"
                ]
            ),
        },
        "metadata": {
            "experiment_name": (
                EXPERIMENT_NAME
            ),
            "model_type": (
                "CatBoostClassifier"
            ),
            "model_params": (
                final_params
            ),
            "validation": (
                "2019-2023 Train / "
                "2024 Validation"
            ),
            "feature_source": (
                "official row-local asof columns only"
            ),
            "target_used_for_feature_generation": False,
            "prior_mean": float(
                PRIOR_MEAN
            ),
            "prior_strength": float(
                PRIOR_STRENGTH
            ),
            "best_ablation": (
                best_summary[
                    "name"
                ]
            ),
            "best_brier": float(
                best_summary[
                    "brier"
                ]
            ),
            "baseline_brier": float(
                baseline_metrics[
                    "brier"
                ]
            ),
            "delta_brier": float(
                best_summary[
                    "delta_brier"
                ]
            ),
            "best_bss": float(
                best_summary[
                    "bss"
                ]
            ),
            "baseline_bss": float(
                baseline_metrics[
                    "bss"
                ]
            ),
            "delta_bss": float(
                best_summary[
                    "delta_bss"
                ]
            ),
            "decision": (
                decision
            ),
            "best_iteration": int(
                best_summary[
                    "best_iteration"
                ]
            ),
            "final_iterations": int(
                final_iterations
            ),
            "ablations": (
                ablation_results
            ),
            "full_training_seconds": float(
                elapsed
            ),
        },
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        bundle,
        MODEL_PATH,
        compress=3,
    )

    shutil.copy2(
        "./train.py",
        MODEL_DIR / "train.py",
    )

    print()
    print("=" * 60)
    print("skill_v001 저장 완료")
    print("=" * 60)

    print(
        f"Model Bundle : {MODEL_PATH}"
    )

    print(
        f"Train Code   : "
        f"{MODEL_DIR / 'train.py'}"
    )

    print(
        f"Best Ablation: "
        f"{best_summary['name']}"
    )

    print(
        f"Final Iter   : "
        f"{final_iterations}"
    )

    print(
        f"Decision     : "
        f"{decision}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()

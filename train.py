"""wl_v002 — D arm(recency x RMSE) 재평가. LB 0회, train.csv 불필요.

배경       : wl_v001 의 FAIL 은 2023->2024 단일 전이 판정이었다. source 2023 은 병리적
             연도(전 성분 음수 skill, s* 반전)라 '2023의 최근 월을 더 믿기'가 무익했던
             것일 수 있다. 2024-2025 연속성 가설 하에서는 정상 연도 전이가 2025 의
             더 나은 대리다.
설계       : A(균등 x Logloss, 대조군) vs D(recency x RMSE) 를 정상 전이 2개
             (2020->2021, 2021->2022)와 참조 전이(2023->2024)에서 비교.
             입력·프로토콜은 wl_v001 과 동일 (m1 의 98열, pitcher-half 내부 holdout
             트리 수, 가중치는 source 학습에만, anchor 의 m1 자리 대입 평가).
판정 (사전 등록):
             정상 전이 2개 모두에서  [D - A] (anchor 대입) >= +1.0bp
             AND 2023->2024 에서     [D - A] >= -0.5bp  (비열세 가드)
             -> PASS: deploy(2024 source) 에 recency x RMSE m1 을 적용한 full run 을
                Day 2 제출 후보로 승격. FAIL: D arm 최종 폐기.
최우선 규칙: 평가 데이터 독립 추론 — train 내부 전이만 사용, test 무관.

사용:
  python wl_v002_decision.py
  WL2_SMOKE=1 python wl_v002_decision.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "ensemble_v008"

ID = "row_id"
TARGET = "control_success"

BASE_CAT_WEIGHT = 0.5655685
BASE_XGB_WEIGHT = 1.0 - BASE_CAT_WEIGHT

# =========================================================
# Feature 실험 Tag Mode
# =========================================================
# Boolean 조합 대신 "모델:기능" Tag를 조합하여 실험을 정의한다.
# 하나의 Tag는 하나의 가설만 담당하도록 유지한다.
#
# 지원 Tag:
# - "cat:temporal"
#     CatBoost에 현재 행의 공식 asof_* 값으로 만든 Temporal Feature 추가
# - "xgb:temporal"
#     XGBoost에 현재 행의 공식 asof_* 값으로 만든 Temporal Feature 추가
# - "xgb:raw-player-id"
#     XGBoost에 pitcher_id / batter_id raw numeric representation 유지
#     이 Tag가 없으면 pitcher_id / batter_id를 XGBoost Feature에서 제거
# - "xgb:reliability"
#     XGBoost에 exact count-based Beta-Binomial Empirical Bayes
#     reliability Feature를 추가
#
# 대표 실험:
# A) Raw / Raw
#    FEATURE_TAGS = {"xgb:raw-player-id"}
#
# B) Cat Temporal / XGB Raw
#    FEATURE_TAGS = {
#        "cat:temporal",
#        "xgb:raw-player-id",
#    }
#
# C) Cat Raw / XGB Temporal
#    FEATURE_TAGS = {
#        "xgb:temporal",
#        "xgb:raw-player-id",
#    }
#
# D) Temporal / Temporal
#    FEATURE_TAGS = {
#        "cat:temporal",
#        "xgb:temporal",
#        "xgb:raw-player-id",
#    }
#
# C-ID-DROP) Cat Raw / XGB Temporal + raw pitcher/batter ID 제거
#    FEATURE_TAGS = {"xgb:temporal"}
#
# E) Cat Raw / XGB Reliability  <- ensemble_v008 권장
#    FEATURE_TAGS = {
#        "xgb:raw-player-id",
#        "xgb:reliability",
#    }
SUPPORTED_FEATURE_TAGS = {
    "cat:temporal",
    "xgb:temporal",
    "xgb:raw-player-id",
    "xgb:reliability",
}

# ensemble_v007 Control 대비 Reliability만 추가한다.
# Temporal은 끈 상태로 유지한다.
FEATURE_TAGS = {
    "xgb:raw-player-id",
    "xgb:reliability",
}

# 최종 blend weight:
# - pooled_oof : 모든 expanding OOF prediction으로 최적화
# - fixed      : 기존 weight 사용
FINAL_WEIGHT_MODE = "pooled_oof"

# Hidden future와 가장 가까운 2024 Fold의 best iteration을
# Final model iteration으로 사용한다.
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# 경로
# =========================================================
DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# Expanding Window Validation
# =========================================================
EXPANDING_FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2019, 2020],
        "validation_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "validation_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "validation_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "validation_season": 2024,
    },
]


# =========================================================
# CatBoost Feature 정책
# =========================================================
# train_cb.py의 categorical 정책을 그대로 유지한다.
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
# XGBoost Feature 정책
# =========================================================
# train_xgb.py의 low-cardinality categorical 정책을 유지한다.
XGB_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

# 기존 XGBoost와의 비교 가능성을 위해 ID-like dtype 정책은 유지한다.
XGB_ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]

# Specialist 실험:
# player raw identity는 CatBoost가 담당하고,
# XGBoost는 pitcher/batter raw ID 대신 temporal representation을 강조한다.
# 팀 ID는 기존 XGB처럼 numeric 유지한다.
XGB_DROP_RAW_ID_COLS = [
    "pitcher_id",
    "batter_id",
]

TEMPORAL_FEATURES = [
    "tf_success_prev1_vs_career",
    "tf_success_prev3_vs_career",
    "tf_success_prev5_vs_career",
    "tf_success_prev1_vs_prev3",
    "tf_success_prev3_vs_prev5",
    "tf_middle_prev1_vs_career",
    "tf_middle_prev3_vs_career",
    "tf_middle_prev5_vs_career",
    "tf_middle_prev1_vs_prev3",
    "tf_middle_prev3_vs_prev5",
]

RELIABILITY_FEATURES = [
    "rf_pitcher_success_eb_rate",
    "rf_pitcher_success_posterior_std",
]

# Beta-Binomial population prior fit 설정.
# Prior는 반드시 각 expanding fold의 train 구간 target만으로 fit한다.
EB_PRIOR_MIN_PITCHER_TRIALS = 1
EB_INITIAL_STRENGTHS = [20.0, 100.0, 500.0]
EB_LOG_PARAM_LOWER = -12.0
EB_LOG_PARAM_UPPER = 14.0

# denominator audit 결과가 이 임계값을 넘으면 exact count 복원을 중단한다.
# 성공 count rounding이 모호해지는 절대 한계는 0.5지만,
# 현재 실제 audit(max≈0.0076)에 맞춰 훨씬 보수적으로 둔다.
EB_MAX_INTEGER_DISTANCE = 0.05


# =========================================================
# CatBoost 설정 — train_cb.py 적용
# =========================================================
CATBOOST_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

CAT_EARLY_STOPPING_ROUNDS = 80


# =========================================================
# XGBoost 설정 — train_xgb.py 적용
# =========================================================
XGB_MAX_ITERATIONS = 5000
XGB_EARLY_STOPPING_ROUNDS = 200
XGB_RANDOM_STATE = 42

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": XGB_RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# Feature Tag 해석
# =========================================================
def validate_feature_tags(feature_tags):
    feature_tags = set(feature_tags)

    unknown_tags = sorted(
        feature_tags - SUPPORTED_FEATURE_TAGS
    )
    if unknown_tags:
        raise ValueError(
            "지원하지 않는 FEATURE_TAGS가 있습니다: "
            + ", ".join(unknown_tags)
        )

    return feature_tags


def resolve_feature_config(feature_tags):
    """
    Tag를 실제 Feature 정책으로 해석한다.

    실험 정의는 FEATURE_TAGS만으로 관리하며,
    Boolean 값은 프로그램 내부에서만 파생한다.
    """
    feature_tags = validate_feature_tags(
        feature_tags
    )

    return {
        "cat_use_temporal": (
            "cat:temporal" in feature_tags
        ),
        "xgb_use_temporal": (
            "xgb:temporal" in feature_tags
        ),
        "xgb_use_raw_player_ids": (
            "xgb:raw-player-id" in feature_tags
        ),
        "xgb_use_reliability": (
            "xgb:reliability" in feature_tags
        ),
    }


# =========================================================
# asof denominator Audit
# =========================================================
def audit_asof_pitcher_success_denominator(df):
    """
    asof_pitcher_n이 asof_pitcher_success_rate의 denominator인지
    train.csv에서 실증적으로 점검한다.

    검사식:
        abs(n * rate - round(n * rate))

    이 값이 거의 항상 0에 가까우면
    n * rate가 실제 과거 성공 count와 일치할 가능성이 높다.

    주의:
    - train.csv만 사용한다.
    - Feature 생성이나 모델 학습에 관여하지 않는다.
    - exact Beta-Binomial 구현 전 사전 검증 용도다.
    """
    required = [
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    ]
    missing = [
        col
        for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "asof denominator audit에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    n = pd.to_numeric(
        df["asof_pitcher_n"],
        errors="coerce",
    )
    rate = pd.to_numeric(
        df["asof_pitcher_success_rate"],
        errors="coerce",
    )

    valid = (
        n.notna()
        & rate.notna()
        & np.isfinite(n.to_numpy(dtype=float))
        & np.isfinite(rate.to_numpy(dtype=float))
        & (n >= 0)
        & (rate >= 0)
        & (rate <= 1)
    )

    n_valid = n.loc[valid].to_numpy(dtype=float)
    rate_valid = rate.loc[valid].to_numpy(dtype=float)

    product = n_valid * rate_valid
    integer_distance = np.abs(
        product - np.rint(product)
    )

    def _quantile(arr, q):
        if len(arr) == 0:
            return float("nan")
        return float(np.quantile(arr, q))

    thresholds = [
        1e-8,
        1e-6,
        1e-4,
        1e-3,
        1e-2,
    ]

    audit = {
        "definition_check": (
            "Tests whether asof_pitcher_n * "
            "asof_pitcher_success_rate is approximately integer."
        ),
        "n_total_rows": int(len(df)),
        "n_valid_rows": int(len(integer_distance)),
        "valid_row_share": (
            float(len(integer_distance) / len(df))
            if len(df) > 0
            else float("nan")
        ),
        "n_zero_history_rows": int(
            np.sum(n_valid == 0)
        ),
        "integer_distance": {
            "mean": (
                float(np.mean(integer_distance))
                if len(integer_distance) > 0
                else float("nan")
            ),
            "median": _quantile(integer_distance, 0.50),
            "p90": _quantile(integer_distance, 0.90),
            "p95": _quantile(integer_distance, 0.95),
            "p99": _quantile(integer_distance, 0.99),
            "p999": _quantile(integer_distance, 0.999),
            "max": (
                float(np.max(integer_distance))
                if len(integer_distance) > 0
                else float("nan")
            ),
        },
        "share_within_tolerance": {
            f"{threshold:.0e}": (
                float(
                    np.mean(
                        integer_distance <= threshold
                    )
                )
                if len(integer_distance) > 0
                else float("nan")
            )
            for threshold in thresholds
        },
    }

    by_season = []
    if "season" in df.columns:
        for season in sorted(
            df.loc[valid, "season"]
            .dropna()
            .unique()
        ):
            season_mask = (
                valid
                & (df["season"] == season)
            )

            season_n = pd.to_numeric(
                df.loc[
                    season_mask,
                    "asof_pitcher_n",
                ],
                errors="coerce",
            ).to_numpy(dtype=float)

            season_rate = pd.to_numeric(
                df.loc[
                    season_mask,
                    "asof_pitcher_success_rate",
                ],
                errors="coerce",
            ).to_numpy(dtype=float)

            season_distance = np.abs(
                season_n * season_rate
                - np.rint(
                    season_n * season_rate
                )
            )

            by_season.append({
                "season": int(season),
                "n": int(len(season_distance)),
                "median_integer_distance": (
                    _quantile(
                        season_distance,
                        0.50,
                    )
                ),
                "p99_integer_distance": (
                    _quantile(
                        season_distance,
                        0.99,
                    )
                ),
                "share_within_1e-3": (
                    float(
                        np.mean(
                            season_distance <= 1e-3
                        )
                    )
                    if len(season_distance) > 0
                    else float("nan")
                ),
            })

    audit["by_season"] = by_season
    return audit


def print_asof_denominator_audit(audit):
    print()
    print("=" * 72)
    print("asof_pitcher_n / success_rate denominator audit")
    print("=" * 72)
    print(
        "Valid rows              :",
        f"{audit['n_valid_rows']:,}",
        "/",
        f"{audit['n_total_rows']:,}",
    )

    distance = audit["integer_distance"]
    print(
        "Median |n*p-round(n*p)| :",
        f"{distance['median']:.12g}",
    )
    print(
        "P99    |n*p-round(n*p)| :",
        f"{distance['p99']:.12g}",
    )
    print(
        "Max    |n*p-round(n*p)| :",
        f"{distance['max']:.12g}",
    )
    print(
        "Share within 1e-3      :",
        f"{audit['share_within_tolerance']['1e-03']:.6f}",
    )
    print(
        "Share within 1e-2      :",
        f"{audit['share_within_tolerance']['1e-02']:.6f}",
    )
    print(
        "주의: 이 audit은 denominator 관계 검증용이며 "
        "Feature 생성에는 사용되지 않습니다."
    )
    print("=" * 72)


# =========================================================
# Exact count-based Beta-Binomial Empirical Bayes
# =========================================================
def validate_exact_count_reconstruction(audit):
    """
    denominator audit가 exact count reconstruction에 충분한지 검사한다.
    """
    max_distance = float(
        audit["integer_distance"]["max"]
    )

    if not np.isfinite(max_distance):
        raise ValueError(
            "asof denominator audit 결과가 유효하지 않습니다."
        )

    if max_distance > EB_MAX_INTEGER_DISTANCE:
        raise ValueError(
            "Exact EB count reconstruction을 중단합니다. "
            f"max |n*p-round(n*p)|={max_distance:.8f} > "
            f"{EB_MAX_INTEGER_DISTANCE:.8f}"
        )


def reconstruct_pitcher_success_counts(df):
    """
    각 row의 공식 asof_* 값만 이용해 과거 성공/실패 count를 복원한다.

    N = asof_pitcher_n
    p = asof_pitcher_success_rate
    S = round(N * p)
    F = N - S

    다른 validation/test row를 참조하지 않는다.
    """
    required = [
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    ]
    missing = [
        col
        for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "Reliability Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    n_raw = pd.to_numeric(
        df["asof_pitcher_n"],
        errors="coerce",
    ).to_numpy(dtype=float)

    rate = pd.to_numeric(
        df["asof_pitcher_success_rate"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(n_raw)
        & np.isfinite(rate)
        & (n_raw >= 0.0)
        & (rate >= 0.0)
        & (rate <= 1.0)
    )

    n = np.full(
        len(df),
        np.nan,
        dtype=float,
    )
    success = np.full(
        len(df),
        np.nan,
        dtype=float,
    )
    failure = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    # asof_pitcher_n은 count이므로 nearest integer로 복원한다.
    n_valid = np.rint(n_raw[valid])
    success_valid = np.rint(
        n_valid * rate[valid]
    )

    success_valid = np.clip(
        success_valid,
        0.0,
        n_valid,
    )
    failure_valid = (
        n_valid - success_valid
    )

    n[valid] = n_valid
    success[valid] = success_valid
    failure[valid] = failure_valid

    return {
        "valid": valid,
        "n": n,
        "success": success,
        "failure": failure,
    }


def build_pitcher_binomial_summary(train_df):
    """
    Beta population prior 추정용 투수별 binomial summary.

    반드시 현재 fold의 train_df만 입력한다.
    Validation/Test data는 절대 사용하지 않는다.
    """
    required = [
        "pitcher_id",
        TARGET,
    ]
    missing = [
        col
        for col in required
        if col not in train_df.columns
    ]
    if missing:
        raise ValueError(
            "Beta prior fit에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    summary = (
        train_df[
            ["pitcher_id", TARGET]
        ]
        .dropna(
            subset=[
                "pitcher_id",
                TARGET,
            ]
        )
        .groupby(
            "pitcher_id",
            observed=True,
        )[TARGET]
        .agg(
            successes="sum",
            trials="count",
        )
        .reset_index()
    )

    summary["successes"] = pd.to_numeric(
        summary["successes"],
        errors="raise",
    ).astype(float)

    summary["trials"] = pd.to_numeric(
        summary["trials"],
        errors="raise",
    ).astype(float)

    summary = summary.loc[
        summary["trials"]
        >= EB_PRIOR_MIN_PITCHER_TRIALS
    ].copy()

    if len(summary) < 2:
        raise ValueError(
            "Beta prior를 추정할 충분한 투수 수가 없습니다."
        )

    summary["failures"] = (
        summary["trials"]
        - summary["successes"]
    )

    if (
        (summary["successes"] < 0).any()
        or (summary["failures"] < 0).any()
    ):
        raise ValueError(
            "Beta prior용 성공/실패 count가 유효하지 않습니다."
        )

    return summary


def beta_binomial_negative_log_likelihood(
    log_params,
    successes,
    failures,
):
    """
    Beta-Binomial marginal likelihood의 negative log likelihood.

    Binomial coefficient는 alpha/beta와 무관하므로
    최적화 목적함수에서는 생략해도 동일한 MLE를 얻는다.
    """
    log_alpha = float(log_params[0])
    log_beta = float(log_params[1])

    alpha = float(np.exp(log_alpha))
    beta = float(np.exp(log_beta))

    if (
        not np.isfinite(alpha)
        or not np.isfinite(beta)
        or alpha <= 0.0
        or beta <= 0.0
    ):
        return float("inf")

    ll = (
        betaln(
            successes + alpha,
            failures + beta,
        )
        - betaln(
            alpha,
            beta,
        )
    )

    if not np.all(np.isfinite(ll)):
        return float("inf")

    return -float(
        np.sum(ll)
    )


def fit_beta_binomial_prior(train_df):
    """
    현재 expanding fold의 train target만 사용해
    Beta(alpha, beta) population prior를 MLE로 추정한다.

    row-level asof snapshot을 prior fitting 표본으로 사용하지 않고,
    투수별 실제 training target count를 하나의 binomial observation으로 사용한다.
    """
    summary = build_pitcher_binomial_summary(
        train_df
    )

    successes = summary[
        "successes"
    ].to_numpy(dtype=float)

    failures = summary[
        "failures"
    ].to_numpy(dtype=float)

    trials = (
        successes + failures
    )

    total_successes = float(
        np.sum(successes)
    )
    total_trials = float(
        np.sum(trials)
    )

    if total_trials <= 0.0:
        raise ValueError(
            "Beta prior fit용 trial 수가 0입니다."
        )

    empirical_mean = float(
        total_successes / total_trials
    )

    # log(0) 방지
    empirical_mean = float(
        np.clip(
            empirical_mean,
            1e-6,
            1.0 - 1e-6,
        )
    )

    best_result = None

    bounds = [
        (
            EB_LOG_PARAM_LOWER,
            EB_LOG_PARAM_UPPER,
        ),
        (
            EB_LOG_PARAM_LOWER,
            EB_LOG_PARAM_UPPER,
        ),
    ]

    for initial_strength in EB_INITIAL_STRENGTHS:
        alpha0 = (
            empirical_mean
            * float(initial_strength)
        )
        beta0 = (
            (1.0 - empirical_mean)
            * float(initial_strength)
        )

        x0 = np.log(
            [
                max(alpha0, 1e-6),
                max(beta0, 1e-6),
            ]
        )

        result = minimize(
            beta_binomial_negative_log_likelihood,
            x0=x0,
            args=(
                successes,
                failures,
            ),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if (
            result.success
            and np.isfinite(result.fun)
        ):
            if (
                best_result is None
                or result.fun < best_result.fun
            ):
                best_result = result

    if best_result is None:
        raise RuntimeError(
            "Beta-Binomial prior MLE 최적화에 실패했습니다."
        )

    alpha = float(
        np.exp(best_result.x[0])
    )
    beta = float(
        np.exp(best_result.x[1])
    )

    prior_strength = float(
        alpha + beta
    )
    prior_mean = float(
        alpha / prior_strength
    )

    prior_variance = float(
        alpha
        * beta
        / (
            prior_strength ** 2
            * (prior_strength + 1.0)
        )
    )
    prior_std = float(
        np.sqrt(prior_variance)
    )

    return {
        "alpha": alpha,
        "beta": beta,
        "prior_mean": prior_mean,
        "prior_strength": prior_strength,
        "prior_std": prior_std,
        "n_pitchers": int(len(summary)),
        "n_trials": int(total_trials),
        "n_successes": int(total_successes),
        "min_pitcher_trials": int(
            EB_PRIOR_MIN_PITCHER_TRIALS
        ),
        "optimizer": "L-BFGS-B",
        "optimizer_success": bool(
            best_result.success
        ),
        "optimizer_message": str(
            best_result.message
        ),
        "negative_log_likelihood": float(
            best_result.fun
        ),
    }


def apply_reliability_features(
    df,
    reliability_state,
):
    """
    Frozen Beta prior와 현재 row 자체의 공식 asof_*만 이용해
    posterior mean/std를 생성한다.

    다른 validation/test row의 값, 빈도, 순서, 분포를 사용하지 않는다.
    """
    df = df.copy()

    alpha = float(
        reliability_state["alpha"]
    )
    beta = float(
        reliability_state["beta"]
    )

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(
            "Reliability prior alpha/beta가 유효하지 않습니다."
        )

    counts = reconstruct_pitcher_success_counts(
        df
    )

    valid = counts["valid"]
    success = counts["success"]
    failure = counts["failure"]

    prior_total = float(
        alpha + beta
    )
    prior_mean = float(
        alpha / prior_total
    )
    prior_var = float(
        alpha
        * beta
        / (
            prior_total ** 2
            * (prior_total + 1.0)
        )
    )
    prior_std = float(
        np.sqrt(prior_var)
    )

    eb_rate = np.full(
        len(df),
        prior_mean,
        dtype=float,
    )
    posterior_std = np.full(
        len(df),
        prior_std,
        dtype=float,
    )

    if np.any(valid):
        posterior_alpha = (
            alpha + success[valid]
        )
        posterior_beta = (
            beta + failure[valid]
        )
        posterior_total = (
            posterior_alpha
            + posterior_beta
        )

        eb_rate[valid] = (
            posterior_alpha
            / posterior_total
        )

        posterior_var = (
            posterior_alpha
            * posterior_beta
            / (
                posterior_total ** 2
                * (posterior_total + 1.0)
            )
        )

        posterior_var = np.maximum(
            posterior_var,
            0.0,
        )

        posterior_std[valid] = np.sqrt(
            posterior_var
        )

    if not np.all(
        np.isfinite(eb_rate)
    ):
        raise ValueError(
            "EB posterior mean에 비정상 값이 있습니다."
        )

    if not np.all(
        np.isfinite(posterior_std)
    ):
        raise ValueError(
            "EB posterior std에 비정상 값이 있습니다."
        )

    if (
        np.any(eb_rate <= 0.0)
        or np.any(eb_rate >= 1.0)
        or np.any(posterior_std < 0.0)
    ):
        raise ValueError(
            "EB posterior Feature 범위가 유효하지 않습니다."
        )

    df[
        "rf_pitcher_success_eb_rate"
    ] = eb_rate

    df[
        "rf_pitcher_success_posterior_std"
    ] = posterior_std

    return df


def summarize_reliability_features(
    df_before,
    df_after,
):
    """
    shrinkage mechanism sanity check.
    모델 선택에는 사용하지 않고 metadata 진단용으로만 기록한다.
    """
    n = pd.to_numeric(
        df_before["asof_pitcher_n"],
        errors="coerce",
    ).to_numpy(dtype=float)

    raw_rate = pd.to_numeric(
        df_before["asof_pitcher_success_rate"],
        errors="coerce",
    ).to_numpy(dtype=float)

    eb_rate = df_after[
        "rf_pitcher_success_eb_rate"
    ].to_numpy(dtype=float)

    posterior_std = df_after[
        "rf_pitcher_success_posterior_std"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(n)
        & np.isfinite(raw_rate)
        & np.isfinite(eb_rate)
        & np.isfinite(posterior_std)
        & (n >= 0.0)
        & (raw_rate >= 0.0)
        & (raw_rate <= 1.0)
    )

    bins = [
        (-np.inf, 10.0, "n<=10"),
        (10.0, 50.0, "10<n<=50"),
        (50.0, 200.0, "50<n<=200"),
        (200.0, 1000.0, "200<n<=1000"),
        (1000.0, np.inf, "n>1000"),
    ]

    by_count = []

    for lower, upper, label in bins:
        mask = (
            valid
            & (n > lower)
            & (n <= upper)
        )

        if not np.any(mask):
            continue

        by_count.append({
            "bin": label,
            "n_rows": int(np.sum(mask)),
            "mean_abs_shrinkage": float(
                np.mean(
                    np.abs(
                        raw_rate[mask]
                        - eb_rate[mask]
                    )
                )
            ),
            "mean_posterior_std": float(
                np.mean(
                    posterior_std[mask]
                )
            ),
        })

    return {
        "n_valid_rows": int(
            np.sum(valid)
        ),
        "n_fallback_rows": int(
            len(df_before)
            - np.sum(valid)
        ),
        "mean_abs_shrinkage": (
            float(
                np.mean(
                    np.abs(
                        raw_rate[valid]
                        - eb_rate[valid]
                    )
                )
            )
            if np.any(valid)
            else float("nan")
        ),
        "mean_posterior_std": (
            float(
                np.mean(
                    posterior_std[valid]
                )
            )
            if np.any(valid)
            else float("nan")
        ),
        "by_count": by_count,
    }


# =========================================================
# Metric
# =========================================================
def brier_metric(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0 * (1.0 - brier / baseline_brier)
        )
        bss = max(0.0, bss_raw)

    return {
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": bss,
    }


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return float("nan")
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def residual_covariance(y_true, cat_pred, xgb_pred):
    y_true = np.asarray(y_true, dtype=float)
    e_cat = y_true - np.asarray(cat_pred, dtype=float)
    e_xgb = y_true - np.asarray(xgb_pred, dtype=float)

    if len(y_true) < 2:
        return float("nan")

    return float(np.cov(e_cat, e_xgb, ddof=1)[0, 1])


def optimal_cat_weight(y_true, cat_pred, xgb_pred):
    """
    Brier loss를 최소화하는 2-model blend의 CatBoost weight.
    0~1 범위로 clip한다.
    """
    y_true = np.asarray(y_true, dtype=float)
    cat_pred = np.asarray(cat_pred, dtype=float)
    xgb_pred = np.asarray(xgb_pred, dtype=float)

    d = cat_pred - xgb_pred
    denom = float(np.dot(d, d))

    if denom <= 1e-15:
        return 0.5

    weight = float(np.dot(d, y_true - xgb_pred) / denom)
    return float(np.clip(weight, 0.0, 1.0))


def blend_predictions(cat_pred, xgb_pred, cat_weight):
    return (
        float(cat_weight) * np.asarray(cat_pred, dtype=float)
        + (1.0 - float(cat_weight))
        * np.asarray(xgb_pred, dtype=float)
    )


# =========================================================
# Temporal Feature
# =========================================================
def add_temporal_features(df):
    """
    현재 행의 공식 asof_* 값끼리만 연산한다.

    다른 validation/test row에 대한 groupby / rolling / lag /
    expanding / frequency / aggregate를 사용하지 않는다.
    """
    df = df.copy()

    required = [
        "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    career_s = df["asof_pitcher_success_rate"]
    prev1_s = df["asof_pitcher_prev1_game_success_rate"]
    prev3_s = df["asof_pitcher_prev3_game_success_rate"]
    prev5_s = df["asof_pitcher_prev5_game_success_rate"]

    career_m = df["asof_pitcher_middle_rate"]
    prev1_m = df["asof_pitcher_prev1_game_middle_rate"]
    prev3_m = df["asof_pitcher_prev3_game_middle_rate"]
    prev5_m = df["asof_pitcher_prev5_game_middle_rate"]

    df["tf_success_prev1_vs_career"] = prev1_s - career_s
    df["tf_success_prev3_vs_career"] = prev3_s - career_s
    df["tf_success_prev5_vs_career"] = prev5_s - career_s
    df["tf_success_prev1_vs_prev3"] = prev1_s - prev3_s
    df["tf_success_prev3_vs_prev5"] = prev3_s - prev5_s

    df["tf_middle_prev1_vs_career"] = prev1_m - career_m
    df["tf_middle_prev3_vs_career"] = prev3_m - career_m
    df["tf_middle_prev5_vs_career"] = prev5_m - career_m
    df["tf_middle_prev1_vs_prev3"] = prev1_m - prev3_m
    df["tf_middle_prev3_vs_prev5"] = prev3_m - prev5_m

    return df


# =========================================================
# CatBoost 전처리 — train_cb.py 적용
# =========================================================
def prepare_cat_features(df, features, cat_cols):
    X = df[features].copy()

    for col in cat_cols:
        if col in X.columns:
            X[col] = (
                X[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return X


# =========================================================
# XGBoost 전처리 — train_xgb.py 적용
# =========================================================
def prepare_xgb_raw_features(
    df,
    features,
    id_like_cols,
):
    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "입력 데이터에 XGBoost Feature가 없습니다: "
            + ", ".join(missing_features)
        )

    X = df[features].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


def create_xgb_preprocessor(
    cat_cols,
    num_cols,
):
    """
    train_xgb.py와 동일:
    - low-cardinality categorical -> OrdinalEncoder
    - unseen/missing category -> -1
    - numeric -> passthrough
    - numeric NaN -> XGBoost가 직접 처리
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# 모델 생성
# =========================================================
def create_cat_validation_model():
    return CatBoostClassifier(
        **CATBOOST_PARAMS
    )


def create_cat_final_model(n_iterations):
    params = {
        **CATBOOST_PARAMS,
        "iterations": int(n_iterations),
    }
    return CatBoostClassifier(**params), params


def create_xgb_validation_model():
    return XGBClassifier(
        n_estimators=XGB_MAX_ITERATIONS,
        eval_metric=brier_metric,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        **XGBOOST_PARAMS,
    )


def create_xgb_final_model(n_estimators):
    return XGBClassifier(
        n_estimators=int(n_estimators),
        **XGBOOST_PARAMS,
    )


# =========================================================
# Fold 평가
# =========================================================
def evaluate_pair(
    y_true,
    cat_pred,
    xgb_pred,
    walk_forward_cat_weight,
):
    y_true = np.asarray(y_true, dtype=float)
    cat_pred = np.asarray(cat_pred, dtype=float)
    xgb_pred = np.asarray(xgb_pred, dtype=float)

    e_cat = y_true - cat_pred
    e_xgb = y_true - xgb_pred

    diagnostic_weight = optimal_cat_weight(
        y_true,
        cat_pred,
        xgb_pred,
    )

    fixed_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        BASE_CAT_WEIGHT,
    )
    walk_forward_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        walk_forward_cat_weight,
    )
    diagnostic_pred = blend_predictions(
        cat_pred,
        xgb_pred,
        diagnostic_weight,
    )

    return {
        "cat_brier": calculate_metrics(
            y_true,
            cat_pred,
        )["brier"],
        "xgb_brier": calculate_metrics(
            y_true,
            xgb_pred,
        )["brier"],
        "prediction_corr": safe_corr(
            cat_pred,
            xgb_pred,
        ),
        "residual_corr": safe_corr(
            e_cat,
            e_xgb,
        ),
        "residual_covariance": residual_covariance(
            y_true,
            cat_pred,
            xgb_pred,
        ),
        "fixed_cat_weight": float(BASE_CAT_WEIGHT),
        "fixed_ensemble_brier": calculate_metrics(
            y_true,
            fixed_pred,
        )["brier"],
        "walk_forward_cat_weight": float(
            walk_forward_cat_weight
        ),
        "walk_forward_ensemble_brier": calculate_metrics(
            y_true,
            walk_forward_pred,
        )["brier"],
        # 같은 fold target으로 찾은 weight라 진단용이다.
        "diagnostic_opt_cat_weight": float(
            diagnostic_weight
        ),
        "diagnostic_opt_ensemble_brier": calculate_metrics(
            y_true,
            diagnostic_pred,
        )["brier"],
        "best_base_brier": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ),
        "fixed_blend_gain": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ) - calculate_metrics(y_true, fixed_pred)["brier"],
        "walk_forward_blend_gain": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ) - calculate_metrics(y_true, walk_forward_pred)["brier"],
        "diagnostic_opt_blend_gain": min(
            calculate_metrics(y_true, cat_pred)["brier"],
            calculate_metrics(y_true, xgb_pred)["brier"],
        ) - calculate_metrics(y_true, diagnostic_pred)["brier"],
    }


# =========================================================
# Final iteration 선택
# =========================================================
def select_final_iterations(
    fold_results,
    key,
):
    if FINAL_ITERATION_STRATEGY != "latest_fold":
        raise ValueError(
            "지원하지 않는 FINAL_ITERATION_STRATEGY: "
            f"{FINAL_ITERATION_STRATEGY}"
        )

    latest_fold = max(
        fold_results,
        key=lambda x: x["validation_season"],
    )

    return int(latest_fold[key])


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
    # 파일 검증
    # -----------------------------------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.csv를 찾을 수 없습니다: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv를 찾을 수 없습니다: {TEST_PATH}"
        )

    # -----------------------------------------------------
    # test schema만 이용해 공식 Raw Feature 결정
    # test의 값/분포/다른 row 정보는 사용하지 않는다.
    # -----------------------------------------------------
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise ValueError(
            f"test.csv에 ID 컬럼 '{ID}'가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID
    ]

    # -----------------------------------------------------
    # Feature Tag Mode 해석
    # -----------------------------------------------------
    # Temporal Feature는 현재 행의 공식 asof_* 값만으로 생성되므로
    # 평가 데이터의 다른 row에 의존하지 않는다.
    feature_tags = validate_feature_tags(
        FEATURE_TAGS
    )
    resolved_feature_config = (
        resolve_feature_config(
            feature_tags
        )
    )

    cat_use_temporal = (
        resolved_feature_config[
            "cat_use_temporal"
        ]
    )
    xgb_use_temporal = (
        resolved_feature_config[
            "xgb_use_temporal"
        ]
    )
    xgb_use_raw_player_ids = (
        resolved_feature_config[
            "xgb_use_raw_player_ids"
        ]
    )
    xgb_use_reliability = (
        resolved_feature_config[
            "xgb_use_reliability"
        ]
    )

    cat_features = raw_features.copy()
    if cat_use_temporal:
        cat_features += TEMPORAL_FEATURES

    if xgb_use_raw_player_ids:
        xgb_dropped_raw_id_cols = []
        xgb_features = raw_features.copy()
    else:
        xgb_dropped_raw_id_cols = (
            XGB_DROP_RAW_ID_COLS.copy()
        )
        xgb_features = [
            col
            for col in raw_features
            if col not in xgb_dropped_raw_id_cols
        ]

    if xgb_use_temporal:
        xgb_features += TEMPORAL_FEATURES

    if xgb_use_reliability:
        xgb_features += RELIABILITY_FEATURES

    cat_cols = [
        col
        for col in CAT_COLS
        if col in cat_features
    ]

    xgb_cat_cols = [
        col
        for col in XGB_CAT_COLS
        if col in xgb_features
    ]

    xgb_id_like_cols = [
        col
        for col in XGB_ID_LIKE_COLS
        if col in xgb_features
    ]

    xgb_num_cols = [
        col
        for col in xgb_features
        if col not in xgb_cat_cols
    ]

    # -----------------------------------------------------
    # Train load
    # -----------------------------------------------------
    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *raw_features,
            TARGET,
        ],
    )

    required_cols = {
        ID,
        TARGET,
        "season",
    }
    missing_cols = required_cols - set(train.columns)

    if missing_cols:
        raise ValueError(
            "train.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_cols))
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"타깃 컬럼 '{TARGET}'에 결측값이 있습니다."
        )

    if cat_use_temporal or xgb_use_temporal:
        train = add_temporal_features(train)

    # -----------------------------------------------------
    # 공식 asof 통계 denominator audit
    # -----------------------------------------------------
    # train.csv만 사용한다. test 값/분포는 보지 않는다.
    asof_denominator_audit = (
        audit_asof_pitcher_success_denominator(
            train
        )
    )
    print_asof_denominator_audit(
        asof_denominator_audit
    )

    if xgb_use_reliability:
        validate_exact_count_reconstruction(
            asof_denominator_audit
        )

    print("=" * 72)
    print("Dual Model Feature-Diverse Ensemble")
    print("=" * 72)
    print("Experiment              :", EXPERIMENT_NAME)
    print("XGBoost version         :", xgb.__version__)
    print("Train shape             :", train.shape)
    print("Raw Feature 수          :", len(raw_features))
    print("CatBoost Feature 수     :", len(cat_features))
    print("CatBoost categorical 수 :", len(cat_cols))
    print("XGBoost Feature 수      :", len(xgb_features))
    print("XGBoost categorical 수  :", len(xgb_cat_cols))
    print(
        "Feature Tags            :",
        sorted(feature_tags),
    )
    print(
        "Resolved Cat Temporal   :",
        cat_use_temporal,
    )
    print(
        "Resolved XGB Temporal   :",
        xgb_use_temporal,
    )
    print(
        "Resolved XGB Raw IDs    :",
        xgb_use_raw_player_ids,
    )
    print(
        "Resolved XGB Reliability:",
        xgb_use_reliability,
    )
    print(
        "Cat Temporal Feature 수 :",
        len(TEMPORAL_FEATURES) if cat_use_temporal else 0,
    )
    print(
        "XGB Temporal Feature 수 :",
        len(TEMPORAL_FEATURES) if xgb_use_temporal else 0,
    )
    print(
        "XGB Reliability Feature 수:",
        len(RELIABILITY_FEATURES) if xgb_use_reliability else 0,
    )
    print("XGB 제거 Raw ID         :", xgb_dropped_raw_id_cols)
    print(
        "Season                  :",
        train["season"].min(),
        "~",
        train["season"].max(),
    )
    print(
        "Target mean             :",
        f"{train[TARGET].mean():.6f}",
    )
    print("=" * 72)

    # -----------------------------------------------------
    # 디렉터리 생성
    # -----------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )
    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    with open(
        VALIDATION_DIR / "asof_denominator_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asof_denominator_audit,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Expanding Window
    # -----------------------------------------------------
    fold_results = []
    oof_frames = []

    previous_y = []
    previous_cat_pred = []
    previous_xgb_pred = []

    for fold_config in EXPANDING_FOLDS:

        fold = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        validation_season = (
            fold_config["validation_season"]
        )

        train_mask = train["season"].isin(
            train_seasons
        )
        val_mask = (
            train["season"]
            == validation_season
        )

        if not train_mask.any():
            raise ValueError(
                f"Fold {fold} 학습 데이터가 없습니다."
            )

        if not val_mask.any():
            raise ValueError(
                f"Fold {fold} Validation 데이터가 없습니다."
            )

        train_df = train.loc[
            train_mask
        ].copy()

        val_df = train.loc[
            val_mask
        ].copy()

        y_train = train_df[
            TARGET
        ].copy()

        y_val = val_df[
            TARGET
        ].copy()

        val_row_id = val_df[
            ID
        ].copy()

        # =================================================
        # XGBoost Reliability — Fold-safe Beta prior
        # =================================================
        reliability_state = None
        reliability_train_summary = None
        reliability_val_summary = None

        if xgb_use_reliability:
            reliability_state = (
                fit_beta_binomial_prior(
                    train_df
                )
            )

            train_df_before_reliability = (
                train_df.copy()
            )
            val_df_before_reliability = (
                val_df.copy()
            )

            train_df = (
                apply_reliability_features(
                    train_df,
                    reliability_state,
                )
            )
            val_df = (
                apply_reliability_features(
                    val_df,
                    reliability_state,
                )
            )

            reliability_train_summary = (
                summarize_reliability_features(
                    train_df_before_reliability,
                    train_df,
                )
            )
            reliability_val_summary = (
                summarize_reliability_features(
                    val_df_before_reliability,
                    val_df,
                )
            )

        print()
        print("=" * 72)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print(
            f"Train={len(train_df):,}, "
            f"Valid={len(val_df):,}"
        )
        print("=" * 72)

        # =================================================
        # CatBoost
        # =================================================
        X_cat_train = prepare_cat_features(
            train_df,
            cat_features,
            cat_cols,
        )

        X_cat_val = prepare_cat_features(
            val_df,
            cat_features,
            cat_cols,
        )

        cat_model = create_cat_validation_model()

        cat_start = time.time()

        cat_model.fit(
            X_cat_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(
                X_cat_val,
                y_val,
            ),
            early_stopping_rounds=(
                CAT_EARLY_STOPPING_ROUNDS
            ),
            use_best_model=True,
        )

        cat_elapsed = (
            time.time() - cat_start
        )

        cat_pred = (
            cat_model.predict_proba(
                X_cat_val
            )[:, 1]
        )

        cat_best_iteration = (
            cat_model.get_best_iteration()
        )

        if (
            cat_best_iteration is None
            or cat_best_iteration < 0
        ):
            cat_final_iterations = (
                CATBOOST_PARAMS["iterations"]
            )
        else:
            cat_final_iterations = (
                int(cat_best_iteration) + 1
            )

        # =================================================
        # XGBoost
        # =================================================
        X_xgb_train_raw = prepare_xgb_raw_features(
            train_df,
            xgb_features,
            xgb_id_like_cols,
        )

        X_xgb_val_raw = prepare_xgb_raw_features(
            val_df,
            xgb_features,
            xgb_id_like_cols,
        )

        # Fold마다 train 구간에만 fit되는 독립 Encoder.
        xgb_preprocessor = create_xgb_preprocessor(
            xgb_cat_cols,
            xgb_num_cols,
        )

        X_xgb_train = (
            xgb_preprocessor
            .fit_transform(
                X_xgb_train_raw
            )
        )

        X_xgb_val = (
            xgb_preprocessor
            .transform(
                X_xgb_val_raw
            )
        )

        xgb_model = create_xgb_validation_model()

        xgb_start = time.time()

        xgb_model.fit(
            X_xgb_train,
            y_train,
            eval_set=[
                (
                    X_xgb_val,
                    y_val,
                ),
            ],
            verbose=100,
        )

        xgb_elapsed = (
            time.time() - xgb_start
        )

        xgb_pred = (
            xgb_model
            .predict_proba(
                X_xgb_val
            )[:, 1]
        )

        xgb_best_iteration = getattr(
            xgb_model,
            "best_iteration",
            None,
        )

        if (
            xgb_best_iteration is None
            or xgb_best_iteration < 0
        ):
            xgb_best_iteration = (
                XGB_MAX_ITERATIONS - 1
            )

        xgb_final_iterations = (
            int(xgb_best_iteration) + 1
        )

        # =================================================
        # Ensemble / Diversity
        # =================================================
        if previous_y:
            previous_y_all = np.concatenate(
                previous_y
            )
            previous_cat_all = np.concatenate(
                previous_cat_pred
            )
            previous_xgb_all = np.concatenate(
                previous_xgb_pred
            )

            walk_forward_cat_weight = (
                optimal_cat_weight(
                    previous_y_all,
                    previous_cat_all,
                    previous_xgb_all,
                )
            )
        else:
            walk_forward_cat_weight = (
                BASE_CAT_WEIGHT
            )

        pair_metrics = evaluate_pair(
            y_val,
            cat_pred,
            xgb_pred,
            walk_forward_cat_weight,
        )

        cat_metrics = calculate_metrics(
            y_val,
            cat_pred,
        )
        xgb_metrics = calculate_metrics(
            y_val,
            xgb_pred,
        )

        fixed_pred = blend_predictions(
            cat_pred,
            xgb_pred,
            BASE_CAT_WEIGHT,
        )

        walk_forward_pred = blend_predictions(
            cat_pred,
            xgb_pred,
            walk_forward_cat_weight,
        )

        fold_result = {
            "fold": int(fold),
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "validation_season": int(
                validation_season
            ),
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "cat_brier": float(
                cat_metrics["brier"]
            ),
            "cat_bss": float(
                cat_metrics["bss"]
            ),
            "cat_best_iteration": int(
                cat_best_iteration
            ),
            "cat_final_iterations": int(
                cat_final_iterations
            ),
            "cat_elapsed_seconds": float(
                cat_elapsed
            ),
            "xgb_brier": float(
                xgb_metrics["brier"]
            ),
            "xgb_bss": float(
                xgb_metrics["bss"]
            ),
            "xgb_best_iteration": int(
                xgb_best_iteration
            ),
            "xgb_final_iterations": int(
                xgb_final_iterations
            ),
            "xgb_elapsed_seconds": float(
                xgb_elapsed
            ),
            "reliability_state": (
                reliability_state
            ),
            "reliability_train_summary": (
                reliability_train_summary
            ),
            "reliability_validation_summary": (
                reliability_val_summary
            ),
            **pair_metrics,
        }

        fold_results.append(
            fold_result
        )

        pred_df = pd.DataFrame({
            ID: val_row_id.to_numpy(),
            "target": y_val.to_numpy(),
            "validation_season": (
                validation_season
            ),
            "cat_prediction": cat_pred,
            "xgb_prediction": xgb_pred,
            "fixed_ensemble_prediction": (
                fixed_pred
            ),
            "walk_forward_ensemble_prediction": (
                walk_forward_pred
            ),
        })

        oof_frames.append(
            pred_df
        )

        fold_dir = (
            VALIDATION_DIR
            / f"fold_{validation_season}"
        )
        fold_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        pred_df.to_csv(
            fold_dir / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            fold_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                fold_result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Cat Brier              : "
            f"{pair_metrics['cat_brier']:.8f}"
        )
        print(
            f"XGB Brier              : "
            f"{pair_metrics['xgb_brier']:.8f}"
        )
        print(
            f"Prediction Corr        : "
            f"{pair_metrics['prediction_corr']:.8f}"
        )
        print(
            f"Residual Corr          : "
            f"{pair_metrics['residual_corr']:.8f}"
        )
        print(
            f"Residual Covariance    : "
            f"{pair_metrics['residual_covariance']:.10f}"
        )
        print(
            f"Fixed Ensemble Brier   : "
            f"{pair_metrics['fixed_ensemble_brier']:.8f}"
        )
        print(
            f"Walk-forward Brier     : "
            f"{pair_metrics['walk_forward_ensemble_brier']:.8f} "
            f"(Cat={walk_forward_cat_weight:.6f})"
        )
        print(
            f"Diagnostic Opt Brier   : "
            f"{pair_metrics['diagnostic_opt_ensemble_brier']:.8f} "
            f"(Cat={pair_metrics['diagnostic_opt_cat_weight']:.6f})"
        )
        print(
            f"Walk-forward Blend Gain: "
            f"{pair_metrics['walk_forward_blend_gain']:+.8f}"
        )
        print(
            f"Cat best/final iter    : "
            f"{cat_best_iteration} / {cat_final_iterations}"
        )
        print(
            f"XGB best/final iter    : "
            f"{xgb_best_iteration} / {xgb_final_iterations}"
        )
        if reliability_state is not None:
            print(
                "EB prior               : "
                f"alpha={reliability_state['alpha']:.6f}, "
                f"beta={reliability_state['beta']:.6f}, "
                f"mean={reliability_state['prior_mean']:.6f}, "
                f"strength={reliability_state['prior_strength']:.3f}"
            )

        previous_y.append(
            y_val.to_numpy(dtype=float)
        )
        previous_cat_pred.append(
            np.asarray(cat_pred, dtype=float)
        )
        previous_xgb_pred.append(
            np.asarray(xgb_pred, dtype=float)
        )

    # -----------------------------------------------------
    # Pooled OOF
    # -----------------------------------------------------
    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    y_oof = oof[
        "target"
    ].to_numpy(dtype=float)

    cat_oof = oof[
        "cat_prediction"
    ].to_numpy(dtype=float)

    xgb_oof = oof[
        "xgb_prediction"
    ].to_numpy(dtype=float)

    pooled_cat_weight = optimal_cat_weight(
        y_oof,
        cat_oof,
        xgb_oof,
    )

    if FINAL_WEIGHT_MODE == "pooled_oof":
        final_cat_weight = pooled_cat_weight
    elif FINAL_WEIGHT_MODE == "fixed":
        final_cat_weight = BASE_CAT_WEIGHT
    else:
        raise ValueError(
            "지원하지 않는 FINAL_WEIGHT_MODE: "
            f"{FINAL_WEIGHT_MODE}"
        )

    final_xgb_weight = (
        1.0 - final_cat_weight
    )

    pooled_pred = blend_predictions(
        cat_oof,
        xgb_oof,
        pooled_cat_weight,
    )

    pooled_fixed_pred = blend_predictions(
        cat_oof,
        xgb_oof,
        BASE_CAT_WEIGHT,
    )

    pooled_metrics = {
        "n_oof": int(len(oof)),
        "cat_brier": float(
            calculate_metrics(
                y_oof,
                cat_oof,
            )["brier"]
        ),
        "xgb_brier": float(
            calculate_metrics(
                y_oof,
                xgb_oof,
            )["brier"]
        ),
        "prediction_corr": float(
            safe_corr(
                cat_oof,
                xgb_oof,
            )
        ),
        "residual_corr": float(
            safe_corr(
                y_oof - cat_oof,
                y_oof - xgb_oof,
            )
        ),
        "residual_covariance": float(
            residual_covariance(
                y_oof,
                cat_oof,
                xgb_oof,
            )
        ),
        "fixed_cat_weight": float(
            BASE_CAT_WEIGHT
        ),
        "fixed_ensemble_brier": float(
            calculate_metrics(
                y_oof,
                pooled_fixed_pred,
            )["brier"]
        ),
        "pooled_opt_cat_weight": float(
            pooled_cat_weight
        ),
        "pooled_opt_xgb_weight": float(
            1.0 - pooled_cat_weight
        ),
        # 전체 OOF를 이용해 weight를 찾았으므로
        # 모델 선택용 지표이며 독립 test score가 아니다.
        "pooled_opt_ensemble_brier": float(
            calculate_metrics(
                y_oof,
                pooled_pred,
            )["brier"]
        ),
        "walk_forward_ensemble_brier": float(
            calculate_metrics(
                y_oof,
                oof[
                    "walk_forward_ensemble_prediction"
                ].to_numpy(dtype=float),
            )["brier"]
        ),
        "best_base_brier": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
        ),
        "fixed_blend_gain": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
            - calculate_metrics(y_oof, pooled_fixed_pred)["brier"]
        ),
        "pooled_opt_blend_gain": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
            - calculate_metrics(y_oof, pooled_pred)["brier"]
        ),
        "walk_forward_blend_gain": float(
            min(
                calculate_metrics(y_oof, cat_oof)["brier"],
                calculate_metrics(y_oof, xgb_oof)["brier"],
            )
            - calculate_metrics(
                y_oof,
                oof[
                    "walk_forward_ensemble_prediction"
                ].to_numpy(dtype=float),
            )["brier"]
        ),
    }

    diagnostic_weights = np.asarray(
        [
            fold[
                "diagnostic_opt_cat_weight"
            ]
            for fold in fold_results
        ],
        dtype=float,
    )

    pooled_metrics[
        "diagnostic_weight_mean"
    ] = float(
        np.mean(diagnostic_weights)
    )

    pooled_metrics[
        "diagnostic_weight_std"
    ] = float(
        np.std(
            diagnostic_weights,
            ddof=1,
        )
    )

    oof["pooled_opt_ensemble_prediction"] = (
        pooled_pred
    )

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        fold_results
    ).to_csv(
        VALIDATION_DIR / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Final iterations
    # -----------------------------------------------------
    cat_final_iterations = (
        select_final_iterations(
            fold_results,
            "cat_final_iterations",
        )
    )

    xgb_final_iterations = (
        select_final_iterations(
            fold_results,
            "xgb_final_iterations",
        )
    )

    print()
    print("=" * 72)
    print("Pooled OOF")
    print("=" * 72)
    print(
        "Cat Brier              :",
        f"{pooled_metrics['cat_brier']:.8f}",
    )
    print(
        "XGB Brier              :",
        f"{pooled_metrics['xgb_brier']:.8f}",
    )
    print(
        "Prediction Corr        :",
        f"{pooled_metrics['prediction_corr']:.8f}",
    )
    print(
        "Residual Corr          :",
        f"{pooled_metrics['residual_corr']:.8f}",
    )
    print(
        "Residual Covariance    :",
        f"{pooled_metrics['residual_covariance']:.10f}",
    )
    print(
        "Fixed Ensemble Brier   :",
        f"{pooled_metrics['fixed_ensemble_brier']:.8f}",
    )
    print(
        "Walk-forward Brier     :",
        f"{pooled_metrics['walk_forward_ensemble_brier']:.8f}",
    )
    print(
        "Pooled Opt Brier       :",
        f"{pooled_metrics['pooled_opt_ensemble_brier']:.8f}",
    )
    print(
        "Pooled Opt Blend Gain  :",
        f"{pooled_metrics['pooled_opt_blend_gain']:+.8f}",
    )
    print(
        "Final weight           :",
        f"Cat={final_cat_weight:.8f}, "
        f"XGB={final_xgb_weight:.8f}",
    )
    print(
        "Diagnostic weight std  :",
        f"{pooled_metrics['diagnostic_weight_std']:.8f}",
    )
    print(
        "Final Cat iterations   :",
        cat_final_iterations,
    )
    print(
        "Final XGB iterations   :",
        xgb_final_iterations,
    )
    print("=" * 72)

    # -----------------------------------------------------
    # 전체 데이터 Final CatBoost
    # -----------------------------------------------------
    X_cat_full = prepare_cat_features(
        train,
        cat_features,
        cat_cols,
    )

    y_full = train[
        TARGET
    ].copy()

    (
        final_cat_model,
        final_cat_params,
    ) = create_cat_final_model(
        cat_final_iterations
    )

    cat_final_start = time.time()

    final_cat_model.fit(
        X_cat_full,
        y_full,
        cat_features=cat_cols,
    )

    cat_final_elapsed = (
        time.time() - cat_final_start
    )

    # -----------------------------------------------------
    # 전체 데이터 Final XGBoost
    # -----------------------------------------------------
    final_reliability_state = None
    final_reliability_summary = None
    xgb_final_train = train

    if xgb_use_reliability:
        final_reliability_state = (
            fit_beta_binomial_prior(
                train
            )
        )

        xgb_final_train = (
            apply_reliability_features(
                train,
                final_reliability_state,
            )
        )

        final_reliability_summary = (
            summarize_reliability_features(
                train,
                xgb_final_train,
            )
        )

    X_xgb_full_raw = prepare_xgb_raw_features(
        xgb_final_train,
        xgb_features,
        xgb_id_like_cols,
    )

    final_xgb_preprocessor = (
        create_xgb_preprocessor(
            xgb_cat_cols,
            xgb_num_cols,
        )
    )

    X_xgb_full = (
        final_xgb_preprocessor
        .fit_transform(
            X_xgb_full_raw
        )
    )

    final_xgb_model = create_xgb_final_model(
        xgb_final_iterations
    )

    xgb_final_start = time.time()

    final_xgb_model.fit(
        X_xgb_full,
        y_full,
        verbose=100,
    )

    xgb_final_elapsed = (
        time.time() - xgb_final_start
    )

    # -----------------------------------------------------
    # Ensemble Artifact
    # -----------------------------------------------------
    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": (
            "CatBoostClassifier + "
            "XGBClassifier tag-ablation ensemble with exact Beta-Binomial EB"
        ),
        "catboost_model": final_cat_model,
        "xgboost_model": final_xgb_model,
        "xgboost_preprocessor": (
            final_xgb_preprocessor
        ),
        "cat_features": cat_features,
        "cat_categorical_features": cat_cols,
        "xgb_features": xgb_features,
        "xgb_categorical_features": (
            xgb_cat_cols
        ),
        "xgb_numeric_features": (
            xgb_num_cols
        ),
        "xgb_id_like_features": (
            xgb_id_like_cols
        ),
        "xgb_dropped_raw_id_features": (
            xgb_dropped_raw_id_cols
        ),
        "feature_tags": sorted(
            feature_tags
        ),
        "resolved_feature_config": {
            "cat_use_temporal": bool(
                cat_use_temporal
            ),
            "xgb_use_temporal": bool(
                xgb_use_temporal
            ),
            "xgb_use_raw_player_ids": bool(
                xgb_use_raw_player_ids
            ),
            "xgb_use_reliability": bool(
                xgb_use_reliability
            ),
        },
        "cat_temporal_features": (
            TEMPORAL_FEATURES if cat_use_temporal else []
        ),
        "xgb_temporal_features": (
            TEMPORAL_FEATURES if xgb_use_temporal else []
        ),
        "temporal_features": (
            TEMPORAL_FEATURES
        ),
        "xgb_reliability_features": (
            RELIABILITY_FEATURES
            if xgb_use_reliability
            else []
        ),
        "reliability_state": (
            final_reliability_state
        ),
        "reliability_summary": (
            final_reliability_summary
        ),
        "cat_weight": float(
            final_cat_weight
        ),
        "xgb_weight": float(
            final_xgb_weight
        ),
        "weight_mode": FINAL_WEIGHT_MODE,
        "asof_denominator_audit": (
            asof_denominator_audit
        ),
        "catboost_params": (
            final_cat_params
        ),
        "xgboost_params": {
            **XGBOOST_PARAMS,
            "n_estimators": int(
                xgb_final_iterations
            ),
        },
        "cat_final_iterations": int(
            cat_final_iterations
        ),
        "xgb_final_iterations": int(
            xgb_final_iterations
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "training_seasons": sorted(
            [
                int(x)
                for x in train[
                    "season"
                ].unique()
            ]
        ),
        "n_train": int(len(train)),
        "raw_features": raw_features,
        "cat_features": cat_features,
        "cat_categorical_features": cat_cols,
        "xgb_features": xgb_features,
        "xgb_categorical_features": (
            xgb_cat_cols
        ),
        "xgb_numeric_features": (
            xgb_num_cols
        ),
        "xgb_id_like_features": (
            xgb_id_like_cols
        ),
        "xgb_dropped_raw_id_features": (
            xgb_dropped_raw_id_cols
        ),
        "feature_tags": sorted(
            feature_tags
        ),
        "resolved_feature_config": {
            "cat_use_temporal": bool(
                cat_use_temporal
            ),
            "xgb_use_temporal": bool(
                xgb_use_temporal
            ),
            "xgb_use_raw_player_ids": bool(
                xgb_use_raw_player_ids
            ),
            "xgb_use_reliability": bool(
                xgb_use_reliability
            ),
        },
        "cat_temporal_features": (
            TEMPORAL_FEATURES if cat_use_temporal else []
        ),
        "xgb_temporal_features": (
            TEMPORAL_FEATURES if xgb_use_temporal else []
        ),
        "temporal_features": (
            TEMPORAL_FEATURES
        ),
        "xgb_reliability_features": (
            RELIABILITY_FEATURES
            if xgb_use_reliability
            else []
        ),
        "final_reliability_state": (
            final_reliability_state
        ),
        "final_reliability_summary": (
            final_reliability_summary
        ),
        "catboost_validation_params": (
            CATBOOST_PARAMS
        ),
        "xgboost_validation_params": (
            XGBOOST_PARAMS
        ),
        "cat_early_stopping_rounds": (
            CAT_EARLY_STOPPING_ROUNDS
        ),
        "xgb_early_stopping_rounds": (
            XGB_EARLY_STOPPING_ROUNDS
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "final_weight_mode": (
            FINAL_WEIGHT_MODE
        ),
        "asof_denominator_audit": (
            asof_denominator_audit
        ),
        "final_cat_weight": float(
            final_cat_weight
        ),
        "final_xgb_weight": float(
            final_xgb_weight
        ),
        "cat_final_iterations": int(
            cat_final_iterations
        ),
        "xgb_final_iterations": int(
            xgb_final_iterations
        ),
        "cat_final_elapsed_seconds": float(
            cat_final_elapsed
        ),
        "xgb_final_elapsed_seconds": float(
            xgb_final_elapsed
        ),
        "folds": fold_results,
        "pooled_oof": pooled_metrics,
        "model_path": str(MODEL_PATH),
    }

    with open(
        VALIDATION_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    feature_manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "feature_tags": sorted(
            feature_tags
        ),
        "resolved_feature_config": {
            "cat_use_temporal": bool(
                cat_use_temporal
            ),
            "xgb_use_temporal": bool(
                xgb_use_temporal
            ),
            "xgb_use_raw_player_ids": bool(
                xgb_use_raw_player_ids
            ),
            "xgb_use_reliability": bool(
                xgb_use_reliability
            ),
        },
        "raw_features": raw_features,
        "cat_features": cat_features,
        "cat_categorical_features": cat_cols,
        "xgb_features": xgb_features,
        "xgb_categorical_features": xgb_cat_cols,
        "xgb_numeric_features": xgb_num_cols,
        "xgb_dropped_raw_id_features": xgb_dropped_raw_id_cols,
        "cat_temporal_features": (
            TEMPORAL_FEATURES if cat_use_temporal else []
        ),
        "xgb_temporal_features": (
            TEMPORAL_FEATURES if xgb_use_temporal else []
        ),
        "xgb_reliability_features": (
            RELIABILITY_FEATURES
            if xgb_use_reliability
            else []
        ),
        "reliability_definition": (
            "Exact count-based Beta-Binomial EB: "
            "S=round(asof_pitcher_n*asof_pitcher_success_rate), "
            "F=N-S; alpha/beta are fitted only on each fold's "
            "training targets grouped by pitcher_id; validation/test "
            "rows use the frozen prior and their own asof_* values only."
        ),
        "asof_denominator_audit": (
            asof_denominator_audit
        ),
        "independent_test_row_rule": (
            "Temporal and Reliability row transforms use only values "
            "from the current row's official asof_* columns. Reliability "
            "prior parameters are fitted from official training targets "
            "and frozen before validation/test inference. No cross-test "
            "aggregation, rolling, lag, frequency, rank, distribution, "
            "or test-row state update is used."
        ),
    }

    with open(
        VALIDATION_DIR / "feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    source_path = Path(
        __file__
    ).resolve()

    shutil.copy2(
        source_path,
        MODEL_DIR / "train.py",
    )

    print()
    print("=" * 72)
    print("Final Ensemble 저장 완료")
    print("- MODEL_PATH :", MODEL_PATH)
    print("- Validation :", VALIDATION_DIR)
    print(
        "- Weight     : "
        f"Cat={final_cat_weight:.8f}, "
        f"XGB={final_xgb_weight:.8f}"
    )
    print("=" * 72)

    print()
    print("[추론 시 필수]")
    print(
        "CatBoostRegressor는 predict()를 사용하고, "
        "최종 제출 확률은 "
        "np.clip(pred_raw, 0.0, 1.0)을 적용합니다."
    )


if __name__ == "__main__":
    main()
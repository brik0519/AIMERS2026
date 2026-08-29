# train.py: CatBoost Regressor cb_reg_v008 — 2023 Corrected-Total Early Stopping
#
# cb_reg_v003의 HB-EB Feature Engineering + Expanding Window Validation 구조를 그대로 유지하면서
# 2023 validation fold에서만 R raw + F train-only corrected 전체 Brier로 stage를 선택한다.
#
# 핵심 비교:
#   Control   : cb_v003     = HB-EB + CatBoostClassifier + Logloss
#   Treatment : cb_reg_v003 = HB-EB + CatBoostRegressor  + RMSE
#
# 평가 데이터 독립 추론 규칙:
# - test.csv는 Feature schema 확인을 위해 header만 읽는다.
# - Feature/Prior는 공식 train.csv만으로 생성한다.
# - 다른 test 행 또는 test 전체 분포를 사용하지 않는다.
#
# Validation 저장:
# - fold별 pred.csv / validation.npz / metadata.json
# - pooled OOF pred.csv / validation.npz / metadata.json
# - Regressor raw prediction과 clipped prediction을 모두 저장
# - [0, 1] 범위 이탈 진단 저장

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v008"

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
# HB-EB 설정
# =========================================================

HBEB_CONFIGS = [
    {
        "group_cols": ["pitcher_hand"],
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "new_col": "hbeb_pitcher_success_rate",
    },
    {
        "group_cols": ["batter_hand"],
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "new_col": "hbeb_batter_success_rate",
    },
]


# =========================================================
# CatBoost 설정
# =========================================================

# cb_v003와 capacity 관련 설정을 동일하게 유지하고
# objective만 Logloss -> RMSE로 변경한다.
MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80
FINAL_ITERATION_STRATEGY = "latest_fold"

# =========================================================
# Experiment C: Corrected-Total Early Stopping
# =========================================================
#
# 2023 validation fold에서만:
#   - CatBoost를 최대 iterations까지 학습
#   - 각 tree stage마다 마지막 학습 시즌(2022)의 F prediction으로
#     train-only logit correction delta를 계산
#   - validation R prediction은 그대로 유지
#   - validation F prediction에는 해당 frozen delta 적용
#   - corrected R+F 전체 Brier를 기준으로 best stage 선택
#
# 중요:
# - delta 계산에 validation/test prediction 평균을 사용하지 않는다.
# - delta 계산에 validation target을 사용하지 않는다.
# - correction 적용은 각 행의 game_type + raw prediction + frozen delta만 사용한다.
# - 2021/2022/2024 fold는 cb_reg_v003과 동일한 full-validation ES를 유지한다.
# - FINAL_ITERATION_STRATEGY="latest_fold"이므로 최종 tree 수는 2024 fold가 결정한다.

CORRECTED_TOTAL_ES_SEASON = 2023
CORRECTED_TOTAL_F_GAME_TYPE = "F"
CORRECTED_TOTAL_F_ANCHOR = 0.50
CORRECTED_TOTAL_CORRECTION_METHOD = "logit_shift"
CORRECTED_TOTAL_EVAL_PERIOD = 1
CORRECTED_TOTAL_PATIENCE = EARLY_STOPPING_ROUNDS
CORRECTION_EPS = 1e-6


# =========================================================
# HB-EB
# =========================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    train subset만으로 HB-EB prior를 추정한다.

    Validation/Test 정보를 prior 추정에 사용하지 않는다.
    """

    g_mu = float(
        df_train[rate_col].mean()
    )

    g_var = float(
        df_train[rate_col].var()
    )

    max_bernoulli_var = (
        g_mu * (1.0 - g_mu)
    )

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = (
            max_bernoulli_var * 0.5
        )

    g_m = (
        max_bernoulli_var / g_var
    ) - 1.0

    g_alpha = g_mu * g_m
    g_beta = (
        1.0 - g_mu
    ) * g_m

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            ["mean", "var", "count"]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0 - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(stats["var"])
        & (stats["var"] > 0.0)
        & (stats["var"] < group_max_var)
        & (stats["count"] >= 30)
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

    if valid_mask.any():
        mu = stats.loc[
            valid_mask,
            "mean",
        ]

        var = stats.loc[
            valid_mask,
            "var",
        ]

        m = (
            mu * (1.0 - mu) / var
        ) - 1.0

        stats.loc[
            valid_mask,
            "alpha",
        ] = mu * m

        stats.loc[
            valid_mask,
            "beta",
        ] = (
            1.0 - mu
        ) * m

    prior_df = stats[
        group_cols
        + [
            "alpha",
            "beta",
        ]
    ].copy()

    return (
        prior_df,
        (
            float(g_alpha),
            float(g_beta),
        ),
    )


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """
    train-only prior를 lookup하여 각 행을 독립적으로 변환한다.
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
        .fillna(global_prior[0])
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup_df["beta"]
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

    denominator = (
        n + alpha + beta
    )

    eb_rate = (
        success_count + alpha
    ) / denominator

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def apply_hbeb_configs(
    df_train,
    df_apply,
    base_features,
):
    """
    df_train만으로 prior를 추정하고 df_apply에 HB-EB feature를 적용한다.

    반환:
    - 변환된 df_apply
    - 최종 feature 목록
    - 추론용 prior artifact
    """

    transformed = (
        df_apply.copy()
    )

    current_features = (
        list(base_features)
    )

    priors = {}

    for config in HBEB_CONFIGS:
        prior_df, global_prior = (
            compute_hbeb_priors(
                df_train=df_train,
                group_cols=config[
                    "group_cols"
                ],
                rate_col=config[
                    "rate_col"
                ],
            )
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=config[
                "group_cols"
            ],
            n_col=config["n_col"],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            config["rate_col"]
            in current_features
        ):
            current_features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in current_features
        ):
            current_features.append(
                config["new_col"]
            )

        priors[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config["group_cols"]
            ),
            "n_col": config["n_col"],
            "rate_col": (
                config["rate_col"]
            ),
        }

    return (
        transformed,
        current_features,
        priors,
    )


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    """
    cb_v003와 동일한 범주형 전처리.
    """

    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


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

    positive_rate = float(
        np.mean(y_true)
    )

    brier = float(
        np.mean(
            (
                y_pred - y_true
            ) ** 2
        )
    )

    baseline_brier = float(
        positive_rate
        * (
            1.0 - positive_rate
        )
    )

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0
            * (
                1.0
                - brier
                / baseline_brier
            )
        )

        bss = max(
            0.0,
            bss_raw,
        )

    return {
        "positive_rate": (
            positive_rate
        ),
        "brier": brier,
        "baseline_brier": (
            baseline_brier
        ),
        "bss_raw": bss_raw,
        "bss": bss,
    }


def _safe_logit(p, eps=CORRECTION_EPS):
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        eps,
        1.0 - eps,
    )
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    pos = z >= 0

    out[pos] = (
        1.0
        / (
            1.0
            + np.exp(-z[pos])
        )
    )

    exp_z = np.exp(z[~pos])

    out[~pos] = (
        exp_z
        / (
            1.0
            + exp_z
        )
    )

    return out


def solve_logit_delta_to_target_mean(
    reference_pred,
    target_mean,
    max_iter=100,
):
    """
    Train-only reference prediction의 보정 후 평균이 target_mean이 되도록
    logit intercept delta를 계산한다.

    Validation/Test prediction vector를 reference_pred로 넣어서는 안 된다.
    """

    reference_pred = np.clip(
        np.asarray(
            reference_pred,
            dtype=np.float64,
        ),
        CORRECTION_EPS,
        1.0 - CORRECTION_EPS,
    )

    if reference_pred.size == 0:
        raise ValueError(
            "F correction reference prediction이 비어 있습니다."
        )

    if not (0.0 < target_mean < 1.0):
        raise ValueError(
            "target_mean은 (0,1) 범위여야 합니다."
        )

    z = _safe_logit(reference_pred)

    lo = -10.0
    hi = 10.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        mean_corrected = float(
            np.mean(
                _sigmoid(z + mid)
            )
        )

        if mean_corrected < target_mean:
            lo = mid
        else:
            hi = mid

    delta = 0.5 * (lo + hi)

    return {
        "delta": float(delta),
        "target_mean": float(target_mean),
        "reference_raw_mean": float(
            np.mean(reference_pred)
        ),
        "reference_corrected_mean": float(
            np.mean(
                _sigmoid(z + delta)
            )
        ),
    }


def apply_f_logit_correction(
    pred,
    game_type,
    delta,
):
    """
    Row-wise deterministic correction.

    R(non-F): raw prediction 유지
    F      : sigmoid(logit(raw) + frozen delta)
    """

    pred = np.asarray(
        pred,
        dtype=np.float64,
    ).copy()

    game_type = np.asarray(
        game_type,
    ).astype(str)

    if len(pred) != len(game_type):
        raise ValueError(
            "prediction과 game_type 길이가 일치하지 않습니다."
        )

    f_mask = (
        game_type
        == CORRECTED_TOTAL_F_GAME_TYPE
    )

    if np.any(f_mask):
        pred[f_mask] = _sigmoid(
            _safe_logit(
                pred[f_mask]
            )
            + float(delta)
        )

    return np.clip(
        pred,
        0.0,
        1.0,
    )


def select_corrected_total_stage(
    estimator,
    X_val,
    y_val,
    val_game_type,
    X_reference_f,
    anchor,
    patience,
):
    """
    2023 Experiment C stage selector.

    각 candidate tree stage마다:
      1. 같은 stage의 train-reference F prediction으로 delta를 fit
      2. validation F에 해당 delta를 적용
      3. corrected R+F 전체 Brier 계산
      4. patience 기준 best stage 선택

    delta fitting에는 validation prediction 평균/target을 사용하지 않는다.
    """

    y_val_np = np.asarray(
        y_val,
        dtype=np.float64,
    )

    game_type_np = np.asarray(
        val_game_type,
    ).astype(str)

    r_mask = (
        game_type_np
        != CORRECTED_TOTAL_F_GAME_TYPE
    )
    f_mask = (
        game_type_np
        == CORRECTED_TOTAL_F_GAME_TYPE
    )

    if not np.any(r_mask):
        raise ValueError(
            "Validation R row가 없습니다."
        )

    if not np.any(f_mask):
        raise ValueError(
            "Validation F row가 없습니다."
        )

    if len(X_reference_f) == 0:
        raise ValueError(
            "Train-reference F row가 없습니다."
        )

    val_generator = estimator.staged_predict(
        X_val,
        eval_period=CORRECTED_TOTAL_EVAL_PERIOD,
    )

    ref_generator = estimator.staged_predict(
        X_reference_f,
        eval_period=CORRECTED_TOTAL_EVAL_PERIOD,
    )

    trajectory = []
    best = None
    no_improve = 0

    for stage_number, (
        val_pred_stage,
        ref_pred_stage,
    ) in enumerate(
        zip(
            val_generator,
            ref_generator,
        ),
        start=1,
    ):
        val_pred_stage = np.asarray(
            val_pred_stage,
            dtype=np.float64,
        )

        ref_pred_stage = np.asarray(
            ref_pred_stage,
            dtype=np.float64,
        )

        delta_info = (
            solve_logit_delta_to_target_mean(
                reference_pred=ref_pred_stage,
                target_mean=anchor,
            )
        )

        raw_clip = np.clip(
            val_pred_stage,
            0.0,
            1.0,
        )

        corrected = (
            apply_f_logit_correction(
                pred=val_pred_stage,
                game_type=game_type_np,
                delta=delta_info["delta"],
            )
        )

        raw_total_brier = float(
            np.mean(
                (y_val_np - raw_clip) ** 2
            )
        )

        corrected_total_brier = float(
            np.mean(
                (y_val_np - corrected) ** 2
            )
        )

        r_brier = float(
            np.mean(
                (
                    y_val_np[r_mask]
                    - raw_clip[r_mask]
                ) ** 2
            )
        )

        f_raw_brier = float(
            np.mean(
                (
                    y_val_np[f_mask]
                    - raw_clip[f_mask]
                ) ** 2
            )
        )

        f_corrected_brier = float(
            np.mean(
                (
                    y_val_np[f_mask]
                    - corrected[f_mask]
                ) ** 2
            )
        )

        row = {
            "stage_number": int(stage_number),
            "best_iteration": int(
                stage_number - 1
            ),
            "raw_total_brier": raw_total_brier,
            "corrected_total_brier": (
                corrected_total_brier
            ),
            "r_brier": r_brier,
            "f_raw_brier": f_raw_brier,
            "f_corrected_brier": (
                f_corrected_brier
            ),
            "delta_f": float(
                delta_info["delta"]
            ),
            "reference_raw_mean": float(
                delta_info[
                    "reference_raw_mean"
                ]
            ),
            "reference_corrected_mean": float(
                delta_info[
                    "reference_corrected_mean"
                ]
            ),
            "anchor": float(anchor),
            "validation_raw_mean": float(
                np.mean(raw_clip)
            ),
            "validation_corrected_mean": float(
                np.mean(corrected)
            ),
            "validation_r_raw_mean": float(
                np.mean(
                    raw_clip[r_mask]
                )
            ),
            "validation_f_raw_mean": float(
                np.mean(
                    raw_clip[f_mask]
                )
            ),
            "validation_f_corrected_mean": float(
                np.mean(
                    corrected[f_mask]
                )
            ),
        }

        trajectory.append(row)

        improved = (
            best is None
            or corrected_total_brier
            < (
                best[
                    "corrected_total_brier"
                ]
                - 1e-15
            )
        )

        if improved:
            best = dict(row)
            no_improve = 0
        else:
            no_improve += (
                CORRECTED_TOTAL_EVAL_PERIOD
            )

        if (
            stage_number == 1
            or stage_number % 25 == 0
            or improved
        ):
            print(
                "Corrected-Total stage "
                f"{stage_number:>3d} | "
                f"raw={raw_total_brier:.8f} | "
                f"corr={corrected_total_brier:.8f} | "
                f"R={r_brier:.8f} | "
                f"Fraw={f_raw_brier:.8f} | "
                f"Fcorr={f_corrected_brier:.8f} | "
                f"delta={delta_info['delta']:+.6f}"
            )

        if no_improve >= patience:
            print(
                "Corrected-Total staged evaluation "
                f"stopped after {patience} "
                "trees without improvement."
            )
            break

    if best is None:
        raise RuntimeError(
            "Corrected-total best stage 선택 실패."
        )

    return best, pd.DataFrame(trajectory)


def calculate_domain_metrics(
    game_type,
    y_true,
    y_pred,
):
    """
    Validation 전체 예측을 R(non-F) / F로 분리해 진단한다.

    - R은 game_type != "F"로 정의한다.
    - 이 함수는 평가/진단용이며 학습 Feature를 생성하지 않는다.
    """

    game_type = pd.Series(
        game_type
    ).astype(str).to_numpy()

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if not (
        len(game_type)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "Domain metric 입력 길이가 일치하지 않습니다."
        )

    masks = {
        "R": (
            game_type
            != CORRECTED_TOTAL_F_GAME_TYPE
        ),
        "F": (
            game_type
            == CORRECTED_TOTAL_F_GAME_TYPE
        ),
    }

    result = {}

    for domain, mask in masks.items():
        n = int(
            np.sum(mask)
        )

        if n == 0:
            result[domain] = {
                "n": 0,
                "share": 0.0,
                "metrics": None,
                "prediction_mean": None,
                "prediction_std": None,
            }
            continue

        metrics = calculate_metrics(
            y_true[mask],
            y_pred[mask],
        )

        result[domain] = {
            "n": n,
            "share": float(
                n / len(y_true)
            ),
            "metrics": metrics,
            "prediction_mean": float(
                np.mean(
                    y_pred[mask]
                )
            ),
            "prediction_std": float(
                np.std(
                    y_pred[mask]
                )
            ),
        }

    return result


def calculate_range_diagnostics(
    y_pred_raw,
):
    y_pred_raw = np.asarray(
        y_pred_raw,
        dtype=np.float64,
    )

    below_zero = (
        y_pred_raw < 0.0
    )

    above_one = (
        y_pred_raw > 1.0
    )

    outside = (
        below_zero
        | above_one
    )

    return {
        "n_prediction_below_0": int(
            np.sum(below_zero)
        ),
        "ratio_prediction_below_0": float(
            np.mean(below_zero)
        ),
        "n_prediction_above_1": int(
            np.sum(above_one)
        ),
        "ratio_prediction_above_1": float(
            np.mean(above_one)
        ),
        "n_prediction_outside_0_1": int(
            np.sum(outside)
        ),
        "ratio_prediction_outside_0_1": float(
            np.mean(outside)
        ),
    }


# =========================================================
# Validation 저장
# =========================================================

def save_fold_validation(
    fold,
    train_seasons,
    validation_season,
    estimator,
    row_ids,
    game_type,
    y_true,
    y_pred_raw,
    y_pred_clip,
    raw_metrics,
    clipped_metrics,
    elapsed_seconds,
    best_iteration,
    final_iterations,
    n_train,
    early_stopping_policy,
    n_early_stopping_eval,
    y_pred_model_raw=None,
    y_pred_model_clip=None,
    correction_metadata=None,
):
    fold_dir = (
        VALIDATION_DIR
        / f"fold_{validation_season}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    row_ids = np.asarray(
        row_ids
    )

    game_type = np.asarray(
        game_type
    )

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred_raw = np.asarray(
        y_pred_raw,
        dtype=np.float64,
    )

    y_pred_clip = np.asarray(
        y_pred_clip,
        dtype=np.float64,
    )

    if y_pred_model_raw is None:
        y_pred_model_raw = (
            y_pred_raw.copy()
        )
    else:
        y_pred_model_raw = np.asarray(
            y_pred_model_raw,
            dtype=np.float64,
        )

    if y_pred_model_clip is None:
        y_pred_model_clip = np.clip(
            y_pred_model_raw,
            0.0,
            1.0,
        )
    else:
        y_pred_model_clip = np.asarray(
            y_pred_model_clip,
            dtype=np.float64,
        )

    if correction_metadata is None:
        correction_metadata = {
            "enabled": False,
        }

    if not (
        len(row_ids)
        == len(game_type)
        == len(y_true)
        == len(y_pred_raw)
        == len(y_pred_clip)
        == len(y_pred_model_raw)
        == len(y_pred_model_clip)
    ):
        raise ValueError(
            "Validation 저장 배열 길이가 일치하지 않습니다."
        )

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "game_type": game_type,
        "target": y_true,
        "prediction_model_raw": (
            y_pred_model_raw
        ),
        "prediction_model_clip": (
            y_pred_model_clip
        ),
        "prediction_raw": (
            y_pred_raw
        ),
        "prediction_clip": (
            y_pred_clip
        ),
        # 기존 OOF/ensemble 분석 코드와 호환
        "prediction": (
            y_pred_clip
        ),
        "validation_season": (
            validation_season
        ),
    })

    pred_df.to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        fold_dir
        / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction_model_raw=(
            y_pred_model_raw
        ),
        prediction_model_clip=(
            y_pred_model_clip
        ),
        prediction_raw=(
            y_pred_raw
        ),
        prediction_clip=(
            y_pred_clip
        ),
        prediction=(
            y_pred_clip
        ),
        validation_season=(
            np.full(
                len(y_true),
                validation_season,
            )
        ),
    )

    range_diag = (
        calculate_range_diagnostics(
            y_pred_raw
        )
    )

    domain_metrics = (
        calculate_domain_metrics(
            game_type=game_type,
            y_true=y_true,
            y_pred=y_pred_clip,
        )
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "fold": int(fold),
        "train_seasons": [
            int(s)
            for s in train_seasons
        ],
        "validation_season": int(
            validation_season
        ),
        "n_train": int(n_train),
        "n_validation": int(
            len(y_true)
        ),
        "early_stopping_policy": (
            early_stopping_policy
        ),
        "n_early_stopping_eval": int(
            n_early_stopping_eval
        ),
        "early_stopping_eval_share": float(
            n_early_stopping_eval
            / len(y_true)
        ),
        "correction_metadata": (
            correction_metadata
        ),
        "domain_metrics": (
            domain_metrics
        ),
        "model_type": (
            type(estimator).__name__
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "positive_rate": float(
            clipped_metrics[
                "positive_rate"
            ]
        ),

        "raw_brier": float(
            raw_metrics["brier"]
        ),
        "raw_bss_raw": float(
            raw_metrics["bss_raw"]
        ),
        "raw_bss": float(
            raw_metrics["bss"]
        ),

        "clipped_brier": float(
            clipped_metrics[
                "brier"
            ]
        ),
        "clipped_bss_raw": float(
            clipped_metrics[
                "bss_raw"
            ]
        ),
        "clipped_bss": float(
            clipped_metrics[
                "bss"
            ]
        ),

        # 기존 분석 코드 호환용
        "brier": float(
            clipped_metrics[
                "brier"
            ]
        ),
        "baseline_brier": float(
            clipped_metrics[
                "baseline_brier"
            ]
        ),
        "bss_raw": float(
            clipped_metrics[
                "bss_raw"
            ]
        ),
        "bss": float(
            clipped_metrics[
                "bss"
            ]
        ),

        "prediction_raw_mean": float(
            np.mean(y_pred_raw)
        ),
        "prediction_raw_std": float(
            np.std(y_pred_raw)
        ),
        "prediction_raw_min": float(
            np.min(y_pred_raw)
        ),
        "prediction_raw_max": float(
            np.max(y_pred_raw)
        ),

        "prediction_clip_mean": float(
            np.mean(y_pred_clip)
        ),
        "prediction_clip_std": float(
            np.std(y_pred_clip)
        ),
        "prediction_clip_min": float(
            np.min(y_pred_clip)
        ),
        "prediction_clip_max": float(
            np.max(y_pred_clip)
        ),

        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "best_iteration": int(
            best_iteration
        ),
        "final_iterations": int(
            final_iterations
        ),
        **range_diag,
    }

    with open(
        fold_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return (
        pred_df,
        metadata,
    )


def save_validation_summary(
    fold_results,
    oof,
):
    oof.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR
        / "oof_validation.npz",
        row_id=oof[
            "row_id"
        ].to_numpy(),
        target=oof[
            "target"
        ].to_numpy(),
        prediction_raw=oof[
            "prediction_raw"
        ].to_numpy(),
        prediction_clip=oof[
            "prediction_clip"
        ].to_numpy(),
        prediction=oof[
            "prediction"
        ].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    raw_oof_metrics = (
        calculate_metrics(
            oof["target"],
            oof[
                "prediction_raw"
            ],
        )
    )

    clipped_oof_metrics = (
        calculate_metrics(
            oof["target"],
            oof[
                "prediction_clip"
            ],
        )
    )

    fold_briers = np.asarray(
        [
            x["brier"]
            for x in fold_results
        ],
        dtype=np.float64,
    )

    fold_bss = np.asarray(
        [
            x["bss"]
            for x in fold_results
        ],
        dtype=np.float64,
    )

    fold_sizes = np.asarray(
        [
            x["n_validation"]
            for x in fold_results
        ],
        dtype=np.float64,
    )

    recent_briers = [
        x["brier"]
        for x in fold_results
        if x["validation_season"]
        in (2023, 2024)
    ]

    range_diag = (
        calculate_range_diagnostics(
            oof[
                "prediction_raw"
            ]
        )
    )

    summary = {
        "n_oof": int(len(oof)),
        "mean_fold_brier": float(
            np.mean(fold_briers)
        ),
        "std_fold_brier": float(
            np.std(fold_briers)
        ),
        "weighted_fold_brier": float(
            np.average(
                fold_briers,
                weights=fold_sizes,
            )
        ),
        "best_fold_brier": float(
            np.min(fold_briers)
        ),
        "worst_fold_brier": float(
            np.max(fold_briers)
        ),
        "fold_brier_range": float(
            np.max(fold_briers)
            - np.min(fold_briers)
        ),
        "recent_2023_2024_mean_brier": float(
            np.mean(
                recent_briers
            )
        ),
        "mean_fold_bss": float(
            np.mean(fold_bss)
        ),
        "weighted_fold_bss": float(
            np.average(
                fold_bss,
                weights=fold_sizes,
            )
        ),

        "pooled_oof_raw_brier": float(
            raw_oof_metrics[
                "brier"
            ]
        ),
        "pooled_oof_raw_bss_raw": float(
            raw_oof_metrics[
                "bss_raw"
            ]
        ),
        "pooled_oof_raw_bss": float(
            raw_oof_metrics[
                "bss"
            ]
        ),

        "pooled_oof_clipped_brier": float(
            clipped_oof_metrics[
                "brier"
            ]
        ),
        "pooled_oof_clipped_baseline_brier": float(
            clipped_oof_metrics[
                "baseline_brier"
            ]
        ),
        "pooled_oof_clipped_bss_raw": float(
            clipped_oof_metrics[
                "bss_raw"
            ]
        ),
        "pooled_oof_clipped_bss": float(
            clipped_oof_metrics[
                "bss"
            ]
        ),

        # 기존 분석 코드 호환
        "pooled_oof_brier": float(
            clipped_oof_metrics[
                "brier"
            ]
        ),
        "pooled_oof_baseline_brier": float(
            clipped_oof_metrics[
                "baseline_brier"
            ]
        ),
        "pooled_oof_bss_raw": float(
            clipped_oof_metrics[
                "bss_raw"
            ]
        ),
        "pooled_oof_bss": float(
            clipped_oof_metrics[
                "bss"
            ]
        ),
        **range_diag,
    }

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "validation_method": (
            "expanding_window"
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "prediction_for_evaluation": (
            "clipped"
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "early_stopping_policy": {
            "default": "full_validation",
            "special_validation_season": int(
                CORRECTED_TOTAL_ES_SEASON
            ),
            "special_policy": "corrected_total_R_raw_F_train_only_shift",
            "corrected_game_type": (
                CORRECTED_TOTAL_F_GAME_TYPE
            ),
            "training_rows": "R_and_F_all",
            "fold_evaluation_rows": "R_and_F_all",
        },
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "folds": fold_results,
        "summary": summary,
    }

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return summary


# =========================================================
# Expanding Window Validation
# =========================================================

def run_expanding_validation(
    train,
    base_features,
    cat_cols,
):
    fold_results = []
    oof_frames = []
    final_features = None

    for fold_config in (
        EXPANDING_FOLDS
    ):
        fold = (
            fold_config["fold"]
        )

        train_seasons = (
            fold_config[
                "train_seasons"
            ]
        )

        validation_season = (
            fold_config[
                "validation_season"
            ]
        )

        train_mask = (
            train["season"].isin(
                train_seasons
            )
        )

        val_mask = (
            train["season"]
            == validation_season
        )

        df_train = (
            train.loc[
                train_mask
            ]
            .copy()
        )

        df_val = (
            train.loc[
                val_mask
            ]
            .copy()
        )

        # ---------------------------------------------
        # Fold train-only HB-EB prior
        # ---------------------------------------------

        current_features = (
            list(base_features)
        )

        for config in (
            HBEB_CONFIGS
        ):
            (
                prior_df,
                global_prior,
            ) = compute_hbeb_priors(
                df_train=df_train,
                group_cols=config[
                    "group_cols"
                ],
                rate_col=config[
                    "rate_col"
                ],
            )

            df_train[
                config["new_col"]
            ] = apply_hbeb(
                df=df_train,
                group_cols=config[
                    "group_cols"
                ],
                n_col=config["n_col"],
                rate_col=config[
                    "rate_col"
                ],
                prior_df=prior_df,
                global_prior=(
                    global_prior
                ),
            )

            df_val[
                config["new_col"]
            ] = apply_hbeb(
                df=df_val,
                group_cols=config[
                    "group_cols"
                ],
                n_col=config["n_col"],
                rate_col=config[
                    "rate_col"
                ],
                prior_df=prior_df,
                global_prior=(
                    global_prior
                ),
            )

            if (
                config["rate_col"]
                in current_features
            ):
                current_features.remove(
                    config["rate_col"]
                )

            if (
                config["new_col"]
                not in current_features
            ):
                current_features.append(
                    config["new_col"]
                )

        final_features = (
            current_features
        )

        X_train = (
            prepare_features(
                df_train[
                    current_features
                ],
                cat_cols,
            )
        )

        y_train = (
            df_train[TARGET]
            .astype(np.float64)
        )

        X_val = (
            prepare_features(
                df_val[
                    current_features
                ],
                cat_cols,
            )
        )

        y_val = (
            df_val[TARGET]
            .astype(np.float64)
        )

        val_row_id = (
            df_val[ID]
        )

        val_game_type = (
            df_val["game_type"]
            .astype(str)
        )

        # ---------------------------------------------
        # Experiment C stage-selection 정책
        # ---------------------------------------------

        if (
            validation_season
            == CORRECTED_TOTAL_ES_SEASON
        ):
            early_stopping_policy = (
                "corrected_total_R_raw_F_train_only_shift"
            )

            reference_season = int(
                max(train_seasons)
            )

            reference_mask = (
                (
                    df_train["season"]
                    == reference_season
                )
                & (
                    df_train["game_type"]
                    .astype(str)
                    == CORRECTED_TOTAL_F_GAME_TYPE
                )
            )

            X_reference_f = (
                X_train.loc[
                    reference_mask
                ]
            )

            if len(X_reference_f) == 0:
                raise ValueError(
                    "마지막 학습 시즌 F reference가 없습니다."
                )

            n_early_stopping_eval = (
                len(y_val)
            )

        else:
            early_stopping_policy = (
                "full_validation"
            )
            reference_season = None
            X_reference_f = None
            n_early_stopping_eval = (
                len(y_val)
            )

        print()
        print("=" * 72)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print("=" * 72)
        print(
            "학습 데이터:",
            X_train.shape,
        )
        print(
            "검증 데이터:",
            X_val.shape,
        )
        print(
            "Training objective    : RMSE"
        )
        print(
            "Evaluation prediction : raw + clipped"
        )
        print(
            "Early stopping policy : "
            f"{early_stopping_policy}"
        )
        print(
            "Early stopping rows   : "
            f"{n_early_stopping_eval:,} "
            f"/ {len(y_val):,}"
        )

        estimator = (
            CatBoostRegressor(
                **MODEL_PARAMS
            )
        )

        start_time = time.time()

        correction_metadata = {
            "enabled": False,
        }
        trajectory_df = None

        if (
            validation_season
            == CORRECTED_TOTAL_ES_SEASON
        ):
            # -------------------------------------------------
            # Experiment C:
            # 2023에서는 내부 raw-overall early stopping을 끈다.
            # -------------------------------------------------
            estimator.fit(
                X_train,
                y_train,
                cat_features=cat_cols,
                verbose=(
                    MODEL_PARAMS[
                        "verbose"
                    ]
                ),
            )

            (
                best_stage,
                trajectory_df,
            ) = select_corrected_total_stage(
                estimator=estimator,
                X_val=X_val,
                y_val=y_val,
                val_game_type=(
                    val_game_type
                ),
                X_reference_f=(
                    X_reference_f
                ),
                anchor=(
                    CORRECTED_TOTAL_F_ANCHOR
                ),
                patience=(
                    CORRECTED_TOTAL_PATIENCE
                ),
            )

            best_iteration = int(
                best_stage[
                    "best_iteration"
                ]
            )

            final_iterations = (
                best_iteration + 1
            )

            estimator.shrink(
                ntree_end=(
                    final_iterations
                )
            )

            val_pred_model_raw = (
                estimator.predict(
                    X_val
                )
            )

            val_pred_model_clip = (
                np.clip(
                    val_pred_model_raw,
                    0.0,
                    1.0,
                )
            )

            selected_delta = float(
                best_stage[
                    "delta_f"
                ]
            )

            # Experiment C의 최종 fold prediction은
            # R raw + F corrected prediction이다.
            val_pred_raw = (
                apply_f_logit_correction(
                    pred=(
                        val_pred_model_raw
                    ),
                    game_type=(
                        val_game_type
                    ),
                    delta=(
                        selected_delta
                    ),
                )
            )

            val_pred_clip = (
                np.clip(
                    val_pred_raw,
                    0.0,
                    1.0,
                )
            )

            correction_metadata = {
                "enabled": True,
                "method": (
                    CORRECTED_TOTAL_CORRECTION_METHOD
                ),
                "domain": (
                    CORRECTED_TOTAL_F_GAME_TYPE
                ),
                "anchor": float(
                    CORRECTED_TOTAL_F_ANCHOR
                ),
                "reference_season": int(
                    reference_season
                ),
                "reference_scope": (
                    "same_model_last_training_season_F"
                ),
                "delta": float(
                    selected_delta
                ),
                "reference_raw_mean": float(
                    best_stage[
                        "reference_raw_mean"
                    ]
                ),
                "reference_corrected_mean": float(
                    best_stage[
                        "reference_corrected_mean"
                    ]
                ),
                "selected_stage_number": int(
                    best_stage[
                        "stage_number"
                    ]
                ),
                "selected_corrected_total_brier": float(
                    best_stage[
                        "corrected_total_brier"
                    ]
                ),
                "selected_raw_total_brier": float(
                    best_stage[
                        "raw_total_brier"
                    ]
                ),
                "selected_r_brier": float(
                    best_stage[
                        "r_brier"
                    ]
                ),
                "selected_f_raw_brier": float(
                    best_stage[
                        "f_raw_brier"
                    ]
                ),
                "selected_f_corrected_brier": float(
                    best_stage[
                        "f_corrected_brier"
                    ]
                ),
                "validation_f_raw_mean": float(
                    best_stage[
                        "validation_f_raw_mean"
                    ]
                ),
                "validation_f_corrected_mean": float(
                    best_stage[
                        "validation_f_corrected_mean"
                    ]
                ),
                "uses_validation_prediction_mean_for_delta": False,
                "uses_validation_target_for_delta": False,
                "uses_test_distribution": False,
                "row_independent_application": True,
            }

        else:
            # -------------------------------------------------
            # cb_reg_v003 original behavior
            # -------------------------------------------------
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

            best_iteration = (
                estimator
                .get_best_iteration()
            )

            if (
                best_iteration
                is not None
                and best_iteration >= 0
            ):
                final_iterations = (
                    best_iteration + 1
                )
            else:
                final_iterations = (
                    MODEL_PARAMS[
                        "iterations"
                    ]
                )

            val_pred_model_raw = (
                estimator.predict(
                    X_val
                )
            )

            val_pred_model_clip = (
                np.clip(
                    val_pred_model_raw,
                    0.0,
                    1.0,
                )
            )

            val_pred_raw = (
                val_pred_model_raw.copy()
            )

            val_pred_clip = (
                val_pred_model_clip.copy()
            )

        elapsed = (
            time.time()
            - start_time
        )

        raw_metrics = (
            calculate_metrics(
                y_val,
                val_pred_raw,
            )
        )

        clipped_metrics = (
            calculate_metrics(
                y_val,
                val_pred_clip,
            )
        )

        model_raw_metrics = (
            calculate_metrics(
                y_val,
                val_pred_model_clip,
            )
        )

        if trajectory_df is not None:
            fold_dir = (
                VALIDATION_DIR
                / f"fold_{validation_season}"
            )

            # save_fold_validation이 exist_ok=False로 디렉터리를 만들므로
            # trajectory는 임시 위치에 저장했다가 fold 저장 후 기록한다.
            trajectory_path = (
                VALIDATION_DIR
                / (
                    f"_trajectory_"
                    f"{validation_season}.csv"
                )
            )

            trajectory_df.to_csv(
                trajectory_path,
                index=False,
            )
        else:
            trajectory_path = None

        pred_df, metadata = (
            save_fold_validation(
                fold=fold,
                train_seasons=(
                    train_seasons
                ),
                validation_season=(
                    validation_season
                ),
                estimator=estimator,
                row_ids=val_row_id,
                game_type=(
                    val_game_type
                ),
                y_true=y_val,
                y_pred_raw=(
                    val_pred_raw
                ),
                y_pred_clip=(
                    val_pred_clip
                ),
                raw_metrics=(
                    raw_metrics
                ),
                clipped_metrics=(
                    clipped_metrics
                ),
                elapsed_seconds=(
                    elapsed
                ),
                best_iteration=(
                    best_iteration
                ),
                final_iterations=(
                    final_iterations
                ),
                n_train=len(y_train),
                early_stopping_policy=(
                    early_stopping_policy
                ),
                n_early_stopping_eval=(
                    n_early_stopping_eval
                ),
                y_pred_model_raw=(
                    val_pred_model_raw
                ),
                y_pred_model_clip=(
                    val_pred_model_clip
                ),
                correction_metadata=(
                    correction_metadata
                ),
            )
        )

        if trajectory_path is not None:
            target_trajectory_path = (
                VALIDATION_DIR
                / f"fold_{validation_season}"
                / "corrected_total_trajectory.csv"
            )

            trajectory_path.replace(
                target_trajectory_path
            )

        fold_results.append(
            metadata
        )

        oof_frames.append(
            pred_df
        )

        range_diag = (
            calculate_range_diagnostics(
                val_pred_raw
            )
        )

        print()
        print("[Raw prediction]")
        print(
            f"Brier Score      : "
            f"{raw_metrics['brier']:.8f}"
        )
        print(
            f"BSS              : "
            f"{raw_metrics['bss']:.2f}"
        )

        print()
        print("[Clipped prediction]")
        print(
            f"Brier Score      : "
            f"{clipped_metrics['brier']:.8f}"
        )
        print(
            f"BSS              : "
            f"{clipped_metrics['bss']:.2f}"
        )

        domain_metrics = (
            metadata[
                "domain_metrics"
            ]
        )

        print()
        print("[Model raw vs evaluated]")
        print(
            "Model raw Brier   : "
            f"{model_raw_metrics['brier']:.8f}"
        )
        print(
            "Evaluated Brier   : "
            f"{clipped_metrics['brier']:.8f}"
        )
        print(
            "Correction enabled: "
            f"{correction_metadata.get('enabled', False)}"
        )

        if correction_metadata.get(
            "enabled",
            False,
        ):
            print(
                "F delta          : "
                f"{correction_metadata['delta']:+.8f}"
            )
            print(
                "F anchor         : "
                f"{correction_metadata['anchor']:.6f}"
            )
            print(
                "Reference season : "
                f"{correction_metadata['reference_season']}"
            )

        print()
        print("[Domain diagnostics]")
        for domain in (
            "R",
            "F",
        ):
            d = domain_metrics[
                domain
            ]

            if d[
                "metrics"
            ] is None:
                print(
                    f"{domain}: no rows"
                )
                continue

            print(
                f"{domain} | "
                f"n={d['n']:,} | "
                f"share={d['share']:.2%} | "
                f"Brier={d['metrics']['brier']:.8f} | "
                f"BSS_raw={d['metrics']['bss_raw']:.2f} | "
                f"pred_mean={d['prediction_mean']:.6f} | "
                f"actual={d['metrics']['positive_rate']:.6f}"
            )

        print()
        print("[Range diagnostics]")
        print(
            f"Below 0          : "
            f"{range_diag['n_prediction_below_0']} "
            f"("
            f"{range_diag['ratio_prediction_below_0']:.6%}"
            f")"
        )
        print(
            f"Above 1          : "
            f"{range_diag['n_prediction_above_1']} "
            f"("
            f"{range_diag['ratio_prediction_above_1']:.6%}"
            f")"
        )
        print(
            f"Outside [0,1]    : "
            f"{range_diag['n_prediction_outside_0_1']} "
            f"("
            f"{range_diag['ratio_prediction_outside_0_1']:.6%}"
            f")"
        )

        print()
        print(
            f"Best iteration   : "
            f"{best_iteration}"
        )
        print(
            f"Final iterations : "
            f"{final_iterations}"
        )
        print(
            f"Elapsed          : "
            f"{elapsed:.1f}s"
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    return (
        fold_results,
        oof,
        final_features,
    )


# =========================================================
# Final iteration 선택
# =========================================================

def select_final_iterations(
    fold_results,
):
    if (
        FINAL_ITERATION_STRATEGY
        == "latest_fold"
    ):
        latest_fold = max(
            fold_results,
            key=lambda x: x[
                "validation_season"
            ],
        )

        return int(
            latest_fold[
                "final_iterations"
            ]
        )

    raise ValueError(
        "Unsupported "
        "FINAL_ITERATION_STRATEGY: "
        f"{FINAL_ITERATION_STRATEGY}"
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )

    # -----------------------------------------------------
    # Feature schema
    # -----------------------------------------------------

    # test.csv의 다른 행이나 분포는 사용하지 않는다.
    # header만 읽어 train/test 공통 input schema를 확정한다.
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        c
        for c in test_cols
        if c != ID
    ]

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + base_features
            + [TARGET]
        ),
    )

    print("=" * 72)
    print(
        f"Experiment           : "
        f"{EXPERIMENT_NAME}"
    )
    print(
        "Model                : "
        "CatBoostRegressor"
    )
    print(
        "Objective            : RMSE"
    )
    print(
        "Validation           : "
        "Expanding Window"
    )
    print(
        "Feature Engineering  : "
        "HB-EB"
    )
    print(
        f"Train shape          : "
        f"{train.shape}"
    )
    print(
        f"Base Feature 수      : "
        f"{len(base_features)}"
    )
    print(
        f"Categorical Feature  : "
        f"{len(cat_cols)}"
    )
    print(
        "시즌                 : "
        f"{train['season'].min()} "
        "~ "
        f"{train['season'].max()}"
    )
    print(
        f"Target mean          : "
        f"{train[TARGET].mean():.6f}"
    )
    print("=" * 72)

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Expanding Window Validation
    # -----------------------------------------------------

    (
        fold_results,
        oof,
        final_features,
    ) = run_expanding_validation(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    summary = (
        save_validation_summary(
            fold_results=fold_results,
            oof=oof,
        )
    )

    print()
    print("=" * 72)
    print(
        "Expanding Window Summary"
    )
    print("=" * 72)

    for fold in fold_results:
        print(
            f"Valid "
            f"{fold['validation_season']} | "
            f"Brier="
            f"{fold['brier']:.8f} | "
            f"BSS="
            f"{fold['bss']:.2f} | "
            f"iter="
            f"{fold['final_iterations']} | "
            f"outside="
            f"{fold['ratio_prediction_outside_0_1']:.6%}"
        )

    print()
    print(
        f"Pooled OOF Brier     : "
        f"{summary['pooled_oof_brier']:.8f}"
    )
    print(
        f"Pooled OOF BSS       : "
        f"{summary['pooled_oof_bss']:.2f}"
    )
    print(
        f"Recent 23~24 Brier   : "
        f"{summary['recent_2023_2024_mean_brier']:.8f}"
    )
    print(
        f"OOF outside [0,1]    : "
        f"{summary['ratio_prediction_outside_0_1']:.6%}"
    )
    print(
        f"Validation 저장      : "
        f"{VALIDATION_DIR}"
    )
    print("=" * 72)

    final_iterations = (
        select_final_iterations(
            fold_results
        )
    )

    print()
    print(
        "Final iteration strategy : "
        f"{FINAL_ITERATION_STRATEGY}"
    )
    print(
        "Final iterations         : "
        f"{final_iterations}"
    )

    # -----------------------------------------------------
    # 전체 train으로 최종 HB-EB prior 생성
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "최종 HB-EB prior 및 "
        "CatBoostRegressor 학습"
    )
    print("=" * 72)

    df_full = (
        train.copy()
    )

    final_priors_dict = {}
    final_features_check = (
        list(base_features)
    )

    for config in (
        HBEB_CONFIGS
    ):
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_full,
            group_cols=config[
                "group_cols"
            ],
            rate_col=config[
                "rate_col"
            ],
        )

        df_full[
            config["new_col"]
        ] = apply_hbeb(
            df=df_full,
            group_cols=config[
                "group_cols"
            ],
            n_col=config[
                "n_col"
            ],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=(
                global_prior
            ),
        )

        if (
            config["rate_col"]
            in final_features_check
        ):
            final_features_check.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in final_features_check
        ):
            final_features_check.append(
                config["new_col"]
            )

        final_priors_dict[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config["group_cols"]
            ),
            "n_col": config[
                "n_col"
            ],
            "rate_col": config[
                "rate_col"
            ],
        }

    if (
        final_features_check
        != final_features
    ):
        raise ValueError(
            "Validation과 Final Training의 "
            "Feature 구성이 일치하지 않습니다."
        )

    X_full = prepare_features(
        df_full[
            final_features
        ],
        cat_cols,
    )

    y_full = (
        df_full[TARGET]
        .astype(np.float64)
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    final_model = (
        CatBoostRegressor(
            **final_params
        )
    )

    start_time = time.time()

    final_model.fit(
        X_full,
        y_full,
        cat_features=cat_cols,
    )

    final_elapsed = (
        time.time()
        - start_time
    )

    # -----------------------------------------------------
    # 단일 Artifact 저장
    # -----------------------------------------------------

    final_artifact = {
        "model": final_model,
        "hbeb_priors": (
            final_priors_dict
        ),
        "features": (
            final_features
        ),
        "cat_cols": cat_cols,
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "early_stopping_policy": {
            "default": "full_validation",
            "special_validation_season": int(
                CORRECTED_TOTAL_ES_SEASON
            ),
            "special_policy": "corrected_total_R_raw_F_train_only_shift",
            "corrected_game_type": (
                CORRECTED_TOTAL_F_GAME_TYPE
            ),
        },
        "final_iterations": int(
            final_iterations
        ),
        "prediction_policy": (
            "np.clip(raw_prediction, 0.0, 1.0)"
        ),
    }

    joblib.dump(
        final_artifact,
        MODEL_PATH,
        compress=3,
    )

    final_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "early_stopping_policy": {
            "default": "full_validation",
            "special_validation_season": int(
                CORRECTED_TOTAL_ES_SEASON
            ),
            "special_policy": "corrected_total_R_raw_F_train_only_shift",
            "corrected_game_type": (
                CORRECTED_TOTAL_F_GAME_TYPE
            ),
            "training_rows": "R_and_F_all",
            "fold_evaluation_rows": "R_and_F_all",
        },
        "validation_method": (
            "expanding_window"
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "final_iterations": int(
            final_iterations
        ),
        "training_seasons": sorted(
            [
                int(x)
                for x in train[
                    "season"
                ].unique()
            ]
        ),
        "n_train": int(
            len(train)
        ),
        "base_features": (
            base_features
        ),
        "features": (
            final_features
        ),
        "categorical_features": (
            cat_cols
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "model_params": (
            final_params
        ),
        "elapsed_seconds": float(
            final_elapsed
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "prediction_policy": (
            "clip_to_0_1"
        ),
    }

    with open(
        MODEL_DIR
        / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    source_path = (
        Path(__file__)
        .resolve()
    )

    shutil.copy2(
        source_path,
        MODEL_DIR
        / "train.py",
    )

    print()
    print("=" * 72)
    print(
        "최종 Artifact 저장 완료"
    )
    print(
        f"- Model   : "
        f"{MODEL_PATH}"
    )
    print(
        f"- Metadata: "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        f"- Train.py: "
        f"{MODEL_DIR / 'train.py'}"
    )
    print(
        f"- elapsed : "
        f"{final_elapsed:.1f}s"
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

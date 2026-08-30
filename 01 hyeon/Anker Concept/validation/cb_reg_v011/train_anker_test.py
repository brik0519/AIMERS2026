# train_cb_reg_v011.py
#
# cb_reg_v003-based minimal Target Centering Ablation
#
# 목적
# -----------------------------------------------------------------------------
# cb_reg_v003의 Feature / HB-EB / CatBoost 설정은 유지하고,
# 오직 target representation만 변경한다.
#
# Control   : y = control_success
# Treatment : y_centered = y - mean(control_success | season, R/F)
#
# 여기서 R/F:
#   F = game_type == "F"
#   R = game_type != "F"
#
# 핵심 질문
# -----------------------------------------------------------------------------
# 1) 2023 best_iteration≈2 문제가 target centering으로 복구되는가?
# 2) 단순히 iteration만 늘어나는 것이 아니라 residual 구조가 개선되는가?
# 3) 2024 성능을 훼손하지 않는가?
#
# 매우 중요
# -----------------------------------------------------------------------------
# - 이 파일은 "검증 실험"용이다.
# - 2025 제출 artifact를 만들지 않는다.
# - validation label로 만든 ORACLE anchor는 진단 전용이며,
#   early stopping / 모델 선택 / 최종 정책 선택에 절대 사용하지 않는다.
# - 실제 deployment-like 평가는 validation season보다 과거 정보만으로 만든
#   blind anchor를 사용한다.
#
# cb_reg_v003 baseline metadata:
#   iterations=800, learning_rate=0.07, depth=8, RMSE,
#   l2_leaf_reg=10, random_strength=1, seed=42,
#   max_ctr_complexity=1, early_stopping_rounds=80
#
# 출력
# -----------------------------------------------------------------------------
# validation/cb_reg_v011/
#   fold_2021/
#   fold_2022/
#   fold_2023/
#   fold_2024/
#   summary.json
#
# 각 fold에서 Control/Treatment를 같은 데이터/feature로 paired 비교한다.

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =============================================================================
# 실험 설정
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v011"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
DOMAIN_SOURCE_COL = "game_type"

F_VALUE = "F"

EARLY_STOPPING_ROUNDS = 80

EXPANDING_FOLDS = [
    ([2019, 2020], 2021),
    ([2019, 2020, 2021], 2022),
    ([2019, 2020, 2021, 2022], 2023),
    ([2019, 2020, 2021, 2022, 2023], 2024),
]

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

# cb_reg_v003의 현행 HB-EB 2개만 유지.
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


# =============================================================================
# Metric
# =============================================================================

def brier_score(y_true, pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def baseline_brier(y_true):
    y = np.asarray(y_true, dtype=np.float64)
    r = float(np.mean(y))
    return float(np.mean((y - r) ** 2))


def bss_raw(y_true, pred):
    base = baseline_brier(y_true)
    score = brier_score(y_true, pred)
    if base <= 0:
        return 0.0
    return float(100000.0 * (base - score) / base)


def bss_clipped_at_zero(y_true, pred):
    return max(0.0, bss_raw(y_true, pred))


# =============================================================================
# Domain helper
# =============================================================================

def make_domain(df):
    """F if game_type == 'F', otherwise R."""
    return np.where(
        df[DOMAIN_SOURCE_COL].astype(str).to_numpy() == F_VALUE,
        "F",
        "R",
    )


# =============================================================================
# cb_reg_v003-compatible HB-EB
# =============================================================================

def _mom_prior(mu, var):
    mu = float(mu)
    var = float(var)

    if not np.isfinite(mu):
        mu = 0.5

    mu = float(np.clip(mu, 1e-9, 1.0 - 1e-9))
    k = mu * (1.0 - mu)

    # cb_reg_v003 guard semantics
    if (
        (not np.isfinite(var))
        or var <= 0.0
        or var >= k
    ):
        var = 0.5 * k

    m = k / var - 1.0

    if (not np.isfinite(m)) or m <= 0.0:
        m = 1.0

    alpha = mu * m
    beta = (1.0 - mu) * m

    return float(alpha), float(beta)


def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    cb_reg_v003의 row-level MoM prior를 재현한다.
    """

    rate = pd.to_numeric(
        df_train[rate_col],
        errors="coerce",
    )

    global_mu = float(rate.mean())
    global_var = float(rate.var(ddof=1))
    global_prior = _mom_prior(
        global_mu,
        global_var,
    )

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
            observed=True,
        )[rate_col]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    alpha_list = []
    beta_list = []

    for _, row in stats.iterrows():
        mu = float(row["mean"])
        var = float(row["var"])
        count = int(row["count"])

        valid = (
            count >= 30
            and np.isfinite(mu)
            and 0.0 < mu < 1.0
            and np.isfinite(var)
            and var > 0.0
            and var < mu * (1.0 - mu)
        )

        if valid:
            alpha, beta = _mom_prior(
                mu,
                var,
            )
        else:
            alpha, beta = global_prior

        alpha_list.append(alpha)
        beta_list.append(beta)

    prior_df = stats[
        list(group_cols)
    ].copy()

    prior_df["alpha"] = np.asarray(
        alpha_list,
        dtype=np.float64,
    )
    prior_df["beta"] = np.asarray(
        beta_list,
        dtype=np.float64,
    )

    return prior_df, global_prior


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """
    저장된 train-only prior를 lookup한다.
    """

    lookup = (
        df[list(group_cols)]
        .merge(
            prior_df,
            on=list(group_cols),
            how="left",
            sort=False,
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
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    denominator = n + alpha + beta

    if np.any(denominator <= 0.0):
        raise ValueError(
            f"HB-EB denominator <= 0: {rate_col}"
        )

    return (
        n * rate + alpha
    ) / denominator


def apply_hbeb_configs(
    df_train,
    df_apply,
    base_features,
    priors=None,
):
    transformed = df_apply.copy()
    features = list(base_features)

    if priors is None:
        fitted_priors = {}
        fit_mode = True
    else:
        fitted_priors = priors
        fit_mode = False

    for cfg in HBEB_CONFIGS:
        new_col = cfg["new_col"]

        if fit_mode:
            prior_df, global_prior = compute_hbeb_priors(
                df_train=df_train,
                group_cols=cfg["group_cols"],
                rate_col=cfg["rate_col"],
            )

            fitted_priors[new_col] = {
                "group_cols": list(cfg["group_cols"]),
                "n_col": cfg["n_col"],
                "rate_col": cfg["rate_col"],
                "prior_df": prior_df,
                "global_prior": global_prior,
            }

        info = fitted_priors[new_col]

        transformed[new_col] = apply_hbeb(
            df=transformed,
            group_cols=info["group_cols"],
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=info["global_prior"],
        )

        # cb_reg_v003: raw rate를 EB rate로 치환
        if cfg["rate_col"] in features:
            features.remove(
                cfg["rate_col"]
            )

        if new_col not in features:
            features.append(new_col)

    return (
        transformed,
        features,
        fitted_priors,
    )


# =============================================================================
# CatBoost preprocessing
# =============================================================================

def prepare_features(
    df,
    cat_cols,
):
    out = df.copy()

    for col in cat_cols:
        if col not in out.columns:
            raise ValueError(
                f"categorical feature missing: {col}"
            )

        out[col] = (
            out[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return out


# =============================================================================
# Target Centering
# =============================================================================

def fit_observed_center_table(
    df_train,
):
    """
    TRAINING TARGET 변환용.

    각 training row의 target에서 그 row가 속한
    (season, R/F)의 관측 평균을 뺀다.

    validation 정보는 사용하지 않는다.
    """

    temp = pd.DataFrame({
        SEASON_COL: df_train[SEASON_COL].to_numpy(),
        "domain": make_domain(df_train),
        TARGET_COL: df_train[TARGET_COL].to_numpy(
            dtype=np.float64
        ),
    })

    table = (
        temp
        .groupby(
            [SEASON_COL, "domain"],
            observed=True,
        )[TARGET_COL]
        .mean()
        .rename("center")
        .reset_index()
    )

    return table


def center_training_target(
    df_train,
    center_table,
):
    key = pd.DataFrame({
        SEASON_COL: df_train[SEASON_COL].to_numpy(),
        "domain": make_domain(df_train),
    })

    lookup = key.merge(
        center_table,
        on=[SEASON_COL, "domain"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if lookup["center"].isna().any():
        raise RuntimeError(
            "training target center lookup failed"
        )

    y = df_train[TARGET_COL].to_numpy(
        dtype=np.float64
    )

    center = lookup["center"].to_numpy(
        dtype=np.float64
    )

    return y - center


def make_blind_anchor(
    df_train,
    validation_season,
):
    """
    ★ Primary deployment-like anchor.

    가장 단순하고 진짜 walk-forward인 정책:
    validation 직전 season의 R/F 실제 prevalence.

    여기서는 2023 F 붕괴를 알고 0.50을 넣는 hindsight policy를
    사용하지 않는다.

    따라서 2023에서 이 anchor가 크게 실패해도 정상이며,
    그 실패 자체가 target centering과 anchor forecasting을
    분리해서 보여주는 중요한 진단 결과다.
    """

    last_season = int(
        df_train[SEASON_COL].max()
    )

    if last_season >= int(validation_season):
        raise RuntimeError(
            "anchor source must be strictly earlier than validation season"
        )

    source = df_train.loc[
        df_train[SEASON_COL] == last_season
    ].copy()

    source_domain = make_domain(source)

    global_mean = float(
        source[TARGET_COL].mean()
    )

    anchor = {
        "policy": "previous_season_R_F_prevalence",
        "source_seasons": [last_season],
    }

    for domain in ["R", "F"]:
        mask = source_domain == domain

        if np.any(mask):
            anchor[domain] = float(
                source.loc[
                    mask,
                    TARGET_COL,
                ].mean()
            )
        else:
            anchor[domain] = global_mean

    return anchor


def make_oracle_anchor(
    df_valid,
):
    """
    [ORACLE - DIAGNOSTIC ONLY]

    validation label을 사용한다.
    절대로:
      - early stopping
      - model selection
      - parameter selection
      - deployment policy
    에 사용하지 않는다.

    목적:
    anchor forecast error를 제거했을 때 centered model의
    residual structure 자체가 좋아졌는지만 확인.
    """

    domain = make_domain(
        df_valid
    )

    global_mean = float(
        df_valid[TARGET_COL].mean()
    )

    anchor = {
        "policy": "ORACLE_DIAGNOSTIC_ONLY",
        "source_seasons": [
            int(df_valid[SEASON_COL].iloc[0])
        ],
    }

    for d in ["R", "F"]:
        mask = domain == d

        if np.any(mask):
            anchor[d] = float(
                df_valid.loc[
                    mask,
                    TARGET_COL,
                ].mean()
            )
        else:
            anchor[d] = global_mean

    return anchor


def anchor_vector(
    df,
    anchor,
):
    domain = make_domain(df)

    return np.where(
        domain == "F",
        float(anchor["F"]),
        float(anchor["R"]),
    ).astype(np.float64)


def reconstruct_probability(
    df,
    residual_pred,
    anchor,
):
    return (
        np.asarray(
            residual_pred,
            dtype=np.float64,
        )
        + anchor_vector(
            df,
            anchor,
        )
    )


# =============================================================================
# Diagnostics
# =============================================================================

def prediction_summary(
    y_true,
    pred_raw,
):
    raw = np.asarray(
        pred_raw,
        dtype=np.float64,
    )
    clipped = np.clip(
        raw,
        0.0,
        1.0,
    )

    return {
        "raw_brier": brier_score(
            y_true,
            raw,
        ),
        "raw_bss_raw": bss_raw(
            y_true,
            raw,
        ),
        "clipped_brier": brier_score(
            y_true,
            clipped,
        ),
        "clipped_bss_raw": bss_raw(
            y_true,
            clipped,
        ),
        "clipped_bss": bss_clipped_at_zero(
            y_true,
            clipped,
        ),
        "prediction_raw_mean": float(
            np.mean(raw)
        ),
        "prediction_raw_std": float(
            np.std(raw)
        ),
        "prediction_raw_min": float(
            np.min(raw)
        ),
        "prediction_raw_max": float(
            np.max(raw)
        ),
        "n_prediction_below_0": int(
            np.sum(raw < 0.0)
        ),
        "n_prediction_above_1": int(
            np.sum(raw > 1.0)
        ),
        "ratio_prediction_outside_0_1": float(
            np.mean(
                (raw < 0.0)
                | (raw > 1.0)
            )
        ),
    }


def domain_summary(
    df,
    y_true,
    pred,
):
    y = np.asarray(
        y_true,
        dtype=np.float64,
    )
    p = np.asarray(
        pred,
        dtype=np.float64,
    )

    domain = make_domain(df)

    result = {}

    for d in ["R", "F"]:
        mask = domain == d

        if not np.any(mask):
            continue

        yd = y[mask]
        pd_ = p[mask]

        result[d] = {
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "actual_mean": float(
                yd.mean()
            ),
            "pred_mean": float(
                pd_.mean()
            ),
            "bias": float(
                pd_.mean()
                - yd.mean()
            ),
            "brier": brier_score(
                yd,
                pd_,
            ),
            "bss_raw": bss_raw(
                yd,
                pd_,
            ),
        }

    return result


def centered_residual_summary(
    df,
    y_centered_oracle,
    residual_pred,
):
    y = np.asarray(
        y_centered_oracle,
        dtype=np.float64,
    )
    p = np.asarray(
        residual_pred,
        dtype=np.float64,
    )

    domain = make_domain(df)

    result = {}

    for d in ["R", "F"]:
        mask = domain == d

        if not np.any(mask):
            continue

        err = p[mask] - y[mask]

        result[d] = {
            "n": int(mask.sum()),
            "target_mean": float(
                y[mask].mean()
            ),
            "prediction_mean": float(
                p[mask].mean()
            ),
            "bias": float(
                err.mean()
            ),
            "mse": float(
                np.mean(
                    err ** 2
                )
            ),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            ),
        }

    return result


# =============================================================================
# Train arms
# =============================================================================

def train_control(
    X_train,
    y_train,
    X_valid,
    y_valid,
    cat_feature_indices,
):
    model = CatBoostRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(
            X_valid,
            y_valid,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    pred = model.predict(
        X_valid
    )

    return (
        model,
        np.asarray(
            pred,
            dtype=np.float64,
        ),
        int(
            model.get_best_iteration()
        ),
    )


def train_centered(
    X_train,
    y_centered_train,
    X_valid,
    y_centered_valid_blind,
    cat_feature_indices,
):
    """
    Treatment의 early stopping도 blind past-only anchor를 기준으로 한다.

    validation label의 R/F 평균(oracle)은 ES에 사용하지 않는다.
    """

    model = CatBoostRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_centered_train,
        cat_features=cat_feature_indices,
        eval_set=(
            X_valid,
            y_centered_valid_blind,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    residual_pred = model.predict(
        X_valid
    )

    return (
        model,
        np.asarray(
            residual_pred,
            dtype=np.float64,
        ),
        int(
            model.get_best_iteration()
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    start_all = time.time()

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. "
            "Remove it or change EXPERIMENT_NAME."
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    df = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
    )

    if ID_COL not in df.columns:
        raise ValueError(
            f"{ID_COL} missing"
        )

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"{TARGET_COL} missing"
        )

    # cb_reg_v003 원본 47 raw feature 구성:
    # row_id, target만 제외
    BASE_FEATURES = [
        col
        for col in df.columns
        if col not in {
            ID_COL,
            TARGET_COL,
        }
    ]

    print("=" * 96)
    print(f"Experiment           : {EXPERIMENT_NAME}")
    print("Model                : CatBoostRegressor")
    print("Objective            : RMSE")
    print("Validation           : Expanding Window")
    print("Baseline             : cb_reg_v003-compatible")
    print("Ablation             : Absolute target vs R/F Target Centering")
    print(f"Train shape          : {df.shape}")
    print(f"Base Feature 수      : {len(BASE_FEATURES)}")
    print(f"Categorical Feature  : {len(CAT_COLS)}")
    print(f"시즌                 : {df[SEASON_COL].min()} ~ {df[SEASON_COL].max()}")
    print(f"Target mean          : {df[TARGET_COL].mean():.6f}")
    print("=" * 96)

    fold_results = []

    for (
        fold_idx,
        (train_seasons, valid_season),
    ) in enumerate(
        EXPANDING_FOLDS,
        start=1,
    ):
        fold_start = time.time()

        print()
        print("=" * 96)
        print(
            f"Fold {fold_idx}: "
            f"{train_seasons[0]}~{train_seasons[-1]} "
            f"-> {valid_season}"
        )
        print("=" * 96)

        df_train = (
            df[
                df[SEASON_COL]
                .isin(train_seasons)
            ]
            .copy()
        )

        df_valid = (
            df[
                df[SEASON_COL]
                == valid_season
            ]
            .copy()
        )

        print(
            f"학습 데이터: {df_train.shape}"
        )
        print(
            f"검증 데이터: {df_valid.shape}"
        )

        # -----------------------------------------------------
        # HB-EB
        # -----------------------------------------------------

        train_fe, final_features, priors = (
            apply_hbeb_configs(
                df_train=df_train,
                df_apply=df_train,
                base_features=BASE_FEATURES,
                priors=None,
            )
        )

        valid_fe, valid_features, _ = (
            apply_hbeb_configs(
                df_train=df_train,
                df_apply=df_valid,
                base_features=BASE_FEATURES,
                priors=priors,
            )
        )

        if (
            final_features
            != valid_features
        ):
            raise RuntimeError(
                "train/validation feature mismatch"
            )

        active_cat_cols = [
            col
            for col in CAT_COLS
            if col in final_features
        ]

        X_train = prepare_features(
            train_fe[
                final_features
            ],
            active_cat_cols,
        )

        X_valid = prepare_features(
            valid_fe[
                final_features
            ],
            active_cat_cols,
        )

        cat_feature_indices = [
            final_features.index(col)
            for col in active_cat_cols
        ]

        y_train = (
            df_train[TARGET_COL]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_valid = (
            df_valid[TARGET_COL]
            .to_numpy(
                dtype=np.float64
            )
        )

        # =====================================================
        # CONTROL
        # =====================================================

        print()
        print("[Control] Absolute target")

        (
            control_model,
            control_pred_raw,
            control_best_iter,
        ) = train_control(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            cat_feature_indices=(
                cat_feature_indices
            ),
        )

        control_pred = np.clip(
            control_pred_raw,
            0.0,
            1.0,
        )

        control_metrics = (
            prediction_summary(
                y_true=y_valid,
                pred_raw=(
                    control_pred_raw
                ),
            )
        )

        print(
            f"Best iteration : "
            f"{control_best_iter}"
        )
        print(
            f"Brier          : "
            f"{control_metrics['clipped_brier']:.8f}"
        )
        print(
            f"BSS            : "
            f"{control_metrics['clipped_bss_raw']:.2f}"
        )

        # =====================================================
        # TREATMENT
        # =====================================================

        print()
        print(
            "[Treatment] "
            "Target = y - mean(y | season, R/F)"
        )

        center_table = (
            fit_observed_center_table(
                df_train
            )
        )

        y_train_centered = (
            center_training_target(
                df_train=df_train,
                center_table=(
                    center_table
                ),
            )
        )

        blind_anchor = (
            make_blind_anchor(
                df_train=df_train,
                validation_season=(
                    valid_season
                ),
            )
        )

        oracle_anchor = (
            make_oracle_anchor(
                df_valid
            )
        )

        # ES에 사용하는 validation target:
        # validation true labels 자체는 metric 계산에 필요하지만,
        # center level은 과거 anchor만 사용한다.
        blind_anchor_vec = (
            anchor_vector(
                df_valid,
                blind_anchor,
            )
        )

        y_valid_centered_blind = (
            y_valid
            - blind_anchor_vec
        )

        (
            centered_model,
            residual_pred,
            centered_best_iter,
        ) = train_centered(
            X_train=X_train,
            y_centered_train=(
                y_train_centered
            ),
            X_valid=X_valid,
            y_centered_valid_blind=(
                y_valid_centered_blind
            ),
            cat_feature_indices=(
                cat_feature_indices
            ),
        )

        # -----------------------------------------------------
        # deployment-like blind reconstruction
        # -----------------------------------------------------

        centered_blind_raw = (
            reconstruct_probability(
                df=df_valid,
                residual_pred=(
                    residual_pred
                ),
                anchor=blind_anchor,
            )
        )

        centered_blind_pred = np.clip(
            centered_blind_raw,
            0.0,
            1.0,
        )

        centered_blind_metrics = (
            prediction_summary(
                y_true=y_valid,
                pred_raw=(
                    centered_blind_raw
                ),
            )
        )

        # -----------------------------------------------------
        # ORACLE reconstruction
        # 진단 전용
        # -----------------------------------------------------

        centered_oracle_raw = (
            reconstruct_probability(
                df=df_valid,
                residual_pred=(
                    residual_pred
                ),
                anchor=oracle_anchor,
            )
        )

        centered_oracle_pred = np.clip(
            centered_oracle_raw,
            0.0,
            1.0,
        )

        centered_oracle_metrics = (
            prediction_summary(
                y_true=y_valid,
                pred_raw=(
                    centered_oracle_raw
                ),
            )
        )

        oracle_anchor_vec = (
            anchor_vector(
                df_valid,
                oracle_anchor,
            )
        )

        y_valid_centered_oracle = (
            y_valid
            - oracle_anchor_vec
        )

        residual_diag = (
            centered_residual_summary(
                df=df_valid,
                y_centered_oracle=(
                    y_valid_centered_oracle
                ),
                residual_pred=(
                    residual_pred
                ),
            )
        )

        print()
        print(
            f"Centered best iteration : "
            f"{centered_best_iter}"
        )
        print(
            "Blind anchor             : "
            f"R={blind_anchor['R']:.6f}, "
            f"F={blind_anchor['F']:.6f}"
        )
        print(
            "Blind source seasons     : "
            f"{blind_anchor['source_seasons']}"
        )
        print(
            f"Blind Brier              : "
            f"{centered_blind_metrics['clipped_brier']:.8f}"
        )
        print(
            f"Blind BSS                : "
            f"{centered_blind_metrics['clipped_bss_raw']:.2f}"
        )
        print(
            f"ΔBSS blind vs Control    : "
            f"{centered_blind_metrics['clipped_bss_raw'] - control_metrics['clipped_bss_raw']:+.2f}"
        )

        print()
        print(
            "[ORACLE - DIAGNOSTIC ONLY]"
        )
        print(
            "Oracle anchor            : "
            f"R={oracle_anchor['R']:.6f}, "
            f"F={oracle_anchor['F']:.6f}"
        )
        print(
            f"Oracle Brier             : "
            f"{centered_oracle_metrics['clipped_brier']:.8f}"
        )
        print(
            f"Oracle BSS               : "
            f"{centered_oracle_metrics['clipped_bss_raw']:.2f}"
        )
        print(
            f"ΔBSS oracle vs Control   : "
            f"{centered_oracle_metrics['clipped_bss_raw'] - control_metrics['clipped_bss_raw']:+.2f}"
        )

        # -----------------------------------------------------
        # Domain diagnostics
        # -----------------------------------------------------

        control_domain = (
            domain_summary(
                df=df_valid,
                y_true=y_valid,
                pred=control_pred,
            )
        )

        blind_domain = (
            domain_summary(
                df=df_valid,
                y_true=y_valid,
                pred=centered_blind_pred,
            )
        )

        oracle_domain = (
            domain_summary(
                df=df_valid,
                y_true=y_valid,
                pred=centered_oracle_pred,
            )
        )

        print()
        print("[Domain diagnostics]")

        for d in ["R", "F"]:
            c = control_domain[d]
            t = blind_domain[d]

            print(
                f"{d} | "
                f"Control Brier={c['brier']:.8f}, "
                f"Centered Blind={t['brier']:.8f}, "
                f"Δ={c['brier'] - t['brier']:+.8f}"
            )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        fold_dir = (
            VALIDATION_DIR
            / f"fold_{valid_season}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        metadata = {
            "experiment_name": (
                EXPERIMENT_NAME
            ),
            "fold": int(fold_idx),
            "train_seasons": [
                int(s)
                for s in train_seasons
            ],
            "validation_season": int(
                valid_season
            ),
            "n_train": int(
                len(df_train)
            ),
            "n_validation": int(
                len(df_valid)
            ),
            "features": list(
                final_features
            ),
            "cat_cols": list(
                active_cat_cols
            ),
            "model_params": dict(
                MODEL_PARAMS
            ),
            "control": {
                "target_representation": (
                    "absolute"
                ),
                "best_iteration": int(
                    control_best_iter
                ),
                "metrics": (
                    control_metrics
                ),
                "domain_metrics": (
                    control_domain
                ),
            },
            "treatment": {
                "target_representation": (
                    "season_x_RF_centered"
                ),
                "best_iteration": int(
                    centered_best_iter
                ),
                "training_center_table": (
                    center_table
                    .to_dict(
                        orient="records"
                    )
                ),
                "blind_anchor": (
                    blind_anchor
                ),
                "blind_metrics": (
                    centered_blind_metrics
                ),
                "blind_domain_metrics": (
                    blind_domain
                ),
                "delta_bss_blind_vs_control": float(
                    centered_blind_metrics[
                        "clipped_bss_raw"
                    ]
                    - control_metrics[
                        "clipped_bss_raw"
                    ]
                ),
                "oracle_DIAGNOSTIC_ONLY": {
                    "anchor": (
                        oracle_anchor
                    ),
                    "metrics": (
                        centered_oracle_metrics
                    ),
                    "domain_metrics": (
                        oracle_domain
                    ),
                    "residual_metrics": (
                        residual_diag
                    ),
                    "delta_bss_vs_control": float(
                        centered_oracle_metrics[
                            "clipped_bss_raw"
                        ]
                        - control_metrics[
                            "clipped_bss_raw"
                        ]
                    ),
                    "warning": (
                        "Validation labels used. "
                        "Never use for selection/deployment."
                    ),
                },
            },
            "elapsed_seconds": float(
                time.time()
                - fold_start
            ),
        }

        with open(
            fold_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        fold_results.append(
            metadata
        )

    # =============================================================================
    # Summary
    # =============================================================================

    print()
    print("=" * 96)
    print("Target Centering Minimal Ablation Summary")
    print("=" * 96)

    for result in fold_results:
        season = result[
            "validation_season"
        ]

        c = result["control"]
        t = result["treatment"]

        print(
            f"Valid {season} | "
            f"Control iter={c['best_iteration']:3d}, "
            f"Brier={c['metrics']['clipped_brier']:.8f}, "
            f"BSS={c['metrics']['clipped_bss_raw']:.2f} | "
            f"Centered iter={t['best_iteration']:3d}, "
            f"Blind Brier={t['blind_metrics']['clipped_brier']:.8f}, "
            f"Blind BSS={t['blind_metrics']['clipped_bss_raw']:.2f}, "
            f"Δ={t['delta_bss_blind_vs_control']:+.2f} | "
            f"[ORACLE] Δ={t['oracle_DIAGNOSTIC_ONLY']['delta_bss_vs_control']:+.2f}"
        )

    summary = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "baseline": (
            "cb_reg_v003-compatible"
        ),
        "experiment_type": (
            "minimal_target_centering_ablation"
        ),
        "control": (
            "absolute_target"
        ),
        "treatment": (
            "season_x_RF_centered_target"
        ),
        "primary_questions": [
            (
                "Does 2023 best_iteration recover?"
            ),
            (
                "Does oracle residual structure improve?"
            ),
            (
                "Does 2024 regress?"
            ),
        ],
        "blind_anchor_policy": (
            "previous_season_R_F_prevalence"
        ),
        "oracle_usage": (
            "diagnostic_only_never_for_selection"
        ),
        "folds": fold_results,
        "elapsed_seconds": float(
            time.time()
            - start_all
        ),
    }

    with open(
        VALIDATION_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("[판정 가이드]")
    print(
        "1) 2023 centered best_iteration 증가만으로 채택하지 않는다."
    )
    print(
        "2) [ORACLE]은 centering의 model-only 구조 진단용이다."
    )
    print(
        "3) 2023 [ORACLE]이 개선되고 2024도 >= control이면 centering 후보."
    )
    print(
        "4) Blind 2023이 붕괴해도 anchor forecast 실패와 centering 실패를 구분한다."
    )
    print(
        "5) 이 실험은 제출 artifact를 생성하지 않는다."
    )

    print()
    print(
        f"Validation 저장: "
        f"{VALIDATION_DIR}"
    )
    print(
        f"Elapsed        : "
        f"{time.time() - start_all:.1f}s"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()

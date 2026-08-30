# train.py
# cb_reg_v018_gated

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =============================================================================
# Paths
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v018_gated"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = Path("./model") / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_DIR = Path("./validation") / EXPERIMENT_NAME


# =============================================================================
# Basic config
# =============================================================================

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


# =============================================================================
# V18 gate config — 이미 discovery에서 freeze한 값
# =============================================================================

V18_PARENT_KEYS = [
    "pitcher_id",
    "batter_hand",
]

V18_DETAIL_KEYS = [
    "pitcher_id",
    "pressure_state",
    "batter_hand",
]

V18_CONFIG = {
    "min_n_parent": 200,
    "min_n_detail": 200,
    "shrink_parent": 300.0,
    "shrink_detail": 100.0,
    "parent_only_scale": 0.5,
    "detail_weight": 0.5,
    "alpha": 0.25,
    "cap": 0.02,
    "min_abs_effect": 0.003,
    "center_residual_by_game_type": True,
}

V18_SOURCE_SEASON = 2024


# =============================================================================
# HB-EB
# =============================================================================

def compute_hbeb_priors(df_train, group_cols, rate_col):
    """Fold/full train만 사용해 HB-EB prior를 추정한다."""

    g_mu = float(df_train[rate_col].mean())
    g_var = float(df_train[rate_col].var())

    max_bernoulli_var = g_mu * (1.0 - g_mu)

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = max_bernoulli_var * 0.5

    g_m = (max_bernoulli_var / g_var) - 1.0
    g_alpha = g_mu * g_m
    g_beta = (1.0 - g_mu) * g_m

    stats = (
        df_train
        .groupby(group_cols, dropna=False)[rate_col]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    group_max_var = stats["mean"] * (1.0 - stats["mean"])

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
        mu = stats.loc[valid_mask, "mean"]
        var = stats.loc[valid_mask, "var"]

        m = (mu * (1.0 - mu) / var) - 1.0

        stats.loc[valid_mask, "alpha"] = mu * m
        stats.loc[valid_mask, "beta"] = (1.0 - mu) * m

    prior_df = stats[
        group_cols + ["alpha", "beta"]
    ].copy()

    return prior_df, (float(g_alpha), float(g_beta))


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """저장된 train-only prior를 각 행에 독립적으로 적용한다."""

    original_index = df.index

    lookup_df = df[group_cols].merge(
        prior_df,
        on=group_cols,
        how="left",
        sort=False,
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
    denominator = n + alpha + beta
    eb_rate = (success_count + alpha) / denominator

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def transform_hbeb_pair(
    df_train,
    df_apply,
    base_features,
):
    """
    df_train에서 prior를 fit하고 df_apply에 적용.
    Validation에서는 df_train=fold train, df_apply=validation.
    """

    transformed = df_apply.copy()
    features = list(base_features)
    priors = {}

    for config in HBEB_CONFIGS:
        prior_df, global_prior = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config["group_cols"],
            rate_col=config["rate_col"],
        )

        transformed[config["new_col"]] = apply_hbeb(
            df=transformed,
            group_cols=config["group_cols"],
            n_col=config["n_col"],
            rate_col=config["rate_col"],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if config["rate_col"] in features:
            features.remove(config["rate_col"])

        if config["new_col"] not in features:
            features.append(config["new_col"])

        priors[config["new_col"]] = {
            "prior_df": prior_df,
            "global_prior": global_prior,
            "group_cols": list(config["group_cols"]),
            "n_col": config["n_col"],
            "rate_col": config["rate_col"],
        }

    return transformed, features, priors


def transform_fold_train_and_val(
    df_train,
    df_val,
    base_features,
):
    """동일한 fold-train prior를 train/validation 양쪽에 적용한다."""

    train_out = df_train.copy()
    val_out = df_val.copy()

    features = list(base_features)

    for config in HBEB_CONFIGS:
        prior_df, global_prior = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config["group_cols"],
            rate_col=config["rate_col"],
        )

        train_out[config["new_col"]] = apply_hbeb(
            df=train_out,
            group_cols=config["group_cols"],
            n_col=config["n_col"],
            rate_col=config["rate_col"],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        val_out[config["new_col"]] = apply_hbeb(
            df=val_out,
            group_cols=config["group_cols"],
            n_col=config["n_col"],
            rate_col=config["rate_col"],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if config["rate_col"] in features:
            features.remove(config["rate_col"])

        if config["new_col"] not in features:
            features.append(config["new_col"])

    return train_out, val_out, features


def prepare_features(df, cat_cols):
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =============================================================================
# Metrics
# =============================================================================

def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(
        np.mean((y_pred - y_true) ** 2)
    )


# =============================================================================
# V18 residual correction
# =============================================================================

def add_pressure_state(df):
    """
    pressure_state는 공개 V18의 정확한 원본 수식이 아니며,
    이번 프로젝트에서 사용한 operational definition.
    """

    df = df.copy()

    balls = (
        pd.to_numeric(
            df["balls_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    strikes = (
        pd.to_numeric(
            df["strikes_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(
            (balls == 3) | (strikes == 2),
            "high",
            "normal",
        ),
    )

    return df


def fit_v18_table(
    source_df,
    key_cols,
    min_n,
    shrink_k,
):
    """
    source season OOF residual로 frozen lookup table 생성.

    effect        = mean(centered residual)
    reliability   = n / (n + shrink_k)
    shrunk_effect = effect * reliability
    """

    stats = (
        source_df
        .groupby(
            key_cols,
            dropna=False,
            sort=False,
        )["conditional_residual"]
        .agg(
            n="size",
            effect="mean",
        )
        .reset_index()
    )

    eligible = (
        (stats["n"] >= int(min_n))
        & (
            stats["effect"].abs()
            >= float(V18_CONFIG["min_abs_effect"])
        )
    )

    stats["reliability"] = (
        stats["n"]
        / (
            stats["n"]
            + float(shrink_k)
        )
    )

    stats["shrunk_effect"] = (
        stats["effect"]
        * stats["reliability"]
    )

    stats.loc[
        ~eligible,
        "shrunk_effect",
    ] = 0.0

    return (
        stats.loc[
            stats["shrunk_effect"] != 0.0
        ]
        .copy()
        .reset_index(drop=True)
    )


def build_final_v18_correction(
    train,
    oof,
):
    """
    최종 deployment correction table 생성.

    source는 2024 OOF만 사용:
      2019~2023으로 학습한 v003 fold model
      -> 2024 prediction
      -> 2024 residual
      -> frozen 2025/test lookup table
    """

    oof_2024 = (
        oof.loc[
            oof["validation_season"]
            == V18_SOURCE_SEASON
        ]
        .copy()
    )

    if oof_2024.empty:
        raise ValueError(
            "2024 OOF rows가 없습니다."
        )

    needed_cols = [
        ID,
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    missing = [
        col
        for col in needed_cols
        if col not in train.columns
    ]

    if missing:
        raise KeyError(
            "V18 correction에 필요한 train column이 없습니다: "
            + ", ".join(missing)
        )

    meta = train[needed_cols].copy()

    source = oof_2024.merge(
        meta,
        on=ID,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if source[
        "pitcher_id"
    ].isna().all():
        raise ValueError(
            "OOF row_id와 train row_id merge 실패."
        )

    source = add_pressure_state(source)

    for col in [
        "game_type",
        "pitcher_id",
        "batter_hand",
        "pressure_state",
    ]:
        source[col] = (
            source[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    source["residual"] = (
        source["target"].to_numpy(dtype=np.float64)
        - source["prediction"].to_numpy(dtype=np.float64)
    )

    if V18_CONFIG[
        "center_residual_by_game_type"
    ]:
        domain_mean = (
            source
            .groupby(
                "game_type",
                dropna=False,
            )["residual"]
            .transform("mean")
            .to_numpy(dtype=np.float64)
        )

        source["conditional_residual"] = (
            source["residual"].to_numpy(dtype=np.float64)
            - domain_mean
        )
    else:
        source["conditional_residual"] = (
            source["residual"].to_numpy(dtype=np.float64)
        )

    parent_table = fit_v18_table(
        source_df=source,
        key_cols=V18_PARENT_KEYS,
        min_n=V18_CONFIG["min_n_parent"],
        shrink_k=V18_CONFIG["shrink_parent"],
    )

    detail_table = fit_v18_table(
        source_df=source,
        key_cols=V18_DETAIL_KEYS,
        min_n=V18_CONFIG["min_n_detail"],
        shrink_k=V18_CONFIG["shrink_detail"],
    )

    if parent_table.empty:
        raise ValueError(
            "최종 V18 parent correction table이 비어 있습니다."
        )

    return {
        "enabled": True,
        "source_season": int(V18_SOURCE_SEASON),
        "source_scope": "official_train_2024_oof_only",
        "residual_definition": "target - clipped_cb_reg_v003_oof_prediction",
        "centering": "subtract source-season game_type mean residual",
        "parent_keys": list(V18_PARENT_KEYS),
        "detail_keys": list(V18_DETAIL_KEYS),
        "parent_table": parent_table,
        "detail_table": detail_table,
        "config": dict(V18_CONFIG),
        "pressure_definition": {
            "full": "balls_before == 3 and strikes_before == 2",
            "high": "balls_before == 3 or strikes_before == 2",
            "normal": "otherwise",
        },
        "selection_note": (
            "Gate hyperparameters were frozen using "
            "2021->2022 and 2022->2023 discovery; "
            "2023->2024 was confirmation only."
        ),
        "uses_test_distribution": False,
        "uses_test_targets": False,
        "row_independent_application": True,
    }


# =============================================================================
# Expanding Window
# =============================================================================

def run_expanding_validation(
    train,
    base_features,
    cat_cols,
):
    fold_results = []
    oof_frames = []
    final_features = None

    for fold_config in EXPANDING_FOLDS:
        fold = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        validation_season = fold_config["validation_season"]

        df_train = (
            train.loc[
                train["season"].isin(train_seasons)
            ]
            .copy()
        )

        df_val = (
            train.loc[
                train["season"] == validation_season
            ]
            .copy()
        )

        (
            df_train,
            df_val,
            current_features,
        ) = transform_fold_train_and_val(
            df_train=df_train,
            df_val=df_val,
            base_features=base_features,
        )

        final_features = list(current_features)

        X_train = prepare_features(
            df_train[current_features],
            cat_cols,
        )

        y_train = (
            df_train[TARGET]
            .astype(np.float64)
        )

        X_val = prepare_features(
            df_val[current_features],
            cat_cols,
        )

        y_val = (
            df_val[TARGET]
            .astype(np.float64)
        )

        print()
        print("=" * 80)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print("=" * 80)
        print(f"Train      : {X_train.shape}")
        print(f"Validation : {X_val.shape}")

        estimator = CatBoostRegressor(
            **MODEL_PARAMS
        )

        start = time.time()

        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )

        elapsed = time.time() - start

        pred_raw = estimator.predict(
            X_val
        )

        pred_clip = np.clip(
            pred_raw,
            0.0,
            1.0,
        )

        best_iteration = (
            estimator.get_best_iteration()
        )

        if (
            best_iteration is not None
            and best_iteration >= 0
        ):
            final_iterations = (
                int(best_iteration) + 1
            )
        else:
            final_iterations = int(
                MODEL_PARAMS["iterations"]
            )

        brier = brier_score(
            y_val,
            pred_clip,
        )

        fold_dir = (
            VALIDATION_DIR
            / f"fold_{validation_season}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        pred_df = pd.DataFrame(
            {
                "row_id": df_val[ID].to_numpy(),
                "target": y_val.to_numpy(),
                "prediction_raw": pred_raw,
                "prediction_clip": pred_clip,
                "prediction": pred_clip,
                "validation_season": validation_season,
            }
        )

        pred_df.to_csv(
            fold_dir / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        np.savez_compressed(
            fold_dir / "validation.npz",
            row_id=pred_df["row_id"].to_numpy(),
            target=pred_df["target"].to_numpy(),
            prediction_raw=pred_df["prediction_raw"].to_numpy(),
            prediction_clip=pred_df["prediction_clip"].to_numpy(),
            prediction=pred_df["prediction"].to_numpy(),
            validation_season=np.full(
                len(pred_df),
                validation_season,
            ),
        )

        fold_meta = {
            "fold": int(fold),
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "validation_season": int(validation_season),
            "n_train": int(len(df_train)),
            "n_validation": int(len(df_val)),
            "brier": float(brier),
            "best_iteration": (
                None
                if best_iteration is None
                else int(best_iteration)
            ),
            "final_iterations": int(final_iterations),
            "elapsed_seconds": float(elapsed),
        }

        with open(
            fold_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                fold_meta,
                f,
                ensure_ascii=False,
                indent=2,
            )

        fold_results.append(
            fold_meta
        )

        oof_frames.append(
            pred_df
        )

        print(
            f"Brier           : {brier:.8f}"
        )
        print(
            f"Best iteration  : {best_iteration}"
        )
        print(
            f"Final iteration : {final_iterations}"
        )
        print(
            f"Elapsed         : {elapsed:.1f}s"
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction_raw=oof["prediction_raw"].to_numpy(),
        prediction_clip=oof["prediction_clip"].to_numpy(),
        prediction=oof["prediction"].to_numpy(),
        validation_season=oof["validation_season"].to_numpy(),
    )

    pooled_brier = brier_score(
        oof["target"],
        oof["prediction"],
    )

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "validation_method": "expanding_window",
        "pooled_oof_brier": float(pooled_brier),
        "folds": fold_results,
    }

    with open(
        VALIDATION_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return (
        fold_results,
        oof,
        final_features,
    )


def select_final_iterations(
    fold_results,
):
    if (
        FINAL_ITERATION_STRATEGY
        != "latest_fold"
    ):
        raise ValueError(
            "Unsupported final iteration strategy."
        )

    latest = max(
        fold_results,
        key=lambda x: x["validation_season"],
    )

    return int(
        latest["final_iterations"]
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. "
            "삭제하거나 EXPERIMENT_NAME을 바꾼 뒤 다시 실행하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. "
            "삭제하거나 EXPERIMENT_NAME을 바꾼 뒤 다시 실행하세요."
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    # -------------------------------------------------------------------------
    # canonical cb_reg_v003 feature schema
    # -------------------------------------------------------------------------
    # 중요:
    # 이전 cb_reg_v003_test와 달리 EXCLUDE_FEATURES를 두지 않는다.
    # 따라서 asof_pitcher_pitchmix_n도 그대로 포함된다.
    # -------------------------------------------------------------------------

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

    if (
        "asof_pitcher_pitchmix_n"
        not in base_features
    ):
        raise KeyError(
            "canonical v003에 필요한 "
            "'asof_pitcher_pitchmix_n'이 test schema에 없습니다."
        )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + base_features
            + [TARGET]
        ),
        low_memory=False,
    )

    # V18 correction용 필수 column 확인
    needed_for_v18 = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    missing_v18 = [
        c
        for c in needed_for_v18
        if c not in train.columns
    ]

    if missing_v18:
        raise KeyError(
            "V18 correction에 필요한 column이 없습니다: "
            + ", ".join(missing_v18)
        )

    print("=" * 80)
    print(f"Experiment          : {EXPERIMENT_NAME}")
    print("Base                : canonical cb_reg_v003")
    print("Model               : CatBoostRegressor")
    print("Objective           : RMSE")
    print("Validation          : Expanding Window")
    print("Feature Engineering : HB-EB")
    print("Residual correction : V18 gated")
    print(
        "pitchmix_n retained :",
        "asof_pitcher_pitchmix_n"
        in base_features,
    )
    print(f"Train shape         : {train.shape}")
    print(f"Base features       : {len(base_features)}")
    print("=" * 80)

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -------------------------------------------------------------------------
    # 1. canonical v003 expanding validation
    # -------------------------------------------------------------------------

    (
        fold_results,
        oof,
        final_features,
    ) = run_expanding_validation(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    final_iterations = (
        select_final_iterations(
            fold_results
        )
    )

    print()
    print("=" * 80)
    print("Validation summary")
    print("=" * 80)

    for fold in fold_results:
        print(
            f"{fold['validation_season']} | "
            f"Brier={fold['brier']:.8f} | "
            f"iter={fold['final_iterations']}"
        )

    print(
        f"Pooled OOF Brier : "
        f"{brier_score(oof['target'], oof['prediction']):.8f}"
    )

    print(
        f"Final iterations : {final_iterations}"
    )

    # -------------------------------------------------------------------------
    # 2. 2024 OOF -> final frozen V18 correction table
    # -------------------------------------------------------------------------

    v18_correction = (
        build_final_v18_correction(
            train=train,
            oof=oof,
        )
    )

    print()
    print("=" * 80)
    print("Final V18 correction")
    print("=" * 80)
    print(
        f"Source season : "
        f"{v18_correction['source_season']}"
    )
    print(
        f"Parent groups : "
        f"{len(v18_correction['parent_table']):,}"
    )
    print(
        f"Detail groups : "
        f"{len(v18_correction['detail_table']):,}"
    )
    print(
        f"Config        : {V18_CONFIG}"
    )

    # -------------------------------------------------------------------------
    # 3. Full train HB-EB
    # -------------------------------------------------------------------------

    df_full = train.copy()

    (
        df_full,
        final_features_check,
        final_priors,
    ) = transform_hbeb_pair(
        df_train=train,
        df_apply=df_full,
        base_features=base_features,
    )

    if (
        final_features_check
        != final_features
    ):
        raise ValueError(
            "Validation과 final training의 feature 순서가 다릅니다."
        )

    if (
        "asof_pitcher_pitchmix_n"
        not in final_features
    ):
        raise ValueError(
            "canonical v003 feature가 손실되었습니다."
        )

    X_full = prepare_features(
        df_full[final_features],
        cat_cols,
    )

    y_full = (
        df_full[TARGET]
        .astype(np.float64)
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(final_iterations),
    }

    final_model = CatBoostRegressor(
        **final_params
    )

    print()
    print("=" * 80)
    print("Final model training")
    print("=" * 80)

    start = time.time()

    final_model.fit(
        X_full,
        y_full,
        cat_features=cat_cols,
    )

    final_elapsed = (
        time.time()
        - start
    )

    # -------------------------------------------------------------------------
    # 4. Final artifact
    # -------------------------------------------------------------------------

    artifact = {
        "model": final_model,
        "hbeb_priors": final_priors,
        "features": list(final_features),
        "cat_cols": list(cat_cols),
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment": "cb_reg_v003",
        "model_type": "CatBoostRegressor",
        "objective": "RMSE",
        "final_iterations": int(final_iterations),
        "prediction_policy": (
            "base=np.clip(model.predict,0,1); "
            "then row-local frozen V18 gated residual correction; "
            "final=np.clip(base+correction,0,1)"
        ),
        "v18_gated_correction": v18_correction,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # reload sanity check
    loaded = joblib.load(
        MODEL_PATH
    )

    required_keys = [
        "model",
        "hbeb_priors",
        "features",
        "cat_cols",
        "v18_gated_correction",
    ]

    missing = [
        k
        for k in required_keys
        if k not in loaded
    ]

    if missing:
        raise RuntimeError(
            "저장 artifact key 누락: "
            + ", ".join(missing)
        )

    if (
        "asof_pitcher_pitchmix_n"
        not in loaded["features"]
    ):
        raise RuntimeError(
            "저장된 모델이 canonical v003 feature를 유지하지 못했습니다."
        )

    final_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment": "cb_reg_v003",
        "model_type": "CatBoostRegressor",
        "objective": "RMSE",
        "validation_method": "expanding_window",
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "final_iterations": int(final_iterations),
        "training_seasons": sorted(
            int(x)
            for x in train["season"].unique()
        ),
        "n_train": int(len(train)),
        "n_base_features": int(len(base_features)),
        "n_final_features": int(len(final_features)),
        "features": list(final_features),
        "categorical_features": list(cat_cols),
        "model_params": final_params,
        "v18_gated_config": V18_CONFIG,
        "v18_source_season": int(V18_SOURCE_SEASON),
        "v18_parent_groups": int(
            len(v18_correction["parent_table"])
        ),
        "v18_detail_groups": int(
            len(v18_correction["detail_table"])
        ),
        "uses_test_distribution": False,
        "row_independent_application": True,
        "elapsed_seconds": float(final_elapsed),
        "model_path": str(MODEL_PATH),
    }

    with open(
        MODEL_DIR / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 실행한 train.py도 같이 보관
    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR / "train.py",
        )
    except shutil.SameFileError:
        pass

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Model    : {MODEL_PATH}")
    print(
        f"Metadata : "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        "Canonical pitchmix_n retained:",
        "asof_pitcher_pitchmix_n"
        in loaded["features"],
    )
    print(
        "V18 correction enabled:",
        loaded[
            "v18_gated_correction"
        ]["enabled"],
    )
    print(
        f"Final train elapsed: {final_elapsed:.1f}s"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

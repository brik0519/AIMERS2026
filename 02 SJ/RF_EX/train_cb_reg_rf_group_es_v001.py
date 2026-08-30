# train.py
# ============================================================
# Experiment: cb_reg_rf_group_es_v001
#
# Control
#   - 전체 train(R+F) 학습
#   - 전체 validation(R+F)로 Early Stopping
#
# Treatment
#   - R model: 전체 train(R+F) 학습, validation R만 eval_set
#   - F model: 전체 train(R+F) 학습, validation F만 eval_set
#   - 각 validation row 자신의 game_type으로 R/F model routing
#
# 그대로 유지
#   - CatBoostRegressor / RMSE
#   - HB-EB
#   - asof_pitcher_pitchmix_n 제외
#   - Expanding Window 4-Fold
#   - random_seed=42 및 나머지 CatBoost hyperparameter
#
# 대회 독립행 규칙
#   - test.csv는 schema 확인용 header만 읽는다.
#   - HB-EB prior는 fold train / final full train에서만 추정한다.
#   - routing은 각 row 자신의 game_type만 사용한다.
# ============================================================

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


EXCLUDE_FEATURES = [
    "asof_pitcher_pitchmix_n",
]

EXPERIMENT_NAME = "cb_reg_rf_group_es_v001"

DATA_DIR = Path("./data")
MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
TEST_PATH = DATA_DIR / "test.csv"

ID = "row_id"
TARGET = "control_success"
GROUP_COL = "game_type"
R_GROUP = "R"
F_GROUP = "F"
ALLOWED_GROUPS = {R_GROUP, F_GROUP}

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
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {"fold": 4, "train_seasons": [2019, 2020, 2021, 2022, 2023], "validation_season": 2024},
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
FINAL_ITERATION_STRATEGY = "latest_fold_groupwise"


def to_python(obj):
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def compute_hbeb_priors(df_train, group_cols, rate_col):
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

    prior_df = stats[group_cols + ["alpha", "beta"]].copy()

    return prior_df, (float(g_alpha), float(g_beta))


def apply_hbeb(df, group_cols, n_col, rate_col, prior_df, global_prior):
    original_index = df.index

    lookup_df = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(prior_df, on=group_cols, how="left", sort=False)
    )

    alpha = lookup_df["alpha"].fillna(global_prior[0]).to_numpy(dtype=np.float64)
    beta = lookup_df["beta"].fillna(global_prior[1]).to_numpy(dtype=np.float64)
    n = df[n_col].fillna(0.0).to_numpy(dtype=np.float64)
    rate = df[rate_col].fillna(0.0).to_numpy(dtype=np.float64)

    success_count = n * rate
    denominator = n + alpha + beta
    eb_rate = (success_count + alpha) / denominator

    return pd.Series(eb_rate, index=original_index, dtype=np.float64)


def fit_hbeb_priors(df_train):
    priors = {}
    for config in HBEB_CONFIGS:
        prior_df, global_prior = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config["group_cols"],
            rate_col=config["rate_col"],
        )
        priors[config["new_col"]] = {
            "prior_df": prior_df,
            "global_prior": global_prior,
            "group_cols": list(config["group_cols"]),
            "n_col": config["n_col"],
            "rate_col": config["rate_col"],
        }
    return priors


def apply_hbeb_with_priors(df, base_features, priors):
    transformed = df.copy()
    current_features = list(base_features)

    for config in HBEB_CONFIGS:
        prior_info = priors[config["new_col"]]

        transformed[config["new_col"]] = apply_hbeb(
            df=transformed,
            group_cols=config["group_cols"],
            n_col=config["n_col"],
            rate_col=config["rate_col"],
            prior_df=prior_info["prior_df"],
            global_prior=prior_info["global_prior"],
        )

        if config["rate_col"] in current_features:
            current_features.remove(config["rate_col"])

        if config["new_col"] not in current_features:
            current_features.append(config["new_col"])

    return transformed, current_features


def prepare_features(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__").astype(str)
    return df


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(100000.0 * (1.0 - brier / baseline_brier))
        bss = float(max(0.0, bss_raw))

    return {
        "n": int(len(y_true)),
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": bss,
    }


def safe_group_metrics(df, prediction, group_value):
    mask = (
        df[GROUP_COL]
        .astype(str)
        .eq(group_value)
        .to_numpy()
    )

    if mask.sum() == 0:
        return {
            "n": 0,
            "positive_rate": None,
            "brier": None,
            "baseline_brier": None,
            "bss_raw": None,
            "bss": None,
        }

    return calculate_metrics(
        df.loc[mask, TARGET].to_numpy(dtype=np.float64),
        np.asarray(prediction, dtype=np.float64)[mask],
    )


def calculate_range_diagnostics(pred_raw):
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    below_zero = pred_raw < 0.0
    above_one = pred_raw > 1.0
    outside = below_zero | above_one

    return {
        "n_prediction_below_0": int(below_zero.sum()),
        "ratio_prediction_below_0": float(below_zero.mean()),
        "n_prediction_above_1": int(above_one.sum()),
        "ratio_prediction_above_1": float(above_one.mean()),
        "n_prediction_outside_0_1": int(outside.sum()),
        "ratio_prediction_outside_0_1": float(outside.mean()),
    }


def validate_game_type_values(df, context):
    values = set(
        df[GROUP_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    unexpected = values - ALLOWED_GROUPS
    if unexpected:
        raise ValueError(
            f"{context}: 예상하지 못한 game_type 발견: {sorted(unexpected)}"
        )

    if R_GROUP not in values or F_GROUP not in values:
        raise ValueError(
            f"{context}: R/F 둘 다 존재해야 합니다. actual={sorted(values)}"
        )


def fit_control_model(X_train, y_train, X_val, y_val, cat_cols):
    model = CatBoostRegressor(**MODEL_PARAMS)
    start = time.time()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    elapsed = time.time() - start
    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        final_iterations = int(MODEL_PARAMS["iterations"])
    else:
        final_iterations = int(best_iteration + 1)

    return model, best_iteration, final_iterations, elapsed


def fit_group_es_model(
    group_value,
    X_train,
    y_train,
    X_val,
    y_val,
    val_group_series,
    cat_cols,
):
    group_mask = (
        val_group_series
        .astype(str)
        .eq(group_value)
        .to_numpy()
    )

    if group_mask.sum() == 0:
        raise ValueError(
            f"Validation에 game_type={group_value} row가 없습니다."
        )

    X_val_group = X_val.loc[group_mask]
    y_val_group = y_val.loc[group_mask]

    model = CatBoostRegressor(**MODEL_PARAMS)
    start = time.time()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val_group, y_val_group),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    elapsed = time.time() - start
    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        final_iterations = int(MODEL_PARAMS["iterations"])
    else:
        final_iterations = int(best_iteration + 1)

    return model, best_iteration, final_iterations, elapsed, int(group_mask.sum())


def routed_group_prediction(r_model, f_model, X, group_series):
    groups = group_series.astype(str).to_numpy()
    r_mask = groups == R_GROUP
    f_mask = groups == F_GROUP

    if not (r_mask | f_mask).all():
        bad = sorted(set(groups[~(r_mask | f_mask)]))
        raise ValueError(f"Routing 불가능한 game_type: {bad}")

    pred_raw = np.empty(len(X), dtype=np.float64)

    if r_mask.any():
        pred_raw[r_mask] = r_model.predict(X.loc[r_mask])

    if f_mask.any():
        pred_raw[f_mask] = f_model.predict(X.loc[f_mask])

    return pred_raw


def run_fold(fold_config, train, base_features, cat_cols):
    fold = int(fold_config["fold"])
    train_seasons = list(fold_config["train_seasons"])
    validation_season = int(fold_config["validation_season"])

    train_mask = train["season"].isin(train_seasons)
    val_mask = train["season"] == validation_season

    df_train_raw = train.loc[train_mask].copy()
    df_val_raw = train.loc[val_mask].copy()

    validate_game_type_values(df_train_raw, context=f"Fold {fold} train")
    validate_game_type_values(df_val_raw, context=f"Fold {fold} validation")

    priors = fit_hbeb_priors(df_train_raw)

    df_train, current_features = apply_hbeb_with_priors(
        df=df_train_raw,
        base_features=base_features,
        priors=priors,
    )

    df_val, current_features_check = apply_hbeb_with_priors(
        df=df_val_raw,
        base_features=base_features,
        priors=priors,
    )

    if current_features != current_features_check:
        raise ValueError("Train/Validation Feature 불일치")

    X_train = prepare_features(df_train[current_features], cat_cols)
    y_train = df_train[TARGET].astype(np.float64)
    X_val = prepare_features(df_val[current_features], cat_cols)
    y_val = df_val[TARGET].astype(np.float64)
    val_group = df_val[GROUP_COL]

    print()
    print("=" * 90)
    print(
        f"Fold {fold}: {train_seasons[0]}~{train_seasons[-1]} "
        f"-> {validation_season}"
    )
    print("=" * 90)
    print(f"Train rows : {len(df_train):,}")
    print(f"Valid rows : {len(df_val):,}")
    print(
        "Valid R/F  : "
        f"R={(val_group.astype(str) == R_GROUP).sum():,}, "
        f"F={(val_group.astype(str) == F_GROUP).sum():,}"
    )

    # Control
    (
        control_model,
        control_best_iter,
        control_final_iter,
        control_elapsed,
    ) = fit_control_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_cols=cat_cols,
    )

    control_raw = np.asarray(control_model.predict(X_val), dtype=np.float64)
    control_pred = np.clip(control_raw, 0.0, 1.0)

    # Treatment: same full train, group-specific eval_set only
    (
        r_model,
        r_best_iter,
        r_final_iter,
        r_elapsed,
        n_val_r,
    ) = fit_group_es_model(
        group_value=R_GROUP,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_group_series=val_group,
        cat_cols=cat_cols,
    )

    (
        f_model,
        f_best_iter,
        f_final_iter,
        f_elapsed,
        n_val_f,
    ) = fit_group_es_model(
        group_value=F_GROUP,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_group_series=val_group,
        cat_cols=cat_cols,
    )

    treatment_raw = routed_group_prediction(
        r_model=r_model,
        f_model=f_model,
        X=X_val,
        group_series=val_group,
    )
    treatment_pred = np.clip(treatment_raw, 0.0, 1.0)

    control_all = calculate_metrics(y_val, control_pred)
    treatment_all = calculate_metrics(y_val, treatment_pred)
    control_r = safe_group_metrics(df_val, control_pred, R_GROUP)
    treatment_r = safe_group_metrics(df_val, treatment_pred, R_GROUP)
    control_f = safe_group_metrics(df_val, control_pred, F_GROUP)
    treatment_f = safe_group_metrics(df_val, treatment_pred, F_GROUP)

    result = {
        "fold": fold,
        "train_seasons": train_seasons,
        "validation_season": validation_season,
        "n_train": int(len(df_train)),
        "n_validation": int(len(df_val)),
        "control": {
            "metrics_all": control_all,
            "metrics_R": control_r,
            "metrics_F": control_f,
            "best_iteration": int(control_best_iter),
            "final_iterations": int(control_final_iter),
            "elapsed_seconds": float(control_elapsed),
            **calculate_range_diagnostics(control_raw),
        },
        "treatment": {
            "metrics_all": treatment_all,
            "metrics_R": treatment_r,
            "metrics_F": treatment_f,
            "R_best_iteration": int(r_best_iter),
            "R_final_iterations": int(r_final_iter),
            "F_best_iteration": int(f_best_iter),
            "F_final_iterations": int(f_final_iter),
            "R_eval_rows": int(n_val_r),
            "F_eval_rows": int(n_val_f),
            "R_elapsed_seconds": float(r_elapsed),
            "F_elapsed_seconds": float(f_elapsed),
            **calculate_range_diagnostics(treatment_raw),
        },
        "delta": {
            # 양수 = Treatment 개선
            "brier_gain_all": float(
                control_all["brier"] - treatment_all["brier"]
            ),
            "brier_gain_R": float(
                control_r["brier"] - treatment_r["brier"]
            ),
            "brier_gain_F": float(
                control_f["brier"] - treatment_f["brier"]
            ),
            "bss_gain_all": float(
                treatment_all["bss"] - control_all["bss"]
            ),
        },
    }

    fold_dir = VALIDATION_DIR / f"fold_{validation_season}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    pred_df = pd.DataFrame({
        "row_id": df_val[ID].to_numpy(),
        "target": y_val.to_numpy(),
        "game_type": df_val[GROUP_COL].astype(str).to_numpy(),
        "control_prediction_raw": control_raw,
        "control_prediction": control_pred,
        "treatment_prediction_raw": treatment_raw,
        "treatment_prediction": treatment_pred,
        "validation_season": validation_season,
    })

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
            to_python(result),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("[Fold 결과]")
    print(
        "Control   Brier : "
        f"{control_all['brier']:.10f} | iter={control_final_iter}"
    )
    print(
        "Treatment Brier : "
        f"{treatment_all['brier']:.10f} "
        f"| gain={result['delta']['brier_gain_all']:+.10f}"
    )
    print(
        "R Brier         : "
        f"{control_r['brier']:.10f} -> {treatment_r['brier']:.10f} "
        f"| gain={result['delta']['brier_gain_R']:+.10f}"
    )
    print(
        "F Brier         : "
        f"{control_f['brier']:.10f} -> {treatment_f['brier']:.10f} "
        f"| gain={result['delta']['brier_gain_F']:+.10f}"
    )
    print(
        "Iterations      : "
        f"Global={control_final_iter}, R={r_final_iter}, F={f_final_iter}"
    )

    return result, pred_df, current_features


def run_expanding_validation(train, base_features, cat_cols):
    fold_results = []
    oof_frames = []
    final_features = None

    for fold_config in EXPANDING_FOLDS:
        result, pred_df, current_features = run_fold(
            fold_config=fold_config,
            train=train,
            base_features=base_features,
            cat_cols=cat_cols,
        )

        if final_features is None:
            final_features = list(current_features)
        elif final_features != current_features:
            raise ValueError("Fold별 Feature 구성이 다릅니다.")

        fold_results.append(result)
        oof_frames.append(pred_df)

    oof = pd.concat(oof_frames, ignore_index=True)
    return fold_results, oof, final_features


def build_summary(fold_results, oof):
    y = oof["target"].to_numpy(dtype=np.float64)
    control_pred = oof["control_prediction"].to_numpy(dtype=np.float64)
    treatment_pred = oof["treatment_prediction"].to_numpy(dtype=np.float64)
    group = oof["game_type"].astype(str).to_numpy()

    control_all = calculate_metrics(y, control_pred)
    treatment_all = calculate_metrics(y, treatment_pred)

    def pooled_group_metrics(pred, value):
        mask = group == value
        return calculate_metrics(y[mask], pred[mask])

    control_r = pooled_group_metrics(control_pred, R_GROUP)
    treatment_r = pooled_group_metrics(treatment_pred, R_GROUP)
    control_f = pooled_group_metrics(control_pred, F_GROUP)
    treatment_f = pooled_group_metrics(treatment_pred, F_GROUP)

    fold_wins = sum(
        1
        for x in fold_results
        if x["delta"]["brier_gain_all"] > 0.0
    )
    fold_losses = sum(
        1
        for x in fold_results
        if x["delta"]["brier_gain_all"] < 0.0
    )

    recent_control = [
        x["control"]["metrics_all"]["brier"]
        for x in fold_results
        if x["validation_season"] in (2023, 2024)
    ]
    recent_treatment = [
        x["treatment"]["metrics_all"]["brier"]
        for x in fold_results
        if x["validation_season"] in (2023, 2024)
    ]

    return {
        "control_pooled": control_all,
        "treatment_pooled": treatment_all,
        "pooled_brier_gain": float(
            control_all["brier"] - treatment_all["brier"]
        ),
        "pooled_bss_gain": float(
            treatment_all["bss"] - control_all["bss"]
        ),
        "R": {
            "control": control_r,
            "treatment": treatment_r,
            "brier_gain": float(
                control_r["brier"] - treatment_r["brier"]
            ),
        },
        "F": {
            "control": control_f,
            "treatment": treatment_f,
            "brier_gain": float(
                control_f["brier"] - treatment_f["brier"]
            ),
        },
        "fold_wins": int(fold_wins),
        "fold_losses": int(fold_losses),
        "recent_2023_2024_control_mean_brier": float(np.mean(recent_control)),
        "recent_2023_2024_treatment_mean_brier": float(np.mean(recent_treatment)),
        "recent_2023_2024_brier_gain": float(
            np.mean(recent_control) - np.mean(recent_treatment)
        ),
    }


def select_group_final_iterations(fold_results):
    if FINAL_ITERATION_STRATEGY != "latest_fold_groupwise":
        raise ValueError(
            f"Unsupported strategy: {FINAL_ITERATION_STRATEGY}"
        )

    latest = max(
        fold_results,
        key=lambda x: x["validation_season"],
    )

    return {
        R_GROUP: int(latest["treatment"]["R_final_iterations"]),
        F_GROUP: int(latest["treatment"]["F_final_iterations"]),
    }


def fit_final_group_models(
    train,
    base_features,
    cat_cols,
    final_features_from_validation,
    group_iterations,
):
    final_priors = fit_hbeb_priors(train)

    df_full, final_features = apply_hbeb_with_priors(
        df=train,
        base_features=base_features,
        priors=final_priors,
    )

    if final_features != final_features_from_validation:
        raise ValueError("Validation/Final Feature 구성 불일치")

    X_full = prepare_features(df_full[final_features], cat_cols)
    y_full = df_full[TARGET].astype(np.float64)

    models = {}
    elapsed = {}

    for group_value in [R_GROUP, F_GROUP]:
        params = {
            **MODEL_PARAMS,
            "iterations": int(group_iterations[group_value]),
        }

        model = CatBoostRegressor(**params)

        print()
        print(
            f"[Final {group_value} model] 전체 train(R+F) 학습, "
            f"iterations={group_iterations[group_value]}"
        )

        start = time.time()
        model.fit(
            X_full,
            y_full,
            cat_features=cat_cols,
        )
        elapsed[group_value] = float(time.time() - start)
        models[group_value] = model

    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "CatBoostRegressor_RF_GroupES",
        "objective": "RMSE",
        "models": models,
        "hbeb_priors": final_priors,
        "features": final_features,
        "cat_cols": cat_cols,
        "group_col": GROUP_COL,
        "allowed_groups": [R_GROUP, F_GROUP],
        "group_iterations": group_iterations,
        "routing_policy": "row_game_type_only",
        "prediction_policy": (
            "predict corresponding R/F model then np.clip(raw,0,1)"
        ),
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "model_params": MODEL_PARAMS,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    return artifact, elapsed


def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. Change EXPERIMENT_NAME."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. Change EXPERIMENT_NAME."
        )

    # test 값/분포는 읽지 않고 header만 사용
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        c
        for c in test_cols
        if c != ID and c not in EXCLUDE_FEATURES
    ]

    if GROUP_COL not in base_features:
        raise ValueError(
            f"{GROUP_COL}이 Feature schema에 없습니다."
        )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[ID] + base_features + [TARGET],
    )

    validate_game_type_values(train, context="Full train")

    print("=" * 90)
    print(f"Experiment           : {EXPERIMENT_NAME}")
    print("Control              : Global Early Stopping")
    print("Treatment            : Shared Train + R/F Group Early Stopping")
    print("Model                : CatBoostRegressor")
    print("Objective            : RMSE")
    print("Feature Engineering  : HB-EB")
    print("Validation           : Expanding Window 4-Fold")
    print(f"Train shape          : {train.shape}")
    print(f"Base Feature 수      : {len(base_features)}")
    print(f"Categorical Feature  : {len(cat_cols)}")
    print(f"Excluded             : {EXCLUDE_FEATURES}")
    print("=" * 90)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    fold_results, oof, final_features = run_expanding_validation(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(
        fold_results,
        oof,
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_type": (
            "paired_ablation_global_es_vs_shared_train_rf_group_es"
        ),
        "validation_method": "expanding_window",
        "model_type": "CatBoostRegressor",
        "objective": "RMSE",
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "exclude_features": EXCLUDE_FEATURES,
        "hbeb_configs": HBEB_CONFIGS,
        "folds": fold_results,
        "summary": summary,
    }

    with open(
        VALIDATION_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(metadata),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 90)
    print("Expanding Window Summary")
    print("=" * 90)

    for result in fold_results:
        print(
            f"Valid {result['validation_season']} | "
            f"Control={result['control']['metrics_all']['brier']:.8f} | "
            f"Treatment={result['treatment']['metrics_all']['brier']:.8f} | "
            f"Gain={result['delta']['brier_gain_all']:+.8f} | "
            f"iters G={result['control']['final_iterations']}, "
            f"R={result['treatment']['R_final_iterations']}, "
            f"F={result['treatment']['F_final_iterations']}"
        )

    print()
    print(
        "Pooled Control Brier   : "
        f"{summary['control_pooled']['brier']:.10f}"
    )
    print(
        "Pooled Treatment Brier : "
        f"{summary['treatment_pooled']['brier']:.10f}"
    )
    print(
        "Pooled Brier Gain      : "
        f"{summary['pooled_brier_gain']:+.10f}"
    )
    print(
        "Pooled BSS Gain        : "
        f"{summary['pooled_bss_gain']:+.4f}"
    )
    print(
        "R Brier Gain           : "
        f"{summary['R']['brier_gain']:+.10f}"
    )
    print(
        "F Brier Gain           : "
        f"{summary['F']['brier_gain']:+.10f}"
    )
    print(
        "Recent 23~24 Gain      : "
        f"{summary['recent_2023_2024_brier_gain']:+.10f}"
    )
    print(
        "Fold wins/losses       : "
        f"{summary['fold_wins']}/{summary['fold_losses']}"
    )
    print("=" * 90)

    group_iterations = select_group_final_iterations(
        fold_results
    )

    print()
    print(
        "Final group iterations (latest fold): "
        f"R={group_iterations[R_GROUP]}, "
        f"F={group_iterations[F_GROUP]}"
    )

    artifact, final_elapsed = fit_final_group_models(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
        final_features_from_validation=final_features,
        group_iterations=group_iterations,
    )

    final_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "training_seasons": sorted(
            int(x)
            for x in train["season"].unique()
        ),
        "n_train": int(len(train)),
        "features": artifact["features"],
        "categorical_features": cat_cols,
        "group_col": GROUP_COL,
        "group_iterations": group_iterations,
        "routing_policy": "row_game_type_only",
        "elapsed_seconds": final_elapsed,
        "model_path": str(MODEL_PATH),
    }

    with open(
        MODEL_DIR / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(final_metadata),
            f,
            indent=2,
            ensure_ascii=False,
        )

    shutil.copy2(
        Path(__file__).resolve(),
        MODEL_DIR / "train.py",
    )

    print()
    print("=" * 90)
    print("Final Artifact 저장 완료")
    print(f"- Model    : {MODEL_PATH}")
    print(f"- Metadata : {MODEL_DIR / 'final_metadata.json'}")
    print(f"- Train.py : {MODEL_DIR / 'train.py'}")
    print("=" * 90)


if __name__ == "__main__":
    main()

#train.py
#Baseline: cb_reg_v003 + Exact-ASOF 5 features


from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Config
# ============================================================

EXPERIMENT_NAME = "cb_reg_v003_exact"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = Path("./model") / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_DIR = Path("./validation") / EXPERIMENT_NAME

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

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

EXACT_COLS = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
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


# ============================================================
# Common
# ============================================================

def prepare_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    brier = float(np.mean((y_pred - y_true) ** 2))
    positive_rate = float(np.mean(y_true))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0:
        bss_raw = 0.0
    else:
        bss_raw = float(
            100000.0 * (1.0 - brier / baseline_brier)
        )

    return {
        "brier": brier,
        "positive_rate": positive_rate,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": max(0.0, bss_raw),
    }


# ============================================================
# Exact-ASOF
# ============================================================

def build_pitcher_season_stats(history: pd.DataFrame) -> pd.DataFrame:
    """
    완료 시즌 train label만 저장한다.

    각 pitcher_id x season에 대해:
      n       = 공식 train 행 수
      success = target 합

    이후 어떤 행의 Exact-ASOF를 만들 때도
    반드시 row의 season보다 과거 시즌만 사용한다.
    """
    required = {
        "pitcher_id",
        SEASON_COL,
        TARGET,
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise KeyError(
            f"pitcher season stats 필수 컬럼 없음: {missing}"
        )

    stats = (
        history
        .groupby(
            ["pitcher_id", SEASON_COL],
            sort=False,
            dropna=False,
        )[TARGET]
        .agg(["size", "sum"])
        .reset_index()
        .rename(
            columns={
                "size": "n",
                "sum": "success",
            }
        )
    )

    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL],
        errors="raise",
    ).astype(int)
    stats["n"] = stats["n"].astype(np.float64)
    stats["success"] = stats["success"].astype(np.float64)

    return stats


def add_pitcher_exact_asof_features(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    각 row를 독립적으로 변환한다.

    official cumulative ASOF에서
    'row season보다 과거인 완료 시즌 train 통계'를 빼서
    현재 시즌 누적 상태를 복원한다.

    다른 평가 row는 절대 사용하지 않는다.
    """
    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 컬럼 없음: {missing}"
        )

    out = rows.copy()

    for col in EXACT_COLS:
        out[col] = np.nan

    row_season = pd.to_numeric(
        out[SEASON_COL],
        errors="raise",
    ).astype(int)

    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL],
        errors="raise",
    ).astype(int)

    for target_year in sorted(row_season.unique()):
        mask = row_season == int(target_year)
        part = out.loc[mask]

        hist = stats.loc[
            stats[SEASON_COL] < int(target_year)
        ]

        if len(hist):
            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[["n", "success"]]
                .sum()
            )
            hist_n = (
                part["pitcher_id"]
                .map(hist_agg["n"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            hist_success = (
                part["pitcher_id"]
                .map(hist_agg["success"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            hist_n = np.zeros(len(part), dtype=np.float64)
            hist_success = np.zeros(len(part), dtype=np.float64)

        asof_n = (
            pd.to_numeric(
                part["asof_pitcher_n"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64)
        )

        asof_rate = (
            pd.to_numeric(
                part["asof_pitcher_success_rate"],
                errors="coerce",
            )
            .to_numpy(dtype=np.float64)
        )

        asof_success = asof_n * asof_rate

        current_n = np.maximum(asof_n - hist_n, 0.0)
        current_success = asof_success - hist_success

        current_success = np.where(
            np.isfinite(current_success),
            np.clip(current_success, 0.0, current_n),
            np.nan,
        )

        current_rate = np.full(
            len(part),
            np.nan,
            dtype=np.float64,
        )
        valid_current = (
            (current_n > 0.0)
            & np.isfinite(current_success)
        )
        current_rate[valid_current] = (
            current_success[valid_current]
            / current_n[valid_current]
        )

        past_rate = np.full(
            len(part),
            np.nan,
            dtype=np.float64,
        )
        valid_past = hist_n > 0.0
        past_rate[valid_past] = (
            hist_success[valid_past]
            / hist_n[valid_past]
        )

        idx = part.index
        out.loc[idx, "exact_pitcher_season_n"] = current_n
        out.loc[idx, "exact_pitcher_season_logn"] = np.log1p(current_n)
        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = current_rate
        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = past_rate
        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = current_rate - past_rate

    return out


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train: pd.DataFrame,
    group_cols: list[str],
    rate_col: str,
):
    """
    canonical v003 HB-EB prior.
    반드시 fold-train / final full-train에서만 추정한다.
    """
    rate = pd.to_numeric(
        df_train[rate_col],
        errors="coerce",
    )

    g_mu = float(rate.mean())
    g_var = float(rate.var())

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

    tmp = df_train[group_cols].copy()
    tmp["_rate_"] = rate

    stats = (
        tmp
        .groupby(
            group_cols,
            dropna=False,
        )["_rate_"]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    group_max_var = stats["mean"] * (1.0 - stats["mean"])

    valid = (
        stats["var"].notna()
        & np.isfinite(stats["var"])
        & (stats["var"] > 0.0)
        & (stats["var"] < group_max_var)
        & (stats["count"] >= 30)
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

    if valid.any():
        mu = stats.loc[valid, "mean"]
        var = stats.loc[valid, "var"]
        m = (mu * (1.0 - mu) / var) - 1.0

        stats.loc[valid, "alpha"] = mu * m
        stats.loc[valid, "beta"] = (1.0 - mu) * m

    prior_df = stats[
        group_cols + ["alpha", "beta"]
    ].copy()

    return prior_df, (float(g_alpha), float(g_beta))


def apply_hbeb(
    df: pd.DataFrame,
    group_cols: list[str],
    n_col: str,
    rate_col: str,
    prior_df: pd.DataFrame,
    global_prior,
):
    """
    저장된 prior lookup + 해당 row의 n/rate만 사용.
    """
    lookup = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(
            prior_df,
            on=group_cols,
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
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.float64)
    )

    success_count = n * rate
    denominator = n + alpha + beta

    return np.divide(
        success_count + alpha,
        denominator,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def add_hbeb_features(
    df_train_for_prior: pd.DataFrame,
    df_apply: pd.DataFrame,
    features: list[str],
):
    out = df_apply.copy()
    final_features = list(features)
    priors = {}

    for cfg in HBEB_CONFIGS:
        prior_df, global_prior = compute_hbeb_priors(
            df_train=df_train_for_prior,
            group_cols=cfg["group_cols"],
            rate_col=cfg["rate_col"],
        )

        out[cfg["new_col"]] = apply_hbeb(
            df=out,
            group_cols=cfg["group_cols"],
            n_col=cfg["n_col"],
            rate_col=cfg["rate_col"],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if cfg["rate_col"] in final_features:
            final_features.remove(cfg["rate_col"])

        if cfg["new_col"] not in final_features:
            final_features.append(cfg["new_col"])

        priors[cfg["new_col"]] = {
            "prior_df": prior_df,
            "global_prior": global_prior,
            "group_cols": list(cfg["group_cols"]),
            "n_col": cfg["n_col"],
            "rate_col": cfg["rate_col"],
        }

    return out, final_features, priors


def apply_existing_hbeb_priors(
    df_apply: pd.DataFrame,
    features: list[str],
    priors: dict,
):
    out = df_apply.copy()
    final_features = list(features)

    for cfg in HBEB_CONFIGS:
        info = priors[cfg["new_col"]]

        out[cfg["new_col"]] = apply_hbeb(
            df=out,
            group_cols=info["group_cols"],
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=info["global_prior"],
        )

        if cfg["rate_col"] in final_features:
            final_features.remove(cfg["rate_col"])

        if cfg["new_col"] not in final_features:
            final_features.append(cfg["new_col"])

    return out, final_features


# ============================================================
# Validation
# ============================================================

def run_expanding_validation(
    train: pd.DataFrame,
    base_features: list[str],
    cat_cols: list[str],
):
    fold_rows = []
    oof_frames = []
    latest_final_features = None

    base_plus_exact = list(base_features) + [
        c for c in EXACT_COLS
        if c not in base_features
    ]

    for cfg in EXPANDING_FOLDS:
        fold = cfg["fold"]
        train_seasons = cfg["train_seasons"]
        val_year = cfg["validation_season"]

        df_tr_raw = train.loc[
            train[SEASON_COL].isin(train_seasons)
        ].copy()

        df_va_raw = train.loc[
            train[SEASON_COL] == val_year
        ].copy()

        # ----------------------------------------------------
        # Exact-ASOF
        # fold-train label로 만든 완료 시즌 통계만 사용
        # ----------------------------------------------------
        pitcher_stats = build_pitcher_season_stats(df_tr_raw)

        df_tr = add_pitcher_exact_asof_features(
            df_tr_raw,
            pitcher_stats,
        )
        df_va = add_pitcher_exact_asof_features(
            df_va_raw,
            pitcher_stats,
        )

        # ----------------------------------------------------
        # HB-EB
        # prior는 fold train에서만 fit
        # ----------------------------------------------------
        (
            df_tr,
            final_features,
            hbeb_priors,
        ) = add_hbeb_features(
            df_train_for_prior=df_tr_raw,
            df_apply=df_tr,
            features=base_plus_exact,
        )

        (
            df_va,
            final_features_check,
        ) = apply_existing_hbeb_priors(
            df_apply=df_va,
            features=base_plus_exact,
            priors=hbeb_priors,
        )

        if final_features_check != final_features:
            raise RuntimeError(
                "train/validation feature 목록 불일치"
            )

        if "asof_pitcher_pitchmix_n" not in final_features:
            raise RuntimeError(
                "canonical v003 feature "
                "'asof_pitcher_pitchmix_n'이 사라졌습니다."
            )

        fold_cat_cols = [
            c for c in cat_cols
            if c in final_features
        ]

        X_tr = prepare_features(
            df_tr[final_features],
            fold_cat_cols,
        )
        X_va = prepare_features(
            df_va[final_features],
            fold_cat_cols,
        )

        y_tr = df_tr[TARGET].astype(np.float64)
        y_va = df_va[TARGET].astype(np.float64)

        model = CatBoostRegressor(**MODEL_PARAMS)

        started = time.time()
        model.fit(
            X_tr,
            y_tr,
            cat_features=fold_cat_cols,
            eval_set=(X_va, y_va),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        elapsed = time.time() - started

        pred_raw = np.asarray(
            model.predict(X_va),
            dtype=np.float64,
        )
        pred = np.clip(pred_raw, 0.0, 1.0)

        metrics = calculate_metrics(y_va, pred)

        best_iter_zero = int(model.get_best_iteration())
        best_tree_count = (
            best_iter_zero + 1
            if best_iter_zero >= 0
            else int(model.tree_count_)
        )

        fold_rows.append({
            "fold": int(fold),
            "train_seasons": [int(x) for x in train_seasons],
            "validation_season": int(val_year),
            "n_train": int(len(df_tr)),
            "n_validation": int(len(df_va)),
            "brier": float(metrics["brier"]),
            "bss_raw": float(metrics["bss_raw"]),
            "bss": float(metrics["bss"]),
            "best_iteration_zero_based": best_iter_zero,
            "best_tree_count": int(best_tree_count),
            "elapsed_seconds": float(elapsed),
        })

        oof_frames.append(
            pd.DataFrame({
                ID: df_va[ID].to_numpy(),
                "target": y_va.to_numpy(dtype=np.float64),
                "prediction_raw": pred_raw,
                "prediction": pred,
                "validation_season": int(val_year),
            })
        )

        latest_final_features = list(final_features)

        print(
            f"[Fold {fold}] "
            f"{train_seasons} -> {val_year} | "
            f"Brier={metrics['brier']:.10f} | "
            f"BSS={metrics['bss_raw']:.2f} | "
            f"trees={best_tree_count}"
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    pooled = calculate_metrics(
        oof["target"],
        oof["prediction"],
    )

    return (
        fold_rows,
        oof,
        latest_final_features,
        pooled,
    )


# ============================================================
# Final training
# ============================================================

def train_final(
    train: pd.DataFrame,
    base_features: list[str],
    cat_cols: list[str],
    final_iterations: int,
):
    base_plus_exact = list(base_features) + [
        c for c in EXACT_COLS
        if c not in base_features
    ]

    # 2019~2024 완료 시즌 통계를 artifact에 저장.
    # 각 training row 계산에서는 row season보다 과거 시즌만 사용.
    pitcher_stats = build_pitcher_season_stats(train)

    df_full = add_pitcher_exact_asof_features(
        train,
        pitcher_stats,
    )

    (
        df_full,
        final_features,
        hbeb_priors,
    ) = add_hbeb_features(
        df_train_for_prior=train,
        df_apply=df_full,
        features=base_plus_exact,
    )

    if "asof_pitcher_pitchmix_n" not in final_features:
        raise RuntimeError(
            "canonical v003 feature "
            "'asof_pitcher_pitchmix_n'이 사라졌습니다."
        )

    final_cat_cols = [
        c for c in cat_cols
        if c in final_features
    ]

    X_full = prepare_features(
        df_full[final_features],
        final_cat_cols,
    )
    y_full = df_full[TARGET].astype(np.float64)

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(final_iterations),
    }

    model = CatBoostRegressor(**final_params)

    started = time.time()
    model.fit(
        X_full,
        y_full,
        cat_features=final_cat_cols,
    )
    elapsed = time.time() - started

    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model": model,
        "features": final_features,
        "cat_cols": final_cat_cols,
        "hbeb_priors": hbeb_priors,
        "pitcher_season_stats": pitcher_stats,
        "exact_cols": list(EXACT_COLS),
        "final_iterations": int(final_iterations),
        "model_params": final_params,
        "prediction_policy": "np.clip(raw_prediction, 0.0, 1.0)",
        "baseline_definition": (
            "canonical cb_reg_v003 + Exact-ASOF5; "
            "no residual, no conditional, no routing"
        ),
    }

    return artifact, elapsed


# ============================================================
# Main
# ============================================================

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보존을 위해 중단: {VALIDATION_DIR}"
        )
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"기존 model 보존을 위해 중단: {MODEL_DIR}"
        )

    # test는 schema(header)만 읽는다.
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    base_features = [
        c for c in test_cols
        if c != ID
    ]

    # 중요: 전에 삭제해서 LB가 크게 하락했던 canonical feature.
    # 이번 baseline에서는 반드시 유지한다.
    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError(
            "test schema에 asof_pitcher_pitchmix_n이 없습니다. "
            "canonical v003 기준선과 다른 schema입니다."
        )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[ID] + base_features + [TARGET],
    )

    required = {
        ID,
        TARGET,
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise KeyError(
            f"train 필수 컬럼 없음: {missing}"
        )

    cat_cols = [
        c for c in CAT_COLS
        if c in base_features
    ]

    print("=" * 88)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print("Baseline   : canonical cb_reg_v003 + Exact-ASOF 5")
    print("Residual   : NO")
    print("Conditional: NO")
    print("Routing    : NO")
    print("pitchmix_n : KEEP")
    print(f"Train rows : {len(train):,}")
    print(f"Base feats : {len(base_features)}")
    print("=" * 88)

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        fold_rows,
        oof,
        final_features,
        pooled,
    ) = run_expanding_validation(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_meta = {
        "experiment_name": EXPERIMENT_NAME,
        "validation_method": "expanding_window",
        "folds": fold_rows,
        "pooled": pooled,
        "final_features": final_features,
        "n_final_features": len(final_features),
        "exact_cols": EXACT_COLS,
        "hbeb_configs": HBEB_CONFIGS,
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
    }

    with open(
        VALIDATION_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            validation_meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 88)
    print("Expanding Window Summary")
    print("=" * 88)
    for row in fold_rows:
        print(
            f"{row['validation_season']} | "
            f"Brier={row['brier']:.10f} | "
            f"BSS={row['bss_raw']:.2f} | "
            f"trees={row['best_tree_count']}"
        )
    print("-" * 88)
    print(
        f"Pooled OOF Brier = {pooled['brier']:.10f}"
    )
    print(
        f"Pooled OOF BSS   = {pooled['bss_raw']:.2f}"
    )
    print("=" * 88)

    final_iterations = int(
        fold_rows[-1]["best_tree_count"]
    )

    print(
        f"Final iterations ({FINAL_ITERATION_STRATEGY}) "
        f"= {final_iterations}"
    )

    artifact, elapsed = train_final(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
        final_iterations=final_iterations,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    final_meta = {
        "experiment_name": EXPERIMENT_NAME,
        "baseline": "canonical cb_reg_v003 + Exact-ASOF5",
        "residual": False,
        "pitcher_conditional": False,
        "routing": False,
        "asof_pitcher_pitchmix_n_kept": True,
        "n_features": len(artifact["features"]),
        "features": artifact["features"],
        "exact_cols": EXACT_COLS,
        "final_iterations": final_iterations,
        "elapsed_seconds": float(elapsed),
        "pooled_oof_brier": float(pooled["brier"]),
        "pooled_oof_bss_raw": float(pooled["bss_raw"]),
    }

    with open(
        MODEL_DIR / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 재현성용 train.py 복사
    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR / "train.py",
        )
    except Exception:
        pass

    print()
    print("=" * 88)
    print("Final model saved")
    print(f"- {MODEL_PATH}")
    print(f"- features: {len(artifact['features'])}")
    print(
        "- asof_pitcher_pitchmix_n: "
        f"{'YES' if 'asof_pitcher_pitchmix_n' in artifact['features'] else 'NO'}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()

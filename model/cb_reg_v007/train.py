#train.py — cb_reg_v007
#Baseline: cb_reg_v003_exact 와 동일 파이프라인 (Exact-ASOF 5 + HB-EB 2, 동일 파라미터).
#변경점: 트리 수 선택에서 검증 연도 정보를 제거한다 (낙관 편향 제거 결정 실험).
#  - 각 fold: fold-train 마지막 시즌을 내부 eval 로 early stopping -> 트리 수 확정(<=T-1 정보만)
#            -> fold-train 전체로 고정 트리 수 재적합 -> 검증 연도는 eval_set 없이 예측
#  - 최종 모델: 2019~2023 학습 + 2024 내부 eval 로 트리 수 확정 -> 2019~2024 재적합
#검증 연도 라벨은 채점에만 쓰이므로 OOF 점수와 블렌드 이득에 낙관 편향이 없다.
#실행 말미: validation/shift_v004a/matrices 가 있으면 anchor 대비 배포모사 블렌드 분석을
#수행하고 blend_analysis.json 에 판정(2024 deploy delta_bp >= +2.0)을 기록한다.


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

EXPERIMENT_NAME = "cb_reg_v007"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
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

EARLY_STOPPING_ROUNDS = 80  # inner holdout 에서만 사용
FINAL_ITERATION_STRATEGY = "inner_holdout"
MIN_TREES = 10  # 내부 eval 붕괴(극소 트리) 방어 하한. 원시 값은 metadata 에 함께 기록.

# 블렌드 분석용 v004a anchor 경로 (없으면 분석 생략)
V004A_MATRICES = Path("./validation") / "shift_v004a" / "matrices"
BLEND_YEARS = [2021, 2022, 2023, 2024]
BLEND_DECISION_THRESHOLD_BP = 2.0


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

def build_fit_matrices(
    df_tr_raw: pd.DataFrame,
    df_va_raw: pd.DataFrame,
    base_plus_exact: list[str],
    cat_cols: list[str],
):
    """fold-train 통계만으로 train/eval 행렬을 만든다 (v003_exact 파이프라인과 동일)."""
    pitcher_stats = build_pitcher_season_stats(df_tr_raw)

    df_tr = add_pitcher_exact_asof_features(df_tr_raw, pitcher_stats)
    df_va = add_pitcher_exact_asof_features(df_va_raw, pitcher_stats)

    df_tr, final_features, hbeb_priors = add_hbeb_features(
        df_train_for_prior=df_tr_raw,
        df_apply=df_tr,
        features=base_plus_exact,
    )
    df_va, final_features_check = apply_existing_hbeb_priors(
        df_apply=df_va,
        features=base_plus_exact,
        priors=hbeb_priors,
    )

    if final_features_check != final_features:
        raise RuntimeError("train/validation feature 목록 불일치")
    if "asof_pitcher_pitchmix_n" not in final_features:
        raise RuntimeError(
            "canonical v003 feature 'asof_pitcher_pitchmix_n'이 사라졌습니다."
        )

    fold_cat_cols = [c for c in cat_cols if c in final_features]
    X_tr = prepare_features(df_tr[final_features], fold_cat_cols)
    X_va = prepare_features(df_va[final_features], fold_cat_cols)
    y_tr = df_tr[TARGET].astype(np.float64)
    y_va = df_va[TARGET].astype(np.float64)
    return X_tr, y_tr, X_va, y_va, final_features, fold_cat_cols


def select_iterations_inner(
    train: pd.DataFrame,
    train_seasons: list[int],
    base_plus_exact: list[str],
    cat_cols: list[str],
):
    """검증 연도 정보 없이 트리 수를 고른다.

    train_seasons 의 마지막 시즌을 내부 eval 로 떼어 early stopping 으로 트리 수를
    확정한다. 사용 정보가 전부 train_seasons(<= T-1) 이므로 deploy regime 에서 합법이며,
    검증/평가 연도의 라벨·분포는 어떤 형태로도 개입하지 않는다.
    """
    if len(train_seasons) < 2:
        raise ValueError(f"inner holdout 에는 시즌 2개 이상 필요: {train_seasons}")

    inner_train_seasons = [int(s) for s in train_seasons[:-1]]
    inner_eval_season = int(train_seasons[-1])

    df_itr = train.loc[train[SEASON_COL].isin(inner_train_seasons)].copy()
    df_iev = train.loc[train[SEASON_COL] == inner_eval_season].copy()

    X_tr, y_tr, X_ev, y_ev, _, cat_fold = build_fit_matrices(
        df_itr, df_iev, base_plus_exact, cat_cols
    )

    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_fold,
        eval_set=(X_ev, y_ev),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    best_zero = int(model.get_best_iteration())
    raw_trees = best_zero + 1 if best_zero >= 0 else int(model.tree_count_)
    fixed_trees = max(int(raw_trees), MIN_TREES)

    return fixed_trees, {
        "inner_train_seasons": inner_train_seasons,
        "inner_eval_season": inner_eval_season,
        "inner_best_tree_count_raw": int(raw_trees),
        "fixed_tree_count": int(fixed_trees),
    }


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
        # (v007) 트리 수 확정 — 검증 연도(val_year) 정보 미사용
        # ----------------------------------------------------
        n_trees, inner_info = select_iterations_inner(
            train,
            [int(s) for s in train_seasons],
            base_plus_exact,
            cat_cols,
        )

        # fold-train 전체 통계로 행렬 구성 (v003_exact 와 동일)
        (
            X_tr,
            y_tr,
            X_va,
            y_va,
            final_features,
            fold_cat_cols,
        ) = build_fit_matrices(
            df_tr_raw,
            df_va_raw,
            base_plus_exact,
            cat_cols,
        )

        fold_params = {**MODEL_PARAMS, "iterations": int(n_trees)}
        model = CatBoostRegressor(**fold_params)

        started = time.time()
        model.fit(
            X_tr,
            y_tr,
            cat_features=fold_cat_cols,
            # eval_set 없음 — 검증 연도는 채점에만 사용된다.
        )
        elapsed = time.time() - started

        pred_raw = np.asarray(
            model.predict(X_va),
            dtype=np.float64,
        )
        pred = np.clip(pred_raw, 0.0, 1.0)

        metrics = calculate_metrics(y_va, pred)

        best_tree_count = int(model.tree_count_)

        fold_rows.append({
            "fold": int(fold),
            "train_seasons": [int(x) for x in train_seasons],
            "validation_season": int(val_year),
            "n_train": int(len(df_tr_raw)),
            "n_validation": int(len(df_va_raw)),
            "brier": float(metrics["brier"]),
            "bss_raw": float(metrics["bss_raw"]),
            "bss": float(metrics["bss"]),
            "best_tree_count": int(best_tree_count),
            **inner_info,
            "elapsed_seconds": float(elapsed),
        })

        oof_frames.append(
            pd.DataFrame({
                ID: df_va_raw[ID].to_numpy(),
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

def run_blend_analysis(oof: pd.DataFrame):
    """v004a anchor 대비 배포모사 블렌드 분석.

    w 는 전년도 연내 최적값을 그대로 다음 해에 적용한다(deploy regime).
    v007 의 OOF 에는 검증 연도 정보가 없으므로 이 표의 delta 에 낙관 편향이 없다.
    """
    if not V004A_MATRICES.exists():
        print(f"[blend] {V004A_MATRICES} 없음 — 분석 생략")
        return None

    per_year = []
    hold = {}
    for year in BLEND_YEARS:
        d = np.load(V004A_MATRICES / f"data_{year}.npz", allow_pickle=True)
        a = np.load(V004A_MATRICES / f"anchor_{year}.npz")
        ids = d["row_id"].astype(str)
        sub = oof.loc[oof["validation_season"] == year].set_index(ID)
        q = sub.reindex(ids)
        ok = q["prediction"].notna().to_numpy()
        y = d["target"].astype(np.float64)[ok]
        pa = np.clip(a["p_base"].astype(np.float64)[ok], 0.0, 1.0)
        pc = q["prediction"].to_numpy(np.float64)[ok]

        base_var = float(y.mean() * (1.0 - y.mean()))

        def bp(p, _y=y, _v=base_var):
            return float((1.0 - np.mean((p - _y) ** 2) / _v) * 1e4)

        ws = np.linspace(0.0, 1.0, 101)
        briers = [float(np.mean(((1.0 - w) * pa + w * pc - y) ** 2)) for w in ws]
        w_best = float(ws[int(np.argmin(briers))])

        per_year.append({
            "year": int(year),
            "n": int(ok.sum()),
            "match_ratio": float(ok.mean()),
            "anchor_bp": bp(pa),
            "v007_bp": bp(pc),
            "w_best_inyear": w_best,
            "blend_bp_inyear": bp((1.0 - w_best) * pa + w_best * pc),
        })
        hold[int(year)] = (pa, pc, y, base_var)

    deploy = []
    for prev, cur in zip(per_year[:-1], per_year[1:]):
        w = prev["w_best_inyear"]
        pa, pc, y, base_var = hold[cur["year"]]
        blend_bp = float(
            (1.0 - np.mean(((1.0 - w) * pa + w * pc - y) ** 2) / base_var) * 1e4
        )
        deploy.append({
            "year": cur["year"],
            "w_from_year": prev["year"],
            "w": w,
            "anchor_bp": cur["anchor_bp"],
            "blend_bp": blend_bp,
            "delta_bp": blend_bp - cur["anchor_bp"],
        })

    decision_delta = deploy[-1]["delta_bp"]
    result = {
        "reference": "validation/shift_v004a/matrices anchor p_base",
        "per_year": per_year,
        "deploy_regime": deploy,
        "decision_rule": (
            f"최근 전이(2023->2024) delta_bp >= +{BLEND_DECISION_THRESHOLD_BP} 이면 "
            "2단계(feature 귀속 ablation) 진행, 아니면 방향 폐기"
        ),
        "decision_2024_delta_bp": decision_delta,
        "decision_pass": bool(decision_delta >= BLEND_DECISION_THRESHOLD_BP),
    }

    print()
    print("=" * 88)
    print("Blend analysis vs shift_v004a anchor (skill bp, 낙관 편향 제거 후)")
    print("=" * 88)
    for r in per_year:
        print(
            f"{r['year']} | anchor={r['anchor_bp']:+8.2f} | v007={r['v007_bp']:+8.2f} | "
            f"w*={r['w_best_inyear']:.2f} | inyear_blend={r['blend_bp_inyear']:+8.2f}"
        )
    print("-" * 88)
    for r in deploy:
        print(
            f"deploy {r['year']} (w={r['w']:.2f} from {r['w_from_year']}) | "
            f"anchor={r['anchor_bp']:+8.2f} | blend={r['blend_bp']:+8.2f} | "
            f"delta={r['delta_bp']:+.2f}"
        )
    print("-" * 88)
    print(
        f"DECISION: 2024 delta = {decision_delta:+.2f} bp -> "
        f"{'PASS (2단계 진행)' if result['decision_pass'] else 'FAIL (방향 폐기 권고)'}"
    )
    print("=" * 88)
    return result


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

    blend_result = run_blend_analysis(oof)
    if blend_result is not None:
        with open(
            VALIDATION_DIR / "blend_analysis.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                blend_result,
                f,
                ensure_ascii=False,
                indent=2,
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
        "iteration_selection": (
            "inner holdout: fold-train 마지막 시즌 eval 로 트리 수 확정, "
            "검증 연도 정보 미사용"
        ),
        "min_trees_floor": MIN_TREES,
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

    base_plus_exact = list(base_features) + [
        c for c in EXACT_COLS
        if c not in base_features
    ]
    all_seasons = sorted(
        pd.to_numeric(train[SEASON_COL], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    final_iterations, final_inner_info = select_iterations_inner(
        train,
        [int(s) for s in all_seasons],
        base_plus_exact,
        cat_cols,
    )
    final_iterations = int(final_iterations)

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
        "final_inner_info": final_inner_info,
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
# train.py: cb_reg_v014
# CONTROL vs A0_ZERO vs A1_NAN vs A2_ZERO_FLAG
# cb_reg_v003-compatible + strictly-past pitcher_id x batter_hand conditional delta ablation

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

EXPERIMENT_NAME = "cb_reg_v014"
DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_DIR = Path("./model") / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
METADATA_PATH = MODEL_DIR / "final_metadata.json"
VALIDATION_DIR = Path("./validation") / EXPERIMENT_NAME

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

BASE_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
]

EXPANDING_FOLDS = [
    ([2019, 2020], 2021),
    ([2019, 2020, 2021], 2022),
    ([2019, 2020, 2021, 2022], 2023),
    ([2019, 2020, 2021, 2022, 2023], 2024),
]

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}
EARLY_STOPPING_ROUNDS = 80

HBEB_CONFIGS = [
    {"group_cols": ["pitcher_hand"], "n_col": "asof_pitcher_n", "rate_col": "asof_pitcher_success_rate", "new_col": "hbeb_pitcher_success_rate"},
    {"group_cols": ["batter_hand"], "n_col": "asof_batter_n", "rate_col": "asof_batter_success_rate", "new_col": "hbeb_batter_success_rate"},
]
HBEB_MIN_GROUP_ROWS = 30

CF_KEY_COLS = ["pitcher_id", "batter_hand"]
CF_PARENT_COLS = ["pitcher_id"]
CF_K = 80.0
CF_CANONICAL = [
    "cf_platoon_delta_nan",
    "cf_platoon_delta_zero",
    "cf_platoon_observed",
    "cf_platoon_reliability",
    "cf_platoon_n",
]

ARMS = {
    "CONTROL": [],
    "A0_ZERO": ["cf_platoon_delta_zero"],
    "A1_NAN": ["cf_platoon_delta_nan"],
    "A2_ZERO_FLAG": ["cf_platoon_delta_zero", "cf_platoon_observed"],
}
AUDIT_SEASONS = [2021, 2022, 2023, 2024]
AUDIT_TOLERANCE = 1e-12
PARITY_FIXTURE_ROWS = 20000
PARITY_FIXTURE_SEED = 20260826


def brier_score(y, p):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    return float(np.mean((p - y) ** 2))


def baseline_brier(y):
    y = np.asarray(y, np.float64)
    r = float(y.mean())
    return float(np.mean((y - r) ** 2))


def bss_raw(y, p):
    bs = brier_score(y, p)
    base = baseline_brier(y)
    return 0.0 if base <= 0 else 1e5 * (base - bs) / base


def ensure_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns: {missing}")


def sanitize_categoricals(df):
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
    return out


def json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


# ---------------- HB-EB: cb_reg_v003-compatible row-level MoM ----------------
def _safe_mom_beta(mu, var):
    mu = float(np.clip(mu, 1e-6, 1 - 1e-6))
    k = mu * (1 - mu)
    if (not np.isfinite(var)) or var <= 0 or var >= k:
        var = 0.5 * k
    m = k / var - 1.0
    if (not np.isfinite(m)) or m <= 0:
        m = 1.0
    return mu * m, (1 - mu) * m


def compute_hbeb_priors(df_train, group_cols, rate_col):
    rate = pd.to_numeric(df_train[rate_col], errors="coerce")
    valid = rate.notna()
    ga, gb = _safe_mom_beta(float(rate[valid].mean()), float(rate[valid].var(ddof=1)))
    work = df_train.loc[valid, group_cols].copy()
    work[rate_col] = rate.loc[valid].to_numpy()
    stats = work.groupby(group_cols, dropna=False, observed=True)[rate_col].agg(["mean", "var", "count"]).reset_index()
    alphas, betas, oks = [], [], []
    for row in stats.itertuples(index=False):
        mu, var, count = float(row.mean), float(row.var), int(row.count)
        ok = count >= HBEB_MIN_GROUP_ROWS and np.isfinite(var) and var > 0 and np.isfinite(mu) and 0 < mu < 1
        a, b = _safe_mom_beta(mu, var) if ok else (ga, gb)
        alphas.append(a); betas.append(b); oks.append(bool(ok))
    prior_df = stats[group_cols].copy()
    prior_df["alpha"] = alphas
    prior_df["beta"] = betas
    prior_df["valid_group_prior"] = oks
    return prior_df, {"alpha": float(ga), "beta": float(gb), "mean": float(ga/(ga+gb)), "strength": float(ga+gb)}


def apply_hbeb(df, cfg, prior_df, global_prior):
    lookup = df[cfg["group_cols"]].reset_index(drop=True).merge(prior_df, on=cfg["group_cols"], how="left", sort=False)
    alpha = pd.to_numeric(lookup["alpha"], errors="coerce").fillna(global_prior["alpha"]).to_numpy(np.float64)
    beta = pd.to_numeric(lookup["beta"], errors="coerce").fillna(global_prior["beta"]).to_numpy(np.float64)
    n = pd.to_numeric(df[cfg["n_col"]], errors="coerce").fillna(0).to_numpy(np.float64)
    rate = pd.to_numeric(df[cfg["rate_col"]], errors="coerce").fillna(0).to_numpy(np.float64)
    n = np.clip(n, 0, None); rate = np.clip(rate, 0, 1)
    denom = n + alpha + beta
    if np.any(denom <= 0):
        raise ValueError(f"HB-EB denominator <= 0: {cfg['rate_col']}")
    return pd.Series((n*rate + alpha)/denom, index=df.index, dtype=np.float64)


def apply_hbeb_configs(df_train, df_apply, priors=None):
    out = df_apply.copy()
    features = list(BASE_FEATURES)
    out_priors = {}
    for cfg in HBEB_CONFIGS:
        new_col = cfg["new_col"]
        if priors is None:
            prior_df, global_prior = compute_hbeb_priors(df_train, cfg["group_cols"], cfg["rate_col"])
        else:
            prior_df = priors[new_col]["prior_df"].copy()
            global_prior = dict(priors[new_col]["global_prior"])
        out[new_col] = apply_hbeb(out, cfg, prior_df, global_prior)
        if cfg["rate_col"] in features: features.remove(cfg["rate_col"])
        if new_col not in features: features.append(new_col)
        out_priors[new_col] = {"config": dict(cfg), "prior_df": prior_df, "global_prior": global_prior}
    return out, features, out_priors


# ---------------- Conditional bank ----------------
def build_platoon_bank(history_df):
    ensure_columns(history_df, [SEASON_COL, TARGET] + CF_KEY_COLS, "build_platoon_bank")
    if len(history_df) == 0:
        return {
            "cell_table": pd.DataFrame(columns=CF_KEY_COLS + ["cell_n", "cell_success"]),
            "parent_table": pd.DataFrame(columns=CF_PARENT_COLS + ["parent_n", "parent_success", "parent_rate"]),
            "global_rate": 0.5, "n_history": 0, "history_seasons": [], "strength": CF_K,
        }
    y = pd.to_numeric(history_df[TARGET], errors="raise").astype(np.float64)
    tmp = history_df[CF_KEY_COLS].copy(); tmp[TARGET] = y.to_numpy()
    cell = tmp.groupby(CF_KEY_COLS, dropna=False, observed=True)[TARGET].agg(cell_n="size", cell_success="sum").reset_index()
    parent = history_df[CF_PARENT_COLS].assign(__y=y.to_numpy()).groupby(CF_PARENT_COLS, dropna=False, observed=True)["__y"].agg(parent_n="size", parent_success="sum").reset_index()
    parent["parent_rate"] = parent["parent_success"] / parent["parent_n"].clip(lower=1)
    return {
        "cell_table": cell, "parent_table": parent, "global_rate": float(y.mean()),
        "n_history": int(len(history_df)),
        "history_seasons": sorted(int(x) for x in pd.unique(history_df[SEASON_COL])),
        "strength": CF_K,
    }


def apply_platoon_bank(df_apply, bank):
    ensure_columns(df_apply, CF_KEY_COLS, "apply_platoon_bank")
    out = df_apply.copy(); n_rows = len(out)
    global_rate, k = float(bank["global_rate"]), float(bank["strength"])
    parent_lookup = out[CF_PARENT_COLS].reset_index(drop=True).merge(bank["parent_table"], on=CF_PARENT_COLS, how="left", sort=False)
    parent_rate = pd.to_numeric(parent_lookup.get("parent_rate", pd.Series(np.nan, index=range(n_rows))), errors="coerce").fillna(global_rate).to_numpy(np.float64)
    cell_lookup = out[CF_KEY_COLS].reset_index(drop=True).merge(bank["cell_table"], on=CF_KEY_COLS, how="left", sort=False)
    cell_n = pd.to_numeric(cell_lookup.get("cell_n", pd.Series(np.nan, index=range(n_rows))), errors="coerce").fillna(0).to_numpy(np.float64)
    cell_success = pd.to_numeric(cell_lookup.get("cell_success", pd.Series(np.nan, index=range(n_rows))), errors="coerce").fillna(0).to_numpy(np.float64)
    cell_n = np.clip(cell_n, 0, None); cell_success = np.clip(cell_success, 0, None)
    observed = (cell_n > 0).astype(np.float64)
    posterior = (cell_success + k*parent_rate) / (cell_n + k)
    reliability = cell_n / (cell_n + k)
    raw_delta = posterior - parent_rate
    delta_nan = np.where(observed > 0, raw_delta, np.nan)
    delta_zero = np.where(observed > 0, raw_delta, 0.0)
    out["cf_platoon_delta_nan"] = delta_nan
    out["cf_platoon_delta_zero"] = delta_zero
    out["cf_platoon_observed"] = observed
    out["cf_platoon_reliability"] = reliability
    out["cf_platoon_n"] = cell_n
    finite = np.isfinite(delta_nan)
    diag = {
        "n_rows": int(n_rows), "history_seasons": list(bank.get("history_seasons", [])),
        "history_n": int(bank.get("n_history", 0)), "strength": k, "global_rate": global_rate,
        "unseen_cell_rows": int(np.sum(observed == 0)), "unseen_cell_rate": float(np.mean(observed == 0)) if n_rows else 0.0,
        "nan_delta_rows": int(np.sum(~finite)), "nan_delta_rate": float(np.mean(~finite)) if n_rows else 0.0,
        "observed_rate": float(np.mean(observed)) if n_rows else 0.0,
        "mean_reliability": float(np.mean(reliability)) if n_rows else 0.0,
        "mean_delta": float(np.mean(delta_nan[finite])) if finite.any() else 0.0,
        "std_delta": float(np.std(delta_nan[finite])) if finite.any() else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(delta_nan[finite]))) if finite.any() else 0.0,
    }
    return out, diag


def cold_start_features(df_apply):
    out = df_apply.copy(); n = len(out)
    out["cf_platoon_delta_nan"] = np.nan
    out["cf_platoon_delta_zero"] = 0.0
    out["cf_platoon_observed"] = 0.0
    out["cf_platoon_reliability"] = 0.0
    out["cf_platoon_n"] = 0.0
    return out, {
        "n_rows": int(n), "history_seasons": [], "history_n": 0, "strength": CF_K, "global_rate": 0.5,
        "unseen_cell_rows": int(n), "unseen_cell_rate": 1.0 if n else 0.0,
        "nan_delta_rows": int(n), "nan_delta_rate": 1.0 if n else 0.0,
        "observed_rate": 0.0, "mean_reliability": 0.0, "mean_delta": 0.0, "std_delta": 0.0, "mean_abs_delta": 0.0,
    }


def build_strictly_past_features(df):
    ensure_columns(df, [SEASON_COL, TARGET] + CF_KEY_COLS, "build_strictly_past_features")
    season_int = df[SEASON_COL].astype(int)
    pieces, diags = [], []
    for season in sorted(int(x) for x in pd.unique(df[SEASON_COL])):
        current = df.loc[season_int == season].copy()
        history = df.loc[season_int < season]
        transformed, diag = cold_start_features(current) if len(history) == 0 else apply_platoon_bank(current, build_platoon_bank(history))
        diag = dict(diag); diag["target_season"] = int(season)
        pieces.append(transformed); diags.append(diag)
    result = pd.concat(pieces).sort_index().reindex(df.index)
    return result, diags


def build_validation_features(history_df, valid_df):
    years = sorted(int(x) for x in pd.unique(valid_df[SEASON_COL]))
    if len(years) != 1: raise ValueError(f"validation_df must contain one season: {years}")
    vy = years[0]
    if bool((history_df[SEASON_COL].astype(int) >= vy).any()):
        raise RuntimeError("Leakage guard: validation/future season in history_df")
    bank = build_platoon_bank(history_df)
    out, diag = apply_platoon_bank(valid_df, bank); diag["target_season"] = vy
    return out, diag, bank


# ---------------- audits ----------------
def feature_max_abs_diff(a_df, b_df, idx):
    max_diff = 0.0
    for c in CF_CANONICAL:
        a = a_df.loc[idx, c].to_numpy(np.float64); b = b_df.loc[idx, c].to_numpy(np.float64)
        an, bn = np.isnan(a), np.isnan(b)
        if np.any(an ^ bn): return float("inf")
        finite = ~an
        if finite.any(): max_diff = max(max_diff, float(np.max(np.abs(a[finite]-b[finite]))))
    return max_diff


def verify_future_invariance(audit_df, year, full_feat):
    trunc = audit_df.loc[audit_df[SEASON_COL].astype(int) <= int(year)].copy()
    trunc_feat, _ = build_strictly_past_features(trunc)
    idx = trunc.index[trunc[SEASON_COL].astype(int) == int(year)]
    d = feature_max_abs_diff(full_feat, trunc_feat, idx)
    return {"year": int(year), "n_rows": int(len(idx)), "max_abs_diff": d, "passed": bool(d <= AUDIT_TOLERANCE)}


def verify_current_label_permutation(audit_df, year, full_feat, seed=123):
    shuf = audit_df.copy(); mask = shuf[SEASON_COL].astype(int) == int(year)
    rng = np.random.default_rng(seed); y = shuf.loc[mask, TARGET].to_numpy(copy=True); rng.shuffle(y); shuf.loc[mask, TARGET] = y
    shuf_feat, _ = build_strictly_past_features(shuf)
    d = feature_max_abs_diff(full_feat, shuf_feat, audit_df.index[mask])
    return {"year": int(year), "n_rows": int(mask.sum()), "max_abs_diff": d, "passed": bool(d <= AUDIT_TOLERANCE)}


def verify_row_independence(frame, bank, n_samples=30, seed=42):
    if len(frame) == 0: return {"passed": True, "max_abs_diff": 0.0, "failures": []}
    rng = np.random.default_rng(seed); positions = rng.choice(len(frame), size=min(n_samples, len(frame)), replace=False)
    batch, _ = apply_platoon_bank(frame, bank); max_diff = 0.0; failures = []
    for pos in positions:
        pos = int(pos); one, _ = apply_platoon_bank(frame.iloc[[pos]].copy(), bank)
        for c in CF_CANONICAL:
            a, b = float(batch.iloc[pos][c]), float(one.iloc[0][c])
            if np.isnan(a) and np.isnan(b): continue
            if not np.isclose(a, b, atol=AUDIT_TOLERANCE, rtol=0, equal_nan=True): failures.append({"position": pos, "feature": c, "batch": a, "single": b})
            elif np.isfinite(a) and np.isfinite(b): max_diff = max(max_diff, abs(a-b))
    return {"passed": len(failures)==0, "max_abs_diff": float(max_diff), "failures": failures, "n_samples": int(len(positions))}


def fit_arm(arm, train_df, valid_df, hbeb_features):
    features = list(hbeb_features) + ARMS[arm]
    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(train_df[features], train_df[TARGET], cat_features=CAT_COLS,
              eval_set=(valid_df[features], valid_df[TARGET]),
              early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True, verbose=100)
    best = int(model.get_best_iteration()); pred_raw = model.predict(valid_df[features]); pred = np.clip(pred_raw, 0, 1)
    y = valid_df[TARGET].to_numpy(np.float64)
    return {
        "model": model, "features": features, "pred_raw": pred_raw, "pred": pred,
        "metadata": {
            "arm": arm, "features": features, "conditional_features": ARMS[arm],
            "best_iteration": best, "final_iterations": best+1,
            "brier": brier_score(y,pred), "baseline_brier": baseline_brier(y), "bss_raw": bss_raw(y,pred),
            "prediction_mean": float(np.mean(pred)), "prediction_std": float(np.std(pred)),
            "feature_importance": dict(zip(features, model.get_feature_importance().tolist())),
        }
    }


def representation_breakdown(year, y, control_pred, arm_results, valid_df):
    observed = valid_df["cf_platoon_observed"].to_numpy(np.float64) > 0.5
    rows = []
    for group, mask in {"all": np.ones(len(y),bool), "seen": observed, "unseen": ~observed}.items():
        if not mask.any(): continue
        cbs = brier_score(y[mask], control_pred[mask])
        for arm in ["A0_ZERO","A1_NAN","A2_ZERO_FLAG"]:
            p = arm_results[arm]["pred"]; abs_ = brier_score(y[mask], p[mask])
            rows.append({
                "validation_season": int(year), "group": group, "arm": arm,
                "n_rows": int(mask.sum()), "share": float(mask.mean()), "actual_mean": float(y[mask].mean()),
                "control_pred_mean": float(control_pred[mask].mean()), "arm_pred_mean": float(p[mask].mean()),
                "control_brier": cbs, "arm_brier": abs_, "delta_brier_control_minus_arm": cbs-abs_,
            })
    return pd.DataFrame(rows)


def train_fold(df, train_seasons, valid_year):
    season_int = df[SEASON_COL].astype(int)
    tr_raw = df.loc[season_int.isin(train_seasons)].copy(); va_raw = df.loc[season_int == valid_year].copy()
    tr_h, hbeb_features, priors = apply_hbeb_configs(tr_raw, tr_raw)
    va_h, va_features, _ = apply_hbeb_configs(tr_raw, va_raw, priors=priors)
    if hbeb_features != va_features: raise RuntimeError("HB-EB feature manifest mismatch")
    tr_cf, tr_diag = build_strictly_past_features(tr_h)
    va_cf, va_diag, bank = build_validation_features(tr_raw, va_h)
    row_check = verify_row_independence(va_h.head(min(5000,len(va_h))), bank)
    if not row_check["passed"]: raise RuntimeError(f"row independence failed: {row_check['failures'][:3]}")
    tr_cf, va_cf = sanitize_categoricals(tr_cf), sanitize_categoricals(va_cf)
    arm_results = {}
    for arm in ARMS:
        print(f"\n[{arm}]")
        arm_results[arm] = fit_arm(arm, tr_cf, va_cf, hbeb_features)
        m = arm_results[arm]["metadata"]
        print(f"Brier={m['brier']:.8f} BSS={m['bss_raw']:.2f} iter={m['final_iterations']}")
    control = arm_results["CONTROL"]["metadata"]
    for arm in ["A0_ZERO","A1_NAN","A2_ZERO_FLAG"]:
        m = arm_results[arm]["metadata"]
        m["delta_brier_vs_control"] = control["brier"] - m["brier"]
        m["delta_bss_vs_control"] = m["bss_raw"] - control["bss_raw"]
    fold_dir = VALIDATION_DIR / f"fold_{valid_year}"; fold_dir.mkdir(parents=True, exist_ok=True)
    y = va_cf[TARGET].to_numpy(np.float64)
    for arm, res in arm_results.items():
        cols = {
            "season": int(valid_year), "target": y, "pred_raw": res["pred_raw"], "pred": res["pred"],
            "cf_platoon_delta_nan": va_cf["cf_platoon_delta_nan"].to_numpy(),
            "cf_platoon_delta_zero": va_cf["cf_platoon_delta_zero"].to_numpy(),
            "cf_platoon_observed": va_cf["cf_platoon_observed"].to_numpy(),
            "cf_platoon_reliability": va_cf["cf_platoon_reliability"].to_numpy(),
            "cf_platoon_n": va_cf["cf_platoon_n"].to_numpy(),
        }
        cols = ({ID: va_cf[ID].to_numpy(), **cols} if ID in va_cf.columns else {"row_index": va_cf.index.to_numpy(), **cols})
        res["pred_df"] = pd.DataFrame(cols)
        res["pred_df"].to_csv(fold_dir/f"pred_{arm}.csv", index=False, encoding="utf-8-sig")
    breakdown = representation_breakdown(valid_year, y, arm_results["CONTROL"]["pred"], arm_results, va_cf)
    breakdown.to_csv(fold_dir/"representation_breakdown.csv", index=False)
    meta = {
        "experiment_name": EXPERIMENT_NAME, "train_seasons": [int(x) for x in train_seasons], "validation_season": int(valid_year),
        "n_train": int(len(tr_cf)), "n_validation": int(len(va_cf)), "hbeb_features": hbeb_features,
        "conditional_train_diagnostics": tr_diag, "conditional_validation_diagnostics": va_diag,
        "conditional_row_independence": row_check,
        "arms": {a:r["metadata"] for a,r in arm_results.items()},
    }
    write_json(fold_dir/"metadata.json", meta)
    return {"metadata": meta, "arm_results": arm_results, "breakdown": breakdown}


def fit_final_arm(arm, final_df, hbeb_features, iterations):
    features = list(hbeb_features) + ARMS[arm]
    params = dict(MODEL_PARAMS); params["iterations"] = int(iterations)
    model = CatBoostRegressor(**params)
    model.fit(final_df[features], final_df[TARGET], cat_features=CAT_COLS, verbose=100)
    return {"model": model, "features": features, "iterations": int(iterations)}


def main():
    start = time.time()
    if VALIDATION_DIR.exists() or MODEL_DIR.exists():
        raise FileExistsError(f"{VALIDATION_DIR} or {MODEL_DIR} exists. Change EXPERIMENT_NAME.")
    VALIDATION_DIR.mkdir(parents=True); MODEL_DIR.mkdir(parents=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    ensure_columns(df, BASE_FEATURES+[TARGET], "train.csv")
    if TEST_PATH.exists():
        test_cols = pd.read_csv(TEST_PATH, nrows=0, encoding="utf-8-sig").columns
        expected = set(test_cols)-{ID}
        if set(BASE_FEATURES) != expected:
            raise RuntimeError(f"BASE_FEATURES schema mismatch: only_train={sorted(set(BASE_FEATURES)-expected)}, only_test={sorted(expected-set(BASE_FEATURES))}")

    print("="*96)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print("Ablation   : CONTROL vs A0_ZERO vs A1_NAN vs A2_ZERO_FLAG")
    print(f"Train      : {df.shape}")
    print(f"K          : {CF_K}")
    print("="*96)

    # Leakage audit once
    audit_df = df[list(dict.fromkeys([SEASON_COL,TARGET]+CF_KEY_COLS))].copy()
    full_feat, _ = build_strictly_past_features(audit_df)
    audit_rows = []
    for year in AUDIT_SEASONS:
        inv = verify_future_invariance(audit_df, year, full_feat)
        perm = verify_current_label_permutation(audit_df, year, full_feat)
        audit_rows.append({"season":year,"future_diff":inv["max_abs_diff"],"future_pass":inv["passed"],"perm_diff":perm["max_abs_diff"],"perm_pass":perm["passed"]})
        if not inv["passed"] or not perm["passed"]: raise RuntimeError(f"Leakage audit failed: {year}, {inv}, {perm}")
    pd.DataFrame(audit_rows).to_csv(VALIDATION_DIR/"conditional_leakage_audit.csv", index=False)
    print("[Leakage audit] PASS")
    del audit_df, full_feat

    fold_results, breakdowns = [], []
    oof_parts = {a:[] for a in ARMS}
    latest_iters = {}
    for i,(train_years,valid_year) in enumerate(EXPANDING_FOLDS,1):
        print("\n"+"="*96)
        print(f"Fold {i}: {min(train_years)}~{max(train_years)} -> {valid_year}")
        print("="*96)
        r = train_fold(df, train_years, valid_year)
        fold_results.append(r["metadata"]); breakdowns.append(r["breakdown"])
        ctrl = r["arm_results"]["CONTROL"]["metadata"]
        print("\n[Fold comparison]")
        for arm in ARMS:
            m = r["arm_results"][arm]["metadata"]
            print(f"{arm:12s} BSS={m['bss_raw']:.2f} Δ={m['bss_raw']-ctrl['bss_raw']:+.2f} iter={m['final_iterations']}")
            oof_parts[arm].append(r["arm_results"][arm]["pred_df"])
            latest_iters[arm] = int(m["final_iterations"])

    summaries = {}
    for arm,parts in oof_parts.items():
        oof = pd.concat(parts, ignore_index=True)
        oof.to_csv(VALIDATION_DIR/f"oof_{arm}.csv", index=False, encoding="utf-8-sig")
        summaries[arm] = {"pooled_brier":brier_score(oof["target"],oof["pred"]), "pooled_bss":bss_raw(oof["target"],oof["pred"]), "n_oof":int(len(oof))}
    pd.concat(breakdowns, ignore_index=True).to_csv(VALIDATION_DIR/"representation_breakdown.csv", index=False)

    summary_rows=[]
    for fm in fold_results:
        cy=fm["arms"]["CONTROL"]
        for arm,m in fm["arms"].items():
            summary_rows.append({"validation_season":fm["validation_season"],"arm":arm,"brier":m["brier"],"bss":m["bss_raw"],"final_iterations":m["final_iterations"],"delta_brier_vs_control":cy["brier"]-m["brier"],"delta_bss_vs_control":m["bss_raw"]-cy["bss_raw"]})
    pd.DataFrame(summary_rows).to_csv(VALIDATION_DIR/"summary.csv", index=False)

    print("\n"+"="*96)
    print("Pooled OOF")
    print("="*96)
    cbss=summaries["CONTROL"]["pooled_bss"]
    for arm,s in summaries.items(): print(f"{arm:12s} Brier={s['pooled_brier']:.8f} BSS={s['pooled_bss']:.2f} Δ={s['pooled_bss']-cbss:+.2f}")

    # Final features and 3 candidate models
    final_h, hbeb_features, final_priors = apply_hbeb_configs(df, df)
    final_cf, final_diag = build_strictly_past_features(final_h)
    final_cf = sanitize_categoricals(final_cf)
    final_models = {}
    for arm in ["A0_ZERO","A1_NAN","A2_ZERO_FLAG"]:
        print(f"\n[Final {arm}] iterations={latest_iters[arm]}")
        final_models[arm] = fit_final_arm(arm, final_cf, hbeb_features, latest_iters[arm])

    final_bank = build_platoon_bank(df)
    row_check = verify_row_independence(df.head(min(5000,len(df))), final_bank, n_samples=40)
    if not row_check["passed"]: raise RuntimeError(f"Final row independence failed: {row_check['failures'][:3]}")

    # parity fixture
    latest_season = int(df[SEASON_COL].max())
    fixture = df.loc[df[SEASON_COL].astype(int)==latest_season].sample(n=min(PARITY_FIXTURE_ROWS, int((df[SEASON_COL].astype(int)==latest_season).sum())), random_state=PARITY_FIXTURE_SEED).sort_index()
    fixture_feat, fixture_diag = apply_platoon_bank(fixture, final_bank)
    key = fixture[ID].to_numpy() if ID in fixture.columns else fixture.index.to_numpy(); key_name = ID if ID in fixture.columns else "row_index"
    ft = fixture[BASE_FEATURES].copy(); ft.insert(0,key_name,key); ft.to_csv(VALIDATION_DIR/"parity_fixture_test.csv", index=False, encoding="utf-8-sig")
    fe = pd.DataFrame({key_name:key})
    for c in CF_CANONICAL: fe[c]=fixture_feat[c].to_numpy()
    fe.to_csv(VALIDATION_DIR/"parity_fixture_expected.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "experiment_name":EXPERIMENT_NAME, "model_type":"CatBoostRegressor", "objective":"RMSE",
        "models":{a:x["model"] for a,x in final_models.items()},
        "features_by_arm":{a:x["features"] for a,x in final_models.items()},
        "iterations_by_arm":{a:x["iterations"] for a,x in final_models.items()},
        "arms":ARMS, "cat_cols":CAT_COLS, "base_features":BASE_FEATURES,
        "hbeb_priors":final_priors, "hbeb_configs":HBEB_CONFIGS,
        "conditional_features":{"type":"strictly_past_pitcher_x_batter_hand","key_cols":CF_KEY_COLS,"parent_cols":CF_PARENT_COLS,"strength":CF_K,"canonical_columns":CF_CANONICAL,"final_2025_bank":final_bank,"row_independence":row_check},
        "prediction_policy":"clip_to_0_1", "training_seasons":sorted(int(x) for x in pd.unique(df[SEASON_COL]))
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)

    final_nan=sum(x["nan_delta_rows"] for x in final_diag); final_rows=sum(x["n_rows"] for x in final_diag)
    metadata = {
        "experiment_name":EXPERIMENT_NAME, "validation_method":"expanding_window",
        "ablation":"CONTROL_vs_A0_ZERO_vs_A1_NAN_vs_A2_ZERO_FLAG",
        "model_params":MODEL_PARAMS, "early_stopping_rounds":EARLY_STOPPING_ROUNDS,
        "folds":fold_results, "oof_summary":summaries, "final_iterations_by_arm":{a:latest_iters[a] for a in ["A0_ZERO","A1_NAN","A2_ZERO_FLAG"]},
        "conditional_formula":"delta = n/(n+K)*(cell_rate-parent_rate)",
        "conditional_strength":CF_K, "final_train_nan_delta_rate":float(final_nan/max(final_rows,1)),
        "final_train_conditional_diagnostics":final_diag,
        "final_2025_bank":{"n_history":final_bank["n_history"],"history_seasons":final_bank["history_seasons"],"n_cells":int(len(final_bank["cell_table"])),"n_parents":int(len(final_bank["parent_table"])),"global_rate":float(final_bank["global_rate"]),"strength":CF_K},
        "final_row_independence":row_check, "parity_fixture":{"season":latest_season,"n_rows":int(len(fixture)),"diagnostics":fixture_diag},
        "elapsed_seconds":float(time.time()-start),
    }
    write_json(METADATA_PATH, metadata)
    try: shutil.copy2(Path(__file__).resolve(), MODEL_DIR/"train.py")
    except Exception: pass
    print("\n"+"="*96)
    print("완료")
    print(f"Model      : {MODEL_PATH}")
    print(f"Validation : {VALIDATION_DIR}")
    print("핵심 판정: A0 vs A1 = unseen representation, A0 vs A2 = observed flag 증분")
    print("representation_breakdown.csv에서 seen/unseen ΔBrier를 반드시 확인")
    print("="*96)


if __name__ == "__main__":
    main()

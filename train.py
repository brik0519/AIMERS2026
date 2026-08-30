"""rresid_v002 — Ridge representation retest on the frozen Treatment OOF bank.

Why this experiment exists
--------------------------
rresid_v001 established that the existing ExtraTrees-adapter 49-column matrix,
when fed to a standardized Ridge, did not transfer R residual Shape robustly.
That does NOT falsify the Ridge idea used by shift_v002/v003 because that layer
used a different 25-column sensitivity representation and scale=0.75.

This audit therefore reuses the already-generated rresid_v001 Treatment OOF
residual bank and compares four strictly-next-year correction designs:

  R0 CONSTANT_S05
     previous-year mean R residual * 0.50 (Level reference)

  R1 RIDGE_CODED49_S05
     exact reproduction of rresid_v001 Ridge representation:
       adapter numeric + integer category codes + base_pred + Exact-ASOF5
       StandardScaler -> Ridge(alpha=1000, fit_intercept=False)
       centered residual target, scale=0.50

  R2 RIDGE_OHE_S05
     Ridge-native rewrite of the same adapter information:
       genuine numeric/engineered features + base_pred + Exact-ASOF5
       small categorical features -> source-fitted OneHotEncoder
       BOTH numeric and one-hot columns source-centered/scaled
       Ridge(alpha=1000, fit_intercept=False)
       centered residual target, scale=0.50

  R3 RIDGE_SENS25_SHIFT_ALLR_S075
     shift_v002/v003 sensitivity representation:
       25 sensitivity columns
       last <=3 labeled seasons, decay=0.65
       pitcher×batter_hand and pitcher×pressure×batter_hand EB deltas/reliability
       count one-hot indicators
       source standardization and clip[-8,8]
       Ridge(alpha=1000, solver='lsqr', fit_intercept=False)
       RAW source residual target, scale=0.75
     NOTE: original shift_v002/v003 applied this only to route_segment=CORE.
     This experiment intentionally applies it to the current R branch
     (game_type != 'F') because danker R does not use the shift router.  The
     feature/model construction is shift-faithful; the routing is R-branch-specific.

  R4 RIDGE_SENS25_REFERENCE_ALLR_S075
     same 25 features but fit_intercept=True.  This mirrors the shift reference
     model and diagnoses whether any gain is only previous-year Level transfer.

Temporal protocol
-----------------
For target t in [2021,2022,2023,2024]:
  fit residual correction ONLY on Treatment OOF residuals from t-1
  apply to Treatment OOF predictions in t.

No final model is trained. No test rows are used.

Fast execution
--------------
The expensive CatBoost OOF stage is NOT repeated. By default this script reads:
  ./validation/rresid_v001/residual_bank.csv
You can override it with RRESID_BANK_SOURCE=/path/to/residual_bank.csv.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Paths / identifiers
# =============================================================================

EXPERIMENT_NAME = "rresid_v002"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"  # intentionally not written

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

DEFAULT_BANK_PATH = VALIDATION_ROOT / "rresid_v001" / "residual_bank.csv"
BANK_SOURCE_PATH = Path(os.environ.get("RRESID_BANK_SOURCE", str(DEFAULT_BANK_PATH)))

RIDGE_METRICS_PATH = VALIDATION_DIR / "ridge_metrics.csv"
RIDGE_PREDICTIONS_PATH = VALIDATION_DIR / "ridge_predictions.csv"
YEAR_SUMMARY_PATH = VALIDATION_DIR / "year_summary.csv"
SEED_SUMMARY_PATH = VALIDATION_DIR / "seed_summary.csv"
META_PATH = VALIDATION_DIR / "final_metadata.json"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

EVAL_YEARS = [2021, 2022, 2023, 2024]
RIDGE_ALPHA = 1000.0
COMMON_SHRINK = 0.50
SHIFT_SCALE = 0.75

# shift_v002/v003 sensitivity constants
SENS_HISTORY_YEARS = 3
SENS_DECAY = 0.65
SENS_PARENT_STRENGTH = 100.0
SENS_HAND_STRENGTH = 35.0
SENS_PRESSURE_HAND_STRENGTH = 45.0

PRIMARY_ARMS = [
    "CONSTANT_S05",
    "RIDGE_CODED49_S05",
    "RIDGE_OHE_S05",
    "RIDGE_SENS25_SHIFT_ALLR_S075",
    "RIDGE_SENS25_REFERENCE_ALLR_S075",
]


# =============================================================================
# Existing adapter information set (rresid_v001 / cb_reg_v003_exact_pc_l2)
# =============================================================================

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

REFERENCE_RAWNUM = [
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "pitcher_team_id",
    "batter_team_id",
]

REFERENCE_CAT_SMALL = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

ENG_COLS = [
    "eng_recent35",
    "eng_recent_delta",
    "eng_sb",
    "eng_same",
    "eng_close",
    "eng_late",
]


# =============================================================================
# Generic metrics
# =============================================================================


def brier_score(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def error_decomposition(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    err = p - y
    level = float(np.mean(err) ** 2)
    total = float(np.mean(err ** 2))
    shape = float(total - level)
    return {"total": total, "level": level, "shape": shape}


def evaluate_arm(y, base, corr, scale, arm, target_year, source_year, seed):
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    corr = np.asarray(corr, dtype=np.float64)
    pred = np.clip(base + float(scale) * corr, 0.0, 1.0)

    b0 = error_decomposition(y, base)
    b1 = error_decomposition(y, pred)
    rate = float(np.mean(y))
    baseline = rate * (1.0 - rate)
    if baseline <= 0:
        raise RuntimeError("R baseline Brier <= 0")

    gains = {
        key: 100000.0 * (b0[key] - b1[key]) / baseline
        for key in ["total", "level", "shape"]
    }
    if abs(gains["total"] - gains["level"] - gains["shape"]) > 1e-6:
        raise RuntimeError("Total != Level + Shape")

    row = {
        "arm": arm,
        "validation_year": int(target_year),
        "source_year": int(source_year),
        "base_seed": int(seed),
        "scale": float(scale),
        "n_R": int(len(y)),
        "R_total_gain_pt": float(gains["total"]),
        "R_level_gain_pt": float(gains["level"]),
        "R_shape_gain_pt": float(gains["shape"]),
        "total_win": bool(gains["total"] > 1e-8),
        "shape_win": bool(gains["shape"] > 1e-6),
        "base_brier": float(b0["total"]),
        "candidate_brier": float(b1["total"]),
        "correction_mean": float(np.mean(corr)),
        "correction_std": float(np.std(corr)),
        "correction_abs_mean": float(np.mean(np.abs(corr))),
    }
    return row, pred


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


# =============================================================================
# Exact-ASOF5
# =============================================================================


def build_pitcher_season_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df_train.groupby(["pitcher_id", SEASON_COL], dropna=False, sort=False)[TARGET]
        .agg(n="size", success="sum")
        .reset_index()
    )
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)
    stats["n"] = stats["n"].astype(np.float64)
    stats["success"] = stats["success"].astype(np.float64)
    return stats


def add_pitcher_exact_asof_features(df, pitcher_season_stats):
    out = df.copy()
    for col in EXACT_FEATURES:
        out[col] = np.nan

    seasons = pd.to_numeric(out[SEASON_COL], errors="raise").astype(int)
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(seasons.unique()):
        row_mask = seasons.eq(int(target_year))
        rows = out.loc[row_mask]
        hist = stats.loc[stats[SEASON_COL] < int(target_year)]

        if len(hist):
            hist_agg = hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]].sum()
            hist_n = rows["pitcher_id"].map(hist_agg["n"]).fillna(0.0).to_numpy(np.float64)
            hist_success = rows["pitcher_id"].map(hist_agg["success"]).fillna(0.0).to_numpy(np.float64)
        else:
            hist_n = np.zeros(len(rows), dtype=np.float64)
            hist_success = np.zeros(len(rows), dtype=np.float64)

        asof_n = (
            pd.to_numeric(rows["asof_pitcher_n"], errors="coerce")
            .fillna(0.0).clip(lower=0.0).to_numpy(np.float64)
        )
        asof_rate = pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce").to_numpy(np.float64)
        raw_n = asof_n - hist_n
        current_n = np.maximum(raw_n, 0.0)
        raw_success = asof_n * asof_rate - hist_success
        current_success = np.where(
            np.isfinite(raw_success), np.clip(raw_success, 0.0, current_n), np.nan
        )

        current_rate = np.full(len(rows), np.nan, dtype=np.float64)
        ok = (current_n > 0) & np.isfinite(current_success)
        current_rate[ok] = current_success[ok] / current_n[ok]

        past_rate = np.full(len(rows), np.nan, dtype=np.float64)
        okp = hist_n > 0
        past_rate[okp] = hist_success[okp] / hist_n[okp]

        idx = rows.index
        out.loc[idx, "exact_pitcher_season_n"] = current_n
        out.loc[idx, "exact_pitcher_season_logn"] = np.log1p(current_n)
        out.loc[idx, "exact_pitcher_season_success_rate"] = current_rate
        out.loc[idx, "exact_pitcher_past_success_rate"] = past_rate
        out.loc[idx, "exact_pitcher_season_minus_past"] = current_rate - past_rate

    return out


def exact5_matrix(rows, pitcher_season_stats):
    t = add_pitcher_exact_asof_features(rows, pitcher_season_stats)
    return (
        t[EXACT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float64)
        .reset_index(drop=True)
    )


# =============================================================================
# Existing adapter matrix (coded-49 reference)
# =============================================================================


def numeric_series(frame, col, default=np.nan):
    if col not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=np.float64), index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def add_eng(frame):
    x = frame.copy()
    career = numeric_series(x, "asof_pitcher_success_rate", 0.5)
    p1 = numeric_series(x, "asof_pitcher_prev1_game_success_rate", np.nan).fillna(career)
    p3 = numeric_series(x, "asof_pitcher_prev3_game_success_rate", np.nan).fillna(p1)
    p5 = numeric_series(x, "asof_pitcher_prev5_game_success_rate", np.nan).fillna(p3)
    x["eng_recent35"] = 0.5 * p1 + 0.3 * p3 + 0.2 * p5
    x["eng_recent_delta"] = x["eng_recent35"] - career
    x["eng_sb"] = numeric_series(x, "asof_pitcher_strike_rate") - numeric_series(x, "asof_pitcher_ball_rate")
    x["eng_same"] = x["pitcher_hand"].astype(str).eq(x["batter_hand"].astype(str)).astype(np.float64)
    x["eng_close"] = numeric_series(x, "score_diff_pitcher_team").abs().le(1).astype(np.float64)
    x["eng_late"] = numeric_series(x, "inning").ge(7).astype(np.float64)
    return x


def fit_adapter_prep(rows):
    m = add_eng(rows.reset_index(drop=True))
    numeric_cols = [c for c in REFERENCE_RAWNUM if c in m.columns] + ENG_COLS
    med = (
        m[numeric_cols].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).median().fillna(0.0).to_dict()
    )
    cat_cols = [c for c in REFERENCE_CAT_SMALL if c in m.columns]
    maps = {}
    for col in cat_cols:
        s = m[col].astype("string").fillna("<NA>").astype(str)
        maps[col] = {v: i for i, v in enumerate(pd.Index(s.unique()))}
    return {"numeric_cols": numeric_cols, "med": med, "maps": maps}


def transform_adapter_core(prep, rows, base_pred):
    r = add_eng(rows.reset_index(drop=True))
    A = r[prep["numeric_cols"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).copy()
    for col in prep["numeric_cols"]:
        A[col] = A[col].fillna(float(prep["med"].get(col, 0.0))).astype(np.float64)
    for col, mapping in prep["maps"].items():
        A["code_" + col] = (
            r[col].astype("string").fillna("<NA>").astype(str)
            .map(mapping).fillna(-1).astype(np.float64)
        )
    A["base_pred"] = np.asarray(base_pred, dtype=np.float64)
    return A.reset_index(drop=True)


def coded49_matrices(source_rows, target_rows, source_base, target_base, pitcher_stats):
    prep = fit_adapter_prep(source_rows)
    cs = transform_adapter_core(prep, source_rows, source_base)
    ct = transform_adapter_core(prep, target_rows, target_base)
    es = exact5_matrix(source_rows, pitcher_stats)
    et = exact5_matrix(target_rows, pitcher_stats)
    Xs = pd.concat([cs, es], axis=1)
    Xt = pd.concat([ct, et], axis=1)
    if list(Xs.columns) != list(Xt.columns) or Xs.shape[1] != 49:
        raise RuntimeError(f"coded49 mismatch: {Xs.shape}, {Xt.shape}")
    return Xs, Xt


def fit_predict_ridge_coded49(Xs, residual, Xt):
    y = np.asarray(residual, dtype=np.float64)
    yc = y - y.mean()
    scaler = StandardScaler()
    Az = scaler.fit_transform(Xs.to_numpy(np.float64))
    Bz = scaler.transform(Xt.to_numpy(np.float64))
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, solver="lsqr", max_iter=200, tol=1e-5)
    model.fit(Az, yc)
    return np.asarray(model.predict(Bz), np.float64), {
        "n_features": int(Az.shape[1]),
        "source_corr_mean": float(np.mean(model.predict(Az))),
        "coef_l2": float(np.linalg.norm(model.coef_)),
    }


# =============================================================================
# Ridge-native OHE representation of the adapter information
# =============================================================================


def _make_ohe(*, dense=True):
    # sklearn >=1.2 uses sparse_output; fallback keeps this script portable.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense, dtype=np.float64)


def ohe_ridge_matrices(source_rows, target_rows, source_base, target_base, pitcher_stats):
    """Same adapter information, but categorical variables are not ordinal-coded.

    Columns that are categorical (including team IDs) are removed from the numeric
    branch to avoid giving Ridge an artificial numeric ordering. One-hot columns
    are source-centered/scaled so fit_intercept=False cannot simply reconstruct a
    source-season constant through a complete dummy partition.
    """
    rs = add_eng(source_rows.reset_index(drop=True))
    rt = add_eng(target_rows.reset_index(drop=True))

    cat_cols = [c for c in REFERENCE_CAT_SMALL if c in rs.columns and c in rt.columns]
    numeric_cols = [
        c for c in ([c for c in REFERENCE_RAWNUM if c in rs.columns] + ENG_COLS)
        if c not in set(cat_cols)
    ]

    med = (
        rs[numeric_cols].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    )
    Ns = rs[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    Nt = rt[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)

    Ns = Ns.reset_index(drop=True)
    Nt = Nt.reset_index(drop=True)
    Ns["base_pred"] = np.asarray(source_base, dtype=np.float64)
    Nt["base_pred"] = np.asarray(target_base, dtype=np.float64)

    Es = exact5_matrix(source_rows, pitcher_stats)
    Et = exact5_matrix(target_rows, pitcher_stats)
    Ns = pd.concat([Ns, Es], axis=1)
    Nt = pd.concat([Nt, Et], axis=1)

    nsc = StandardScaler()
    A_num = nsc.fit_transform(Ns.to_numpy(np.float64))
    B_num = nsc.transform(Nt.to_numpy(np.float64))

    Cs = rs[cat_cols].astype("string").fillna("<NA>").astype(str)
    Ct = rt[cat_cols].astype("string").fillna("<NA>").astype(str)
    enc = _make_ohe(dense=True)
    A_cat_raw = np.asarray(enc.fit_transform(Cs), dtype=np.float64)
    B_cat_raw = np.asarray(enc.transform(Ct), dtype=np.float64)

    # Center/scale one-hot columns on source. This preserves category-specific
    # contrasts while enforcing zero source-column means for no-intercept Ridge.
    csc = StandardScaler(with_mean=True, with_std=True)
    A_cat = csc.fit_transform(A_cat_raw)
    B_cat = csc.transform(B_cat_raw)

    A = np.concatenate([A_num, A_cat], axis=1)
    B = np.concatenate([B_num, B_cat], axis=1)
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise RuntimeError("OHE Ridge matrix contains NaN/Inf")

    return A, B, {
        "numeric_cols": list(Ns.columns),
        "categorical_cols": cat_cols,
        "ohe_feature_count": int(A_cat.shape[1]),
        "total_feature_count": int(A.shape[1]),
    }


def fit_predict_ridge_ohe(A, residual, B):
    y = np.asarray(residual, dtype=np.float64)
    yc = y - y.mean()
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, solver="lsqr", max_iter=200, tol=1e-5)
    model.fit(A, yc)
    return np.asarray(model.predict(B), np.float64), {
        "n_features": int(A.shape[1]),
        "source_corr_mean": float(np.mean(model.predict(A))),
        "coef_l2": float(np.linalg.norm(model.coef_)),
    }


# =============================================================================
# shift_v002/v003 sensitivity 25 representation
# =============================================================================


def _pressure_frame(frame):
    r = frame.copy()
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int)
    r["sens_pressure"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(balls == 3, "3ball", np.where(strikes == 2, "2strike", "normal")),
    )
    r["sens_pressure_bhand"] = r["sens_pressure"].astype(str) + "|" + r["batter_hand"].astype(str)
    return r


def _weighted_sensitivity_table(history, ctx, strength, cutoff_year):
    h = _pressure_frame(history.reset_index(drop=True))
    h["_w"] = np.power(
        SENS_DECAY,
        np.maximum(int(cutoff_year) - 1 - pd.to_numeric(h[SEASON_COL], errors="raise").astype(int), 0),
    ).astype(np.float64)
    h["_wy"] = h["_w"] * pd.to_numeric(h[TARGET], errors="coerce").to_numpy(np.float64)
    mu = float(np.average(h[TARGET].to_numpy(np.float64), weights=h["_w"]))

    pg = h.groupby("pitcher_id", sort=False).agg(sw=("_w", "sum"), sy=("_wy", "sum")).reset_index()
    pg["parent_rate"] = (pg["sy"] + SENS_PARENT_STRENGTH * mu) / (pg["sw"] + SENS_PARENT_STRENGTH)

    g = (
        h.groupby(["pitcher_id", ctx], sort=False)
        .agg(sw=("_w", "sum"), sy=("_wy", "sum"))
        .reset_index()
        .merge(pg[["pitcher_id", "parent_rate"]], on="pitcher_id", how="left")
    )
    g["rate"] = (g["sy"] + float(strength) * g["parent_rate"]) / (g["sw"] + float(strength))
    g["delta"] = g["rate"] - g["parent_rate"]
    g["reliability"] = g["sw"] / (g["sw"] + float(strength))
    return g[["pitcher_id", ctx, "delta", "reliability"]]


def build_sensitivity_specs(train, cutoff_year):
    min_year = int(pd.to_numeric(train[SEASON_COL], errors="raise").min())
    lo = max(min_year, int(cutoff_year) - SENS_HISTORY_YEARS)
    seasons = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)
    h = train.loc[seasons.lt(int(cutoff_year)) & seasons.ge(lo)].reset_index(drop=True)
    if h.empty:
        raise RuntimeError(f"no sensitivity history for cutoff={cutoff_year}")

    specs = []
    for ctx, strength, prefix in [
        ("batter_hand", SENS_HAND_STRENGTH, "sens__batter_hand"),
        ("sens_pressure_bhand", SENS_PRESSURE_HAND_STRENGTH, "sens__pressure_bhand"),
    ]:
        specs.append({
            "ctx": ctx,
            "prefix": prefix,
            "table": _weighted_sensitivity_table(h, ctx, strength, cutoff_year),
        })
    return specs


def _lookup(table, rows, keys, field, default=0.0):
    if len(table) == 0:
        return np.full(len(rows), default, dtype=np.float64)
    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float64)
    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float64)


def transform_sensitivity_specs(specs, rows):
    r = _pressure_frame(rows.reset_index(drop=True))
    z = pd.DataFrame(index=np.arange(len(r)))
    for spec in specs:
        de = _lookup(spec["table"], r, ["pitcher_id", spec["ctx"]], "delta", 0.0)
        re = _lookup(spec["table"], r, ["pitcher_id", spec["ctx"]], "reliability", 0.0)
        z[spec["prefix"] + "__delta"] = de
        z[spec["prefix"] + "__weighted_delta"] = de * re
        z[spec["prefix"] + "__reliability"] = re

    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    for b in range(4):
        z[f"ctx__ball_{b}"] = (balls == b).astype(np.float64)
    for s in range(3):
        z[f"ctx__strike_{s}"] = (strikes == s).astype(np.float64)
    for b in range(4):
        for s in range(3):
            z[f"ctx__count_{b}_{s}"] = ((balls == b) & (strikes == s)).astype(np.float64)

    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    if z.shape[1] != 25:
        raise RuntimeError(f"sensitivity feature count !=25: {z.shape}")
    return z


def sensitivity25_matrices(train, source_year, target_year, source_rows, target_rows):
    # Faithful to shift_v002/v003: specs are constructed separately for each
    # cutoff year from official labeled history strictly before that year.
    ss = build_sensitivity_specs(train, source_year)
    st = build_sensitivity_specs(train, target_year)
    A = transform_sensitivity_specs(ss, source_rows).to_numpy(np.float64)
    B = transform_sensitivity_specs(st, target_rows).to_numpy(np.float64)

    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Az = np.clip((A - mu) / sd, -8.0, 8.0)
    Bz = np.clip((B - mu) / sd, -8.0, 8.0)
    return Az, Bz


def fit_predict_sens25_pair(Az, residual, Bz):
    y = np.asarray(residual, dtype=np.float64)  # raw residual, shift-faithful
    ref = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=200, tol=1e-5, fit_intercept=True)
    sh = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=200, tol=1e-5, fit_intercept=False)
    ref.fit(Az, y)
    sh.fit(Az, y)
    return (
        np.asarray(sh.predict(Bz), np.float64),
        np.asarray(ref.predict(Bz), np.float64),
        {
            "shift_source_corr_mean": float(np.mean(sh.predict(Az))),
            "reference_intercept": float(np.asarray(ref.intercept_).reshape(-1)[0]),
            "coef_gap_max": float(np.max(np.abs(np.asarray(ref.coef_) - np.asarray(sh.coef_)))),
            "shift_coef_l2": float(np.linalg.norm(sh.coef_)),
        },
    )


# =============================================================================
# Summaries
# =============================================================================


def build_year_summary(metrics):
    return (
        metrics.groupby(["arm", "validation_year"], as_index=False)
        .agg(
            seed_count=("base_seed", "size"),
            total_win_count=("total_win", "sum"),
            shape_win_count=("shape_win", "sum"),
            mean_R_total_gain_pt=("R_total_gain_pt", "mean"),
            std_R_total_gain_pt=("R_total_gain_pt", "std"),
            min_R_total_gain_pt=("R_total_gain_pt", "min"),
            mean_R_level_gain_pt=("R_level_gain_pt", "mean"),
            mean_R_shape_gain_pt=("R_shape_gain_pt", "mean"),
            std_R_shape_gain_pt=("R_shape_gain_pt", "std"),
            min_R_shape_gain_pt=("R_shape_gain_pt", "min"),
        )
    )


def build_seed_summary(metrics):
    return (
        metrics.groupby(["arm", "base_seed"], as_index=False)
        .agg(
            fold_count=("validation_year", "size"),
            total_fold_wins=("total_win", "sum"),
            shape_fold_wins=("shape_win", "sum"),
            mean_R_total_gain_pt=("R_total_gain_pt", "mean"),
            mean_R_level_gain_pt=("R_level_gain_pt", "mean"),
            mean_R_shape_gain_pt=("R_shape_gain_pt", "mean"),
            worst_R_total_gain_pt=("R_total_gain_pt", "min"),
            worst_R_shape_gain_pt=("R_shape_gain_pt", "min"),
        )
    )


# =============================================================================
# Main
# =============================================================================


def main():
    t0 = time.time()

    print("=" * 118)
    print(EXPERIMENT_NAME)
    print("=" * 118)
    print("Purpose      : Ridge representation retest on frozen Treatment OOF residual bank")
    print(f"Bank source  : {BANK_SOURCE_PATH}")
    print("Eval years   :", EVAL_YEARS)
    print(f"Ridge alpha  : {RIDGE_ALPHA}")
    print(f"Adapter/OHE  : scale={COMMON_SHRINK}")
    print(f"Shift Sens25 : scale={SHIFT_SCALE}, fit_intercept=False, raw residual")
    print("Final model  : NOT TRAINED")
    print("=" * 118)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)
    if not BANK_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Residual bank missing: {BANK_SOURCE_PATH}\n"
            "Run rresid_v001 once, or set RRESID_BANK_SOURCE to its residual_bank.csv."
        )
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보호를 위해 중단: {VALIDATION_DIR}\n"
            "기존 폴더를 이동/삭제 후 다시 실행하세요."
        )
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    train[SEASON_COL] = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)

    required_train = {
        ID, TARGET, SEASON_COL, GAME_TYPE_COL, "pitcher_id", "batter_hand",
        "balls_before", "strikes_before", "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }
    missing = sorted(required_train - set(train.columns))
    if missing:
        raise KeyError(f"train required columns missing: {missing}")

    bank = pd.read_csv(
        BANK_SOURCE_PATH,
        encoding="utf-8-sig",
        usecols=[ID, "validation_year", "base_seed", "y_true", "treatment_pred", "residual"],
    )
    bank["validation_year"] = pd.to_numeric(bank["validation_year"], errors="raise").astype(int)
    bank["base_seed"] = pd.to_numeric(bank["base_seed"], errors="raise").astype(int)
    seeds = sorted(bank["base_seed"].unique().tolist())

    needed_years = sorted(set([y - 1 for y in EVAL_YEARS] + EVAL_YEARS))
    missing_years = sorted(set(needed_years) - set(bank["validation_year"].unique().tolist()))
    if missing_years:
        raise RuntimeError(f"bank missing OOF years: {missing_years}")

    pitcher_stats = build_pitcher_season_stats(train)
    metrics_rows = []
    pred_parts = []
    diag_rows = []

    for target_year in EVAL_YEARS:
        source_year = target_year - 1
        raw_source = train.loc[
            train[SEASON_COL].eq(source_year) & ~train[GAME_TYPE_COL].fillna("__MISSING__").astype(str).eq("F")
        ].copy().reset_index(drop=True)
        raw_target = train.loc[
            train[SEASON_COL].eq(target_year) & ~train[GAME_TYPE_COL].fillna("__MISSING__").astype(str).eq("F")
        ].copy().reset_index(drop=True)

        print("\n" + "#" * 118)
        print(f"TARGET YEAR={target_year} | SOURCE={source_year} | source_R={len(raw_source):,} target_R={len(raw_target):,}")
        print("#" * 118)

        # Sensitivity rows do not depend on CatBoost seed: build once per transition.
        sens_A, sens_B = sensitivity25_matrices(
            train=train,
            source_year=source_year,
            target_year=target_year,
            source_rows=raw_source,
            target_rows=raw_target,
        )

        for seed in seeds:
            sb = bank.loc[(bank["validation_year"] == source_year) & (bank["base_seed"] == seed)].set_index(ID)
            tb = bank.loc[(bank["validation_year"] == target_year) & (bank["base_seed"] == seed)].set_index(ID)

            if not raw_source[ID].isin(sb.index).all() or not raw_target[ID].isin(tb.index).all():
                raise RuntimeError(f"residual bank alignment failed {source_year}->{target_year} seed={seed}")

            source_base = raw_source[ID].map(sb["treatment_pred"]).to_numpy(np.float64)
            source_y = raw_source[ID].map(sb["y_true"]).to_numpy(np.float64)
            source_resid = source_y - source_base

            target_base = raw_target[ID].map(tb["treatment_pred"]).to_numpy(np.float64)
            target_y = raw_target[ID].map(tb["y_true"]).to_numpy(np.float64)

            print(
                f"\n[TRANSITION] {source_year}->{target_year} seed={seed} "
                f"source residual mean={source_resid.mean():+.8f} std={source_resid.std():.8f}"
            )

            # Constant level reference
            constant_corr = np.full(len(target_y), float(source_resid.mean()), dtype=np.float64)

            # v001 coded-49 reference
            X49s, X49t = coded49_matrices(
                raw_source, raw_target, source_base, target_base, pitcher_stats
            )
            coded_corr, coded_diag = fit_predict_ridge_coded49(X49s, source_resid, X49t)

            # Ridge-native OHE adapter information
            Aohe, Bohe, ohe_schema = ohe_ridge_matrices(
                raw_source, raw_target, source_base, target_base, pitcher_stats
            )
            ohe_corr, ohe_diag = fit_predict_ridge_ohe(Aohe, source_resid, Bohe)

            # shift-v002/v003 25-col sensitivity pair
            sens_shift_corr, sens_ref_corr, sens_diag = fit_predict_sens25_pair(
                sens_A, source_resid, sens_B
            )

            arm_specs = [
                ("CONSTANT_S05", constant_corr, COMMON_SHRINK),
                ("RIDGE_CODED49_S05", coded_corr, COMMON_SHRINK),
                ("RIDGE_OHE_S05", ohe_corr, COMMON_SHRINK),
                ("RIDGE_SENS25_SHIFT_ALLR_S075", sens_shift_corr, SHIFT_SCALE),
                ("RIDGE_SENS25_REFERENCE_ALLR_S075", sens_ref_corr, SHIFT_SCALE),
            ]

            payload = {
                ID: raw_target[ID].to_numpy(),
                "validation_year": int(target_year),
                "source_year": int(source_year),
                "base_seed": int(seed),
                "y_true": target_y,
                "treatment_pred": target_base,
            }

            for arm, corr, scale in arm_specs:
                m, p = evaluate_arm(
                    target_y, target_base, corr, scale, arm, target_year, source_year, seed
                )
                metrics_rows.append(m)
                payload[arm.lower() + "__correction"] = corr
                payload[arm.lower() + "__prediction"] = p
                print(
                    f"{arm:<40} Total={m['R_total_gain_pt']:+9.3f} "
                    f"Level={m['R_level_gain_pt']:+9.3f} Shape={m['R_shape_gain_pt']:+9.3f}"
                )

            diag_rows.append({
                "validation_year": int(target_year),
                "source_year": int(source_year),
                "base_seed": int(seed),
                "source_residual_mean": float(source_resid.mean()),
                "coded49": coded_diag,
                "ohe": {**ohe_schema, **ohe_diag},
                "sens25": sens_diag,
            })
            pred_parts.append(pd.DataFrame(payload))

            del X49s, X49t, Aohe, Bohe

    metrics = pd.DataFrame(metrics_rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    ys = build_year_summary(metrics)
    ss = build_seed_summary(metrics)

    metrics.to_csv(RIDGE_METRICS_PATH, index=False, encoding="utf-8-sig")
    preds.to_csv(RIDGE_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    ys.to_csv(YEAR_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ss.to_csv(SEED_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "purpose": "Ridge representation retest after rresid_v001",
        "validation_only": True,
        "final_model_saved": False,
        "bank_source": str(BANK_SOURCE_PATH),
        "base_seeds": seeds,
        "eval_years": EVAL_YEARS,
        "temporal_protocol": "source year t-1 Treatment OOF residual -> target year t",
        "arms": {
            "CONSTANT_S05": "previous-year R residual mean * 0.50",
            "RIDGE_CODED49_S05": "rresid_v001 coded 49 features, standardized, centered target, no intercept, scale .50",
            "RIDGE_OHE_S05": "numeric+Exact standardized; small categoricals source-centered one-hot; centered target; no intercept; scale .50",
            "RIDGE_SENS25_SHIFT_ALLR_S075": {
                "source": "shift_v002/v003 sensitivity feature construction",
                "n_features": 25,
                "history_years": SENS_HISTORY_YEARS,
                "decay": SENS_DECAY,
                "parent_strength": SENS_PARENT_STRENGTH,
                "hand_strength": SENS_HAND_STRENGTH,
                "pressure_hand_strength": SENS_PRESSURE_HAND_STRENGTH,
                "alpha": RIDGE_ALPHA,
                "fit_intercept": False,
                "target": "raw previous-year Treatment OOF residual",
                "scale": SHIFT_SCALE,
                "routing_note": "applied to all current R rows; original shift applied only CORE",
            },
            "RIDGE_SENS25_REFERENCE_ALLR_S075": "same Sens25 but fit_intercept=True, scale .75",
        },
        "diagnostics": diag_rows,
        "rule_compliance": {
            "official_train_labels_only": True,
            "test_rows_used": False,
            "test_row_aggregation": False,
            "residual_source": "strictly previous OOF season only",
            "sensitivity_history": "official labeled seasons strictly before each row year",
        },
        "outputs": {
            "ridge_metrics": str(RIDGE_METRICS_PATH),
            "ridge_predictions": str(RIDGE_PREDICTIONS_PATH),
            "year_summary": str(YEAR_SUMMARY_PATH),
            "seed_summary": str(SEED_SUMMARY_PATH),
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    META_PATH.write_text(json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("YEAR SUMMARY")
    print("=" * 118)
    print(ys.loc[ys["arm"].isin(PRIMARY_ARMS)].to_string(index=False))

    print("\nInterpretation keys:")
    print("  OHE > CODED49        : categorical ordinal encoding was materially hurting Ridge")
    print("  SENS25 shift > OHE   : shift-style sensitivity representation is the key inductive bias")
    print("  REF >> SHIFT         : gain is mostly previous-year Level/intercept transfer")
    print("  SHIFT Shape > 0      : intercept-free sensitivity Shape transfers to next season")
    print("  2024 3/3 seed Shape  : strongest evidence for current R-branch adoption")
    print(f"\nSaved under: {VALIDATION_DIR}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""rresid_v003: historical, validation-only Ridge shrink audit.

Run from the AIMERS repository root:
    python /path/to/this/train.py

Required: data/train.csv and validation/rresid_v001/residual_bank.csv.
Optional: v002/ridge_predictions.csv (exact replay) and ridge_metrics.csv
(reference parity). No CatBoost refit, final model, test read, or submission.

The 2024 results were already inspected before this experiment was designed.
2024 is excluded from THIS RUN's scale selection, but is NOT an untouched
holdout. All gate results are exploratory, never deployment approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Project-standard paths. Paths are relative to the working repository root.
EXPERIMENT_NAME = "rresid_v003"
DATA_DIR = Path(os.environ.get("RRESID_DATA_DIR", "./data"))
TRAIN_PATH = DATA_DIR / "train.csv"
MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"  # NOT written: validation only
VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"  # interface only; NEVER opened
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"  # NEVER opened
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"  # NEVER written

BANK_SOURCE_PATH = Path(os.environ.get(
    "RRESID_BANK_SOURCE", str(VALIDATION_ROOT / "rresid_v001" / "residual_bank.csv")))
RIDGE_PREDICTIONS_SOURCE_PATH = Path(os.environ.get(
    "RRESID_RIDGE_PREDICTIONS",
    str(VALIDATION_ROOT / "rresid_v002" / "ridge_predictions.csv")))
REFERENCE_METRICS_PATH = Path(os.environ.get(
    "RRESID_REFERENCE_METRICS",
    str(VALIDATION_ROOT / "rresid_v002" / "ridge_metrics.csv")))

ID, TARGET, SEASON_COL, GAME_TYPE_COL = (
    "row_id", "control_success", "season", "game_type")
BASE_SEEDS = (42, 2026, 314)
OOF_YEARS = (2020, 2021, 2022, 2023, 2024)
DISCOVERY_YEARS = (2021, 2022, 2023)
REPORT_YEAR = 2024
SCALE_GRID = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0)
FIXED_WEAK_SCALE = 0.15  # hypothesis suggested AFTER viewing v001/v002 2024
RIDGE_ALPHA = 1000.0
SENS_HISTORY_YEARS = 3
SENS_DECAY = 0.65
SENS_PARENT_STRENGTH = 100.0
SENS_HAND_STRENGTH = 35.0
SENS_PRESSURE_HAND_STRENGTH = 45.0
POINT_EPS = 1e-6
TIE_EPS = 1e-9
PARITY_POINT_ATOL = 1e-6
PARITY_VALUE_ATOL = 1e-10

# The representation/solver is copied from v002, not redesigned.
FAMILIES = ("RIDGE_CODED49", "RIDGE_SENS25")
NATIVE_SCALE = {"RIDGE_CODED49": 0.50, "RIDGE_SENS25": 0.75}
LEGACY_ARM = {
    "RIDGE_CODED49": "RIDGE_CODED49_S05",
    "RIDGE_SENS25": "RIDGE_SENS25_SHIFT_ALLR_S075",
    "CONSTANT": "CONSTANT_S05",
}
CACHE_CORRECTION = {
    family: LEGACY_ARM[family].lower() + "__correction" for family in FAMILIES
}

# Below: verbatim v002 feature helpers. They fit preprocessing on source rows
# and construct history features using official seasons before the row's year.

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


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    text = json.dumps(json_safe(payload), ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def strict_int_column(frame, name):
    x = pd.to_numeric(frame[name], errors="raise")
    if x.isna().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: missing/nonfinite integer")
    if not np.equal(x.to_numpy(), np.floor(x.to_numpy())).all():
        raise ValueError(f"{name}: fractional integer")
    frame[name] = x.astype(np.int64)


def check_finite(values, name):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: NaN/Inf")
    return a


def r_mask(frame):
    if frame[GAME_TYPE_COL].isna().any():
        raise ValueError("game_type contains missing values; routing is ambiguous")
    return frame[GAME_TYPE_COL].astype(str).ne("F")


def read_inputs():
    # Check all indispensable files BEFORE creating an output directory.
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Official training data missing: {TRAIN_PATH}")
    if not BANK_SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Frozen Treatment OOF bank missing: {BANK_SOURCE_PATH}\n"
            "Restore residual_bank.csv from the original v001 run, or set "
            "RRESID_BANK_SOURCE to its path. Summary CSVs cannot replace this "
            "bank. v003 will NOT silently train a different CatBoost baseline.")
    for env_name, path in [
        ("RRESID_RIDGE_PREDICTIONS", RIDGE_PREDICTIONS_SOURCE_PATH),
        ("RRESID_REFERENCE_METRICS", REFERENCE_METRICS_PATH),
    ]:
        if env_name in os.environ and not path.is_file():
            raise FileNotFoundError(
                f"Explicit {env_name} file is missing: {path}. "
                "Refusing to switch modes or skip parity silently.")
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {VALIDATION_DIR}. "
            "Preserve that run and choose a different EXPERIMENT_NAME.")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig",
                        dtype={ID: "string"}, low_memory=False)
    required = set(REFERENCE_RAWNUM + REFERENCE_CAT_SMALL) | {
        ID, TARGET, SEASON_COL, GAME_TYPE_COL, "pitcher_id"}
    require_columns(train, required, "train.csv")
    if train[ID].isna().any() or train[ID].duplicated().any():
        raise ValueError("train.csv must have unique, nonmissing row_id")
    strict_int_column(train, SEASON_COL)
    train[TARGET] = check_finite(pd.to_numeric(train[TARGET], errors="raise"),
                                "official labels")
    if not train[TARGET].isin([0.0, 1.0]).all():
        raise ValueError("Official labels must be binary")
    if train["pitcher_id"].isna().any() or train["batter_hand"].isna().any():
        raise ValueError("Missing pitcher_id/batter_hand; history keys ambiguous")
    r_mask(train)
    if int(train[SEASON_COL].min()) > min(OOF_YEARS) - 1:
        raise ValueError("Official pre-2020 history is required")

    bank = pd.read_csv(BANK_SOURCE_PATH, encoding="utf-8-sig",
                       dtype={ID: "string"}, low_memory=False)
    require_columns(bank, [ID, "validation_year", "base_seed",
                           "y_true", "treatment_pred", "residual"], "OOF bank")
    for column in ["validation_year", "base_seed"]:
        strict_int_column(bank, column)
    if set(bank["base_seed"]) != set(BASE_SEEDS):
        raise ValueError(f"OOF bank must contain exactly base seeds {BASE_SEEDS}")
    if set(bank["validation_year"]) != set(OOF_YEARS):
        raise ValueError(f"OOF bank must contain exactly years {OOF_YEARS}")
    key = ["validation_year", "base_seed", ID]
    if bank[ID].isna().any() or bank.duplicated(key).any():
        raise ValueError("OOF bank contains duplicate/missing row keys")
    for column in ["y_true", "treatment_pred", "residual"]:
        bank[column] = check_finite(
            pd.to_numeric(bank[column], errors="raise"), f"bank.{column}")
    if not bank["treatment_pred"].between(0, 1).all():
        raise ValueError("Treatment bank predictions must already lie in [0,1]")
    if not np.allclose(bank["residual"], bank["y_true"] - bank["treatment_pred"],
                       rtol=0, atol=PARITY_VALUE_ATOL):
        raise ValueError("OOF residual != y_true - treatment_pred")

    official = train.set_index(ID, verify_integrity=True)
    if not bank[ID].isin(official.index).all():
        raise ValueError("OOF bank contains row_id absent from official train")
    official_year = bank[ID].map(official[SEASON_COL])
    official_y = bank[ID].map(official[TARGET])
    if not np.array_equal(official_year.to_numpy(), bank["validation_year"].to_numpy()):
        raise ValueError("OOF validation_year != official row season")
    if not np.array_equal(official_y.to_numpy(), bank["y_true"].to_numpy()):
        raise ValueError("OOF labels differ from official labels")

    # Each year/seed must contain every R row exactly once, not just some rows.
    bank_is_r = bank[ID].map(official[GAME_TYPE_COL]).astype(str).ne("F")
    bank_r = bank.loc[bank_is_r].copy()
    folds = {}
    for year in OOF_YEARS:
        expected = set(train.loc[train[SEASON_COL].eq(year) & r_mask(train), ID])
        if not expected:
            raise ValueError(f"No R rows for year {year}")
        for seed in BASE_SEEDS:
            fold = bank_r.loc[bank_r["validation_year"].eq(year)
                              & bank_r["base_seed"].eq(seed)].set_index(ID)
            if set(fold.index) != expected:
                raise ValueError(f"Missing/extra R rows: OOF year={year}, seed={seed}")
            folds[(year, seed)] = fold

    cache = None
    if RIDGE_PREDICTIONS_SOURCE_PATH.is_file():
        cache = pd.read_csv(RIDGE_PREDICTIONS_SOURCE_PATH, encoding="utf-8-sig",
                            dtype={ID: "string"}, low_memory=False)
        require_columns(cache, key + ["source_year", "y_true", "treatment_pred"]
                        + list(CACHE_CORRECTION.values()), "v002 predictions")
        for column in ["validation_year", "source_year", "base_seed"]:
            strict_int_column(cache, column)
        if cache[ID].isna().any() or cache.duplicated(key).any():
            raise ValueError("v002 prediction cache contains duplicate row keys")
        if not cache["source_year"].eq(cache["validation_year"] - 1).all():
            raise ValueError("v002 correction source must be target_year - 1")
        if set(cache["base_seed"]) != set(BASE_SEEDS):
            raise ValueError("v002 cache has a different seed set")
        if set(cache["validation_year"]) != set(DISCOVERY_YEARS + (REPORT_YEAR,)):
            raise ValueError("v002 cache has a different evaluation-year set")
        for column in ["y_true", "treatment_pred"] + list(CACHE_CORRECTION.values()):
            cache[column] = check_finite(pd.to_numeric(cache[column], errors="raise"),
                                         f"cache.{column}")
        cache = cache.set_index(["validation_year", "base_seed", ID],
                                verify_integrity=True)

    reference = None
    if REFERENCE_METRICS_PATH.is_file():
        reference = pd.read_csv(REFERENCE_METRICS_PATH, encoding="utf-8-sig")
        require_columns(reference, ["arm", "validation_year", "source_year",
                                    "base_seed", "scale", "n_R", "base_brier",
                                    "candidate_brier", "R_total_gain_pt",
                                    "R_level_gain_pt", "R_shape_gain_pt"],
                        "v002 reference metrics")
        if reference.duplicated(["arm", "validation_year", "base_seed"]).any():
            raise ValueError("Duplicate reference metrics keys")
        reference = reference.set_index(["arm", "validation_year", "base_seed"])
    return train, folds, cache, reference


def decompose(y, prediction):
    err = check_finite(prediction, "prediction") - check_finite(y, "labels")
    total = float(np.mean(err * err))
    level = float(np.mean(err) ** 2)
    return total, level, total - level


def evaluate(y, base, correction, scale, family, year, seed):
    y, base, correction = [check_finite(x, name) for x, name in [
        (y, "y"), (base, "base"), (correction, "correction")]]
    if y.ndim != 1 or y.shape != base.shape or y.shape != correction.shape:
        raise ValueError("Prediction vectors are not aligned")
    before_clip = base + float(scale) * correction
    pred = np.clip(before_clip, 0.0, 1.0)
    b0, b1 = decompose(y, base), decompose(y, pred)
    rate = float(y.mean())
    denominator = rate * (1.0 - rate)
    if denominator <= 0:
        raise ValueError("R reference Brier is zero; metrics undefined")
    gains = 100000.0 * (np.array(b0) - np.array(b1)) / denominator
    if abs(gains[0] - gains[1] - gains[2]) > POINT_EPS:
        raise AssertionError("Total gain != Level gain + Shape gain")
    applied = pred - base
    return {
        "family": family, "validation_year": int(year),
        "source_year": int(year - 1), "base_seed": int(seed),
        "scale": float(scale), "n_R": int(len(y)),
        "base_brier": b0[0], "candidate_brier": b1[0],
        "R_total_gain_pt": float(gains[0]), "R_level_gain_pt": float(gains[1]),
        "R_shape_gain_pt": float(gains[2]),
        "total_win": bool(gains[0] > POINT_EPS),
        "shape_win": bool(gains[2] > POINT_EPS),
        "correction_mean": float(correction.mean()),
        "correction_std": float(correction.std()),
        "applied_correction_mean": float(applied.mean()),
        "applied_correction_std": float(applied.std()),
        "clip_count": int(np.count_nonzero((before_clip < 0) | (before_clip > 1))),
    }


def check_reference(row, reference):
    if reference is None:
        return {"status": "not_checked_missing_reference"}
    key = (LEGACY_ARM[row["family"]], row["validation_year"], row["base_seed"])
    if key not in reference.index:
        raise ValueError(f"Reference metric missing: {key}")
    ref = reference.loc[key]
    for field in ["source_year", "n_R"]:
        if int(ref[field]) != row[field]:
            raise ValueError(f"Reference {field} mismatch: {key}")
    gaps = {}
    for field in ["scale", "base_brier", "candidate_brier",
                  "R_total_gain_pt", "R_level_gain_pt", "R_shape_gain_pt"]:
        a, b = float(row[field]), float(ref[field])
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"Reference metric is nonfinite: {key}, {field}")
        gap = abs(a - b)
        tolerance = PARITY_POINT_ATOL if field.endswith("_pt") else PARITY_VALUE_ATOL
        gaps[field] = gap
        if gap > tolerance:
            raise ValueError(
                f"v002 replay parity failed: {key}, {field}, gap={gap:.12g}. "
                "Restore the original ridge_predictions.csv or reproduce its "
                "runtime. Do not widen the tolerance to disguise a different learner.")
    return {"status": "pass", "arm": key[0], "validation_year": key[1],
            "base_seed": key[2], "absolute_gaps": gaps}


def cached_corrections(cache, target_rows, year, seed, target_bank):
    fold = cache.xs((year, seed), level=("validation_year", "base_seed"))
    if set(fold.index) != set(target_rows[ID]):
        raise ValueError(f"Cached R row set mismatch: {year}, {seed}")
    fold = fold.loc[target_rows[ID]]
    if not np.array_equal(fold["y_true"].to_numpy(), target_bank["y_true"].to_numpy()):
        raise ValueError("Cache labels differ from bank")
    if not np.allclose(fold["treatment_pred"], target_bank["treatment_pred"],
                       rtol=0, atol=PARITY_VALUE_ATOL):
        raise ValueError("Cache Treatment differs from frozen OOF bank")
    result = {family: fold[column].to_numpy(np.float64)
              for family, column in CACHE_CORRECTION.items()}
    # If saved native-scale predictions are present, validate their semantics.
    for family in FAMILIES:
        column = LEGACY_ARM[family].lower() + "__prediction"
        if column in fold:
            expected = np.clip(target_bank["treatment_pred"].to_numpy(np.float64)
                               + NATIVE_SCALE[family] * result[family], 0, 1)
            if not np.allclose(expected, fold[column].to_numpy(np.float64),
                               rtol=0, atol=PARITY_VALUE_ATOL):
                raise ValueError(f"Cache correction is not unscaled: {family}")
    return result


def fit_sens25(A, residual, B):
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=200, tol=1e-5,
                  fit_intercept=False)
    model.fit(A, np.asarray(residual, np.float64))
    return np.asarray(model.predict(B), np.float64), {
        "source_correction_mean": float(model.predict(A).mean()),
        "coefficient_norm": float(np.linalg.norm(model.coef_)),
    }


def evaluate_year(train, folds, cache, reference, year, output_dir, selection=None):
    source_year = year - 1
    source_rows = train.loc[train[SEASON_COL].eq(source_year) & r_mask(train)]
    target_rows = train.loc[train[SEASON_COL].eq(year) & r_mask(train)]
    # Target labels cannot enter feature generation or fitting.
    source_x = source_rows.drop(columns=[TARGET]).reset_index(drop=True)
    target_x = target_rows.drop(columns=[TARGET]).reset_index(drop=True)
    history = train.loc[train[SEASON_COL].lt(year)].copy()
    if int(history[SEASON_COL].max()) >= year:
        raise AssertionError("Target-year labels leaked into history")
    pitcher_stats = None
    sens_A = sens_B = None
    if cache is None:
        pitcher_stats = build_pitcher_season_stats(history)
        sens_A, sens_B = sensitivity25_matrices(
            history, source_year, year, source_x, target_x)

    rows, fits, parity = [], [], []
    correction_dir = output_dir / "corrections"
    correction_dir.mkdir(exist_ok=True)
    for seed in BASE_SEEDS:
        source = folds[(source_year, seed)].loc[source_x[ID]]
        target = folds[(year, seed)].loc[target_x[ID]]
        source_base = source["treatment_pred"].to_numpy(np.float64)
        target_base = target["treatment_pred"].to_numpy(np.float64)
        residual = source["y_true"].to_numpy(np.float64) - source_base
        target_y = target["y_true"].to_numpy(np.float64)
        fit_info = {"validation_year": year, "source_year": source_year,
                    "base_seed": seed, "source_residual_mean": float(residual.mean()),
                    "n_source_R": len(source), "n_target_R": len(target)}
        if cache is not None:
            corrections = cached_corrections(cache, target_x, year, seed, target)
            fit_info["mode"] = "replay_frozen_v002_corrections"
        else:
            Xs, Xt = coded49_matrices(source_x, target_x, source_base,
                                     target_base, pitcher_stats)
            coded, coded_diag = fit_predict_ridge_coded49(Xs, residual, Xt)
            sens, sens_diag = fit_sens25(sens_A, residual, sens_B)
            corrections = {"RIDGE_CODED49": coded, "RIDGE_SENS25": sens}
            fit_info.update(mode="refit_v002_recipe_on_frozen_bank",
                            coded49=coded_diag, sens25=sens_diag)
            del Xs, Xt
        corrections["CONSTANT"] = np.full(len(target_y), residual.mean())
        payload = pd.DataFrame({
            ID: target_x[ID].to_numpy(), "validation_year": year,
            "source_year": source_year, "base_seed": seed,
            "y_true": target_y, "treatment_pred": target_base,
        })
        for family, correction in corrections.items():
            payload[family.lower() + "__correction"] = correction
            for scale in SCALE_GRID:
                row = evaluate(target_y, target_base, correction, scale,
                               family, year, seed)
                rows.append(row)
                anchor = NATIVE_SCALE.get(family, 0.50)
                if scale == anchor:
                    parity.append(check_reference(row, reference))
            if selection is not None and family in FAMILIES:
                chosen = float(selection[family]["scale"])
                payload[family.lower() + "__selected_prediction"] = np.clip(
                    target_base + chosen * correction, 0, 1)
                payload[family.lower() + "__fixed015_prediction"] = np.clip(
                    target_base + FIXED_WEAK_SCALE * correction, 0, 1)
        payload.to_csv(correction_dir / f"fold_{year}_seed_{seed}.csv.gz",
                       index=False, compression="gzip")
        fits.append(fit_info)
        print(f"[fold={year} seed={seed}] {fit_info['mode']} complete", flush=True)
    return rows, fits, parity


def select_scales(discovery):
    actual_years = set(discovery["validation_year"])
    if actual_years != set(DISCOVERY_YEARS):
        raise ValueError("Selection accepts ONLY all 2021-2023 discovery folds")
    if REPORT_YEAR in actual_years:
        raise AssertionError("Report-year metric entered scale selection")
    selection, summaries = {}, []
    for family in FAMILIES:
        candidates = []
        for scale in SCALE_GRID:
            # Larger scales are diagnostics; selection only tests shrinkage.
            if scale > NATIVE_SCALE[family]:
                continue
            block = discovery.loc[discovery["family"].eq(family)
                                  & discovery["scale"].eq(scale)]
            if len(block) != len(DISCOVERY_YEARS) * len(BASE_SEEDS):
                raise ValueError(f"Incomplete discovery block: {family}, {scale}")
            if block.duplicated(["validation_year", "base_seed"]).any():
                raise ValueError("Duplicate discovery fold/seed metric")
            if set(block["base_seed"]) != set(BASE_SEEDS):
                raise ValueError("Discovery seed set mismatch")
            numeric = block[["R_shape_gain_pt", "R_level_gain_pt", "R_total_gain_pt"]]
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError("Nonfinite discovery metric")
            by_year = block.groupby("validation_year")[
                ["R_shape_gain_pt", "R_level_gain_pt", "R_total_gain_pt"]].mean()
            item = {
                "family": family, "scale": float(scale),
                "score_equal_year_mean_shape_pt": float(by_year["R_shape_gain_pt"].mean()),
                "worst_year_mean_shape_pt": float(by_year["R_shape_gain_pt"].min()),
                "mean_total_gain_pt": float(by_year["R_total_gain_pt"].mean()),
                "max_abs_year_mean_level_pt": float(by_year["R_level_gain_pt"].abs().max()),
                "positive_shape_year_count": int((by_year["R_shape_gain_pt"] > POINT_EPS).sum()),
                "positive_shape_seed_fold_count": int((block["R_shape_gain_pt"] > POINT_EPS).sum()),
            }
            candidates.append(item)
        maximum = max(x["score_equal_year_mean_shape_pt"] for x in candidates)
        tied = [x for x in candidates
                if maximum - x["score_equal_year_mean_shape_pt"] <= TIE_EPS]
        chosen = min(tied, key=lambda x: x["scale"])
        selection[family] = dict(chosen)
        selection[family]["reason"] = (
            "maximize equal-weight mean of discovery-year mean Shape gains; "
            "ties choose smaller scale; scale 0 allows no change")
        for item in candidates:
            summaries.append(dict(item, selected=(item["scale"] == chosen["scale"])))
    return selection, pd.DataFrame(summaries)


def policy_metrics(metrics, selection):
    blocks = []
    for family in FAMILIES:
        policies = {
            "discovery_selected": float(selection[family]["scale"]),
            "fixed_015_posthoc_hypothesis": FIXED_WEAK_SCALE,
            "legacy_anchor": NATIVE_SCALE[family],
        }
        for variant, scale in policies.items():
            block = metrics.loc[metrics["family"].eq(family)
                                & metrics["scale"].eq(scale)].copy()
            block["variant"] = variant
            blocks.append(block)
    block = metrics.loc[metrics["family"].eq("CONSTANT")
                        & metrics["scale"].eq(0.50)].copy()
    block["variant"] = "constant_level_control"
    blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def gate_report(block):
    if len(block) != 3 or set(block["base_seed"]) != set(BASE_SEEDS):
        raise ValueError("Report gate requires exactly three base seeds")
    shape = block["R_shape_gain_pt"].to_numpy(np.float64)
    mean = float(shape.mean())
    std = float(shape.std(ddof=1))
    tu = 2.0 * std / math.sqrt(3)
    tp = max(5.0, tu)
    level = float(block["R_level_gain_pt"].mean())
    primary_pass = bool(np.all(shape > POINT_EPS) and mean >= tp)
    level_pass = bool(abs(level) <= 5.0)
    scale = float(block["scale"].iloc[0])
    if scale == 0:
        status = "NO_CHANGE"
    elif primary_pass and level_pass:
        status = "POSTHOC_R_GATES_PASS"
    elif primary_pass:
        status = "LEVEL_GATE_FAIL"
    elif mean <= POINT_EPS or not np.all(shape > POINT_EPS):
        status = "PRIMARY_FAIL"
    elif mean < tu:
        status = "UNCERTAIN_SEED_VARIATION"
    else:
        status = "WEAK_POSITIVE_BELOW_5PT"
    return {
        "status": status, "scale": scale,
        "shape_mean_pt": mean, "shape_sample_std_pt": std,
        "shape_seed_wins": int(np.sum(shape > POINT_EPS)),
        "seed_count": 3, "T_u": tu, "T_p": tp,
        "primary_pass": primary_pass, "level_mean_pt": level,
        "level_pass": level_pass,
        "total_mean_pt": float(block["R_total_gain_pt"].mean()),
        "F_level_gate": {
            "status": "not_measured",
            "routing_contract": "correction applies only to non-F rows; F must remain unchanged",
            "measured_F_gain": None,
        },
        "overall_deployment_approval": False,
        "interpretation": "historical posthoc audit; 2024 was previously inspected",
    }


def summarize(selected):
    group = ["variant", "family", "scale"]
    year = selected.groupby(group + ["validation_year"], as_index=False).agg(
        seed_count=("base_seed", "size"),
        total_win_count=("total_win", "sum"),
        shape_win_count=("shape_win", "sum"),
        mean_R_total_gain_pt=("R_total_gain_pt", "mean"),
        std_R_total_gain_pt=("R_total_gain_pt", "std"),
        mean_R_level_gain_pt=("R_level_gain_pt", "mean"),
        mean_R_shape_gain_pt=("R_shape_gain_pt", "mean"),
        std_R_shape_gain_pt=("R_shape_gain_pt", "std"),
        min_R_shape_gain_pt=("R_shape_gain_pt", "min"),
        clip_count=("clip_count", "sum"),
    )
    seeds = selected.groupby(group + ["base_seed"], as_index=False).agg(
        fold_count=("validation_year", "size"),
        total_fold_wins=("total_win", "sum"),
        shape_fold_wins=("shape_win", "sum"),
        mean_R_total_gain_pt=("R_total_gain_pt", "mean"),
        mean_R_level_gain_pt=("R_level_gain_pt", "mean"),
        mean_R_shape_gain_pt=("R_shape_gain_pt", "mean"),
    )
    return year, seeds


def main():
    started = time.time()
    print(EXPERIMENT_NAME, flush=True)
    print("HISTORICAL POSTHOC AUDIT: 2024 already seen; NOT an untouched holdout.",
          flush=True)
    train, folds, cache, reference = read_inputs()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    try:
        source_files = {
            "official_train": TRAIN_PATH, "frozen_bank": BANK_SOURCE_PATH,
            "this_script": Path(__file__).resolve(),
        }
        if cache is not None:
            source_files["v002_predictions"] = RIDGE_PREDICTIONS_SOURCE_PATH
        if reference is not None:
            source_files["v002_metrics"] = REFERENCE_METRICS_PATH
        fingerprints = {k: {"path": str(v.resolve()), "sha256": file_digest(v)}
                        for k, v in source_files.items()}
        shutil.copyfile(Path(__file__).resolve(), VALIDATION_DIR / "train_source.py")
        protocol = {
            "experiment_name": EXPERIMENT_NAME,
            "historical_posthoc": True, "report_year_previously_seen": True,
            "validation_only": True, "final_model_saved": False,
            "selection_years": list(DISCOVERY_YEARS), "report_year": REPORT_YEAR,
            "base_seeds": list(BASE_SEEDS), "scale_grid": list(SCALE_GRID),
            "fixed_weak_scale": FIXED_WEAK_SCALE,
            "fixed_weak_scale_origin": "posthoc hypothesis after viewing v001/v002 2024",
            "families": list(FAMILIES), "native_scale": NATIVE_SCALE,
            "selection_rule": "equal-year mean Shape; ties smaller scale; scale <= native",
            "residual_training": "strict previous-year Treatment OOF residual only",
            "target_year_correction_recentered": False, "seed_predictions_averaged": False,
            "source_mode": ("replay_frozen_v002_corrections" if cache is not None
                            else "refit_v002_recipe_on_frozen_bank"),
            "F_gate": "not_measured; R-only correction contract",
            "extra_trees": "excluded: exact v001 execution source unavailable",
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "source_files": fingerprints,
        }
        write_json(VALIDATION_DIR / "protocol.json", protocol)
        write_json(VALIDATION_DIR / "run_status.json", {"status": "running_discovery"})
        all_rows, fits, parity = [], [], []
        for year in DISCOVERY_YEARS:
            rows, diagnostic, checks = evaluate_year(
                train, folds, cache, reference, year, VALIDATION_DIR)
            all_rows.extend(rows)
            fits.extend(diagnostic)
            parity.extend(checks)

        discovery = pd.DataFrame(all_rows)
        selection, score_table = select_scales(discovery)
        frozen = {"protocol": protocol, "selection": selection,
                  "2024_metric_used_by_selector": False}
        freeze_path = VALIDATION_DIR / "discovery_selection.json"
        write_json(freeze_path, frozen)  # written BEFORE fitting/evaluating 2024
        freeze_hash = file_digest(freeze_path)
        score_table.to_csv(VALIDATION_DIR / "discovery_scores.csv", index=False)
        discovery.to_csv(VALIDATION_DIR / "discovery_metrics.csv", index=False)
        print("Frozen discovery scales:", {k: v["scale"] for k, v in selection.items()},
              flush=True)
        write_json(VALIDATION_DIR / "run_status.json",
                   {"status": "running_historical_2024_report",
                    "selection_sha256": freeze_hash})

        rows, diagnostic, checks = evaluate_year(
            train, folds, cache, reference, REPORT_YEAR, VALIDATION_DIR, selection)
        all_rows.extend(rows)
        fits.extend(diagnostic)
        parity.extend(checks)
        if file_digest(freeze_path) != freeze_hash:
            raise RuntimeError("Frozen selection changed during report evaluation")

        metrics = pd.DataFrame(all_rows)
        chosen = policy_metrics(metrics, selection)
        year_summary, seed_summary = summarize(chosen)
        metrics.to_csv(VALIDATION_DIR / "scale_metrics.csv", index=False)
        chosen.to_csv(VALIDATION_DIR / "selected_metrics.csv", index=False)
        year_summary.to_csv(VALIDATION_DIR / "year_summary.csv", index=False)
        seed_summary.to_csv(VALIDATION_DIR / "seed_summary.csv", index=False)
        report = {}
        for (variant, family), block in chosen.loc[
                chosen["validation_year"].eq(REPORT_YEAR)].groupby(["variant", "family"]):
            report[f"{variant}/{family}"] = gate_report(block)
        write_json(VALIDATION_DIR / "confirmation.json", {
            "kind": "historical_posthoc_2024_report",
            "not_untouched_confirmation": True,
            "selection_sha256": freeze_hash, "results": report,
            "no_model_family_selected_using_2024": True,
        })
        write_json(VALIDATION_DIR / "parity_report.json", {"checks": parity})
        write_json(VALIDATION_DIR / "fit_diagnostics.json", {"fits": fits})
        write_json(VALIDATION_DIR / "final_metadata.json", {
            **protocol, "status": "complete",
            "selection": selection, "selection_sha256": freeze_hash,
            "reference_parity": ("passed" if reference is not None
                                 else "not_checked_missing_reference"),
            "row_level_metrics": "actual clipped predictions, not quadratic estimates",
            "rules": {
                "test_read": False, "test_row_aggregation": False,
                "target_feature_history": "official seasons strictly before target year",
                "source_preprocessing": "fit on source R rows only",
                "target_correction_mean_subtracted": False,
                "seed_std_is_statistical_significance": False,
                "external_OOF_bank_training_code_audited": False,
            },
            "model_path_reserved_not_written": str(MODEL_PATH),
            "elapsed_seconds": time.time() - started,
        })
        write_json(VALIDATION_DIR / "run_status.json", {"status": "complete"})
        print("\n2024 historical report (within-R BSS points):", flush=True)
        print(year_summary.loc[year_summary["validation_year"].eq(REPORT_YEAR)].to_string(
            index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
        print(f"Saved: {VALIDATION_DIR.resolve()}", flush=True)
        print("No final model or submission was produced.", flush=True)
    except Exception as exc:
        write_json(VALIDATION_DIR / "run_status.json",
                   {"status": "failed", "exception_type": type(exc).__name__,
                    "message": str(exc), "partial_results_are_not_validated": True})
        raise


if __name__ == "__main__":
    main()

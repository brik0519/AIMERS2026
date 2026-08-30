"""rresid_v003 — Sens25 Ridge + original CORE routing temporal audit.

Purpose
-------
rresid_v002 showed that shift_v002/v003-style Sens25 Ridge is materially more
stable than generic residual representations, but the v002 audit applied the
correction to ALL R rows.  The original shift layer applied its sensitivity
correction only to route_segment == 'CORE'.

This experiment reuses the frozen Treatment OOF residual bank from rresid_v001
and asks whether restoring that router rescues the 2024 transition.

Important scope
---------------
* Base = Treatment OOF from rresid_v001.
* Ridge fit source = ALL source-year R residual rows, as in rresid_v002.
* Apply target = CORE only for the main routed arms.
* FUTURES/F rows are outside this R-branch audit.
* The full shift_v003 Phase-5 used its own adjusted OOF over all segments;
  this experiment intentionally tests the shift Sens25/router inductive bias
  on the current Treatment R branch instead.
* No final model is trained. No test rows are used.

Arms
----
R0 CONSTANT_S05
   previous-year mean R residual * 0.50 on all R (Level reference)

R1 SENS25_SHIFT_ALLR_S075
   rresid_v002 reproduction: Sens25 Ridge(no intercept) * 0.75 on all R

R2 SENS25_SHIFT_CORE_S075   [PRIMARY]
   same correction, applied only to route_segment == CORE

R3 SENS25_REFERENCE_CORE_S075
   same Sens25 but fit_intercept=True, CORE only; Level/intercept diagnostic

R4 CONST_S05_PLUS_SENS25_SHIFT_CORE_S075
   all-R Constant Level correction + CORE-only Sens25 Shape correction
   (secondary decomposition candidate; no additional fit/tuning)

Temporal protocol
-----------------
For target t in [2021, 2022, 2023, 2024]:
  fit correction on Treatment OOF residuals from t-1 only
  apply to Treatment OOF predictions in t.

Router
------
Original shift router:
  FUTURES     : game_type == 'F'
  ANCHOR_TEAM : non-F and anchor team involved
  CORE        : remaining non-F rows

The original anchor team was discovered from F rows with season <= 2022.
For strict temporal audit, each transition also discovers the anchor using only
F history <= source_year.  The script asserts that this causal anchor equals the
original fixed anchor; if not, it fails loudly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# =============================================================================
# Paths / identifiers
# =============================================================================

EXPERIMENT_NAME = "rresid_v003"

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

TRANSITION_METRICS_PATH = VALIDATION_DIR / "transition_metrics.csv"
SEGMENT_METRICS_PATH = VALIDATION_DIR / "segment_metrics.csv"
PREDICTIONS_PATH = VALIDATION_DIR / "predictions.csv"
YEAR_SUMMARY_PATH = VALIDATION_DIR / "year_summary.csv"
SEED_SUMMARY_PATH = VALIDATION_DIR / "seed_summary.csv"
ROUTER_AUDIT_PATH = VALIDATION_DIR / "router_audit.csv"
META_PATH = VALIDATION_DIR / "final_metadata.json"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

EVAL_YEARS = [2021, 2022, 2023, 2024]

# shift_v002/v003 frozen settings
RIDGE_ALPHA = 1000.0
SHIFT_SCALE = 0.75
CONSTANT_SCALE = 0.50
SHIFT_SEGMENT = "CORE"
ANCHOR_DISCOVERY_MAX_SEASON = 2022

SENS_HISTORY_YEARS = 3
SENS_DECAY = 0.65
SENS_PARENT_STRENGTH = 100.0
SENS_HAND_STRENGTH = 35.0
SENS_PRESSURE_HAND_STRENGTH = 45.0

PRIMARY_ARM = "SENS25_SHIFT_CORE_S075"
COMBINED_ARM = "CONST_S05_PLUS_SENS25_SHIFT_CORE_S075"

ARMS = [
    "CONSTANT_S05",
    "SENS25_SHIFT_ALLR_S075",
    PRIMARY_ARM,
    "SENS25_REFERENCE_CORE_S075",
    COMBINED_ARM,
]


# =============================================================================
# Metrics
# =============================================================================


def error_decomposition(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    err = p - y
    level = float(np.mean(err) ** 2)
    total = float(np.mean(err ** 2))
    shape = float(total - level)
    return {"total": total, "level": level, "shape": shape}


def gain_decomposition(y, base, pred) -> dict:
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    b0 = error_decomposition(y, base)
    b1 = error_decomposition(y, pred)
    rate = float(np.mean(y))
    baseline = rate * (1.0 - rate)
    if baseline <= 0.0:
        raise RuntimeError("baseline Brier <= 0")
    g = {
        k: 100000.0 * (b0[k] - b1[k]) / baseline
        for k in ["total", "level", "shape"]
    }
    if abs(g["total"] - g["level"] - g["shape"]) > 1e-6:
        raise RuntimeError("Total != Level + Shape")
    return g


def metric_row(*, arm, year, source_year, seed, segment, y, base, pred, correction):
    g = gain_decomposition(y, base, pred)
    corr = np.asarray(correction, dtype=np.float64)
    return {
        "arm": arm,
        "validation_year": int(year),
        "source_year": int(source_year),
        "base_seed": int(seed),
        "segment": str(segment),
        "n_rows": int(len(y)),
        "R_total_gain_pt": float(g["total"]),
        "R_level_gain_pt": float(g["level"]),
        "R_shape_gain_pt": float(g["shape"]),
        "total_win": bool(g["total"] > 1e-8),
        "shape_win": bool(g["shape"] > 1e-6),
        "correction_mean": float(np.mean(corr)) if len(corr) else 0.0,
        "correction_std": float(np.std(corr)) if len(corr) else 0.0,
        "correction_abs_mean": float(np.mean(np.abs(corr))) if len(corr) else 0.0,
    }


# =============================================================================
# Original shift router
# =============================================================================


def discover_anchor_team(screen: pd.DataFrame, max_season: int) -> int:
    """Discover the persistent F anchor using only rows up to max_season."""
    f_hist = screen.loc[
        screen[GAME_TYPE_COL].eq("F")
        & pd.to_numeric(screen[SEASON_COL], errors="raise").astype(int).le(int(max_season))
    ]
    if f_hist.empty:
        raise RuntimeError(f"no F history for router cutoff={max_season}")

    teams = sorted(
        set(f_hist["pitcher_team_id"].dropna().unique())
        | set(f_hist["batter_team_id"].dropna().unique())
    )
    involvement = {
        int(t): float(
            (
                (f_hist["pitcher_team_id"] == t)
                | (f_hist["batter_team_id"] == t)
            ).mean()
        )
        for t in teams
    }
    anchors = [t for t, v in involvement.items() if v > 0.999999]
    if len(anchors) != 1:
        raise RuntimeError(
            f"expected exactly one persistent F anchor at cutoff={max_season}; "
            f"anchors={anchors}, involvement={involvement}"
        )
    return int(anchors[0])


def attach_route_segment(frame: pd.DataFrame, anchor_team_id: int) -> pd.DataFrame:
    out = frame.copy()
    out["route_segment"] = np.where(
        out[GAME_TYPE_COL].eq("F"),
        "FUTURES",
        np.where(
            (out["pitcher_team_id"] == anchor_team_id)
            | (out["batter_team_id"] == anchor_team_id),
            "ANCHOR_TEAM",
            "CORE",
        ),
    )
    return out


# =============================================================================
# shift_v002/v003 Sens25 — faithful representation
# =============================================================================


def _pressure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame.copy()
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int)
    r["sens_pressure"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(balls == 3, "3ball", np.where(strikes == 2, "2strike", "normal")),
    )
    r["sens_pressure_bhand"] = (
        r["sens_pressure"].astype(str) + "|" + r["batter_hand"].astype(str)
    )
    return r


def _weighted_sensitivity_table(history, ctx, strength, cutoff_year):
    h = _pressure_frame(history.reset_index(drop=True))
    seasons = pd.to_numeric(h[SEASON_COL], errors="raise").astype(int)
    h["_w"] = np.power(
        SENS_DECAY,
        np.maximum(int(cutoff_year) - 1 - seasons, 0),
    ).astype(np.float64)
    h["_wy"] = h["_w"] * pd.to_numeric(h[TARGET], errors="coerce").to_numpy(np.float64)

    mu = float(np.average(h[TARGET].to_numpy(np.float64), weights=h["_w"]))
    pg = (
        h.groupby("pitcher_id", sort=False)
        .agg(sw=("_w", "sum"), sy=("_wy", "sum"))
        .reset_index()
    )
    pg["parent_rate"] = (
        pg["sy"] + SENS_PARENT_STRENGTH * mu
    ) / (
        pg["sw"] + SENS_PARENT_STRENGTH
    )

    g = (
        h.groupby(["pitcher_id", ctx], sort=False)
        .agg(sw=("_w", "sum"), sy=("_wy", "sum"))
        .reset_index()
        .merge(pg[["pitcher_id", "parent_rate"]], on="pitcher_id", how="left")
    )
    g["rate"] = (
        g["sy"] + float(strength) * g["parent_rate"]
    ) / (
        g["sw"] + float(strength)
    )
    g["delta"] = g["rate"] - g["parent_rate"]
    g["reliability"] = g["sw"] / (g["sw"] + float(strength))
    return g[["pitcher_id", ctx, "delta", "reliability"]]


def build_sensitivity_specs(train: pd.DataFrame, cutoff_year: int) -> list:
    seasons = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)
    min_year = int(seasons.min())
    lo = max(min_year, int(cutoff_year) - SENS_HISTORY_YEARS)
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


def transform_sensitivity_specs(specs: list, rows: pd.DataFrame) -> pd.DataFrame:
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
        raise RuntimeError(f"sensitivity feature count != 25: {z.shape}")
    if not np.isfinite(z.to_numpy(np.float64)).all():
        raise RuntimeError("Sens25 contains NaN/Inf")
    return z


def sensitivity25_matrices(train, source_year, target_year, source_rows, target_rows):
    # shift-faithful: specs are built separately for each cutoff year.
    source_specs = build_sensitivity_specs(train, source_year)
    target_specs = build_sensitivity_specs(train, target_year)
    A = transform_sensitivity_specs(source_specs, source_rows).to_numpy(np.float64)
    B = transform_sensitivity_specs(target_specs, target_rows).to_numpy(np.float64)

    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Az = np.clip((A - mu) / sd, -8.0, 8.0)
    Bz = np.clip((B - mu) / sd, -8.0, 8.0)
    return Az, Bz, mu, sd


def ridge_predict_rowwise(model: Ridge, X) -> np.ndarray:
    """Deterministic row-independent Ridge prediction by fixed-order accumulation."""
    X64 = np.asarray(X, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    icpt = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    acc = np.full(X64.shape[0], float(icpt[0]) if icpt.size else 0.0, dtype=np.float64)
    for j in range(X64.shape[1]):
        acc = acc + X64[:, j] * coef[j]
    return acc


def fit_predict_sens25_pair(Az, residual, Bz):
    y = np.asarray(residual, dtype=np.float64)
    ref = Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
        max_iter=200,
        tol=1e-5,
        fit_intercept=True,
    )
    shift = Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
        max_iter=200,
        tol=1e-5,
        fit_intercept=False,
    )
    ref.fit(Az, y)
    shift.fit(Az, y)

    corr_ref = ridge_predict_rowwise(ref, Bz)
    corr_shift = ridge_predict_rowwise(shift, Bz)
    coef_gap = float(np.max(np.abs(np.asarray(ref.coef_) - np.asarray(shift.coef_))))

    return corr_shift, corr_ref, {
        "reference_intercept": float(np.asarray(ref.intercept_).reshape(-1)[0]),
        "coef_gap_max": coef_gap,
        "shift_coef_l2": float(np.linalg.norm(shift.coef_)),
        "source_shift_corr_mean": float(np.mean(ridge_predict_rowwise(shift, Az))),
        "source_reference_corr_mean": float(np.mean(ridge_predict_rowwise(ref, Az))),
    }


# =============================================================================
# Prediction arms
# =============================================================================


def apply_all(base, corr, scale):
    return np.clip(
        np.asarray(base, np.float64) + float(scale) * np.asarray(corr, np.float64),
        0.0,
        1.0,
    )


def apply_masked(base, corr, mask, scale):
    p = np.asarray(base, np.float64).copy()
    c = np.asarray(corr, np.float64)
    m = np.asarray(mask, dtype=bool)
    p[m] = np.clip(p[m] + float(scale) * c[m], 0.0, 1.0)
    return p


def apply_constant_plus_core(base, const_corr, sens_corr, core_mask):
    # Level expert on all R, then Shape expert on CORE only.
    p = apply_all(base, const_corr, CONSTANT_SCALE)
    m = np.asarray(core_mask, dtype=bool)
    c = np.asarray(sens_corr, np.float64)
    p[m] = np.clip(p[m] + SHIFT_SCALE * c[m], 0.0, 1.0)
    return p


# =============================================================================
# Summaries / gates
# =============================================================================


def build_year_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    allr = metrics.loc[metrics["segment"].eq("ALL_R")]
    return (
        allr.groupby(["arm", "validation_year"], as_index=False)
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


def build_seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    allr = metrics.loc[metrics["segment"].eq("ALL_R")]
    return (
        allr.groupby(["arm", "base_seed"], as_index=False)
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


def summarize_gates(metrics: pd.DataFrame, year_summary: pd.DataFrame) -> dict:
    y24 = year_summary.loc[
        year_summary["validation_year"].eq(2024)
        & year_summary["arm"].isin([PRIMARY_ARM, COMBINED_ARM])
    ].set_index("arm")

    out = {}
    for arm in [PRIMARY_ARM, COMBINED_ARM]:
        if arm not in y24.index:
            continue
        row = y24.loc[arm]
        temporal = year_summary.loc[year_summary["arm"].eq(arm)]
        core24 = metrics.loc[
            metrics["arm"].eq(arm)
            & metrics["validation_year"].eq(2024)
            & metrics["segment"].eq("CORE")
        ]

        out[arm] = {
            "2024_allR_total_3of3_positive": bool(int(row["total_win_count"]) == 3),
            "2024_allR_shape_3of3_positive": bool(int(row["shape_win_count"]) == 3),
            "2024_allR_mean_total_positive": bool(float(row["mean_R_total_gain_pt"]) > 0.0),
            "2024_allR_mean_shape_positive": bool(float(row["mean_R_shape_gain_pt"]) > 0.0),
            "2024_core_total_3of3_positive": bool((core24["R_total_gain_pt"] > 0).all()),
            "2024_core_shape_3of3_positive": bool((core24["R_shape_gain_pt"] > 0).all()),
            "2024_core_mean_total_pt": float(core24["R_total_gain_pt"].mean()),
            "2024_core_mean_shape_pt": float(core24["R_shape_gain_pt"].mean()),
            "temporal_allR_total_positive_years": int((temporal["mean_R_total_gain_pt"] > 0).sum()),
            "temporal_allR_shape_positive_years": int((temporal["mean_R_shape_gain_pt"] > 0).sum()),
            # Original shift-like diagnostics; baseline is Treatment here, not adjusted_oof.
            "diagnostic_g23_ge_8": bool(float(
                temporal.loc[temporal["validation_year"].eq(2023), "mean_R_total_gain_pt"].iloc[0]
            ) >= 8.0),
            "diagnostic_g24_ge_10": bool(float(row["mean_R_total_gain_pt"]) >= 10.0),
            "diagnostic_core24_ge_15": bool(float(core24["R_total_gain_pt"].mean()) >= 15.0),
        }
    return out


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
# Main
# =============================================================================


def main():
    t0 = time.time()

    print("=" * 122)
    print(EXPERIMENT_NAME)
    print("=" * 122)
    print("Purpose       : shift Sens25 Ridge + original CORE router temporal audit")
    print(f"Bank source   : {BANK_SOURCE_PATH}")
    print(f"Eval years    : {EVAL_YEARS}")
    print(f"Ridge         : alpha={RIDGE_ALPHA}, fit_intercept=False, solver=lsqr")
    print(f"Sens history  : <=3 seasons, decay={SENS_DECAY}")
    print(f"Shift scale   : {SHIFT_SCALE}")
    print(f"Constant      : previous-year residual mean * {CONSTANT_SCALE}")
    print("Router        : FUTURES / ANCHOR_TEAM / CORE (original shift definition)")
    print("Ridge fit     : source ALL-R residuals; routed application is CORE-only")
    print("Final model   : NOT TRAINED")
    print("Test usage    : NONE")
    print("=" * 122)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)
    if not BANK_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Residual bank missing: {BANK_SOURCE_PATH}\n"
            "Run rresid_v001 first, or set RRESID_BANK_SOURCE."
        )
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보호를 위해 중단: {VALIDATION_DIR}\n"
            "기존 폴더를 이동/삭제 후 다시 실행하세요."
        )
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    train[SEASON_COL] = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)

    required = {
        ID, TARGET, SEASON_COL, GAME_TYPE_COL,
        "pitcher_id", "batter_hand", "balls_before", "strikes_before",
        "pitcher_team_id", "batter_team_id",
    }
    missing = sorted(required - set(train.columns))
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
        raise RuntimeError(f"residual bank missing OOF years: {missing_years}")

    # Original fixed router anchor, as shift_v003 implemented it.
    fixed_anchor = discover_anchor_team(train, ANCHOR_DISCOVERY_MAX_SEASON)
    print(f"Original fixed anchor team (F<=2022): {fixed_anchor}")

    metric_rows = []
    pred_parts = []
    router_rows = []
    diag_rows = []

    for target_year in EVAL_YEARS:
        source_year = target_year - 1

        # Strict temporal router audit: discover anchor only from information
        # available by source_year, then assert it equals the original fixed router.
        causal_anchor = discover_anchor_team(train, source_year)
        if causal_anchor != fixed_anchor:
            raise RuntimeError(
                f"router anchor instability {source_year}->{target_year}: "
                f"causal={causal_anchor}, fixed={fixed_anchor}"
            )

        raw_source = train.loc[
            train[SEASON_COL].eq(source_year)
            & ~train[GAME_TYPE_COL].fillna("__MISSING__").astype(str).eq("F")
        ].copy().reset_index(drop=True)
        raw_target = train.loc[
            train[SEASON_COL].eq(target_year)
            & ~train[GAME_TYPE_COL].fillna("__MISSING__").astype(str).eq("F")
        ].copy().reset_index(drop=True)

        raw_source = attach_route_segment(raw_source, causal_anchor)
        raw_target = attach_route_segment(raw_target, causal_anchor)

        source_counts = raw_source["route_segment"].value_counts().to_dict()
        target_counts = raw_target["route_segment"].value_counts().to_dict()
        if set(raw_target["route_segment"].unique()) - {"CORE", "ANCHOR_TEAM"}:
            raise RuntimeError("R-only target unexpectedly contains FUTURES/unknown segment")

        router_rows.append({
            "source_year": int(source_year),
            "target_year": int(target_year),
            "fixed_anchor_team": int(fixed_anchor),
            "causal_anchor_team": int(causal_anchor),
            "anchor_equal": bool(fixed_anchor == causal_anchor),
            "source_core_n": int(source_counts.get("CORE", 0)),
            "source_anchor_team_n": int(source_counts.get("ANCHOR_TEAM", 0)),
            "target_core_n": int(target_counts.get("CORE", 0)),
            "target_anchor_team_n": int(target_counts.get("ANCHOR_TEAM", 0)),
            "target_core_share_R": float(target_counts.get("CORE", 0) / len(raw_target)),
        })

        print("\n" + "#" * 122)
        print(
            f"TARGET={target_year} SOURCE={source_year} | anchor={causal_anchor} | "
            f"source_R={len(raw_source):,} target_R={len(raw_target):,} | "
            f"target CORE={target_counts.get('CORE', 0):,} "
            f"ANCHOR_TEAM={target_counts.get('ANCHOR_TEAM', 0):,}"
        )
        print("#" * 122)

        Az, Bz, sens_mu, sens_sd = sensitivity25_matrices(
            train=train,
            source_year=source_year,
            target_year=target_year,
            source_rows=raw_source,
            target_rows=raw_target,
        )
        core = raw_target["route_segment"].eq(SHIFT_SEGMENT).to_numpy()
        anchor_mask = raw_target["route_segment"].eq("ANCHOR_TEAM").to_numpy()

        for seed in seeds:
            sb = bank.loc[
                bank["validation_year"].eq(source_year)
                & bank["base_seed"].eq(seed)
            ].set_index(ID)
            tb = bank.loc[
                bank["validation_year"].eq(target_year)
                & bank["base_seed"].eq(seed)
            ].set_index(ID)

            if not raw_source[ID].isin(sb.index).all():
                raise RuntimeError(f"source bank alignment failed {source_year} seed={seed}")
            if not raw_target[ID].isin(tb.index).all():
                raise RuntimeError(f"target bank alignment failed {target_year} seed={seed}")

            source_base = raw_source[ID].map(sb["treatment_pred"]).to_numpy(np.float64)
            source_y = raw_source[ID].map(sb["y_true"]).to_numpy(np.float64)
            source_resid = source_y - source_base

            target_base = raw_target[ID].map(tb["treatment_pred"]).to_numpy(np.float64)
            target_y = raw_target[ID].map(tb["y_true"]).to_numpy(np.float64)

            corr_shift, corr_ref, fit_diag = fit_predict_sens25_pair(Az, source_resid, Bz)
            const_corr = np.full(len(target_y), float(source_resid.mean()), dtype=np.float64)

            preds = {
                "CONSTANT_S05": apply_all(target_base, const_corr, CONSTANT_SCALE),
                "SENS25_SHIFT_ALLR_S075": apply_all(target_base, corr_shift, SHIFT_SCALE),
                PRIMARY_ARM: apply_masked(target_base, corr_shift, core, SHIFT_SCALE),
                "SENS25_REFERENCE_CORE_S075": apply_masked(target_base, corr_ref, core, SHIFT_SCALE),
                COMBINED_ARM: apply_constant_plus_core(target_base, const_corr, corr_shift, core),
            }

            effective_corr = {
                "CONSTANT_S05": CONSTANT_SCALE * const_corr,
                "SENS25_SHIFT_ALLR_S075": SHIFT_SCALE * corr_shift,
                PRIMARY_ARM: SHIFT_SCALE * corr_shift * core.astype(np.float64),
                "SENS25_REFERENCE_CORE_S075": SHIFT_SCALE * corr_ref * core.astype(np.float64),
                COMBINED_ARM: (
                    CONSTANT_SCALE * const_corr + SHIFT_SCALE * corr_shift * core.astype(np.float64)
                ),
            }

            print(
                f"\n[TRANSITION] {source_year}->{target_year} seed={seed} | "
                f"source resid mean={source_resid.mean():+.8f} std={source_resid.std():.8f} | "
                f"CORE share={core.mean():.4f}"
            )

            payload = {
                ID: raw_target[ID].to_numpy(),
                "validation_year": int(target_year),
                "source_year": int(source_year),
                "base_seed": int(seed),
                "route_segment": raw_target["route_segment"].to_numpy(),
                "y_true": target_y,
                "treatment_pred": target_base,
                "sens25_shift_correction_raw": corr_shift,
                "sens25_reference_correction_raw": corr_ref,
                "constant_correction_raw": const_corr,
            }

            for arm in ARMS:
                p = preds[arm]
                c = effective_corr[arm]

                all_row = metric_row(
                    arm=arm,
                    year=target_year,
                    source_year=source_year,
                    seed=seed,
                    segment="ALL_R",
                    y=target_y,
                    base=target_base,
                    pred=p,
                    correction=c,
                )
                metric_rows.append(all_row)

                for seg_name, mask in [("CORE", core), ("ANCHOR_TEAM", anchor_mask)]:
                    if not mask.any():
                        continue
                    metric_rows.append(metric_row(
                        arm=arm,
                        year=target_year,
                        source_year=source_year,
                        seed=seed,
                        segment=seg_name,
                        y=target_y[mask],
                        base=target_base[mask],
                        pred=p[mask],
                        correction=c[mask],
                    ))

                payload[arm.lower() + "__prediction"] = p

                print(
                    f"{arm:<45} ALL_R "
                    f"Total={all_row['R_total_gain_pt']:+9.3f} "
                    f"Level={all_row['R_level_gain_pt']:+9.3f} "
                    f"Shape={all_row['R_shape_gain_pt']:+9.3f}"
                )

            # Print main routed segment decomposition explicitly.
            for arm in [PRIMARY_ARM, COMBINED_ARM]:
                z = [
                    r for r in metric_rows
                    if r["arm"] == arm
                    and r["validation_year"] == target_year
                    and r["base_seed"] == seed
                    and r["segment"] in ["CORE", "ANCHOR_TEAM"]
                ]
                for r in z:
                    print(
                        f"  {arm:<43} {r['segment']:<11} "
                        f"Total={r['R_total_gain_pt']:+9.3f} "
                        f"Level={r['R_level_gain_pt']:+9.3f} "
                        f"Shape={r['R_shape_gain_pt']:+9.3f}"
                    )

            diag_rows.append({
                "source_year": int(source_year),
                "target_year": int(target_year),
                "base_seed": int(seed),
                "source_residual_mean": float(source_resid.mean()),
                "source_residual_std": float(source_resid.std()),
                "core_share_target_R": float(core.mean()),
                "sens_mu_abs_mean": float(np.mean(np.abs(sens_mu))),
                "sens_sd_mean": float(np.mean(sens_sd)),
                **fit_diag,
            })
            pred_parts.append(pd.DataFrame(payload))

    metrics = pd.DataFrame(metric_rows)
    allr_metrics = metrics.loc[metrics["segment"].eq("ALL_R")].copy()
    segment_metrics = metrics.loc[~metrics["segment"].eq("ALL_R")].copy()
    predictions = pd.concat(pred_parts, ignore_index=True)
    year_summary = build_year_summary(metrics)
    seed_summary = build_seed_summary(metrics)
    router_audit = pd.DataFrame(router_rows)
    gates = summarize_gates(metrics, year_summary)

    allr_metrics.to_csv(TRANSITION_METRICS_PATH, index=False, encoding="utf-8-sig")
    segment_metrics.to_csv(SEGMENT_METRICS_PATH, index=False, encoding="utf-8-sig")
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    year_summary.to_csv(YEAR_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    seed_summary.to_csv(SEED_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    router_audit.to_csv(ROUTER_AUDIT_PATH, index=False, encoding="utf-8-sig")

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "purpose": "Sens25 Ridge + original CORE routing temporal audit on Treatment R OOF residuals",
        "validation_only": True,
        "final_model_saved": False,
        "bank_source": str(BANK_SOURCE_PATH),
        "eval_years": EVAL_YEARS,
        "base_seeds": seeds,
        "temporal_protocol": "source year t-1 Treatment OOF residual -> target year t",
        "router": {
            "definition": "FUTURES if F; else ANCHOR_TEAM if anchor team involved; else CORE",
            "original_anchor_discovery_max_season": ANCHOR_DISCOVERY_MAX_SEASON,
            "original_fixed_anchor_team": int(fixed_anchor),
            "audit": router_audit.to_dict(orient="records"),
            "strict_temporal_note": "each transition rediscovers anchor using F rows through source_year and asserts equality to fixed original anchor",
        },
        "sens25": {
            "n_features": 25,
            "history_years": SENS_HISTORY_YEARS,
            "decay": SENS_DECAY,
            "parent_strength": SENS_PARENT_STRENGTH,
            "hand_strength": SENS_HAND_STRENGTH,
            "pressure_hand_strength": SENS_PRESSURE_HAND_STRENGTH,
            "ridge_alpha": RIDGE_ALPHA,
            "solver": "lsqr",
            "fit_intercept_primary": False,
            "scale": SHIFT_SCALE,
            "fit_source": "all source-year R rows",
            "apply_primary": "target CORE rows only",
        },
        "arms": {
            "CONSTANT_S05": "all-R previous-year residual mean * 0.50",
            "SENS25_SHIFT_ALLR_S075": "no-intercept Sens25 * .75 on all R; rresid_v002 comparison",
            PRIMARY_ARM: "no-intercept Sens25 * .75 on CORE only; primary router audit",
            "SENS25_REFERENCE_CORE_S075": "intercept Sens25 * .75 on CORE only; Level diagnostic",
            COMBINED_ARM: "constant .50 all R + no-intercept Sens25 .75 CORE only",
        },
        "gates": gates,
        "diagnostics": diag_rows,
        "rule_compliance": {
            "official_train_labels_only": True,
            "test_rows_used": False,
            "test_row_aggregation": False,
            "residual_source": "strictly previous OOF season only",
            "sensitivity_history": "official labeled seasons strictly before each cutoff year",
            "router_current_transition_source": "only F covariates through source_year; no target labels used",
        },
        "scope_note": "This is not a full reproduction of shift_v003 Phase 5: base is Treatment R OOF, and F residuals are excluded from Ridge fit. It isolates Sens25 + CORE routing for the current R branch.",
        "outputs": {
            "transition_metrics": str(TRANSITION_METRICS_PATH),
            "segment_metrics": str(SEGMENT_METRICS_PATH),
            "predictions": str(PREDICTIONS_PATH),
            "year_summary": str(YEAR_SUMMARY_PATH),
            "seed_summary": str(SEED_SUMMARY_PATH),
            "router_audit": str(ROUTER_AUDIT_PATH),
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    META_PATH.write_text(
        json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 122)
    print("YEAR SUMMARY — ALL R")
    print("=" * 122)
    print(year_summary.to_string(index=False))

    print("\n" + "=" * 122)
    print("2024 SEGMENT SUMMARY — PRIMARY / COMBINED")
    print("=" * 122)
    seg24 = segment_metrics.loc[
        segment_metrics["validation_year"].eq(2024)
        & segment_metrics["arm"].isin([PRIMARY_ARM, COMBINED_ARM])
    ]
    if len(seg24):
        print(
            seg24.groupby(["arm", "segment"], as_index=False)
            .agg(
                seed_count=("base_seed", "size"),
                total_win_count=("total_win", "sum"),
                shape_win_count=("shape_win", "sum"),
                mean_total=("R_total_gain_pt", "mean"),
                mean_level=("R_level_gain_pt", "mean"),
                mean_shape=("R_shape_gain_pt", "mean"),
                min_total=("R_total_gain_pt", "min"),
                min_shape=("R_shape_gain_pt", "min"),
            )
            .to_string(index=False)
        )

    print("\n" + "=" * 122)
    print("GATES")
    print("=" * 122)
    for arm, gate in gates.items():
        print(arm)
        for k, v in gate.items():
            print(f"  {k}: {v}")

    print("\nSaved:")
    for path in [
        TRANSITION_METRICS_PATH,
        SEGMENT_METRICS_PATH,
        PREDICTIONS_PATH,
        YEAR_SUMMARY_PATH,
        SEED_SUMMARY_PATH,
        ROUTER_AUDIT_PATH,
        META_PATH,
    ]:
        print(path)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("=" * 122)


if __name__ == "__main__":
    main()

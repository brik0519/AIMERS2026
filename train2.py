"""rresid_v004: fixed Sens25 x 0.15 on frozen FINAL temporal OOF residuals.

Validation only. Requires a full-stack OOF bank and its provenance manifest.
No baseline refit, model serialization, test access, or submission generation.
Paths default to the repository working directory, never a machine-specific path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

EXPERIMENT_NAME = "rresid_v004"
DATA_DIR = Path("./data")
MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"  # Reserved; never written.
VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"  # Never opened.
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"  # Never opened.
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"  # Never written.
ID, TARGET, SEASON_COL, GAME_TYPE_COL = "row_id", "control_success", "season", "game_type"
RIDGE_ALPHA = 1000.0
FIXED_SCALE = 0.15
SENS_HISTORY_YEARS = 3
SENS_DECAY = 0.65
SENS_PARENT_STRENGTH = 100.0
SENS_HAND_STRENGTH = 35.0
SENS_PRESSURE_HAND_STRENGTH = 45.0
POINT_EPS = 1e-6
SENS25_HELPERS_SHA256 = "6f10fb32f37d61c28ec1a4ed6018bbd78703bd751faad3e7ec31df092dbe3715"

# The Sens25 feature functions below are copied verbatim from rresid_v003.
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{name}: missing columns {missing}")


def strict_integer(frame, column):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(),
            f"{column}: expected finite integers")
    frame[column] = values.astype(np.int64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-id", required=True, help="Exact baseline recipe/version; no automatic guessing.")
    p.add_argument("--bank", type=Path, required=True, help="FINAL OOF CSV or CSV.gz, including R and F.")
    p.add_argument("--manifest", type=Path, required=True, help="OOF provenance JSON; see README.")
    p.add_argument("--years", required=True, help="Explicit evaluation years, e.g. 2023,2024; each needs year-1 OOF.")
    p.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    p.add_argument("--output-dir", type=Path, default=VALIDATION_DIR)
    p.add_argument("--check-only", action="store_true", help="Validate inputs without creating output or fitting Ridge.")
    args = p.parse_args()
    try:
        years = [int(x) for x in args.years.split(",")]
    except ValueError:
        p.error("--years must be comma-separated integer years")
    if not years or len(set(years)) != len(years) or max(years) > 2024:
        p.error("Use unique historical evaluation years through 2024")
    args.years = sorted(years)
    for field in ("bank", "manifest", "train_path", "output_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    return args


def read_inputs(args):
    for name in ("bank", "manifest", "train_path"):
        path = getattr(args, name)
        require(path.is_file(), f"Missing {name}: {path}. L1-only CSVs cannot replace a FINAL OOF bank.")
    if not args.check_only:
        require(not args.output_dir.exists(), f"Refusing to overwrite existing results: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    require(manifest.get("schema_version") == 1, "Expected manifest schema_version=1")
    require(manifest.get("baseline_id") == args.baseline_id, "Baseline ID differs from the manifest")
    require(manifest.get("prediction_stage") == "FINAL", "Baseline must include the full final stack, not L1")
    require(manifest.get("oof_kind") == "expanding_temporal", "Expected expanding temporal OOF")
    require(manifest.get("row_scope") == "ALL", "Full R/F OOF is required to measure F preservation")
    require(manifest.get("replicate_mode") in
            ("independent_base_seeds", "fixed_seed_ensemble", "single_base_model"), "Unknown replicate_mode")
    replicas = manifest.get("replicate_ids", [])
    require(isinstance(replicas, list) and replicas and all(isinstance(x, str) and x.strip() for x in replicas),
            "replicate_ids must contain nonempty strings")
    require(len(set(replicas)) == len(replicas), "Duplicate replicate IDs")
    if manifest["replicate_mode"] != "independent_base_seeds":
        require(len(replicas) == 1, "A fixed ensemble or single model is one replicate; do not duplicate its predictions")
    else:
        require(len(replicas) >= 2, "independent_base_seeds requires at least two distinct base runs")
    components = manifest.get("full_stack_components", [])
    require(isinstance(components, list) and components and all(isinstance(x, str) and x.strip() for x in components),
            "Declare every fitted final-stack component, including L2/SHIFT when present")
    require(len(set(components)) == len(components), "Duplicate stack component names")
    code = Path(manifest.get("generation_code_path", ""))
    if not code.is_absolute():
        code = args.manifest.parent / code
    require(code.is_file(), f"Missing OOF generation code: {code}")
    hashes = {"bank_sha256": digest(args.bank), "official_train_sha256": digest(args.train_path),
              "generation_code_sha256": digest(code)}
    for key, actual in hashes.items():
        require(manifest.get(key) == actual, f"{key} mismatch; do not mix baseline artifacts or change provenance")

    cols = [ID, TARGET, SEASON_COL, GAME_TYPE_COL, "pitcher_id", "batter_hand", "balls_before", "strikes_before"]
    train = pd.read_csv(args.train_path, encoding="utf-8-sig", usecols=cols, dtype={ID: "string"})
    require(train[ID].notna().all() and not train[ID].duplicated().any(), "Official row IDs missing/duplicated")
    strict_integer(train, SEASON_COL)
    train[TARGET] = pd.to_numeric(train[TARGET], errors="raise")
    require(train[TARGET].isin([0, 1]).all(), "Official target must be binary")
    require(train[GAME_TYPE_COL].notna().all(), "Missing game_type; R/F routing is ambiguous")
    train = train.set_index(ID, drop=False)
    bank = pd.read_csv(args.bank, encoding="utf-8-sig", dtype={ID: "string", "replicate_id": "string"})
    check_columns(bank, [ID, "validation_year", "replicate_id", "y_true", "final_pred"], "FINAL OOF bank")
    strict_integer(bank, "validation_year")
    require(bank[ID].notna().all() and bank["replicate_id"].notna().all(), "OOF keys cannot be missing")
    require(not bank.duplicated(["validation_year", "replicate_id", ID]).any(), "Duplicate OOF keys")
    for col in ("y_true", "final_pred"):
        bank[col] = pd.to_numeric(bank[col], errors="raise")
        require(np.isfinite(bank[col]).all(), f"{col}: nonfinite values")
    require(bank["final_pred"].between(0, 1).all(), "FINAL probabilities must be in [0,1]")
    require(bank["y_true"].isin([0, 1]).all(), "OOF labels must be binary")
    require(set(bank["replicate_id"]) == set(replicas), "OOF replicate IDs differ from manifest")
    require(set(bank[ID]).issubset(set(train.index)), "OOF row IDs absent from official train")
    aligned = train.loc[bank[ID]]
    require(np.array_equal(aligned[SEASON_COL].to_numpy(), bank["validation_year"].to_numpy()), "OOF row years differ")
    require(np.array_equal(aligned[TARGET].to_numpy(), bank["y_true"].to_numpy()), "OOF labels differ from official train")
    if "residual" in bank:
        residual = pd.to_numeric(bank["residual"], errors="raise").to_numpy(float)
        require(np.allclose(residual, bank["y_true"] - bank["final_pred"], atol=1e-12, rtol=0), "Residual != y - FINAL")
    needed_years = sorted(set(args.years) | {year - 1 for year in args.years})
    require(set(needed_years).issubset(set(bank["validation_year"])),
            f"Missing OOF source/target years. Required: {needed_years}; do not substitute in-sample predictions")
    require(min(needed_years) > int(train[SEASON_COL].min()), "Insufficient earlier official history for Sens25")
    lineage = {}
    for entry in manifest.get("folds", []):
        year, replica = entry.get("validation_year"), entry.get("replicate_id")
        require(type(year) is int and isinstance(replica, str), "Invalid lineage fold key")
        key = (year, replica)
        require(key not in lineage, f"Duplicate lineage: {key}")
        maximum = entry.get("component_fit_max_years", {})
        require(set(maximum) == set(components), f"Incomplete stack lineage: {key}")
        require(all(type(value) is int and value < year for value in maximum.values()),
                f"Same/future-year component training in {key}: {maximum}")
        require(type(entry.get("target_year_used_for_early_stopping")) is bool,
                f"Declare target-year early stopping explicitly: {key}")
        lineage[key] = entry
    folds = {}
    for year in sorted(bank["validation_year"].unique()):
        expected = train.loc[train[SEASON_COL].eq(year), ID]
        require(len(expected) > 0, f"No official rows for {year}")
        for replica in replicas:
            key = (int(year), replica)
            require(key in lineage, f"Missing temporal lineage: {key}")
            fold = bank.loc[bank["validation_year"].eq(year) & bank["replicate_id"].eq(replica)].set_index(ID)
            require(len(fold) == len(expected) and set(fold.index) == set(expected), f"Incomplete all-row OOF: {key}")
            folds[key] = fold.loc[expected]
        if len(replicas) > 1:
            for i, first in enumerate(replicas):
                for second in replicas[i + 1:]:
                    require(not np.array_equal(folds[(int(year), first)].final_pred.to_numpy(),
                                               folds[(int(year), second)].final_pred.to_numpy()),
                            f"Identical base predictions under different replicate IDs in {year}")
    require(set(lineage) == set(folds), "Manifest folds differ from actual OOF folds")
    return train, folds, manifest, hashes, code.resolve()


def decompose(y, pred):
    err = pred - y
    total, level = float(np.mean(err * err)), float(np.mean(err) ** 2)
    return np.array([total, level, total - level])


def measure(y, base, prediction, mask, branch):
    y, base, prediction = y[mask], base[mask], prediction[mask]
    require(len(y) > 0, f"No {branch} rows")
    denominator = float(y.mean() * (1 - y.mean()))
    require(denominator > 0, f"Undefined {branch} Brier reference")
    before, after = decompose(y, base), decompose(y, prediction)
    gains = 100000 * (before - after) / denominator
    require(abs(gains[0] - gains[1] - gains[2]) < POINT_EPS, "Total != Level + Shape")
    return {f"n_{branch}": len(y), f"{branch}_base_brier": float(before[0]),
            f"{branch}_candidate_brier": float(after[0]),
            **{f"{branch}_{name}_gain_pt": float(value) for name, value in zip(("total", "level", "shape"), gains)}}


def run_year(train, folds, replicas, year, output):
    source_year = year - 1
    src = train.loc[train[SEASON_COL].eq(source_year) & train[GAME_TYPE_COL].ne("F")].copy()
    target = train.loc[train[SEASON_COL].eq(year)].copy()
    r = target[GAME_TYPE_COL].ne("F").to_numpy()
    f = ~r
    require(r.any() and f.any(), f"Both R and F rows are required: {year}")
    # Target labels are excluded from every feature transform. History is < year.
    sx = src.drop(columns=[TARGET]).reset_index(drop=True)
    tx = target.drop(columns=[TARGET]).reset_index(drop=True)
    A, B = sensitivity25_matrices(train.reset_index(drop=True), source_year, year, sx, tx.loc[r])
    y = target[TARGET].to_numpy(float)
    result, diagnostics = [], []
    for replica in replicas:
        source = folds[(source_year, replica)].loc[src[ID]]
        base = folds[(year, replica)].loc[target[ID], "final_pred"].to_numpy(float)
        residual = source["y_true"].to_numpy(float) - source["final_pred"].to_numpy(float)
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=200, tol=1e-5, fit_intercept=False)
        model.fit(A, residual)
        c = np.zeros(len(target), dtype=float)
        c[r] = model.predict(B)
        require(np.isfinite(c).all(), "Nonfinite Sens25 correction")
        predicted = base.copy()
        predicted[r] = np.clip(base[r] + FIXED_SCALE * c[r], 0, 1)
        require(np.array_equal(predicted[f], base[f]), "F predictions changed")
        identity = dict(validation_year=year, source_year=source_year, replicate_id=replica)
        for arm, pred in (("NO_CHANGE", base), ("SENS25_FIXED_015", predicted)):
            row = {**identity, "arm": arm, "scale": 0.0 if arm == "NO_CHANGE" else FIXED_SCALE}
            for branch, mask in (("ALL", np.ones(len(y), bool)), ("R", r), ("F", f)):
                row.update(measure(y, base, pred, mask, branch))
            row["F_prediction_max_abs_diff"] = float(np.max(np.abs(pred[f] - base[f])))
            row["clip_count"] = 0 if arm == "NO_CHANGE" else int(np.sum(
                (base[r] + FIXED_SCALE * c[r] < 0) | (base[r] + FIXED_SCALE * c[r] > 1)))
            result.append(row)
        pd.DataFrame({ID: target[ID].to_numpy(), "validation_year": year, "source_year": source_year,
                      "replicate_id": replica, GAME_TYPE_COL: target[GAME_TYPE_COL].to_numpy(), "y_true": y,
                      "final_pred": base, "sens25_correction": c, "candidate_pred": predicted}).to_csv(
                          output / "predictions" / f"fold_{year}_replicate_{replicas.index(replica)}.csv.gz", index=False)
        diagnostics.append({**identity, "source_R_rows": len(src), "feature_count": int(A.shape[1]),
                            "source_residual_mean": float(residual.mean()), "coef_l2": float(np.linalg.norm(model.coef_)),
                            "target_R_correction_mean": float(c[r].mean()), "target_R_correction_std": float(c[r].std()),
                            "coefficients": model.coef_.tolist()})
        print(f"[year={year} replicate={replica}] FINAL + Sens25 x 0.15 complete", flush=True)
    return result, diagnostics


def report_gates(metrics, manifest):
    reports = []
    for year, block in metrics.loc[metrics.arm.eq("SENS25_FIXED_015")].groupby("validation_year"):
        values = block.R_shape_gain_pt.to_numpy(float)
        n = len(values)
        mean, level = float(values.mean()), float(block.R_level_gain_pt.mean())
        std = float(values.std(ddof=1)) if n > 1 else None
        three_seed = manifest["replicate_mode"] == "independent_base_seeds" and n == 3
        tu = 2 * std / math.sqrt(3) if three_seed else None
        tp = max(5.0, tu) if three_seed else None
        primary = bool(np.all(values > POINT_EPS) and mean >= tp) if three_seed else None
        level_pass = abs(level) <= 5.0
        f_pass = bool(block.F_prediction_max_abs_diff.eq(0).all() and block.F_level_gain_pt.ge(-10).all())
        if not three_seed:
            status = "DESCRIPTIVE_ONLY_NO_THREE_SEED_GATE"
        elif primary and level_pass and f_pass:
            status = "POSTHOC_GATES_PASS"
        else:
            status = "POSTHOC_GATE_FAIL"
        reports.append({"validation_year": int(year), "status": status, "replicate_count": n,
                        "shape_mean_pt": mean, "shape_sample_std_pt": std, "shape_wins": int(np.sum(values > POINT_EPS)),
                        "total_mean_pt": float(block.R_total_gain_pt.mean()), "level_mean_pt": level,
                        "T_u": tu, "T_p": tp, "primary_pass": primary, "R_level_pass": level_pass,
                        "F_identity_and_level_pass": f_pass, "deployment_approved": False})
    return {"baseline_id": manifest["baseline_id"], "replicate_mode": manifest["replicate_mode"],
            "interpretation": "Historical posthoc audit; seed variation is not statistical significance.",
            "source_code_lineage_audited": False,
            "lineage_limit": "Manifest, hashes and year bounds checked; metadata alone cannot prove how OOF was generated.",
            "reports": reports, "deployment_approved": False}


def main():
    args = parse_args()
    train, folds, manifest, hashes, code = read_inputs(args)
    print(f"Validated FINAL OOF inputs for {args.baseline_id}; target years={args.years}", flush=True)
    if args.check_only:
        print("Input preflight passed. No output or model was created.")
        return
    protocol = {"experiment_name": EXPERIMENT_NAME, "baseline_id": args.baseline_id,
                "validation_only": True, "historical_posthoc": True, "2024_previously_seen": True,
                "scale_origin": "Fixed 0.15 hypothesis from previously inspected rresid_v001/v002/v003 results",
                "family": "RIDGE_SENS25", "scale": FIXED_SCALE, "ridge_alpha": RIDGE_ALPHA,
                "fit_intercept": False, "source_policy": "strict previous-year FINAL OOF residuals, R rows only",
                "feature_history": "official labeled history strictly before each row year; at most 3 years",
                "standardization": "source R rows only, ddof=0, feature clip +/-8",
                "target_correction_recentered": False, "scale_search": False, "test_read": False,
                "years": args.years, "replicate_mode": manifest["replicate_mode"],
                "replicate_ids": manifest["replicate_ids"], "input_hashes": hashes,
                "sens25_helpers_sha256": SENS25_HELPERS_SHA256,
                "train_script_sha256": digest(__file__), "manifest_sha256": digest(args.manifest),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "deployment_approved": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run_status.json", {"status": "running"})
    try:
        write_json(args.output_dir / "protocol.json", protocol)
        locked = digest(args.output_dir / "protocol.json")
        shutil.copy2(__file__, args.output_dir / "train_source.py")
        shutil.copy2(args.manifest, args.output_dir / "baseline_manifest.json")
        shutil.copy2(code, args.output_dir / "baseline_generation_source.py")
        (args.output_dir / "predictions").mkdir()
        rows, diagnostics = [], []
        for year in args.years:
            found, diag = run_year(train, folds, manifest["replicate_ids"], year, args.output_dir)
            rows.extend(found)
            diagnostics.extend(diag)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        summary = metrics.groupby(["arm", "validation_year"], as_index=False).agg(
            replicate_count=("replicate_id", "size"), mean_R_total_gain_pt=("R_total_gain_pt", "mean"),
            mean_R_level_gain_pt=("R_level_gain_pt", "mean"), mean_R_shape_gain_pt=("R_shape_gain_pt", "mean"),
            std_R_shape_gain_pt=("R_shape_gain_pt", "std"), min_R_shape_gain_pt=("R_shape_gain_pt", "min"),
            mean_ALL_total_gain_pt=("ALL_total_gain_pt", "mean"), mean_F_level_gain_pt=("F_level_gain_pt", "mean"),
            max_F_prediction_abs_diff=("F_prediction_max_abs_diff", "max"), clip_count=("clip_count", "sum"))
        summary.to_csv(args.output_dir / "year_summary.csv", index=False)
        write_json(args.output_dir / "fit_diagnostics.json", diagnostics)
        write_json(args.output_dir / "confirmation.json", report_gates(metrics, manifest))
        require(digest(args.output_dir / "protocol.json") == locked, "Protocol changed during evaluation")
        write_json(args.output_dir / "run_status.json", {"status": "complete", "metric_rows": len(metrics)})
        print(summary.to_string(index=False))
        print(f"Saved validation only: {args.output_dir}")
    except Exception as exc:
        write_json(args.output_dir / "run_status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        sys.exit(f"Preflight/evaluation stopped: {exc}")

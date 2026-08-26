"""V18 Reproduction — R1 / R2 / R3 numerical parity gate.

v18_reference_dump.py 가 만든 ground truth 와 train.py 재구현 산출물을 비교한다.

  R1 Data    : row count / row_id order / target / season / domain3 / pitcher-half split
  R2 Feature : self_state / gseason / pitcher_conditional / exact_asof_state /
               adapter_matrix / v18_stable  (+ 미투입 3블록)
               column names · column order · shape · dtype · NaN mask ·
               max_abs_diff < 1e-7 · mean_abs_diff < 1e-9
  R3 Anchor  : trend_slope / trend_prior (diff < 1e-12)
               p_exact_lgb / p_exact_ridge / p_trend_lgb / p_base / residual_l1
               (max_abs_diff < 1e-6)

사용법:
    # 1) reference 생성
    python3 v18_reference_dump.py --notebook <nb> --data-dir ./data \
        --out validation/v18_repro_v001/reference --year 2020 ... --stage anchor
    # 2) train.py 실행 (V18_REPRO_PHASE=3)
    python3 train.py
    # 3) gate
    python3 v18_gate_check.py \
        --reference validation/v18_repro_v001/reference \
        --candidate validation/v18_repro_v001/matrices \
        --validation-dir validation/v18_repro_v001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R2_MAX_ABS = 1e-7
R2_MEAN_ABS = 1e-9
R3_PRED_MAX_ABS = 1e-6
R3_SCALAR_ABS = 1e-12

BLOCKS = [
    ("self_state", "R2 Self State"),
    ("gseason", "R2 Global Season"),
    ("pitcher_conditional", "R2 Pitcher Conditional"),
    ("exact_asof_state", "R2 Exact ASOF"),
    ("adapter_matrix_surrogate", "R2 Adapter Matrix"),
    ("v18_stable", "R2 Stable"),
]
UNUSED_BLOCKS = [
    ("cross_domain_profile", "R2 [unused] Cross Domain"),
    ("batter_conditional", "R2 [unused] Batter Conditional"),
    ("role_deployment", "R2 [unused] Role Deployment"),
]
ANCHOR_KEYS = [
    ("p_exact_lgb", "R3 Exact LGB"),
    ("p_exact_ridge", "R3 Ridge"),
    ("p_trend_lgb", "R3 Trend LGB"),
    ("p_base", "R3 Anchor"),
    ("residual_l1", "R3 Residual"),
]


def load_matrix(path: Path):
    z = np.load(path, allow_pickle=True)
    return z["values"], [str(c) for c in z["columns"]], [str(d) for d in z["dtypes"]]


def compare_matrix(ref_path: Path, cand_path: Path) -> dict:
    if not ref_path.exists():
        return {"status": "SKIP", "note": f"reference missing: {ref_path.name}"}
    if not cand_path.exists():
        return {"status": "SKIP", "note": f"candidate missing: {cand_path.name}"}

    a, acols, adt = load_matrix(ref_path)
    b, bcols, bdt = load_matrix(cand_path)

    res: dict = {
        "ref_shape": list(a.shape),
        "cand_shape": list(b.shape),
        "n_cols_ref": len(acols),
        "n_cols_cand": len(bcols),
        "column_order_match": acols == bcols,
        "dtype_match": adt == bdt,
    }
    if a.shape != b.shape:
        res.update({"status": "FAIL", "note": "shape mismatch",
                    "max_abs_diff": None, "mean_abs_diff": None})
        return res
    if acols != bcols:
        only_ref = [c for c in acols if c not in bcols]
        only_cand = [c for c in bcols if c not in acols]
        res.update({"status": "FAIL", "note": "column mismatch",
                    "only_in_reference": only_ref[:20],
                    "only_in_candidate": only_cand[:20],
                    "max_abs_diff": None, "mean_abs_diff": None})
        return res

    na, nb = np.isnan(a), np.isnan(b)
    nan_match = bool(np.array_equal(na, nb))
    res["nan_mask_match"] = nan_match
    res["n_nan_ref"] = int(na.sum())
    res["n_nan_cand"] = int(nb.sum())

    both = ~(na | nb)
    if both.any():
        d = np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
        max_abs = float(d.max())
        mean_abs = float(d.mean())
    else:
        max_abs = 0.0
        mean_abs = 0.0
    res["max_abs_diff"] = max_abs
    res["mean_abs_diff"] = mean_abs

    if not nan_match:
        res["status"] = "FAIL"
        res["note"] = "NaN mask mismatch"
        return res

    ok = (max_abs < R2_MAX_ABS) and (mean_abs < R2_MEAN_ABS) and res["dtype_match"]
    res["status"] = "PASS" if ok else "FAIL"
    if not ok and max_abs >= R2_MAX_ABS:
        # 어느 열이 문제인지 지목한다
        colmax = np.zeros(a.shape[1])
        for j in range(a.shape[1]):
            m = both[:, j]
            if m.any():
                colmax[j] = np.abs(a[m, j].astype(np.float64) - b[m, j].astype(np.float64)).max()
        worst = np.argsort(-colmax)[:8]
        res["worst_columns"] = [{"col": acols[int(j)], "max_abs_diff": float(colmax[int(j)])}
                                for j in worst if colmax[int(j)] > 0]
    return res


def compare_r1(ref_dir: Path, cand_dir: Path, year: int) -> dict:
    rp = ref_dir / f"data_{year}.npz"
    cp = cand_dir / f"data_{year}.npz"
    if not rp.exists() or not cp.exists():
        return {"status": "SKIP", "note": "data npz missing"}
    a = np.load(rp, allow_pickle=True)
    b = np.load(cp, allow_pickle=True)
    checks = {}
    checks["n_rows"] = (int(len(a["row_id"])), int(len(b["row_id"])))
    checks["row_id_order"] = bool(np.array_equal(a["row_id"].astype(str), b["row_id"].astype(str)))
    checks["target"] = float(np.max(np.abs(a["target"] - b["target"])))
    checks["season"] = bool(np.array_equal(a["season"], b["season"]))
    checks["domain3"] = bool(np.array_equal(a["domain3"].astype(str), b["domain3"].astype(str)))
    ph = {}
    for k in a.files:
        if k.startswith("pitcher_half_"):
            ph[k] = bool(np.array_equal(a[k], b[k])) if k in b.files else False
    checks["pitcher_half"] = ph
    ok = (
        checks["n_rows"][0] == checks["n_rows"][1]
        and checks["row_id_order"]
        and checks["target"] == 0.0
        and checks["season"]
        and checks["domain3"]
        and all(ph.values())
    )
    checks["status"] = "PASS" if ok else "FAIL"
    return checks


def compare_r3_scalars(ref_manifest: dict, cand_summary: dict, year: int,
                       fallback_summary: dict | None = None) -> dict:
    ry = ref_manifest.get("years", {}).get(str(year), {})
    cy = cand_summary.get("years", {}).get(str(year), {})
    if not cy and fallback_summary:
        # Phase 2 단독 실행 시 trend scalar 는 phase2_summary 에 있다.
        cy = fallback_summary.get("years", {}).get(str(year), {})
    if not ry or not cy:
        return {"status": "SKIP", "note": "manifest/summary missing"}
    out = {}
    if "trend_slope" in ry and "trend_slope" in cy:
        out["trend_slope_diff"] = abs(float(ry["trend_slope"]) - float(cy["trend_slope"]))
    prior_diff = 0.0
    if "trend_prior" in ry and "trend_prior" in cy:
        for d in ry["trend_prior"]:
            prior_diff = max(prior_diff, abs(float(ry["trend_prior"][d]) - float(cy["trend_prior"][d])))
        out["trend_prior_max_diff"] = prior_diff
    vals = [v for k, v in out.items() if isinstance(v, float)]
    out["status"] = "PASS" if vals and max(vals) < R3_SCALAR_ABS else ("FAIL" if vals else "SKIP")
    return out


def compare_anchor(ref_dir: Path, cand_csv: Path, year: int) -> dict:
    rp = ref_dir / f"anchor_{year}.npz"
    if not rp.exists():
        return {k: {"status": "SKIP", "note": "reference anchor missing (stage=features)"}
                for k, _ in ANCHOR_KEYS}
    if not cand_csv.exists():
        return {k: {"status": "SKIP", "note": "candidate base_{year}.csv missing"}
                for k, _ in ANCHOR_KEYS}
    a = np.load(rp)
    df = pd.read_csv(cand_csv)
    out = {}
    for key, _label in ANCHOR_KEYS:
        if key not in a.files or key not in df.columns:
            out[key] = {"status": "SKIP", "note": "column missing"}
            continue
        d = np.abs(a[key].astype(np.float64) - df[key].to_numpy(np.float64))
        mx, mn = float(d.max()), float(d.mean())
        corr = float(np.corrcoef(a[key], df[key].to_numpy(np.float64))[0, 1])
        out[key] = {
            "max_abs_diff": mx,
            "mean_abs_diff": mn,
            "corr": corr,
            "status": "PASS" if mx < R3_PRED_MAX_ABS else "FAIL",
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True,
                    help="train.py 의 FEATURE_DUMP_DIR (기본 validation/<exp>/matrices)")
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--year", type=int, action="append", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_dir = Path(args.reference)
    cand_dir = Path(args.candidate)
    vdir = Path(args.validation_dir)
    out_path = Path(args.out) if args.out else vdir / "gate_r1_r2_r3.json"

    ref_manifest: dict = {"years": {}}
    for p in sorted(ref_dir.glob("manifest_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        ref_manifest.setdefault("meta", m)
        ref_manifest["years"].update(m.get("years", {}))

    cand_summary = {}
    p3 = vdir / "phase3_summary.json"
    if p3.exists():
        cand_summary = json.loads(p3.read_text(encoding="utf-8"))
    cand_summary2 = {}
    p2 = vdir / "phase2_summary.json"
    if p2.exists():
        cand_summary2 = json.loads(p2.read_text(encoding="utf-8"))

    years = args.year or sorted(int(y) for y in ref_manifest["years"])
    report: dict = {
        "reference_dir": str(ref_dir),
        "candidate_dir": str(cand_dir),
        "thresholds": {
            "r2_max_abs_diff": R2_MAX_ABS,
            "r2_mean_abs_diff": R2_MEAN_ABS,
            "r3_pred_max_abs_diff": R3_PRED_MAX_ABS,
            "r3_scalar_abs_diff": R3_SCALAR_ABS,
        },
        "years": {},
    }

    rows = []
    for year in years:
        y: dict = {}
        y["R1 Data"] = compare_r1(ref_dir, cand_dir, year)
        for name, label in BLOCKS + UNUSED_BLOCKS:
            y[label] = compare_matrix(ref_dir / f"{name}_{year}.npz",
                                      cand_dir / f"{name}_{year}.npz")
        y["R3 Trend Scalars"] = compare_r3_scalars(ref_manifest, cand_summary, year,
                                                   cand_summary2)
        anchor = compare_anchor(ref_dir, vdir / f"base_{year}.csv", year)
        for key, label in ANCHOR_KEYS:
            y[label] = anchor.get(key, {"status": "SKIP"})
        report["years"][str(year)] = y

        for label, res in y.items():
            rows.append(
                {
                    "year": year,
                    "gate": label,
                    "status": res.get("status", "SKIP"),
                    "max_abs_diff": res.get("max_abs_diff"),
                    "mean_abs_diff": res.get("mean_abs_diff"),
                    "note": res.get("note", ""),
                }
            )

    tab = pd.DataFrame(rows)
    statuses = set(tab.status)
    report["overall"] = (
        "FAIL" if "FAIL" in statuses else ("PASS" if "PASS" in statuses else "SKIP")
    )
    report["counts"] = tab.status.value_counts().to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    tab.to_csv(vdir / "gate_r1_r2_r3.csv", index=False)

    # ---- feature_parity/<block>.csv ----
    fp = vdir / "feature_parity"
    fp.mkdir(parents=True, exist_ok=True)
    for name, label in BLOCKS + UNUSED_BLOCKS:
        recs = []
        for year in years:
            res = report["years"][str(year)].get(label, {})
            recs.append(
                {
                    "year": year,
                    "block": name,
                    "ref_shape": res.get("ref_shape"),
                    "cand_shape": res.get("cand_shape"),
                    "n_cols": res.get("n_cols_cand"),
                    "column_order_match": res.get("column_order_match"),
                    "dtype_match": res.get("dtype_match"),
                    "nan_mask_match": res.get("nan_mask_match"),
                    "n_nan_ref": res.get("n_nan_ref"),
                    "n_nan_cand": res.get("n_nan_cand"),
                    "max_abs_diff": res.get("max_abs_diff"),
                    "mean_abs_diff": res.get("mean_abs_diff"),
                    "status": res.get("status"),
                }
            )
        pd.DataFrame(recs).to_csv(fp / f"{name}.csv", index=False)

    # ---- 콘솔 표 ----
    def fmt(v):
        if v is None:
            return "-"
        if v == 0:
            return "0"
        return f"{v:.3e}"

    print("=" * 96)
    print(f"{'Year':<6}{'Gate':<34}{'Result':<26}{'Max Diff':<13}{'Status'}")
    print("=" * 96)
    for _, r in tab.iterrows():
        res = report["years"][str(r["year"])][r["gate"]]
        if r["gate"] == "R1 Data":
            detail = f"rows={res.get('n_rows')}" if res.get("status") != "SKIP" else res.get("note", "")
        elif r["gate"].startswith("R2"):
            detail = (
                f"cols={res.get('n_cols_cand')} order={'Y' if res.get('column_order_match') else 'N'}"
                f" nan={'Y' if res.get('nan_mask_match') else 'N'}"
                if res.get("status") != "SKIP" else res.get("note", "")
            )
        elif r["gate"] == "R3 Trend Scalars":
            detail = (f"slope={fmt(res.get('trend_slope_diff'))} prior={fmt(res.get('trend_prior_max_diff'))}"
                      if res.get("status") != "SKIP" else res.get("note", ""))
        else:
            detail = (f"corr={res.get('corr'):.8f}" if res.get("corr") is not None
                      else res.get("note", ""))
        md = fmt(r["max_abs_diff"]) if r["gate"] != "R3 Trend Scalars" else fmt(
            max([v for v in [res.get("trend_slope_diff"), res.get("trend_prior_max_diff")]
                 if v is not None], default=None)
        )
        print(f"{r['year']:<6}{r['gate']:<34}{detail[:25]:<26}{md:<13}{r['status']}")
    print("=" * 96)
    print("OVERALL:", report["overall"], report["counts"])
    print(f"  -> {out_path}")
    print(f"  -> {vdir / 'gate_r1_r2_r3.csv'}")


if __name__ == "__main__":
    main()

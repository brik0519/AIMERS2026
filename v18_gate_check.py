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
# Phase 4 — Layer 1 V15 residual adapter
V15_KEYS = [
    ("p_seed42", "R4 Seed 42"),
    ("p_seed2026", "R4 Seed 2026"),
    ("p_seed314", "R4 Seed 314"),
    ("v15_oof", "R4 V15 OOF"),
]
R4_BSS_ABS = 0.5
# Phase 5 — Layer 2 V18 temporal-stable Ridge
P18_KEYS = [
    ("corr", "R5 L2 Correction"),
    ("p18", "R5 P18 Final"),
]
P18_TRANSITIONS = (2021, 2022, 2023, 2024)


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


def compare_anchor(ref_dir: Path, cand_csv: Path, year: int,
                   cand_npz: Path | None = None) -> dict:
    rp = ref_dir / f"anchor_{year}.npz"
    if not rp.exists():
        return {k: {"status": "SKIP", "note": "reference anchor missing (stage=features)"}
                for k, _ in ANCHOR_KEYS}

    # 무손실 npz 를 우선한다. pandas to_csv 는 float64 round-trip 을 보장하지 않아
    # CSV 경로에서는 최대 1 ulp(약 5.55e-17) 의 직렬화 오차가 섞인다.
    source = None
    if cand_npz is not None and cand_npz.exists():
        z = np.load(cand_npz)
        cand = {k: z[k].astype(np.float64) for k in z.files}
        source = "npz"
    elif cand_csv.exists():
        df = pd.read_csv(cand_csv)
        cand = {c: df[c].to_numpy(np.float64) for c in df.columns if c in
                {k for k, _ in ANCHOR_KEYS}}
        source = "csv(lossy)"
    else:
        return {k: {"status": "SKIP", "note": "candidate anchor npz / base csv 둘 다 없음"}
                for k, _ in ANCHOR_KEYS}

    a = np.load(rp)
    out = {}
    for key, _label in ANCHOR_KEYS:
        if key not in a.files or key not in cand:
            out[key] = {"status": "SKIP", "note": "column missing"}
            continue
        d = np.abs(a[key].astype(np.float64) - cand[key])
        mx, mn = float(d.max()), float(d.mean())
        corr = float(np.corrcoef(a[key], cand[key])[0, 1])
        out[key] = {
            "max_abs_diff": mx,
            "mean_abs_diff": mn,
            "corr": corr,
            "candidate_source": source,
            "bit_identical": bool(mx == 0.0),
            "status": "PASS" if mx < R3_PRED_MAX_ABS else "FAIL",
        }
    return out


def raw_bss(y, p):
    y = np.asarray(y, np.float64)
    p = np.clip(np.asarray(p, np.float64), 0, 1)
    ref = max(float(y.mean() * (1 - y.mean())), 1e-12)
    return float(100000.0 * (1.0 - np.mean((p - y) ** 2) / ref))


def compare_v15_oof(ref_dir: Path, cand_dir: Path, year: int) -> dict:
    """R4 — Layer 1 V15 OOF parity. 양쪽 모두 무손실 npz(float64) 를 쓴다."""
    rp = ref_dir / f"v15_oof_{year}.npz"
    cp = cand_dir / f"v15_oof_{year}.npz"
    if not rp.exists():
        return {k: {"status": "SKIP", "note": "reference v15_oof 없음 (stage != v15_oof)"}
                for k, _ in V15_KEYS}
    if not cp.exists():
        return {k: {"status": "SKIP", "note": "candidate v15_oof npz 없음 (PHASE<4)"}
                for k, _ in V15_KEYS}
    a = np.load(rp)
    b = np.load(cp)
    out = {}

    # base / target 이 같은 것이 전제다. 어긋나면 R4 는 의미가 없다.
    guard = {}
    for k in ("p_base", "target"):
        if k in a.files and k in b.files:
            guard[k] = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())

    for key, _label in V15_KEYS:
        if key not in a.files or key not in b.files:
            out[key] = {"status": "SKIP", "note": f"{key} 없음"}
            continue
        ra = a[key].astype(np.float64)
        cb_ = b[key].astype(np.float64)
        if ra.shape != cb_.shape:
            out[key] = {"status": "FAIL", "note": "shape mismatch",
                        "ref_shape": list(ra.shape), "cand_shape": list(cb_.shape)}
            continue
        d = np.abs(ra - cb_)
        mx, mn = float(d.max()), float(d.mean())
        corr = float(np.corrcoef(ra, cb_)[0, 1])
        tgt = a["target"].astype(np.float64) if "target" in a.files else None
        bss_diff = (abs(raw_bss(tgt, ra) - raw_bss(tgt, cb_))
                    if tgt is not None else None)
        ok = mx < R3_PRED_MAX_ABS and corr > 0.99999
        if bss_diff is not None:
            ok = ok and bss_diff < R4_BSS_ABS
        out[key] = {
            "max_abs_diff": mx,
            "mean_abs_diff": mn,
            "corr": corr,
            "bss_ref": raw_bss(tgt, ra) if tgt is not None else None,
            "bss_cand": raw_bss(tgt, cb_) if tgt is not None else None,
            "bss_abs_diff": bss_diff,
            "bit_identical": bool(mx == 0.0),
            "base_target_guard": guard,
            "status": "PASS" if ok else "FAIL",
        }
    return out


def compare_p18(ref_dir: Path, cand_dir: Path, year: int) -> dict:
    """R5 — Layer 2 parity.

    잔여 오차는 0 이 아닐 것으로 예상한다. Layer 2 Ridge 의 학습 타깃이
    `Y[T-1] - V15_OOF[T-1]` 이라 V15_OOF 의 2^-52 실행 잡음이 계수로 전파되고,
    solver='lsqr' 는 tol=1e-5 의 반복 해법이다. ~1e-14 수준은 정상이다.
    """
    if year not in P18_TRANSITIONS:
        return {k: {"status": "SKIP", "note": f"{year} 는 Layer 2 전이 대상이 아님"}
                for k, _ in P18_KEYS}
    rp = ref_dir / f"p18_{year}.npz"
    cp = cand_dir / f"p18_{year}.npz"
    if not rp.exists():
        return {k: {"status": "SKIP", "note": "reference p18 없음 (stage != p18)"}
                for k, _ in P18_KEYS}
    if not cp.exists():
        return {k: {"status": "SKIP", "note": "candidate p18 npz 없음 (PHASE<5)"}
                for k, _ in P18_KEYS}
    a, b = np.load(rp), np.load(cp)
    guard = {}
    for k in ("v15_oof", "target", "core"):
        if k in a.files and k in b.files:
            if k == "core":
                guard[k] = bool(np.array_equal(a[k], b[k]))
            else:
                guard[k] = float(np.abs(a[k].astype(np.float64)
                                        - b[k].astype(np.float64)).max())
    out = {}
    for key, _label in P18_KEYS:
        if key not in a.files or key not in b.files:
            out[key] = {"status": "SKIP", "note": f"{key} 없음"}
            continue
        ra = a[key].astype(np.float64)
        cb_ = b[key].astype(np.float64)
        if ra.shape != cb_.shape:
            out[key] = {"status": "FAIL", "note": "shape mismatch"}
            continue
        d = np.abs(ra - cb_)
        mx, mn = float(d.max()), float(d.mean())
        corr = float(np.corrcoef(ra, cb_)[0, 1])
        tgt = a["target"].astype(np.float64) if "target" in a.files else None
        bss_diff = (abs(raw_bss(tgt, ra) - raw_bss(tgt, cb_))
                    if tgt is not None and key == "p18" else None)
        ok = mx < R3_PRED_MAX_ABS and corr > 0.99999
        if bss_diff is not None:
            ok = ok and bss_diff < R4_BSS_ABS
        out[key] = {
            "max_abs_diff": mx, "mean_abs_diff": mn, "corr": corr,
            "bss_ref": raw_bss(tgt, ra) if (tgt is not None and key == "p18") else None,
            "bss_cand": raw_bss(tgt, cb_) if (tgt is not None and key == "p18") else None,
            "bss_abs_diff": bss_diff,
            "bit_identical": bool(mx == 0.0),
            "v15_oof_target_guard": guard,
            "status": "PASS" if ok else "FAIL",
        }
    return out


def compare_promotion_gate(ref_dir: Path, vdir: Path) -> dict:
    """R5 — 승격 gate 일치. V18_SUBMIT_RECOMMENDED 불리언과 전이 gain 을 비교한다."""
    rm = ref_dir / "manifest_p18.json"
    cg = vdir / "gate.json"
    if not rm.exists():
        return {"status": "SKIP", "note": "reference manifest_p18.json 없음"}
    if not cg.exists():
        return {"status": "SKIP", "note": "candidate gate.json 없음 (PHASE<5)"}
    r = json.loads(rm.read_text(encoding="utf-8")).get("p18", {})
    c = json.loads(cg.read_text(encoding="utf-8"))
    rs = r.get("summary", {})
    out = {
        "ref_submit_recommended": bool(r.get("v18_submit_recommended")),
        "cand_submit_recommended": bool(c.get("submit_recommended")),
    }
    out["submit_recommended_match"] = (
        out["ref_submit_recommended"] == out["cand_submit_recommended"]
    )
    diffs = {}
    for k in ("temporal_2022_to_2023_gain", "temporal_2023_to_2024_gain",
              "core_2024_gain", "early_gain", "late_gain"):
        if k in rs and k in c:
            diffs[k] = abs(float(rs[k]) - float(c[k]))
    for k in ("p05", "prob_positive"):
        if "bootstrap" in rs and "bootstrap" in c:
            diffs["boot_" + k] = abs(float(rs["bootstrap"][k]) - float(c["bootstrap"][k]))
    out["metric_abs_diffs"] = diffs
    out["max_metric_abs_diff"] = max(diffs.values()) if diffs else None
    ok = out["submit_recommended_match"] and (
        out["max_metric_abs_diff"] is None or out["max_metric_abs_diff"] < R4_BSS_ABS
    )
    out["status"] = "PASS" if ok else "FAIL"
    return out


def run_selfcheck(dir_a: Path, dir_b: Path, years: list[int]) -> dict:
    """R3-0 — 자기 결정성.

    같은 코드/같은 환경에서 reference 를 두 번 돌린 산출물을 비교한다.
    여기서 나온 차이는 **재현 오차가 아니라 환경 잡음의 하한**이다.
    R3 / R4 의 잔여 차이를 해석하는 기준선이 된다.
    """
    report = {"dir_a": str(dir_a), "dir_b": str(dir_b), "years": {}}
    print("=" * 84)
    print(f"{'Year':<6}{'Artifact':<18}{'Key':<16}{'max_abs_diff':>16}{'bit_identical':>16}")
    print("=" * 84)
    for year in years:
        y = {}
        for art in ("anchor", "v15_oof"):
            pa = dir_a / f"{art}_{year}.npz"
            pb = dir_b / f"{art}_{year}.npz"
            if not (pa.exists() and pb.exists()):
                y[art] = {"status": "SKIP", "note": f"{art}_{year}.npz 없음"}
                print(f"{year:<6}{art:<18}{'-':<16}{'SKIP':>16}{'':>16}")
                continue
            a, b = np.load(pa), np.load(pb)
            keys = [k for k in a.files if k in b.files]
            per = {}
            for k in keys:
                d = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())
                per[k] = {"max_abs_diff": d, "bit_identical": bool(d == 0.0)}
                print(f"{year:<6}{art:<18}{k:<16}{d:>16.3e}{str(d == 0.0):>16}")
            y[art] = per
        report["years"][str(year)] = y
    print("=" * 84)
    allv = [v["max_abs_diff"]
            for yv in report["years"].values()
            for art in yv.values() if isinstance(art, dict)
            for v in art.values() if isinstance(v, dict) and "max_abs_diff" in v]
    report["environment_noise_floor"] = max(allv) if allv else None
    print(f"환경 잡음 하한 (같은 코드 2회 실행 최대 차이): "
          f"{report['environment_noise_floor']}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=False, default=None)
    ap.add_argument("--candidate", required=False, default=None,
                    help="train.py 의 FEATURE_DUMP_DIR (기본 validation/<exp>/matrices)")
    ap.add_argument("--validation-dir", required=True)  # selfcheck 에서도 출력 위치로 쓴다
    ap.add_argument("--year", type=int, action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", nargs=2, metavar=("REF_A", "REF_B"), default=None,
                    help="R3-0 자기 결정성: reference 2회 실행분을 서로 비교한다")
    args = ap.parse_args()

    if args.selfcheck:
        a, b = Path(args.selfcheck[0]), Path(args.selfcheck[1])
        years = args.year or sorted(
            {int(p.stem.rsplit("_", 1)[1]) for p in a.glob("v15_oof_*.npz")}
            | {int(p.stem.rsplit("_", 1)[1]) for p in a.glob("anchor_*.npz")}
        )
        rep = run_selfcheck(a, b, years)
        out = Path(args.out) if args.out else Path(args.validation_dir) / "gate_r3_0_selfcheck.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        print("  ->", out)
        return

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
        anchor = compare_anchor(ref_dir, vdir / f"base_{year}.csv", year,
                                cand_dir / f"anchor_{year}.npz")
        for key, label in ANCHOR_KEYS:
            y[label] = anchor.get(key, {"status": "SKIP"})
        v15 = compare_v15_oof(ref_dir, cand_dir, year)
        for key, label in V15_KEYS:
            y[label] = v15.get(key, {"status": "SKIP"})
        p18 = compare_p18(ref_dir, cand_dir, year)
        for key, label in P18_KEYS:
            y[label] = p18.get(key, {"status": "SKIP"})
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

    gate5 = compare_promotion_gate(ref_dir, vdir)
    report["promotion_gate"] = gate5
    rows.append({"year": "-", "gate": "R5 Promotion Gate", "status": gate5.get("status"),
                 "max_abs_diff": gate5.get("max_metric_abs_diff"),
                 "mean_abs_diff": None, "note": gate5.get("note", "")})

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
        if r["gate"] == "R5 Promotion Gate":
            g5 = report["promotion_gate"]
            det = (f"submit={g5.get('cand_submit_recommended')} "
                   f"match={g5.get('submit_recommended_match')}"
                   if g5.get("status") != "SKIP" else g5.get("note", ""))
            print(f"{'-':<6}{r['gate']:<34}{det[:25]:<26}"
                  f"{fmt(r['max_abs_diff']):<13}{r['status']}")
            continue
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
        elif r["gate"].startswith(("R4", "R5")):
            detail = (
                f"corr={res.get('corr'):.8f} dBSS={res.get('bss_abs_diff'):.3f}"
                if res.get("status") != "SKIP" and res.get("bss_abs_diff") is not None
                else res.get("note", "")
            )
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
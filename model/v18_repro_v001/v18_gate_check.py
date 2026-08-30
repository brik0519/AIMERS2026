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
# Phase 7 — script.py 산출 parity (notebook 자체 smoke 와 동일 임계)
R7_SCRIPT_MAX_ABS = 1e-10
# Phase 5 — Layer 2 V18 temporal-stable Ridge
P18_KEYS = [
    ("corr", "R5 L2 Correction"),
    ("p18", "R5 P18 Final"),
]
P18_TRANSITIONS = (2021, 2022, 2023, 2024)
# Phase 6 — deploy artifact (2024 -> 2025)
DEPLOY_KEYS = [
    ("p1", "R6 Deploy exact_lgb"),
    ("p2", "R6 Deploy ridge"),
    ("p3", "R6 Deploy trend_lgb"),
    ("p_base25", "R6 Deploy p_base25"),
    ("correction", "R6 Deploy L1 corr"),
    ("final_v15_direct", "R6 Deploy V15 direct"),
    ("stable_corr25", "R6 Deploy L2 corr"),
    ("final_direct", "R6 Deploy FINAL"),
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


def compare_deploy(ref_dir: Path, cand_dir: Path, vdir: Path) -> dict:
    """R6 — Phase 6 배포 경로 parity (2024 -> 2025).

    validation 이 아니라 deploy regime 이다. Layer 1 이 crossfit 없이 2024 전체로
    학습되므로 ExtraTrees 병렬 누적 잡음(2^-53 근처)이 그대로 실린다.
    """
    rp = ref_dir / "deploy_2025.npz"
    cp = cand_dir / "deploy_2025.npz"
    if not rp.exists():
        return {k: {"status": "SKIP", "note": "reference deploy_2025.npz 없음 (stage != deploy)"}
                for k, _ in DEPLOY_KEYS}
    if not cp.exists():
        return {k: {"status": "SKIP", "note": "candidate deploy_2025.npz 없음 (PHASE<6)"}
                for k, _ in DEPLOY_KEYS}
    a, b = np.load(rp, allow_pickle=True), np.load(cp, allow_pickle=True)
    guard = {}
    for k in ("row_id", "domain3"):
        if k in a.files and k in b.files:
            guard[k] = bool(np.array_equal(a[k].astype(str), b[k].astype(str)))
    out = {}
    for key, _label in DEPLOY_KEYS:
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
        mx = float(d.max())
        corr = (float(np.corrcoef(ra, cb_)[0, 1]) if ra.size > 1 and ra.std() > 0 else 1.0)
        ok = mx < R3_PRED_MAX_ABS and all(guard.values())
        out[key] = {
            "max_abs_diff": mx, "mean_abs_diff": float(d.mean()), "corr": corr,
            "n": int(ra.size), "bit_identical": bool(mx == 0.0),
            "row_domain_guard": guard,
            "status": "PASS" if ok else "FAIL",
        }
    return out


def check_deploy_artifact(model_path: Path, ref_manifest: dict) -> dict:
    """R6b — 단일 pkl artifact 가 배포에 필요한 키를 전부 담고 있는가 + 차원 일치."""
    if not model_path.exists():
        return {"status": "SKIP", "note": f"{model_path} 없음 (PHASE<6)"}
    import pickle

    class _TolerantUnpickler(pickle.Unpickler):
        """lightgbm / sklearn 이 없는 환경에서도 **구조**만 검사할 수 있게 한다.

        artifact 의 키·차원·설정값 확인에는 모델 객체 실체가 필요 없다.
        복원할 수 없는 클래스는 빈 더미로 대체한다 (예측에는 절대 쓰지 않는다).
        """

        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                # __reduce__ 가 인자를 넘기거나 __setstate__ 를 부르는 경우까지 견딘다
                return type(
                    f"_Missing_{module}_{name}", (),
                    {"__init__": lambda self, *a, **k: None,
                     "__setstate__": lambda self, state: None,
                     "__reduce__": lambda self: (dict, ())},
                )

    loaded_with = "pickle"
    try:
        with open(model_path, "rb") as f:
            art = pickle.load(f)
    except (ImportError, ModuleNotFoundError, AttributeError):
        try:
            with open(model_path, "rb") as f:
                art = _TolerantUnpickler(f).load()
            loaded_with = "tolerant (모델 모듈 부재 — 구조만 검사)"
        except Exception as e:  # noqa: BLE001
            return {"status": "SKIP",
                    "note": f"artifact 를 열 수 없다: {type(e).__name__}: {e}"}
    required = [
        "expected_year", "anchor_team_id", "selected_blocks",
        "base_exact_lgb", "exact_feature_cols", "exact_cat_maps",
        "base_ridge", "ridge_feature_cols", "ridge_med", "ridge_mu", "ridge_sd",
        "base_trend_lgb", "trend_num_cols", "trend_cat_maps", "trend_feature_cols",
        "trend_prior", "target_global_bank",
        "adapter_prep", "adapter_base_feature_cols", "adapter_feature_cols",
        "pitcher_conditional_specs", "pitcher_conditional_cols", "exact_adapter_cols",
        "adapter_seeds", "adapter_shrink", "adapter_models",
        "v18_stable_enabled", "v18_stable_specs", "v18_stable_cols",
        "v18_stable_mu", "v18_stable_sd", "v18_stable_model",
        "v18_stable_gamma", "v18_route_domain",
    ]
    missing = [k for k in required if k not in art]
    dims = {
        "exact_lgb": len(art.get("exact_feature_cols", [])),
        "ridge": len(art.get("ridge_feature_cols", [])),
        "trend_lgb": len(art.get("trend_feature_cols", [])),
        "adapter": len(art.get("adapter_feature_cols", [])),
        "adapter_base": len(art.get("adapter_base_feature_cols", [])),
        "pitcher_conditional": len(art.get("pitcher_conditional_cols", [])),
        "exact_adapter": len(art.get("exact_adapter_cols", [])),
        "v18_stable": len(art.get("v18_stable_cols", [])),
    }
    expect = {"exact_lgb": 98, "ridge": 89, "trend_lgb": 42, "adapter": 128,
              "adapter_base": 45, "pitcher_conditional": 21, "exact_adapter": 62,
              "v18_stable": 25}
    dim_bad = {k: (dims[k], expect[k]) for k in expect if dims[k] != expect[k]}
    n_models = len(art.get("adapter_models", {}))
    ref = ref_manifest.get("deploy", {})
    checks = {
        "loaded_with": loaded_with,
        "reference_available": bool(ref),   # False 면 아래 대조가 vacuous 하다
        "missing_keys": missing,
        "dims": dims, "dim_mismatch": dim_bad,
        "n_adapter_models": n_models,
        "artifact_mb": round(model_path.stat().st_size / 1e6, 2),
        "v18_stable_enabled": bool(art.get("v18_stable_enabled")),
        "ref_v18_stable_enabled": ref.get("v18_stable_enabled"),
    }
    if ref:
        checks["stable_enabled_match"] = (
            bool(art.get("v18_stable_enabled")) == bool(ref.get("v18_stable_enabled"))
        )
        checks["expected_year_match"] = (
            int(art.get("expected_year", -1)) == int(ref.get("expected_year", -2))
        )
        checks["route_domain_match"] = (
            art.get("v18_route_domain") == ref.get("v18_route_domain")
        )
        checks["gamma_match"] = (
            abs(float(art.get("v18_stable_gamma", -1))
                - float(ref.get("v18_stable_gamma", -2))) < 1e-12
        )
        checks["shrink_match"] = (
            abs(float(art.get("adapter_shrink", -1))
                - float(ref.get("adapter_shrink", -2))) < 1e-12
        )
        checks["seeds_match"] = (
            list(art.get("adapter_seeds", [])) == list(ref.get("adapter_seeds", []))
        )
        checks["selected_blocks_match"] = (
            list(art.get("selected_blocks", [])) == list(ref.get("selected_blocks", []))
        )
        ref_dims = {k.replace("_cols_n", "").replace("_feature", ""): v
                    for k, v in ref.items() if k.endswith("_cols_n")}
        checks["ref_dims"] = ref_dims
        checks["ref_dim_mismatch"] = {
            k: (dims[k], ref_dims[k]) for k in dims if k in ref_dims and dims[k] != ref_dims[k]
        }
    else:
        checks["note"] = ("reference manifest_deploy.json 의 deploy 섹션이 없어 "
                          "대조가 vacuous 하다 — --stage deploy 를 먼저 실행할 것")
    ok = (not missing and not dim_bad and n_models == 9
          and checks.get("stable_enabled_match", True)
          and checks.get("expected_year_match", True)
          and checks.get("route_domain_match", True)
          and checks.get("gamma_match", True)
          and checks.get("shrink_match", True)
          and checks.get("seeds_match", True)
          and checks.get("selected_blocks_match", True)
          and not checks.get("ref_dim_mismatch", {}))
    checks["status"] = "PASS" if ok else "FAIL"
    return checks


def compare_script_output(cand_dir: Path, submission: Path) -> dict:
    """R7 — script.py 산출 submission.csv 가 train.py 내부 예측과 같은가.

    notebook cell 16 의 자기 smoke(`WORKING_TREE_SMOKE_OK max_abs_diff < 1e-10`) 와 같은 검사다.
    잔여 오차 요인 2가지가 원본 설계에 이미 있다 — 둘 다 임계보다 6자릿수 작다.
      · script.py 는 `corr += c/len(seeds)` (누적 평균),
        train.py Phase 6 은 `np.mean(np.column_stack(...))` (일괄 평균)
      · submission.csv 는 pandas 기본 float 포맷 → CSV 왕복 오차 (§G-6)
    """
    # hardened 산출물이 있으면 그쪽이 artifact / submission 과 일치하는 값이다.
    # (규칙 준수 조치로 Ridge 계수를 float64 로 올린 경우 verbatim 과 ~4e-9 다르다)
    dep = cand_dir / "deploy_2025_hardened.npz"
    which = "hardened"
    if not dep.exists():
        dep = cand_dir / "deploy_2025.npz"
        which = "verbatim"
    if not dep.exists():
        return {"status": "SKIP", "note": "deploy_2025[_hardened].npz 없음 (PHASE<6)"}
    if not submission.exists():
        return {"status": "SKIP", "note": f"{submission} 없음 (script.py 미실행)"}
    z = np.load(dep, allow_pickle=True)
    sub = pd.read_csv(submission, encoding="utf-8-sig")
    cols = list(sub.columns)
    if cols != ["row_id", "control_success"]:
        return {"status": "FAIL", "note": f"submission 컬럼 불일치: {cols}"}
    ref = z["final_direct"].astype(np.float64)
    cand = sub["control_success"].to_numpy(np.float64)
    order_ok = bool(np.array_equal(z["row_id"].astype(str),
                                   sub["row_id"].astype(str).to_numpy()))
    if ref.shape != cand.shape:
        return {"status": "FAIL", "note": "행 수 불일치",
                "n_ref": int(ref.size), "n_cand": int(cand.size)}
    d = np.abs(ref - cand)
    mx = float(d.max())
    ok = order_ok and mx < R7_SCRIPT_MAX_ABS and bool(
        np.isfinite(cand).all() and (cand >= 0).all() and (cand <= 1).all())
    return {
        "n": int(ref.size), "row_id_order_match": order_ok,
        "compared_against": which,
        "max_abs_diff": mx, "mean_abs_diff": float(d.mean()),
        "bit_identical": bool(mx == 0.0),
        "threshold": R7_SCRIPT_MAX_ABS,
        "range_ok": bool(np.isfinite(cand).all() and (cand >= 0).all() and (cand <= 1).all()),
        "status": "PASS" if ok else "FAIL",
    }


def check_artifact_chain(cand_dir: Path, year: int) -> dict:
    """R0 — candidate 자체 산출물 체인 정합성 (reference 불필요).

    각 Phase 를 따로 재실행하면 앞 단계만 갱신되고 뒤 단계는 이전 산출물이 남는다.
    그 상태로도 reference 대비 gate 는 통과하므로 **stale artifact 가 조용히 지나간다.**
    Layer 0 이 bit-결정적인 동안에는 무해하지만, Phase 6/7 처럼 산출물이 늘어나면
    실질 위험이 된다. 다음 연결이 실제로 같은 값인지 확인한다.

        anchor_{y}.p_base   ==  v15_oof_{y}.p_base     (Layer 1 이 그 anchor 를 썼는가)
        v15_oof_{y}.v15_oof ==  p18_{y}.v15_oof        (Layer 2 가 그 OOF 를 썼는가)
        target 은 세 산출물 전부 동일
    """
    links = []
    a = cand_dir / f"anchor_{year}.npz"
    v = cand_dir / f"v15_oof_{year}.npz"
    p = cand_dir / f"p18_{year}.npz"
    if not (a.exists() and v.exists()):
        return {"status": "SKIP", "note": "anchor / v15_oof npz 부족"}
    za, zv = np.load(a), np.load(v)
    links.append(("anchor.p_base -> v15_oof.p_base",
                  float(np.abs(za["p_base"].astype(np.float64)
                               - zv["p_base"].astype(np.float64)).max())))
    if "residual_l1" in za.files and "target" in zv.files:
        # anchor 는 target 을 직접 저장하지 않으므로 residual 로 역산해 대조한다
        tgt_from_anchor = za["residual_l1"].astype(np.float64) + za["p_base"].astype(np.float64)
        links.append(("anchor.target(역산) -> v15_oof.target",
                      float(np.abs(tgt_from_anchor - zv["target"].astype(np.float64)).max())))
    if p.exists():
        zp = np.load(p)
        links.append(("v15_oof.v15_oof -> p18.v15_oof",
                      float(np.abs(zv["v15_oof"].astype(np.float64)
                                   - zp["v15_oof"].astype(np.float64)).max())))
        links.append(("v15_oof.target -> p18.target",
                      float(np.abs(zv["target"].astype(np.float64)
                                   - zp["target"].astype(np.float64)).max())))
    worst = max(d for _, d in links)
    return {
        "links": {k: d for k, d in links},
        "max_abs_diff": worst,
        # 같은 실행이라면 저장된 배열이 비트 단위로 같아야 한다.
        "status": "PASS" if worst == 0.0 else "FAIL",
        "note": "" if worst == 0.0 else "stale artifact 의심 — 해당 Phase 를 다시 실행할 것",
    }


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
    ap.add_argument("--submission", default=None,
                    help="R7 검사용 submission.csv (기본 ./output/submission.csv)")
    ap.add_argument("--model-path", default=None,
                    help="R6b 검사용 단일 pkl 경로 (기본 ./model/<exp>/<exp>.pkl)")
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
        # manifest_p18.json / manifest_deploy.json 의 stage 섹션도 그대로 실어야
        # R5 promotion gate / R6b artifact 검사가 reference 와 대조할 수 있다.
        # (없으면 비교가 조용히 vacuous 하게 통과한다)
        for stage_key in ("p18", "deploy"):
            if stage_key in m:
                ref_manifest[stage_key] = m[stage_key]

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
        y["R0 Artifact Chain"] = check_artifact_chain(cand_dir, year)
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

    # ---- R6 : Phase 6 deploy (2024 -> 2025) ----
    dep = compare_deploy(ref_dir, cand_dir, vdir)
    report["deploy"] = dep
    for key, label in DEPLOY_KEYS:
        r_ = dep.get(key, {"status": "SKIP"})
        report.setdefault("deploy_by_gate", {})[label] = r_
        rows.append({"year": "-", "gate": label, "status": r_.get("status"),
                     "max_abs_diff": r_.get("max_abs_diff"),
                     "mean_abs_diff": r_.get("mean_abs_diff"), "note": r_.get("note", "")})
    exp = Path(str(vdir)).name
    model_path = (Path(args.model_path) if args.model_path
                  else Path("./model") / exp / f"{exp}.pkl")
    art = check_deploy_artifact(model_path, ref_manifest)
    art["model_path"] = str(model_path)
    report["deploy_artifact"] = art
    rows.append({"year": "-", "gate": "R6b Deploy Artifact", "status": art.get("status"),
                 "max_abs_diff": None, "mean_abs_diff": None, "note": art.get("note", "")})

    # ---- R7 : script.py 산출 parity ----
    sub_path = Path(args.submission) if args.submission else Path("./output/submission.csv")
    r7 = compare_script_output(cand_dir, sub_path)
    r7["submission"] = str(sub_path)
    report["script_parity"] = r7
    rows.append({"year": "-", "gate": "R7 script.py Parity", "status": r7.get("status"),
                 "max_abs_diff": r7.get("max_abs_diff"),
                 "mean_abs_diff": r7.get("mean_abs_diff"), "note": r7.get("note", "")})

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
        if r["gate"].startswith("R6 Deploy"):
            rr = report["deploy_by_gate"][r["gate"]]
            det = (f"n={rr.get('n')} bit={rr.get('bit_identical')}"
                   if rr.get("status") != "SKIP" else rr.get("note", ""))
            print(f"{'-':<6}{r['gate']:<34}{det[:25]:<26}"
                  f"{fmt(r['max_abs_diff']):<13}{r['status']}")
            continue
        if r["gate"] == "R7 script.py Parity":
            rr = report["script_parity"]
            det = (f"n={rr.get('n')} order={'Y' if rr.get('row_id_order_match') else 'N'}"
                   if rr.get("status") != "SKIP" else rr.get("note", ""))
            print(f"{'-':<6}{r['gate']:<34}{det[:25]:<26}"
                  f"{fmt(r['max_abs_diff']):<13}{r['status']}")
            continue
        if r["gate"] == "R6b Deploy Artifact":
            aa = report["deploy_artifact"]
            det = (f"{aa.get('artifact_mb')}MB models={aa.get('n_adapter_models')}"
                   if aa.get("status") != "SKIP" else aa.get("note", ""))
            print(f"{'-':<6}{r['gate']:<34}{det[:25]:<26}{'-':<13}{r['status']}")
            continue
        if r["gate"] == "R5 Promotion Gate":
            g5 = report["promotion_gate"]
            det = (f"submit={g5.get('cand_submit_recommended')} "
                   f"match={g5.get('submit_recommended_match')}"
                   if g5.get("status") != "SKIP" else g5.get("note", ""))
            print(f"{'-':<6}{r['gate']:<34}{det[:25]:<26}"
                  f"{fmt(r['max_abs_diff']):<13}{r['status']}")
            continue
        res = report["years"][str(r["year"])][r["gate"]]
        if r["gate"] == "R0 Artifact Chain":
            detail = (f"links={len(res.get('links', {}))}"
                      if res.get("status") != "SKIP" else res.get("note", ""))
        elif r["gate"] == "R1 Data":
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
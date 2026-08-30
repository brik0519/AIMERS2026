"""V18 Reproduction — R3 Anchor **Input** parity gate.

Layer 0 의 세 모델(LGBMClassifier / Ridge / LGBMRegressor)은 입력이 같으면
결정적이다. 따라서 lightgbm / scikit-learn 을 설치할 수 없는 환경에서도
`fit_recent_base` 가 모델에 넘기는 **입력 행렬 자체**를 비교하면
R3 의 대부분을 실측할 수 있다.

비교 대상 (연도별, source = year-1 / target = year):
    exact_lgb   A(98) / B(98)      RAWNUM + self_state + gseason + code_cat
    ridge       A(89) / B(89)      median fill -> z-score -> clip(+-8) 까지 적용한 최종 배열
    trend_lgb   A(42) / B(42)      simple_pair 결과
    trend       base(source 행별 DOMAIN_MEANS) / bt(target 행별 prior)

메모리를 아끼기 위해 reference 를 먼저 요약(columns / 열별 checksum / NaN 수 /
결정적 subsample)으로 축약한 뒤 namespace 를 해제하고, 그다음 train.py 측을 계산한다.

사용법:
    python3 v18_anchor_input_gate.py --notebook <nb> --data-dir ./data --year 2024 \
        --validation-dir validation/v18_repro_v001
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import v18_reference_dump as refmod

SUBSAMPLE_STRIDE = 13
SUBSAMPLE_MAX = 20000
MAX_ABS = 1e-7
MEAN_ABS = 1e-9


def digest(name: str, arr: np.ndarray, columns: list[str]) -> dict:
    a = np.asarray(arr, np.float32)
    idx = np.arange(0, a.shape[0], SUBSAMPLE_STRIDE)[:SUBSAMPLE_MAX]
    a64 = a.astype(np.float64)
    return {
        "name": name,
        "shape": list(a.shape),
        "columns": list(columns),
        "col_checksum": np.nansum(a64, axis=0),
        "col_nan": np.isnan(a).sum(axis=0),
        "sub_idx": idx,
        "sub": a[idx].copy(),
    }


def compare(ref: dict, cand: dict) -> dict:
    out = {
        "ref_shape": ref["shape"],
        "cand_shape": cand["shape"],
        "column_order_match": ref["columns"] == cand["columns"],
        "n_cols": len(cand["columns"]),
    }
    if ref["shape"] != cand["shape"] or ref["columns"] != cand["columns"]:
        out["status"] = "FAIL"
        out["note"] = "shape/column mismatch"
        only_r = [c for c in ref["columns"] if c not in cand["columns"]][:10]
        only_c = [c for c in cand["columns"] if c not in ref["columns"]][:10]
        out["only_in_reference"] = only_r
        out["only_in_candidate"] = only_c
        return out

    out["nan_count_match"] = bool(np.array_equal(ref["col_nan"], cand["col_nan"]))
    out["n_nan"] = int(cand["col_nan"].sum())

    cs = np.abs(ref["col_checksum"] - cand["col_checksum"])
    out["col_checksum_max_abs_diff"] = float(np.nanmax(cs)) if cs.size else 0.0

    a, b = ref["sub"], cand["sub"]
    na, nb = np.isnan(a), np.isnan(b)
    out["sub_nan_mask_match"] = bool(np.array_equal(na, nb))
    both = ~(na | nb)
    if both.any():
        d = np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
        out["sub_max_abs_diff"] = float(d.max())
        out["sub_mean_abs_diff"] = float(d.mean())
    else:
        out["sub_max_abs_diff"] = 0.0
        out["sub_mean_abs_diff"] = 0.0
    out["sub_rows"] = int(a.shape[0])

    ok = (
        out["column_order_match"]
        and out["nan_count_match"]
        and out["sub_nan_mask_match"]
        and out["sub_max_abs_diff"] < MAX_ABS
        and out["sub_mean_abs_diff"] < MEAN_ABS
        and out["col_checksum_max_abs_diff"] < 1e-2  # float64 합, 25만행 누적 허용
    )
    out["status"] = "PASS" if ok else "FAIL"
    if not ok:
        colmax = np.zeros(a.shape[1])
        for j in range(a.shape[1]):
            m = both[:, j]
            if m.any():
                colmax[j] = np.abs(a[m, j].astype(np.float64) - b[m, j].astype(np.float64)).max()
        worst = np.argsort(-colmax)[:8]
        out["worst_columns"] = [
            {"col": cand["columns"][int(j)], "sub_max_abs_diff": float(colmax[int(j)]),
             "checksum_diff": float(cs[int(j)])}
            for j in worst
        ]
    return out


# ------------------------------------------------------------------
# reference 측 — notebook cell 5 fit_recent_base 의 행렬 구성만 그대로 재현
# ------------------------------------------------------------------
def reference_anchor_inputs(ns: dict, year: int) -> dict:
    TARGET = ns["TARGET"]
    screen = ns["screen"]
    RAWNUM = ns["RAWNUM"]
    source_year = int(year) - 1

    source = screen.loc[screen.season.eq(int(source_year))].reset_index(drop=True)
    target = screen.loc[screen.season.eq(int(year))].reset_index(drop=True)

    ss, gs = ns["year_state"](source, source_year)
    st, gt = ns["year_state"](target, year)
    A0 = pd.concat(
        [source[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True), ss, gs],
        axis=1,
    )
    B0 = pd.concat(
        [target[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True), st, gt],
        axis=1,
    )
    ca, cb, _ = ns["encode_cats"](source, target)
    A = pd.concat([A0, ca], axis=1)
    B = pd.concat([B0, cb], axis=1)
    out = {
        "exact_lgb_A": digest("exact_lgb_A", A.to_numpy(np.float32), list(A.columns)),
        "exact_lgb_B": digest("exact_lgb_B", B.to_numpy(np.float32), list(B.columns)),
    }
    del A, B, ca, cb
    gc.collect()

    Ar = A0.copy()
    Br = B0.copy()
    ridge_cols = list(A0.columns)
    del A0, B0, ss, gs, st, gt
    gc.collect()
    Ar = Ar.replace([np.inf, -np.inf], np.nan)
    Br = Br.replace([np.inf, -np.inf], np.nan)
    med = Ar.median().fillna(0.0)
    Ar = Ar.fillna(med).to_numpy(np.float32)
    Br = Br.fillna(med).to_numpy(np.float32)
    mu = Ar.mean(0)
    sd = Ar.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Ar = np.clip((Ar - mu) / sd, -8, 8)
    Br = np.clip((Br - mu) / sd, -8, 8)
    out["ridge_A"] = digest("ridge_A", Ar, ridge_cols)
    out["ridge_B"] = digest("ridge_B", Br, ridge_cols)
    del Ar, Br
    gc.collect()

    At, Bt = ns["simple_pair"](source, target)
    out["trend_lgb_A"] = digest("trend_lgb_A", At.to_numpy(np.float32), list(At.columns))
    out["trend_lgb_B"] = digest("trend_lgb_B", Bt.to_numpy(np.float32), list(Bt.columns))
    del At, Bt
    gc.collect()

    DOMAIN_MEANS = ns["DOMAIN_MEANS"]
    slope = ns["trailing_core_slope"](source_year)
    base = np.array(
        [float(DOMAIN_MEANS.loc[int(source_year), d]) for d in source.domain3.to_numpy()],
        dtype=np.float64,
    )
    prior = {d: float(DOMAIN_MEANS.loc[int(source_year), d] + slope)
             for d in ["R_CORE", "R_ANCHOR", "F"]}
    bt = np.array([prior[d] for d in target.domain3.to_numpy()], dtype=np.float64)
    out["trend_base"] = digest("trend_base", base.reshape(-1, 1), ["trend_base"])
    out["trend_bt"] = digest("trend_bt", bt.reshape(-1, 1), ["trend_bt"])
    out["_scalars"] = {"trend_slope": float(slope), "trend_prior": prior,
                       "residual_target_mean": float((source[TARGET].to_numpy(np.float64) - base).mean())}
    del source, target
    gc.collect()
    return out


# ------------------------------------------------------------------
# candidate 측 — train.py 의 함수만 사용
# ------------------------------------------------------------------
def candidate_anchor_inputs(tp, year: int) -> dict:
    screen = tp.load_train()
    tp.validate_schema(screen)
    anchor_team_id = tp.fit_router(screen)
    screen = tp.attach_domain3(screen, anchor_team_id)
    tp.finalize_column_lists(screen)
    dm = tp.DomainMeans(screen)

    TARGET = tp.TARGET
    RAWNUM = tp.RAWNUM
    source_year = int(year) - 1
    source = screen.loc[screen[tp.SEASON_COL].eq(int(source_year))].reset_index(drop=True)
    target = screen.loc[screen[tp.SEASON_COL].eq(int(year))].reset_index(drop=True)

    ss, gs, _ = tp.year_state(screen, source, source_year)
    st, gt, _ = tp.year_state(screen, target, year)
    A0 = pd.concat(
        [source[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True), ss, gs],
        axis=1,
    )
    B0 = pd.concat(
        [target[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True), st, gt],
        axis=1,
    )
    ca, cb, _ = tp.encode_cats(source, target)
    A = pd.concat([A0, ca], axis=1)
    B = pd.concat([B0, cb], axis=1)
    out = {
        "exact_lgb_A": digest("exact_lgb_A", A.to_numpy(np.float32), list(A.columns)),
        "exact_lgb_B": digest("exact_lgb_B", B.to_numpy(np.float32), list(B.columns)),
    }
    del A, B, ca, cb
    gc.collect()

    Ar = A0.copy()
    Br = B0.copy()
    ridge_cols = list(A0.columns)
    del A0, B0, ss, gs, st, gt
    gc.collect()
    Ar = Ar.replace([np.inf, -np.inf], np.nan)
    Br = Br.replace([np.inf, -np.inf], np.nan)
    med = Ar.median().fillna(0.0)
    Ar = Ar.fillna(med).to_numpy(np.float32)
    Br = Br.fillna(med).to_numpy(np.float32)
    mu = Ar.mean(0)
    sd = Ar.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Ar = np.clip((Ar - mu) / sd, -8, 8)
    Br = np.clip((Br - mu) / sd, -8, 8)
    out["ridge_A"] = digest("ridge_A", Ar, ridge_cols)
    out["ridge_B"] = digest("ridge_B", Br, ridge_cols)
    del Ar, Br
    gc.collect()

    At, Bt = tp.simple_pair(source, target)
    out["trend_lgb_A"] = digest("trend_lgb_A", At.to_numpy(np.float32), list(At.columns))
    out["trend_lgb_B"] = digest("trend_lgb_B", Bt.to_numpy(np.float32), list(Bt.columns))
    del At, Bt
    gc.collect()

    slope = dm.trailing_core_slope(source_year)
    base = np.array(
        [dm.value(int(source_year), d, source_year) for d in source.domain3.to_numpy()],
        dtype=np.float64,
    )
    prior = {d: dm.value(int(source_year), d, source_year) + slope
             for d in ["R_CORE", "R_ANCHOR", "F"]}
    bt = np.array([prior[d] for d in target.domain3.to_numpy()], dtype=np.float64)
    out["trend_base"] = digest("trend_base", base.reshape(-1, 1), ["trend_base"])
    out["trend_bt"] = digest("trend_bt", bt.reshape(-1, 1), ["trend_bt"])
    out["_scalars"] = {
        "trend_slope": float(slope), "trend_prior": prior,
        "residual_target_mean": float((source[TARGET].to_numpy(np.float64) - base).mean()),
        "domain_means_future_access": int(dm.future_access_count()),
    }
    del source, target, screen
    gc.collect()
    return out


KEYS = ["exact_lgb_A", "exact_lgb_B", "ridge_A", "ridge_B",
        "trend_lgb_A", "trend_lgb_B", "trend_base", "trend_bt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--year", type=int, action="append", required=True)
    ap.add_argument("--validation-dir", default="validation/v18_repro_v001")
    args = ap.parse_args()

    vdir = Path(args.validation_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    out_path = vdir / "gate_r3_anchor_input.json"
    report = {"thresholds": {"sub_max_abs": MAX_ABS, "sub_mean_abs": MEAN_ABS},
              "subsample": {"stride": SUBSAMPLE_STRIDE, "max": SUBSAMPLE_MAX},
              "years": {}}
    if out_path.exists():
        try:
            report = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for year in args.year:
        print(f"[gate] === year {year} : reference side ===", flush=True)
        ns, patches = refmod.build_namespace(Path(args.notebook), Path(args.data_dir), "features")
        ref = reference_anchor_inputs(ns, year)
        ref_scalars = ref.pop("_scalars")
        del ns
        gc.collect()

        print(f"[gate] === year {year} : candidate side (train.py) ===", flush=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        tp = importlib.import_module("train")
        cand = candidate_anchor_inputs(tp, year)
        cand_scalars = cand.pop("_scalars")
        gc.collect()

        y = {k: compare(ref[k], cand[k]) for k in KEYS}
        y["scalars"] = {
            "trend_slope_diff": abs(ref_scalars["trend_slope"] - cand_scalars["trend_slope"]),
            "trend_prior_max_diff": max(
                abs(ref_scalars["trend_prior"][d] - cand_scalars["trend_prior"][d])
                for d in ref_scalars["trend_prior"]
            ),
            "residual_target_mean_diff": abs(
                ref_scalars["residual_target_mean"] - cand_scalars["residual_target_mean"]
            ),
            "domain_means_future_access": cand_scalars["domain_means_future_access"],
        }
        y["scalars"]["status"] = (
            "PASS"
            if max(y["scalars"]["trend_slope_diff"], y["scalars"]["trend_prior_max_diff"]) < 1e-12
            and y["scalars"]["domain_means_future_access"] == 0
            else "FAIL"
        )
        report["years"][str(year)] = y
        del ref, cand
        gc.collect()

        print(f"\n{'Gate':<26}{'cols':>6}{'order':>7}{'nan':>6}"
              f"{'sub_max':>12}{'chk_max':>12}  Status")
        print("-" * 80)
        for k in KEYS:
            r = y[k]
            print(f"{k:<26}{r.get('n_cols', 0):>6}"
                  f"{'Y' if r.get('column_order_match') else 'N':>7}"
                  f"{'Y' if r.get('nan_count_match') else 'N':>6}"
                  f"{r.get('sub_max_abs_diff', float('nan')):>12.3e}"
                  f"{r.get('col_checksum_max_abs_diff', float('nan')):>12.3e}"
                  f"  {r['status']}")
        print(f"{'scalars':<26}{'':>6}{'':>7}{'':>6}"
              f"{y['scalars']['trend_slope_diff']:>12.3e}"
              f"{y['scalars']['trend_prior_max_diff']:>12.3e}  {y['scalars']['status']}")
        print()

    statuses = {v["status"] for yv in report["years"].values() for v in yv.values()}
    report["overall"] = "FAIL" if "FAIL" in statuses else "PASS"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print("OVERALL:", report["overall"])
    print("  ->", out_path)


if __name__ == "__main__":
    main()

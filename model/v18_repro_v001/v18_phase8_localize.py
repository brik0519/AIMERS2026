"""Phase 8 진단 — R8 실패 지점을 층별로 특정한다.

R8 이 `~1e-8` 로 실패했다. 잡음 하한 `2.22e-16`(§I-11) 보다 8자릿수 크므로
**단순 부동소수 잡음으로 설명되지 않는다.** 어느 단계에서 커지는지 찾는다.

측정 지점 (같은 행을 전체 프레임 / 단일 행 두 경로로 계산해 비교):

    1. selfx / gx / B0        row-local feature (T2 에서 0 확인된 구간)
    2. p1  LGBMClassifier
    3. p2  Ridge              ← BLAS 가 (1,89) 와 (N,89) 에서 다른 커널을 쓴다
    4. p3  LGBMRegressor
    5. base = clip((p1+p2+p3)/3)
    6. base_pred = float32(base)   ← float32 양자화 경계
    7. X (adapter 128열)           ← base_pred 가 열 하나로 들어간다
    8. corr (ExtraTrees seed 평균)  ← X 가 1 ulp 바뀌면 split 이 뒤집힐 수 있다
    9. final

추가:
    · 전체 배치 2회 실행 재현성 (같은 shape 를 두 번 → 잡음인가 shape 의존인가)
    · 배치 크기별 예측 (1 / 8 / 64 / 512 / 전체) → 크기 의존성 곡선

사용법:
    python3 v18_phase8_localize.py --rows 4000 --samples 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script as S  # noqa: E402
import v18_phase8_audit as A8  # noqa: E402


def base_parts(test: pd.DataFrame) -> dict:
    """script.py `transform_base` 를 단계별로 쪼개 중간값을 남긴다 (계산은 동일)."""
    rawnum = S.META["rawnum"]
    selfx = S.build_self_state(test)
    gx = S.transform_global_season(S.META["target_global_bank"], test)
    B0 = pd.concat(
        [test[rawnum].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
         selfx.reset_index(drop=True), gx.reset_index(drop=True)], axis=1)

    cb = S.encode_with_maps(test, S.META["exact_cat_maps"])
    B_exact = pd.concat([B0, cb], axis=1).reindex(columns=S.META["exact_feature_cols"])
    p1 = S.META["base_exact_lgb"].predict_proba(B_exact)[:, 1]

    B = B0.reindex(columns=S.META["ridge_feature_cols"]).replace([np.inf, -np.inf], np.nan)
    med = pd.Series(S.META["ridge_med"])
    Bz = B.fillna(med).to_numpy(np.float32)
    mu = np.asarray(S.META["ridge_mu"], dtype=np.float32)
    sd = np.asarray(S.META["ridge_sd"], dtype=np.float32)
    Bz = np.clip((Bz - mu) / sd, -8, 8)
    p2 = np.clip(S.META["base_ridge"].predict(Bz), 0.001, 0.999)

    t = S.add_eng(test)
    Bnum = t[S.META["trend_num_cols"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    cb2 = S.encode_with_maps(t, S.META["trend_cat_maps"])
    Bt = pd.concat([Bnum, cb2], axis=1).astype(np.float32).reindex(
        columns=S.META["trend_feature_cols"])
    prior = S.META["trend_prior"]
    bt = np.array([float(prior[d]) for d in test.domain3.to_numpy()], dtype=np.float64)
    p3 = np.clip(bt + S.META["base_trend_lgb"].predict(Bt), 0.001, 0.999)

    base = np.clip((np.asarray(p1, float) + np.asarray(p2, float)
                    + np.asarray(p3, float)) / 3.0, 0, 1)
    return {"selfx": selfx, "gx": gx, "B0": B0, "Bz_ridge": Bz,
            "p1": p1, "p2": p2, "p3": p3, "base": base}


def stage_values(test: pd.DataFrame) -> dict:
    """전 단계 값을 한 번에 만든다."""
    bp = base_parts(test)
    base = bp["base"]
    X = S.transform_v15_adapter(test, base)
    corr = np.zeros(len(test), dtype=np.float64)
    per_seed = {}
    for seed in S.META["adapter_seeds"]:
        c = np.zeros(len(test), dtype=np.float64)
        for d in ["R_CORE", "R_ANCHOR", "F"]:
            mask = test.domain3.eq(d).to_numpy()
            if not mask.any():
                continue
            c[mask] = S.META["adapter_models"][(int(seed), d)].predict(X.loc[mask])
        per_seed[int(seed)] = c
        corr += c / len(S.META["adapter_seeds"])
    final = np.clip(base + float(S.META["adapter_shrink"]) * corr, 0, 1)
    if S.META.get("v18_stable_enabled", False):
        Xs = S.transform_v18_stable(test)
        arr = Xs.to_numpy(np.float32)
        mu = np.asarray(S.META["v18_stable_mu"], dtype=np.float32)
        sd = np.asarray(S.META["v18_stable_sd"], dtype=np.float32)
        arr = np.clip((arr - mu) / sd, -8, 8)
        sc = np.asarray(S.META["v18_stable_model"].predict(arr), dtype=np.float64)
        core = test.domain3.eq(S.META.get("v18_route_domain", "R_CORE")).to_numpy()
        final[core] = np.clip(final[core] + float(S.META["v18_stable_gamma"]) * sc[core], 0, 1)
    else:
        sc = np.zeros(len(test))
    return {
        "selfx": bp["selfx"].to_numpy(np.float64),
        "gx": bp["gx"].to_numpy(np.float64),
        "B0": bp["B0"].to_numpy(np.float64),
        "Bz_ridge": bp["Bz_ridge"].astype(np.float64),
        "p1": np.asarray(bp["p1"], np.float64),
        "p2": np.asarray(bp["p2"], np.float64),
        "p3": np.asarray(bp["p3"], np.float64),
        "base": base,
        "base_pred_f32": np.asarray(base, np.float32).astype(np.float64),
        "X": X.to_numpy(np.float64),
        "X_base_pred_col": X["base_pred"].to_numpy(np.float64),
        **{f"corr_seed{s}": v for s, v in per_seed.items()},
        "corr": corr,
        "stable_corr": sc,
        "final": final,
    }


STAGES = ["selfx", "gx", "B0", "Bz_ridge", "p1", "p2", "p3", "base",
          "base_pred_f32", "X_base_pred_col", "X", "corr", "stable_corr", "final"]


def _row(v, i):
    a = np.asarray(v)
    return a[i] if a.ndim == 1 else a[i, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--validation-dir", default="validation/v18_repro_v001")
    args = ap.parse_args()

    print("=" * 84)
    print("Phase 8 진단 — R8 실패 지점 층별 특정")
    print("=" * 84)
    frame = A8.build_proxy_test(args.rows, args.seed)
    n = len(frame)
    print(f"proxy n={n}  domain3={frame.domain3.value_counts().to_dict()}")
    report = {"n_rows": n, "samples": args.samples}

    # ---------- 0a. 가설 검증 : Ridge coef_ 가 float32 인가 ----------
    # 관측된 차이 9.9341e-09 = 2^-25 / 3 이다.
    #   2^-25 = 2.98e-08 는 [0.25,0.5) 구간 float32 ulp
    #   /3    은 base = (p1+p2+p3)/3
    # V18 은 Ridge 를 float32 행렬(Az/Ar)로 적합한다. sklearn 은 float32 입력을 보존하므로
    # coef_ 도 float32 가 되고, predict 는 **float32 BLAS** 로 계산된다.
    # float32 gemv 는 행 수에 따라 blocking/vectorization 이 달라져 행별 결과가
    # 1 float32 ulp 수준으로 흔들린다 → 이것이 R8 실패의 근원이라는 가설.
    print("\n[0a] 모델 계수 dtype (가설: float32 이면 predict 가 float32 BLAS)")
    dtypes = {}
    for name in ("base_ridge", "v18_stable_model"):
        m = S.META.get(name)
        for attr in ("coef_", "intercept_"):
            v = getattr(m, attr, None)
            if v is not None:
                dt = str(np.asarray(v).dtype)
                dtypes[f"{name}.{attr}"] = dt
                print(f"    {name}.{attr:<12} {dt}")
    report["model_coef_dtypes"] = dtypes

    # 같은 계수를 float64 로 올려 직접 계산하면 shape 의존이 사라지는가
    print("\n[0b] Ridge 를 float64 로 직접 계산했을 때 shape 의존 (결정적 확인)")
    bp_full = base_parts(frame)
    Bz_full = bp_full["Bz_ridge"]
    coef = np.asarray(S.META["base_ridge"].coef_, np.float64).ravel()
    icpt = float(np.asarray(S.META["base_ridge"].intercept_).ravel()[0])
    rng0 = np.random.default_rng(args.seed)
    probe0 = sorted(rng0.choice(n, size=min(20, n), replace=False).tolist())
    d_sklearn, d_manual = 0.0, 0.0
    for i in probe0:
        one = frame.iloc[[i]].reset_index(drop=True)
        Bz_one = base_parts(one)["Bz_ridge"]
        # (a) sklearn predict
        d_sklearn = max(d_sklearn, abs(
            float(S.META["base_ridge"].predict(Bz_one)[0])
            - float(S.META["base_ridge"].predict(Bz_full[i:i + 1])[0])))
        # (b) float64 수동 계산
        v_one = float(Bz_one.astype(np.float64) @ coef + icpt)
        v_full = float(Bz_full[i].astype(np.float64) @ coef + icpt)
        d_manual = max(d_manual, abs(v_one - v_full))
        # 입력 자체가 같은지도 확인
        assert np.array_equal(Bz_one[0], Bz_full[i]), f"Bz 입력이 다르다 row={i}"
    print(f"    Bz 입력 동일성            : OK (모든 표본 비트 일치)")
    print(f"    sklearn predict 차이       : {d_sklearn:.3e}")
    print(f"    float64 수동계산 차이      : {d_manual:.3e}")
    report["ridge_shape_dependence"] = {
        "sklearn_predict_max_diff": d_sklearn,
        "manual_float64_max_diff": d_manual,
        "input_identical": True,
    }
    if d_manual < 1e-15 <= d_sklearn:
        print("    -> 가설 확정: 입력은 같은데 sklearn predict 만 shape 에 따라 달라진다")
        report["ridge_hypothesis"] = "CONFIRMED: float32 BLAS shape dependence"

    # ---------- 0. 전체 배치 2회 실행 재현성 ----------
    print("\n[0] 같은 shape 2회 실행 (잡음인가 shape 의존인가)")
    a = stage_values(frame)
    b = stage_values(frame)
    rep = {}
    for k in STAGES:
        d = float(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)).max())
        rep[k] = d
        print(f"    {k:18s} {d:.3e}")
    report["run_to_run_full_batch"] = rep
    noise = max(rep.values())
    print(f"    -> 실행 잡음 하한 = {noise:.3e}")

    # ---------- 1. 단일 행 vs 전체 : 단계별 ----------
    rng = np.random.default_rng(args.seed)
    idx = set()
    for d in ["R_CORE", "R_ANCHOR", "F"]:
        pool = np.flatnonzero(frame.domain3.eq(d).to_numpy())
        if len(pool):
            idx |= set(rng.choice(pool, size=min(max(args.samples // 3, 2), len(pool)),
                                  replace=False).tolist())
    idx = sorted(idx)
    print(f"\n[1] 단일 행 vs 전체 — 단계별 최대 차이 ({len(idx)}행)")
    worst = {k: 0.0 for k in STAGES}
    first_break = None
    for i in idx:
        one = stage_values(frame.iloc[[i]].reset_index(drop=True))
        for k in STAGES:
            d = float(np.abs(np.asarray(_row(a[k], i), np.float64)
                             - np.asarray(_row(one[k], 0), np.float64)).max())
            worst[k] = max(worst[k], d)
    for k in STAGES:
        flag = ""
        if worst[k] > max(noise, 1e-15) * 10 and first_break is None:
            first_break = k
            flag = "   <-- 여기서 커진다"
        print(f"    {k:18s} {worst[k]:.3e}{flag}")
    report["single_vs_full_by_stage"] = worst
    report["first_amplifying_stage"] = first_break

    # ---------- 2. 배치 크기 의존성 ----------
    print("\n[2] 배치 크기별 final 차이 (전체 배치 기준)")
    sizes = [s for s in (1, 8, 64, 512, 4096) if s <= n]
    probe = idx[: min(6, len(idx))]
    size_rep = {}
    for sz in sizes:
        worst_sz = 0.0
        for i in probe:
            lo = max(0, min(i, n - sz))
            sub = frame.iloc[lo:lo + sz].reset_index(drop=True)
            pos = i - lo
            v = stage_values(sub)["final"][pos]
            worst_sz = max(worst_sz, abs(float(v) - float(a["final"][i])))
        size_rep[sz] = worst_sz
        print(f"    batch={sz:>5}  max_abs_diff={worst_sz:.3e}")
    report["batch_size_dependence"] = size_rep

    out = Path(args.validation_dir) / "gate_r8_localize.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("\n" + "=" * 84)
    print(f"실행 잡음(같은 shape 2회) : {noise:.3e}")
    print(f"최초 증폭 단계            : {first_break}")
    print(f"final 단일행 vs 전체      : {worst['final']:.3e}")
    print("  ->", out)


if __name__ == "__main__":
    main()

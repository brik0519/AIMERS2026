"""deploy_validation.py — 실험 deployval_v001.

목적
----
train.py Phase 6 (2024 -> 2025 배포 regime) 을 **T-1 -> T** 로 그대로 재연해 라벨이 있는 T 에서
채점한다. 지금까지의 검증(Phase 4 crossfit / Phase 5 전이)은 배포 regime 과 다섯 가지가 달라
LB 추정치가 아니었다. 이 harness 가 처음으로 배포와 같은 경로를 채점한다.

    validation (기존)                 deploy (실제 제출)                이 harness
    L1 : T 안에서 crossfit, 100트리   L1 : T-1 전체 학습, 120트리        deploy 와 동일
    L1 target : y_T - anchor(T-1->T)  L1 target : y_{T-1} - anchor(T-2->T-1)  deploy 와 동일
    L1 적용 : T 의 OOF                L1 적용 : anchor(T-1->T) 의 direct 예측 deploy 와 동일
    L2 : T-1 OOF 잔차로 학습, T OOF 보정   L2 : T-1 OOF 잔차로 학습, direct 보정  deploy 와 동일

ExtraTrees 는 전이당 한 번만 적합하고(9회), 보정값을 저장해 두므로 아래는 재적합 없이 격자를 돈다.
    RESIDUAL_SHRINK  ∈ SHRINK_GRID   (배포 상수 0.50 이 배포 regime 에서도 최적인가)
    SHIFT_SCALE      ∈ SCALE_GRID    (0.75 가 배포 regime 에서도 최적인가)
    Layer 2 shift vs reference

규칙
----
모든 통계·모델은 T-1 이하의 공식 train 만 사용한다. T 는 채점에만 쓰인다 (라벨은 BSS 계산에만).
격자 선택도 train 전이에서만 이루어지며 test 는 어디에도 관여하지 않는다.

사용
----
    python deploy_validation.py                      # T = 2021..2024
    DEPLOYVAL_YEARS=2023,2024 python deploy_validation.py
    python deploy_validation.py --check-2025         # 2024->2025 경로가 Phase 6 산출물과 bit-identical 인지
입력 캐시: SHIFT_CACHE_DIR (기본 validation/shift_v001/matrices) 의 anchor_{y}.npz, adjusted_oof_{y}.npz
출력    : validation/deployval_v001/{deploy_transitions.csv, shrink_grid.csv, scale_grid.csv,
           gap_vs_validation.csv, summary.json, matrices/deploy_{T}.npz}
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train as T   # 학습 코드의 함수를 그대로 사용한다 (오프라인 실험 전용, 제출물 아님)

# =========================================================
# 경로
# =========================================================
EXPERIMENT_NAME = "deployval_v001"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
MATRIX_DIR = VALIDATION_DIR / "matrices"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

_CACHE_ENV = os.environ.get("SHIFT_CACHE_DIR", "").strip()
CACHE_DIR = Path(_CACHE_ENV) if _CACHE_ENV else (VALIDATION_ROOT / "shift_v001" / "matrices")

_YEARS_ENV = os.environ.get("DEPLOYVAL_YEARS", "").strip()
TARGET_YEARS = [int(y) for y in _YEARS_ENV.split(",")] if _YEARS_ENV else [2021, 2022, 2023, 2024]

SHRINK_GRID = [0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
SCALE_GRID = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]

DEPLOY_SHRINK = float(T.RESIDUAL_SHRINK)    # 0.50
DEPLOY_SCALE = float(T.SHIFT_SCALE)         # 0.75


def _load_npz(name: str, aliases: list[str]):
    for n in [name] + aliases:
        p = CACHE_DIR / n
        if p.exists():
            return np.load(p, allow_pickle=True), p
    raise FileNotFoundError(f"{name} not in {CACHE_DIR}")


# =========================================================
# 배포 경로 재연 (train.py run_phase6 의 학습부를 source/target 으로 파라미터화)
# =========================================================

def deploy_transition(screen: pd.DataFrame, segment_means: T.SegmentMeans,
                      source_year: int, target: pd.DataFrame, target_year: int,
                      log=print) -> dict:
    """run_phase6 과 동일한 학습 경로. target 은 라벨을 떼고 test 처럼 취급한다.

    반환: p_base(direct), residual correction(seed 평균, shrink 미적용),
          shift/reference correction(scale 미적용), core mask, 메타.
    """
    T._import_phase3_libs()
    T._import_phase4_libs()
    lgb, Ridge, ExtraTreesRegressor = T.lgb, T.Ridge, T.ExtraTreesRegressor
    RAWNUM, ENG_COLS, BANK_COLS = T.RAWNUM, T.ENG_COLS, T.BANK_COLS

    raw = screen.loc[screen[T.SEASON_COL].eq(source_year)].reset_index(drop=True)
    y_src = raw[T.TARGET].to_numpy(np.float64)
    test = target.drop(columns=[T.TARGET], errors="ignore").reset_index(drop=True)

    za, za_path = _load_npz(f"anchor_{source_year}.npz", [])
    p_base_src = za["p_base"].astype(np.float64)
    zo, zo_path = _load_npz(f"adjusted_oof_{source_year}.npz", [f"v15_oof_{source_year}.npz"])
    adjusted_oof_src = (zo["adjusted_oof"] if "adjusted_oof" in zo.files else zo["v15_oof"]).astype(np.float64)
    assert np.array_equal(zo["target"].astype(np.float64), y_src), "adjusted_oof target mismatch"
    assert float(np.abs(zo["p_base"].astype(np.float64) - p_base_src).max()) == 0.0
    assert len(p_base_src) == len(raw)
    log(f"    anchor <- {za_path}\n    adjusted_oof <- {zo_path}")

    # ---- 1) shared base : source 전체 학습 -> target direct ----
    source = raw
    ss = T.build_self_state(source.drop(columns=[T.TARGET], errors="ignore"))
    source_bank = T.fit_global_season_bank(
        screen.loc[screen[T.SEASON_COL].lt(source_year), BANK_COLS], source_year)
    gs = T.transform_global_season(source_bank, source.drop(columns=[T.TARGET], errors="ignore"))
    target_bank = T.fit_global_season_bank(
        screen.loc[screen[T.SEASON_COL].lt(target_year), BANK_COLS], target_year)
    st = T.build_self_state(test)
    gt = T.transform_global_season(target_bank, test)
    A0 = pd.concat([source[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                    ss.reset_index(drop=True), gs.reset_index(drop=True)], axis=1)
    B0 = pd.concat([test[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                    st.reset_index(drop=True), gt.reset_index(drop=True)], axis=1)

    ca, cb, _ = T.encode_cats(source, test)
    A_exact = pd.concat([A0, ca], axis=1)
    B_exact = pd.concat([B0, cb], axis=1)
    assert A_exact.shape[1] == T.EXPECTED_DIMS["exact_lgb_features"]
    m1 = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=T.N_JOBS, **T.BASE_CFG["exact_lgb"])
    m1.fit(A_exact, y_src)
    p1 = m1.predict_proba(B_exact)[:, 1]
    del A_exact, B_exact, ca, cb
    gc.collect()

    Ar = A0.replace([np.inf, -np.inf], np.nan)
    Br = B0.replace([np.inf, -np.inf], np.nan)
    ridge_med = Ar.median().fillna(0.0)
    Ar = Ar.fillna(ridge_med).to_numpy(np.float32)
    Br = Br.fillna(ridge_med).to_numpy(np.float32)
    ridge_mu = Ar.mean(0)
    ridge_sd = Ar.std(0)
    ridge_sd = np.where(ridge_sd > 1e-6, ridge_sd, 1.0)
    Ar = np.clip((Ar - ridge_mu) / ridge_sd, -8, 8)
    Br = np.clip((Br - ridge_mu) / ridge_sd, -8, 8)
    m2 = Ridge(alpha=T.BASE_CFG["ridge_alpha"], solver="lsqr", max_iter=100, tol=1e-3)
    m2.fit(Ar, y_src)
    p2 = np.clip(T.ridge_predict_rowwise(m2, Br), 0.001, 0.999)
    del Ar, Br, A0, B0, ss, gs
    gc.collect()

    s = T.add_eng(source)
    t = T.add_eng(test)
    trend_num_cols = RAWNUM + ENG_COLS
    An = s[trend_num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    Bn = t[trend_num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    cta, ctb, _ = T.encode_cats(s, t)
    At = pd.concat([An, cta], axis=1).astype(np.float32)
    Bt = pd.concat([Bn, ctb], axis=1).astype(np.float32)
    slope = segment_means.trailing_core_slope(source_year)
    base_dom = np.array([segment_means.value(source_year, d, source_year)
                         for d in source.route_segment.to_numpy()], dtype=np.float64)
    trend_prior = {d: segment_means.value(source_year, d, source_year) + slope
                   for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]}
    bt = np.array([trend_prior[d] for d in test.route_segment.to_numpy()], dtype=np.float64)
    m3 = lgb.LGBMRegressor(objective="regression_l2", verbosity=-1, n_jobs=T.N_JOBS, **T.BASE_CFG["trend_lgb"])
    m3.fit(At, y_src - base_dom)
    p3 = np.clip(bt + m3.predict(Bt), 0.001, 0.999)
    del At, Bt, An, Bn, cta, ctb, s, t
    gc.collect()

    p_base_tgt = np.clip((np.asarray(p1, float) + np.asarray(p2, float) + np.asarray(p3, float)) / 3.0, 0, 1)
    log(f"    p_base({source_year}->{target_year}) mean={float(p_base_tgt.mean()):.6f}  slope={slope:.10f}")

    # ---- 2) context 블록 : 학습 EB = source-1, 추론 EB = source ----
    pc_train_specs = T.make_context_specs(screen, source_year - 1)
    pc_test_specs = T.make_context_specs(screen, source_year)
    pc_train = T.transform_context_specs(pc_train_specs, raw)
    pc_test = T.transform_context_specs(pc_test_specs, test)
    ex_train = T.build_asof_state(screen, raw, source_year)
    ex_test = pd.concat([st.add_prefix("exact__"), gt.add_prefix("exact__")], axis=1) \
        .replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    prep = T.fit_residual_prep(raw)
    X_src_base = T.transform_residual_features(prep, raw, p_base_src)
    X_tgt_base = T.transform_residual_features(prep, test, p_base_tgt)
    X_src = pd.concat([X_src_base.reset_index(drop=True), pc_train.reset_index(drop=True),
                       ex_train.reset_index(drop=True)], axis=1)
    X_tgt = pd.concat([X_tgt_base.reset_index(drop=True), pc_test.reset_index(drop=True),
                       ex_test.reset_index(drop=True)], axis=1)
    assert list(X_src.columns) == list(X_tgt.columns)
    assert X_src.shape[1] == T.EXPECTED_DIMS["residual_matrix"] == 128
    assert np.isfinite(X_src.to_numpy(float)).all() and np.isfinite(X_tgt.to_numpy(float)).all()

    # ---- 3) residual adjuster : source 전체, 120 트리 ----
    res_src = y_src - p_base_src
    corrections = []
    for seed in T.CF_SEEDS:
        c = np.zeros(len(test), dtype=np.float64)
        for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]:
            tr = raw.route_segment.eq(d).to_numpy()
            te = test.route_segment.eq(d).to_numpy()
            t0 = time.time()
            m = ExtraTreesRegressor(n_jobs=T.N_JOBS, random_state=16200 + int(seed), **T.RESIDUAL_FINAL_CFG)
            m.fit(X_src.loc[tr], res_src[tr])
            if te.any():
                c[te] = m.predict(X_tgt.loc[te])
            log(f"      residual seed={seed} segment={d:<11} n_tr={int(tr.sum()):>7} "
                f"n_te={int(te.sum()):>7} sec={time.time() - t0:.1f}")
            del m
            gc.collect()
        corrections.append(c)
    correction = np.mean(np.column_stack(corrections), axis=1)

    # ---- 4) Layer 2 : source OOF 잔차 -> target direct 보정 ----
    sens_src = T.transform_sensitivity_specs(T.build_sensitivity_specs(screen, source_year), raw)
    sens_tgt = T.transform_sensitivity_specs(T.build_sensitivity_specs(screen, target_year), test)
    assert list(sens_src.columns) == list(sens_tgt.columns)
    A = sens_src.to_numpy(np.float32)
    B = sens_tgt.to_numpy(np.float32)
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    Az = np.clip((A - mu) / sd, -8, 8)
    Bz = np.clip((B - mu) / sd, -8, 8)
    ref_model, shift_model, corr_ref, corr_shift = T._fit_shift_pair(Az, Bz, y_src - adjusted_oof_src, rowwise=True)
    core = test.route_segment.eq(T.SHIFT_SEGMENT).to_numpy()

    return {
        "p_base": p_base_tgt, "correction": correction,
        "corr_shift": corr_shift, "corr_reference": corr_ref, "core": core,
        "segment": test.route_segment.to_numpy(),
        "meta": {"source_year": source_year, "target_year": target_year, "slope": float(slope),
                 "trend_prior": trend_prior, "reference_intercept": float(ref_model.intercept_),
                 "residual_target_mean": float(res_src.mean()),
                 "n_source": int(len(raw)), "n_target": int(len(test))},
    }


def compose(out: dict, shrink: float, scale: float, mode: str = "shift") -> np.ndarray:
    p = np.clip(out["p_base"] + shrink * out["correction"], 0, 1)
    if scale != 0.0:
        c = out["corr_shift"] if mode == "shift" else out["corr_reference"]
        core = out["core"]
        p[core] = np.clip(p[core] + scale * c[core], 0, 1)
    return p


# =========================================================
# main
# =========================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-2025", action="store_true",
                    help="2024->2025 경로를 재연해 Phase 6 deploy_2025.npz 와 bit-identical 인지 확인")
    args = ap.parse_args(argv)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("=" * 78)
    print(f"Experiment : {EXPERIMENT_NAME}   (deploy-faithful validation of shift_v001 pipeline)")
    print(f"CACHE_DIR  : {CACHE_DIR}")
    print(f"Targets    : {TARGET_YEARS}   shrink grid {SHRINK_GRID}   scale grid {SCALE_GRID}")
    print("=" * 78)

    # train.py main() 과 같은 순서: route_segment 를 붙인 뒤 컬럼 목록을 확정한다
    screen = T.load_train()
    T.validate_schema(screen)
    anchor_team_id = T.fit_router(screen)
    screen = T.attach_route_segment(screen, anchor_team_id)
    T.finalize_column_lists(screen)
    segment_means = T.SegmentMeans(screen)

    if args.check_2025:
        z = np.load(CACHE_DIR / "deploy_2025.npz", allow_pickle=True)
        test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
        test[T.SEASON_COL] = pd.to_numeric(test[T.SEASON_COL], errors="raise").astype(int)
        test["game_type"] = test["game_type"].astype("string").fillna("<NA>").astype(str)
        test = T.attach_route_segment(test, anchor_team_id)
        out = deploy_transition(screen, segment_means, 2024, test, 2025)
        p = compose(out, DEPLOY_SHRINK, DEPLOY_SCALE, "shift")
        d_base = float(np.abs(out["p_base"] - z["p_base25"].astype(np.float64)).max())
        d_final = float(np.abs(p - z["final_direct"].astype(np.float64)).max())
        print(f"[check-2025] max|Δ p_base|={d_base:.3e}  max|Δ final|={d_final:.3e}  "
              f"(0.0 이면 harness == Phase 6)")
        return

    rows, shrink_rows, scale_rows = [], [], []
    for ty in TARGET_YEARS:
        sy = ty - 1
        print(f"\n[deploy regime] {sy} -> {ty}")
        target = screen.loc[screen[T.SEASON_COL].eq(ty)].reset_index(drop=True)
        y = target[T.TARGET].to_numpy(np.float64)
        out = deploy_transition(screen, segment_means, sy, target, ty)
        bss = lambda p: T.brier_skill(y, p)
        core = out["core"]

        p_anchor = out["p_base"]
        p_adj = compose(out, DEPLOY_SHRINK, 0.0)
        p_ref = compose(out, DEPLOY_SHRINK, DEPLOY_SCALE, "reference")
        p_new = compose(out, DEPLOY_SHRINK, DEPLOY_SCALE, "shift")
        row = {
            "source_year": sy, "target_year": ty, "n": int(len(y)),
            "anchor_direct_bss": bss(p_anchor),
            "adjusted_direct_bss": bss(p_adj),
            "reference_bss": bss(p_ref),
            "final_bss": bss(p_new),
            "layer1_gain": bss(p_adj) - bss(p_anchor),
            "layer2_gain_reference": bss(p_ref) - bss(p_adj),
            "layer2_gain_shift": bss(p_new) - bss(p_adj),
            "final_vs_reference": bss(p_new) - bss(p_ref),
            "core_final_vs_reference": T.brier_skill(y[core], p_new[core]) - T.brier_skill(y[core], p_ref[core]),
            "anchor_bias": float(p_anchor.mean() - y.mean()),
            "adjusted_bias": float(p_adj.mean() - y.mean()),
            **{k: v for k, v in out["meta"].items() if k not in ("trend_prior",)},
        }
        rows.append(row)
        print(f"    anchor={row['anchor_direct_bss']:.2f}  adjusted={row['adjusted_direct_bss']:.2f}  "
              f"reference={row['reference_bss']:.2f}  final={row['final_bss']:.2f}  "
              f"L1 {row['layer1_gain']:+.2f}  L2(shift) {row['layer2_gain_shift']:+.2f}  "
              f"final_vs_ref {row['final_vs_reference']:+.2f}  bias anchor {row['anchor_bias']:+.5f} adj {row['adjusted_bias']:+.5f}")

        for sh in SHRINK_GRID:
            shrink_rows.append({"target_year": ty, "shrink": sh,
                                "bss_no_l2": bss(compose(out, sh, 0.0)),
                                "bss_shift": bss(compose(out, sh, DEPLOY_SCALE, "shift"))})
        for sc in SCALE_GRID:
            scale_rows.append({"target_year": ty, "scale": sc,
                               "bss_shift": bss(compose(out, DEPLOY_SHRINK, sc, "shift")),
                               "bss_reference": bss(compose(out, DEPLOY_SHRINK, sc, "reference"))})
        np.savez_compressed(MATRIX_DIR / f"deploy_{ty}.npz", target=y, p_base=p_anchor,
                            correction=out["correction"], corr_shift=out["corr_shift"],
                            corr_reference=out["corr_reference"], core=core,
                            segment=out["segment"].astype(str))
        del out, target
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(VALIDATION_DIR / "deploy_transitions.csv", index=False)
    sg = pd.DataFrame(shrink_rows)
    sg.to_csv(VALIDATION_DIR / "shrink_grid.csv", index=False)
    cg = pd.DataFrame(scale_rows)
    cg.to_csv(VALIDATION_DIR / "scale_grid.csv", index=False)

    # ---- 기존 검증(Phase 5) 과의 gap ----
    gap_rows = []
    p5 = VALIDATION_ROOT / "shift_v001" / "phase5_summary.json"
    if p5.exists():
        tr = {r["target_year"]: r for r in json.loads(p5.read_text(encoding="utf-8"))["transition_rows"]}
        for r in rows:
            v = tr.get(r["target_year"])
            if v:
                gap_rows.append({
                    "target_year": r["target_year"],
                    "validation_adjusted": v["adjusted_bss"], "deploy_adjusted": r["adjusted_direct_bss"],
                    "validation_final": v["final_bss"], "deploy_final": r["final_bss"],
                    "gap_adjusted": r["adjusted_direct_bss"] - v["adjusted_bss"],
                    "gap_final": r["final_bss"] - v["final_bss"],
                    "validation_l2_gain": v["final_gain"], "deploy_l2_gain": r["layer2_gain_shift"],
                })
        pd.DataFrame(gap_rows).to_csv(VALIDATION_DIR / "gap_vs_validation.csv", index=False)

    # ---- 보고 ----
    print("\n" + "=" * 78)
    print("[deploy-faithful] transition | anchor | adjusted(L1 .50) | reference | final | L1 | L2 | final_vs_ref")
    for r in rows:
        print(f"  {r['source_year']}->{r['target_year']}  {r['anchor_direct_bss']:>8.2f}  "
              f"{r['adjusted_direct_bss']:>8.2f}  {r['reference_bss']:>8.2f}  {r['final_bss']:>8.2f}  "
              f"{r['layer1_gain']:>+7.2f}  {r['layer2_gain_shift']:>+6.2f}  {r['final_vs_reference']:>+6.2f}")
    print(f"  sum final_vs_reference = {df.final_vs_reference.sum():+.2f}   "
          f"min = {df.final_vs_reference.min():+.2f}")

    piv = sg.pivot(index="shrink", columns="target_year", values="bss_shift")
    piv["sum"] = piv.sum(axis=1)
    piv["min_gain_vs_default"] = (piv[TARGET_YEARS].sub(piv.loc[DEPLOY_SHRINK, TARGET_YEARS])).min(axis=1)
    print("\n[shrink grid — Layer 2 shift 포함, BSS]")
    print(piv.round(2).to_string())
    best_shrink = float(piv["sum"].idxmax())
    print(f"  sum 최대 shrink = {best_shrink}  (배포 상수 {DEPLOY_SHRINK})")

    piv2 = cg.pivot(index="scale", columns="target_year", values="bss_shift")
    piv2["sum"] = piv2.sum(axis=1)
    print("\n[scale grid — shrink 0.50 고정, Layer 2 shift, BSS]")
    print(piv2.round(2).to_string())
    best_scale = float(piv2["sum"].idxmax())
    print(f"  sum 최대 scale = {best_scale}  (배포 상수 {DEPLOY_SCALE})")

    if gap_rows:
        g = pd.DataFrame(gap_rows)
        print("\n[gap: deploy-faithful − Phase 5 validation]")
        print(g[["target_year", "validation_final", "deploy_final", "gap_final",
                 "validation_l2_gain", "deploy_l2_gain"]].round(2).to_string(index=False))

    # ---- 판정 규칙 (train 전이만 사용) ----
    # 배포 상수 변경은 4개 전이 모두에서 기본값 이상이고 합이 개선될 때만 제안한다.
    def robust_choice(piv_, default):
        cand = piv_[piv_["min_gain_vs_default"] >= 0.0] if "min_gain_vs_default" in piv_ else piv_
        if cand.empty:
            return default, "no candidate dominates default on every transition"
        b = float(cand["sum"].idxmax())
        return b, f"dominates default on every transition; sum {piv_.loc[b, 'sum']:.2f} vs {piv_.loc[default, 'sum']:.2f}"
    piv2["min_gain_vs_default"] = (piv2[TARGET_YEARS].sub(piv2.loc[DEPLOY_SCALE, TARGET_YEARS])).min(axis=1)
    shrink_choice, shrink_why = robust_choice(piv, DEPLOY_SHRINK)
    scale_choice, scale_why = robust_choice(piv2, DEPLOY_SCALE)
    print(f"\n[recommend] RESIDUAL_SHRINK {DEPLOY_SHRINK} -> {shrink_choice}  ({shrink_why})")
    print(f"[recommend] SHIFT_SCALE     {DEPLOY_SCALE} -> {scale_choice}  ({scale_why})")

    summary = {
        "experiment_name": EXPERIMENT_NAME, "targets": TARGET_YEARS,
        "deploy_transitions": rows, "shrink_grid": shrink_rows, "scale_grid": scale_rows,
        "gap_vs_validation": gap_rows,
        "recommendation": {"residual_shrink": shrink_choice, "residual_shrink_reason": shrink_why,
                           "shift_scale": scale_choice, "shift_scale_reason": scale_why,
                           "rule": "change a deploy constant only if it is >= default on every transition and improves the sum"},
        "library_versions": T.library_versions(),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    T.write_json(VALIDATION_DIR / "summary.json", summary)
    print(f"\nDONE  elapsed={time.time() - t_all:.1f}s  -> {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
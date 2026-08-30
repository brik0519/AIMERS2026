# -*- coding: utf-8 -*-
"""rfold_v001 — R 브랜치 공통 fold 하네스 (청사진 34_*).

Phase 1  danker L1(treatment) walk-forward 재생성 (esfix: eval_set=R-only)
         + fold 2024 한정 control/dynamic (v002 median 용)
Phase 2  iter_table + T_final seed median (v002 상수)
Phase 3  L2 단일 vs 다년화 — 전이별 paired (L2 첫 OOT 실측 포함)
Phase 4  rshift (V18 sensitivity shift 이식) — L1 위(A4), L2 위(A3)

fold: train 2019~(y-1) → valid y,  y ∈ {2021, 2022, 2023, 2024}
전이: fit t → apply t+1,  t+1 ∈ {2022, 2023, 2024}
seed: CatBoost {42, 2026, 314} (paired) · L2 내부 adapter seed는 danker 사양 유지

arm (R 행만 보정, F 불변):
  A0 = L1(treatment)
  A1 = L1 + L2_single(fold t 잔차)
  A2 = L1 + L2_multi(fold ≤t 잔차 stack)
  A4 = L1 + shift(fold t L1 잔차)
  A3 = A1 + shift(fold t A1-chain 잔차)   # t+1 ∈ {2023, 2024}만 (chain 필요)

Gate: PRIMARY ΔR_shape · |ΔR_level| ≤ 5 · 판정 불가 |mean| < 2·std/√n
규칙: 모든 상수 fold-train freeze, test.csv 미사용, 행 독립 유지.

실행: 실험 루트에서  python rfold_v001/run_phases.py   (예상 1~2시간)
"""

from __future__ import annotations

import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

# =========================================================
# 경로
# =========================================================
EXPERIMENT_NAME = "rfold_v001"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"            # header only

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

DANKER_LIB_CANDIDATES = [
    Path("./model/danker_v001/train.py"),    # 배포 동결 사본 (우선)
    Path("./danker_v001/train.py"),
]
L3_LIB_CANDIDATES = [
    Path("./l3logit_v001/l3_link_validation.py"),
    Path(__file__).resolve().parent.parent / "l3logit_v001" / "l3_link_validation.py",
]

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

FOLDS = [2021, 2022, 2023, 2024]
TRANSITIONS = [2022, 2023, 2024]
SEEDS = [42, 2026, 314]

SHIFT_ALPHA = 1000.0        # V18 Phase 5 사양 동결
SHIFT_SCALE = 0.75
GATE_CFG = {"r_level_abs_max": 5.0, "inconclusive_k": 2.0}


def load_module(candidates, name):
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location(name, cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, cand
    raise FileNotFoundError(f"{name} 없음: {candidates}")


def seg_components(y_all, p, mask, var_tot, n_tot):
    ys, ps = y_all[mask], p[mask]
    w = mask.sum() / n_tot
    m = float(np.mean(ys - ps))
    b = float(np.mean((ys - ps) ** 2))
    b_base = float(np.mean((ys - np.mean(y_all)) ** 2))
    seg_pt = 100000.0 * w * (b_base - b) / var_tot
    level_debt = -100000.0 * w * (m ** 2) / var_tot
    return {"seg_pt": seg_pt, "level_debt_pt": level_debt,
            "shape_pt": seg_pt - level_debt, "mean_residual": m}


def r_delta(y, p_new, p_base_arm, r_mask, var, n):
    a = seg_components(y, p_new, r_mask, var, n)
    b = seg_components(y, p_base_arm, r_mask, var, n)
    return {"d_R_total": a["seg_pt"] - b["seg_pt"],
            "d_R_level": a["level_debt_pt"] - b["level_debt_pt"],
            "d_R_shape": a["shape_pt"] - b["shape_pt"]}


def main():
    t0 = time.time()
    if VALIDATION_DIR.exists():
        raise FileExistsError(f"기존 validation 보호: {VALIDATION_DIR}")

    lib, lib_path = load_module(DANKER_LIB_CANDIDATES, "danker_lib")
    l3, l3_path = load_module(L3_LIB_CANDIDATES, "l3_lib")
    print(f"danker lib: {lib_path}\nl3 lib    : {l3_path}")

    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns.tolist()
    base_features = [c for c in test_cols if c != ID]

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    train[SEASON_COL] = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)

    # 행별 사용은 '해당 행 season 미만'으로 함수 내부 제한 → 전체 train 통계 1회 구축 안전
    full_stats = lib.build_pitcher_season_stats(train)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)

    # =========================================================
    # Phase 1 — treatment walk-forward (+ fold2024 control/dynamic)
    # =========================================================
    PRED = {}          # PRED[y][seed] = treatment 예측 (fold_val 전체 행)
    RAW = {}           # RAW[y] = fold_val 원본 rows
    Y, RMASK = {}, {}
    iter_rows = []

    for y in FOLDS:
        t_fold = time.time()
        fold_train = train.loc[train[SEASON_COL] < y].reset_index(drop=True)
        fold_val = train.loc[train[SEASON_COL] == y].reset_index(drop=True)
        views = lib.make_l1_views(fold_train, fold_val, base_features, scope=f"rfold{y}")

        y_va = fold_val[TARGET].to_numpy(np.float64)
        gt = fold_val[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
        r_mask = gt != "F"
        r_idx = np.flatnonzero(r_mask)

        X_tr = lib.prepare_features(
            views["train_treatment"][views["treatment_features"]], views["treatment_cat_cols"])
        X_va = lib.prepare_features(
            views["apply_treatment"][views["treatment_features"]], views["treatment_cat_cols"])
        y_tr = fold_train[TARGET].to_numpy(np.float64)

        PRED[y], RAW[y], Y[y], RMASK[y] = {}, fold_val, y_va, r_mask

        for seed in SEEDS:
            params = {**lib.MODEL_PARAMS, "random_seed": int(seed)}
            m = CatBoostRegressor(**params)
            m.fit(X_tr, y_tr,
                  eval_set=(X_va.iloc[r_idx], y_va[r_idx]),          # esfix
                  cat_features=views["treatment_cat_cols"],
                  early_stopping_rounds=lib.EARLY_STOPPING_ROUNDS,
                  use_best_model=True, verbose=False)
            PRED[y][seed] = np.clip(np.asarray(m.predict(X_va), np.float64), 0, 1)
            iter_rows.append({"fold": y, "model": "treatment", "seed": seed,
                              "best_iter": int(m.get_best_iteration())})
            print(f"[P1 fold {y} treatment seed {seed}] iter={iter_rows[-1]['best_iter']} "
                  f"({time.time()-t_fold:.0f}s)")
            del m

        # fold 2024: control/dynamic (v002 median 용)
        if y == 2024:
            Xc_tr = lib.prepare_features(
                views["train_control"][views["control_features"]], views["control_cat_cols"])
            Xc_va = lib.prepare_features(
                views["apply_control"][views["control_features"]], views["control_cat_cols"])
            means = lib.build_season_league_means(fold_train)
            prev_tr = lib.compute_prev_anchor(fold_train, views["pitcher_season_stats"], means)
            prev_va = lib.compute_prev_anchor(fold_val, views["pitcher_season_stats"], means)
            dyn_tr, _ = lib.compute_dynamic_anchor(
                exact_view=views["train_control"], prev_anchor=prev_tr)
            dyn_va, _ = lib.compute_dynamic_anchor(
                exact_view=views["apply_control"], prev_anchor=prev_va)
            for seed in SEEDS:
                params = {**lib.MODEL_PARAMS, "random_seed": int(seed)}
                mc = CatBoostRegressor(**params)
                mc.fit(Xc_tr, y_tr, eval_set=(Xc_va.iloc[r_idx], y_va[r_idx]),
                       cat_features=views["control_cat_cols"],
                       early_stopping_rounds=lib.EARLY_STOPPING_ROUNDS,
                       use_best_model=True, verbose=False)
                iter_rows.append({"fold": y, "model": "control", "seed": seed,
                                  "best_iter": int(mc.get_best_iteration())})
                del mc
                md = CatBoostRegressor(**params)
                md.fit(Xc_tr, y_tr - dyn_tr,
                       eval_set=(Xc_va.iloc[r_idx], (y_va - dyn_va)[r_idx]),
                       cat_features=views["control_cat_cols"],
                       early_stopping_rounds=lib.EARLY_STOPPING_ROUNDS,
                       use_best_model=True, verbose=False)
                iter_rows.append({"fold": y, "model": "dynamic", "seed": seed,
                                  "best_iter": int(md.get_best_iteration())})
                del md
            print(f"[P1 fold 2024 control/dynamic done]")

        np.savez_compressed(
            VALIDATION_DIR / f"l1_fold_{y}.npz",
            y_true=y_va, game_type=gt,
            **{f"treatment_seed{s}": PRED[y][s] for s in SEEDS})
        del views, X_tr, X_va
        gc.collect()

    # =========================================================
    # Phase 2 — v002 상수
    # =========================================================
    iter_df = pd.DataFrame(iter_rows)
    iter_df.to_csv(VALIDATION_DIR / "iter_table.csv", index=False)
    med = {}
    for (y, mdl), g in iter_df.groupby(["fold", "model"]):
        med[f"{mdl}_{y}"] = int(np.median(g["best_iter"])) + 1
    (VALIDATION_DIR / "t_final_median.json").write_text(
        json.dumps({"tree_count_median_plus1": med,
                    "note": "v002: T_final = seed median (+1, zero-based 보정)"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("[P2] t_final_median.json 저장:", med)

    # =========================================================
    # Phase 3 — L2 single vs multi (전이별 paired)
    # =========================================================
    def build_l2_X(prep, rows, base_pred):
        core = lib.transform_adapter_core(prep, rows, base_pred)
        exact = lib.exact5_matrix(rows, full_stats)
        X = pd.concat([core, exact], axis=1)
        assert not X.columns.duplicated().any()
        return X

    def fit_l2(X_fit, resid_fit, r_fit):
        models = []
        for aseed in lib.ADAPTER_SEEDS:
            et = ExtraTreesRegressor(n_jobs=lib.N_JOBS,
                                     random_state=16200 + int(aseed),
                                     **lib.FINAL_ADAPTER_CFG)
            et.fit(X_fit.loc[r_fit], resid_fit[r_fit])
            models.append(et)
        return models

    def apply_l2(models, X, base, r_mask):
        corrs = []
        for et in models:
            et.n_jobs = 1                      # 행 독립 결정론
            corrs.append(np.asarray(et.predict(X), np.float64))
        corr = np.mean(np.column_stack(corrs), axis=1)
        out = base.copy()
        out[r_mask] = np.clip(base[r_mask] + lib.ADAPTER_SHRINK * corr[r_mask], 0, 1)
        return out

    l2_rows = []
    A1_STORE = {}     # A1_STORE[(t+1, seed)] — Phase 4 A3 chain 용
    for tgt in TRANSITIONS:
        src = tgt - 1
        for seed in SEEDS:
            base_s, base_t = PRED[src][seed], PRED[tgt][seed]
            yv, r_t = Y[tgt], RMASK[tgt]
            var = float(np.mean(yv) * (1 - np.mean(yv)))
            n = len(yv)

            # single
            prep_s = lib.fit_adapter_prep(RAW[src])
            X_s = build_l2_X(prep_s, RAW[src], base_s)
            X_t = build_l2_X(prep_s, RAW[tgt], base_t)
            m_single = fit_l2(X_s, Y[src] - base_s, RMASK[src])
            A1 = apply_l2(m_single, X_t, base_t, r_t)
            A1_STORE[(tgt, seed)] = A1

            # multi (fold 2021..src stack)
            stack_years = [f for f in FOLDS if f <= src]
            rows_m = pd.concat([RAW[f] for f in stack_years], ignore_index=True)
            prep_m = lib.fit_adapter_prep(rows_m)
            X_m = pd.concat(
                [build_l2_X(prep_m, RAW[f], PRED[f][seed]) for f in stack_years],
                ignore_index=True)
            resid_m = np.concatenate([Y[f] - PRED[f][seed] for f in stack_years])
            rmask_m = np.concatenate([RMASK[f] for f in stack_years])
            m_multi = fit_l2(X_m, resid_m, rmask_m)
            X_t_m = build_l2_X(prep_m, RAW[tgt], base_t)
            A2 = apply_l2(m_multi, X_t_m, base_t, r_t)

            d1 = r_delta(yv, A1, base_t, r_t, var, n)
            d2 = r_delta(yv, A2, base_t, r_t, var, n)
            d21 = r_delta(yv, A2, A1, r_t, var, n)
            l2_rows.append({"transition": tgt, "seed": seed,
                            "stack_years": len(stack_years),
                            **{f"A1_{k}": v for k, v in d1.items()},
                            **{f"A2_{k}": v for k, v in d2.items()},
                            **{f"A2vsA1_{k}": v for k, v in d21.items()}})
            print(f"[P3 {src}->{tgt} seed {seed}] "
                  f"A1 dR={d1['d_R_total']:+.1f} | A2 dR={d2['d_R_total']:+.1f} | "
                  f"A2-A1={d21['d_R_total']:+.1f}")
            del X_s, X_t, X_m, X_t_m, m_single, m_multi
            gc.collect()

    l2_df = pd.DataFrame(l2_rows)
    l2_df.to_csv(VALIDATION_DIR / "l2_transition_summary.csv", index=False)

    # =========================================================
    # Phase 4 — rshift (A4: L1 위 / A3: A1 위)
    # =========================================================
    SENS = {y: l3.transform_sensitivity_specs(
        l3.build_sensitivity_specs(train, y), RAW[y]).to_numpy(np.float32)
        for y in FOLDS}

    def fit_apply_shift(Zs, Zt, resid_src, r_src, base_tgt, r_tgt):
        A = Zs[r_src]
        mu, sd = A.mean(0), A.std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        Az = np.clip((A - mu) / sd, -8, 8).astype(np.float64)
        Bz = np.clip((Zt[r_tgt] - mu) / sd, -8, 8).astype(np.float64)
        beta = l3.ridge_no_intercept(Az, resid_src[r_src], SHIFT_ALPHA)
        out = base_tgt.copy()
        out[r_tgt] = np.clip(out[r_tgt] + SHIFT_SCALE * (Bz @ beta), 0, 1)
        return out

    sh_rows = []
    for tgt in TRANSITIONS:
        src = tgt - 1
        for seed in SEEDS:
            yv, r_t = Y[tgt], RMASK[tgt]
            var = float(np.mean(yv) * (1 - np.mean(yv)))
            n = len(yv)

            # A4: L1 + shift
            A4 = fit_apply_shift(SENS[src], SENS[tgt],
                                 Y[src] - PRED[src][seed], RMASK[src],
                                 PRED[tgt][seed], r_t)
            d4 = r_delta(yv, A4, PRED[tgt][seed], r_t, var, n)
            row = {"transition": tgt, "seed": seed,
                   **{f"A4_{k}": v for k, v in d4.items()}}

            # A3: A1 + shift (chain: src 의 A1 필요 → src ∈ TRANSITIONS 인 경우만)
            if (src, seed) in A1_STORE:
                A1_src, A1_tgt = A1_STORE[(src, seed)], A1_STORE[(tgt, seed)]
                A3 = fit_apply_shift(SENS[src], SENS[tgt],
                                     Y[src] - A1_src, RMASK[src], A1_tgt, r_t)
                d3 = r_delta(yv, A3, A1_tgt, r_t, var, n)
                row.update({f"A3_{k}": v for k, v in d3.items()})
            sh_rows.append(row)
            print(f"[P4 {src}->{tgt} seed {seed}] A4 dR={d4['d_R_total']:+.1f}"
                  + (f" | A3 dR={row.get('A3_d_R_total', float('nan')):+.1f}"
                     if f"A3_d_R_total" in row else ""))

    sh_df = pd.DataFrame(sh_rows)
    sh_df.to_csv(VALIDATION_DIR / "shift_transition_summary.csv", index=False)

    # =========================================================
    # Gate
    # =========================================================
    def gate_block(df, col_total, col_level):
        sub = df.dropna(subset=[col_total])
        mean_d = float(sub[col_total].mean())
        std_d = float(sub[col_total].std(ddof=1))
        nn = len(sub)
        wins = int((sub[col_total] > 0).sum())
        lvl = float(sub[col_level].mean())
        inconclusive = abs(mean_d) < GATE_CFG["inconclusive_k"] * std_d / np.sqrt(max(nn, 1))
        verdict = ("INCONCLUSIVE" if inconclusive else
                   ("ADOPT_CANDIDATE" if mean_d > 0 and wins > nn * 0.75 else "REJECT"))
        if abs(lvl) > GATE_CFG["r_level_abs_max"]:
            verdict += " (LEVEL_GATE_FAIL)"
        return {"mean_d_R_total": mean_d, "std": std_d, "wins": f"{wins}/{nn}",
                "mean_d_R_level": lvl, "verdict": verdict}

    gate = {
        "experiment": EXPERIMENT_NAME,
        "L2_single_vs_L1 (A1-A0)": gate_block(l2_df, "A1_d_R_total", "A1_d_R_level"),
        "L2_multi_vs_single (A2-A1)": gate_block(l2_df, "A2vsA1_d_R_total", "A2vsA1_d_R_level"),
        "shift_on_L1 (A4-A0)": gate_block(sh_df, "A4_d_R_total", "A4_d_R_level"),
        "shift_on_L2 (A3-A1)": gate_block(sh_df, "A3_d_R_total", "A3_d_R_level")
                                if "A3_d_R_total" in sh_df else "N/A",
        "gate_cfg": GATE_CFG,
        "decision_rule": "ADOPT_CANDIDATE 성분만 v002 에 반영. 최종 판정은 LB (vs 994.58).",
    }
    (VALIDATION_DIR / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    (VALIDATION_DIR / "metadata.json").write_text(json.dumps({
        "experiment": EXPERIMENT_NAME, "folds": FOLDS, "transitions": TRANSITIONS,
        "seeds": SEEDS, "esfix": "eval_set=R-only",
        "shift_spec": {"alpha": SHIFT_ALPHA, "scale": SHIFT_SCALE, "intercept": False,
                       "apply": "R rows only"},
        "l2_spec": {"cfg": lib.FINAL_ADAPTER_CFG, "shrink": lib.ADAPTER_SHRINK,
                    "adapter_seeds": lib.ADAPTER_SEEDS},
        "rule_compliance": {"labels": "official train only", "test_usage": "header only",
                            "constants": "fold-train frozen"},
        "elapsed_seconds": float(time.time() - t0),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    for k in ["L2_single_vs_L1 (A1-A0)", "L2_multi_vs_single (A2-A1)",
              "shift_on_L1 (A4-A0)", "shift_on_L2 (A3-A1)"]:
        print(f"{k}: {json.dumps(gate[k], ensure_ascii=False)}")
    print(f"elapsed {time.time()-t0:.0f}s | saved -> {VALIDATION_DIR}/")


if __name__ == "__main__":
    main()
"""danker_v003 — D0 드라이버 (4 fold x 3 seed)

설계 근거: 38_danker_v002_구조개편_설계_v001.md §7-1 D0, §8-2, §8-5

`model/danker_v003/train.py` 는 단일 자기완결 산출물이다. 이 드라이버는 그 파일의
함수만 조합해 expanding-window walk-forward 를 돌린다. train.py 의 main()(단일 fold
배포 경로)은 건드리지 않는다.

    fold 2021 : 2019~2020 -> 2021      큰 과거 level mismatch 에서 폭발하지 않는가
    fold 2022 : 2019~2021 -> 2022      정상 regime 에서 이득이 유지되는가
    fold 2023 : 2019~2022 -> 2023      regime break 에서 tail-risk defense 가 되는가
    fold 2024 : 2019~2023 -> 2024      PRIMARY

사용
    # 첫 fold 만 스모크 (가장 먼저 깨질 지점을 싸게 잡는다)
    python danker_v003/run_d0.py --tag smoke --folds 2021 --seeds 42

    # D0 기준선 — 재학습 12회
    python danker_v003/run_d0.py --tag baseline

    # 후보 실험: train.py 상단 플래그 하나만 바꾸고
    python danker_v003/run_d0.py --tag E1a \\
        --baseline validation/danker_v003/ls_metrics_baseline.csv --gate F

산출 (validation/danker_v003/)
    ls_metrics_<tag>.csv        fold x seed x branch, within/total 두 단위
    oof_<fold>_<seed>.csv       L1 성분별 OOF (E3c 가산항 원천)
    gate_<tag>.json             게이트 판정 (--baseline 있을 때)
    run_<tag>.json              실행 메타

행 독립성
    학습/검증 데이터만 다룬다. test.csv 는 컬럼 목록을 읽는 데만 쓴다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "./"
TRAIN_PATH = "./train.py"


def load_train():
    if not TRAIN_PATH.exists():
        raise SystemExit(f"{TRAIN_PATH} 가 없다")
    spec = importlib.util.spec_from_file_location("danker_v003_train", TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["danker_v003_train"] = mod
    spec.loader.exec_module(mod)          # main() 은 실행되지 않는다
    return mod


# =========================================================
def run_one_fold(T, train_df, base_features, source_year: int, seeds, out_dir: Path):
    """한 fold 를 seeds 만큼 돈다.

    feature view 와 dynamic anchor 는 seed 와 무관하므로 fold 당 1회만 만든다.
    (가장 비싼 부분이라 3배 절약된다)
    """
    fold_train = train_df.loc[train_df[T.SEASON_COL] < source_year].copy().reset_index(drop=True)
    fold_val = train_df.loc[train_df[T.SEASON_COL] == source_year].copy().reset_index(drop=True)
    if len(fold_train) == 0 or len(fold_val) == 0:
        raise RuntimeError(f"fold {source_year}: 학습/검증 행 없음")

    print(f"\n{'=' * 96}")
    print(f"FOLD {source_year}   train={len(fold_train):,} -> val={len(fold_val):,}")
    print("=" * 96)

    t0 = time.time()
    views = T.make_l1_views(
        df_train_raw=fold_train,
        df_apply_raw=fold_val,
        base_features=base_features,
        scope=f"fold{source_year}",
    )
    print(f"[views] {time.time() - t0:.1f}s  control={len(views['control_features'])} "
          f"treatment={len(views['treatment_features'])}")

    y_tr = fold_train[T.TARGET].to_numpy(dtype=np.float64)
    y_va = fold_val[T.TARGET].to_numpy(dtype=np.float64)
    gt = fold_val[T.GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    f_mask = gt == T.F_LABEL
    if not f_mask.any() or f_mask.all():
        raise RuntimeError(f"fold {source_year}: R/F 한쪽이 비어 있다")

    league = T.build_season_league_means(fold_train)
    prev_tr = T.compute_prev_anchor(
        rows=fold_train, pitcher_season_stats=views["pitcher_season_stats"],
        season_league_means=league,
    )
    prev_va = T.compute_prev_anchor(
        rows=fold_val, pitcher_season_stats=views["pitcher_season_stats"],
        season_league_means=league,
    )
    anchor_tr, _ = T.compute_dynamic_anchor(
        exact_view=views["train_control"], prev_anchor=prev_tr)
    anchor_va, diag_va = T.compute_dynamic_anchor(
        exact_view=views["apply_control"], prev_anchor=prev_va)

    rows, trees = [], {}
    for seed in seeds:
        s0 = time.time()
        _, p_ctrl, ctrl_best, _ = T.fit_fold_model(
            views["train_control"], views["apply_control"],
            views["control_features"], views["control_cat_cols"],
            y_tr, y_va, f"CONTROL[{source_year}/{seed}]", seed=seed,
        )
        _, p_treat, treat_best, _ = T.fit_fold_model(
            views["train_treatment"], views["apply_treatment"],
            views["treatment_features"], views["treatment_cat_cols"],
            y_tr, y_va, f"TREATMENT[{source_year}/{seed}]", seed=seed,
        )
        _, p_dyn, dyn_corr, dyn_best, _ = T.fit_dynamic_fold_model(
            train_view=views["train_control"], val_view=views["apply_control"],
            features=views["control_features"], cat_cols=views["control_cat_cols"],
            y_train=y_tr, y_val=y_va,
            anchor_train=anchor_tr, anchor_val=anchor_va,
            label=f"DYNAMIC[{source_year}/{seed}]", seed=seed,
        )

        # --- L1 합성: 마스크는 값에만 (I2) ---
        p_main = np.where(f_mask, p_ctrl, p_treat)
        w = T.blend_weights(gt, w_r=T.W_R, w_f=T.F_BLEND_WEIGHT)
        p_l1 = T.blend(p_main, p_dyn, w)

        if T.W_R == 0.0:                                    # I5 회귀
            f_blend = np.clip(
                (1.0 - T.F_BLEND_WEIGHT) * p_ctrl + T.F_BLEND_WEIGHT * p_dyn, 0.0, 1.0)
            ref = np.clip(np.where(~f_mask, p_treat, f_blend), 0.0, 1.0)
            if not np.array_equal(p_l1, ref):
                raise RuntimeError(
                    f"fold {source_year}/{seed}: W_R=0 인데 v002 라우팅과 값이 다르다 "
                    f"(max|diff|={float(np.abs(p_l1 - ref).max()):.3e})"
                )

        rows += T.fold_seed_rows(T.EXPERIMENT_NAME, source_year, seed, y_va, p_l1, gt)
        trees[str(seed)] = {
            "control": int(ctrl_best + 1),
            "treatment": int(treat_best + 1),
            "dynamic": int(dyn_best + 1),
            "seconds": round(time.time() - s0, 1),
        }

        pd.DataFrame({
            T.ID: fold_val[T.ID].to_numpy(),
            T.SEASON_COL: fold_val[T.SEASON_COL].to_numpy(),
            T.GAME_TYPE_COL: gt,
            "y_true": y_va,
            "CONTROL": p_ctrl,
            "TREATMENT": p_treat,
            "DYNAMIC_ANCHOR": anchor_va,
            "DYNAMIC": p_dyn,
            "DYNAMIC_CORR": dyn_corr,
            "L1": p_l1,
        }).to_csv(out_dir / f"oof_{source_year}_{seed}.csv", index=False, encoding="utf-8-sig")

        b, l, s = T.brier_parts(p_l1, y_va)
        print(f"  [{source_year}/{seed}] Brier={b:.10f}  Level={l:.3e}  Shape={s:.10f}  "
              f"BSS={T.bss_of(b, T.b0_of(y_va)):.4f}  ({trees[str(seed)]['seconds']}s)")

    return rows, {
        "trees": trees, "anchor_diag": diag_va,
        "n_train": int(len(fold_train)), "n_val": int(len(fold_val)),
        "n_R": int((~f_mask).sum()), "n_F": int(f_mask.sum()),
    }


# =========================================================
def main() -> int:
    T = load_train()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="baseline", help="산출물 접미사")
    ap.add_argument("--folds", type=int, nargs="+", default=list(T.FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(T.SEEDS))
    ap.add_argument("--baseline", type=Path, default=None,
                    help="paired 비교 대상 ls_metrics csv")
    ap.add_argument("--gate", choices=["R", "F", "RF", "none"], default="none")
    a = ap.parse_args()

    T.assert_invariants()
    out_dir = T.VALIDATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"train.py : {TRAIN_PATH}")
    print(f"flags    : W_R={T.W_R} W_F={T.F_BLEND_WEIGHT} L2_APPLY_F={T.L2_APPLY_F} "
          f"L2_TRAIN_ON_ALL={T.L2_TRAIN_ON_ALL} shrink={T.ADAPTER_SHRINK} "
          f"neutral={T.is_neutral()}")

    base_features = [
        c for c in pd.read_csv(T.TEST_PATH, encoding="utf-8-sig", nrows=0).columns
        if c != T.ID
    ]
    train_df = pd.read_csv(T.TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    train_df[T.SEASON_COL] = pd.to_numeric(train_df[T.SEASON_COL], errors="raise").astype(int)
    print(f"train    : {len(train_df):,} rows  seasons={sorted(train_df[T.SEASON_COL].unique())}")

    all_rows, fold_meta = [], {}
    t0 = time.time()
    for fold in a.folds:
        r, meta = run_one_fold(T, train_df, base_features, int(fold), a.seeds, out_dir)
        all_rows += r
        fold_meta[str(fold)] = meta

    # ---- ls_metrics ----
    base_rows = None
    if a.baseline is not None:
        if not a.baseline.exists():
            raise SystemExit(f"baseline 파일이 없다: {a.baseline}")
        base_rows = pd.read_csv(a.baseline)
    ls = T.build_ls_metrics(all_rows, base_rows=base_rows,
                            experiment=f"{T.EXPERIMENT_NAME}:{a.tag}")
    ls_path = out_dir / f"ls_metrics_{a.tag}.csv"
    T.save_ls_metrics(ls, ls_path)
    print(f"\nls_metrics -> {ls_path}")

    # ---- 게이트 ----
    if a.gate != "none":
        if base_rows is None:
            raise SystemExit("--gate 를 쓰려면 --baseline 이 필요하다 (paired delta 없이 판정 불가)")
        gt_types = {"R": ("R",), "F": ("F",), "RF": ("R", "F")}[a.gate]
        res = T.run_gates(ls, experiment=f"{T.EXPERIMENT_NAME}:{a.tag}", gate_types=gt_types)
        gpath = out_dir / f"gate_{a.tag}.json"
        T.save_gate(res, gpath)
        print("\n" + T.format_report(res))
        print(f"\ngate -> {gpath}")

    # ---- 실행 메타 ----
    (out_dir / f"run_{a.tag}.json").write_text(
        json.dumps({
            "experiment": T.EXPERIMENT_NAME,
            "tag": a.tag,
            "protocol": "D0_EXPANDING_WALKFORWARD_4FOLD_3SEED",
            "folds": list(a.folds),
            "seeds": list(a.seeds),
            "flags": T.freeze(),
            "fold_meta": fold_meta,
            "elapsed_sec": round(time.time() - t0, 1),
            "versions": {
                "numpy": np.__version__, "pandas": pd.__version__,
                "python": sys.version.split()[0],
            },
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"총 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

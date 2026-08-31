"""danker_v003 — D0 드라이버 (4 fold x 3 seed)

설계 근거: 38_danker_v002_구조개편_설계_v001.md §7-1 D0, §8-2, §8-5

`train.py` 는 단일 자기완결 산출물이다. 이 드라이버는 그 파일의 함수만 조합해
expanding-window walk-forward 를 돌린다. train.py 의 main()(단일 fold 배포 경로)은
건드리지 않는다.

    fold 2021 : 2019~2020 -> 2021      큰 과거 level mismatch 에서 폭발하지 않는가
    fold 2022 : 2019~2021 -> 2022      정상 regime 에서 이득이 유지되는가
    fold 2023 : 2019~2022 -> 2023      regime break 에서 tail-risk defense 가 되는가
    fold 2024 : 2019~2023 -> 2024      PRIMARY


경로 정책 — 전부 절대경로
------------------------
train.py 는 제출용이라 상대경로(`./data`, `./model`, `./validation`)를 쓴다. 그래서
cwd 가 프로젝트 루트가 아니면 조용히 엉뚱한 위치를 읽고 쓴다. 이 드라이버는

    1. 프로젝트 루트를 **절대경로로 확정**하고 (자기 위치에서 위로 올라가며 탐색)
    2. 그 위치로 **chdir 한 뒤** train.py 를 로드한다

따라서 어디서 실행해도 결과가 같다. 확정된 절대경로는 시작 시 전부 출력된다.

    프로젝트 루트 판정 기준 : data/train.csv · data/test.csv · model/ · validation/
    우선순위               : --root  >  $AIMERS_ROOT  >  자기 위치에서 상향 탐색
    train.py               : --train  >  이 파일과 같은 디렉터리의 train.py


기준선을 같은 학습 결과에서 파생한다 (`--baseline-wr`)
----------------------------------------------------
W_R 은 **학습 후 합성 가중치**다. CONTROL / TREATMENT / DYNAMIC 세 예측은 W_R 과
무관하게 동일하므로, 한 번 학습한 결과에서 임의의 W_R 에 대한 L1 을 계산할 수 있다.

    L1(w) = clip( (1-w)*where(F, CONTROL, TREATMENT) + w*DYNAMIC , 0, 1 )

따라서 `--baseline-wr 0.0` 을 주면 별도 12회 재학습 없이 기준선을 같은 패스에서
만든다. 시드·데이터·early stopping 이 완전히 같으므로 paired 비교가 정확해진다.

    ⚠️ 이 방식은 W_R 만 다른 후보에서 유효하다. 다른 축(E1/E3/E4)을 바꾼 실험은
       모델 자체가 달라지므로 반드시 별도 기준선 파일과 비교해야 한다.


사용
    python ~/AIMERS2026/model/danker_v003/run_d0.py --tag smoke --folds 2021 --seeds 42

    # E2b — 기준선을 같은 패스에서 파생 (재학습 12회로 끝. 24회가 아니다)
    python ~/AIMERS2026/model/danker_v003/run_d0.py --tag E2b --baseline-wr 0.0 --gate R

    # 기존 기준선 파일과 비교 (다른 축 실험)
    python ~/AIMERS2026/model/danker_v003/run_d0.py --tag E1a \\
        --baseline validation/danker_v003_d0/ls_metrics_baseline.csv --gate R

    # 어디서 실행하든 동일 (경로는 내부에서 절대화된다)
    cd /tmp && python ~/AIMERS2026/model/danker_v003/run_d0.py --tag E2b --baseline-wr 0.0 --gate R


산출 (기본 <루트>/validation/danker_v003/, `--out` 으로 변경)
    ls_metrics_<tag>.csv                fold x seed x branch, within/total 두 단위
    ls_metrics_<tag>.partial.csv        fold 완료마다 갱신되는 중간 저장본
    ls_metrics_<tag>_base_wr<W>.csv     --baseline-wr 로 파생한 기준선
    oof_<tag>_<fold>_<seed>.csv         L1 성분별 OOF (E3c 가산항 원천)
    gate_<tag>.json                     게이트 판정
    run_<tag>.json                      실행 메타 (flags + 해석된 절대경로 포함)

⚠️ train.py 는 `VALIDATION_DIR.mkdir(exist_ok=False)` 라 재실행하려면 그 디렉터리를
   지워야 한다. 기본 산출 위치를 train.py 와 공유하면 **train.py 재실행 한 번에
   D0 산출물이 전부 사라진다.** `--out validation/danker_v003_d0` 를 권한다.

행 독립성
    학습/검증 데이터만 다룬다. test.csv 는 컬럼 목록을 읽는 데만 쓴다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# 프로젝트 루트라면 반드시 있어야 하는 것들.
ROOT_MARKERS = (
    Path("data") / "train.csv",
    Path("data") / "test.csv",
    Path("model"),
    Path("validation"),
)


# =========================================================
# 경로 확정
# =========================================================
def looks_like_root(p: Path) -> bool:
    return all((p / m).exists() for m in ROOT_MARKERS)


def resolve_project_root(cli_root: Path | None) -> Path:
    """프로젝트 루트를 절대경로로 확정한다.

    우선순위: --root  >  $AIMERS_ROOT  >  이 파일 위치에서 상향 탐색  >  cwd
    """
    candidates: list[tuple[str, Path]] = []
    if cli_root is not None:
        candidates.append(("--root", cli_root))
    env = os.environ.get("AIMERS_ROOT")
    if env:
        candidates.append(("$AIMERS_ROOT", Path(env)))

    for label, cand in candidates:
        cand = cand.expanduser().resolve()
        if not cand.is_dir():
            raise SystemExit(f"{label} 경로가 디렉터리가 아니다: {cand}")
        if not looks_like_root(cand):
            missing = [str(m) for m in ROOT_MARKERS if not (cand / m).exists()]
            raise SystemExit(
                f"{label} 이 프로젝트 루트로 보이지 않는다: {cand}\n"
                "  없는 항목: " + ", ".join(missing)
            )
        return cand

    for cand in (HERE, *HERE.parents):
        if looks_like_root(cand):
            return cand

    cwd = Path.cwd().resolve()
    if looks_like_root(cwd):
        return cwd

    raise SystemExit(
        "프로젝트 루트를 찾지 못했다.\n"
        f"  탐색 시작: {HERE}\n"
        f"  cwd      : {cwd}\n"
        "  필요 항목: " + ", ".join(str(m) for m in ROOT_MARKERS) + "\n"
        "  --root /절대/경로  또는  AIMERS_ROOT=/절대/경로  로 지정하라"
    )


def load_train(train_path: Path):
    """지정된 train.py 를 절대경로로 로드한다.

    드라이버와 train.py 는 항상 한 쌍으로 움직인다. 다른 train.py 를 읽으면
    W_R / VALIDATION_DIR 가 통째로 달라진 채 에러 없이 다른 실험이 돌아간다.
    """
    if not train_path.is_file():
        raise SystemExit(f"train.py 가 없다: {train_path}")
    spec = importlib.util.spec_from_file_location("danker_v003_train", train_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["danker_v003_train"] = mod
    spec.loader.exec_module(mod)          # main() 은 실행되지 않는다
    return mod


def abspaths(T, root: Path) -> dict:
    """train.py 의 상대경로 상수를 절대경로로 해석해 돌려준다 (표시·검증용)."""
    return {
        "root": root,
        "train_csv": (root / T.TRAIN_PATH).resolve(),
        "test_csv": (root / T.TEST_PATH).resolve(),
        "validation_dir": (root / T.VALIDATION_DIR).resolve(),
        "model_dir": (root / T.MODEL_DIR).resolve(),
    }


# =========================================================
# 정합성 가드
# =========================================================
def check_tag_flags(T, tag: str, force: bool) -> None:
    """tag 와 실제 구조 플래그가 어긋나지 않는지 확인한다.

    무엇이 기준선인지는 tag 가 아니라 train.py 상수가 정한다. 이름만 baseline
    인 파일이 생기면 이후 모든 paired delta 가 조용히 오염된다.
    """
    neutral = T.is_neutral()
    if tag == "baseline" and not neutral:
        msg = (
            f"--tag baseline 인데 train.py 가 E0 중립이 아니다 (W_R={T.W_R}, "
            f"L2_APPLY_F={T.L2_APPLY_F}, SHIFT_SCALE={T.SHIFT_SCALE}, "
            f"AFFINE_ALPHA={T.AFFINE_ALPHA}, L1_TARGET_MODE={T.L1_TARGET_MODE!r})"
        )
        if not force:
            raise SystemExit(msg + "\n의도한 것이면 --force")
        print(f"  경고: {msg}  (--force)")
    if tag != "baseline" and neutral:
        print(
            f"  경고: --tag {tag} 인데 train.py 가 E0 중립이다. "
            "기준선과 완전히 동일한 결과가 나오고 paired delta 는 전부 0 이 된다."
        )


def check_baseline_differs(T, baseline: Path, force: bool) -> None:
    """기준선이 현재와 다른 플래그로 만들어졌는지 확인한다.

    같은 플래그로 만든 기준선과 비교하면 delta 가 전부 0 이고, 게이트는
    그것을 '효과 없음'으로 판정한다 — 실험이 아니라 측정 실수다.
    """
    tag = baseline.stem.replace("ls_metrics_", "", 1)
    meta = baseline.with_name(f"run_{tag}.json")
    if not meta.exists():
        print(f"  참고: {meta.name} 이 없어 기준선 플래그를 확인하지 못했다")
        return
    try:
        base_flags = json.loads(meta.read_text(encoding="utf-8")).get("flags")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  참고: {meta.name} 을 읽지 못했다 ({exc})")
        return
    if base_flags == T.freeze():
        msg = f"기준선({meta.name})과 현재 플래그가 동일하다 — paired delta 가 전부 0 이 된다"
        if not force:
            raise SystemExit(msg + "\n의도한 것이면 --force")
        print(f"  경고: {msg}  (--force)")


# =========================================================
def run_one_fold(T, train_df, base_features, source_year: int, seeds, out_dir: Path,
                 tag: str, overwrite: bool, baseline_wr: float | None = None):
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

    rows, base_rows_derived, trees = [], [], {}
    for seed in seeds:
        oof_path = out_dir / f"oof_{tag}_{source_year}_{seed}.csv"
        if oof_path.exists() and not overwrite:
            raise SystemExit(
                f"{oof_path} 가 이미 있다. 덮어쓰려면 --overwrite, "
                "아니면 --tag 를 바꿔라 (기존 OOF 는 후속 실험의 원천이다)"
            )

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

        # --- 같은 학습 결과에서 기준선 L1 을 파생한다 (재학습 없음) ---
        if baseline_wr is not None:
            w_b = T.blend_weights(gt, w_r=float(baseline_wr), w_f=T.F_BLEND_WEIGHT)
            p_base = T.blend(p_main, p_dyn, w_b)
            base_rows_derived += T.fold_seed_rows(
                f"{T.EXPERIMENT_NAME}:base_wr{baseline_wr:g}",
                source_year, seed, y_va, p_base, gt,
            )
            bb, _, _ = T.brier_parts(p_base, y_va)
            print(f"    파생기준선 W_R={baseline_wr:g}  Brier={bb:.10f}  "
                  f"BSS={T.bss_of(bb, T.b0_of(y_va)):.4f}")

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
        }).to_csv(oof_path, index=False, encoding="utf-8-sig")

        b, l, s = T.brier_parts(p_l1, y_va)
        print(f"  [{source_year}/{seed}] Brier={b:.10f}  Level={l:.3e}  Shape={s:.10f}  "
              f"BSS={T.bss_of(b, T.b0_of(y_va)):.4f}  ({trees[str(seed)]['seconds']}s)")

    return rows, base_rows_derived, {
        "trees": trees, "anchor_diag": diag_va,
        "n_train": int(len(fold_train)), "n_val": int(len(fold_val)),
        "n_R": int((~f_mask).sum()), "n_F": int(f_mask.sum()),
    }


# =========================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="baseline", help="산출물 접미사")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="paired 비교 대상 ls_metrics csv (루트 기준 상대 또는 절대)")
    ap.add_argument("--baseline-wr", type=float, default=None,
                    help="기준선을 같은 학습 결과에서 이 W_R 로 파생한다 (보통 0.0). "
                         "W_R 만 다른 실험에서만 유효")
    ap.add_argument("--gate", choices=["R", "F", "RF", "none"], default="none")
    ap.add_argument("--root", type=Path, default=None,
                    help="프로젝트 루트 절대경로 (미지정 시 자동 탐색)")
    ap.add_argument("--out", type=Path, default=None,
                    help="산출 디렉터리 (기본 train.py 의 VALIDATION_DIR). "
                         "train.py 재실행이 D0 산출물을 지우지 않게 하려면 분리 권장")
    ap.add_argument("--train", type=Path, default=None,
                    help="사용할 train.py 절대경로 (미지정 시 이 파일과 같은 디렉터리)")
    ap.add_argument("--overwrite", action="store_true",
                    help="같은 tag 의 기존 OOF 를 덮어쓴다")
    ap.add_argument("--force", action="store_true",
                    help="tag/flag 정합성 경고를 무시하고 진행한다")
    a = ap.parse_args()

    # ---- 경로 확정. train.py 로드 전에 chdir 한다 (상대경로 상수가 여기서 굳는다) ----
    root = resolve_project_root(a.root)
    train_py = (a.train.expanduser().resolve() if a.train is not None else HERE / "train.py")
    os.chdir(root)

    T = load_train(train_py)

    folds = list(a.folds) if a.folds else list(T.FOLDS)
    seeds = list(a.seeds) if a.seeds else list(T.SEEDS)

    # train.py 의 게이트 계산(_pull)은 모듈 상수 SEEDS 를 하드 참조한다.
    # --seeds 로 다른 시드를 쓰면 ls_metrics 에는 새 시드가 들어가는데
    # _pull 은 여전히 (42, 2026, 314) 를 찾아 KeyError 로 죽는다.
    # 이 프로세스 안에서만 상수를 실제 실행값으로 맞춘다.
    # (main() 은 호출하지 않으므로 배포 경로에는 영향이 없다)
    if tuple(seeds) != tuple(T.SEEDS):
        print(f"  SEEDS 재정의: {tuple(T.SEEDS)} -> {tuple(seeds)}  (게이트 계산용)")
        T.SEEDS = tuple(seeds)

    T.assert_invariants()          # len(SEEDS) >= 2 검사가 실제 시드로 걸린다
    check_tag_flags(T, a.tag, a.force)

    if a.baseline is not None and a.baseline_wr is not None:
        raise SystemExit("--baseline 과 --baseline-wr 은 함께 쓸 수 없다")
    if a.baseline_wr is not None and not (0.0 <= a.baseline_wr <= 1.0):
        raise SystemExit(f"--baseline-wr 은 [0,1] 이어야 한다: {a.baseline_wr}")
    if a.baseline_wr is not None and float(a.baseline_wr) == float(T.W_R):
        raise SystemExit(
            f"--baseline-wr {a.baseline_wr:g} 이 현재 W_R 과 같다 — delta 가 전부 0 이 된다"
        )

    P = abspaths(T, root)
    if a.out is not None:
        out_dir = a.out.expanduser()
        out_dir = (out_dir if out_dir.is_absolute() else root / out_dir).resolve()
    else:
        out_dir = P["validation_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"루트     : {root}")
    print(f"train.py : {train_py}")
    print(f"train.csv: {P['train_csv']}")
    print(f"test.csv : {P['test_csv']}")
    print(f"산출     : {out_dir}")
    print(f"flags    : W_R={T.W_R} W_F={T.F_BLEND_WEIGHT} L2_APPLY_F={T.L2_APPLY_F} "
          f"L2_TRAIN_ON_ALL={T.L2_TRAIN_ON_ALL} shrink={T.ADAPTER_SHRINK} "
          f"neutral={T.is_neutral()}")
    print(f"folds    : {folds}   seeds: {seeds}   tag: {a.tag}")
    if a.baseline_wr is not None:
        print(f"기준선   : 같은 학습 결과에서 W_R={a.baseline_wr:g} 로 파생 (재학습 없음)")

    for label, p in (("train.csv", P["train_csv"]), ("test.csv", P["test_csv"])):
        if not p.is_file():
            raise SystemExit(f"{label} 이 없다: {p}")

    # train.py 의 VALIDATION_DIR.mkdir(exist_ok=False) 와 같은 곳을 쓰고 있는가?
    if out_dir == P["validation_dir"] and (out_dir / "fold4_metadata.json").exists():
        print(
            "\n  ⚠️  산출 위치가 train.py 의 VALIDATION_DIR 과 같고 배포 산출물이 이미 있다.\n"
            "      train.py 는 mkdir(exist_ok=False) 이므로 재실행하려면 이 디렉터리를\n"
            "      지워야 하고, 그때 D0 산출물(ls_metrics / oof / gate)도 함께 사라진다.\n"
            f"      권장: --out {out_dir.name}_d0\n"
        )

    # ---- 기준선 ----
    base_rows, baseline_abs = None, None
    if a.baseline is not None:
        baseline_abs = a.baseline.expanduser()
        if not baseline_abs.is_absolute():
            baseline_abs = (root / baseline_abs)
        baseline_abs = baseline_abs.resolve()
        if not baseline_abs.is_file():
            raise SystemExit(f"baseline 파일이 없다: {baseline_abs}")
        print(f"baseline : {baseline_abs}")
        check_baseline_differs(T, baseline_abs, a.force)
        base_rows = pd.read_csv(baseline_abs)
    if a.gate != "none" and base_rows is None and a.baseline_wr is None:
        raise SystemExit(
            "--gate 를 쓰려면 --baseline 또는 --baseline-wr 이 필요하다 "
            "(paired delta 없이 판정 불가)"
        )

    # ---- 데이터 ----
    base_features = [
        c for c in pd.read_csv(P["test_csv"], encoding="utf-8-sig", nrows=0).columns
        if c != T.ID
    ]
    train_df = pd.read_csv(P["train_csv"], encoding="utf-8-sig", low_memory=False)
    train_df[T.SEASON_COL] = pd.to_numeric(train_df[T.SEASON_COL], errors="raise").astype(int)
    print(f"train    : {len(train_df):,} rows  seasons={sorted(train_df[T.SEASON_COL].unique())}")

    ls_path = out_dir / f"ls_metrics_{a.tag}.csv"
    partial_path = out_dir / f"ls_metrics_{a.tag}.partial.csv"

    all_rows, derived_rows, fold_meta = [], [], {}
    t0 = time.time()
    for fold in folds:
        r, rb, meta = run_one_fold(
            T, train_df, base_features, int(fold), seeds, out_dir,
            tag=a.tag, overwrite=a.overwrite, baseline_wr=a.baseline_wr,
        )
        all_rows += r
        derived_rows += rb
        fold_meta[str(fold)] = meta

        # fold 완료마다 중간 저장 — 뒤쪽 fold 에서 깨져도 앞선 결과는 남는다.
        try:
            T.save_ls_metrics(
                T.build_ls_metrics(all_rows,
                                   base_rows=(derived_rows if derived_rows else base_rows),
                                   experiment=f"{T.EXPERIMENT_NAME}:{a.tag}"),
                partial_path,
            )
        except Exception as exc:                              # noqa: BLE001
            print(f"  경고: 중간 저장 실패 ({exc})")

    # ---- 파생 기준선 저장 ----
    if a.baseline_wr is not None:
        base_rows = derived_rows
        derived_path = out_dir / f"ls_metrics_{a.tag}_base_wr{a.baseline_wr:g}.csv"
        T.save_ls_metrics(
            T.build_ls_metrics(derived_rows, base_rows=None,
                               experiment=f"{T.EXPERIMENT_NAME}:base_wr{a.baseline_wr:g}"),
            derived_path,
        )
        print(f"\n파생 기준선 -> {derived_path}")

    # ---- ls_metrics ----
    ls = T.build_ls_metrics(all_rows, base_rows=base_rows,
                            experiment=f"{T.EXPERIMENT_NAME}:{a.tag}")
    T.save_ls_metrics(ls, ls_path)
    print(f"ls_metrics -> {ls_path}")
    if partial_path.exists():
        partial_path.unlink()

    # ---- 게이트 ----
    if a.gate != "none":
        gt_types = {"R": ("R",), "F": ("F",), "RF": ("R", "F")}[a.gate]
        try:
            res = T.run_gates(ls, experiment=f"{T.EXPERIMENT_NAME}:{a.tag}", gate_types=gt_types)
        except Exception as exc:                              # noqa: BLE001
            # 게이트가 깨져도 12회 재학습 결과는 이미 저장돼 있다.
            # 오프라인으로 다시 판정할 수 있도록 알려주고 종료 코드만 남긴다.
            print(f"\n게이트 계산 실패: {type(exc).__name__}: {exc}")
            print(f"ls_metrics 는 저장됐다 -> {ls_path}")
            print("게이트는 그 파일로 오프라인 재판정할 수 있다. 재학습은 불필요하다.")
            raise
        # run_gates 는 모듈 상수 SEEDS 를 기록한다. 실제 실행값으로 바로잡는다.
        res["seeds"] = seeds
        res["folds"] = folds
        res["baseline"] = (
            f"derived:W_R={a.baseline_wr:g}" if a.baseline_wr is not None
            else str(baseline_abs)
        )
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
            "folds": folds,
            "seeds": seeds,
            "paths": {
                "project_root": str(root),
                "train_py": str(train_py),
                "train_csv": str(P["train_csv"]),
                "test_csv": str(P["test_csv"]),
                "out_dir": str(out_dir),
                "baseline": str(baseline_abs) if baseline_abs else None,
            },
            "baseline_wr": a.baseline_wr,
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
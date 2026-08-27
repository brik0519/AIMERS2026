"""V18 Reference Ground Truth Dumper.

kaggle_kbo_v18_RECOVERED_FROM_V20_exact.ipynb 의 코드 셀을 **그대로 exec** 하여
intermediate output 을 dump 한다. train.py 재구현과의 numerical parity 를 재는
R1 / R2 / R3 gate 의 ground truth 다.

원본 코드를 수정하지 않는 것이 이 파일의 유일한 목적이다.
아래 3가지 패치만 적용하며, 어느 것도 계산값에 영향을 주지 않는다.

  P1) cell 1 의 `import lightgbm` / `from sklearn...` 세 줄을
      --stage features 에서만 제거한다 (Phase 2 는 numpy/pandas 만 필요).
      --stage anchor 에서는 제거하지 않는다.
  P1b) cell 1 의 `from IPython.display import display` 를 항상 제거하고
      no-op 스텁을 주입한다. 표시 전용이라 계산값에 영향이 없다.
  P2) cell 3 discover_csv 의 `roots=[...]` 를 실제 data 디렉터리로 바꾼다.
      파일 탐색 경로만 바뀌고 읽는 파일과 dtype 처리는 동일하다.
  P3) IPython `display` 를 no-op 스텁으로 대체한다.
  P4) --stage v15_oof 에서 cell 13 의 top-level 실행 루프를 제거하고
      함수 정의 블록만 **원본 텍스트 그대로** 잘라 exec 한다 (연도별 실행을 위해).

사용법:
    python3 v18_reference_dump.py --notebook <path.ipynb> --data-dir ./data \
        --out validation/v18_repro_v001/reference --year 2024 --stage features

    # 메모리가 빠듯하면 연도별로 프로세스를 나눈다
    for y in 2020 2021 2022 2023 2024; do
      python3 v18_reference_dump.py ... --year $y --stage features
    done
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

V18_CODE_CELLS = [1, 2, 3, 4, 5, 6, 7, 8]

# --stage v15_oof 는 Layer 1 crossfit 함수가 더 필요하다.
#   cell 12 : one_adapter_crossfit_cfg  (전부 def + print → 통째로 exec)
#   cell 13 : build_v15_matrix_for_year / v15_oof_year
#             (하단에 5개 연도 전체 루프가 있으므로 def 노드만 골라 exec = 패치 P4)
V15_OOF_EXTRA_CELLS = [12, 13]
DEFS_ONLY_CELLS = {13}

# --stage p18 : Layer 2 (PASS 2) + 승격 gate.
#   cell 14 · cell 15 는 top-level 코드라 함수로 호출할 수 없다. 대신
#   ROWS / Y / V15_OOF 를 미리 namespace 에 심어두고 **셀을 통째로 verbatim exec** 한다.
#   → 재현 충실도가 가장 높은 형태다 (재작성 0줄).
P18_EXTRA_CELLS = [14, 15]

# --stage deploy : Phase 6 (2024→2025 배포 artifact) + script.py 생성 + zip smoke.
#   cell 16 이 cell 1~8 외에 필요로 하는 이름은 AST 확인 결과 정확히 6개이고
#   그중 V18_SUBMIT_RECOMMENDED / summary 는 cell 15 가 만든다.
#   → ROWS / Y / V15_OOF / BASE_BY_YEAR 4개만 주입하면 14·15·16 을 통째로 verbatim exec 할 수 있다.
DEPLOY_EXTRA_CELLS = [14, 15, 16]

_LIB_IMPORT_LINES = (
    "import lightgbm as lgb",
    "from sklearn.linear_model import Ridge",
    "from sklearn.ensemble import ExtraTreesRegressor",
)
_IPY_IMPORT_LINE = "from IPython.display import display"
_ROOTS_LINE = "    roots=[Path('/kaggle/input'),Path('/content'),Path('.'),Path('/mnt/data')]"


def surrogate_base_pred(n: int) -> np.ndarray:
    """train.py 의 surrogate_base_pred 와 반드시 동일해야 한다."""
    return (((np.arange(n) % 997) / 997.0) * 0.98 + 0.01).astype(np.float64)


def load_cells(nb_path: Path, wanted: list[int]) -> dict[int, str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    out = {}
    for i in wanted:
        assert cells[i]["cell_type"] == "code", (i, cells[i]["cell_type"])
        out[i] = "".join(cells[i]["source"])
    return out


def defs_only(src: str) -> str:
    """top-level 함수 정의 블록만 **원본 텍스트 그대로** 잘라 남긴다 (패치 P4).

    ast.unparse 를 쓰지 않는다. lineno / end_lineno 로 원본 줄을 그대로 슬라이스하므로
    남는 코드는 notebook 과 문자 단위로 동일하다. 제거되는 것은 5개 연도 전체를 도는
    top-level 실행 루프뿐이다.
    """
    import ast

    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out.append("".join(lines[n.lineno - 1: n.end_lineno]))
    return "\n".join(out)


def patch_cell(idx: int, src: str, data_dir: Path, stage: str) -> tuple[str, list[str]]:
    patches = []
    if idx == 1:
        if _IPY_IMPORT_LINE in src:
            src = src.replace(_IPY_IMPORT_LINE + "\n", "")
            patches.append(f"P1b removed: {_IPY_IMPORT_LINE}")
        if stage == "features":
            for line in _LIB_IMPORT_LINES:
                if line in src:
                    src = src.replace(line + "\n", "")
                    patches.append(f"P1 removed: {line}")
    if idx == 3:
        assert _ROOTS_LINE in src, "cell 3 roots line not found — notebook changed?"
        src = src.replace(
            _ROOTS_LINE, f"    roots=[Path({str(data_dir.resolve())!r})]"
        )
        patches.append(f"P2 roots -> {data_dir.resolve()}")
    return src, patches


def build_namespace(nb_path: Path, data_dir: Path, stage: str) -> tuple[dict, list[str]]:
    wanted = list(V18_CODE_CELLS)
    if stage == "v15_oof":
        wanted += V15_OOF_EXTRA_CELLS
    cells = load_cells(nb_path, wanted)
    ns: dict = {"__name__": "__v18_reference__"}
    ns["display"] = lambda *a, **k: None  # P3
    all_patches = []
    for i in wanted:
        src, patches = patch_cell(i, cells[i], data_dir, stage)
        if i in DEFS_ONLY_CELLS:
            src = defs_only(src)
            patches.append("P4 top-level 실행 루프 제거, 함수 정의만 원본 텍스트로 exec")
        all_patches += [f"cell {i}: {p}" for p in patches]
        t0 = time.time()
        exec(compile(src, f"<v18_cell_{i}>", "exec"), ns)
        print(f"  [ref] cell {i} exec  {time.time() - t0:6.1f}s", flush=True)
        gc.collect()
    return ns, all_patches


def save_matrix(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=df.to_numpy(np.float32),
        columns=np.array(list(df.columns), dtype=object),
        dtypes=np.array([str(t) for t in df.dtypes], dtype=object),
    )


def dump_year(ns: dict, year: int, out_dir: Path, stage: str) -> dict:
    TARGET = ns["TARGET"]
    screen = ns["screen"]
    source_year = int(year) - 1
    rows = screen.loc[screen.season.eq(int(year))].reset_index(drop=True)
    n = len(rows)
    info: dict = {"year": int(year), "source_year": source_year, "n_rows": n}

    # ---- R1 data ----
    np.savez_compressed(
        out_dir / f"data_{year}.npz",
        row_id=rows.row_id.to_numpy().astype(str),
        target=rows[TARGET].to_numpy(np.float64),
        season=rows.season.to_numpy(np.int64),
        domain3=rows.domain3.to_numpy().astype(str),
        **{
            f"pitcher_half_{s}": ns["pitcher_half"](rows, s) for s in ns["CF_SEEDS"]
        },
    )
    info["domain3_counts"] = rows.domain3.value_counts().to_dict()
    info["target_mean"] = float(rows[TARGET].mean())

    # ---- self_state (34) ----
    self_state = ns["build_self_state"](rows.drop(columns=[TARGET], errors="ignore"))
    save_matrix(out_dir / f"self_state_{year}.npz", self_state)
    info["self_state_shape"] = list(self_state.shape)

    # ---- gseason (28)  via year_state ----
    sx, gx = ns["year_state"](rows, year)
    save_matrix(out_dir / f"gseason_{year}.npz", gx)
    info["gseason_shape"] = list(gx.shape)
    assert np.array_equal(
        sx.to_numpy(np.float32), self_state.to_numpy(np.float32), equal_nan=True
    ), "year_state self-state differs from direct build_self_state"
    del sx
    gc.collect()

    # ---- V15 blocks (EXACT_ASOF_STATE / PITCHER_CONDITIONAL + 미투입 3블록) ----
    blocks = ns["build_latest_history_blocks"](source_year, rows, year)
    for key, fname in [
        ("EXACT_ASOF_STATE", "exact_asof_state"),
        ("PITCHER_CONDITIONAL", "pitcher_conditional"),
        ("CROSS_DOMAIN_PROFILE", "cross_domain_profile"),
        ("BATTER_CONDITIONAL", "batter_conditional"),
        ("ROLE_DEPLOYMENT", "role_deployment"),
    ]:
        save_matrix(out_dir / f"{fname}_{year}.npz", blocks[key])
        info[fname + "_shape"] = list(blocks[key].shape)

    # ---- V18 STABLE (25) ----
    stable, stable_rows, _specs = ns["v18_stable_matrix"](year)
    assert len(stable_rows) == n
    save_matrix(out_dir / f"v18_stable_{year}.npz", stable)
    info["v18_stable_shape"] = list(stable.shape)

    # ---- adapter matrix (128), surrogate base_pred ----
    map_rows = screen.loc[screen.season.eq(source_year)].reset_index(drop=True)
    prep = ns["fit_adapter_prep"](map_rows)
    x0 = ns["transform_adapter"](prep, rows, surrogate_base_pred(n))
    X = pd.concat(
        [
            x0.reset_index(drop=True),
            blocks["PITCHER_CONDITIONAL"].reset_index(drop=True),
            blocks["EXACT_ASOF_STATE"].reset_index(drop=True),
        ],
        axis=1,
    )
    assert not X.columns.duplicated().any()
    save_matrix(out_dir / f"adapter_matrix_surrogate_{year}.npz", X)
    info["adapter_matrix_shape"] = list(X.shape)
    info["adapter_base_shape"] = list(x0.shape)
    info["adapter_prep_cols"] = list(prep["cols"])
    info["adapter_prep_cat_levels"] = {c: len(v) for c, v in prep["maps"].items()}
    del map_rows, x0, X, blocks, stable
    gc.collect()

    # ---- temporal scalars ----
    DOMAIN_MEANS = ns["DOMAIN_MEANS"]
    slope = float(ns["trailing_core_slope"](source_year))
    info["trend_slope"] = slope
    info["trend_prior"] = {
        d: float(DOMAIN_MEANS.loc[int(source_year), d] + slope)
        for d in ["R_CORE", "R_ANCHOR", "F"]
    }
    info["domain_means_source_year"] = {
        d: float(DOMAIN_MEANS.loc[int(source_year), d]) for d in ["R_CORE", "R_ANCHOR", "F"]
    }

    # ---- Phase 3 anchor ----
    if stage in ("anchor", "v15_oof"):
        p, parts, meta = ns["fit_recent_base"](source_year, rows, year, verbose=False)
        np.savez_compressed(
            out_dir / f"anchor_{year}.npz",
            p_base=np.asarray(p, np.float64),
            p_exact_lgb=np.asarray(parts["exact_lgb"], np.float64),
            p_exact_ridge=np.asarray(parts["exact_ridge"], np.float64),
            p_trend_lgb=np.asarray(parts["trend_center_lgb"], np.float64),
            residual_l1=rows[TARGET].to_numpy(np.float64) - np.asarray(p, np.float64),
        )
        info["base_bss"] = float(ns["raw_bss"](rows[TARGET].to_numpy(np.float64), p))
        info["p_base_mean"] = float(np.mean(p))
        info["anchor_meta"] = {
            "trend_slope": float(meta["trend_slope"]),
            "trend_prior": {k: float(v) for k, v in meta["trend_prior"].items()},
        }
        del parts
        gc.collect()

    # ---- Phase 4 : Layer 1 V15 OOF ----
    if stage == "v15_oof":
        y = rows[TARGET].to_numpy(np.float64)
        base = np.asarray(p, np.float64)
        # notebook cell 13 build_v15_matrix_for_year 는 전역 ROWS / BASE_BY_YEAR 를 읽는다.
        ns["ROWS"] = {int(year): rows}
        ns["BASE_BY_YEAR"] = {int(year): base}
        ns["Y"] = {int(year): y}
        X = ns["build_v15_matrix_for_year"](int(year))
        info["v15_matrix_shape"] = list(X.shape)
        info["v15_matrix_columns"] = list(X.columns)

        # cell 13 v15_oof_year 본문과 동일하다. 다만 seed 별 예측을 보존하기 위해
        # one_adapter_crossfit_cfg 를 직접 호출한다 (같은 함수, 같은 인자).
        preds, seed_rows, fold_rows = [], [], []
        for seed in ns["CF_SEEDS"]:
            t0 = time.time()
            pp, frows = ns["one_adapter_crossfit_cfg"](
                seed, X, y, base, rows, ns["V15_CF_CFG"]
            )
            preds.append(np.asarray(pp, np.float64))
            fold_rows.extend(frows)
            seed_rows.append(
                {
                    "year": int(year), "seed": int(seed),
                    "bss": float(ns["raw_bss"](y, pp)),
                    "gain_vs_base": float(ns["raw_bss"](y, pp) - ns["raw_bss"](y, base)),
                    "sec": round(time.time() - t0, 1),
                }
            )
            print(f"  [ref] {seed_rows[-1]}", flush=True)
            gc.collect()
        consensus = np.mean(np.column_stack(preds), axis=1)
        np.savez_compressed(
            out_dir / f"v15_oof_{year}.npz",
            v15_oof=consensus,
            p_base=base,
            target=y,
            **{f"p_seed{int(s)}": p_ for s, p_ in zip(ns["CF_SEEDS"], preds)},
        )
        info["v15_oof_bss"] = float(ns["raw_bss"](y, consensus))
        info["v15_oof_mean"] = float(consensus.mean())
        info["v15_seed_rows"] = seed_rows
        info["v15_fold_rows"] = fold_rows
        print(f"  [ref] V15 OOF CONSENSUS {year} {info['v15_oof_bss']:.2f}", flush=True)
        del X, preds, consensus
        gc.collect()

    if stage in ("anchor", "v15_oof"):
        del p
        gc.collect()

    del rows, self_state, gx
    gc.collect()
    return info


def _seed_pass1_globals(ns: dict, out_dir: Path, need_base: bool = False) -> tuple:
    """Phase 3/4 산출물(npz)에서 ROWS / Y / V15_OOF / BASE_BY_YEAR 를 복원해 주입한다."""
    TARGET = ns["TARGET"]
    screen = ns["screen"]
    years = [2020, 2021, 2022, 2023, 2024]
    ROWS, Y, V15_OOF, BASE = {}, {}, {}, {}
    for y in years:
        rows = screen.loc[screen.season.eq(int(y))].reset_index(drop=True)
        p = out_dir / f"v15_oof_{y}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"v15_oof npz 5개가 필요하다. 없음: {p}\n"
                f"  먼저 --stage v15_oof 로 2020~2024 를 모두 dump 할 것."
            )
        z = np.load(p)
        oof = z["v15_oof"].astype(np.float64)
        assert len(oof) == len(rows), (y, len(oof), len(rows))
        assert np.array_equal(z["target"].astype(np.float64),
                              rows[TARGET].to_numpy(np.float64)), f"target mismatch {y}"
        ROWS[y] = rows
        Y[y] = rows[TARGET].to_numpy(np.float64)
        V15_OOF[y] = oof
        BASE[y] = z["p_base"].astype(np.float64)
        del z
    ns["ROWS"] = ROWS
    ns["Y"] = Y
    ns["V15_OOF"] = V15_OOF
    if need_base:
        ns["BASE_BY_YEAR"] = BASE
    print("  [ref] ROWS / Y / V15_OOF"
          + (" / BASE_BY_YEAR" if need_base else "")
          + " 주입 완료 (2020~2024)", flush=True)
    return ROWS, Y, V15_OOF, BASE


def dump_deploy(ns: dict, out_dir: Path, nb_path: Path, data_dir: Path) -> dict:
    """Phase 6 — cell 14·15·16 을 통째로 verbatim exec 한다.

    cell 16 은 배포 artifact 학습 + script.py 생성 + zip + subprocess smoke 까지 수행하므로
    **Phase 7 의 ground truth 도 함께 만들어진다.**
    """
    ROWS, Y, V15_OOF, BASE = _seed_pass1_globals(ns, out_dir, need_base=True)

    cells = load_cells(nb_path, DEPLOY_EXTRA_CELLS)
    for i in DEPLOY_EXTRA_CELLS:
        src, _ = patch_cell(i, cells[i], data_dir, "deploy")
        t0 = time.time()
        exec(compile(src, f"<v18_cell_{i}>", "exec"), ns)
        print(f"  [ref] cell {i} exec  {time.time() - t0:8.1f}s", flush=True)
        gc.collect()

    test = ns["test"]
    np.savez_compressed(
        out_dir / "deploy_2025.npz",
        row_id=test.row_id.to_numpy().astype(str),
        domain3=test.domain3.to_numpy().astype(str),
        p_base25=np.asarray(ns["p_base25"], np.float64),
        p1=np.asarray(ns["p1"], np.float64),
        p2=np.asarray(ns["p2"], np.float64),
        p3=np.asarray(ns["p3"], np.float64),
        correction=np.asarray(ns["correction"], np.float64),
        final_v15_direct=np.asarray(ns["final_v15_direct"], np.float64),
        stable_corr25=np.asarray(ns["stable_corr25"], np.float64),
        final_direct=np.asarray(ns["final_direct"], np.float64),
    )
    pre = ns["preprocess"]
    info = {
        "expected_year": int(ns["expected_year"]),
        "n_test_rows": int(len(test)),
        "v18_stable_enabled": bool(pre.get("v18_stable_enabled")),
        "v18_submit_recommended": bool(ns["V18_SUBMIT_RECOMMENDED"]),
        "protocol": pre.get("protocol"),
        "selected_blocks": list(pre.get("selected_blocks", [])),
        "exact_feature_cols_n": len(pre["exact_feature_cols"]),
        "ridge_feature_cols_n": len(pre["ridge_feature_cols"]),
        "trend_feature_cols_n": len(pre["trend_feature_cols"]),
        "adapter_feature_cols_n": len(pre["adapter_feature_cols"]),
        "adapter_base_feature_cols_n": len(pre["adapter_base_feature_cols"]),
        "pitcher_conditional_cols_n": len(pre["pitcher_conditional_cols"]),
        "exact_adapter_cols_n": len(pre["exact_adapter_cols"]),
        "v18_stable_cols_n": len(pre["v18_stable_cols"]),
        "trend_prior": {k: float(v) for k, v in pre["trend_prior"].items()},
        "adapter_seeds": list(pre["adapter_seeds"]),
        "adapter_shrink": float(pre["adapter_shrink"]),
        "v18_stable_gamma": float(pre["v18_stable_gamma"]),
        "v18_route_domain": pre["v18_route_domain"],
        "final_direct": {"mean": float(np.mean(ns["final_direct"])),
                         "min": float(np.min(ns["final_direct"])),
                         "max": float(np.max(ns["final_direct"]))},
        "work_root": str(ns["WORK_ROOT"]),
        "zip_path": str(ns.get("zip_path", "")),
    }
    # cell 16 이 만든 script.py 를 Phase 7 ground truth 로 보존한다
    src_script = Path(ns["WORK_ROOT"]) / "script.py"
    if src_script.exists():
        (out_dir / "script_reference.py").write_text(
            src_script.read_text(encoding="utf-8"), encoding="utf-8"
        )
        info["script_reference"] = str(out_dir / "script_reference.py")
    print(f"  [ref] deploy 완료 · v18_stable_enabled={info['v18_stable_enabled']} "
          f"· test rows={info['n_test_rows']}", flush=True)
    return info


def dump_p18(ns: dict, out_dir: Path, nb_path: Path, data_dir: Path) -> dict:
    """Layer 2 (PASS 2). cell 14 / cell 15 를 통째로 verbatim exec 한다.

    선행 조건: 같은 out_dir 에 `--stage v15_oof` 로 만든
    v15_oof_{2020..2024}.npz 5개가 있어야 한다.
    """
    ROWS, Y, V15_OOF, _ = _seed_pass1_globals(ns, out_dir, need_base=False)

    cells = load_cells(nb_path, P18_EXTRA_CELLS)
    for i in P18_EXTRA_CELLS:
        src, _ = patch_cell(i, cells[i], data_dir, "p18")
        t0 = time.time()
        exec(compile(src, f"<v18_cell_{i}>", "exec"), ns)
        print(f"  [ref] cell {i} exec  {time.time() - t0:6.1f}s", flush=True)
        gc.collect()

    P18 = ns["P18"]
    CORR = ns["CORR"]
    info: dict = {
        "transitions": sorted(int(k) for k in P18),
        "temporal_rows": ns["TEMPORAL_ROWS"],
        "v18_submit_recommended": bool(ns["V18_SUBMIT_RECOMMENDED"]),
        "summary": ns["summary"],
    }
    for ty in sorted(P18):
        core = ROWS[ty].domain3.eq(ns["V18_ROUTE_DOMAIN"]).to_numpy()
        np.savez_compressed(
            out_dir / f"p18_{ty}.npz",
            p18=np.asarray(P18[ty], np.float64),
            v15_oof=np.asarray(V15_OOF[ty], np.float64),
            corr=np.asarray(CORR[ty], np.float64),
            target=Y[ty],
            core=core,
        )
    print(f"  [ref] V18_SUBMIT_RECOMMENDED: {info['v18_submit_recommended']}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="validation/v18_repro_v001/reference")
    ap.add_argument("--year", type=int, action="append", default=None)
    ap.add_argument("--stage", choices=["features", "anchor", "v15_oof", "p18", "deploy"],
                    default="features")
    args = ap.parse_args()
    if args.stage not in ("p18", "deploy") and not args.year:
        ap.error("--year 는 --stage p18 이 아닌 경우 필수다")

    nb_path = Path(args.notebook)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ref] notebook : {nb_path}")
    print(f"[ref] data_dir : {data_dir.resolve()}")
    print(f"[ref] out      : {out_dir.resolve()}")
    print(f"[ref] stage    : {args.stage}")
    ns, patches = build_namespace(nb_path, data_dir, args.stage)
    print("[ref] patches applied:")
    for p in patches:
        print("   ", p)

    screen = ns["screen"]
    manifest = {
        "notebook": str(nb_path),
        "patches": patches,
        "stage": args.stage,
        "train_rows": int(len(screen)),
        "seasons": sorted(int(s) for s in screen.season.unique()),
        "n_columns_loaded": int(screen.shape[1]),
        "columns_loaded": list(screen.columns),
        "anchor_team_id": int(ns["ANCHOR_TEAM_ID"]),
        "rawnum": list(ns["RAWNUM"]),
        "cat_small": list(ns["CAT_SMALL"]),
        "adapter_rawnum": list(ns["ADAPTER_RAWNUM"]),
        "eng_cols": list(ns["ENG_COLS"]),
        "bank_cols": list(ns["BANK_COLS"]),
        "cf_seeds": list(ns["CF_SEEDS"]),
        "years": {},
    }
    if args.stage in ("p18", "deploy"):
        label = "Layer 2 (PASS 2)" if args.stage == "p18" else "Deploy artifact (2024->2025)"
        print(f"[ref] {label} ...", flush=True)
        t0 = time.time()
        fn = dump_p18 if args.stage == "p18" else dump_deploy
        manifest[args.stage] = fn(ns, out_dir, nb_path, data_dir)
        print(f"[ref] {args.stage} done  {time.time() - t0:.1f}s", flush=True)
        mpath = out_dir / f"manifest_{args.stage}.json"
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
        print(f"[ref] manifest -> {mpath}")
        return

    for year in args.year:
        print(f"[ref] dumping year {year} ...", flush=True)
        t0 = time.time()
        manifest["years"][str(year)] = dump_year(ns, year, out_dir, args.stage)
        print(f"[ref] year {year} done  {time.time() - t0:.1f}s", flush=True)
        gc.collect()

    mpath = out_dir / f"manifest_{'_'.join(str(y) for y in args.year)}_{args.stage}.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
    print(f"[ref] manifest -> {mpath}")


if __name__ == "__main__":
    main()
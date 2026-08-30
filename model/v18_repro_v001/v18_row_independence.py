"""V18 Reproduction — T2 : Feature 단계 행 독립성 선행 검사.

Phase 8 의 최종 gate 는
    predict(test.iloc[[i]])[0] == predict(test)[i]
이지만, 그중 **feature 변환 부분은 모델 없이 지금 검증할 수 있다.**
여기서 실패하면 Phase 6·7 결과가 통째로 무효가 되므로 먼저 돌린다.

검사 대상 (전부 train-only frozen artifact + 해당 행 입력만 사용):

    build_self_state            34
    transform_global_season     28   (frozen bank)
    exact_asof_state            62   (위 둘 결합)
    transform_pc_specs          21   (frozen EB table)
    transform_v18_stable_specs  25   (frozen EB table)
    transform_adapter           45   (frozen median/maps)
    adapter_matrix             128

검사 항목:

    A. 단일 행 vs 전체 프레임        무작위 N행, domain3 3종 포함
    B. 행 순서 셔플 불변성
    C. 프레임 분할 후 결합 == 전체
    D. 단일 행 CSV 왕복 후 동일성    (§B-5 의 "3_2" vs "3.0_2.0" dtype 위험)
    E. context 문자열 키 동일성      v15_* / v18_pressure_bhand

기준: 전부 max_abs_diff == 0.  모델이 개입하지 않으므로 1 ulp 잡음도 허용하지 않는다.

사용법:
    python3 v18_row_independence.py --year 2024 --samples 60
    python3 v18_row_independence.py --year 2024 --year 2023 --validation-dir validation/v18_repro_v001
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

CTX_COLS = ["v15_count", "v15_inning_phase", "v15_pressure", "v15_runners", "v15_li_bucket"]


# ---------------------------------------------------------------------------
# frozen artifact — 추론 시점에 존재하는 것만 만든다 (전부 공식 train 유래)
# ---------------------------------------------------------------------------
def build_frozen_artifacts(screen: pd.DataFrame, year: int) -> dict:
    source_year = int(year) - 1
    hist = screen.loc[screen[T.SEASON_COL].lt(int(year)), T.BANK_COLS]
    art = {
        "year": int(year),
        "source_year": source_year,
        "bank": T.fit_global_season_bank(hist, int(year)),
        "pc_specs": T.make_pc_specs(screen, source_year),
        "stable_specs": T.build_v18_stable_specs(screen, int(year)),
        "prep": T.fit_adapter_prep(
            screen.loc[screen[T.SEASON_COL].eq(source_year)].reset_index(drop=True)
        ),
    }
    # 규칙 재확인: artifact 는 target 연도 이전 데이터만 봤는가
    assert art["bank"]["cutoff"] == int(year)
    assert art["bank"]["source_max_season"] == source_year
    del hist
    gc.collect()
    return art


def transform_rows(art: dict, rows: pd.DataFrame, base_pred) -> dict:
    """frozen artifact + 해당 행 입력만으로 feature 를 만든다. screen 을 보지 않는다."""
    r = rows.drop(columns=[T.TARGET], errors="ignore").reset_index(drop=True)
    self_state = T.build_self_state(r)
    gseason = T.transform_global_season(art["bank"], r)
    exact = pd.concat(
        [self_state.add_prefix("exact__"), gseason.add_prefix("exact__")], axis=1
    ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    pc = T.transform_pc_specs(art["pc_specs"], r)
    stable = T.transform_v18_stable_specs(art["stable_specs"], r)
    adapter_base = T.transform_adapter(art["prep"], r, base_pred)
    X = pd.concat(
        [adapter_base.reset_index(drop=True), pc.reset_index(drop=True),
         exact.reset_index(drop=True)], axis=1
    )
    assert X.shape[1] == T.EXPECTED_DIMS["adapter_matrix"], X.shape
    return {
        "self_state": self_state, "gseason": gseason, "exact_asof_state": exact,
        "pitcher_conditional": pc, "v18_stable": stable,
        "adapter_base": adapter_base, "adapter_matrix": X,
    }


def context_keys(rows: pd.DataFrame) -> pd.DataFrame:
    """문자열 context 키만 뽑는다 (dtype 사고 탐지용)."""
    a = T._ctx_frame(rows.reset_index(drop=True))
    b = T._v18_pressure_frame(rows.reset_index(drop=True))
    out = a[CTX_COLS].copy()
    out["v18_pressure"] = b["v18_pressure"].to_numpy()
    out["v18_pressure_bhand"] = b["v18_pressure_bhand"].to_numpy()
    return out.astype(str)


def diff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, bool, bool]:
    """(max_abs_diff, 열순서 일치, NaN mask 일치)"""
    if list(a.columns) != list(b.columns):
        return float("inf"), False, False
    A = a.to_numpy(np.float64)
    B = b.to_numpy(np.float64)
    if A.shape != B.shape:
        return float("inf"), True, False
    na, nb = np.isnan(A), np.isnan(B)
    if not np.array_equal(na, nb):
        return float("inf"), True, False
    both = ~na
    d = float(np.abs(A[both] - B[both]).max()) if both.any() else 0.0
    return d, True, True


# ---------------------------------------------------------------------------
# 검사
# ---------------------------------------------------------------------------
def run_year(screen: pd.DataFrame, year: int, n_samples: int, seed: int) -> dict:
    t0 = time.time()
    rows = screen.loc[screen[T.SEASON_COL].eq(int(year))].reset_index(drop=True)
    n = len(rows)
    base = T.surrogate_base_pred(n)
    art = build_frozen_artifacts(screen, year)
    full = transform_rows(art, rows, base)
    blocks = list(full.keys())
    res: dict = {"year": int(year), "n_rows": n, "n_samples": int(n_samples), "blocks": blocks}
    print(f"  [{year}] 전체 프레임 변환 완료  n={n}  blocks={len(blocks)}")

    # ---- A. 단일 행 vs 전체 ----
    rng = np.random.default_rng(seed)
    idx = set()
    for d in ["R_CORE", "R_ANCHOR", "F"]:
        pool = np.flatnonzero(rows.domain3.eq(d).to_numpy())
        take = min(max(n_samples // 6, 3), len(pool))
        idx |= set(rng.choice(pool, size=take, replace=False).tolist())
    while len(idx) < n_samples:
        idx.add(int(rng.integers(0, n)))
    idx = sorted(idx)
    worst_a = {b: 0.0 for b in blocks}
    for i in idx:
        one = transform_rows(art, rows.iloc[[i]], base[i:i + 1])
        for b in blocks:
            d, order_ok, nan_ok = diff(full[b].iloc[[i]].reset_index(drop=True),
                                       one[b].reset_index(drop=True))
            assert order_ok and nan_ok, (year, b, i, "column/NaN mismatch")
            worst_a[b] = max(worst_a[b], d)
    res["A_single_vs_full"] = {
        "max_abs_diff": worst_a, "worst": max(worst_a.values()),
        "n_checked": len(idx),
        "domain_counts": rows.domain3.iloc[idx].value_counts().to_dict(),
        "status": "PASS" if max(worst_a.values()) == 0.0 else "FAIL",
    }
    print(f"  [{year}] A 단일행 vs 전체     worst={res['A_single_vs_full']['worst']!r}"
          f"  ({len(idx)}행)  {res['A_single_vs_full']['status']}")

    # ---- B. 행 순서 셔플 불변성 ----
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    sh = transform_rows(art, rows.iloc[perm], base[perm])
    worst_b = {}
    for b in blocks:
        back = sh[b].iloc[inv].reset_index(drop=True)
        d, order_ok, nan_ok = diff(full[b], back)
        assert order_ok and nan_ok, (year, b, "shuffle column/NaN mismatch")
        worst_b[b] = d
    res["B_shuffle"] = {"max_abs_diff": worst_b, "worst": max(worst_b.values()),
                        "status": "PASS" if max(worst_b.values()) == 0.0 else "FAIL"}
    print(f"  [{year}] B 셔플 불변          worst={res['B_shuffle']['worst']!r}"
          f"  {res['B_shuffle']['status']}")
    del sh
    gc.collect()

    # ---- C. 분할 후 결합 ----
    cuts = [0, n // 3, 2 * n // 3, n]
    parts = {b: [] for b in blocks}
    for s, e in zip(cuts[:-1], cuts[1:]):
        chunk = transform_rows(art, rows.iloc[s:e], base[s:e])
        for b in blocks:
            parts[b].append(chunk[b])
        del chunk
    worst_c = {}
    for b in blocks:
        joined = pd.concat(parts[b], ignore_index=True)
        d, order_ok, nan_ok = diff(full[b], joined)
        assert order_ok and nan_ok, (year, b, "split column/NaN mismatch")
        worst_c[b] = d
    res["C_split_concat"] = {"max_abs_diff": worst_c, "worst": max(worst_c.values()),
                             "n_chunks": len(cuts) - 1,
                             "status": "PASS" if max(worst_c.values()) == 0.0 else "FAIL"}
    print(f"  [{year}] C 분할 결합          worst={res['C_split_concat']['worst']!r}"
          f"  {res['C_split_concat']['status']}")
    del parts
    gc.collect()

    # ---- D. 단일 행 CSV 왕복 ----
    tmp = Path("/tmp") / f"_v18_rowindep_{year}.csv"
    worst_d = {b: 0.0 for b in blocks}
    dtype_changes = {}
    # 알려진 위험값을 반드시 포함한다.
    #   base_state='123' 은 단일 행 CSV 에서 int64 로 재적재되어
    #   _lookup 의 MultiIndex 키가 어긋날 수 있다 (object "123" vs int64 123).
    # ⚠ vacuous 방지: lookup 이 애초에 빗나가 0.0 인 행을 고르면 검사가 무의미하다.
    #   해당 context 의 EB 조회가 **실제로 적중한** 행만 고른다.
    pc_hit = full["pitcher_conditional"]["pc__base_state__rate"].to_numpy() != 0.0
    st_hit = full["v18_stable"]["stable__batter_hand__delta"].to_numpy() != 0.0
    live = pc_hit & st_hit
    d_idx = list(idx[: min(len(idx), 15)])
    vacuous_warn = []
    for col in ["base_state", "top_bottom", "domain3"]:
        for v in rows[col].astype(str).unique():
            mask = rows[col].astype(str).eq(v).to_numpy()
            hit = np.flatnonzero(mask & live)
            if len(hit):
                d_idx.append(int(hit[0]))
            else:
                alt = np.flatnonzero(mask)
                if len(alt):
                    d_idx.append(int(alt[0]))
                    vacuous_warn.append(f"{col}={v} (적중 행 없음 — vacuous)")
    d_idx = sorted(set(d_idx))
    for i in d_idx:
        one_df = rows.iloc[[i]].drop(columns=[T.TARGET], errors="ignore")
        one_df.to_csv(tmp, index=False)
        back = pd.read_csv(tmp, encoding="utf-8-sig", low_memory=False)
        for c in one_df.columns:
            if str(one_df[c].dtype) != str(back[c].dtype):
                dtype_changes.setdefault(c, (str(one_df[c].dtype), str(back[c].dtype)))
        assert "domain3" in back.columns, "domain3 는 screen 에서 이미 부여되어 CSV 에도 실린다"
        one = transform_rows(art, back, base[i:i + 1])
        for b in blocks:
            d, order_ok, nan_ok = diff(full[b].iloc[[i]].reset_index(drop=True),
                                       one[b].reset_index(drop=True))
            assert order_ok and nan_ok, (year, b, i, "csv column/NaN mismatch")
            worst_d[b] = max(worst_d[b], d)
    tmp.unlink(missing_ok=True)
    res["D_csv_roundtrip"] = {
        "max_abs_diff": worst_d, "worst": max(worst_d.values()),
        "n_checked": len(d_idx), "dtype_changes": dtype_changes,
        "forced_values": {c: sorted(rows[c].astype(str).unique().tolist())
                          for c in ["base_state", "top_bottom", "domain3"]},
        "non_vacuous_rows": int(live[d_idx].sum()),
        "vacuous_warnings": vacuous_warn,
        # dtype 이 바뀌어도 feature 값이 같으면 통과다. 값이 갈라지는 것만 실패로 본다.
        "status": "PASS" if max(worst_d.values()) == 0.0 else "FAIL",
    }
    print(f"  [{year}] D CSV 왕복           worst={res['D_csv_roundtrip']['worst']!r}"
          f"  dtype 변화={dtype_changes if dtype_changes else '없음'}"
          f"  {res['D_csv_roundtrip']['status']}")

    # ---- E. context 문자열 키 ----
    ck_full = context_keys(rows)
    mism = {}
    for i in idx:
        ck_one = context_keys(rows.iloc[[i]])
        for c in ck_full.columns:
            a_, b_ = ck_full[c].iloc[i], ck_one[c].iloc[0]
            if a_ != b_:
                mism.setdefault(c, []).append({"row": int(i), "full": a_, "single": b_})
    sample = {c: ck_full[c].iloc[idx[0]] for c in ck_full.columns}
    res["E_context_keys"] = {"mismatches": mism, "sample_row_keys": sample,
                             "status": "PASS" if not mism else "FAIL"}
    print(f"  [{year}] E context 키         {'일치' if not mism else mism}"
          f"  {res['E_context_keys']['status']}")
    print(f"        예시: {sample}")

    res["status"] = ("PASS" if all(res[k]["status"] == "PASS"
                                   for k in ["A_single_vs_full", "B_shuffle", "C_split_concat",
                                             "D_csv_roundtrip", "E_context_keys"]) else "FAIL")
    res["sec"] = round(time.time() - t0, 1)
    del full, art, rows
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, action="append", default=None)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--validation-dir", default="validation/v18_repro_v001")
    args = ap.parse_args()
    years = args.year or [2024]

    print("=" * 78)
    print("T2 — Feature 행 독립성 선행 검사 (모델 미개입, 기준 max_abs_diff == 0)")
    print("=" * 78)
    screen = T.load_train()
    T.validate_schema(screen)
    anchor = T.fit_router(screen)
    screen = T.attach_domain3(screen, anchor)
    T.finalize_column_lists(screen)
    print(f"ANCHOR_TEAM_ID {anchor} · rows {len(screen)} · years {years}")
    print()

    report = {"anchor_team_id": anchor, "samples": args.samples, "seed": args.seed,
              "criterion": "max_abs_diff == 0 (모델 미개입)", "years": {}}
    for y in years:
        report["years"][str(y)] = run_year(screen, y, args.samples, args.seed)
        print()
    report["overall"] = ("PASS" if all(v["status"] == "PASS" for v in report["years"].values())
                         else "FAIL")

    out = Path(args.validation_dir) / "gate_t2_row_independence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("=" * 78)
    print("OVERALL:", report["overall"])
    for y, v in report["years"].items():
        print(f"  {y}: {v['status']}  "
              f"A={v['A_single_vs_full']['worst']!r} B={v['B_shuffle']['worst']!r} "
              f"C={v['C_split_concat']['worst']!r} D={v['D_csv_roundtrip']['worst']!r} "
              f"E={v['E_context_keys']['status']}  ({v['sec']}s)")
    print("  ->", out)
    if report["overall"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

"""V18 Reproduction — Phase 8 : Row-independence 최종 audit (R8).

대회 최우선 규칙의 최종 검증이다. **실제 artifact 와 실제 추론 경로**로 확인한다.

    predict(test.iloc[[i]])[0] == predict(test)[i]

T2(`v18_row_independence.py`) 는 feature 변환만 봤다. 여기서는 `script.py` 의
`predict_rows` 를 그대로 호출하므로 **Layer 0/1/2 모델 예측까지 포함**된다.

검사:
    A. 단일 행 vs 전체        무작위 N행, domain3 3종 포함
    B. 행 순서 셔플 불변
    C. 분할 후 결합 == 전체    (배치 추론 시뮬레이션)
    D. 실제 test.csv 자체 검사

프레임 2종:
    real   : data/test.csv (형식용 5행). 실제 제출 입력과 같은 파일.
    proxy  : train 2024 에서 N행을 뽑아 season 만 expected_year 로 바꾼 대리 프레임.
             형식용 파일이 5행·단일 domain 뿐이라 커버리지가 부족해서 함께 쓴다.
             **점수를 재는 것이 아니라 코드 경로의 행 독립성을 재는 것이므로 타당하다.**

기준: `atol = 1e-12`.  §I-13 대로 `0` 을 요구하지 않는다 —
sklearn forest 의 병렬 누적 잡음이 ~1e-16 수준으로 남을 수 있다.

사용법:
    python3 v18_phase8_audit.py --samples 60 --proxy-rows 20000
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
import script as S  # noqa: E402  (import 시 MODEL_PATH 의 artifact 를 로드한다)

ATOL = 1e-12
TRAIN_PATH = Path("./data/train.csv")


def load_real_test() -> pd.DataFrame:
    t = pd.read_csv(S.TEST_PATH, encoding="utf-8-sig", low_memory=False)
    return S.validate_input(t)


def build_proxy_test(n_rows: int, seed: int) -> pd.DataFrame:
    """train 2024 에서 뽑아 season 만 바꾼 대리 test 프레임.

    형식용 test.csv 가 5행·R_CORE 뿐이라 domain3 3종과 다수 투수를 커버하기 위한 것이다.
    입력 스키마는 test.csv 와 동일해야 하므로 target 을 제거한다.
    """
    expected = int(S.META["expected_year"])
    test_cols = list(pd.read_csv(S.TEST_PATH, encoding="utf-8-sig", nrows=0).columns)
    src = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False,
                      usecols=test_cols + ["season"])
    src = src.loc[src.season.eq(expected - 1)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(src), size=min(n_rows, len(src)), replace=False)
    idx.sort()
    proxy = src.iloc[idx].reset_index(drop=True).copy()
    proxy["season"] = expected
    proxy = proxy[test_cols]
    return S.validate_input(proxy)


def audit(frame: pd.DataFrame, label: str, n_samples: int, seed: int) -> dict:
    t0 = time.time()
    n = len(frame)
    full = S.predict_rows(frame)
    res: dict = {"label": label, "n_rows": n,
                 "domain3": frame.domain3.value_counts().to_dict(),
                 "pred": {"mean": float(full.mean()), "min": float(full.min()),
                          "max": float(full.max())}}
    print(f"  [{label}] n={n}  domain3={res['domain3']}  "
          f"pred_mean={full.mean():.10f}")

    # ---- A. 단일 행 vs 전체 ----
    rng = np.random.default_rng(seed)
    idx = set()
    for d in ["R_CORE", "R_ANCHOR", "F"]:
        pool = np.flatnonzero(frame.domain3.eq(d).to_numpy())
        if len(pool):
            take = min(max(n_samples // 4, 2), len(pool))
            idx |= set(rng.choice(pool, size=take, replace=False).tolist())
    while len(idx) < min(n_samples, n):
        idx.add(int(rng.integers(0, n)))
    idx = sorted(idx)
    worst = 0.0
    detail = []
    for i in idx:
        one = S.predict_rows(frame.iloc[[i]].reset_index(drop=True))
        d = abs(float(one[0]) - float(full[i]))
        worst = max(worst, d)
        detail.append({"row": int(i), "domain3": str(frame.domain3.iloc[i]),
                       "full": float(full[i]), "single": float(one[0]), "abs_diff": d})
    res["A_single_vs_full"] = {
        "n_checked": len(idx), "max_abs_diff": worst,
        "domain_counts": frame.domain3.iloc[idx].value_counts().to_dict(),
        "worst_rows": sorted(detail, key=lambda r: -r["abs_diff"])[:5],
        "status": "PASS" if worst < ATOL else "FAIL",
    }
    print(f"  [{label}] A 단일행 vs 전체  max={worst:.3e}  ({len(idx)}행)  "
          f"{res['A_single_vs_full']['status']}")

    # ---- B. 셔플 불변 ----
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    sh = S.predict_rows(frame.iloc[perm].reset_index(drop=True))[inv]
    db = float(np.abs(sh - full).max())
    res["B_shuffle"] = {"max_abs_diff": db, "status": "PASS" if db < ATOL else "FAIL"}
    print(f"  [{label}] B 셔플 불변       max={db:.3e}  {res['B_shuffle']['status']}")
    del sh
    gc.collect()

    # ---- C. 분할 결합 ----
    cuts = [0, n // 3, 2 * n // 3, n] if n >= 3 else [0, n]
    parts = [S.predict_rows(frame.iloc[s:e].reset_index(drop=True))
             for s, e in zip(cuts[:-1], cuts[1:]) if e > s]
    joined = np.concatenate(parts)
    dc = float(np.abs(joined - full).max())
    res["C_split_concat"] = {"n_chunks": len(parts), "max_abs_diff": dc,
                             "status": "PASS" if dc < ATOL else "FAIL"}
    print(f"  [{label}] C 분할 결합       max={dc:.3e}  {res['C_split_concat']['status']}")
    del parts, joined
    gc.collect()

    res["status"] = ("PASS" if all(res[k]["status"] == "PASS"
                                   for k in ["A_single_vs_full", "B_shuffle",
                                             "C_split_concat"]) else "FAIL")
    res["sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--proxy-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--validation-dir", default="validation/v18_repro_v001")
    ap.add_argument("--skip-proxy", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("Phase 8 (R8) — Row-independence 최종 audit · 실제 artifact + script.py 추론 경로")
    print("=" * 78)
    print(f"artifact         : {S.MODEL_PATH}")
    print(f"expected_year    : {S.META['expected_year']}")
    print(f"v18_stable_enabled: {S.META.get('v18_stable_enabled')}")
    print(f"atol             : {ATOL}")
    print()

    report = {"artifact": str(S.MODEL_PATH), "atol": ATOL,
              "expected_year": int(S.META["expected_year"]),
              "v18_stable_enabled": bool(S.META.get("v18_stable_enabled")),
              "samples": args.samples, "seed": args.seed, "frames": {}}

    real = load_real_test()
    report["frames"]["real_test_csv"] = audit(real, "real", args.samples, args.seed)
    print()

    if not args.skip_proxy:
        if not TRAIN_PATH.exists():
            print(f"  [proxy] {TRAIN_PATH} 없음 — 건너뜀")
        else:
            proxy = build_proxy_test(args.proxy_rows, args.seed)
            report["frames"]["proxy_from_train2024"] = audit(
                proxy, "proxy", args.samples, args.seed)
            print()

    report["overall"] = ("PASS" if all(v["status"] == "PASS"
                                       for v in report["frames"].values()) else "FAIL")
    out = Path(args.validation_dir) / "gate_r8_row_independence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("=" * 78)
    print("OVERALL:", report["overall"])
    for k, v in report["frames"].items():
        print(f"  {k:24s} {v['status']}  n={v['n_rows']:>6}  "
              f"A={v['A_single_vs_full']['max_abs_diff']:.3e} "
              f"B={v['B_shuffle']['max_abs_diff']:.3e} "
              f"C={v['C_split_concat']['max_abs_diff']:.3e}  ({v['sec']}s)")
    print("  ->", out)
    if report["overall"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

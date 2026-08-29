#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PITCHER_CONDITIONAL R/F × Fold 진단

목적
----
cb_reg_v020_pitcher_conditional_ablation의 OOF 결과를 이용해
PITCHER_CONDITIONAL이 R/F에서 시즌별로 일관되게 작동하는지 확인한다.

입력
----
validation/cb_reg_v020_pitcher_conditional_ablation/oof_predictions.csv

출력
----
1) 터미널: 시즌 × R/F별 Control/Treatment Brier와 delta
2) CSV:
   validation/cb_reg_v020_pitcher_conditional_ablation/rf_by_fold_diagnostic.csv
3) 최종 판정:
   - R_ONLY_CANDIDATE : R에서 충분히 일관된 개선 → R에만 PC 사용 실험 가치 있음
   - REJECT_PC        : R도 불안정 → PITCHER_CONDITIONAL 종료 권장
"""

from pathlib import Path
import numpy as np
import pandas as pd

OOF_PATH = Path("validation/cb_reg_v020_pitcher_conditional_ablation/oof_predictions.csv")
OUT_PATH = Path("validation/cb_reg_v020_pitcher_conditional_ablation/rf_by_fold_diagnostic.csv")

SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

REQUIRED = {
    SEASON_COL,
    GAME_TYPE_COL,
    "y_true",
    "control_pred",
    "treatment_pred",
}


def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def summarize_part(df, season, domain):
    if domain == "R":
        part = df[(df[SEASON_COL] == season) & (df[GAME_TYPE_COL].astype(str) != "F")].copy()
    else:
        part = df[(df[SEASON_COL] == season) & (df[GAME_TYPE_COL].astype(str) == "F")].copy()

    if len(part) == 0:
        return {
            "season": season,
            "domain": domain,
            "n": 0,
            "control_brier": np.nan,
            "treatment_brier": np.nan,
            "delta_C_minus_T": np.nan,
            "win": False,
        }

    c = brier(part["y_true"], part["control_pred"])
    t = brier(part["y_true"], part["treatment_pred"])
    delta = c - t  # + 이면 Treatment가 더 좋음

    return {
        "season": int(season),
        "domain": domain,
        "n": int(len(part)),
        "control_brier": c,
        "treatment_brier": t,
        "delta_C_minus_T": delta,
        "win": bool(delta > 0),
    }


def pooled_domain(df, domain):
    if domain == "R":
        part = df[df[GAME_TYPE_COL].astype(str) != "F"].copy()
    else:
        part = df[df[GAME_TYPE_COL].astype(str) == "F"].copy()

    c = brier(part["y_true"], part["control_pred"])
    t = brier(part["y_true"], part["treatment_pred"])
    return {
        "n": int(len(part)),
        "control_brier": c,
        "treatment_brier": t,
        "delta": c - t,
    }


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF 파일을 찾을 수 없습니다:\n{OOF_PATH}\n\n"
            "프로젝트 루트에서 실행했는지 확인하세요."
        )

    df = pd.read_csv(OOF_PATH)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"OOF에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="raise").astype(int)

    # 안전성 검사
    for col in ["y_true", "control_pred", "treatment_pred"]:
        vals = pd.to_numeric(df[col], errors="raise").to_numpy(np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(f"{col}에 NaN/Inf가 있습니다.")

    seasons = sorted(df[SEASON_COL].unique().tolist())
    rows = []
    for season in seasons:
        rows.append(summarize_part(df, season, "R"))
        rows.append(summarize_part(df, season, "F"))

    result = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 96)
    print("PITCHER_CONDITIONAL | R/F × FOLD DIAGNOSTIC")
    print("=" * 96)

    for season in seasons:
        print(f"\n[{season}]")
        for domain in ["R", "F"]:
            row = result[(result["season"] == season) & (result["domain"] == domain)].iloc[0]
            if int(row["n"]) == 0:
                print(f"{domain} | no rows")
                continue
            print(
                f"{domain} | n={int(row['n']):6d} | "
                f"C={row['control_brier']:.10f} | "
                f"T={row['treatment_brier']:.10f} | "
                f"delta(C-T)={row['delta_C_minus_T']:+.10f} | "
                f"{'WIN' if bool(row['win']) else 'LOSS'}"
            )

    r_pool = pooled_domain(df, "R")
    f_pool = pooled_domain(df, "F")

    r_rows = result[result["domain"] == "R"].dropna(subset=["delta_C_minus_T"])
    f_rows = result[result["domain"] == "F"].dropna(subset=["delta_C_minus_T"])

    r_wins = int((r_rows["delta_C_minus_T"] > 0).sum())
    f_wins = int((f_rows["delta_C_minus_T"] > 0).sum())
    n_folds = len(r_rows)

    recent_seasons = [s for s in seasons if s >= 2023]
    r_recent = result[
        (result["domain"] == "R") & result["season"].isin(recent_seasons)
    ]["delta_C_minus_T"].mean()
    f_recent = result[
        (result["domain"] == "F") & result["season"].isin(recent_seasons)
    ]["delta_C_minus_T"].mean()

    print("\n" + "-" * 96)
    print("SUMMARY")
    print("-" * 96)
    print(f"R pooled delta       : {r_pool['delta']:+.10f}")
    print(f"R fold wins          : {r_wins}/{n_folds}")
    print(f"R recent 23-24 mean  : {r_recent:+.10f}")
    print()
    print(f"F pooled delta       : {f_pool['delta']:+.10f}")
    print(f"F fold wins          : {f_wins}/{len(f_rows)}")
    print(f"F recent 23-24 mean  : {f_recent:+.10f}")

    # 매우 보수적인 다음 단계 판정
    # R-only 실험은:
    # - R pooled 개선
    # - 4 folds 중 최소 3 wins
    # - 최근 2023~24 평균도 양수
    r_only_candidate = (
        r_pool["delta"] > 0
        and r_wins >= max(3, n_folds - 1)
        and r_recent > 0
    )

    print("\n" + "=" * 96)
    if r_only_candidate:
        print("VERDICT: R_ONLY_CANDIDATE")
        print(
            "→ PITCHER_CONDITIONAL은 global 적용은 실패했지만, "
            "R에서 시간적으로 비교적 안정적인 신호가 확인됨.\n"
            "→ 다음 실험: global CatBoost는 유지하고, "
            "R row에서만 PC를 활성화 / F row에서는 neutral 처리하는 ablation."
        )
    else:
        print("VERDICT: REJECT_PC")
        print(
            "→ R pooled 개선만으로는 부족하고 시즌별 일관성이 확인되지 않음.\n"
            "→ PITCHER_CONDITIONAL은 여기서 종료하는 것을 권장."
        )

    print(f"\nSaved: {OUT_PATH}")
    print("=" * 96)


if __name__ == "__main__":
    main()

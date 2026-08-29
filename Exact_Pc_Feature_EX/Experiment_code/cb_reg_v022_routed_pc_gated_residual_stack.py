#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v022_routed_pc_gated_residual_stack.py

목적
----
현재 R-only PITCHER_CONDITIONAL Routed Base 위에
기존에 성공했던 Gated Residual Correction을 "다시 계산해서" 붙였을 때
추가 이득이 유지되는지 검증한다.

중요한 비교
-----------
A) Control raw
   = Exact-ASOF CatBoost

B) Current v019-local
   = Control raw + Control OOF로 다시 만든 frozen Gated Residual
   -> 기존 v019 residual stacking sanity check

C) New routed raw
   = R: R-only PITCHER_CONDITIONAL CatBoost
     F: Control Exact-ASOF prediction

D) New routed + rebuilt Gated Residual
   = C의 OOF residual을 새로 계산한 뒤
     동일한 frozen Gated config로 correction table을 다시 생성

핵심 원칙
---------
- Residual 파라미터 재튜닝 금지.
- 기존 v019에서 확정한 config를 그대로 사용.
- 새 Routed Base의 residual로 correction table을 반드시 새로 생성.
- 2021 residual -> 2022
  2022 residual -> 2023
  2023 residual -> 2024
  strict temporal transfer만 평가.
- test 데이터 사용 없음.

입력
----
./data/train.csv
./validation/cb_reg_v021_pitcher_conditional_r_only_ablation/oof_predictions.csv

출력
----
./validation/cb_reg_v022_routed_pc_gated_residual_stack/
    transition_results.csv
    pooled_predictions.csv
    metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v022_routed_pc_gated_residual_stack"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"

BASE_OOF_PATH = (
    Path("./validation")
    / "cb_reg_v021_pitcher_conditional_r_only_ablation"
    / "oof_predictions.csv"
)

OUT_DIR = Path("./validation") / EXPERIMENT_NAME

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

PARENT_KEYS = ["pitcher_id", "batter_hand"]
DETAIL_KEYS = ["pitcher_id", "pressure_state", "batter_hand"]

TRANSITIONS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]

# -------------------------------------------------------------------------
# Frozen config: 기존 successful Gated Residual에서 확정한 값을 그대로 사용.
# 이번 실험에서는 절대 튜닝하지 않는다.
# -------------------------------------------------------------------------
FROZEN_CONFIG = {
    "min_n_parent": 200,
    "min_n_detail": 200,
    "shrink_parent": 300.0,
    "shrink_detail": 100.0,
    "parent_only_scale": 0.50,
    "detail_weight": 0.50,
    "alpha": 0.25,
    "cap": 0.02,
    "min_abs_effect": 0.003,
    "center_residual_by_game_type": True,
}


# =============================================================================
# Metrics
# =============================================================================

def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def metric_block(y_true, raw_pred, corrected_pred):
    raw = brier_score(y_true, raw_pred)
    corrected = brier_score(y_true, corrected_pred)
    return {
        "raw_brier": raw,
        "corrected_brier": corrected,
        "gain": float(raw - corrected),
    }


def subset_gain(y, raw, corrected, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            "n": 0,
            "raw_brier": None,
            "corrected_brier": None,
            "gain": None,
        }

    yy = np.asarray(y, dtype=np.float64)[mask]
    rr = np.asarray(raw, dtype=np.float64)[mask]
    cc = np.asarray(corrected, dtype=np.float64)[mask]

    return {
        "n": int(mask.sum()),
        **metric_block(yy, rr, cc),
    }


# =============================================================================
# Load / prepare data
# =============================================================================

def add_pressure_state(df):
    df = df.copy()

    balls = (
        pd.to_numeric(df["balls_before"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    strikes = (
        pd.to_numeric(df["strikes_before"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(
            (balls == 3) | (strikes == 2),
            "high",
            "normal",
        ),
    )

    return df


def load_frame():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train.csv: {TRAIN_PATH}")
    if not BASE_OOF_PATH.exists():
        raise FileNotFoundError(
            f"Missing v021 OOF: {BASE_OOF_PATH}\n"
            "먼저 cb_reg_v021_pitcher_conditional_r_only_ablation.py를 실행하세요."
        )

    required_train = [
        ID_COL,
        TARGET_COL,
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=required_train,
        encoding="utf-8-sig",
        low_memory=False,
    )
    train = add_pressure_state(train)

    oof = pd.read_csv(BASE_OOF_PATH)

    required_oof = {
        ID_COL,
        SEASON_COL,
        GAME_TYPE_COL,
        "y_true",
        "control_pred",
        "routed_pred",
    }
    missing = sorted(required_oof - set(oof.columns))
    if missing:
        raise KeyError(f"v021 OOF missing columns: {missing}")

    if oof[ID_COL].duplicated().any():
        raise ValueError("v021 OOF row_id duplicated")

    frame = oof.merge(
        train[
            [
                ID_COL,
                TARGET_COL,
                SEASON_COL,
                GAME_TYPE_COL,
                "pitcher_id",
                "batter_hand",
                "pressure_state",
            ]
        ],
        on=ID_COL,
        how="left",
        validate="one_to_one",
        suffixes=("_oof", "_train"),
        sort=False,
    )

    # Merge 후 season/game_type consistency 확인.
    if frame[TARGET_COL].isna().any():
        raise ValueError("OOF rows missing from train.csv")

    oof_season = pd.to_numeric(frame[f"{SEASON_COL}_oof"], errors="raise").astype(int)
    tr_season = pd.to_numeric(frame[f"{SEASON_COL}_train"], errors="raise").astype(int)

    if not np.array_equal(oof_season.to_numpy(), tr_season.to_numpy()):
        raise ValueError("OOF/train season mismatch")

    oof_gt = frame[f"{GAME_TYPE_COL}_oof"].fillna("__MISSING__").astype(str)
    tr_gt = frame[f"{GAME_TYPE_COL}_train"].fillna("__MISSING__").astype(str)
    if not np.array_equal(oof_gt.to_numpy(), tr_gt.to_numpy()):
        raise ValueError("OOF/train game_type mismatch")

    frame[SEASON_COL] = tr_season
    frame[GAME_TYPE_COL] = tr_gt

    # y_true도 공식 train label과 동일한지 확인.
    if not np.allclose(
        frame["y_true"].to_numpy(np.float64),
        frame[TARGET_COL].to_numpy(np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("OOF y_true != train control_success")

    for col in ["pitcher_id", "batter_hand", "pressure_state", GAME_TYPE_COL]:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

    for col in ["control_pred", "routed_pred", "y_true"]:
        arr = pd.to_numeric(frame[col], errors="raise").to_numpy(np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"{col} contains NaN/Inf")
        if col != "y_true" and ((arr < 0.0) | (arr > 1.0)).any():
            raise ValueError(f"{col} outside [0,1]")

    # 불필요 merge duplicate columns 제거.
    frame = frame.drop(
        columns=[
            f"{SEASON_COL}_oof",
            f"{SEASON_COL}_train",
            f"{GAME_TYPE_COL}_oof",
            f"{GAME_TYPE_COL}_train",
        ]
    )

    return frame


# =============================================================================
# Residual table
# =============================================================================

def add_residual_columns(frame, pred_col):
    out = frame.copy()

    out["residual"] = (
        out[TARGET_COL].to_numpy(np.float64)
        - out[pred_col].to_numpy(np.float64)
    )

    if FROZEN_CONFIG["center_residual_by_game_type"]:
        domain_mean = (
            out.groupby(
                [SEASON_COL, GAME_TYPE_COL],
                dropna=False,
                sort=False,
            )["residual"]
            .transform("mean")
            .to_numpy(np.float64)
        )
        out["conditional_residual"] = (
            out["residual"].to_numpy(np.float64)
            - domain_mean
        )
    else:
        out["conditional_residual"] = out["residual"].to_numpy(np.float64)

    return out


def fit_table(source, key_cols, min_n, shrink_k):
    stats = (
        source.groupby(
            key_cols,
            dropna=False,
            sort=False,
        )["conditional_residual"]
        .agg(n="size", effect="mean")
        .reset_index()
    )

    stats["eligible"] = (
        (stats["n"] >= int(min_n))
        & (
            stats["effect"].abs()
            >= float(FROZEN_CONFIG["min_abs_effect"])
        )
    )

    stats["reliability"] = (
        stats["n"]
        / (stats["n"] + float(shrink_k))
    )

    stats["shrunk_effect"] = (
        stats["effect"] * stats["reliability"]
    )

    stats.loc[~stats["eligible"], "shrunk_effect"] = 0.0

    return (
        stats.loc[
            stats["shrunk_effect"] != 0.0,
            key_cols
            + [
                "n",
                "effect",
                "reliability",
                "shrunk_effect",
            ],
        ]
        .copy()
        .reset_index(drop=True)
    )


def lookup_effect(target, table, key_cols):
    if len(table) == 0:
        return np.zeros(len(target), dtype=np.float64)

    return (
        target[key_cols]
        .merge(
            table[key_cols + ["shrunk_effect"]],
            on=key_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )["shrunk_effect"]
        .fillna(0.0)
        .to_numpy(np.float64)
    )


def build_gated_correction(parent_effect, detail_effect):
    parent_effect = np.asarray(parent_effect, dtype=np.float64)
    detail_effect = np.asarray(detail_effect, dtype=np.float64)

    parent_available = parent_effect != 0.0
    detail_available = detail_effect != 0.0

    same_sign = (
        parent_available
        & detail_available
        & (np.sign(parent_effect) == np.sign(detail_effect))
    )

    parent_only = parent_available & ~detail_available
    conflict = parent_available & detail_available & ~same_sign
    detail_only = ~parent_available & detail_available

    correction = np.zeros(len(parent_effect), dtype=np.float64)

    correction[parent_only] = (
        parent_effect[parent_only]
        * float(FROZEN_CONFIG["parent_only_scale"])
    )

    w = float(FROZEN_CONFIG["detail_weight"])
    correction[same_sign] = (
        (1.0 - w) * parent_effect[same_sign]
        + w * detail_effect[same_sign]
    )

    correction *= float(FROZEN_CONFIG["alpha"])
    correction = np.clip(
        correction,
        -float(FROZEN_CONFIG["cap"]),
        float(FROZEN_CONFIG["cap"]),
    )

    diag = {
        "parent_available_rows": int(parent_available.sum()),
        "detail_available_rows": int(detail_available.sum()),
        "same_sign_rows": int(same_sign.sum()),
        "parent_only_rows": int(parent_only.sum()),
        "conflict_rows": int(conflict.sum()),
        "detail_only_rows": int(detail_only.sum()),
        "corrected_rows": int((correction != 0.0).sum()),
        "coverage": float(np.mean(correction != 0.0)),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "max_abs_correction": float(np.max(np.abs(correction))) if len(correction) else 0.0,
    }

    return correction, diag


# =============================================================================
# Temporal transfer
# =============================================================================

def evaluate_transition(frame_with_residual, pred_col, source_year, target_year, label):
    source = (
        frame_with_residual.loc[
            frame_with_residual[SEASON_COL].eq(int(source_year))
        ]
        .copy()
        .reset_index(drop=True)
    )

    target = (
        frame_with_residual.loc[
            frame_with_residual[SEASON_COL].eq(int(target_year))
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(source) == 0 or len(target) == 0:
        raise RuntimeError(f"Empty transition {source_year}->{target_year}")

    parent_table = fit_table(
        source,
        PARENT_KEYS,
        FROZEN_CONFIG["min_n_parent"],
        FROZEN_CONFIG["shrink_parent"],
    )

    detail_table = fit_table(
        source,
        DETAIL_KEYS,
        FROZEN_CONFIG["min_n_detail"],
        FROZEN_CONFIG["shrink_detail"],
    )

    parent_effect = lookup_effect(target, parent_table, PARENT_KEYS)
    detail_effect = lookup_effect(target, detail_table, DETAIL_KEYS)

    correction, diag = build_gated_correction(
        parent_effect,
        detail_effect,
    )

    raw_pred = target[pred_col].to_numpy(np.float64)
    corrected_pred = np.clip(raw_pred + correction, 0.0, 1.0)
    y = target[TARGET_COL].to_numpy(np.float64)

    total = metric_block(y, raw_pred, corrected_pred)

    gt = target[GAME_TYPE_COL].to_numpy()
    r_block = subset_gain(y, raw_pred, corrected_pred, gt != "F")
    f_block = subset_gain(y, raw_pred, corrected_pred, gt == "F")

    result = {
        "system": label,
        "pred_col": pred_col,
        "source_year": int(source_year),
        "target_year": int(target_year),
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "parent_groups": int(len(parent_table)),
        "detail_groups": int(len(detail_table)),
        **diag,
        "raw_brier": total["raw_brier"],
        "corrected_brier": total["corrected_brier"],
        "gain": total["gain"],
        "R_n": r_block["n"],
        "R_raw_brier": r_block["raw_brier"],
        "R_corrected_brier": r_block["corrected_brier"],
        "R_gain": r_block["gain"],
        "F_n": f_block["n"],
        "F_raw_brier": f_block["raw_brier"],
        "F_corrected_brier": f_block["corrected_brier"],
        "F_gain": f_block["gain"],
    }

    prediction_frame = pd.DataFrame({
        ID_COL: target[ID_COL].to_numpy(),
        SEASON_COL: int(target_year),
        GAME_TYPE_COL: target[GAME_TYPE_COL].to_numpy(),
        "y_true": y,
        f"{label}_raw_pred": raw_pred,
        f"{label}_correction": correction,
        f"{label}_corrected_pred": corrected_pred,
    })

    return result, prediction_frame


# =============================================================================
# Summary
# =============================================================================

def pooled_metrics(predictions, raw_col, corrected_col):
    y = predictions["y_true"].to_numpy(np.float64)
    raw = predictions[raw_col].to_numpy(np.float64)
    corrected = predictions[corrected_col].to_numpy(np.float64)

    total = metric_block(y, raw, corrected)

    gt = predictions[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    r = subset_gain(y, raw, corrected, gt != "F")
    f = subset_gain(y, raw, corrected, gt == "F")

    return {
        "n": int(len(predictions)),
        **total,
        "R": r,
        "F": f,
    }


def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {OUT_DIR}\n"
            "기존 실험을 보존하기 위해 폴더를 지우거나 EXPERIMENT_NAME을 바꾸세요."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    frame = load_frame()

    # ---------------------------------------------------------------------
    # 두 시스템의 residual을 각각 독립적으로 계산.
    # Control tables를 Routed에 재사용하지 않는다.
    # ---------------------------------------------------------------------
    control_frame = add_residual_columns(frame, "control_pred")
    routed_frame = add_residual_columns(frame, "routed_pred")

    transition_rows = []
    control_preds = []
    routed_preds = []

    print("=" * 108)
    print("V022 | ROUTED PITCHER_CONDITIONAL + REBUILT GATED RESIDUAL")
    print("=" * 108)
    print("Frozen residual config:")
    print(json.dumps(FROZEN_CONFIG, indent=2, ensure_ascii=False))
    print()

    for source_year, target_year in TRANSITIONS:
        c_result, c_pred = evaluate_transition(
            control_frame,
            "control_pred",
            source_year,
            target_year,
            "control",
        )
        r_result, r_pred = evaluate_transition(
            routed_frame,
            "routed_pred",
            source_year,
            target_year,
            "routed",
        )

        transition_rows.extend([c_result, r_result])
        control_preds.append(c_pred)
        routed_preds.append(r_pred)

        print(f"[{source_year} -> {target_year}]")
        print(
            "CURRENT v019-local | "
            f"raw={c_result['raw_brier']:.10f} "
            f"corrected={c_result['corrected_brier']:.10f} "
            f"gain={c_result['gain']:+.10f} | "
            f"R={c_result['R_gain']:+.10f} "
            f"F={c_result['F_gain']:+.10f} | "
            f"coverage={c_result['coverage']:.4f}"
        )
        print(
            "NEW routed+gated  | "
            f"raw={r_result['raw_brier']:.10f} "
            f"corrected={r_result['corrected_brier']:.10f} "
            f"gain={r_result['gain']:+.10f} | "
            f"R={r_result['R_gain']:+.10f} "
            f"F={r_result['F_gain']:+.10f} | "
            f"coverage={r_result['coverage']:.4f}"
        )
        print(
            "FINAL corrected improvement vs current v019-local: "
            f"{c_result['corrected_brier'] - r_result['corrected_brier']:+.10f}"
        )
        print("-" * 108)

    transition_df = pd.DataFrame(transition_rows)
    transition_df.to_csv(
        OUT_DIR / "transition_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------------
    # Pooled 2022-2024 comparison
    # ---------------------------------------------------------------------
    control_pred_df = pd.concat(control_preds, ignore_index=True)
    routed_pred_df = pd.concat(routed_preds, ignore_index=True)

    merged = control_pred_df.merge(
        routed_pred_df[
            [
                ID_COL,
                f"routed_raw_pred",
                f"routed_correction",
                f"routed_corrected_pred",
            ]
        ],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(control_pred_df):
        raise RuntimeError("Pooled prediction merge row count mismatch")

    control_pool = pooled_metrics(
        merged,
        "control_raw_pred",
        "control_corrected_pred",
    )
    routed_pool = pooled_metrics(
        merged,
        "routed_raw_pred",
        "routed_corrected_pred",
    )

    # 2023-2024 recent
    recent = merged[merged[SEASON_COL].isin([2023, 2024])].copy()
    control_recent = pooled_metrics(
        recent,
        "control_raw_pred",
        "control_corrected_pred",
    )
    routed_recent = pooled_metrics(
        recent,
        "routed_raw_pred",
        "routed_corrected_pred",
    )

    current_final = control_pool["corrected_brier"]
    new_final = routed_pool["corrected_brier"]
    final_improvement = current_final - new_final

    current_recent_final = control_recent["corrected_brier"]
    new_recent_final = routed_recent["corrected_brier"]
    recent_final_improvement = current_recent_final - new_recent_final

    routed_transition_rows = transition_df[transition_df["system"] == "routed"]
    routed_residual_wins = int((routed_transition_rows["gain"] > 0).sum())

    final_vs_current_wins = 0
    for _, row in transition_df[transition_df["system"] == "routed"].iterrows():
        target_year = int(row["target_year"])
        current_row = transition_df[
            (transition_df["system"] == "control")
            & (transition_df["target_year"] == target_year)
        ].iloc[0]
        if float(row["corrected_brier"]) < float(current_row["corrected_brier"]):
            final_vs_current_wins += 1

    # Conservative verdict:
    # 1) 새 Routed base 위 residual 자체가 3/3 추가 gain
    # 2) 최종 corrected가 current v019-local보다 pooled 우수
    # 3) recent 2023-24도 우수
    # 4) target seasons 3/3에서 current v019-local corrected보다 우수
    if (
        routed_residual_wins == 3
        and final_improvement > 0
        and recent_final_improvement > 0
        and final_vs_current_wins == 3
    ):
        verdict = "STRONG_PASS"
    elif (
        routed_residual_wins >= 2
        and final_improvement > 0
        and recent_final_improvement > 0
    ):
        verdict = "PASS"
    elif final_improvement > 0:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    merged.to_csv(
        OUT_DIR / "pooled_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "input_oof": str(BASE_OOF_PATH),
        "frozen_config": FROZEN_CONFIG,
        "transitions": TRANSITIONS,
        "control_system": "Exact-ASOF control + rebuilt frozen Gated Residual (v019-local sanity)",
        "new_system": "Routed R-only PITCHER_CONDITIONAL + rebuilt frozen Gated Residual",
        "control_pooled_2022_2024": control_pool,
        "routed_pooled_2022_2024": routed_pool,
        "control_recent_2023_2024": control_recent,
        "routed_recent_2023_2024": routed_recent,
        "final_corrected_improvement_vs_current": float(final_improvement),
        "recent_corrected_improvement_vs_current": float(recent_final_improvement),
        "routed_residual_transition_wins": routed_residual_wins,
        "final_vs_current_transition_wins": final_vs_current_wins,
        "verdict": verdict,
    }

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 108)
    print("FINAL STACKING SUMMARY | 2022-2024")
    print("=" * 108)
    print(
        f"Current v019-local corrected : {current_final:.10f}"
    )
    print(
        f"New routed+gated corrected   : {new_final:.10f}"
    )
    print(
        f"Improvement vs current       : {final_improvement:+.10f}"
    )
    print("-" * 108)
    print(
        f"Recent 23-24 current         : {current_recent_final:.10f}"
    )
    print(
        f"Recent 23-24 new             : {new_recent_final:.10f}"
    )
    print(
        f"Recent improvement           : {recent_final_improvement:+.10f}"
    )
    print("-" * 108)
    print(
        f"New residual gain wins       : {routed_residual_wins}/3"
    )
    print(
        f"Final vs current wins        : {final_vs_current_wins}/3"
    )
    print(f"Verdict                      : {verdict}")
    print("-" * 108)
    print(
        "Control residual sanity gains: "
        + ", ".join(
            f"{int(row.target_year)}={row.gain:+.10f}"
            for row in transition_df[
                transition_df["system"] == "control"
            ].itertuples()
        )
    )
    print(
        "New routed residual gains    : "
        + ", ".join(
            f"{int(row.target_year)}={row.gain:+.10f}"
            for row in transition_df[
                transition_df["system"] == "routed"
            ].itertuples()
        )
    )
    print(f"Saved: {OUT_DIR}")
    print("=" * 108)


if __name__ == "__main__":
    main()

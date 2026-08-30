#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v024_pitcher_conditional_block_ablation.py
=================================================

목적
----
R-only PITCHER_CONDITIONAL(PC) 7개 context 중
어떤 block이 실제 다음 시즌 Brier/BSS 개선에 기여하는지
Grouped Leave-One-Block-Out Ablation으로 확인한다.

중요
----
Feature Importance가 아니라 "실제 성능 기여"를 보는 실험이다.

Control
    canonical cb_reg_v003 + Exact-ASOF 5 + HB-EB

Full
    Control + R-only PITCHER_CONDITIONAL 7 blocks (21 features)
    Routed prediction:
        R -> Full PC model
        F -> Control model

Ablation
    Full에서 context 1개(3 features)를 제거하고 동일 조건으로 재학습
    Routed prediction:
        R -> 해당 Ablation model
        F -> Control model

비교
----
Full vs -Count
Full vs -BatterHand
Full vs -InningPhase
Full vs -BaseState
Full vs -Pressure
Full vs -Runners
Full vs -LIBucket

해석
----
delta_bss_full_minus_ablation > 0
    -> 그 block을 제거했더니 성능이 나빠짐
    -> block이 도움

delta_bss_full_minus_ablation < 0
    -> 그 block을 제거했더니 성능이 좋아짐
    -> block이 noise / 제거 후보

주의
----
- block별 contribution은 서로 더하면 안 된다.
  correlated feature가 있으므로 Leave-One-Out 효과는 비가산적이다.
- 이 단계에서는 Residual Correction을 일부러 붙이지 않는다.
  Residual이 빠진 PC 자체의 기여를 먼저 분리하기 위함이다.
- 유망한 제거/유지 후보만 이후 Multi-seed 또는 Residual rebuild로 확인한다.

필수 파일
---------
프로젝트 루트에 아래 파일이 있어야 한다.

    cb_reg_v021_pitcher_conditional_r_only_ablation.py

실행
----
    python cb_reg_v024_pitcher_conditional_block_ablation.py

예상 학습 횟수
-------------
4 folds × (Control 1 + Full 1 + 7 Ablations) = 36 CatBoost fits

출력
----
validation/cb_reg_v024_pitcher_conditional_block_ablation/
    block_ablation_summary.csv
    fold_results.csv
    oof_predictions.csv
    pitcher_conditional_coverage.csv
    metadata.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v024_pitcher_conditional_block_ablation"

ROOT = Path(".")
SOURCE_PATH = ROOT / "cb_reg_v021_pitcher_conditional_r_only_ablation.py"

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_DIR = ROOT / "validation" / EXPERIMENT_NAME
SUMMARY_PATH = VALIDATION_DIR / "block_ablation_summary.csv"
FOLD_PATH = VALIDATION_DIR / "fold_results.csv"
OOF_PATH = VALIDATION_DIR / "oof_predictions.csv"
COVERAGE_PATH = VALIDATION_DIR / "pitcher_conditional_coverage.csv"
METADATA_PATH = VALIDATION_DIR / "metadata.json"

# v021 seed=42 routed-PC 결과 reproduction guard.
EXPECTED_CONTROL_POOLED_BRIER = 0.2469124012
EXPECTED_FULL_ROUTED_POOLED_DELTA = 0.0000479983
EXPECTED_FULL_ROUTED_FOLD_WINS = 4

CONTROL_REPRO_TOL = 2e-7
FULL_DELTA_REPRO_TOL = 3e-7


# 보기 좋은 block 이름.
BLOCK_LABELS = {
    "v15_count": "count",
    "batter_hand": "batter_hand",
    "v15_inning_phase": "inning_phase",
    "base_state": "base_state",
    "v15_pressure": "pressure",
    "v15_runners": "runners",
    "v15_li_bucket": "li_bucket",
}


# =============================================================================
# Import exact v021 implementation
# =============================================================================

def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"필수 파일이 없습니다: {path}\n"
            "cb_reg_v021_pitcher_conditional_r_only_ablation.py를 "
            "프로젝트 루트에 두고 실행하세요."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Metrics
# =============================================================================

def calc_metrics(base, y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    return {
        "n": int(len(y)),
        "brier": float(base.brier_score(y, pred)),
        "bss": float(base.calculate_bss(y, pred)),
    }


def routed_metrics(base, frame: pd.DataFrame, pred_col: str):
    return calc_metrics(
        base,
        frame["y_true"].to_numpy(np.float64),
        frame[pred_col].to_numpy(np.float64),
    )


# =============================================================================
# Block definitions
# =============================================================================

def get_block_features(base) -> Dict[str, List[str]]:
    blocks = {}

    for ctx, _strength in base.PC_CONTEXTS:
        blocks[ctx] = [
            f"pc__{ctx}__{field}"
            for field in base.PC_FIELDS
        ]

    flattened = [
        feature
        for features in blocks.values()
        for feature in features
    ]

    if flattened != list(base.PC_FEATURES):
        raise RuntimeError(
            "PC block definition does not reproduce v021 PC feature order"
        )

    for ctx, feats in blocks.items():
        if len(feats) != 3:
            raise RuntimeError(
                f"{ctx}: expected 3 PC fields, got {len(feats)}"
            )

    return blocks


def get_variant_feature_map(base, blocks):
    """
    FULL + 7 leave-one-block-out variants.
    """
    variants = {
        "FULL": list(base.PC_FEATURES),
    }

    for removed_ctx in blocks:
        variants[f"DROP__{removed_ctx}"] = [
            c for c in base.PC_FEATURES
            if c not in set(blocks[removed_ctx])
        ]

    return variants


# =============================================================================
# One complete validation run
# =============================================================================

def run_validation(base, train, base_features):
    blocks = get_block_features(base)
    variants = get_variant_feature_map(base, blocks)

    fold_rows = []
    oof_frames = []
    coverage_frames = []

    for cfg in base.EXPANDING_FOLDS:
        fold = int(cfg["fold"])
        train_seasons = list(cfg["train_seasons"])
        val_year = int(cfg["validation_season"])

        print("\n" + "=" * 110)
        print(f"Fold {fold}: {train_seasons} -> {val_year}")
        print("=" * 110)

        df_train = train.loc[
            train[base.SEASON_COL].isin(train_seasons)
        ].copy()

        df_val = train.loc[
            train[base.SEASON_COL].eq(val_year)
        ].copy()

        if len(df_train) == 0 or len(df_val) == 0:
            raise RuntimeError(
                f"Fold {fold}: empty train/validation"
            )

        # ------------------------------------------------------------------
        # 1) Exact-ASOF: 모든 variant 공통
        # ------------------------------------------------------------------
        pitcher_season_stats = base.build_pitcher_season_stats(
            df_train
        )

        train_exact, _train_exact_diag = (
            base.add_pitcher_exact_asof_features(
                df_train,
                pitcher_season_stats,
            )
        )

        val_exact, _val_exact_diag = (
            base.add_pitcher_exact_asof_features(
                df_val,
                pitcher_season_stats,
            )
        )

        exact_features = list(base_features)
        for col in base.EXACT_FEATURES:
            if col not in exact_features:
                exact_features.append(col)

        # ------------------------------------------------------------------
        # 2) HB-EB: 모든 variant 공통
        # ------------------------------------------------------------------
        (
            train_control,
            control_features,
            _train_hbeb,
        ) = base.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )

        (
            val_control,
            val_control_features,
            _val_hbeb,
        ) = base.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=val_exact,
            base_features=exact_features,
        )

        if control_features != val_control_features:
            raise RuntimeError(
                f"Fold {fold}: control train/val feature order mismatch"
            )

        if len(control_features) != base.EXPECTED_CONTROL_FEATURES:
            raise RuntimeError(
                f"Fold {fold}: control feature count "
                f"{len(control_features)} != {base.EXPECTED_CONTROL_FEATURES}"
            )

        if "asof_pitcher_pitchmix_n" not in control_features:
            raise RuntimeError(
                "canonical v003 feature asof_pitcher_pitchmix_n missing"
            )

        # ------------------------------------------------------------------
        # 3) PC 21개는 단 한 번 생성.
        #    이후 variant마다 block column만 선택.
        # ------------------------------------------------------------------
        pc_train, cov_train = base.build_pitcher_conditional_for_rows(
            rows=df_train,
            available_history=df_train,
            scope="train",
        )

        pc_val, cov_val = base.build_pitcher_conditional_for_rows(
            rows=df_val,
            available_history=df_train,
            scope="validation",
        )

        # R-only masking: F PC=0.
        pc_train = base.mask_pitcher_conditional_to_r_only(
            pc_train,
            df_train,
        )
        pc_val = base.mask_pitcher_conditional_to_r_only(
            pc_val,
            df_val,
        )

        cov_train["fold"] = fold
        cov_train["validation_season"] = val_year
        cov_val["fold"] = fold
        cov_val["validation_season"] = val_year
        coverage_frames.extend([cov_train, cov_val])

        # ------------------------------------------------------------------
        # 4) Control fit: fold당 1회
        # ------------------------------------------------------------------
        control_cat = [
            c for c in base.CAT_COLS
            if c in control_features
        ]

        print(
            f"[CONTROL] features={len(control_features)}"
        )

        (
            control_pred,
            control_iter,
            control_time,
        ) = base.fit_catboost_and_predict(
            train_control,
            val_control,
            control_features,
            control_cat,
        )

        y_val = df_val[base.TARGET].to_numpy(
            dtype=np.float64
        )

        game_type = (
            df_val[base.GAME_TYPE_COL]
            .fillna("__MISSING__")
            .astype(str)
            .to_numpy()
        )

        r_mask = game_type != "F"
        f_mask = game_type == "F"

        control_metric = calc_metrics(
            base,
            y_val,
            control_pred,
        )

        fold_rows.append({
            "fold": fold,
            "validation_season": val_year,
            "variant": "CONTROL",
            "removed_block": "",
            "n_features": int(len(control_features)),
            "best_iteration": int(control_iter),
            "elapsed_seconds": float(control_time),
            "brier": float(control_metric["brier"]),
            "bss": float(control_metric["bss"]),
            "R_brier": float(
                base.brier_score(
                    y_val[r_mask],
                    control_pred[r_mask],
                )
            ),
            "R_bss": float(
                base.calculate_bss(
                    y_val[r_mask],
                    control_pred[r_mask],
                )
            ),
            "F_brier": float(
                base.brier_score(
                    y_val[f_mask],
                    control_pred[f_mask],
                )
            ) if f_mask.any() else np.nan,
            "F_bss": float(
                base.calculate_bss(
                    y_val[f_mask],
                    control_pred[f_mask],
                )
            ) if f_mask.any() else np.nan,
        })

        fold_oof = pd.DataFrame({
            base.ID: df_val[base.ID].to_numpy(),
            base.SEASON_COL: val_year,
            base.GAME_TYPE_COL: game_type,
            "y_true": y_val,
            "CONTROL": control_pred,
        })

        # ------------------------------------------------------------------
        # 5) FULL + 7 LOO models
        # ------------------------------------------------------------------
        for variant_name, selected_pc_features in variants.items():
            removed_ctx = (
                ""
                if variant_name == "FULL"
                else variant_name.replace("DROP__", "", 1)
            )

            train_variant = train_control.copy()
            val_variant = val_control.copy()

            for col in selected_pc_features:
                train_variant[col] = pc_train[col]
                val_variant[col] = pc_val[col]

            features = (
                list(control_features)
                + list(selected_pc_features)
            )

            if len(set(features)) != len(features):
                raise RuntimeError(
                    f"{variant_name}: duplicated feature"
                )

            expected_count = (
                base.EXPECTED_CONTROL_FEATURES
                + len(selected_pc_features)
            )

            if len(features) != expected_count:
                raise RuntimeError(
                    f"{variant_name}: feature count "
                    f"{len(features)} != {expected_count}"
                )

            variant_cat = [
                c for c in base.CAT_COLS
                if c in features
            ]

            print(
                f"[{variant_name}] "
                f"PC={len(selected_pc_features):2d} "
                f"total={len(features):2d}"
            )

            (
                variant_pred,
                best_iter,
                elapsed,
            ) = base.fit_catboost_and_predict(
                train_variant,
                val_variant,
                features,
                variant_cat,
            )

            # Deployment candidate:
            # R = variant
            # F = exact control
            routed_pred = control_pred.copy()
            routed_pred[r_mask] = variant_pred[r_mask]

            metric = calc_metrics(
                base,
                y_val,
                routed_pred,
            )

            fold_rows.append({
                "fold": fold,
                "validation_season": val_year,
                "variant": variant_name,
                "removed_block": (
                    BLOCK_LABELS.get(
                        removed_ctx,
                        removed_ctx,
                    )
                    if removed_ctx
                    else ""
                ),
                "n_features": int(len(features)),
                "best_iteration": int(best_iter),
                "elapsed_seconds": float(elapsed),
                "brier": float(metric["brier"]),
                "bss": float(metric["bss"]),
                "R_brier": float(
                    base.brier_score(
                        y_val[r_mask],
                        routed_pred[r_mask],
                    )
                ),
                "R_bss": float(
                    base.calculate_bss(
                        y_val[r_mask],
                        routed_pred[r_mask],
                    )
                ),
                # Routed이므로 F는 CONTROL과 정확히 같아야 함.
                "F_brier": float(
                    base.brier_score(
                        y_val[f_mask],
                        routed_pred[f_mask],
                    )
                ) if f_mask.any() else np.nan,
                "F_bss": float(
                    base.calculate_bss(
                        y_val[f_mask],
                        routed_pred[f_mask],
                    )
                ) if f_mask.any() else np.nan,
            })

            fold_oof[variant_name] = routed_pred

            if f_mask.any():
                if not np.array_equal(
                    routed_pred[f_mask],
                    control_pred[f_mask],
                ):
                    raise RuntimeError(
                        f"{variant_name}: routed F prediction != control"
                    )

        oof_frames.append(fold_oof)

    fold_df = pd.DataFrame(fold_rows)
    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )
    coverage = pd.concat(
        coverage_frames,
        ignore_index=True,
    )

    return (
        fold_df,
        oof,
        coverage,
        blocks,
        variants,
    )


# =============================================================================
# Summaries
# =============================================================================

def make_block_summary(base, fold_df, oof, blocks):
    full_all = routed_metrics(
        base,
        oof,
        "FULL",
    )

    full_recent = routed_metrics(
        base,
        oof.loc[
            oof[base.SEASON_COL].isin([2023, 2024])
        ],
        "FULL",
    )

    rows = []

    for ctx, block_features in blocks.items():
        variant = f"DROP__{ctx}"

        ab_all = routed_metrics(
            base,
            oof,
            variant,
        )

        recent_frame = oof.loc[
            oof[base.SEASON_COL].isin([2023, 2024])
        ]

        ab_recent = routed_metrics(
            base,
            recent_frame,
            variant,
        )

        # R-only comparison.
        r_frame = oof.loc[
            oof[base.GAME_TYPE_COL] != "F"
        ]

        full_r = routed_metrics(
            base,
            r_frame,
            "FULL",
        )

        ab_r = routed_metrics(
            base,
            r_frame,
            variant,
        )

        per_fold = []

        for val_year in [2021, 2022, 2023, 2024]:
            full_row = fold_df.loc[
                (fold_df["validation_season"] == val_year)
                & (fold_df["variant"] == "FULL")
            ]

            ab_row = fold_df.loc[
                (fold_df["validation_season"] == val_year)
                & (fold_df["variant"] == variant)
            ]

            if len(full_row) != 1 or len(ab_row) != 1:
                raise RuntimeError(
                    f"{ctx}/{val_year}: missing fold result"
                )

            full_row = full_row.iloc[0]
            ab_row = ab_row.iloc[0]

            # positive = Full is better = removed block was useful.
            delta_bss = float(
                full_row["bss"] - ab_row["bss"]
            )
            delta_brier = float(
                ab_row["brier"] - full_row["brier"]
            )

            per_fold.append({
                "year": val_year,
                "delta_bss": delta_bss,
                "delta_brier": delta_brier,
            })

        help_wins = int(
            sum(x["delta_bss"] > 0 for x in per_fold)
        )
        removal_wins = int(
            sum(x["delta_bss"] < 0 for x in per_fold)
        )

        pooled_delta_bss = float(
            full_all["bss"] - ab_all["bss"]
        )
        pooled_delta_brier = float(
            ab_all["brier"] - full_all["brier"]
        )

        recent_delta_bss = float(
            full_recent["bss"] - ab_recent["bss"]
        )

        r_delta_bss = float(
            full_r["bss"] - ab_r["bss"]
        )

        delta_2024 = next(
            x["delta_bss"]
            for x in per_fold
            if x["year"] == 2024
        )

        # 판정은 sign + temporal consistency만 사용.
        # 임의의 BSS threshold는 넣지 않는다.
        if (
            help_wins == 4
            and pooled_delta_bss > 0
            and recent_delta_bss > 0
        ):
            verdict = "KEEP_STRONG"
        elif (
            help_wins >= 3
            and pooled_delta_bss > 0
            and delta_2024 > 0
        ):
            verdict = "KEEP"
        elif pooled_delta_bss > 0:
            verdict = "WEAK_KEEP"
        elif (
            removal_wins == 4
            and pooled_delta_bss < 0
            and recent_delta_bss < 0
        ):
            verdict = "REMOVE_STRONG"
        elif (
            removal_wins >= 3
            and pooled_delta_bss < 0
            and delta_2024 < 0
        ):
            verdict = "REMOVE_CANDIDATE"
        else:
            verdict = "MIXED"

        row = {
            "context": ctx,
            "block": BLOCK_LABELS.get(ctx, ctx),
            "n_features": int(len(block_features)),
            "features": " | ".join(block_features),

            "full_pooled_brier": float(full_all["brier"]),
            "ablation_pooled_brier": float(ab_all["brier"]),
            "delta_brier_ablation_minus_full": pooled_delta_brier,

            "full_pooled_bss": float(full_all["bss"]),
            "ablation_pooled_bss": float(ab_all["bss"]),
            "delta_bss_full_minus_ablation": pooled_delta_bss,

            "recent_2023_2024_delta_bss": recent_delta_bss,
            "R_pooled_delta_bss": r_delta_bss,

            "delta_bss_2021": per_fold[0]["delta_bss"],
            "delta_bss_2022": per_fold[1]["delta_bss"],
            "delta_bss_2023": per_fold[2]["delta_bss"],
            "delta_bss_2024": per_fold[3]["delta_bss"],

            "block_help_fold_wins": help_wins,
            "removal_fold_wins": removal_wins,
            "verdict": verdict,
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    # 실제 기여가 큰 block부터.
    summary = summary.sort_values(
        "delta_bss_full_minus_ablation",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def full_vs_control_sanity(base, fold_df, oof):
    control = routed_metrics(
        base,
        oof,
        "CONTROL",
    )
    full = routed_metrics(
        base,
        oof,
        "FULL",
    )

    pooled_delta = float(
        control["brier"] - full["brier"]
    )

    fold_wins = 0

    for year in [2021, 2022, 2023, 2024]:
        c = fold_df.loc[
            (fold_df["validation_season"] == year)
            & (fold_df["variant"] == "CONTROL"),
            "brier",
        ]

        f = fold_df.loc[
            (fold_df["validation_season"] == year)
            & (fold_df["variant"] == "FULL"),
            "brier",
        ]

        if len(c) != 1 or len(f) != 1:
            raise RuntimeError(
                f"Sanity missing fold {year}"
            )

        if float(c.iloc[0]) > float(f.iloc[0]):
            fold_wins += 1

    control_repro_error = abs(
        control["brier"]
        - EXPECTED_CONTROL_POOLED_BRIER
    )

    full_delta_error = abs(
        pooled_delta
        - EXPECTED_FULL_ROUTED_POOLED_DELTA
    )

    return {
        "control_pooled_brier": float(control["brier"]),
        "expected_control_pooled_brier": EXPECTED_CONTROL_POOLED_BRIER,
        "control_repro_abs_error": float(control_repro_error),
        "control_repro_pass": bool(
            control_repro_error <= CONTROL_REPRO_TOL
        ),

        "full_pooled_brier": float(full["brier"]),
        "full_minus_control_delta_brier": pooled_delta,
        "expected_full_minus_control_delta_brier": (
            EXPECTED_FULL_ROUTED_POOLED_DELTA
        ),
        "full_delta_repro_abs_error": float(full_delta_error),
        "full_delta_repro_pass": bool(
            full_delta_error <= FULL_DELTA_REPRO_TOL
        ),

        "full_fold_wins_vs_control": int(fold_wins),
        "expected_full_fold_wins": (
            EXPECTED_FULL_ROUTED_FOLD_WINS
        ),
        "full_fold_win_repro_pass": bool(
            fold_wins == EXPECTED_FULL_ROUTED_FOLD_WINS
        ),

        "control_pooled_bss": float(control["bss"]),
        "full_pooled_bss": float(full["bss"]),
        "full_minus_control_delta_bss": float(
            full["bss"] - control["bss"]
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {VALIDATION_DIR}\n"
            "기존 결과 보존을 위해 폴더를 지우거나 "
            "EXPERIMENT_NAME을 변경하세요."
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    base = import_module_from_path(
        "v021_pc_source",
        SOURCE_PATH,
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
    )

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    required_train = {
        base.ID,
        base.TARGET,
        base.SEASON_COL,
        base.GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    }

    missing_train = sorted(
        required_train - set(train.columns)
    )

    if missing_train:
        raise KeyError(
            f"train required columns missing: {missing_train}"
        )

    base_features = [
        c for c in test_cols
        if c != base.ID
    ]

    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError(
            "canonical v003의 asof_pitcher_pitchmix_n이 없습니다."
        )

    # PC source가 test row-local inference에서도 존재 가능한지 header만 확인.
    required_pc_test = {
        base.SEASON_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
    }

    missing_pc_test = sorted(
        required_pc_test - set(test_cols)
    )

    if missing_pc_test:
        raise KeyError(
            f"test header PC source missing: {missing_pc_test}"
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 110)
    print(EXPERIMENT_NAME)
    print("=" * 110)
    print("Purpose : 7-block Grouped Leave-One-Out Ablation")
    print("Base    : canonical v003 + Exact-ASOF5 + HB-EB")
    print("PC      : previous labeled season only, exact v021 semantics")
    print("Routing : R=PC model, F=Exact Control")
    print("Residual: DISABLED")
    print("Seed    :", base.MODEL_PARAMS["random_seed"])
    print("Fits    : 36 CatBoost models")
    print("=" * 110)

    (
        fold_df,
        oof,
        coverage,
        blocks,
        variants,
    ) = run_validation(
        base=base,
        train=train,
        base_features=base_features,
    )

    summary = make_block_summary(
        base=base,
        fold_df=fold_df,
        oof=oof,
        blocks=blocks,
    )

    sanity = full_vs_control_sanity(
        base=base,
        fold_df=fold_df,
        oof=oof,
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "purpose": (
            "Grouped leave-one-context-out ablation of "
            "R-only PITCHER_CONDITIONAL"
        ),
        "source_script": str(SOURCE_PATH),
        "seed": int(base.MODEL_PARAMS["random_seed"]),
        "residual_enabled": False,
        "routing": "R=variant PC model, F=Exact Control",
        "pc_source_policy": "exactly previous labeled season only",
        "pc_parent_strength": float(base.PC_PARENT_STRENGTH),
        "pc_contexts": [
            {
                "context": ctx,
                "label": BLOCK_LABELS.get(ctx, ctx),
                "strength": float(strength),
                "features": blocks[ctx],
            }
            for ctx, strength in base.PC_CONTEXTS
        ],
        "variants": {
            name: features
            for name, features in variants.items()
        },
        "sanity": sanity,
        "interpretation": {
            "delta_bss_full_minus_ablation_positive": (
                "block helps; removing it lowers BSS"
            ),
            "delta_bss_full_minus_ablation_negative": (
                "removing block improves BSS; removal candidate"
            ),
            "warning": (
                "LOO contributions are not additive because PC blocks "
                "can be correlated/substitutable."
            ),
        },
    }

    fold_df.to_csv(
        FOLD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    oof.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("SANITY: FULL R-ONLY PC vs CONTROL")
    print("=" * 110)
    print(
        f"Control pooled Brier : "
        f"{sanity['control_pooled_brier']:.10f} "
        f"(expected {EXPECTED_CONTROL_POOLED_BRIER:.10f})"
    )
    print(
        f"Full pooled Brier    : "
        f"{sanity['full_pooled_brier']:.10f}"
    )
    print(
        f"Full gain Brier      : "
        f"{sanity['full_minus_control_delta_brier']:+.10f} "
        f"(expected {EXPECTED_FULL_ROUTED_POOLED_DELTA:+.10f})"
    )
    print(
        f"Full gain BSS        : "
        f"{sanity['full_minus_control_delta_bss']:+.2f}"
    )
    print(
        f"Fold wins            : "
        f"{sanity['full_fold_wins_vs_control']}/4"
    )
    print(
        "Reproduction         : "
        f"Control={'PASS' if sanity['control_repro_pass'] else 'FAIL'}, "
        f"FullDelta={'PASS' if sanity['full_delta_repro_pass'] else 'FAIL'}, "
        f"FoldWins={'PASS' if sanity['full_fold_win_repro_pass'] else 'FAIL'}"
    )

    print("\n" + "=" * 110)
    print("PC BLOCK ABLATION SUMMARY")
    print("positive ΔBSS = KEEP signal | negative ΔBSS = removal signal")
    print("=" * 110)

    display_cols = [
        "block",
        "delta_bss_full_minus_ablation",
        "recent_2023_2024_delta_bss",
        "delta_bss_2021",
        "delta_bss_2022",
        "delta_bss_2023",
        "delta_bss_2024",
        "block_help_fold_wins",
        "verdict",
    ]

    print(
        summary[display_cols]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.2f}",
        )
    )

    print("\nSaved:")
    print(f"- {SUMMARY_PATH}")
    print(f"- {FOLD_PATH}")
    print(f"- {OOF_PATH}")
    print(f"- {COVERAGE_PATH}")
    print(f"- {METADATA_PATH}")
    print("=" * 110)


if __name__ == "__main__":
    main()

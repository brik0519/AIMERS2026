#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v025_pitcher_conditional_lean_subset.py
==============================================

목적
----
현재 Feature Engineering 공식 BASE:

    canonical cb_reg_v003 + Exact-ASOF 5 + HB-EB

위에서 R-only PITCHER_CONDITIONAL의 축약 subset을 직접 비교한다.

비교 모델
---------
CONTROL
    cb_reg_v003 + Exact-ASOF 5 + HB-EB

FULL
    CONTROL + R-only PC 7 blocks (21 features)
    Routed: R = FULL PC model, F = CONTROL

LEAN3
    CONTROL + R-only PC 3 blocks
      - count
      - batter_hand
      - li_bucket
    Routed: R = LEAN3 model, F = CONTROL

LEAN4
    CONTROL + R-only PC 4 blocks
      - count
      - batter_hand
      - li_bucket
      - runners
    Routed: R = LEAN4 model, F = CONTROL

핵심 원칙
---------
- Residual Correction 사용하지 않음.
- v021의 PC 생성 로직을 그대로 import하여 사용.
- 각 target season의 PC는 정확히 이전 labeled season만 사용.
- validation/test 같은 시즌의 다른 row/target을 사용하지 않음.
- test.csv는 schema(header) 확인용으로만 읽음.
- seed / CatBoost params / validation 구조는 v021 그대로 유지.

필수 파일
---------
프로젝트 루트에 아래 파일 필요:

    cb_reg_v021_pitcher_conditional_r_only_ablation.py

실행
----
    python cb_reg_v025_pitcher_conditional_lean_subset.py

출력
----
validation/cb_reg_v025_pitcher_conditional_lean_subset/
    fold_results.csv
    oof_predictions.csv
    subset_summary.csv
    pitcher_conditional_coverage.csv
    metadata.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v025_pitcher_conditional_lean_subset"

ROOT = Path(".")
SOURCE_PATH = ROOT / "cb_reg_v021_pitcher_conditional_r_only_ablation.py"

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_DIR = ROOT / "validation" / EXPERIMENT_NAME

FOLD_PATH = VALIDATION_DIR / "fold_results.csv"
OOF_PATH = VALIDATION_DIR / "oof_predictions.csv"
SUMMARY_PATH = VALIDATION_DIR / "subset_summary.csv"
COVERAGE_PATH = VALIDATION_DIR / "pitcher_conditional_coverage.csv"
METADATA_PATH = VALIDATION_DIR / "metadata.json"

# cb_reg_v003_exact pooled reproduction guard
EXPECTED_CONTROL_POOLED_BRIER = 0.2469124012
CONTROL_REPRO_TOL = 2e-7

# 기존 v021 FULL R-only PC reproduction guard
EXPECTED_FULL_ROUTED_POOLED_GAIN = 0.0000479983
FULL_REPRO_TOL = 3e-7

# 현재 확인할 subset
VARIANT_CONTEXTS = {
    "FULL": [
        "v15_count",
        "batter_hand",
        "v15_inning_phase",
        "base_state",
        "v15_pressure",
        "v15_runners",
        "v15_li_bucket",
    ],
    "LEAN3": [
        "v15_count",
        "batter_hand",
        "v15_li_bucket",
    ],
    "LEAN4": [
        "v15_count",
        "batter_hand",
        "v15_li_bucket",
        "v15_runners",
    ],
}

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
# Import v021 exactly
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


def summary_for_prediction(base, frame, pred_col):
    return calc_metrics(
        base,
        frame["y_true"].to_numpy(np.float64),
        frame[pred_col].to_numpy(np.float64),
    )


def get_pc_features_for_contexts(base, contexts):
    context_set = set(contexts)

    features = [
        f"pc__{ctx}__{field}"
        for ctx, _strength in base.PC_CONTEXTS
        if ctx in context_set
        for field in base.PC_FIELDS
    ]

    expected = len(contexts) * len(base.PC_FIELDS)

    if len(features) != expected:
        raise RuntimeError(
            f"context feature count mismatch: "
            f"{len(features)} != {expected}"
        )

    return features


# =============================================================================
# Validation
# =============================================================================

def run_validation(base, train, base_features):
    fold_rows = []
    oof_frames = []
    coverage_frames = []

    variant_features = {
        name: get_pc_features_for_contexts(base, contexts)
        for name, contexts in VARIANT_CONTEXTS.items()
    }

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
        # 1) Exact-ASOF
        # ------------------------------------------------------------------
        pitcher_season_stats = base.build_pitcher_season_stats(
            df_train
        )

        train_exact, _ = base.add_pitcher_exact_asof_features(
            df_train,
            pitcher_season_stats,
        )

        val_exact, _ = base.add_pitcher_exact_asof_features(
            df_val,
            pitcher_season_stats,
        )

        exact_features = list(base_features)

        for col in base.EXACT_FEATURES:
            if col not in exact_features:
                exact_features.append(col)

        # ------------------------------------------------------------------
        # 2) HB-EB
        # ------------------------------------------------------------------
        (
            train_control,
            control_features,
            _,
        ) = base.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )

        (
            val_control,
            val_control_features,
            _,
        ) = base.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=val_exact,
            base_features=exact_features,
        )

        if control_features != val_control_features:
            raise RuntimeError(
                f"Fold {fold}: control train/val feature mismatch"
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
        # 3) PC 21개 생성 후 R-only masking
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

        coverage_frames.extend(
            [cov_train, cov_val]
        )

        # ------------------------------------------------------------------
        # 4) CONTROL
        # ------------------------------------------------------------------
        control_cat = [
            c for c in base.CAT_COLS
            if c in control_features
        ]

        print(
            f"[CONTROL] total_features={len(control_features)}"
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
            "contexts": "",
            "n_pc_features": 0,
            "n_total_features": int(len(control_features)),
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
        # 5) FULL / LEAN3 / LEAN4
        # ------------------------------------------------------------------
        for variant_name in ["FULL", "LEAN3", "LEAN4"]:
            selected_pc_features = variant_features[
                variant_name
            ]

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

            variant_cat = [
                c for c in base.CAT_COLS
                if c in features
            ]

            context_labels = [
                BLOCK_LABELS.get(c, c)
                for c in VARIANT_CONTEXTS[variant_name]
            ]

            print(
                f"[{variant_name}] "
                f"contexts={','.join(context_labels)} | "
                f"PC={len(selected_pc_features)} | "
                f"total={len(features)}"
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

            # R = variant model
            # F = exact CONTROL
            routed_pred = control_pred.copy()
            routed_pred[r_mask] = variant_pred[r_mask]

            if f_mask.any():
                if not np.array_equal(
                    routed_pred[f_mask],
                    control_pred[f_mask],
                ):
                    raise RuntimeError(
                        f"{variant_name}: F routing mismatch"
                    )

            metric = calc_metrics(
                base,
                y_val,
                routed_pred,
            )

            fold_rows.append({
                "fold": fold,
                "validation_season": val_year,
                "variant": variant_name,
                "contexts": " | ".join(context_labels),
                "n_pc_features": int(len(selected_pc_features)),
                "n_total_features": int(len(features)),
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

            print(
                f"  Brier={metric['brier']:.10f} | "
                f"Δ vs CONTROL="
                f"{control_metric['brier'] - metric['brier']:+.10f}"
            )

        oof_frames.append(
            fold_oof
        )

    fold_df = pd.DataFrame(
        fold_rows
    )

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
        variant_features,
    )


# =============================================================================
# Summary
# =============================================================================

def build_summary(base, fold_df, oof):
    control_all = summary_for_prediction(
        base,
        oof,
        "CONTROL",
    )

    recent_frame = oof.loc[
        oof[base.SEASON_COL].isin([2023, 2024])
    ]

    control_recent = summary_for_prediction(
        base,
        recent_frame,
        "CONTROL",
    )

    rows = []

    for variant in ["FULL", "LEAN3", "LEAN4"]:
        all_metric = summary_for_prediction(
            base,
            oof,
            variant,
        )

        recent_metric = summary_for_prediction(
            base,
            recent_frame,
            variant,
        )

        r_frame = oof.loc[
            oof[base.GAME_TYPE_COL] != "F"
        ]

        control_r = summary_for_prediction(
            base,
            r_frame,
            "CONTROL",
        )

        variant_r = summary_for_prediction(
            base,
            r_frame,
            variant,
        )

        per_year = {}

        wins = 0

        for year in [2021, 2022, 2023, 2024]:
            c_row = fold_df.loc[
                (fold_df["validation_season"] == year)
                & (fold_df["variant"] == "CONTROL")
            ].iloc[0]

            v_row = fold_df.loc[
                (fold_df["validation_season"] == year)
                & (fold_df["variant"] == variant)
            ].iloc[0]

            gain_brier = float(
                c_row["brier"]
                - v_row["brier"]
            )

            gain_bss = float(
                v_row["bss"]
                - c_row["bss"]
            )

            if gain_brier > 0:
                wins += 1

            per_year[year] = {
                "gain_brier": gain_brier,
                "gain_bss": gain_bss,
            }

        pooled_gain = float(
            control_all["brier"]
            - all_metric["brier"]
        )

        pooled_bss_gain = float(
            all_metric["bss"]
            - control_all["bss"]
        )

        recent_gain = float(
            control_recent["brier"]
            - recent_metric["brier"]
        )

        recent_bss_gain = float(
            recent_metric["bss"]
            - control_recent["bss"]
        )

        r_gain = float(
            control_r["brier"]
            - variant_r["brier"]
        )

        if (
            pooled_gain > 0
            and recent_gain > 0
            and per_year[2024]["gain_brier"] > 0
            and wins == 4
        ):
            verdict = "STRONG_PASS"
        elif (
            pooled_gain > 0
            and recent_gain > 0
            and per_year[2024]["gain_brier"] > 0
            and wins >= 3
        ):
            verdict = "PASS"
        elif pooled_gain > 0:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        rows.append({
            "variant": variant,
            "contexts": " | ".join(
                BLOCK_LABELS.get(c, c)
                for c in VARIANT_CONTEXTS[variant]
            ),
            "pooled_brier": float(all_metric["brier"]),
            "pooled_gain_vs_control": pooled_gain,
            "pooled_bss": float(all_metric["bss"]),
            "pooled_bss_gain_vs_control": pooled_bss_gain,
            "recent_2023_2024_brier": float(recent_metric["brier"]),
            "recent_2023_2024_gain_vs_control": recent_gain,
            "recent_2023_2024_bss_gain_vs_control": recent_bss_gain,
            "R_pooled_gain_vs_control": r_gain,
            "gain_2021": per_year[2021]["gain_brier"],
            "gain_2022": per_year[2022]["gain_brier"],
            "gain_2023": per_year[2023]["gain_brier"],
            "gain_2024": per_year[2024]["gain_brier"],
            "bss_gain_2021": per_year[2021]["gain_bss"],
            "bss_gain_2022": per_year[2022]["gain_bss"],
            "bss_gain_2023": per_year[2023]["gain_bss"],
            "bss_gain_2024": per_year[2024]["gain_bss"],
            "fold_wins": int(wins),
            "verdict": verdict,
        })

    summary = pd.DataFrame(
        rows
    ).sort_values(
        "pooled_brier",
        ascending=True,
    ).reset_index(drop=True)

    return (
        summary,
        control_all,
        control_recent,
    )


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
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

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
        required_train
        - set(train.columns)
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
            "canonical v003 feature "
            "asof_pitcher_pitchmix_n missing"
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 110)
    print(EXPERIMENT_NAME)
    print("=" * 110)
    print("BASE    : canonical cb_reg_v003 + Exact-ASOF5 + HB-EB")
    print("Compare : CONTROL vs FULL vs LEAN3 vs LEAN4")
    print("LEAN3   : count + batter_hand + li_bucket")
    print("LEAN4   : count + batter_hand + li_bucket + runners")
    print("Routing : R=PC variant, F=CONTROL")
    print("Residual: DISABLED")
    print("Seed    :", base.MODEL_PARAMS["random_seed"])
    print("Fits    : 4 folds × 4 models = 16 CatBoost fits")
    print("=" * 110)

    (
        fold_df,
        oof,
        coverage,
        variant_features,
    ) = run_validation(
        base,
        train,
        base_features,
    )

    (
        summary,
        control_all,
        control_recent,
    ) = build_summary(
        base,
        fold_df,
        oof,
    )

    # ---------------------------------------------------------------------
    # Reproduction checks
    # ---------------------------------------------------------------------
    control_error = abs(
        float(control_all["brier"])
        - EXPECTED_CONTROL_POOLED_BRIER
    )

    control_pass = (
        control_error
        <= CONTROL_REPRO_TOL
    )

    full_row = summary.loc[
        summary["variant"] == "FULL"
    ].iloc[0]

    full_gain = float(
        full_row["pooled_gain_vs_control"]
    )

    full_error = abs(
        full_gain
        - EXPECTED_FULL_ROUTED_POOLED_GAIN
    )

    full_pass = (
        full_error
        <= FULL_REPRO_TOL
    )

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
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

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "base": (
            "canonical cb_reg_v003 + "
            "Exact-ASOF 5 + HB-EB"
        ),
        "residual_enabled": False,
        "routing": (
            "R=variant PC model, "
            "F=Exact Control"
        ),
        "pc_source_policy": (
            "exactly previous labeled season only"
        ),
        "seed": int(
            base.MODEL_PARAMS["random_seed"]
        ),
        "variants": {
            name: {
                "contexts": VARIANT_CONTEXTS[name],
                "labels": [
                    BLOCK_LABELS.get(c, c)
                    for c in VARIANT_CONTEXTS[name]
                ],
                "features": variant_features[name],
            }
            for name in variant_features
        },
        "control_reproduction": {
            "expected": EXPECTED_CONTROL_POOLED_BRIER,
            "actual": float(control_all["brier"]),
            "abs_error": float(control_error),
            "pass": bool(control_pass),
        },
        "full_reproduction": {
            "expected_gain": EXPECTED_FULL_ROUTED_POOLED_GAIN,
            "actual_gain": float(full_gain),
            "abs_error": float(full_error),
            "pass": bool(full_pass),
        },
        "summary": summary.to_dict(
            orient="records"
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Print
    # ---------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("REPRODUCTION CHECK")
    print("=" * 110)
    print(
        f"CONTROL pooled Brier : "
        f"{control_all['brier']:.10f} "
        f"(expected {EXPECTED_CONTROL_POOLED_BRIER:.10f}) "
        f"=> {'PASS' if control_pass else 'FAIL'}"
    )
    print(
        f"FULL pooled gain     : "
        f"{full_gain:+.10f} "
        f"(expected {EXPECTED_FULL_ROUTED_POOLED_GAIN:+.10f}) "
        f"=> {'PASS' if full_pass else 'FAIL'}"
    )

    print("\n" + "=" * 110)
    print("LEAN PC SUBSET SUMMARY")
    print("positive gain = better than CONTROL")
    print("=" * 110)

    display_cols = [
        "variant",
        "pooled_brier",
        "pooled_gain_vs_control",
        "pooled_bss_gain_vs_control",
        "recent_2023_2024_gain_vs_control",
        "gain_2021",
        "gain_2022",
        "gain_2023",
        "gain_2024",
        "fold_wins",
        "verdict",
    ]

    print(
        summary[display_cols]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.10f}",
        )
    )

    print("\n" + "=" * 110)
    print("DECISION RULE")
    print("=" * 110)
    print(
        "1) LEAN3/LEAN4가 CONTROL보다 pooled 개선"
    )
    print(
        "2) recent 2023-24도 개선"
    )
    print(
        "3) 2024 개선"
    )
    print(
        "4) 최소 3/4 fold win"
    )
    print(
        "→ 만족하면 PC 후보 유지 후 multi-seed"
    )
    print(
        "→ 둘 다 미달이면 PC Feature 실험 종료"
    )

    print("\nSaved:")
    print(f"- {SUMMARY_PATH}")
    print(f"- {FOLD_PATH}")
    print(f"- {OOF_PATH}")
    print(f"- {COVERAGE_PATH}")
    print(f"- {METADATA_PATH}")
    print("=" * 110)

    if not control_pass:
        raise RuntimeError(
            "CONTROL reproduction failed. "
            "Feature 결과를 해석하지 마세요."
        )

    if not full_pass:
        raise RuntimeError(
            "FULL PC reproduction failed. "
            "v021/v024 실험 조건과 일치하지 않습니다."
        )


if __name__ == "__main__":
    main()

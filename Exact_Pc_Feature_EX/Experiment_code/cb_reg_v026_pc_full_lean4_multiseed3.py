#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v026_pc_full_lean4_multiseed3.py
=======================================

목적
----
현재 공식 Feature Engineering BASE:

    canonical cb_reg_v003 + Exact-ASOF 5 + HB-EB

위에서 유망한 두 PC 후보를 3-seed paired validation으로 검증한다.

비교
----
CONTROL
    cb_reg_v003 + Exact-ASOF5 + HB-EB

FULL
    CONTROL + R-only PC 7 blocks (21 features)
    Routed: R=FULL model, F=CONTROL

LEAN4
    CONTROL + R-only PC 4 blocks (12 features)
      - count
      - batter_hand
      - runners
      - li_bucket
    Routed: R=LEAN4 model, F=CONTROL

Seeds
-----
7, 42, 2026

원칙
----
- 같은 seed의 CONTROL과 treatment를 paired 비교한다.
- Residual correction 사용하지 않는다.
- v021의 PC 생성 / R-only masking / CatBoost 설정을 그대로 사용한다.
- target season의 PC는 정확히 이전 labeled season만 사용한다.
- test.csv는 header(schema) 확인용으로만 사용한다.

필수 파일
---------
프로젝트 루트:
    cb_reg_v021_pitcher_conditional_r_only_ablation.py

실행
----
    python cb_reg_v026_pc_full_lean4_multiseed3.py

출력
----
validation/cb_reg_v026_pc_full_lean4_multiseed3/
    fold_results.csv
    seed_summary.csv
    multiseed_summary.csv
    oof_predictions.csv
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

EXPERIMENT_NAME = "cb_reg_v026_pc_full_lean4_multiseed3"

ROOT = Path(".")
SOURCE_PATH = ROOT / "cb_reg_v021_pitcher_conditional_r_only_ablation.py"

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_DIR = ROOT / "validation" / EXPERIMENT_NAME
FOLD_PATH = VALIDATION_DIR / "fold_results.csv"
SEED_SUMMARY_PATH = VALIDATION_DIR / "seed_summary.csv"
MULTISEED_SUMMARY_PATH = VALIDATION_DIR / "multiseed_summary.csv"
OOF_PATH = VALIDATION_DIR / "oof_predictions.csv"
METADATA_PATH = VALIDATION_DIR / "metadata.json"

SEEDS = [7, 42, 2026]

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
    "LEAN4": [
        "v15_count",
        "batter_hand",
        "v15_runners",
        "v15_li_bucket",
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
# Import source
# =============================================================================

def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"필수 파일 없음: {path}\n"
            "cb_reg_v021_pitcher_conditional_r_only_ablation.py를 "
            "프로젝트 루트에 두세요."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Helpers
# =============================================================================

def pc_features_for_contexts(base, contexts):
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
            f"PC feature count mismatch: {len(features)} != {expected}"
        )
    return features


def brier(base, y, p):
    return float(
        base.brier_score(
            np.asarray(y, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
        )
    )


def bss(base, y, p):
    return float(
        base.calculate_bss(
            np.asarray(y, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
        )
    )


# =============================================================================
# Run one seed
# =============================================================================

def run_one_seed(base, train, base_features, seed, variant_features):
    # v021 fit function이 MODEL_PARAMS를 참조하므로 seed만 변경.
    base.MODEL_PARAMS["random_seed"] = int(seed)

    fold_rows = []
    oof_frames = []

    for cfg in base.EXPANDING_FOLDS:
        fold = int(cfg["fold"])
        train_seasons = list(cfg["train_seasons"])
        val_year = int(cfg["validation_season"])

        print("\n" + "=" * 110)
        print(f"SEED {seed} | Fold {fold}: {train_seasons} -> {val_year}")
        print("=" * 110)

        df_train = train.loc[
            train[base.SEASON_COL].isin(train_seasons)
        ].copy()

        df_val = train.loc[
            train[base.SEASON_COL].eq(val_year)
        ].copy()

        # ------------------------------------------------------------------
        # Exact-ASOF
        # ------------------------------------------------------------------
        pitcher_stats = base.build_pitcher_season_stats(df_train)

        train_exact, _ = base.add_pitcher_exact_asof_features(
            df_train,
            pitcher_stats,
        )
        val_exact, _ = base.add_pitcher_exact_asof_features(
            df_val,
            pitcher_stats,
        )

        exact_features = list(base_features)
        for col in base.EXACT_FEATURES:
            if col not in exact_features:
                exact_features.append(col)

        # ------------------------------------------------------------------
        # HB-EB
        # ------------------------------------------------------------------
        train_control, control_features, _ = base.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )
        val_control, val_control_features, _ = base.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=val_exact,
            base_features=exact_features,
        )

        if control_features != val_control_features:
            raise RuntimeError(
                f"seed={seed}, fold={fold}: control feature mismatch"
            )

        if "asof_pitcher_pitchmix_n" not in control_features:
            raise RuntimeError("canonical pitchmix_n missing")

        # ------------------------------------------------------------------
        # PC 생성 / R-only mask
        # ------------------------------------------------------------------
        pc_train, _ = base.build_pitcher_conditional_for_rows(
            rows=df_train,
            available_history=df_train,
            scope="train",
        )
        pc_val, _ = base.build_pitcher_conditional_for_rows(
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

        y_val = df_val[base.TARGET].to_numpy(np.float64)

        game_type = (
            df_val[base.GAME_TYPE_COL]
            .fillna("__MISSING__")
            .astype(str)
            .to_numpy()
        )
        r_mask = game_type != "F"
        f_mask = game_type == "F"

        # ------------------------------------------------------------------
        # CONTROL
        # ------------------------------------------------------------------
        control_cat = [
            c for c in base.CAT_COLS
            if c in control_features
        ]

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

        fold_rows.append({
            "seed": int(seed),
            "fold": fold,
            "validation_season": val_year,
            "variant": "CONTROL",
            "brier": brier(base, y_val, control_pred),
            "bss": bss(base, y_val, control_pred),
            "best_iteration": int(control_iter),
            "elapsed_seconds": float(control_time),
        })

        fold_oof = pd.DataFrame({
            base.ID: df_val[base.ID].to_numpy(),
            base.SEASON_COL: val_year,
            base.GAME_TYPE_COL: game_type,
            "seed": int(seed),
            "y_true": y_val,
            "CONTROL": control_pred,
        })

        # ------------------------------------------------------------------
        # FULL / LEAN4
        # ------------------------------------------------------------------
        for variant in ["FULL", "LEAN4"]:
            selected_pc = variant_features[variant]

            train_variant = train_control.copy()
            val_variant = val_control.copy()

            for col in selected_pc:
                train_variant[col] = pc_train[col]
                val_variant[col] = pc_val[col]

            features = list(control_features) + list(selected_pc)

            variant_cat = [
                c for c in base.CAT_COLS
                if c in features
            ]

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

            # deployment routing:
            # R = PC model
            # F = exact CONTROL
            routed_pred = control_pred.copy()
            routed_pred[r_mask] = variant_pred[r_mask]

            if f_mask.any() and not np.array_equal(
                routed_pred[f_mask],
                control_pred[f_mask],
            ):
                raise RuntimeError(
                    f"seed={seed}, fold={fold}, {variant}: F routing mismatch"
                )

            vb = brier(base, y_val, routed_pred)
            cb = brier(base, y_val, control_pred)

            fold_rows.append({
                "seed": int(seed),
                "fold": fold,
                "validation_season": val_year,
                "variant": variant,
                "brier": vb,
                "bss": bss(base, y_val, routed_pred),
                "gain_brier_vs_control": float(cb - vb),
                "best_iteration": int(best_iter),
                "elapsed_seconds": float(elapsed),
            })

            fold_oof[variant] = routed_pred

            print(
                f"[{variant}] "
                f"CONTROL={cb:.10f} | "
                f"{variant}={vb:.10f} | "
                f"gain={cb - vb:+.10f}"
            )

        oof_frames.append(fold_oof)

    return (
        pd.DataFrame(fold_rows),
        pd.concat(oof_frames, ignore_index=True),
    )


# =============================================================================
# Seed summaries
# =============================================================================

def summarize_seed(base, oof, seed):
    frame = oof.loc[oof["seed"] == int(seed)].copy()

    y = frame["y_true"].to_numpy(np.float64)
    c = frame["CONTROL"].to_numpy(np.float64)

    recent_mask = frame[base.SEASON_COL].isin([2023, 2024]).to_numpy()
    mask_2024 = frame[base.SEASON_COL].eq(2024).to_numpy()

    rows = []

    for variant in ["FULL", "LEAN4"]:
        p = frame[variant].to_numpy(np.float64)

        pooled_gain = brier(base, y, c) - brier(base, y, p)

        recent_gain = (
            brier(base, y[recent_mask], c[recent_mask])
            - brier(base, y[recent_mask], p[recent_mask])
        )

        gain_2024 = (
            brier(base, y[mask_2024], c[mask_2024])
            - brier(base, y[mask_2024], p[mask_2024])
        )

        fold_gains = {}
        fold_wins = 0

        for year in [2021, 2022, 2023, 2024]:
            m = frame[base.SEASON_COL].eq(year).to_numpy()
            gain = (
                brier(base, y[m], c[m])
                - brier(base, y[m], p[m])
            )
            fold_gains[year] = float(gain)
            if gain > 0:
                fold_wins += 1

        rows.append({
            "seed": int(seed),
            "variant": variant,
            "pooled_gain": float(pooled_gain),
            "recent_2023_2024_gain": float(recent_gain),
            "gain_2024": float(gain_2024),
            "gain_2021": fold_gains[2021],
            "gain_2022": fold_gains[2022],
            "gain_2023": fold_gains[2023],
            "gain_2024_check": fold_gains[2024],
            "fold_wins": int(fold_wins),
            "seed_win": bool(pooled_gain > 0),
        })

    return pd.DataFrame(rows)


def summarize_multiseed(seed_summary):
    rows = []

    for variant in ["FULL", "LEAN4"]:
        part = seed_summary.loc[
            seed_summary["variant"] == variant
        ].copy()

        pooled = part["pooled_gain"].to_numpy(np.float64)
        recent = part["recent_2023_2024_gain"].to_numpy(np.float64)
        latest = part["gain_2024"].to_numpy(np.float64)

        seed_wins = int((pooled > 0).sum())

        mean_gain = float(np.mean(pooled))
        std_gain = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0

        mean_recent = float(np.mean(recent))
        mean_2024 = float(np.mean(latest))

        # 3-seed screening rule
        if (
            seed_wins == 3
            and mean_gain > 0
            and mean_recent > 0
            and mean_2024 > 0
        ):
            verdict = "STRONG_PASS"
        elif (
            seed_wins >= 2
            and mean_gain > 0
            and mean_recent > 0
            and mean_2024 > 0
        ):
            verdict = "PASS"
        elif mean_gain > 0:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        rows.append({
            "variant": variant,
            "seed_wins": seed_wins,
            "mean_pooled_gain": mean_gain,
            "std_pooled_gain": std_gain,
            "gain_to_std_ratio": (
                float(mean_gain / std_gain)
                if std_gain > 0
                else np.inf
            ),
            "mean_recent_2023_2024_gain": mean_recent,
            "mean_2024_gain": mean_2024,
            "min_pooled_gain": float(np.min(pooled)),
            "max_pooled_gain": float(np.max(pooled)),
            "mean_fold_wins": float(part["fold_wins"].mean()),
            "verdict": verdict,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_pooled_gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {VALIDATION_DIR}\n"
            "기존 결과 보존을 위해 폴더를 삭제하거나 "
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

    base_features = [
        c for c in test_cols
        if c != base.ID
    ]

    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError(
            "canonical v003 feature asof_pitcher_pitchmix_n missing"
        )

    variant_features = {
        name: pc_features_for_contexts(base, contexts)
        for name, contexts in VARIANT_CONTEXTS.items()
    }

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 110)
    print(EXPERIMENT_NAME)
    print("=" * 110)
    print("BASE     : cb_reg_v003 + Exact-ASOF5 + HB-EB")
    print("Variants : FULL, LEAN4")
    print("Seeds    :", SEEDS)
    print("Routing  : R=variant PC model, F=CONTROL")
    print("Residual : DISABLED")
    print(
        "Fits     : "
        f"{len(SEEDS)} seeds × 4 folds × 3 models = "
        f"{len(SEEDS) * 4 * 3}"
    )
    print("=" * 110)

    fold_frames = []
    oof_frames = []
    seed_summary_frames = []

    original_seed = int(
        base.MODEL_PARAMS["random_seed"]
    )

    try:
        for seed in SEEDS:
            fold_df, oof = run_one_seed(
                base,
                train,
                base_features,
                seed,
                variant_features,
            )

            fold_frames.append(fold_df)
            oof_frames.append(oof)

            seed_summary_frames.append(
                summarize_seed(
                    base,
                    oof,
                    seed,
                )
            )
    finally:
        base.MODEL_PARAMS["random_seed"] = original_seed

    fold_all = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    oof_all = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    seed_summary = pd.concat(
        seed_summary_frames,
        ignore_index=True,
    )

    multiseed_summary = summarize_multiseed(
        seed_summary
    )

    fold_all.to_csv(
        FOLD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    oof_all.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary.to_csv(
        SEED_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    multiseed_summary.to_csv(
        MULTISEED_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "seeds": SEEDS,
        "base": "canonical cb_reg_v003 + Exact-ASOF5 + HB-EB",
        "variants": {
            name: {
                "contexts": contexts,
                "labels": [
                    BLOCK_LABELS.get(c, c)
                    for c in contexts
                ],
                "features": variant_features[name],
            }
            for name, contexts in VARIANT_CONTEXTS.items()
        },
        "residual_enabled": False,
        "routing": "R=variant PC model, F=Exact CONTROL",
        "comparison_policy": "paired within identical random seed",
        "seed_summary": seed_summary.to_dict(orient="records"),
        "multiseed_summary": multiseed_summary.to_dict(orient="records"),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("3-SEED DETAIL")
    print("=" * 110)

    print(
        seed_summary[
            [
                "seed",
                "variant",
                "pooled_gain",
                "recent_2023_2024_gain",
                "gain_2024",
                "fold_wins",
                "seed_win",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.10f}",
        )
    )

    print("\n" + "=" * 110)
    print("3-SEED FINAL SUMMARY")
    print("positive gain = better than same-seed CONTROL")
    print("=" * 110)

    print(
        multiseed_summary[
            [
                "variant",
                "seed_wins",
                "mean_pooled_gain",
                "std_pooled_gain",
                "gain_to_std_ratio",
                "mean_recent_2023_2024_gain",
                "mean_2024_gain",
                "mean_fold_wins",
                "verdict",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.10f}",
        )
    )

    print("\nDecision:")
    print("- 3/3 wins + pooled/recent/2024 mean positive => STRONG_PASS")
    print("- 2/3 wins + pooled/recent/2024 mean positive => PASS")
    print("- mean pooled only positive => WEAK")
    print("- otherwise => FAIL")
    print()
    print("Saved:")
    print(f"- {FOLD_PATH}")
    print(f"- {SEED_SUMMARY_PATH}")
    print(f"- {MULTISEED_SUMMARY_PATH}")
    print(f"- {OOF_PATH}")
    print(f"- {METADATA_PATH}")
    print("=" * 110)


if __name__ == "__main__":
    main()

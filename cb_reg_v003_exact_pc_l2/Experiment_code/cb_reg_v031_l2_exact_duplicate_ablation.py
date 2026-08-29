#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v031_l2_exact_duplicate_ablation.py
==========================================

목적
----
현재 L2 champion D0에서 Exact-ASOF5를 L2에도 다시 넣는 것이
실제로 필요한지 직접 검증한다.

현재 D0 (기존 v028, 재학습 없음)
---------------------------------
L1:
    CatBoost + Exact-ASOF5 + R-only PC FULL

L2:
    ExtraTrees residual adapter
    + reference-style adapter core
    + Exact-ASOF5
    - PC

새 후보 E0
----------
L1:
    동일 (CatBoost + Exact-ASOF5 + R-only PC FULL)

L2:
    ExtraTrees residual adapter
    + reference-style adapter core
    - Exact-ASOF5
    - PC

핵심 비교
---------
E0 - D0

양수:
    L1이 이미 Exact를 사용하므로
    L2에서는 Exact 중복을 제거하는 것이 더 좋음

음수:
    Exact는 L1과 L2 양쪽에서 사용하는 것이 더 좋음

중요
----
- CatBoost 재학습 0회
- D0 ExtraTrees 재학습 0회
- 기존 v028 OOF를 그대로 재사용
- 새로 학습하는 것은 E0 ExtraTrees만
- residual target은 D0/E0 동일:
      y - B_L1_PC_OOF
- PC는 두 구조 모두 L2에서 제외
- routing도 동일:
      R = L2 corrected
      F = L1 prediction 보호
- 즉 바뀌는 것은 "L2 입력에서 Exact-ASOF5 5개를 빼는 것" 딱 하나

필수 파일
---------
./data/train.csv

./validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/
    oof_predictions.csv

프로젝트 루트:
    cb_reg_v027_pc_placement_l1_vs_l2.py

출력
----
./validation/cb_reg_v031_l2_exact_duplicate_ablation/
    fold_results.csv
    seed_summary.csv
    final_summary.csv
    oof_predictions.csv
    metadata.json

실행
----
python cb_reg_v031_l2_exact_duplicate_ablation.py
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

EXPERIMENT_NAME = "cb_reg_v031_l2_exact_duplicate_ablation"

ROOT = Path(".")

TRAIN_PATH = ROOT / "data" / "train.csv"

V028_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
    / "oof_predictions.csv"
)

V027_SOURCE_CANDIDATES = [
    ROOT / "cb_reg_v027_pc_placement_l1_vs_l2.py",
    ROOT / "cb_reg_v027_pc_placement_l1_vs_l2(1).py",
]

OUT_DIR = ROOT / "validation" / EXPERIMENT_NAME

FOLD_PATH = OUT_DIR / "fold_results.csv"
SEED_SUMMARY_PATH = OUT_DIR / "seed_summary.csv"
FINAL_SUMMARY_PATH = OUT_DIR / "final_summary.csv"
OOF_PATH = OUT_DIR / "oof_predictions.csv"
METADATA_PATH = OUT_DIR / "metadata.json"

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"

REQUESTED_BASE_SEEDS = [7, 42, 2026]


# =============================================================================
# Common
# =============================================================================

def find_v027_source():
    for path in V027_SOURCE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "프로젝트 루트에 "
        "cb_reg_v027_pc_placement_l1_vs_l2.py가 필요합니다."
    )


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def brier(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def bss(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    score = brier(
        y,
        p,
    )

    mu = float(
        y.mean()
    )

    reference = max(
        mu * (1.0 - mu),
        1e-12,
    )

    return float(
        100000.0
        * (
            1.0
            - score / reference
        )
    )


# =============================================================================
# Inputs
# =============================================================================

def load_inputs():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not V028_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{V028_OOF_PATH}\n"
            "먼저 v028 결과가 필요합니다."
        )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    train[SEASON] = (
        pd.to_numeric(
            train[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    oof = pd.read_csv(
        V028_OOF_PATH,
        encoding="utf-8-sig",
    )

    required = {
        ID,
        SEASON,
        GAME_TYPE,
        "base_seed",
        "y_true",
        "B_L1_PC",
        "D_L1PC_L2",
    }

    missing = sorted(
        required
        - set(
            oof.columns
        )
    )

    if missing:
        raise KeyError(
            f"v028 OOF missing: {missing}"
        )

    oof[SEASON] = (
        pd.to_numeric(
            oof[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    oof["base_seed"] = (
        pd.to_numeric(
            oof["base_seed"],
            errors="raise",
        )
        .astype(int)
    )

    available = sorted(
        oof[
            "base_seed"
        ]
        .unique()
        .tolist()
    )

    seeds = [
        seed
        for seed in REQUESTED_BASE_SEEDS
        if seed in available
    ]

    if not seeds:
        raise RuntimeError(
            f"requested={REQUESTED_BASE_SEEDS}, "
            f"available={available}"
        )

    oof = (
        oof.loc[
            oof[
                "base_seed"
            ]
            .isin(
                seeds
            )
        ]
        .copy()
    )

    return (
        train,
        oof,
        seeds,
    )


# =============================================================================
# One fold
# =============================================================================

def evaluate_fold(
    v027,
    train,
    part,
    base_seed,
    target_year,
):
    part = (
        part.copy()
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------------------
    # Recover raw official train rows corresponding to OOF rows
    # -----------------------------------------------------------------

    raw_cols = [
        c
        for c in train.columns
        if c != TARGET
    ]

    raw = (
        part[
            [ID]
        ]
        .merge(
            train[
                raw_cols
            ],
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if (
        raw[SEASON]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "OOF row missing from train"
        )

    raw[SEASON] = (
        pd.to_numeric(
            raw[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    if not (
        raw[SEASON]
        .eq(
            int(target_year)
        )
        .all()
    ):
        raise RuntimeError(
            "OOF/train season mismatch"
        )

    y = (
        part[
            "y_true"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # -----------------------------------------------------------------
    # B = frozen L1
    # CatBoost + Exact-ASOF5 + PC FULL
    # -----------------------------------------------------------------

    l1_pc = (
        part[
            "B_L1_PC"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # -----------------------------------------------------------------
    # D0 = frozen v028 L2 champion candidate
    # core + Exact-ASOF5
    # -----------------------------------------------------------------

    d0 = (
        part[
            "D_L1PC_L2"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # -----------------------------------------------------------------
    # Build exactly the same adapter core as v028.
    #
    # The ONLY change:
    # v028 X = core + Exact5
    # v031 X = core
    # -----------------------------------------------------------------

    source_year = (
        int(target_year)
        - 1
    )

    map_rows = (
        train.loc[
            train[SEASON]
            .eq(
                source_year
            )
        ]
        .copy()
    )

    if len(map_rows) == 0:
        raise RuntimeError(
            f"No source season {source_year}"
        )

    prep = (
        v027.fit_adapter_prep(
            map_rows
        )
    )

    core = (
        v027.transform_adapter_core(
            prep=prep,
            rows=raw,
            base_pred=l1_pc,
        )
    )

    # E0:
    # Exact-ASOF5 deliberately NOT concatenated here.
    X_e0 = (
        core
        .reset_index(
            drop=True
        )
        .copy()
    )

    if (
        X_e0.columns
        .duplicated()
        .any()
    ):
        duplicated = (
            X_e0.columns[
                X_e0.columns
                .duplicated()
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate E0 columns: {duplicated}"
        )

    values = (
        X_e0
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "E0 feature matrix contains NaN/Inf"
        )

    # -----------------------------------------------------------------
    # E0 ExtraTrees
    #
    # residual target:
    # y - B_L1_PC_OOF
    #
    # Same function/config/shrink/routing as v028.
    # -----------------------------------------------------------------

    (
        e0,
        adapter_seed_diag,
    ) = (
        v027.crossfit_l2_variant(
            X=X_e0,
            y=y,
            base_pred=l1_pc,
            frame=raw,
        )
    )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    b_brier = brier(
        y,
        l1_pc,
    )

    d0_brier = brier(
        y,
        d0,
    )

    e0_brier = brier(
        y,
        e0,
    )

    b_bss = bss(
        y,
        l1_pc,
    )

    d0_bss = bss(
        y,
        d0,
    )

    e0_bss = bss(
        y,
        e0,
    )

    row = {
        "base_seed": int(
            base_seed
        ),

        "target_year": int(
            target_year
        ),

        "n": int(
            len(part)
        ),

        "B_l1_pc_brier": (
            b_brier
        ),

        "B_l1_pc_bss": (
            b_bss
        ),

        "D0_l2_with_exact_brier": (
            d0_brier
        ),

        "D0_l2_with_exact_bss": (
            d0_bss
        ),

        "E0_l2_without_exact_brier": (
            e0_brier
        ),

        "E0_l2_without_exact_bss": (
            e0_bss
        ),

        # L2 total effect vs L1
        "D0_minus_B_bss": float(
            d0_bss
            - b_bss
        ),

        "E0_minus_B_bss": float(
            e0_bss
            - b_bss
        ),

        # PRIMARY:
        # positive => removing Exact from L2 is better
        "E0_minus_D0_bss": float(
            e0_bss
            - d0_bss
        ),

        "D0_minus_E0_brier": float(
            d0_brier
            - e0_brier
        ),

        "n_e0_features": int(
            X_e0.shape[1]
        ),

        "adapter_seed_mean_bss": float(
            adapter_seed_diag[
                "bss"
            ]
            .mean()
        ),

        "adapter_seed_std_bss": float(
            adapter_seed_diag[
                "bss"
            ]
            .std(
                ddof=1
            )
        ),
    }

    pred = pd.DataFrame({
        ID: (
            part[ID]
            .to_numpy()
        ),

        SEASON: int(
            target_year
        ),

        GAME_TYPE: (
            raw[
                GAME_TYPE
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .to_numpy()
        ),

        "base_seed": int(
            base_seed
        ),

        "y_true": y,

        "B_L1_PC": (
            l1_pc
        ),

        "D0_L2_WITH_EXACT": (
            d0
        ),

        "E0_L2_WITHOUT_EXACT": (
            e0
        ),
    })

    return (
        row,
        pred,
    )


# =============================================================================
# Seed summary
# =============================================================================

def summarize_seed(
    pred_all,
    seed,
):
    part = (
        pred_all.loc[
            pred_all[
                "base_seed"
            ]
            .eq(
                int(seed)
            )
        ]
        .copy()
    )

    y = (
        part[
            "y_true"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    b = (
        part[
            "B_L1_PC"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    d0 = (
        part[
            "D0_L2_WITH_EXACT"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    e0 = (
        part[
            "E0_L2_WITHOUT_EXACT"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    recent = (
        part[SEASON]
        .isin(
            [2023, 2024]
        )
        .to_numpy()
    )

    latest = (
        part[SEASON]
        .eq(
            2024
        )
        .to_numpy()
    )

    fold_gains = {}
    fold_wins = 0

    for year in [
        2021,
        2022,
        2023,
        2024,
    ]:
        mask = (
            part[SEASON]
            .eq(
                year
            )
            .to_numpy()
        )

        gain = (
            bss(
                y[mask],
                e0[mask],
            )
            - bss(
                y[mask],
                d0[mask],
            )
        )

        fold_gains[
            year
        ] = float(
            gain
        )

        if gain > 0:
            fold_wins += 1

    d0_gain_vs_b = (
        bss(
            y,
            d0,
        )
        - bss(
            y,
            b,
        )
    )

    e0_gain_vs_b = (
        bss(
            y,
            e0,
        )
        - bss(
            y,
            b,
        )
    )

    e0_minus_d0 = (
        bss(
            y,
            e0,
        )
        - bss(
            y,
            d0,
        )
    )

    recent_gain = (
        bss(
            y[recent],
            e0[recent],
        )
        - bss(
            y[recent],
            d0[recent],
        )
    )

    latest_gain = (
        bss(
            y[latest],
            e0[latest],
        )
        - bss(
            y[latest],
            d0[latest],
        )
    )

    return {
        "base_seed": int(
            seed
        ),

        "D0_gain_vs_B_bss": float(
            d0_gain_vs_b
        ),

        "E0_gain_vs_B_bss": float(
            e0_gain_vs_b
        ),

        # Primary
        "E0_minus_D0_bss": float(
            e0_minus_d0
        ),

        "recent_2023_2024_E0_minus_D0_bss": float(
            recent_gain
        ),

        "gain_2024_E0_minus_D0_bss": float(
            latest_gain
        ),

        "gain_2021_bss": (
            fold_gains[
                2021
            ]
        ),

        "gain_2022_bss": (
            fold_gains[
                2022
            ]
        ),

        "gain_2023_bss": (
            fold_gains[
                2023
            ]
        ),

        "gain_2024_bss": (
            fold_gains[
                2024
            ]
        ),

        "fold_wins": int(
            fold_wins
        ),

        "seed_win": bool(
            e0_minus_d0
            > 0
        ),
    }


# =============================================================================
# Final summary
# =============================================================================

def build_final_summary(
    seed_summary,
):
    pooled = (
        seed_summary[
            "E0_minus_D0_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    recent = (
        seed_summary[
            "recent_2023_2024_E0_minus_D0_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    latest = (
        seed_summary[
            "gain_2024_E0_minus_D0_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    n = len(
        seed_summary
    )

    seed_wins = int(
        (
            pooled > 0
        )
        .sum()
    )

    mean_pooled = float(
        pooled.mean()
    )

    std_pooled = float(
        pooled.std(
            ddof=1
        )
        if n > 1
        else 0.0
    )

    mean_recent = float(
        recent.mean()
    )

    mean_latest = float(
        latest.mean()
    )

    mean_fold_wins = float(
        seed_summary[
            "fold_wins"
        ]
        .mean()
    )

    if (
        seed_wins == n
        and mean_pooled > 0
        and mean_recent > 0
        and mean_latest > 0
    ):
        verdict = (
            "REMOVE_EXACT_FROM_L2"
        )

    elif (
        seed_wins
        >= max(
            2,
            n - 1,
        )
        and mean_pooled > 0
    ):
        verdict = (
            "REMOVE_EXACT_WEAK_PASS"
        )

    elif (
        seed_wins == 0
        and mean_pooled < 0
    ):
        verdict = (
            "KEEP_EXACT_IN_L2"
        )

    elif mean_pooled < 0:
        verdict = (
            "EXACT_REMOVAL_FAIL"
        )

    else:
        verdict = (
            "MIXED"
        )

    return pd.DataFrame([
        {
            "n_base_seeds": int(
                n
            ),

            "seed_wins": int(
                seed_wins
            ),

            "mean_D0_gain_vs_B_bss": float(
                seed_summary[
                    "D0_gain_vs_B_bss"
                ]
                .mean()
            ),

            "mean_E0_gain_vs_B_bss": float(
                seed_summary[
                    "E0_gain_vs_B_bss"
                ]
                .mean()
            ),

            "mean_E0_minus_D0_bss": (
                mean_pooled
            ),

            "std_E0_minus_D0_bss": (
                std_pooled
            ),

            "gain_to_std_ratio": (
                float(
                    mean_pooled
                    / std_pooled
                )
                if std_pooled > 0
                else np.inf
            ),

            "mean_recent_2023_2024_E0_minus_D0_bss": (
                mean_recent
            ),

            "mean_2024_E0_minus_D0_bss": (
                mean_latest
            ),

            "mean_fold_wins": (
                mean_fold_wins
            ),

            "verdict": (
                verdict
            ),
        }
    ])


# =============================================================================
# Main
# =============================================================================

def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {OUT_DIR}\n"
            "기존 결과 보호를 위해 overwrite하지 않습니다."
        )

    v027_source = (
        find_v027_source()
    )

    v027 = (
        import_module_from_path(
            "v027_source",
            v027_source,
        )
    )

    train, oof, seeds = (
        load_inputs()
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "=" * 118
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 118
    )

    print(
        "CatBoost fits : 0"
    )

    print(
        "D0 fits       : 0 (reuse v028 OOF)"
    )

    print(
        "E0 fits       : ExtraTrees only"
    )

    print(
        f"Base seeds    : {seeds}"
    )

    print(
        f"Adapter seeds : {v027.ADAPTER_SEEDS}"
    )

    print(
        f"Adapter cfg   : {v027.ADAPTER_CFG}"
    )

    print(
        f"Shrink        : {v027.ADAPTER_SHRINK}"
    )

    print(
        "B             : L1 Exact + PC"
    )

    print(
        "D0            : B + L2(core + Exact5)"
    )

    print(
        "E0            : B + L2(core only)"
    )

    print(
        "PC in L2      : NO for both"
    )

    print(
        "Primary       : E0 - D0"
    )

    print(
        "Routing       : R=L2 corrected / F=B protected"
    )

    print(
        "=" * 118
    )

    fold_rows = []
    pred_frames = []

    for seed in seeds:
        print()
        print(
            "#" * 118
        )

        print(
            f"BASE SEED {seed}"
        )

        print(
            "#" * 118
        )

        oof_seed = (
            oof.loc[
                oof[
                    "base_seed"
                ]
                .eq(
                    int(seed)
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        for target_year in sorted(
            oof_seed[
                SEASON
            ]
            .unique()
            .tolist()
        ):
            part = (
                oof_seed.loc[
                    oof_seed[
                        SEASON
                    ]
                    .eq(
                        int(target_year)
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            row, pred = (
                evaluate_fold(
                    v027=v027,
                    train=train,
                    part=part,
                    base_seed=seed,
                    target_year=target_year,
                )
            )

            fold_rows.append(
                row
            )

            pred_frames.append(
                pred
            )

            print(
                f"[{target_year}] "
                f"D0-B={row['D0_minus_B_bss']:+.3f} | "
                f"E0-B={row['E0_minus_B_bss']:+.3f} | "
                f"E0-D0={row['E0_minus_D0_bss']:+.3f} BSS"
            )

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_all = pd.concat(
        pred_frames,
        ignore_index=True,
    )

    seed_summary = pd.DataFrame([
        summarize_seed(
            pred_all,
            seed,
        )
        for seed in seeds
    ])

    final_summary = (
        build_final_summary(
            seed_summary
        )
    )

    fold_df.to_csv(
        FOLD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary.to_csv(
        SEED_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    final_summary.to_csv(
        FINAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pred_all.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "v028_oof": str(
            V028_OOF_PATH
        ),

        "v027_source": str(
            v027_source
        ),

        "catboost_retrained": (
            False
        ),

        "d0_retrained": (
            False
        ),

        "new_model": (
            "E0 ExtraTrees residual adapter only"
        ),

        "base_seeds": [
            int(x)
            for x in seeds
        ],

        "structure": {
            "B": (
                "L1 CatBoost Exact-ASOF5 + R-only PC FULL"
            ),

            "D0": (
                "B + ExtraTrees residual core + Exact-ASOF5; no PC in L2"
            ),

            "E0": (
                "B + ExtraTrees residual core only; no Exact5/PC in L2"
            ),
        },

        "single_changed_factor": (
            "remove Exact-ASOF5 from L2 input"
        ),

        "residual_target": (
            "y - B_L1_PC_OOF"
        ),

        "primary_metric": (
            "E0 BSS - D0 BSS"
        ),

        "adapter_cfg": (
            v027.ADAPTER_CFG
        ),

        "adapter_seeds": (
            v027.ADAPTER_SEEDS
        ),

        "adapter_shrink": float(
            v027.ADAPTER_SHRINK
        ),

        "routing": (
            "R=L2 corrected; F=B protected"
        ),

        "seed_summary": (
            seed_summary
            .to_dict(
                orient="records"
            )
        ),

        "final_summary": (
            final_summary
            .to_dict(
                orient="records"
            )
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

    print()
    print(
        "=" * 118
    )

    print(
        "L2 EXACT DUPLICATION | BASE-SEED SUMMARY"
    )

    print(
        "=" * 118
    )

    print(
        seed_summary.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:+.4f}"
            ),
        )
    )

    print()
    print(
        "=" * 118
    )

    print(
        "L2 EXACT DUPLICATION | FINAL SUMMARY"
    )

    print(
        "=" * 118
    )

    print(
        final_summary.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:+.4f}"
            ),
        )
    )

    print()
    print(
        "Decision:"
    )

    print(
        "- REMOVE_EXACT_FROM_L2 : "
        "Exact는 L1-only로 확정 후보"
    )

    print(
        "- KEEP_EXACT_IN_L2     : "
        "Exact는 L1+L2 양쪽 유지"
    )

    print(
        "- WEAK/MIXED           : "
        "fold/recent/latest 확인 후 결정"
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {FOLD_PATH}"
    )

    print(
        f"- {SEED_SUMMARY_PATH}"
    )

    print(
        f"- {FINAL_SUMMARY_PATH}"
    )

    print(
        f"- {OOF_PATH}"
    )

    print(
        f"- {METADATA_PATH}"
    )

    print(
        "=" * 118
    )


if __name__ == "__main__":
    main()

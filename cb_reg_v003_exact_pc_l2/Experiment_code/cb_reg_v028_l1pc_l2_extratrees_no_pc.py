#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v028_l1pc_l2_extratrees_no_pc.py
=======================================

목적
----
현재까지 가장 유력한 구조를 직접 검증한다.

B) L1 champion
   = CatBoost + Exact-ASOF + R-only PC FULL

D) L1 + L2
   = B
   + ExtraTrees residual adapter
   + adapter core features + Exact-ASOF5
   - L2에는 PC feature를 넣지 않음

중요
----
- CatBoost는 재학습하지 않는다.
- 기존 v026 OOF의 FULL prediction을 그대로 L1 base로 사용한다.
- ExtraTrees residual target만 새로 계산한다:

      residual = y - FULL_OOF

- L2 correction은 R에만 적용하고 F는 L1 prediction을 그대로 보호한다.
- ExtraTrees 설정 / pitcher crossfit / 3 adapter seed consensus는
  이전 v027과 동일하게 유지한다.

필수 파일
---------
프로젝트 루트:

    cb_reg_v027_pc_placement_l1_vs_l2.py

기존 결과:

    validation/
    └─ cb_reg_v026_pc_full_lean4_multiseed3/
       └─ oof_predictions.csv

실행
----
    python cb_reg_v028_l1pc_l2_extratrees_no_pc.py

출력
----
validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/
    fold_results.csv
    seed_summary.csv
    final_summary.csv
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

EXPERIMENT_NAME = "cb_reg_v028_l1pc_l2_extratrees_no_pc"

ROOT = Path(".")

V027_SOURCE_CANDIDATES = [
    ROOT / "cb_reg_v027_pc_placement_l1_vs_l2.py",
    ROOT / "cb_reg_v027_pc_placement_l1_vs_l2(1).py",
]

TRAIN_PATH = ROOT / "data" / "train.csv"

V026_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v026_pc_full_lean4_multiseed3"
    / "oof_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "validation"
    / EXPERIMENT_NAME
)

FOLD_PATH = OUT_DIR / "fold_results.csv"
SEED_SUMMARY_PATH = OUT_DIR / "seed_summary.csv"
FINAL_SUMMARY_PATH = OUT_DIR / "final_summary.csv"
OOF_PATH = OUT_DIR / "oof_predictions.csv"
METADATA_PATH = OUT_DIR / "metadata.json"

REQUESTED_BASE_SEEDS = [7, 42, 2026]

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"


# =============================================================================
# Import v027 exactly
# =============================================================================

def find_v027_source():
    for path in V027_SOURCE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "cb_reg_v027_pc_placement_l1_vs_l2.py를 "
        "프로젝트 루트에서 찾지 못했습니다."
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


# =============================================================================
# Metrics
# =============================================================================

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

    ref = max(
        mu * (1.0 - mu),
        1e-12,
    )

    return float(
        100000.0
        * (
            1.0
            - score / ref
        )
    )


# =============================================================================
# Load existing CatBoost OOF
# =============================================================================

def load_existing_oof():
    if not V026_OOF_PATH.exists():
        raise FileNotFoundError(
            f"v026 OOF 없음:\n{V026_OOF_PATH}\n\n"
            "CatBoost 재학습을 하지 않는 실험이므로 "
            "기존 v026 결과 폴더가 필요합니다."
        )

    oof = pd.read_csv(
        V026_OOF_PATH,
        encoding="utf-8-sig",
    )

    required = {
        ID,
        SEASON,
        GAME_TYPE,
        "seed",
        "y_true",
        "CONTROL",
        "FULL",
    }

    missing = sorted(
        required
        - set(
            oof.columns
        )
    )

    if missing:
        raise KeyError(
            f"v026 OOF missing: {missing}"
        )

    oof[SEASON] = (
        pd.to_numeric(
            oof[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    oof["seed"] = (
        pd.to_numeric(
            oof["seed"],
            errors="raise",
        )
        .astype(int)
    )

    available = sorted(
        oof["seed"]
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

    return (
        oof.loc[
            oof["seed"]
            .isin(
                seeds
            )
        ]
        .copy(),
        seeds,
    )


# =============================================================================
# One seed
# =============================================================================

def evaluate_one_seed(
    v027,
    train,
    oof_seed,
    base_seed,
):
    fold_rows = []
    pred_frames = []

    for target_year in sorted(
        oof_seed[SEASON]
        .unique()
        .tolist()
    ):
        part = (
            oof_seed.loc[
                oof_seed[SEASON]
                .eq(
                    int(
                        target_year
                    )
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        # -------------------------------------------------------------
        # official train row 복원
        # -------------------------------------------------------------
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
                f"OOF row missing from train: {target_year}"
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
                int(
                    target_year
                )
            )
            .all()
        ):
            raise RuntimeError(
                "OOF/train season mismatch"
            )

        y = (
            part["y_true"]
            .to_numpy(
                dtype=np.float64
            )
        )

        # B = current L1 champion
        # CatBoost + Exact + PC FULL
        l1_pc = (
            part["FULL"]
            .to_numpy(
                dtype=np.float64
            )
        )

        # -------------------------------------------------------------
        # L2 feature prep
        # previous labeled season only
        # -------------------------------------------------------------
        source_year = (
            int(
                target_year
            )
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

        # 핵심:
        # base_pred도 CONTROL이 아니라 FULL을 사용.
        core = (
            v027.transform_adapter_core(
                prep=prep,
                rows=raw,
                base_pred=l1_pc,
            )
        )

        # L2에서는 PC 제외.
        # reference-style adapter core + Exact-ASOF5만 사용.
        exact5 = (
            v027.build_exact5_for_target_year(
                base_module=(
                    v027.import_module_from_path(
                        "v021_inner",
                        v027.find_source_path(),
                    )
                ),
                train=train,
                target_rows=raw,
                target_year=target_year,
            )
        )

        X_l2 = pd.concat(
            [
                core.reset_index(
                    drop=True
                ),
                exact5.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        if (
            X_l2.columns
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                "Duplicate L2 feature columns"
            )

        values = (
            X_l2
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                "L2 matrix contains NaN/Inf"
            )

        # -------------------------------------------------------------
        # D = L1-PC + ExtraTrees residual(no PC)
        # residual target is automatically:
        # y - l1_pc
        # -------------------------------------------------------------
        (
            l1_l2,
            adapter_seed_diag,
        ) = v027.crossfit_l2_variant(
            X=X_l2,
            y=y,
            base_pred=l1_pc,
            frame=raw,
        )

        B_brier = brier(
            y,
            l1_pc,
        )

        D_brier = brier(
            y,
            l1_l2,
        )

        B_bss = bss(
            y,
            l1_pc,
        )

        D_bss = bss(
            y,
            l1_l2,
        )

        fold_rows.append({
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
                B_brier
            ),
            "B_l1_pc_bss": (
                B_bss
            ),

            "D_l1pc_l2_brier": (
                D_brier
            ),
            "D_l1pc_l2_bss": (
                D_bss
            ),

            "gain_brier_B_minus_D": float(
                B_brier
                - D_brier
            ),

            "gain_bss_D_minus_B": float(
                D_bss
                - B_bss
            ),

            "n_l2_features": int(
                X_l2.shape[1]
            ),

            "adapter_seed_mean_bss": float(
                adapter_seed_diag[
                    "bss"
                ].mean()
            ),

            "adapter_seed_std_bss": float(
                adapter_seed_diag[
                    "bss"
                ].std(
                    ddof=1
                )
            ),
        })

        pred_frames.append(
            pd.DataFrame({
                ID: (
                    part[ID]
                    .to_numpy()
                ),
                SEASON: int(
                    target_year
                ),
                GAME_TYPE: (
                    raw[GAME_TYPE]
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
                "D_L1PC_L2": (
                    l1_l2
                ),
            })
        )

        print(
            f"[seed={base_seed} | {target_year}] "
            f"B={B_bss:+.2f} | "
            f"D={D_bss:+.2f} | "
            f"gain={D_bss - B_bss:+.2f} BSS"
        )

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.concat(
            pred_frames,
            ignore_index=True,
        ),
    )


# =============================================================================
# Summaries
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
                int(
                    seed
                )
            )
        ]
        .copy()
    )

    y = (
        part["y_true"]
        .to_numpy(
            dtype=np.float64
        )
    )

    B = (
        part["B_L1_PC"]
        .to_numpy(
            dtype=np.float64
        )
    )

    D = (
        part["D_L1PC_L2"]
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
        .eq(2024)
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
                D[mask],
            )
            - bss(
                y[mask],
                B[mask],
            )
        )

        fold_gains[
            year
        ] = float(
            gain
        )

        if gain > 0:
            fold_wins += 1

    pooled_gain = (
        bss(
            y,
            D,
        )
        - bss(
            y,
            B,
        )
    )

    recent_gain = (
        bss(
            y[recent],
            D[recent],
        )
        - bss(
            y[recent],
            B[recent],
        )
    )

    latest_gain = (
        bss(
            y[latest],
            D[latest],
        )
        - bss(
            y[latest],
            B[latest],
        )
    )

    return {
        "base_seed": int(
            seed
        ),

        "B_l1_pc_brier": (
            brier(
                y,
                B,
            )
        ),

        "D_l1pc_l2_brier": (
            brier(
                y,
                D,
            )
        ),

        "pooled_gain_bss": float(
            pooled_gain
        ),

        "recent_2023_2024_gain_bss": float(
            recent_gain
        ),

        "gain_2024_bss": float(
            latest_gain
        ),

        "gain_2021_bss": (
            fold_gains[2021]
        ),

        "gain_2022_bss": (
            fold_gains[2022]
        ),

        "gain_2023_bss": (
            fold_gains[2023]
        ),

        "gain_2024_bss_check": (
            fold_gains[2024]
        ),

        "fold_wins": int(
            fold_wins
        ),

        "seed_win": bool(
            pooled_gain > 0
        ),
    }


def build_final_summary(
    seed_summary,
):
    pooled = (
        seed_summary[
            "pooled_gain_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    recent = (
        seed_summary[
            "recent_2023_2024_gain_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    latest = (
        seed_summary[
            "gain_2024_bss"
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
        ).sum()
    )

    mean_gain = float(
        pooled.mean()
    )

    std_gain = float(
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

    if (
        seed_wins == n
        and mean_gain > 0
        and mean_recent > 0
        and mean_latest > 0
    ):
        verdict = (
            "STRONG_PASS"
        )
    elif (
        seed_wins
        >= max(
            2,
            n - 1,
        )
        and mean_gain > 0
        and mean_recent > 0
    ):
        verdict = (
            "PASS"
        )
    elif mean_gain > 0:
        verdict = (
            "WEAK"
        )
    else:
        verdict = (
            "FAIL"
        )

    return pd.DataFrame(
        [
            {
                "n_base_seeds": int(
                    n
                ),
                "seed_wins": int(
                    seed_wins
                ),
                "mean_pooled_gain_bss": (
                    mean_gain
                ),
                "std_pooled_gain_bss": (
                    std_gain
                ),
                "gain_to_std_ratio": (
                    float(
                        mean_gain
                        / std_gain
                    )
                    if std_gain > 0
                    else np.inf
                ),
                "mean_recent_2023_2024_gain_bss": (
                    mean_recent
                ),
                "mean_2024_gain_bss": (
                    mean_latest
                ),
                "mean_fold_wins": float(
                    seed_summary[
                        "fold_wins"
                    ]
                    .mean()
                ),
                "verdict": (
                    verdict
                ),
            }
        ]
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {OUT_DIR}\n"
            "기존 결과 보존을 위해 폴더를 삭제하거나 "
            "EXPERIMENT_NAME을 변경하세요."
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
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

    existing_oof, seeds = (
        load_existing_oof()
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "=" * 110
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 110
    )
    print(
        "CatBoost fits : 0"
    )
    print(
        f"Base seeds    : {seeds}"
    )
    print(
        "B             : CatBoost + Exact + PC FULL"
    )
    print(
        "D             : B + ExtraTrees residual adapter (NO PC in L2)"
    )
    print(
        "Residual      : y - B_L1_PC OOF"
    )
    print(
        "Routing       : R=L2 corrected, F=B protected"
    )
    print(
        f"Adapter cfg   : {v027.ADAPTER_CFG}"
    )
    print(
        f"Adapter seeds : {v027.ADAPTER_SEEDS}"
    )
    print(
        f"Shrink        : {v027.ADAPTER_SHRINK}"
    )
    print(
        "=" * 110
    )

    fold_frames = []
    pred_frames = []

    for seed in seeds:
        print()
        print(
            "#" * 110
        )
        print(
            f"BASE SEED {seed}"
        )
        print(
            "#" * 110
        )

        oof_seed = (
            existing_oof.loc[
                existing_oof[
                    "seed"
                ]
                .eq(
                    int(
                        seed
                    )
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        fold_df, pred_df = (
            evaluate_one_seed(
                v027=v027,
                train=train,
                oof_seed=oof_seed,
                base_seed=seed,
            )
        )

        fold_frames.append(
            fold_df
        )

        pred_frames.append(
            pred_df
        )

    fold_all = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    pred_all = pd.concat(
        pred_frames,
        ignore_index=True,
    )

    seed_summary = pd.DataFrame(
        [
            summarize_seed(
                pred_all,
                seed,
            )
            for seed in seeds
        ]
    )

    final_summary = (
        build_final_summary(
            seed_summary
        )
    )

    fold_all.to_csv(
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
        "catboost_retrained": False,
        "existing_catboost_oof": str(
            V026_OOF_PATH
        ),
        "v027_source": str(
            v027_source
        ),
        "base_seeds": [
            int(
                x
            )
            for x in seeds
        ],
        "structure_B": (
            "CatBoost Exact + R-only PC FULL"
        ),
        "structure_D": (
            "B + ExtraTrees residual adapter; "
            "PC excluded from L2"
        ),
        "residual_target": (
            "y - B_L1_PC OOF"
        ),
        "routing": (
            "R=L2 corrected; F=B protected"
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
        "=" * 110
    )
    print(
        "L1-PC + L2 EXTRATREES | BASE-SEED SUMMARY"
    )
    print(
        "=" * 110
    )

    print(
        seed_summary[
            [
                "base_seed",
                "pooled_gain_bss",
                "recent_2023_2024_gain_bss",
                "gain_2024_bss",
                "gain_2021_bss",
                "gain_2022_bss",
                "gain_2023_bss",
                "gain_2024_bss_check",
                "fold_wins",
                "seed_win",
            ]
        ]
        .to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:+.4f}"
            ),
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "L1-PC + L2 EXTRATREES | FINAL SUMMARY"
    )
    print(
        "=" * 110
    )

    print(
        final_summary
        .to_string(
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
        "- STRONG_PASS => L1+L2 구조 확정 후 L3 Ridge로 이동"
    )
    print(
        "- PASS        => L3 전 한 번 더 세부 안정성 확인 가능"
    )
    print(
        "- WEAK/FAIL   => L1 PC만 유지하거나 L2 구조 재검토"
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
        "=" * 110
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v039_shadow_control_state_sanity.py
==========================================

목적
----
v038에서 관찰된 큰 gain이 진짜 current-season state feature 효과인지 검증한다.

핵심
----
v038 Treatment는 이미 학습 완료되어 OOF가 저장되어 있다.
따라서 Treatment를 다시 학습하지 않는다.

이번 실험에서 새로 학습하는 것은 딱:
    SHADOW_CONTROL = v038과 완전히 같은 training pipeline
                     + extra current-state feature 0개

4 expanding folds만 새로 학습한다.

검증 질문
---------
1) 기존 v028 B_L1_PC OOF와 v038 pipeline의 SHADOW_CONTROL이 사실상 동일한가?

2) v038의 STATE_SUCCESS_EXT / STATE_FULL R_ONLY gain을
   SHADOW_CONTROL 기준으로 다시 계산해도 유지되는가?

3) STATE_FULL - STATE_SUCCESS_EXT의 incremental gain이
   각 fold에서 얼마나 안정적인가?

4) FULL의 큰 recent gain이 특정 한 fold(특히 2023)에만 집중된 것인가?

입력
----
data/train.csv
data/test.csv

validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/oof_predictions.csv

validation/cb_reg_v038_current_season_asof_state_ablation/
    oof_predictions.csv

필요 source
-----------
cb_reg_v038_current_season_asof_state_ablation.py
cb_reg_v021_pitcher_conditional_r_only_ablation*.py

출력
----
validation/cb_reg_v039_shadow_control_state_sanity/
    shadow_control_oof.csv
    shadow_equivalence_by_fold.csv
    shadow_equivalence_summary.csv
    corrected_state_ablation_by_fold.csv
    corrected_state_ablation_summary.csv
    full_minus_success_by_fold.csv
    metadata.json

실행
----
python cb_reg_v039_shadow_control_state_sanity.py
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

EXPERIMENT_NAME = "cb_reg_v039_shadow_control_state_sanity"

ROOT = Path(".")

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

V028_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
    / "oof_predictions.csv"
)

V038_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v038_current_season_asof_state_ablation"
    / "oof_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "validation"
    / EXPERIMENT_NAME
)

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"

CONTROL_SEED = 42

TREATMENTS = [
    "STATE_SUCCESS_EXT",
    "STATE_FULL",
]

ROUTE = "R_ONLY"

# Equivalence는 hard pass/fail보다 연속적인 수치를 보는 게 중요하다.
# 아래 값들은 보조 진단용.
PRED_EXACT_TOL = 1e-12
PRED_PRACTICAL_TOL = 1e-8
BSS_PRACTICAL_TOL = 1e-4


# =============================================================================
# Generic helpers
# =============================================================================

def find_script(pattern):
    direct = ROOT / pattern

    if direct.exists():
        return direct

    matches = sorted(
        p
        for p in ROOT.rglob(pattern)
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and "validation" not in p.parts
    )

    if len(matches) == 1:
        print(
            f"[AUTO] source found: {matches[0]}"
        )
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"{pattern} not found in project."
        )

    raise RuntimeError(
        f"Multiple candidates for {pattern}:\n"
        + "\n".join(
            f"- {p}"
            for p in matches
        )
    )


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(path)

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


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
        np.mean(
            y
        )
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
# Load frozen OOFs
# =============================================================================

def load_v028_control():
    if not V028_OOF_PATH.exists():
        raise FileNotFoundError(
            V028_OOF_PATH
        )

    df = pd.read_csv(
        V028_OOF_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        ID,
        SEASON,
        GAME_TYPE,
        "base_seed",
        "y_true",
        "B_L1_PC",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"v028 missing columns: {missing}"
        )

    df = (
        df.loc[
            pd.to_numeric(
                df["base_seed"],
                errors="raise",
            )
            .astype(int)
            .eq(
                CONTROL_SEED
            )
        ]
        .copy()
    )

    df[SEASON] = (
        pd.to_numeric(
            df[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    df["y_true"] = (
        pd.to_numeric(
            df["y_true"],
            errors="raise",
        )
        .astype(np.float64)
    )

    df["v028_control_pred"] = (
        pd.to_numeric(
            df["B_L1_PC"],
            errors="raise",
        )
        .astype(np.float64)
    )

    out = (
        df[
            [
                ID,
                SEASON,
                GAME_TYPE,
                "y_true",
                "v028_control_pred",
            ]
        ]
        .copy()
    )

    if out[ID].duplicated().any():
        raise RuntimeError(
            "v028 seed42 row_id duplicate"
        )

    return out


def load_v038_treatments():
    if not V038_OOF_PATH.exists():
        raise FileNotFoundError(
            V038_OOF_PATH
        )

    df = pd.read_csv(
        V038_OOF_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        ID,
        SEASON,
        GAME_TYPE,
        "treatment",
        "route",
        "y_true",
        "control_pred",
        "treatment_raw_pred",
        "pred",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"v038 missing columns: {missing}"
        )

    df = (
        df.loc[
            df["treatment"]
            .isin(
                TREATMENTS
            )
            & df["route"]
            .eq(
                ROUTE
            )
        ]
        .copy()
    )

    df[SEASON] = (
        pd.to_numeric(
            df[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    for c in [
        "y_true",
        "control_pred",
        "treatment_raw_pred",
        "pred",
    ]:
        df[c] = (
            pd.to_numeric(
                df[c],
                errors="raise",
            )
            .astype(np.float64)
        )

    if df.empty:
        raise RuntimeError(
            "No v038 R_ONLY treatment rows found."
        )

    for treatment in TREATMENTS:
        d = df.loc[
            df["treatment"]
            .eq(
                treatment
            )
        ]

        if d[ID].duplicated().any():
            raise RuntimeError(
                f"v038 duplicate row_id for {treatment}"
            )

    return df


# =============================================================================
# Fresh shadow control
# =============================================================================

def train_shadow_control(
    v038,
    src,
    train,
    base_features,
):
    pred_frames = []
    fit_rows = []

    for cfg in src.EXPANDING_FOLDS:
        fold = int(
            cfg["fold"]
        )

        val_year = int(
            cfg["validation_season"]
        )

        print()
        print(
            "=" * 126
        )
        print(
            f"SHADOW CONTROL FOLD {fold}: "
            f"{cfg['train_seasons']} -> {val_year}"
        )
        print(
            "=" * 126
        )

        (
            df_tr,
            df_va,
            tr_view,
            va_view,
            base_l1_features,
            diagnostics,
        ) = v038.build_fold_l1_view(
            src=src,
            train=train,
            base_features=base_features,
            cfg=cfg,
        )

        # 핵심:
        # state columns는 tr_view / va_view에 존재하지만,
        # features에 단 하나도 넣지 않는다.
        # 따라서 v038 training path의 exact "0 extra state" control.
        (
            shadow_pred,
            best_iteration,
            elapsed,
        ) = v038.fit_treatment(
            src=src,
            train_view=tr_view,
            val_view=va_view,
            df_train=df_tr,
            df_val=df_va,
            features=base_l1_features,
        )

        y = (
            df_va[TARGET]
            .to_numpy(
                np.float64
            )
        )

        pred_frames.append(
            pd.DataFrame({
                ID: (
                    df_va[ID]
                    .to_numpy()
                ),
                SEASON: int(
                    val_year
                ),
                GAME_TYPE: (
                    df_va[GAME_TYPE]
                    .fillna(
                        "__MISSING__"
                    )
                    .astype(str)
                    .to_numpy()
                ),
                "y_true": y,
                "shadow_control_pred": (
                    shadow_pred
                ),
            })
        )

        fit_rows.append({
            "fold": fold,
            "validation_year": (
                val_year
            ),
            "n_train": int(
                len(
                    df_tr
                )
            ),
            "n_validation": int(
                len(
                    df_va
                )
            ),
            "n_features": int(
                len(
                    base_l1_features
                )
            ),
            "best_iteration": int(
                best_iteration
            ),
            "fit_seconds": float(
                elapsed
            ),
            "shadow_control_bss": float(
                bss(
                    y,
                    shadow_pred,
                )
            ),
        })

        print(
            f"best_iter={best_iteration} | "
            f"shadow BSS={bss(y, shadow_pred):+.6f}"
        )

    return (
        pd.concat(
            pred_frames,
            ignore_index=True,
        ),
        pd.DataFrame(
            fit_rows
        ),
    )


# =============================================================================
# Shadow equivalence
# =============================================================================

def compare_shadow_to_v028(
    shadow,
    v028_control,
):
    x = (
        shadow.merge(
            v028_control[
                [
                    ID,
                    "v028_control_pred",
                ]
            ],
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if x[
        "v028_control_pred"
    ].isna().any():
        raise RuntimeError(
            "Shadow/v028 alignment failed."
        )

    x[
        "abs_pred_diff"
    ] = np.abs(
        x[
            "shadow_control_pred"
        ]
        - x[
            "v028_control_pred"
        ]
    )

    rows = []

    for year, g in x.groupby(
        SEASON,
        sort=True,
    ):
        y = (
            g["y_true"]
            .to_numpy(
                np.float64
            )
        )

        s = (
            g["shadow_control_pred"]
            .to_numpy(
                np.float64
            )
        )

        old = (
            g["v028_control_pred"]
            .to_numpy(
                np.float64
            )
        )

        rows.append({
            "year": int(
                year
            ),
            "n": int(
                len(
                    g
                )
            ),
            "max_abs_pred_diff": float(
                np.max(
                    np.abs(
                        s - old
                    )
                )
            ),
            "mean_abs_pred_diff": float(
                np.mean(
                    np.abs(
                        s - old
                    )
                )
            ),
            "p99_abs_pred_diff": float(
                np.quantile(
                    np.abs(
                        s - old
                    ),
                    0.99,
                )
            ),
            "shadow_bss": float(
                bss(
                    y,
                    s,
                )
            ),
            "v028_bss": float(
                bss(
                    y,
                    old,
                )
            ),
            "shadow_minus_v028_bss": float(
                bss(
                    y,
                    s,
                )
                - bss(
                    y,
                    old,
                )
            ),
        })

    by_fold = pd.DataFrame(
        rows
    )

    y = x[
        "y_true"
    ].to_numpy(
        np.float64
    )

    s = x[
        "shadow_control_pred"
    ].to_numpy(
        np.float64
    )

    old = x[
        "v028_control_pred"
    ].to_numpy(
        np.float64
    )

    max_diff = float(
        np.max(
            np.abs(
                s - old
            )
        )
    )

    mean_diff = float(
        np.mean(
            np.abs(
                s - old
            )
        )
    )

    bss_diff = float(
        bss(
            y,
            s,
        )
        - bss(
            y,
            old,
        )
    )

    summary = pd.DataFrame([
        {
            "n": int(
                len(
                    x
                )
            ),
            "max_abs_pred_diff": (
                max_diff
            ),
            "mean_abs_pred_diff": (
                mean_diff
            ),
            "p99_abs_pred_diff": float(
                np.quantile(
                    np.abs(
                        s - old
                    ),
                    0.99,
                )
            ),
            "shadow_bss": float(
                bss(
                    y,
                    s,
                )
            ),
            "v028_bss": float(
                bss(
                    y,
                    old,
                )
            ),
            "shadow_minus_v028_bss": (
                bss_diff
            ),
            "exact_prediction_match": bool(
                max_diff
                <= PRED_EXACT_TOL
            ),
            "practical_prediction_match": bool(
                max_diff
                <= PRED_PRACTICAL_TOL
            ),
            "practical_bss_match": bool(
                abs(
                    bss_diff
                )
                <= BSS_PRACTICAL_TOL
            ),
        }
    ])

    return (
        x,
        by_fold,
        summary,
    )


# =============================================================================
# Corrected same-pipeline treatment evaluation
# =============================================================================

def build_same_pipeline_eval(
    shadow,
    v038_treatments,
):
    frames = []

    for treatment in TREATMENTS:
        t = (
            v038_treatments.loc[
                v038_treatments[
                    "treatment"
                ]
                .eq(
                    treatment
                )
            ]
            [
                [
                    ID,
                    "treatment_raw_pred",
                ]
            ]
            .copy()
        )

        x = shadow.merge(
            t,
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if x[
            "treatment_raw_pred"
        ].isna().any():
            raise RuntimeError(
                f"Treatment alignment failed: {treatment}"
            )

        gt = (
            x[GAME_TYPE]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .to_numpy()
        )

        r_mask = (
            gt != "F"
        )

        routed = (
            x[
                "shadow_control_pred"
            ]
            .to_numpy(
                np.float64
            )
            .copy()
        )

        raw = (
            x[
                "treatment_raw_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        routed[
            r_mask
        ] = raw[
            r_mask
        ]

        frames.append(
            pd.DataFrame({
                ID: (
                    x[ID]
                    .to_numpy()
                ),
                SEASON: (
                    x[SEASON]
                    .to_numpy()
                ),
                GAME_TYPE: (
                    gt
                ),
                "y_true": (
                    x[
                        "y_true"
                    ]
                    .to_numpy(
                        np.float64
                    )
                ),
                "treatment": (
                    treatment
                ),
                "shadow_control_pred": (
                    x[
                        "shadow_control_pred"
                    ]
                    .to_numpy(
                        np.float64
                    )
                ),
                "treatment_raw_pred": (
                    raw
                ),
                "routed_pred": (
                    routed
                ),
            })
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def summarize_treatments(
    eval_df,
):
    by_fold_rows = []
    summary_rows = []

    for treatment, g in eval_df.groupby(
        "treatment",
        sort=True,
    ):
        fold_gains = []

        for year, gy in g.groupby(
            SEASON,
            sort=True,
        ):
            y = (
                gy["y_true"]
                .to_numpy(
                    np.float64
                )
            )

            c = (
                gy[
                    "shadow_control_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            p = (
                gy[
                    "routed_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            gain = float(
                bss(
                    y,
                    p,
                )
                - bss(
                    y,
                    c,
                )
            )

            fold_gains.append(
                gain
            )

            by_fold_rows.append({
                "treatment": str(
                    treatment
                ),
                "year": int(
                    year
                ),
                "n": int(
                    len(
                        gy
                    )
                ),
                "shadow_control_bss": float(
                    bss(
                        y,
                        c,
                    )
                ),
                "treatment_bss": float(
                    bss(
                        y,
                        p,
                    )
                ),
                "gain_vs_shadow_control_bss": (
                    gain
                ),
            })

        y = (
            g["y_true"]
            .to_numpy(
                np.float64
            )
        )

        c = (
            g[
                "shadow_control_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        p = (
            g[
                "routed_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        recent = (
            g[SEASON]
            .isin(
                [
                    2023,
                    2024,
                ]
            )
            .to_numpy()
        )

        latest = (
            g[SEASON]
            .eq(
                2024
            )
            .to_numpy()
        )

        summary_rows.append({
            "treatment": str(
                treatment
            ),
            "route": (
                ROUTE
            ),
            "pooled_gain_vs_shadow_bss": float(
                bss(
                    y,
                    p,
                )
                - bss(
                    y,
                    c,
                )
            ),
            "recent_2023_2024_gain_vs_shadow_bss": float(
                bss(
                    y[
                        recent
                    ],
                    p[
                        recent
                    ],
                )
                - bss(
                    y[
                        recent
                    ],
                    c[
                        recent
                    ],
                )
            ),
            "gain_2024_vs_shadow_bss": float(
                bss(
                    y[
                        latest
                    ],
                    p[
                        latest
                    ],
                )
                - bss(
                    y[
                        latest
                    ],
                    c[
                        latest
                    ],
                )
            ),
            "fold_wins": int(
                sum(
                    z > 0
                    for z in fold_gains
                )
            ),
            "n_folds": int(
                len(
                    fold_gains
                )
            ),
            "min_fold_gain_bss": float(
                np.min(
                    fold_gains
                )
            ),
            "mean_fold_gain_bss": float(
                np.mean(
                    fold_gains
                )
            ),
        })

    return (
        pd.DataFrame(
            by_fold_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# =============================================================================
# FULL minus SUCCESS incremental audit
# =============================================================================

def build_full_minus_success(
    eval_df,
):
    success = (
        eval_df.loc[
            eval_df[
                "treatment"
            ]
            .eq(
                "STATE_SUCCESS_EXT"
            ),
            [
                ID,
                SEASON,
                GAME_TYPE,
                "y_true",
                "routed_pred",
            ],
        ]
        .rename(
            columns={
                "routed_pred": (
                    "success_pred"
                ),
            }
        )
    )

    full = (
        eval_df.loc[
            eval_df[
                "treatment"
            ]
            .eq(
                "STATE_FULL"
            ),
            [
                ID,
                "routed_pred",
            ],
        ]
        .rename(
            columns={
                "routed_pred": (
                    "full_pred"
                ),
            }
        )
    )

    x = success.merge(
        full,
        on=ID,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    rows = []

    for year, g in x.groupby(
        SEASON,
        sort=True,
    ):
        y = (
            g["y_true"]
            .to_numpy(
                np.float64
            )
        )

        s = (
            g["success_pred"]
            .to_numpy(
                np.float64
            )
        )

        f = (
            g["full_pred"]
            .to_numpy(
                np.float64
            )
        )

        rows.append({
            "year": int(
                year
            ),
            "n": int(
                len(
                    g
                )
            ),
            "success_bss": float(
                bss(
                    y,
                    s,
                )
            ),
            "full_bss": float(
                bss(
                    y,
                    f,
                )
            ),
            "full_minus_success_bss": float(
                bss(
                    y,
                    f,
                )
                - bss(
                    y,
                    s,
                )
            ),
        })

    # Pooled/recent summary rows also append.
    y = (
        x["y_true"]
        .to_numpy(
            np.float64
        )
    )

    s = (
        x["success_pred"]
        .to_numpy(
            np.float64
        )
    )

    f = (
        x["full_pred"]
        .to_numpy(
            np.float64
        )
    )

    recent = (
        x[SEASON]
        .isin(
            [
                2023,
                2024,
            ]
        )
        .to_numpy()
    )

    summary = pd.DataFrame([
        {
            "scope": "POOLED",
            "full_minus_success_bss": float(
                bss(
                    y,
                    f,
                )
                - bss(
                    y,
                    s,
                )
            ),
        },
        {
            "scope": "RECENT_2023_2024",
            "full_minus_success_bss": float(
                bss(
                    y[
                        recent
                    ],
                    f[
                        recent
                    ],
                )
                - bss(
                    y[
                        recent
                    ],
                    s[
                        recent
                    ],
                )
            ),
        },
        {
            "scope": "2024",
            "full_minus_success_bss": float(
                bss(
                    y[
                        x[SEASON]
                        .eq(
                            2024
                        )
                        .to_numpy()
                    ],
                    f[
                        x[SEASON]
                        .eq(
                            2024
                        )
                        .to_numpy()
                    ],
                )
                - bss(
                    y[
                        x[SEASON]
                        .eq(
                            2024
                        )
                        .to_numpy()
                    ],
                    s[
                        x[SEASON]
                        .eq(
                            2024
                        )
                        .to_numpy()
                    ],
                )
            ),
        },
    ])

    return (
        pd.DataFrame(
            rows
        ),
        summary,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"{OUT_DIR} already exists.\n"
            "기존 결과 보호를 위해 overwrite하지 않습니다."
        )

    v038_path = find_script(
        "cb_reg_v038_current_season_asof_state_ablation.py"
    )

    v038 = import_module(
        "state_v038",
        v038_path,
    )

    v021_path = v038.find_v021_source()

    src = import_module(
        "pc_source_v021",
        v021_path,
    )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    test_cols = (
        pd.read_csv(
            TEST_PATH,
            encoding="utf-8-sig",
            nrows=0,
        )
        .columns
        .tolist()
    )

    base_features = [
        c
        for c in test_cols
        if c != ID
    ]

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
        ]
        + base_features
        + [
            TARGET,
        ],
        low_memory=False,
    )

    train[SEASON] = (
        pd.to_numeric(
            train[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    v028_control = load_v028_control()
    v038_treatments = load_v038_treatments()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "=" * 126
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 126
    )
    print(
        f"v038 source  : {v038_path}"
    )
    print(
        f"v021 source  : {v021_path}"
    )
    print(
        "Fresh fits   : SHADOW_CONTROL only (4 folds)"
    )
    print(
        "Treatments   : REUSE v038 OOF"
    )
    print(
        "=" * 126
    )

    (
        shadow,
        shadow_fit_info,
    ) = train_shadow_control(
        v038=v038,
        src=src,
        train=train,
        base_features=base_features,
    )

    (
        shadow_with_old,
        equivalence_by_fold,
        equivalence_summary,
    ) = compare_shadow_to_v028(
        shadow=shadow,
        v028_control=v028_control,
    )

    eval_df = build_same_pipeline_eval(
        shadow=shadow,
        v038_treatments=v038_treatments,
    )

    (
        corrected_by_fold,
        corrected_summary,
    ) = summarize_treatments(
        eval_df
    )

    (
        incremental_by_fold,
        incremental_summary,
    ) = build_full_minus_success(
        eval_df
    )

    # Save
    shadow_with_old.to_csv(
        OUT_DIR
        / "shadow_control_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shadow_fit_info.to_csv(
        OUT_DIR
        / "shadow_fit_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    equivalence_by_fold.to_csv(
        OUT_DIR
        / "shadow_equivalence_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    equivalence_summary.to_csv(
        OUT_DIR
        / "shadow_equivalence_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corrected_by_fold.to_csv(
        OUT_DIR
        / "corrected_state_ablation_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corrected_summary.to_csv(
        OUT_DIR
        / "corrected_state_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    incremental_by_fold.to_csv(
        OUT_DIR
        / "full_minus_success_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    incremental_summary.to_csv(
        OUT_DIR
        / "full_minus_success_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 126
    )
    print(
        "SHADOW CONTROL EQUIVALENCE"
    )
    print(
        "=" * 126
    )

    print(
        equivalence_by_fold.to_string(
            index=False,
            float_format=lambda x: f"{x:+.10f}",
        )
    )

    print()
    print(
        "OVERALL:"
    )

    print(
        equivalence_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.10f}",
        )
    )

    print()
    print(
        "=" * 126
    )
    print(
        "SAME-PIPELINE STATE ABLATION"
    )
    print(
        "=" * 126
    )

    print(
        corrected_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "=" * 126
    )
    print(
        "FULL - SUCCESS_EXT BY FOLD"
    )
    print(
        "=" * 126
    )

    print(
        incremental_by_fold.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "SUMMARY:"
    )

    print(
        incremental_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    # Decision helper
    e = equivalence_summary.iloc[0]

    shadow_ok = bool(
        e[
            "practical_prediction_match"
        ]
        or e[
            "practical_bss_match"
        ]
    )

    full_row = (
        corrected_summary.loc[
            corrected_summary[
                "treatment"
            ]
            .eq(
                "STATE_FULL"
            )
        ]
        .iloc[0]
    )

    success_row = (
        corrected_summary.loc[
            corrected_summary[
                "treatment"
            ]
            .eq(
                "STATE_SUCCESS_EXT"
            )
        ]
        .iloc[0]
    )

    print()
    print(
        "=" * 126
    )
    print(
        "DECISION"
    )
    print(
        "=" * 126
    )

    if shadow_ok:
        print(
            "PASS: v028 control and v038 shadow control are practically equivalent."
        )
    else:
        print(
            "WARNING: shadow control differs materially from v028."
        )

    print(
        f"STATE_SUCCESS_EXT same-pipeline pooled gain: "
        f"{success_row['pooled_gain_vs_shadow_bss']:+.5f} BSS"
    )

    print(
        f"STATE_FULL same-pipeline pooled gain: "
        f"{full_row['pooled_gain_vs_shadow_bss']:+.5f} BSS"
    )

    print(
        f"STATE_FULL same-pipeline 2024 gain: "
        f"{full_row['gain_2024_vs_shadow_bss']:+.5f} BSS"
    )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),
        "v038_source": str(
            v038_path
        ),
        "v021_source": str(
            v021_path
        ),
        "fresh_fits": (
            "shadow control only; 4 expanding folds"
        ),
        "treatment_predictions": (
            "reused from v038"
        ),
        "shadow_control_definition": (
            "same v038 build/fitting pipeline with base_l1_features only, "
            "zero current-state features"
        ),
        "practical_prediction_tolerance": float(
            PRED_PRACTICAL_TOL
        ),
        "practical_bss_tolerance": float(
            BSS_PRACTICAL_TOL
        ),
        "notes": [
            (
                "For same-pipeline R-only evaluation, R rows use frozen v038 "
                "treatment predictions and F rows use the freshly trained shadow control."
            ),
            (
                "This isolates current-state feature effect from any possible "
                "difference between the old v028 training path and v038 training path."
            ),
            (
                "FULL minus SUCCESS_EXT is evaluated directly on routed OOF "
                "to reveal fold concentration of component-state incremental value."
            ),
        ],
    }

    (
        OUT_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {OUT_DIR}"
    )
    print(
        "=" * 126
    )


if __name__ == "__main__":
    main()

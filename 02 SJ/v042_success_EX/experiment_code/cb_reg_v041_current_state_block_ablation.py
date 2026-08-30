#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v041_current_state_block_ablation.py
===========================================

목적
----
v038/v040에서 확인된 current-season state improvement를 세부 block으로 분해한다.

확정된 사실
------------
- R에서 v028 Control == v039 Shadow Control prediction exactly.
- STATE_SUCCESS_EXT R_ONLY:
    4/4 folds positive
- STATE_FULL R_ONLY:
    4/4 folds positive
- FULL - SUCCESS는 2023 기여가 매우 큼.

따라서 이번 실험은:

A. SUCCESS_EXT의 +효과가 어디서 나오는가?
   1) pitcher success extension
   2) batter current state
   3) pitcher-batter current diff
   4) pitcher + batter, no matchup diff
   5) full SUCCESS_EXT

B. COMPONENT의 추가효과가 어디서 나오는가?
   SUCCESS_EXT 위에
   1) pitch outcome
   2) pitchmix
   3) batter-middle
   4) sample reliability only
   5) full COMPONENT

재학습 정책
-----------
REUSE:
- v028 seed42 B_L1_PC control
- v038 STATE_SUCCESS_EXT raw treatment prediction
- v038 STATE_FULL raw treatment prediction

NEW FITS:
- PITCHER_SUCCESS_EXT
- BATTER_CURRENT_STATE
- MATCHUP_CURRENT_DIFF
- SUCCESS_NO_MATCHUP
- SUCCESS_PLUS_OUTCOME
- SUCCESS_PLUS_PITCHMIX
- SUCCESS_PLUS_BATTER_MIDDLE
- SUCCESS_PLUS_SAMPLE_RELIABILITY

Control CatBoost는 재학습하지 않는다.

평가 route
----------
R_ONLY only:
- R -> treatment
- F -> existing v028 control

왜 R_ONLY만 보는가?
-------------------
v040에서 R control prediction은 v028 == v039 shadow exactly.
F current-state는 v038에서 명확히 악화되었으므로 보호가 확정적이다.

검증
----
Expanding Window:
2019-20 -> 2021
2019-21 -> 2022
2019-22 -> 2023
2019-23 -> 2024

출력
----
validation/cb_reg_v041_current_state_block_ablation/
    fold_results.csv
    oof_predictions.csv
    summary.csv
    incremental_vs_success_by_fold.csv
    incremental_vs_success_summary.csv
    feature_manifest.csv
    metadata.json

실행
----
python cb_reg_v041_current_state_block_ablation.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v041_current_state_block_ablation"

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
ROUTE = "R_ONLY"


# =============================================================================
# Feature blocks
# =============================================================================

PITCHER_SUCCESS_EXT = [
    "gseason__pitcher_rate_k40",
    "gseason__pitcher_rate_k80",
    "gseason__pitcher_rate_k160",
    "gseason__pitcher_reliability",
    "gseason__pitcher_minus_career",
]

BATTER_CURRENT_STATE = [
    "gseason__batter_rate_k40",
    "gseason__batter_rate_k80",
    "gseason__batter_rate_k160",
    "gseason__batter_raw",
    "gseason__batter_log_n",
    "gseason__batter_reliability",
    "gseason__batter_minus_career",
]

MATCHUP_CURRENT_DIFF = [
    "gseason__pitcher_minus_batter_k80",
]

SUCCESS_NO_MATCHUP = (
    PITCHER_SUCCESS_EXT
    + BATTER_CURRENT_STATE
)

SUCCESS_FULL = (
    SUCCESS_NO_MATCHUP
    + MATCHUP_CURRENT_DIFF
)

# -------------------------------------------------------------------------
# Components
# -------------------------------------------------------------------------

SAMPLE_RELIABILITY = [
    "gseason__pitcher_component_log_n",
    "gseason__pitcher_component_reliability",
]

PITCH_OUTCOME = [
    "gseason__pitcher_strike_rate",
    "gseason__pitcher_ball_rate",
    "gseason__pitcher_strike_ball_margin",
    "gseason__pitcher_middle_rate",
    "gseason__pitcher_reverse_rate",
]

PITCHMIX = [
    "gseason__pitcher_fastball_rate",
    "gseason__pitcher_breaking_rate",
    "gseason__pitcher_offspeed_rate",
    "gseason__pitcher_pitchmix_entropy",
]

BATTER_MIDDLE = [
    "gseason__batter_middle_rate",
    "gseason__batter_middle_minus_pitcher_middle",
]

COMPONENT_FULL = (
    SAMPLE_RELIABILITY
    + PITCH_OUTCOME
    + PITCHMIX
    + BATTER_MIDDLE
)


# =============================================================================
# Treatment definitions
# =============================================================================

# New fits only.
NEW_TREATMENTS = {
    "PITCHER_SUCCESS_EXT": (
        PITCHER_SUCCESS_EXT
    ),

    "BATTER_CURRENT_STATE": (
        BATTER_CURRENT_STATE
    ),

    "MATCHUP_CURRENT_DIFF": (
        MATCHUP_CURRENT_DIFF
    ),

    "SUCCESS_NO_MATCHUP": (
        SUCCESS_NO_MATCHUP
    ),

    "SUCCESS_PLUS_OUTCOME": (
        SUCCESS_FULL
        + PITCH_OUTCOME
    ),

    "SUCCESS_PLUS_PITCHMIX": (
        SUCCESS_FULL
        + PITCHMIX
    ),

    "SUCCESS_PLUS_BATTER_MIDDLE": (
        SUCCESS_FULL
        + BATTER_MIDDLE
    ),

    "SUCCESS_PLUS_SAMPLE_RELIABILITY": (
        SUCCESS_FULL
        + SAMPLE_RELIABILITY
    ),
}

# Frozen v038 predictions.
REUSED_TREATMENTS = {
    "SUCCESS_FULL": "STATE_SUCCESS_EXT",
    "SUCCESS_PLUS_FULL_COMPONENT": "STATE_FULL",
}

ALL_TREATMENT_NAMES = (
    list(
        NEW_TREATMENTS.keys()
    )
    + list(
        REUSED_TREATMENTS.keys()
    )
)


# =============================================================================
# Helpers
# =============================================================================

def find_script(pattern):
    direct = (
        ROOT
        / pattern
    )

    if direct.exists():
        return direct

    matches = sorted(
        p
        for p in ROOT.rglob(
            pattern
        )
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and "validation" not in p.parts
    )

    if len(
        matches
    ) == 1:
        print(
            f"[AUTO] source found: {matches[0]}"
        )
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"{pattern} not found."
        )

    raise RuntimeError(
        f"Multiple candidates for {pattern}:\n"
        + "\n".join(
            f"- {p}"
            for p in matches
        )
    )


def import_module(
    name,
    path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            path
        )

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
        mu
        * (
            1.0
            - mu
        ),
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
# Frozen control
# =============================================================================

def load_control():
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
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            f"v028 missing: {missing}"
        )

    df = (
        df.loc[
            pd.to_numeric(
                df[
                    "base_seed"
                ],
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

    df["control_pred"] = (
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
                "control_pred",
            ]
        ]
        .copy()
    )

    if out[
        ID
    ].duplicated().any():
        raise RuntimeError(
            "Control row_id duplicate"
        )

    return out


# =============================================================================
# Frozen v038 treatment OOF
# =============================================================================

def load_reused_v038():
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
        "treatment_raw_pred",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            f"v038 missing: {missing}"
        )

    df = (
        df.loc[
            df[
                "treatment"
            ]
            .isin(
                list(
                    REUSED_TREATMENTS.values()
                )
            )
            & df[
                "route"
            ]
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

    df["treatment_raw_pred"] = (
        pd.to_numeric(
            df[
                "treatment_raw_pred"
            ],
            errors="raise",
        )
        .astype(np.float64)
    )

    return df


# =============================================================================
# R-only routing
# =============================================================================

def route_r_only(
    control,
    treatment,
    game_type,
):
    control = np.asarray(
        control,
        dtype=np.float64,
    )

    treatment = np.asarray(
        treatment,
        dtype=np.float64,
    )

    gt = (
        pd.Series(
            game_type
        )
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    out = control.copy()

    r_mask = (
        gt != "F"
    )

    out[
        r_mask
    ] = treatment[
        r_mask
    ]

    return out


# =============================================================================
# Align control
# =============================================================================

def align_control(
    control_oof,
    df_val,
    val_year,
):
    part = (
        control_oof.loc[
            control_oof[
                SEASON
            ]
            .eq(
                int(
                    val_year
                )
            ),
            [
                ID,
                "y_true",
                "control_pred",
            ],
        ]
        .copy()
    )

    aligned = (
        df_val[
            [
                ID,
                TARGET,
                GAME_TYPE,
            ]
        ]
        .merge(
            part,
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if (
        aligned[
            "control_pred"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            f"Control alignment failed: {val_year}"
        )

    y_raw = (
        aligned[
            TARGET
        ]
        .to_numpy(
            np.float64
        )
    )

    y_oof = (
        aligned[
            "y_true"
        ]
        .to_numpy(
            np.float64
        )
    )

    if np.max(
        np.abs(
            y_raw
            - y_oof
        )
    ) > 1e-12:
        raise RuntimeError(
            f"y mismatch: {val_year}"
        )

    return aligned


# =============================================================================
# Fit new treatment
# =============================================================================

def fit_new_treatment(
    v038,
    src,
    df_tr,
    df_va,
    tr_view,
    va_view,
    base_l1_features,
    extra_features,
):
    features = (
        list(
            base_l1_features
        )
        + list(
            extra_features
        )
    )

    if len(
        features
    ) != len(
        set(
            features
        )
    ):
        raise RuntimeError(
            "Duplicate feature names."
        )

    (
        pred,
        best_iteration,
        elapsed,
    ) = v038.fit_treatment(
        src=src,
        train_view=tr_view,
        val_view=va_view,
        df_train=df_tr,
        df_val=df_va,
        features=features,
    )

    return (
        pred,
        best_iteration,
        elapsed,
        features,
    )


# =============================================================================
# Evaluation summaries
# =============================================================================

def summarize_predictions(
    pred_df,
):
    by_fold_rows = []
    summary_rows = []

    for treatment, g in pred_df.groupby(
        "treatment",
        sort=True,
    ):
        fold_gains = []

        for year, gy in g.groupby(
            SEASON,
            sort=True,
        ):
            y = (
                gy[
                    "y_true"
                ]
                .to_numpy(
                    np.float64
                )
            )

            c = (
                gy[
                    "control_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            p = (
                gy[
                    "pred"
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

            # R-only direct gain
            gt = (
                gy[
                    GAME_TYPE
                ]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
                .to_numpy()
            )

            r = (
                gt != "F"
            )

            r_gain = float(
                bss(
                    y[
                        r
                    ],
                    p[
                        r
                    ],
                )
                - bss(
                    y[
                        r
                    ],
                    c[
                        r
                    ],
                )
            )

            by_fold_rows.append({
                "treatment": str(
                    treatment
                ),
                "year": int(
                    year
                ),
                "gain_vs_control_bss": (
                    gain
                ),
                "R_gain_vs_control_bss": (
                    r_gain
                ),
            })

        y = (
            g[
                "y_true"
            ]
            .to_numpy(
                np.float64
            )
        )

        c = (
            g[
                "control_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        p = (
            g[
                "pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        recent = (
            g[
                SEASON
            ]
            .isin(
                [
                    2023,
                    2024,
                ]
            )
            .to_numpy()
        )

        latest = (
            g[
                SEASON
            ]
            .eq(
                2024
            )
            .to_numpy()
        )

        summary_rows.append({
            "treatment": str(
                treatment
            ),

            "pooled_gain_vs_control_bss": float(
                bss(
                    y,
                    p,
                )
                - bss(
                    y,
                    c,
                )
            ),

            "recent_2023_2024_gain_bss": float(
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

            "gain_2024_bss": float(
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
                    x > 0
                    for x in fold_gains
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

    summary = pd.DataFrame(
        summary_rows
    )

    summary[
        "strong_candidate"
    ] = (
        (
            summary[
                "pooled_gain_vs_control_bss"
            ]
            > 0
        )
        & (
            summary[
                "recent_2023_2024_gain_bss"
            ]
            > 0
        )
        & (
            summary[
                "gain_2024_bss"
            ]
            > 0
        )
        & (
            summary[
                "fold_wins"
            ]
            >= 3
        )
    )

    summary = (
        summary.sort_values(
            [
                "strong_candidate",
                "gain_2024_bss",
                "recent_2023_2024_gain_bss",
                "pooled_gain_vs_control_bss",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return (
        pd.DataFrame(
            by_fold_rows
        ),
        summary,
    )


# =============================================================================
# Incremental vs SUCCESS_FULL
# =============================================================================

def build_incremental_vs_success(
    pred_df,
):
    success = (
        pred_df.loc[
            pred_df[
                "treatment"
            ]
            .eq(
                "SUCCESS_FULL"
            ),
            [
                ID,
                SEASON,
                GAME_TYPE,
                "y_true",
                "pred",
            ],
        ]
        .rename(
            columns={
                "pred": (
                    "success_pred"
                ),
            }
        )
    )

    if success.empty:
        raise RuntimeError(
            "SUCCESS_FULL prediction missing."
        )

    compare_names = [
        "SUCCESS_PLUS_OUTCOME",
        "SUCCESS_PLUS_PITCHMIX",
        "SUCCESS_PLUS_BATTER_MIDDLE",
        "SUCCESS_PLUS_SAMPLE_RELIABILITY",
        "SUCCESS_PLUS_FULL_COMPONENT",
    ]

    by_fold_rows = []
    summary_rows = []

    for treatment in compare_names:
        t = (
            pred_df.loc[
                pred_df[
                    "treatment"
                ]
                .eq(
                    treatment
                ),
                [
                    ID,
                    "pred",
                ],
            ]
            .rename(
                columns={
                    "pred": (
                        "candidate_pred"
                    ),
                }
            )
        )

        if t.empty:
            continue

        x = success.merge(
            t,
            on=ID,
            how="inner",
            validate="one_to_one",
            sort=False,
        )

        fold_deltas = []

        for year, g in x.groupby(
            SEASON,
            sort=True,
        ):
            y = (
                g[
                    "y_true"
                ]
                .to_numpy(
                    np.float64
                )
            )

            s = (
                g[
                    "success_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            p = (
                g[
                    "candidate_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            delta = float(
                bss(
                    y,
                    p,
                )
                - bss(
                    y,
                    s,
                )
            )

            fold_deltas.append(
                delta
            )

            by_fold_rows.append({
                "treatment": (
                    treatment
                ),
                "year": int(
                    year
                ),
                "incremental_vs_success_bss": (
                    delta
                ),
            })

        y = (
            x[
                "y_true"
            ]
            .to_numpy(
                np.float64
            )
        )

        s = (
            x[
                "success_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        p = (
            x[
                "candidate_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        recent = (
            x[
                SEASON
            ]
            .isin(
                [
                    2023,
                    2024,
                ]
            )
            .to_numpy()
        )

        latest = (
            x[
                SEASON
            ]
            .eq(
                2024
            )
            .to_numpy()
        )

        summary_rows.append({
            "treatment": (
                treatment
            ),

            "pooled_incremental_vs_success_bss": float(
                bss(
                    y,
                    p,
                )
                - bss(
                    y,
                    s,
                )
            ),

            "recent_incremental_vs_success_bss": float(
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
                    s[
                        recent
                    ],
                )
            ),

            "2024_incremental_vs_success_bss": float(
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
                    s[
                        latest
                    ],
                )
            ),

            "incremental_fold_wins": int(
                sum(
                    x > 0
                    for x in fold_deltas
                )
            ),

            "n_folds": int(
                len(
                    fold_deltas
                )
            ),

            "min_incremental_fold_bss": float(
                np.min(
                    fold_deltas
                )
            ),

            "mean_incremental_fold_bss": float(
                np.mean(
                    fold_deltas
                )
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    if len(
        summary
    ):
        summary[
            "stable_incremental"
        ] = (
            (
                summary[
                    "pooled_incremental_vs_success_bss"
                ]
                > 0
            )
            & (
                summary[
                    "recent_incremental_vs_success_bss"
                ]
                > 0
            )
            & (
                summary[
                    "2024_incremental_vs_success_bss"
                ]
                > 0
            )
            & (
                summary[
                    "incremental_fold_wins"
                ]
                >= 3
            )
        )

        summary = (
            summary.sort_values(
                [
                    "stable_incremental",
                    "2024_incremental_vs_success_bss",
                    "recent_incremental_vs_success_bss",
                    "pooled_incremental_vs_success_bss",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    return (
        pd.DataFrame(
            by_fold_rows
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

    v021_path = (
        v038.find_v021_source()
    )

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

    train[
        SEASON
    ] = pd.to_numeric(
        train[
            SEASON
        ],
        errors="raise",
    ).astype(int)

    control_oof = (
        load_control()
    )

    reused = (
        load_reused_v038()
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "=" * 136
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 136
    )
    print(
        f"v038 source      : {v038_path}"
    )
    print(
        f"v021 source      : {v021_path}"
    )
    print(
        "Control fits     : 0 (reuse v028 seed42)"
    )
    print(
        "Reused treatment : SUCCESS_FULL / SUCCESS_PLUS_FULL_COMPONENT"
    )
    print(
        f"New fits/fold    : {len(NEW_TREATMENTS)}"
    )
    print(
        "Route            : R_ONLY"
    )
    print(
        "=" * 136
    )

    fold_rows = []
    pred_frames = []
    fit_rows = []

    for cfg in src.EXPANDING_FOLDS:
        fold = int(
            cfg[
                "fold"
            ]
        )

        val_year = int(
            cfg[
                "validation_season"
            ]
        )

        print()
        print(
            "=" * 136
        )
        print(
            f"FOLD {fold}: "
            f"{cfg['train_seasons']} -> {val_year}"
        )
        print(
            "=" * 136
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

        aligned = align_control(
            control_oof=control_oof,
            df_val=df_va,
            val_year=val_year,
        )

        y = (
            aligned[
                "y_true"
            ]
            .to_numpy(
                np.float64
            )
        )

        control = (
            aligned[
                "control_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        gt = (
            aligned[
                GAME_TYPE
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .to_numpy()
        )

        # -------------------------------------------------------------
        # New fits
        # -------------------------------------------------------------

        for (
            treatment_name,
            extra_features,
        ) in NEW_TREATMENTS.items():

            started = time.time()

            (
                raw_pred,
                best_iteration,
                elapsed,
                features,
            ) = fit_new_treatment(
                v038=v038,
                src=src,
                df_tr=df_tr,
                df_va=df_va,
                tr_view=tr_view,
                va_view=va_view,
                base_l1_features=base_l1_features,
                extra_features=extra_features,
            )

            routed = route_r_only(
                control=control,
                treatment=raw_pred,
                game_type=gt,
            )

            gain = float(
                bss(
                    y,
                    routed,
                )
                - bss(
                    y,
                    control,
                )
            )

            print(
                f"{treatment_name:34s} "
                f"best_iter={best_iteration:4d} "
                f"Δ={gain:+.4f} BSS"
            )

            fit_rows.append({
                "fold": fold,
                "validation_year": (
                    val_year
                ),
                "treatment": (
                    treatment_name
                ),
                "n_extra_features": int(
                    len(
                        extra_features
                    )
                ),
                "n_total_features": int(
                    len(
                        features
                    )
                ),
                "best_iteration": int(
                    best_iteration
                ),
                "fit_seconds": float(
                    elapsed
                ),
                "reused": False,
            })

            pred_frames.append(
                pd.DataFrame({
                    ID: (
                        df_va[
                            ID
                        ]
                        .to_numpy()
                    ),
                    SEASON: (
                        int(
                            val_year
                        )
                    ),
                    GAME_TYPE: (
                        gt
                    ),
                    "treatment": (
                        treatment_name
                    ),
                    "y_true": (
                        y
                    ),
                    "control_pred": (
                        control
                    ),
                    "raw_pred": (
                        raw_pred
                    ),
                    "pred": (
                        routed
                    ),
                })
            )

        # -------------------------------------------------------------
        # Reuse v038 treatment predictions
        # -------------------------------------------------------------

        for (
            new_name,
            old_name,
        ) in REUSED_TREATMENTS.items():

            part = (
                reused.loc[
                    reused[
                        "treatment"
                    ]
                    .eq(
                        old_name
                    )
                    & reused[
                        SEASON
                    ]
                    .eq(
                        val_year
                    ),
                    [
                        ID,
                        "treatment_raw_pred",
                    ],
                ]
                .copy()
            )

            joined = (
                df_va[
                    [
                        ID,
                    ]
                ]
                .merge(
                    part,
                    on=ID,
                    how="left",
                    validate="one_to_one",
                    sort=False,
                )
            )

            if (
                joined[
                    "treatment_raw_pred"
                ]
                .isna()
                .any()
            ):
                raise RuntimeError(
                    f"Reuse alignment failed: {new_name}, {val_year}"
                )

            raw_pred = (
                joined[
                    "treatment_raw_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            routed = route_r_only(
                control=control,
                treatment=raw_pred,
                game_type=gt,
            )

            gain = float(
                bss(
                    y,
                    routed,
                )
                - bss(
                    y,
                    control,
                )
            )

            print(
                f"{new_name:34s} "
                f"[REUSED] Δ={gain:+.4f} BSS"
            )

            fit_rows.append({
                "fold": fold,
                "validation_year": (
                    val_year
                ),
                "treatment": (
                    new_name
                ),
                "n_extra_features": (
                    len(
                        SUCCESS_FULL
                    )
                    if new_name
                    == "SUCCESS_FULL"
                    else len(
                        SUCCESS_FULL
                        + COMPONENT_FULL
                    )
                ),
                "n_total_features": np.nan,
                "best_iteration": np.nan,
                "fit_seconds": 0.0,
                "reused": True,
            })

            pred_frames.append(
                pd.DataFrame({
                    ID: (
                        df_va[
                            ID
                        ]
                        .to_numpy()
                    ),
                    SEASON: (
                        int(
                            val_year
                        )
                    ),
                    GAME_TYPE: (
                        gt
                    ),
                    "treatment": (
                        new_name
                    ),
                    "y_true": (
                        y
                    ),
                    "control_pred": (
                        control
                    ),
                    "raw_pred": (
                        raw_pred
                    ),
                    "pred": (
                        routed
                    ),
                })
            )

    pred_df = pd.concat(
        pred_frames,
        ignore_index=True,
    )

    fit_df = pd.DataFrame(
        fit_rows
    )

    (
        fold_summary,
        summary,
    ) = summarize_predictions(
        pred_df
    )

    (
        inc_by_fold,
        inc_summary,
    ) = build_incremental_vs_success(
        pred_df
    )

    # -------------------------------------------------------------------------
    # Feature manifest
    # -------------------------------------------------------------------------

    manifest_rows = []

    all_defs = dict(
        NEW_TREATMENTS
    )

    all_defs[
        "SUCCESS_FULL"
    ] = (
        SUCCESS_FULL
    )

    all_defs[
        "SUCCESS_PLUS_FULL_COMPONENT"
    ] = (
        SUCCESS_FULL
        + COMPONENT_FULL
    )

    for name, feats in all_defs.items():
        for feat in feats:
            if feat in PITCHER_SUCCESS_EXT:
                block = (
                    "PITCHER_SUCCESS"
                )
            elif feat in BATTER_CURRENT_STATE:
                block = (
                    "BATTER_SUCCESS"
                )
            elif feat in MATCHUP_CURRENT_DIFF:
                block = (
                    "MATCHUP_DIFF"
                )
            elif feat in SAMPLE_RELIABILITY:
                block = (
                    "SAMPLE_RELIABILITY"
                )
            elif feat in PITCH_OUTCOME:
                block = (
                    "PITCH_OUTCOME"
                )
            elif feat in PITCHMIX:
                block = (
                    "PITCHMIX"
                )
            elif feat in BATTER_MIDDLE:
                block = (
                    "BATTER_MIDDLE"
                )
            else:
                block = (
                    "OTHER"
                )

            manifest_rows.append({
                "treatment": (
                    name
                ),
                "feature": (
                    feat
                ),
                "block": (
                    block
                ),
            })

    manifest = pd.DataFrame(
        manifest_rows
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    fit_df.to_csv(
        OUT_DIR
        / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        OUT_DIR
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_summary.to_csv(
        OUT_DIR
        / "summary_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    inc_by_fold.to_csv(
        OUT_DIR
        / "incremental_vs_success_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    inc_summary.to_csv(
        OUT_DIR
        / "incremental_vs_success_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest.to_csv(
        OUT_DIR
        / "feature_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 136
    )
    print(
        "CURRENT-STATE BLOCK ABLATION SUMMARY"
    )
    print(
        "=" * 136
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "=" * 136
    )
    print(
        "SUCCESS INTERNAL BLOCKS"
    )
    print(
        "=" * 136
    )

    success_names = [
        "PITCHER_SUCCESS_EXT",
        "BATTER_CURRENT_STATE",
        "MATCHUP_CURRENT_DIFF",
        "SUCCESS_NO_MATCHUP",
        "SUCCESS_FULL",
    ]

    ss = (
        summary.loc[
            summary[
                "treatment"
            ]
            .isin(
                success_names
            )
        ]
        .copy()
    )

    print(
        ss.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "=" * 136
    )
    print(
        "COMPONENT INCREMENTAL VS SUCCESS_FULL"
    )
    print(
        "=" * 136
    )

    print(
        inc_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
        if len(
            inc_summary
        )
        else "None"
    )

    print()
    print(
        "=" * 136
    )
    print(
        "COMPONENT INCREMENTAL BY FOLD"
    )
    print(
        "=" * 136
    )

    print(
        inc_by_fold.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
        if len(
            inc_by_fold
        )
        else "None"
    )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),

        "control": (
            "v028 seed42 B_L1_PC"
        ),

        "route": (
            ROUTE
        ),

        "reused_predictions": (
            REUSED_TREATMENTS
        ),

        "new_treatments": {
            k: list(
                v
            )
            for k, v in NEW_TREATMENTS.items()
        },

        "feature_blocks": {
            "pitcher_success_ext": (
                PITCHER_SUCCESS_EXT
            ),
            "batter_current_state": (
                BATTER_CURRENT_STATE
            ),
            "matchup_current_diff": (
                MATCHUP_CURRENT_DIFF
            ),
            "sample_reliability": (
                SAMPLE_RELIABILITY
            ),
            "pitch_outcome": (
                PITCH_OUTCOME
            ),
            "pitchmix": (
                PITCHMIX
            ),
            "batter_middle": (
                BATTER_MIDDLE
            ),
        },

        "notes": [
            (
                "v040 proved R control prediction is exactly identical "
                "between v028 and v039 shadow control."
            ),
            (
                "Therefore v028 is a valid frozen control for R-only "
                "current-state block ablation."
            ),
            (
                "STATE_SUCCESS_EXT and STATE_FULL are reused from v038 "
                "to avoid unnecessary retraining."
            ),
            (
                "Component blocks are evaluated only on top of SUCCESS_FULL, "
                "because component-only state failed in v038."
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
        "=" * 136
    )


if __name__ == "__main__":
    main()

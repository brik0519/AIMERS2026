#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep.py
=====================================================

목적
----
Final L1 = v42 (Public LB 1004.5151022194) 를 고정하고
old L2 ExtraTrees residual adapter를 정확히 복원하여 새 L1 residual에 재학습한다.

old L2 exact schema
-------------------
adapter core:
    REFERENCE_RAWNUM
    + ENG_COLS
    + code_<REFERENCE_CAT_SMALL>
    + base_pred

then:
    adapter core + Exact-ASOF5

PC는 L2에 넣지 않는다.

ExtraTrees validation:
    n_estimators=100
    max_depth=10
    min_samples_leaf=200
    max_features=0.7

Final deployment:
    n_estimators=120
    max_depth=10
    min_samples_leaf=200
    max_features=0.7

Seeds:
    [42, 2026, 314]
    actual random_state = 16200 + seed

새 validation
-------------
Deployment 구조:
    2024 OOT residual을 학습 -> 2025 test correction

이에 맞춰 동일한 1-season transfer를 과거에 재현:
    2021 residual -> 2022
    2022 residual -> 2023
    2023 residual -> 2024

즉 같은 연도의 label로 같은 연도를 cross-fit하는 대신
전년도 residual만 이용하여 다음 연도를 평가한다.

Shrink grid:
    0.25, 0.50, 0.70, 0.80, 0.90,
    0.95, 1.00, 1.05, 1.10, 1.20

Free routing audit:
    ALL_R
    R_CORE_ONLY
    R_ANCHOR_ONLY

F는 모든 경우 v42 prediction을 정확히 보호한다.

Selection gate
--------------
- pooled(2022~2024) gain > 0
- recent(2023~2024) gain > 0
- 2024 gain > 0
- temporal wins >= 2/3
- 2024 seed wins >= 2/3

통과 후보 중:
    2024 gain -> recent gain -> pooled gain
순으로 선택.

최종 artifact
-------------
선택된 route/shrink가 있으면:
    source = 2024 v42 OOT residual
    fit scope = 2024 R rows
    3 ExtraTrees models
    n_estimators=120

저장:
    model/cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep/
        cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep.pkl
        metadata.json

    validation/cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep/
        transfer_predictions.csv.gz
        transfer_fit_diagnostics.csv
        route_shrink_by_year.csv
        route_shrink_summary.csv
        selected_candidate.json

중요
----
- test.csv를 읽지 않는다.
- 공식 train.csv + v42 OOF만 사용한다.
- OOF는 각 season에 대해 과거 labeled seasons로 학습된 v42 prediction이다.
- F는 L2 correction을 받지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


# =============================================================================
# Config
# =============================================================================

ROOT = Path(".")

EXPERIMENT_NAME = (
    "cb_reg_v048_v42_l2_temporal_transfer_shrink_sweep"
)

TRAIN_CANDIDATES = [
    ROOT / "data" / "train.csv",
    ROOT / "open" / "train.csv",
]

V42_OOF_CANDIDATES = [
    ROOT
    / "validation"
    / "cb_reg_v042_success_full_r_only_submit"
    / "oof_predictions.csv",

    ROOT
    / "validation"
    / "cb_reg_v042_success_full_r_only_submit"
    / "oof_predictions.csv.gz",
]

OLD_L2_SOURCE_CANDIDATES = [
    ROOT
    / "model"
    / "cb_reg_v003_exact_pc_l2"
    / "train.py",
]

VALIDATION_DIR = (
    ROOT
    / "validation"
    / EXPERIMENT_NAME
)

MODEL_DIR = (
    ROOT
    / "model"
    / EXPERIMENT_NAME
)

MODEL_PATH = (
    MODEL_DIR
    / f"{EXPERIMENT_NAME}.pkl"
)

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"

PITCHER_TEAM = "pitcher_team_id"
BATTER_TEAM = "batter_team_id"

ANCHOR_TEAM_ID = 13

TRANSFER_PAIRS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]

ADAPTER_SEEDS = [
    42,
    2026,
    314,
]

VALIDATION_ET_CFG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}

FINAL_ET_CFG = {
    "n_estimators": 120,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}

N_JOBS = max(
    1,
    min(
        6,
        os.cpu_count() or 1,
    ),
)

SHRINK_GRID = [
    0.25,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.20,
]

ROUTES = [
    "ALL_R",
    "R_CORE_ONLY",
    "R_ANCHOR_ONLY",
]

EXPECTED_CORE_FEATURES = 44
EXPECTED_EXACT_FEATURES = 5
EXPECTED_TOTAL_FEATURES = 49

V42_PUBLIC_LB = 1004.5151022194


# =============================================================================
# Exact recovered constants
# =============================================================================

REFERENCE_RAWNUM = [
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "pitcher_team_id",
    "batter_team_id",
]

REFERENCE_CAT_SMALL = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

ENG_COLS = [
    "eng_recent35",
    "eng_recent_delta",
    "eng_sb",
    "eng_same",
    "eng_close",
    "eng_late",
]

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


# =============================================================================
# Utilities
# =============================================================================

def first_existing(
    candidates,
    label,
):
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"{label} not found. Checked:\n"
        + "\n".join(
            f"  - {p}"
            for p in candidates
        )
    )


def import_module_from_path(
    name,
    path,
):
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Could not import: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def brier(
    y,
    pred,
):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    return float(
        np.mean(
            (
                y
                - pred
            )
            ** 2
        )
    )


def bss(
    y,
    pred,
):
    """
    Project BSS scale.
    Larger is better.
    """
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    p = float(
        np.mean(
            y
        )
    )

    ref = (
        p
        * (
            1.0
            - p
        )
    )

    if ref <= 0.0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier(
                y,
                pred,
            )
            / ref
        )
    )


def numeric_series(
    frame,
    col,
    default=np.nan,
):
    if col not in frame.columns:
        return pd.Series(
            np.full(
                len(frame),
                default,
                dtype=np.float64,
            ),
            index=frame.index,
        )

    return pd.to_numeric(
        frame[
            col
        ],
        errors="coerce",
    )


# =============================================================================
# Exact old L2 adapter core
# =============================================================================

def add_eng(
    frame,
):
    x = frame.copy()

    career = numeric_series(
        x,
        "asof_pitcher_success_rate",
        0.5,
    )

    p1 = numeric_series(
        x,
        "asof_pitcher_prev1_game_success_rate",
        np.nan,
    )

    p3 = numeric_series(
        x,
        "asof_pitcher_prev3_game_success_rate",
        np.nan,
    )

    p5 = numeric_series(
        x,
        "asof_pitcher_prev5_game_success_rate",
        np.nan,
    )

    p1 = p1.fillna(
        career
    )

    p3 = p3.fillna(
        p1
    )

    p5 = p5.fillna(
        p3
    )

    x[
        "eng_recent35"
    ] = (
        0.5
        * p1
        + 0.3
        * p3
        + 0.2
        * p5
    )

    x[
        "eng_recent_delta"
    ] = (
        x[
            "eng_recent35"
        ]
        - career
    )

    x[
        "eng_sb"
    ] = (
        numeric_series(
            x,
            "asof_pitcher_strike_rate",
            np.nan,
        )
        - numeric_series(
            x,
            "asof_pitcher_ball_rate",
            np.nan,
        )
    )

    if (
        "pitcher_hand"
        in x.columns
        and
        "batter_hand"
        in x.columns
    ):
        x[
            "eng_same"
        ] = (
            x[
                "pitcher_hand"
            ]
            .astype(str)
            .eq(
                x[
                    "batter_hand"
                ]
                .astype(str)
            )
            .astype(
                np.float32
            )
        )

    else:
        x[
            "eng_same"
        ] = 0.0

    x[
        "eng_close"
    ] = (
        numeric_series(
            x,
            "score_diff_pitcher_team",
            np.nan,
        )
        .abs()
        .le(1)
        .astype(
            np.float32
        )
    )

    x[
        "eng_late"
    ] = (
        numeric_series(
            x,
            "inning",
            np.nan,
        )
        .ge(7)
        .astype(
            np.float32
        )
    )

    return x


def fit_adapter_prep(
    map_rows,
):
    """
    Fit only on source-year official train rows.
    """

    m = add_eng(
        map_rows.reset_index(
            drop=True
        )
    )

    numeric_cols = (
        [
            c
            for c in REFERENCE_RAWNUM
            if c in m.columns
        ]
        + ENG_COLS
    )

    missing_eng = [
        c
        for c in ENG_COLS
        if c not in m.columns
    ]

    if missing_eng:
        raise KeyError(
            f"ENG columns missing: {missing_eng}"
        )

    med = (
        m[
            numeric_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .median()
        .fillna(
            0.0
        )
        .to_dict()
    )

    cat_cols = [
        c
        for c in REFERENCE_CAT_SMALL
        if c in m.columns
    ]

    maps = {}

    for col in cat_cols:
        s = (
            m[
                col
            ]
            .astype(
                "string"
            )
            .fillna(
                "<NA>"
            )
            .astype(str)
        )

        maps[
            col
        ] = {
            value: idx
            for idx, value in enumerate(
                pd.Index(
                    s.unique()
                )
            )
        }

    return {
        "numeric_cols": (
            numeric_cols
        ),
        "med": med,
        "maps": maps,
    }


def transform_adapter_core(
    prep,
    rows,
    base_pred,
):
    r = add_eng(
        rows.reset_index(
            drop=True
        )
    )

    A = (
        r[
            prep[
                "numeric_cols"
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .copy()
    )

    for col in prep[
        "numeric_cols"
    ]:
        A[
            col
        ] = (
            A[
                col
            ]
            .fillna(
                float(
                    prep[
                        "med"
                    ].get(
                        col,
                        0.0,
                    )
                )
            )
            .astype(
                np.float32
            )
        )

    for (
        col,
        mapping,
    ) in prep[
        "maps"
    ].items():
        A[
            "code_"
            + col
        ] = (
            r[
                col
            ]
            .astype(
                "string"
            )
            .fillna(
                "<NA>"
            )
            .astype(str)
            .map(
                mapping
            )
            .fillna(
                -1
            )
            .astype(
                np.float32
            )
        )

    A[
        "base_pred"
    ] = np.asarray(
        base_pred,
        dtype=np.float32,
    )

    if not np.isfinite(
        A.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "adapter core contains NaN/Inf"
        )

    return A.reset_index(
        drop=True
    )


# =============================================================================
# Exact5 bridge to old exact implementation
# =============================================================================

def validate_old_source(
    old,
):
    required_functions = [
        "build_pitcher_season_stats",
        "add_pitcher_exact_asof_features",
    ]

    for fn in required_functions:
        if not hasattr(
            old,
            fn,
        ):
            raise AttributeError(
                f"old L2 source missing function: {fn}"
            )

    # Hard recovered-schema checks.
    if list(
        getattr(
            old,
            "EXACT_FEATURES",
            [],
        )
    ) != EXACT_FEATURES:
        raise RuntimeError(
            "Old source Exact5 mismatch."
        )

    if list(
        getattr(
            old,
            "REFERENCE_RAWNUM",
            [],
        )
    ) != REFERENCE_RAWNUM:
        raise RuntimeError(
            "Old source REFERENCE_RAWNUM mismatch."
        )

    if list(
        getattr(
            old,
            "REFERENCE_CAT_SMALL",
            [],
        )
    ) != REFERENCE_CAT_SMALL:
        raise RuntimeError(
            "Old source REFERENCE_CAT_SMALL mismatch."
        )

    if list(
        getattr(
            old,
            "ENG_COLS",
            [],
        )
    ) != ENG_COLS:
        raise RuntimeError(
            "Old source ENG_COLS mismatch."
        )


def exact5_matrix(
    old,
    rows,
    pitcher_season_stats,
):
    result = (
        old.add_pitcher_exact_asof_features(
            rows,
            pitcher_season_stats,
        )
    )

    # Old source may return either dataframe or (dataframe, diagnostics).
    if isinstance(
        result,
        tuple,
    ):
        transformed = (
            result[
                0
            ]
        )

    else:
        transformed = result

    exact = (
        transformed[
            EXACT_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .astype(
            np.float32
        )
        .reset_index(
            drop=True
        )
    )

    if not np.isfinite(
        exact.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Exact5 matrix contains NaN/Inf."
        )

    return exact


def build_adapter_matrix(
    old,
    prep,
    rows,
    base_pred,
    pitcher_season_stats,
):
    core = (
        transform_adapter_core(
            prep=prep,
            rows=rows,
            base_pred=base_pred,
        )
    )

    exact = (
        exact5_matrix(
            old=old,
            rows=rows,
            pitcher_season_stats=(
                pitcher_season_stats
            ),
        )
    )

    X = pd.concat(
        [
            core,
            exact,
        ],
        axis=1,
    )

    if (
        X.columns
        .duplicated()
        .any()
    ):
        dup = (
            X.columns[
                X.columns
                .duplicated()
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate L2 columns: {dup}"
        )

    if not np.isfinite(
        X.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "L2 matrix contains NaN/Inf."
        )

    if (
        core.shape[
            1
        ]
        != EXPECTED_CORE_FEATURES
    ):
        raise RuntimeError(
            "Adapter core feature count mismatch: "
            f"{core.shape[1]} != {EXPECTED_CORE_FEATURES}"
        )

    if (
        exact.shape[
            1
        ]
        != EXPECTED_EXACT_FEATURES
    ):
        raise RuntimeError(
            "Exact feature count mismatch."
        )

    if (
        X.shape[
            1
        ]
        != EXPECTED_TOTAL_FEATURES
    ):
        raise RuntimeError(
            "Total L2 feature count mismatch: "
            f"{X.shape[1]} != {EXPECTED_TOTAL_FEATURES}"
        )

    return (
        X,
        list(
            core.columns
        ),
        list(
            exact.columns
        ),
    )


# =============================================================================
# Data alignment / routing
# =============================================================================

def align_train_and_oof(
    train,
    oof,
):
    required_oof = {
        ID,
        SEASON,
        "y_true",
        "routed_pred",
    }

    missing_oof = sorted(
        required_oof
        - set(
            oof.columns
        )
    )

    if missing_oof:
        raise KeyError(
            f"v42 OOF missing: {missing_oof}"
        )

    if oof[
        ID
    ].duplicated().any():
        raise RuntimeError(
            "v42 OOF row_id duplicated."
        )

    if train[
        ID
    ].duplicated().any():
        raise RuntimeError(
            "train row_id duplicated."
        )

    o = (
        oof[
            [
                ID,
                SEASON,
                "y_true",
                "routed_pred",
            ]
        ]
        .rename(
            columns={
                SEASON: (
                    "oof_season"
                ),
                "y_true": (
                    "oof_y"
                ),
                "routed_pred": (
                    "v42_pred"
                ),
            }
        )
    )

    merged = (
        train.merge(
            o,
            on=ID,
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        merged
    ) != len(
        oof
    ):
        raise RuntimeError(
            "train/v42 OOF row count mismatch: "
            f"{len(merged)} != {len(oof)}"
        )

    train_year = (
        pd.to_numeric(
            merged[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    oof_year = (
        pd.to_numeric(
            merged[
                "oof_season"
            ],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        train_year,
        oof_year,
    ):
        raise RuntimeError(
            "train season != OOF season."
        )

    y = (
        pd.to_numeric(
            merged[
                TARGET
            ],
            errors="raise",
        )
        .to_numpy(
            np.float64
        )
    )

    oof_y = (
        pd.to_numeric(
            merged[
                "oof_y"
            ],
            errors="raise",
        )
        .to_numpy(
            np.float64
        )
    )

    max_diff = float(
        np.max(
            np.abs(
                y
                - oof_y
            )
        )
    )

    if max_diff > 1e-12:
        raise RuntimeError(
            f"train target != OOF y_true: max diff={max_diff}"
        )

    merged[
        "v42_pred"
    ] = pd.to_numeric(
        merged[
            "v42_pred"
        ],
        errors="raise",
    ).astype(
        np.float64
    )

    if not np.isfinite(
        merged[
            "v42_pred"
        ]
        .to_numpy(
            np.float64
        )
    ).all():
        raise RuntimeError(
            "v42 OOF contains NaN/Inf."
        )

    return merged


def domain_masks(
    rows,
):
    gt = (
        rows[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    is_f = (
        gt == "F"
    )

    is_anchor = (
        rows[
            PITCHER_TEAM
        ]
        .eq(
            ANCHOR_TEAM_ID
        )
        .to_numpy()
        |
        rows[
            BATTER_TEAM
        ]
        .eq(
            ANCHOR_TEAM_ID
        )
        .to_numpy()
    )

    return {
        "F": (
            is_f
        ),
        "R": (
            ~is_f
        ),
        "R_ANCHOR": (
            (~is_f)
            & is_anchor
        ),
        "R_CORE": (
            (~is_f)
            & (~is_anchor)
        ),
    }


def route_apply_mask(
    rows,
    route,
):
    masks = domain_masks(
        rows
    )

    if route == "ALL_R":
        return masks[
            "R"
        ]

    if route == "R_CORE_ONLY":
        return masks[
            "R_CORE"
        ]

    if route == "R_ANCHOR_ONLY":
        return masks[
            "R_ANCHOR"
        ]

    raise ValueError(
        route
    )


# =============================================================================
# Transfer fits
# =============================================================================

def fit_transfer_models(
    X_source,
    source_rows,
    residual_source,
):
    masks = domain_masks(
        source_rows
    )

    r_mask = (
        masks[
            "R"
        ]
    )

    minimum = (
        VALIDATION_ET_CFG[
            "min_samples_leaf"
        ]
        * 2
    )

    n_r = int(
        r_mask.sum()
    )

    if n_r <= minimum:
        raise RuntimeError(
            f"Source R rows insufficient: {n_r} <= {minimum}"
        )

    models = {}
    rows = []

    for seed in ADAPTER_SEEDS:
        model = (
            ExtraTreesRegressor(
                n_jobs=N_JOBS,
                random_state=(
                    16200
                    + int(
                        seed
                    )
                ),
                **VALIDATION_ET_CFG,
            )
        )

        t0 = time.time()

        model.fit(
            X_source.loc[
                r_mask
            ],
            residual_source[
                r_mask
            ],
        )

        elapsed = (
            time.time()
            - t0
        )

        models[
            int(
                seed
            )
        ] = model

        train_corr = np.asarray(
            model.predict(
                X_source.loc[
                    r_mask
                ]
            ),
            dtype=np.float64,
        )

        rows.append({
            "seed": int(
                seed
            ),
            "n_source_R": (
                n_r
            ),
            "fit_seconds": float(
                elapsed
            ),
            "train_corr_mean": float(
                train_corr.mean()
            ),
            "train_corr_std": float(
                train_corr.std()
            ),
            "train_corr_abs_mean": float(
                np.mean(
                    np.abs(
                        train_corr
                    )
                )
            ),
        })

    return (
        models,
        pd.DataFrame(
            rows
        ),
    )


def predict_models(
    models,
    X_target,
):
    per_seed = {}

    for seed in ADAPTER_SEEDS:
        pred = np.asarray(
            models[
                int(
                    seed
                )
            ].predict(
                X_target
            ),
            dtype=np.float64,
        )

        if not np.isfinite(
            pred
        ).all():
            raise RuntimeError(
                f"L2 correction NaN/Inf seed={seed}"
            )

        per_seed[
            int(
                seed
            )
        ] = pred

    matrix = np.column_stack(
        [
            per_seed[
                int(
                    seed
                )
            ]
            for seed in ADAPTER_SEEDS
        ]
    )

    ensemble = (
        matrix.mean(
            axis=1
        )
    )

    return (
        ensemble,
        per_seed,
    )


# =============================================================================
# Evaluation
# =============================================================================

def routed_prediction(
    rows,
    base_pred,
    correction,
    shrink,
    route,
):
    base = np.asarray(
        base_pred,
        dtype=np.float64,
    )

    corr = np.asarray(
        correction,
        dtype=np.float64,
    )

    pred = (
        base.copy()
    )

    apply_mask = (
        route_apply_mask(
            rows,
            route,
        )
    )

    pred[
        apply_mask
    ] = np.clip(
        base[
            apply_mask
        ]
        + float(
            shrink
        )
        * corr[
            apply_mask
        ],
        0.0,
        1.0,
    )

    # Every non-applied row is exactly protected.
    protected = (
        ~apply_mask
    )

    if (
        protected.any()
        and not np.array_equal(
            pred[
                protected
            ],
            base[
                protected
            ],
        )
    ):
        raise RuntimeError(
            f"Route protection failed: {route}"
        )

    return pred


def build_grid_metrics(
    transfer_pred,
):
    by_year_rows = []
    summary_rows = []

    for route in ROUTES:
        for shrink in SHRINK_GRID:
            gains = []

            for target_year, g in (
                transfer_pred.groupby(
                    "target_year",
                    sort=True,
                )
            ):
                y = (
                    g[
                        "y_true"
                    ]
                    .to_numpy(
                        np.float64
                    )
                )

                base = (
                    g[
                        "v42_pred"
                    ]
                    .to_numpy(
                        np.float64
                    )
                )

                corr = (
                    g[
                        "corr_mean"
                    ]
                    .to_numpy(
                        np.float64
                    )
                )

                pred = (
                    routed_prediction(
                        rows=g,
                        base_pred=base,
                        correction=corr,
                        shrink=shrink,
                        route=route,
                    )
                )

                gain = float(
                    bss(
                        y,
                        pred,
                    )
                    - bss(
                        y,
                        base,
                    )
                )

                gains.append(
                    gain
                )

                by_year_rows.append({
                    "route": (
                        route
                    ),
                    "shrink": float(
                        shrink
                    ),
                    "year": int(
                        target_year
                    ),
                    "n": int(
                        len(
                            g
                        )
                    ),
                    "gain_vs_v42_bss": (
                        gain
                    ),
                })

            y_all = (
                transfer_pred[
                    "y_true"
                ]
                .to_numpy(
                    np.float64
                )
            )

            base_all = (
                transfer_pred[
                    "v42_pred"
                ]
                .to_numpy(
                    np.float64
                )
            )

            corr_all = (
                transfer_pred[
                    "corr_mean"
                ]
                .to_numpy(
                    np.float64
                )
            )

            pred_all = (
                routed_prediction(
                    rows=(
                        transfer_pred
                    ),
                    base_pred=(
                        base_all
                    ),
                    correction=(
                        corr_all
                    ),
                    shrink=(
                        shrink
                    ),
                    route=(
                        route
                    ),
                )
            )

            years = (
                transfer_pred[
                    "target_year"
                ]
                .to_numpy(
                    int
                )
            )

            recent = np.isin(
                years,
                [
                    2023,
                    2024,
                ],
            )

            latest = (
                years
                == 2024
            )

            pooled_gain = float(
                bss(
                    y_all,
                    pred_all,
                )
                - bss(
                    y_all,
                    base_all,
                )
            )

            recent_gain = float(
                bss(
                    y_all[
                        recent
                    ],
                    pred_all[
                        recent
                    ],
                )
                - bss(
                    y_all[
                        recent
                    ],
                    base_all[
                        recent
                    ],
                )
            )

            latest_gain = float(
                bss(
                    y_all[
                        latest
                    ],
                    pred_all[
                        latest
                    ],
                )
                - bss(
                    y_all[
                        latest
                    ],
                    base_all[
                        latest
                    ],
                )
            )

            wins = int(
                np.sum(
                    np.asarray(
                        gains
                    )
                    > 0.0
                )
            )

            # ---------------------------------------------
            # 2024 per-seed stability
            # ---------------------------------------------
            g24 = (
                transfer_pred.loc[
                    transfer_pred[
                        "target_year"
                    ].eq(
                        2024
                    )
                ]
                .reset_index(
                    drop=True
                )
            )

            seed_gains = []

            for seed in ADAPTER_SEEDS:
                corr_seed = (
                    g24[
                        f"corr_seed_{seed}"
                    ]
                    .to_numpy(
                        np.float64
                    )
                )

                pred_seed = (
                    routed_prediction(
                        rows=g24,
                        base_pred=(
                            g24[
                                "v42_pred"
                            ]
                            .to_numpy(
                                np.float64
                            )
                        ),
                        correction=(
                            corr_seed
                        ),
                        shrink=(
                            shrink
                        ),
                        route=(
                            route
                        ),
                    )
                )

                seed_gain = float(
                    bss(
                        g24[
                            "y_true"
                        ]
                        .to_numpy(
                            np.float64
                        ),
                        pred_seed,
                    )
                    - bss(
                        g24[
                            "y_true"
                        ]
                        .to_numpy(
                            np.float64
                        ),
                        g24[
                            "v42_pred"
                        ]
                        .to_numpy(
                            np.float64
                        ),
                    )
                )

                seed_gains.append(
                    seed_gain
                )

            seed_wins = int(
                np.sum(
                    np.asarray(
                        seed_gains
                    )
                    > 0.0
                )
            )

            passed = bool(
                pooled_gain > 0.0
                and recent_gain > 0.0
                and latest_gain > 0.0
                and wins >= 2
                and seed_wins >= 2
            )

            summary_rows.append({
                "route": (
                    route
                ),
                "shrink": float(
                    shrink
                ),
                "pooled_2022_2024_gain_bss": (
                    pooled_gain
                ),
                "recent_2023_2024_gain_bss": (
                    recent_gain
                ),
                "gain_2024_bss": (
                    latest_gain
                ),
                "temporal_wins": (
                    wins
                ),
                "n_temporal_folds": 3,
                "min_temporal_gain_bss": float(
                    np.min(
                        gains
                    )
                ),
                "mean_temporal_gain_bss": float(
                    np.mean(
                        gains
                    )
                ),
                "seed42_2024_gain_bss": float(
                    seed_gains[
                        0
                    ]
                ),
                "seed2026_2024_gain_bss": float(
                    seed_gains[
                        1
                    ]
                ),
                "seed314_2024_gain_bss": float(
                    seed_gains[
                        2
                    ]
                ),
                "seed_wins_2024": (
                    seed_wins
                ),
                "pass_gate": (
                    passed
                ),
            })

    summary = pd.DataFrame(
        summary_rows
    )

    summary = (
        summary.sort_values(
            by=[
                "pass_gate",
                "gain_2024_bss",
                "recent_2023_2024_gain_bss",
                "pooled_2022_2024_gain_bss",
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
            by_year_rows
        ),
        summary,
    )


def choose_candidate(
    summary,
):
    passed = (
        summary.loc[
            summary[
                "pass_gate"
            ].astype(bool)
        ]
        .copy()
    )

    if len(
        passed
    ) == 0:
        return {
            "selected": False,
            "route": None,
            "shrink": 0.0,
            "reason": (
                "No L2 route/shrink passed temporal + latest + seed gate."
            ),
            "recommended_action": (
                "Keep v42 L1 without L2."
            ),
        }

    best = (
        passed.iloc[
            0
        ]
    )

    return {
        "selected": True,
        "route": str(
            best[
                "route"
            ]
        ),
        "shrink": float(
            best[
                "shrink"
            ]
        ),
        "pooled_gain_bss": float(
            best[
                "pooled_2022_2024_gain_bss"
            ]
        ),
        "recent_gain_bss": float(
            best[
                "recent_2023_2024_gain_bss"
            ]
        ),
        "gain_2024_bss": float(
            best[
                "gain_2024_bss"
            ]
        ),
        "temporal_wins": int(
            best[
                "temporal_wins"
            ]
        ),
        "seed_wins_2024": int(
            best[
                "seed_wins_2024"
            ]
        ),
        "reason": (
            "Best passing candidate by 2024 -> recent -> pooled."
        ),
        "recommended_action": (
            "Use 2024 residual-trained final L2 artifact for 2025 inference."
        ),
    }


# =============================================================================
# Final 2024 -> 2025 L2 fit
# =============================================================================

def fit_final_models(
    X24,
    raw24,
    residual24,
):
    masks = (
        domain_masks(
            raw24
        )
    )

    r_mask = (
        masks[
            "R"
        ]
    )

    minimum = (
        FINAL_ET_CFG[
            "min_samples_leaf"
        ]
        * 2
    )

    if int(
        r_mask.sum()
    ) <= minimum:
        raise RuntimeError(
            "Final 2024 R rows insufficient."
        )

    models = {}
    diagnostics = []

    for seed in ADAPTER_SEEDS:
        model = (
            ExtraTreesRegressor(
                n_jobs=N_JOBS,
                random_state=(
                    16200
                    + int(
                        seed
                    )
                ),
                **FINAL_ET_CFG,
            )
        )

        t0 = time.time()

        model.fit(
            X24.loc[
                r_mask
            ],
            residual24[
                r_mask
            ],
        )

        elapsed = (
            time.time()
            - t0
        )

        corr = np.asarray(
            model.predict(
                X24.loc[
                    r_mask
                ]
            ),
            dtype=np.float64,
        )

        models[
            int(
                seed
            )
        ] = model

        diagnostics.append({
            "seed": int(
                seed
            ),
            "n_train_R": int(
                r_mask.sum()
            ),
            "fit_seconds": float(
                elapsed
            ),
            "train_corr_mean": float(
                corr.mean()
            ),
            "train_corr_std": float(
                corr.std()
            ),
            "train_corr_abs_mean": float(
                np.mean(
                    np.abs(
                        corr
                    )
                )
            ),
        })

    return (
        models,
        pd.DataFrame(
            diagnostics
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if (
        VALIDATION_DIR.exists()
        or MODEL_DIR.exists()
    ):
        raise FileExistsError(
            "Existing v048 output detected. "
            "Delete/rename old validation/model directories first."
        )

    train_path = (
        first_existing(
            TRAIN_CANDIDATES,
            "train.csv",
        )
    )

    oof_path = (
        first_existing(
            V42_OOF_CANDIDATES,
            "v42 OOF",
        )
    )

    old_source_path = (
        first_existing(
            OLD_L2_SOURCE_CANDIDATES,
            "old L2 train.py",
        )
    )

    old = (
        import_module_from_path(
            "old_l2_exact_source_v048",
            old_source_path,
        )
    )

    validate_old_source(
        old
    )

    print(
        "=" * 128
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 128
    )
    print(
        f"train       : {train_path}"
    )
    print(
        f"v42 OOF     : {oof_path}"
    )
    print(
        f"old L2 src  : {old_source_path}"
    )
    print(
        f"Final L1 LB : {V42_PUBLIC_LB:.10f}"
    )
    print(
        f"transfer    : {TRANSFER_PAIRS}"
    )
    print(
        f"routes      : {ROUTES}"
    )
    print(
        f"shrinks     : {SHRINK_GRID}"
    )
    print(
        f"seeds       : {ADAPTER_SEEDS}"
    )
    print(
        f"validation ET: {VALIDATION_ET_CFG}"
    )
    print(
        f"final ET     : {FINAL_ET_CFG}"
    )
    print(
        "=" * 128
    )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------
    train = pd.read_csv(
        train_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    oof = pd.read_csv(
        oof_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    train[
        SEASON
    ] = (
        pd.to_numeric(
            train[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    aligned = (
        align_train_and_oof(
            train=train,
            oof=oof,
        )
    )

    required_raw = (
        set(
            REFERENCE_RAWNUM
        )
        | set(
            REFERENCE_CAT_SMALL
        )
        | {
            ID,
            TARGET,
            SEASON,
            GAME_TYPE,
            PITCHER_TEAM,
            BATTER_TEAM,
        }
    )

    missing_raw = sorted(
        required_raw
        - set(
            aligned.columns
        )
    )

    if missing_raw:
        raise KeyError(
            f"train missing L2 raw columns: {missing_raw}"
        )

    # Full official train stats are safe because exact transform uses
    # completed seasons strictly earlier than each row's own season.
    pitcher_stats = (
        old.build_pitcher_season_stats(
            train
        )
    )

    # -----------------------------------------------------------------
    # Temporal transfers
    # -----------------------------------------------------------------
    prediction_frames = []
    fit_diag_frames = []

    for (
        source_year,
        target_year,
    ) in TRANSFER_PAIRS:
        print()
        print(
            "=" * 128
        )
        print(
            f"TRANSFER {source_year} residual -> {target_year}"
        )
        print(
            "=" * 128
        )

        src = (
            aligned.loc[
                aligned[
                    SEASON
                ].eq(
                    source_year
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        tgt = (
            aligned.loc[
                aligned[
                    SEASON
                ].eq(
                    target_year
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if (
            len(
                src
            ) == 0
            or
            len(
                tgt
            ) == 0
        ):
            raise RuntimeError(
                f"Missing transfer rows: {source_year}->{target_year}"
            )

        y_src = (
            src[
                TARGET
            ]
            .to_numpy(
                np.float64
            )
        )

        base_src = (
            src[
                "v42_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        residual_src = (
            y_src
            - base_src
        )

        prep = (
            fit_adapter_prep(
                src
            )
        )

        (
            X_src,
            core_cols,
            exact_cols,
        ) = (
            build_adapter_matrix(
                old=old,
                prep=prep,
                rows=src,
                base_pred=base_src,
                pitcher_season_stats=(
                    pitcher_stats
                ),
            )
        )

        base_tgt = (
            tgt[
                "v42_pred"
            ]
            .to_numpy(
                np.float64
            )
        )

        (
            X_tgt,
            core_cols_t,
            exact_cols_t,
        ) = (
            build_adapter_matrix(
                old=old,
                prep=prep,
                rows=tgt,
                base_pred=base_tgt,
                pitcher_season_stats=(
                    pitcher_stats
                ),
            )
        )

        if (
            list(
                X_src.columns
            )
            != list(
                X_tgt.columns
            )
        ):
            raise RuntimeError(
                f"Feature order mismatch: {source_year}->{target_year}"
            )

        if (
            core_cols
            != core_cols_t
            or
            exact_cols
            != exact_cols_t
        ):
            raise RuntimeError(
                "Core/Exact column mismatch."
            )

        (
            models,
            fit_diag,
        ) = (
            fit_transfer_models(
                X_source=X_src,
                source_rows=src,
                residual_source=(
                    residual_src
                ),
            )
        )

        fit_diag[
            "source_year"
        ] = int(
            source_year
        )

        fit_diag[
            "target_year"
        ] = int(
            target_year
        )

        fit_diag_frames.append(
            fit_diag
        )

        (
            corr_mean,
            corr_per_seed,
        ) = (
            predict_models(
                models=models,
                X_target=X_tgt,
            )
        )

        frame = (
            tgt[
                [
                    ID,
                    SEASON,
                    GAME_TYPE,
                    PITCHER_TEAM,
                    BATTER_TEAM,
                    TARGET,
                    "v42_pred",
                ]
            ]
            .copy()
        )

        frame[
            "source_year"
        ] = int(
            source_year
        )

        frame[
            "target_year"
        ] = int(
            target_year
        )

        frame[
            "y_true"
        ] = (
            frame[
                TARGET
            ]
            .to_numpy(
                np.float64
            )
        )

        frame[
            "corr_mean"
        ] = (
            corr_mean
        )

        for seed in ADAPTER_SEEDS:
            frame[
                f"corr_seed_{seed}"
            ] = (
                corr_per_seed[
                    int(
                        seed
                    )
                ]
            )

        masks = (
            domain_masks(
                tgt
            )
        )

        frame[
            "domain3"
        ] = np.where(
            masks[
                "F"
            ],
            "F",
            np.where(
                masks[
                    "R_ANCHOR"
                ],
                "R_ANCHOR",
                "R_CORE",
            ),
        )

        prediction_frames.append(
            frame
        )

        print(
            f"source rows={len(src):,} "
            f"| target rows={len(tgt):,} "
            f"| L2 features={X_src.shape[1]}"
        )

        print(
            f"source residual mean={residual_src.mean():+.8f} "
            f"std={residual_src.std():.8f}"
        )

        print(
            f"target corr mean={corr_mean.mean():+.8f} "
            f"abs={np.mean(np.abs(corr_mean)):.8f}"
        )

    transfer_pred = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    fit_diag_all = pd.concat(
        fit_diag_frames,
        ignore_index=True,
    )

    (
        by_year,
        summary,
    ) = build_grid_metrics(
        transfer_pred
    )

    selection = (
        choose_candidate(
            summary
        )
    )

    # -----------------------------------------------------------------
    # Final 2024 residual -> 2025 artifact
    # -----------------------------------------------------------------
    raw24 = (
        aligned.loc[
            aligned[
                SEASON
            ].eq(
                2024
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        raw24
    ) == 0:
        raise RuntimeError(
            "2024 OOF source rows missing."
        )

    y24 = (
        raw24[
            TARGET
        ]
        .to_numpy(
            np.float64
        )
    )

    base24 = (
        raw24[
            "v42_pred"
        ]
        .to_numpy(
            np.float64
        )
    )

    residual24 = (
        y24
        - base24
    )

    prep24 = (
        fit_adapter_prep(
            raw24
        )
    )

    (
        X24,
        core24_cols,
        exact24_cols,
    ) = (
        build_adapter_matrix(
            old=old,
            prep=prep24,
            rows=raw24,
            base_pred=base24,
            pitcher_season_stats=(
                pitcher_stats
            ),
        )
    )

    (
        final_models,
        final_fit_diag,
    ) = (
        fit_final_models(
            X24=X24,
            raw24=raw24,
            residual24=(
                residual24
            ),
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    transfer_pred.to_csv(
        VALIDATION_DIR
        / "transfer_predictions.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    fit_diag_all.to_csv(
        VALIDATION_DIR
        / "transfer_fit_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_year.to_csv(
        VALIDATION_DIR
        / "route_shrink_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        VALIDATION_DIR
        / "route_shrink_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_fit_diag.to_csv(
        VALIDATION_DIR
        / "final_2024_fit_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        VALIDATION_DIR
        / "selected_candidate.json"
    ).write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base_l1": (
            "cb_reg_v042_success_full_r_only_submit"
        ),
        "base_l1_public_lb": (
            V42_PUBLIC_LB
        ),
        "model_type": (
            "v42 L1 + ExtraTrees residual L2"
        ),
        "source_year": 2024,
        "expected_target_year": 2025,
        "selected_route": (
            selection[
                "route"
            ]
        ),
        "selected_shrink": float(
            selection[
                "shrink"
            ]
        ),
        "selection_passed": bool(
            selection[
                "selected"
            ]
        ),
        "adapter_models": (
            final_models
        ),
        "adapter_seeds": list(
            ADAPTER_SEEDS
        ),
        "validation_cfg": dict(
            VALIDATION_ET_CFG
        ),
        "final_cfg": dict(
            FINAL_ET_CFG
        ),
        "adapter_prep": (
            prep24
        ),
        "adapter_feature_cols": list(
            X24.columns
        ),
        "adapter_core_feature_cols": (
            core24_cols
        ),
        "exact_cols": (
            exact24_cols
        ),
        "pitcher_season_stats": (
            pitcher_stats
        ),
        "reference_rawnum": list(
            REFERENCE_RAWNUM
        ),
        "reference_cat_small": list(
            REFERENCE_CAT_SMALL
        ),
        "eng_cols": list(
            ENG_COLS
        ),
        "anchor_team_id": (
            ANCHOR_TEAM_ID
        ),
        "routing_policy": {
            "F": (
                "always v42 protected"
            ),
            "ALL_R": (
                "apply L2 to game_type != F"
            ),
            "R_CORE_ONLY": (
                "apply L2 only to non-F rows without anchor team 13"
            ),
            "R_ANCHOR_ONLY": (
                "apply L2 only to non-F rows involving anchor team 13"
            ),
        },
        "residual_definition": (
            "official_y_2024 - v42_2019_2023_to_2024_OOT_routed_prediction"
        ),
        "rule_compliance": {
            "test_rows_used_during_training": False,
            "test_row_aggregation": False,
            "l2_source": (
                "official 2024 labels + v42 2024 OOT prediction"
            ),
        },
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "v42_public_lb": (
            V42_PUBLIC_LB
        ),
        "transfer_pairs": (
            TRANSFER_PAIRS
        ),
        "selected_candidate": (
            selection
        ),
        "validation_cfg": (
            VALIDATION_ET_CFG
        ),
        "final_cfg": (
            FINAL_ET_CFG
        ),
        "adapter_seeds": (
            ADAPTER_SEEDS
        ),
        "n_adapter_features": int(
            X24.shape[
                1
            ]
        ),
        "n_core_features": int(
            len(
                core24_cols
            )
        ),
        "n_exact_features": int(
            len(
                exact24_cols
            )
        ),
        "n_2024_source_rows": int(
            len(
                raw24
            )
        ),
        "n_2024_R_rows": int(
            domain_masks(
                raw24
            )[
                "R"
            ].sum()
        ),
        "residual24_mean": float(
            residual24.mean()
        ),
        "residual24_std": float(
            residual24.std()
        ),
    }

    (
        MODEL_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console
    # -----------------------------------------------------------------
    print()
    print(
        "=" * 128
    )
    print(
        "TEMPORAL TRANSFER — TOP CANDIDATES"
    )
    print(
        "=" * 128
    )

    print(
        summary.head(
            20
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "=" * 128
    )
    print(
        "SELECTED L2"
    )
    print(
        "=" * 128
    )

    print(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
        )
    )

    if selection[
        "selected"
    ]:
        route = (
            selection[
                "route"
            ]
        )

        shrink = float(
            selection[
                "shrink"
            ]
        )

        detail = (
            by_year.loc[
                by_year[
                    "route"
                ].eq(
                    route
                )
                & np.isclose(
                    by_year[
                        "shrink"
                    ],
                    shrink,
                )
            ]
            .sort_values(
                "year"
            )
        )

        print()
        print(
            "SELECTED BY YEAR"
        )

        print(
            detail.to_string(
                index=False,
                float_format=lambda x: f"{x:+.5f}",
            )
        )

    print()
    print(
        f"Final L2 artifact: {MODEL_PATH}"
    )

    print(
        f"Validation       : {VALIDATION_DIR}"
    )

    print(
        "=" * 128
    )


if __name__ == "__main__":
    main()

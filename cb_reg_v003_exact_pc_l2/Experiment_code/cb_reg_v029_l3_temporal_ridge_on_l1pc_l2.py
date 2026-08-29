#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v029_l3_temporal_ridge_on_l1pc_l2.py
============================================

목적
----
v028에서 확정 후보인 L1+L2 OOF 위에
V18-style temporal Ridge residual correction을 strict next-season 방식으로 검증한다.

Base system D
-------------
L1:
    CatBoost + Exact-ASOF + R-only PC FULL

L2:
    ExtraTrees residual adapter
    PC는 L2 입력에서 제외

L3 candidate E
--------------
source season의 D OOF residual:
    residual_source = y_source - D_OOF_source

를 Ridge로 학습하고 바로 다음 season에만 적용한다.

Strict transitions:
    2021 residual -> 2022 correction
    2022 residual -> 2023 correction
    2023 residual -> 2024 correction

중요
----
- CatBoost 재학습 0회
- ExtraTrees 재학습 0회
- v028 OOF prediction을 그대로 사용
- 새로 학습하는 것은 Ridge L3뿐
- same-season residual fitting 후 same-season 평가 금지
- F는 완전히 보호: L3 correction은 R에만 적용
- feature 통계도 target season 이전 공식 train label만 사용

V18-style frozen config
-----------------------
History window           : recent max 3 completed seasons
Decay                    : 0.65
Pitcher parent strength  : 100
Pitcher x batter_hand    : 35
Pitcher x pressure x hand: 45
Ridge alpha              : 1000
Primary gamma            : 0.75
Gamma sensitivity        : 0.65 / 0.75 / 0.85
Standardization          : source mean/std, z clip [-8, 8]

Stable conditional features
---------------------------
1) pitcher x batter_hand
    delta
    reliability
    weighted_delta

2) pitcher x pressure x batter_hand
    delta
    reliability
    weighted_delta

Count context
-------------
balls one-hot    : 4
strikes one-hot  : 3
ball x strike    : 12

총 25 features.

필수 입력
---------
./data/train.csv

./validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/
    oof_predictions.csv

출력
----
./validation/cb_reg_v029_l3_temporal_ridge_on_l1pc_l2/
    transition_results.csv
    seed_summary.csv
    final_summary.csv
    gamma_sensitivity.csv
    oof_predictions.csv
    metadata.json

실행
----
python cb_reg_v029_l3_temporal_ridge_on_l1pc_l2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v029_l3_temporal_ridge_on_l1pc_l2"

ROOT = Path(".")
TRAIN_PATH = ROOT / "data" / "train.csv"

V028_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
    / "oof_predictions.csv"
)

OUT_DIR = ROOT / "validation" / EXPERIMENT_NAME

TRANSITION_PATH = OUT_DIR / "transition_results.csv"
SEED_SUMMARY_PATH = OUT_DIR / "seed_summary.csv"
FINAL_SUMMARY_PATH = OUT_DIR / "final_summary.csv"
GAMMA_PATH = OUT_DIR / "gamma_sensitivity.csv"
OOF_PATH = OUT_DIR / "oof_predictions.csv"
METADATA_PATH = OUT_DIR / "metadata.json"

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"

REQUESTED_BASE_SEEDS = [7, 42, 2026]

HISTORY_YEARS = 3
DECAY = 0.65

PARENT_STRENGTH = 100.0
HAND_STRENGTH = 35.0
PRESSURE_HAND_STRENGTH = 45.0

RIDGE_ALPHA = 1000.0

PRIMARY_GAMMA = 0.75
GAMMAS = [0.65, 0.75, 0.85]

Z_CLIP = 8.0

TRANSITIONS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]


# =============================================================================
# Metrics
# =============================================================================

def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        0.0,
        1.0,
    )
    return float(
        np.mean((p - y) ** 2)
    )


def bss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        0.0,
        1.0,
    )

    score = brier(y, p)

    mu = float(y.mean())
    ref = max(
        mu * (1.0 - mu),
        1e-12,
    )

    return float(
        100000.0
        * (1.0 - score / ref)
    )


# =============================================================================
# Context
# =============================================================================

def add_pressure(frame):
    out = frame.copy()

    balls = pd.to_numeric(
        out["balls_before"],
        errors="coerce",
    )

    strikes = pd.to_numeric(
        out["strikes_before"],
        errors="coerce",
    )

    out["l3_pressure"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(
            balls == 3,
            "three_ball",
            np.where(
                strikes == 2,
                "two_strike",
                "normal",
            ),
        ),
    )

    return out


# =============================================================================
# Decayed historical conditional statistics
# =============================================================================

def select_history(
    train,
    target_year,
):
    """
    target year y의 features는 y 이전 season label만 사용한다.
    최근 최대 3개 completed seasons, 최신 season weight=1.
    """

    season = pd.to_numeric(
        train[SEASON],
        errors="raise",
    ).astype(int)

    prior_years = sorted(
        season.loc[
            season < int(target_year)
        ]
        .unique()
        .tolist()
    )

    use_years = (
        prior_years[
            -HISTORY_YEARS:
        ]
    )

    hist = (
        train.loc[
            season.isin(
                use_years
            )
        ]
        .copy()
    )

    if len(hist) == 0:
        raise RuntimeError(
            f"No history before target_year={target_year}"
        )

    latest = max(use_years)

    hist["_l3_weight"] = (
        DECAY
        ** (
            latest
            - pd.to_numeric(
                hist[SEASON],
                errors="raise",
            ).astype(int)
        )
    ).astype(np.float64)

    return hist


def weighted_group_stats(
    frame,
    keys,
):
    x = frame[
        keys
        + [
            TARGET,
            "_l3_weight",
        ]
    ].copy()

    x["_wn"] = x[
        "_l3_weight"
    ].astype(np.float64)

    x["_ws"] = (
        x["_l3_weight"]
        .astype(np.float64)
        * pd.to_numeric(
            x[TARGET],
            errors="raise",
        ).astype(np.float64)
    )

    return (
        x.groupby(
            keys,
            sort=False,
            dropna=False,
        )[
            [
                "_wn",
                "_ws",
            ]
        ]
        .sum()
        .reset_index()
    )


def global_weighted_rate(hist):
    w = (
        hist["_l3_weight"]
        .to_numpy(
            dtype=np.float64
        )
    )

    y = (
        pd.to_numeric(
            hist[TARGET],
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    den = float(
        w.sum()
    )

    if den <= 0:
        return float(
            np.mean(y)
        )

    return float(
        np.sum(
            w * y
        )
        / den
    )


def build_conditional_table(
    hist,
    child_keys,
    child_strength,
):
    """
    Parent = pitcher_id.

    parent_rate:
      (parent weighted success + parent_strength * global_rate)
      /
      (parent weighted n + parent_strength)

    child_rate:
      (child weighted success + child_strength * parent_rate)
      /
      (child weighted n + child_strength)

    delta:
      child_rate - parent_rate

    reliability:
      child weighted n / (child weighted n + child_strength)

    weighted_delta:
      delta * reliability
    """

    global_rate = (
        global_weighted_rate(
            hist
        )
    )

    parent = weighted_group_stats(
        hist,
        ["pitcher_id"],
    ).rename(
        columns={
            "_wn": "parent_n",
            "_ws": "parent_success",
        }
    )

    parent["parent_rate"] = (
        (
            parent["parent_success"]
            + PARENT_STRENGTH
            * global_rate
        )
        / (
            parent["parent_n"]
            + PARENT_STRENGTH
        )
    )

    child = weighted_group_stats(
        hist,
        child_keys,
    ).rename(
        columns={
            "_wn": "child_n",
            "_ws": "child_success",
        }
    )

    child = child.merge(
        parent[
            [
                "pitcher_id",
                "parent_rate",
            ]
        ],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    child["parent_rate"] = (
        child["parent_rate"]
        .fillna(
            global_rate
        )
    )

    child["child_rate"] = (
        (
            child["child_success"]
            + float(child_strength)
            * child["parent_rate"]
        )
        / (
            child["child_n"]
            + float(child_strength)
        )
    )

    child["delta"] = (
        child["child_rate"]
        - child["parent_rate"]
    )

    child["reliability"] = (
        child["child_n"]
        / (
            child["child_n"]
            + float(child_strength)
        )
    )

    child["weighted_delta"] = (
        child["delta"]
        * child["reliability"]
    )

    return (
        child,
        parent,
        global_rate,
    )


def lookup_conditional_features(
    rows,
    table,
    parent,
    global_rate,
    keys,
    prefix,
):
    """
    unseen child:
      delta=0
      reliability=0
      weighted_delta=0

    이것은 conditional evidence가 없다는 의미.
    """

    source = (
        rows[
            keys
        ]
        .reset_index(
            drop=True
        )
        .merge(
            table[
                keys
                + [
                    "delta",
                    "reliability",
                    "weighted_delta",
                ]
            ],
            on=keys,
            how="left",
            sort=False,
        )
    )

    result = pd.DataFrame(
        index=np.arange(
            len(rows)
        )
    )

    for field in [
        "delta",
        "reliability",
        "weighted_delta",
    ]:
        result[
            f"{prefix}_{field}"
        ] = (
            pd.to_numeric(
                source[field],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(np.float64)
        )

    return result


# =============================================================================
# L3 feature matrix
# =============================================================================

def build_l3_features(
    train,
    rows,
    target_year,
):
    """
    target_year의 rows를 위한 L3 feature.
    conditional statistics는 target_year 이전 공식 train labels만 사용.
    """

    rows = (
        add_pressure(
            rows.reset_index(
                drop=True
            )
        )
    )

    hist = (
        add_pressure(
            select_history(
                train,
                target_year,
            )
        )
    )

    # ---------------------------------------------------------
    # 1) pitcher x batter_hand
    # ---------------------------------------------------------

    hand_table, hand_parent, hand_global = (
        build_conditional_table(
            hist=hist,
            child_keys=[
                "pitcher_id",
                "batter_hand",
            ],
            child_strength=(
                HAND_STRENGTH
            ),
        )
    )

    hand = (
        lookup_conditional_features(
            rows=rows,
            table=hand_table,
            parent=hand_parent,
            global_rate=hand_global,
            keys=[
                "pitcher_id",
                "batter_hand",
            ],
            prefix="hand",
        )
    )

    # ---------------------------------------------------------
    # 2) pitcher x pressure x batter_hand
    # ---------------------------------------------------------

    ph_table, ph_parent, ph_global = (
        build_conditional_table(
            hist=hist,
            child_keys=[
                "pitcher_id",
                "l3_pressure",
                "batter_hand",
            ],
            child_strength=(
                PRESSURE_HAND_STRENGTH
            ),
        )
    )

    ph = (
        lookup_conditional_features(
            rows=rows,
            table=ph_table,
            parent=ph_parent,
            global_rate=ph_global,
            keys=[
                "pitcher_id",
                "l3_pressure",
                "batter_hand",
            ],
            prefix="pressure_hand",
        )
    )

    # ---------------------------------------------------------
    # 3) Count one-hot
    # ---------------------------------------------------------

    balls = (
        pd.to_numeric(
            rows["balls_before"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )

    strikes = (
        pd.to_numeric(
            rows["strikes_before"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )

    count = pd.DataFrame(
        index=np.arange(
            len(rows)
        )
    )

    for b in range(4):
        count[
            f"balls_{b}"
        ] = (
            balls == b
        ).astype(
            np.float64
        )

    for s in range(3):
        count[
            f"strikes_{s}"
        ] = (
            strikes == s
        ).astype(
            np.float64
        )

    for b in range(4):
        for s in range(3):
            count[
                f"count_{b}_{s}"
            ] = (
                (balls == b)
                & (strikes == s)
            ).astype(
                np.float64
            )

    X = pd.concat(
        [
            hand,
            ph,
            count,
        ],
        axis=1,
    )

    if X.shape[1] != 25:
        raise RuntimeError(
            f"L3 feature count={X.shape[1]} != 25"
        )

    values = X.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "L3 features contain NaN/Inf"
        )

    return X


# =============================================================================
# Standardization
# =============================================================================

def fit_standardizer(X):
    values = X.to_numpy(
        dtype=np.float64
    )

    mean = values.mean(
        axis=0
    )

    std = values.std(
        axis=0
    )

    std = np.where(
        std > 1e-12,
        std,
        1.0,
    )

    return (
        mean,
        std,
    )


def transform_standardized(
    X,
    mean,
    std,
):
    z = (
        X.to_numpy(
            dtype=np.float64
        )
        - mean
    ) / std

    z = np.clip(
        z,
        -Z_CLIP,
        Z_CLIP,
    )

    if not np.isfinite(
        z
    ).all():
        raise RuntimeError(
            "standardized L3 contains NaN/Inf"
        )

    return z


# =============================================================================
# Data loading
# =============================================================================

def load_inputs():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not V028_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{V028_OOF_PATH}\n\n"
            "먼저 cb_reg_v028_l1pc_l2_extratrees_no_pc.py를 실행하세요."
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

    required_oof = {
        ID,
        SEASON,
        GAME_TYPE,
        "base_seed",
        "y_true",
        "B_L1_PC",
        "D_L1PC_L2",
    }

    missing = sorted(
        required_oof
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
        s
        for s in REQUESTED_BASE_SEEDS
        if s in available
    ]

    if not seeds:
        raise RuntimeError(
            f"No requested seeds. available={available}"
        )

    oof = (
        oof.loc[
            oof[
                "base_seed"
            ].isin(seeds)
        ]
        .copy()
    )

    return (
        train,
        oof,
        seeds,
    )


def raw_rows_for_oof(
    train,
    part,
):
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

    return raw


# =============================================================================
# One transition
# =============================================================================

def run_transition(
    train,
    oof_seed,
    base_seed,
    source_year,
    target_year,
):
    source = (
        oof_seed.loc[
            oof_seed[SEASON]
            .eq(source_year)
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    target = (
        oof_seed.loc[
            oof_seed[SEASON]
            .eq(target_year)
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        len(source) == 0
        or len(target) == 0
    ):
        raise RuntimeError(
            f"Missing transition {source_year}->{target_year}"
        )

    raw_source = (
        raw_rows_for_oof(
            train,
            source,
        )
    )

    raw_target = (
        raw_rows_for_oof(
            train,
            target,
        )
    )

    y_source = source[
        "y_true"
    ].to_numpy(
        dtype=np.float64
    )

    p_source = source[
        "D_L1PC_L2"
    ].to_numpy(
        dtype=np.float64
    )

    residual_source = (
        y_source
        - p_source
    )

    y_target = target[
        "y_true"
    ].to_numpy(
        dtype=np.float64
    )

    p_target = target[
        "D_L1PC_L2"
    ].to_numpy(
        dtype=np.float64
    )

    # Features available in each season at prediction time.
    X_source = (
        build_l3_features(
            train=train,
            rows=raw_source,
            target_year=source_year,
        )
    )

    X_target = (
        build_l3_features(
            train=train,
            rows=raw_target,
            target_year=target_year,
        )
    )

    if list(
        X_source.columns
    ) != list(
        X_target.columns
    ):
        raise RuntimeError(
            "source/target L3 feature order mismatch"
        )

    mean, std = (
        fit_standardizer(
            X_source
        )
    )

    Z_source = (
        transform_standardized(
            X_source,
            mean,
            std,
        )
    )

    Z_target = (
        transform_standardized(
            X_target,
            mean,
            std,
        )
    )

    # Reference philosophy:
    # fit residual model on source season only.
    source_r = (
        raw_source[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .ne("F")
        .to_numpy()
    )

    target_r = (
        raw_target[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .ne("F")
        .to_numpy()
    )

    if (
        source_r.sum() < 100
        or target_r.sum() == 0
    ):
        raise RuntimeError(
            "Insufficient R rows"
        )

    ridge = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
    )

    ridge.fit(
        Z_source[
            source_r
        ],
        residual_source[
            source_r
        ],
    )

    raw_correction = np.zeros(
        len(target),
        dtype=np.float64,
    )

    raw_correction[
        target_r
    ] = (
        ridge.predict(
            Z_target[
                target_r
            ]
        )
    )

    gamma_rows = []
    gamma_predictions = {}

    base_brier = brier(
        y_target,
        p_target,
    )

    base_bss = bss(
        y_target,
        p_target,
    )

    for gamma in GAMMAS:
        pred = (
            p_target.copy()
        )

        pred[
            target_r
        ] = np.clip(
            p_target[
                target_r
            ]
            + float(gamma)
            * raw_correction[
                target_r
            ],
            0.0,
            1.0,
        )

        # F exact protection
        pred[
            ~target_r
        ] = (
            p_target[
                ~target_r
            ]
        )

        gamma_predictions[
            float(gamma)
        ] = pred

        gamma_rows.append({
            "base_seed": int(
                base_seed
            ),
            "source_year": int(
                source_year
            ),
            "target_year": int(
                target_year
            ),
            "gamma": float(
                gamma
            ),
            "base_brier": (
                base_brier
            ),
            "candidate_brier": (
                brier(
                    y_target,
                    pred,
                )
            ),
            "brier_gain": (
                base_brier
                - brier(
                    y_target,
                    pred,
                )
            ),
            "base_bss": (
                base_bss
            ),
            "candidate_bss": (
                bss(
                    y_target,
                    pred,
                )
            ),
            "gain_bss": (
                bss(
                    y_target,
                    pred,
                )
                - base_bss
            ),
        })

    primary_pred = (
        gamma_predictions[
            PRIMARY_GAMMA
        ]
    )

    # R/F diagnostics
    base_r_bss = bss(
        y_target[
            target_r
        ],
        p_target[
            target_r
        ],
    )

    primary_r_bss = bss(
        y_target[
            target_r
        ],
        primary_pred[
            target_r
        ],
    )

    result = {
        "base_seed": int(
            base_seed
        ),
        "source_year": int(
            source_year
        ),
        "target_year": int(
            target_year
        ),
        "n_source": int(
            len(source)
        ),
        "n_source_R": int(
            source_r.sum()
        ),
        "n_target": int(
            len(target)
        ),
        "n_target_R": int(
            target_r.sum()
        ),
        "base_brier": (
            base_brier
        ),
        "l3_brier": (
            brier(
                y_target,
                primary_pred,
            )
        ),
        "brier_gain": (
            base_brier
            - brier(
                y_target,
                primary_pred,
            )
        ),
        "base_bss": (
            base_bss
        ),
        "l3_bss": (
            bss(
                y_target,
                primary_pred,
            )
        ),
        "gain_bss": (
            bss(
                y_target,
                primary_pred,
            )
            - base_bss
        ),
        "R_gain_bss": (
            primary_r_bss
            - base_r_bss
        ),
        "mean_abs_raw_correction_R": float(
            np.mean(
                np.abs(
                    raw_correction[
                        target_r
                    ]
                )
            )
        ),
        "ridge_intercept": float(
            ridge.intercept_
        ),
        "ridge_coef_l2_norm": float(
            np.linalg.norm(
                ridge.coef_
            )
        ),
    }

    pred_out = pd.DataFrame({
        ID: (
            target[ID]
            .to_numpy()
        ),
        SEASON: int(
            target_year
        ),
        GAME_TYPE: (
            raw_target[
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
        "source_year": int(
            source_year
        ),
        "y_true": (
            y_target
        ),
        "D_L1PC_L2": (
            p_target
        ),
        "L3_RAW_CORRECTION": (
            raw_correction
        ),
        "E_L1PC_L2_L3": (
            primary_pred
        ),
    })

    return (
        result,
        pd.DataFrame(
            gamma_rows
        ),
        pred_out,
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
            .eq(seed)
        ]
        .copy()
    )

    y = part[
        "y_true"
    ].to_numpy(
        dtype=np.float64
    )

    base = part[
        "D_L1PC_L2"
    ].to_numpy(
        dtype=np.float64
    )

    cand = part[
        "E_L1PC_L2_L3"
    ].to_numpy(
        dtype=np.float64
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

    gains = {}

    wins = 0

    for year in [
        2022,
        2023,
        2024,
    ]:
        mask = (
            part[SEASON]
            .eq(year)
            .to_numpy()
        )

        gain = (
            bss(
                y[mask],
                cand[mask],
            )
            - bss(
                y[mask],
                base[mask],
            )
        )

        gains[
            year
        ] = float(
            gain
        )

        if gain > 0:
            wins += 1

    return {
        "base_seed": int(
            seed
        ),
        "pooled_gain_bss": float(
            bss(y, cand)
            - bss(y, base)
        ),
        "recent_2023_2024_gain_bss": float(
            bss(
                y[recent],
                cand[recent],
            )
            - bss(
                y[recent],
                base[recent],
            )
        ),
        "gain_2024_bss": float(
            bss(
                y[latest],
                cand[latest],
            )
            - bss(
                y[latest],
                base[latest],
            )
        ),
        "gain_2022_bss": (
            gains[2022]
        ),
        "gain_2023_bss": (
            gains[2023]
        ),
        "gain_2024_bss_check": (
            gains[2024]
        ),
        "transition_wins": int(
            wins
        ),
        "seed_win": bool(
            bss(y, cand)
            > bss(y, base)
        ),
    }


def build_final_summary(
    seed_summary,
    gamma_all,
):
    pooled = seed_summary[
        "pooled_gain_bss"
    ].to_numpy(
        dtype=np.float64
    )

    recent = seed_summary[
        "recent_2023_2024_gain_bss"
    ].to_numpy(
        dtype=np.float64
    )

    latest = seed_summary[
        "gain_2024_bss"
    ].to_numpy(
        dtype=np.float64
    )

    n = len(
        seed_summary
    )

    seed_wins = int(
        (pooled > 0)
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

    # gamma robustness:
    # each gamma's mean gain across all seed-transition combinations
    gamma_mean = (
        gamma_all
        .groupby(
            "gamma"
        )[
            "gain_bss"
        ]
        .mean()
    )

    gamma_all_positive = bool(
        (
            gamma_mean
            > 0
        )
        .all()
    )

    if (
        seed_wins == n
        and mean_pooled > 0
        and float(
            recent.mean()
        ) > 0
        and float(
            latest.mean()
        ) > 0
        and gamma_all_positive
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
        and mean_pooled > 0
        and float(
            recent.mean()
        ) > 0
    ):
        verdict = (
            "PASS"
        )

    elif mean_pooled > 0:
        verdict = (
            "WEAK"
        )

    else:
        verdict = (
            "FAIL"
        )

    return pd.DataFrame([
        {
            "n_base_seeds": int(
                n
            ),
            "seed_wins": int(
                seed_wins
            ),
            "mean_pooled_gain_bss": (
                mean_pooled
            ),
            "std_pooled_gain_bss": (
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
            "mean_recent_2023_2024_gain_bss": float(
                recent.mean()
            ),
            "mean_2024_gain_bss": float(
                latest.mean()
            ),
            "mean_transition_wins": float(
                seed_summary[
                    "transition_wins"
                ]
                .mean()
            ),
            "gamma_065_mean_gain_bss": float(
                gamma_mean.loc[
                    0.65
                ]
            ),
            "gamma_075_mean_gain_bss": float(
                gamma_mean.loc[
                    0.75
                ]
            ),
            "gamma_085_mean_gain_bss": float(
                gamma_mean.loc[
                    0.85
                ]
            ),
            "all_gamma_mean_positive": (
                gamma_all_positive
            ),
            "verdict": verdict,
        }
    ])


# =============================================================================
# Main
# =============================================================================

def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {OUT_DIR}\n"
            "기존 결과 보호를 위해 자동 overwrite하지 않습니다."
        )

    train, oof, seeds = (
        load_inputs()
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "=" * 116
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 116
    )
    print(
        "CatBoost fits   : 0"
    )
    print(
        "ExtraTrees fits : 0"
    )
    print(
        "New model       : Ridge L3 only"
    )
    print(
        f"Base seeds      : {seeds}"
    )
    print(
        f"Transitions     : {TRANSITIONS}"
    )
    print(
        f"History         : max {HISTORY_YEARS} seasons"
    )
    print(
        f"Decay           : {DECAY}"
    )
    print(
        f"Strengths       : parent={PARENT_STRENGTH}, "
        f"hand={HAND_STRENGTH}, "
        f"pressure_hand={PRESSURE_HAND_STRENGTH}"
    )
    print(
        f"Ridge alpha     : {RIDGE_ALPHA}"
    )
    print(
        f"Primary gamma   : {PRIMARY_GAMMA}"
    )
    print(
        f"Gamma check     : {GAMMAS}"
    )
    print(
        "Routing         : R=L3 corrected / F=D protected"
    )
    print(
        "=" * 116
    )

    transition_rows = []
    gamma_frames = []
    pred_frames = []

    for seed in seeds:
        print()
        print(
            "#" * 116
        )
        print(
            f"BASE SEED {seed}"
        )
        print(
            "#" * 116
        )

        oof_seed = (
            oof.loc[
                oof[
                    "base_seed"
                ]
                .eq(seed)
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        for source_year, target_year in TRANSITIONS:
            (
                result,
                gamma_df,
                pred_df,
            ) = run_transition(
                train=train,
                oof_seed=oof_seed,
                base_seed=seed,
                source_year=source_year,
                target_year=target_year,
            )

            transition_rows.append(
                result
            )
            gamma_frames.append(
                gamma_df
            )
            pred_frames.append(
                pred_df
            )

            print(
                f"[{source_year}->{target_year}] "
                f"gain={result['gain_bss']:+.3f} BSS | "
                f"R={result['R_gain_bss']:+.3f} | "
                f"|corr|={result['mean_abs_raw_correction_R']:.6f}"
            )

    transition_df = pd.DataFrame(
        transition_rows
    )

    gamma_all = pd.concat(
        gamma_frames,
        ignore_index=True,
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
            seed_summary,
            gamma_all,
        )
    )

    transition_df.to_csv(
        TRANSITION_PATH,
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

    gamma_all.to_csv(
        GAMMA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pred_all.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "base_oof": str(
            V028_OOF_PATH
        ),
        "catboost_retrained": False,
        "extratrees_retrained": False,
        "l3_model": "Ridge",
        "strict_next_season": True,
        "transitions": [
            {
                "source": int(s),
                "target": int(t),
            }
            for s, t
            in TRANSITIONS
        ],
        "config": {
            "history_years": (
                HISTORY_YEARS
            ),
            "decay": DECAY,
            "parent_strength": (
                PARENT_STRENGTH
            ),
            "hand_strength": (
                HAND_STRENGTH
            ),
            "pressure_hand_strength": (
                PRESSURE_HAND_STRENGTH
            ),
            "ridge_alpha": (
                RIDGE_ALPHA
            ),
            "primary_gamma": (
                PRIMARY_GAMMA
            ),
            "gamma_sensitivity": (
                GAMMAS
            ),
            "z_clip": Z_CLIP,
            "routing": (
                "R=L3 corrected; F=D protected"
            ),
        },
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
        "=" * 116
    )
    print(
        "L3 TEMPORAL RIDGE | SEED SUMMARY"
    )
    print(
        "=" * 116
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
        "=" * 116
    )
    print(
        "L3 TEMPORAL RIDGE | FINAL SUMMARY"
    )
    print(
        "=" * 116
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
        "- STRONG_PASS: L1+L2+L3 구조 최종 후보"
    )
    print(
        "- PASS       : transition별 세부 안정성 확인 후 승격"
    )
    print(
        "- WEAK/FAIL  : L3 제외, L1+L2 유지"
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"- {TRANSITION_PATH}"
    )
    print(
        f"- {SEED_SUMMARY_PATH}"
    )
    print(
        f"- {FINAL_SUMMARY_PATH}"
    )
    print(
        f"- {GAMMA_PATH}"
    )
    print(
        f"- {OOF_PATH}"
    )
    print(
        f"- {METADATA_PATH}"
    )
    print(
        "=" * 116
    )


if __name__ == "__main__":
    main()

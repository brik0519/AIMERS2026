#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v027_pc_placement_l1_vs_l2.py
====================================

목적
----
PITCHER_CONDITIONAL(PC)이 실제로 어느 위치에서 더 가치가 있는지 검증한다.

이미 완료된 CatBoost 학습 결과를 재사용한다.
CatBoost는 이 스크립트에서 단 한 번도 다시 학습하지 않는다.

비교 구조
---------
A) CONTROL
   = cb_reg_v003 + Exact-ASOF
   = 기존 v026 OOF의 CONTROL prediction

B) L1_PC
   = CatBoost L1에 R-only PC FULL 21개를 직접 넣은 기존 방식
   = 기존 v026 OOF의 FULL prediction

C0) L2_BASE
   = A의 prediction을 base로 사용
   + reference-style ExtraTrees residual adapter
   + adapter core features + Exact-ASOF5
   - PC는 넣지 않음

C1) L2_PC
   = C0와 동일한 ExtraTrees residual adapter
   + PITCHER_CONDITIONAL 21개 추가

핵심 비교
---------
1) PC의 L1 incremental value
       B - A

2) PC의 L2 incremental value
       C1 - C0

3) 최종 구조 비교
       C1 vs B

따라서 L2 자체의 이득과 PC의 위치 효과를 분리한다.

Reference V15에서 가져온 부분
----------------------------
ExtraTrees residual adapter:
    n_estimators=100
    max_depth=10
    min_samples_leaf=200
    max_features=0.7

Adapter shrink:
    0.50

Crossfit seeds:
    42, 2026, 314

Crossfit:
    pitcher_id를 seed별 2분할
    한쪽 pitcher residual을 학습하고 반대쪽 pitcher에 예측
    두 fold를 합쳐 OOF correction 생성
    3 adapter seed prediction consensus

우리 프로젝트에 맞춘 adaptation
-------------------------------
- L1은 reference LightGBM/Ridge ensemble이 아니라 현재 CatBoost Exact baseline.
- PC는 이전 실험 결과에 따라 R에서만 평가/적용.
- F는 CONTROL prediction을 그대로 보호.
- reference의 R_CORE/R_ANCHOR 세분화는 이번 placement 실험에서는 사용하지 않는다.
- reference EXACT_ASOF_STATE 대신 현재 프로젝트의 Exact-ASOF5를 L2에 제공한다.
- L2 feature preprocessing의 median/category map은 target year가 아니라
  바로 이전 labeled season에서만 만든다.
- PC도 정확히 이전 labeled season label만 이용한다.

입력
----
./data/train.csv

./validation/cb_reg_v026_pc_full_lean4_multiseed3/
    oof_predictions.csv

프로젝트 루트에:
    cb_reg_v021_pitcher_conditional_r_only_ablation.py

출력
----
./validation/cb_reg_v027_pc_placement_l1_vs_l2/
    fold_results.csv
    seed_summary.csv
    placement_summary.csv
    oof_predictions.csv
    metadata.json

실행
----
python cb_reg_v027_pc_placement_l1_vs_l2.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v027_pc_placement_l1_vs_l2"

ROOT = Path(".")
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"

V026_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v026_pc_full_lean4_multiseed3"
    / "oof_predictions.csv"
)

SOURCE_CANDIDATES = [
    ROOT / "cb_reg_v021_pitcher_conditional_r_only_ablation.py",
    ROOT / "cb_reg_v021_pitcher_conditional_r_only_ablation(1).py",
]

OUT_DIR = ROOT / "validation" / EXPERIMENT_NAME
FOLD_PATH = OUT_DIR / "fold_results.csv"
SEED_SUMMARY_PATH = OUT_DIR / "seed_summary.csv"
PLACEMENT_SUMMARY_PATH = OUT_DIR / "placement_summary.csv"
OOF_PATH = OUT_DIR / "oof_predictions.csv"
METADATA_PATH = OUT_DIR / "metadata.json"

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"

# 방금 v026에서 이미 검증한 CatBoost seeds.
# OOF 파일에 존재하는 seed와 교집합만 자동 사용.
REQUESTED_BASE_SEEDS = [7, 42, 2026]

# Reference V15 residual adapter 설정.
ADAPTER_SEEDS = [42, 2026, 314]
ADAPTER_CFG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}
ADAPTER_SHRINK = 0.50
N_JOBS = max(1, min(6, os.cpu_count() or 1))

# Reference V15 adapter core의 row-local numeric view.
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
    # Reference ADAPTER_RAWNUM에 추가됐던 team ids.
    "pitcher_team_id",
    "batter_team_id",
]

REFERENCE_CAT_SMALL = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    # reference에는 domain3도 있지만 현재 프로젝트 placement에서는 제외.
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


# =============================================================================
# Import exact PC / Exact-ASOF implementation from v021
# =============================================================================

def find_source_path():
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "cb_reg_v021_pitcher_conditional_r_only_ablation.py를 "
        "프로젝트 루트에서 찾지 못했습니다."
    )


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Metrics
# =============================================================================

def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def bss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    score = brier(y, p)
    mu = float(y.mean())
    ref = max(mu * (1.0 - mu), 1e-12)

    return float(
        100000.0
        * (1.0 - score / ref)
    )


# =============================================================================
# Reference-style adapter core
# =============================================================================

def numeric_series(frame, col, default=np.nan):
    if col not in frame.columns:
        return pd.Series(
            np.full(len(frame), default, dtype=np.float64),
            index=frame.index,
        )

    return pd.to_numeric(
        frame[col],
        errors="coerce",
    )


def add_eng(frame):
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

    p1 = p1.fillna(career)
    p3 = p3.fillna(p1)
    p5 = p5.fillna(p3)

    x["eng_recent35"] = (
        0.5 * p1
        + 0.3 * p3
        + 0.2 * p5
    )

    x["eng_recent_delta"] = (
        x["eng_recent35"]
        - career
    )

    x["eng_sb"] = (
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
        "pitcher_hand" in x.columns
        and "batter_hand" in x.columns
    ):
        x["eng_same"] = (
            x["pitcher_hand"]
            .astype(str)
            .eq(
                x["batter_hand"]
                .astype(str)
            )
            .astype(np.float32)
        )
    else:
        x["eng_same"] = 0.0

    x["eng_close"] = (
        numeric_series(
            x,
            "score_diff_pitcher_team",
            np.nan,
        )
        .abs()
        .le(1)
        .astype(np.float32)
    )

    x["eng_late"] = (
        numeric_series(
            x,
            "inning",
            np.nan,
        )
        .ge(7)
        .astype(np.float32)
    )

    return x


def fit_adapter_prep(map_rows):
    """
    Reference처럼 target-year 전체에서 prep을 만들지 않고,
    직전 labeled season의 row-local inputs만 이용한다.
    target은 사용하지 않는다.
    """
    m = add_eng(
        map_rows.reset_index(drop=True)
    )

    numeric_cols = [
        c
        for c in REFERENCE_RAWNUM
        if c in m.columns
    ] + ENG_COLS

    med = (
        m[numeric_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .median()
        .fillna(0.0)
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
            m[col]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
        )
        maps[col] = {
            value: idx
            for idx, value
            in enumerate(
                pd.Index(
                    s.unique()
                )
            )
        }

    return {
        "numeric_cols": numeric_cols,
        "med": med,
        "maps": maps,
    }


def transform_adapter_core(
    prep,
    rows,
    base_pred,
):
    r = add_eng(
        rows.reset_index(drop=True)
    )

    A = (
        r[
            prep["numeric_cols"]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .copy()
    )

    for col in prep["numeric_cols"]:
        A[col] = (
            A[col]
            .fillna(
                float(
                    prep["med"].get(
                        col,
                        0.0,
                    )
                )
            )
            .astype(np.float32)
        )

    for col, mapping in prep["maps"].items():
        A[
            "code_" + col
        ] = (
            r[col]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.float32)
        )

    A["base_pred"] = np.asarray(
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

    return A.reset_index(drop=True)


# =============================================================================
# Exact-ASOF5 for adapter
# =============================================================================

def build_exact5_for_target_year(
    base_module,
    train,
    target_rows,
    target_year,
):
    history = train.loc[
        pd.to_numeric(
            train[SEASON],
            errors="raise",
        ).astype(int)
        < int(target_year)
    ].copy()

    if len(history) == 0:
        raise RuntimeError(
            f"No prior history for Exact-ASOF {target_year}"
        )

    stats = (
        base_module
        .build_pitcher_season_stats(
            history
        )
    )

    transformed, _ = (
        base_module
        .add_pitcher_exact_asof_features(
            target_rows,
            stats,
        )
    )

    exact = (
        transformed[
            base_module.EXACT_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )

    return exact


# =============================================================================
# PITCHER_CONDITIONAL21
# =============================================================================

def build_pc21_for_target_year(
    base_module,
    train,
    target_rows,
    target_year,
):
    source_year = int(target_year) - 1

    source_history = train.loc[
        pd.to_numeric(
            train[SEASON],
            errors="raise",
        ).astype(int)
        == source_year
    ].copy()

    if len(source_history) == 0:
        raise RuntimeError(
            f"No previous labeled season {source_year} "
            f"for PC target {target_year}"
        )

    pc, _ = (
        base_module
        .build_pitcher_conditional_for_rows(
            rows=target_rows,
            available_history=source_history,
            scope=f"placement_{target_year}",
        )
    )

    pc = (
        pc[
            base_module.PC_FEATURES
        ]
        .astype(np.float32)
        .reset_index(drop=True)
    )

    if not np.isfinite(
        pc.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            f"PC contains NaN/Inf: {target_year}"
        )

    return pc


# =============================================================================
# Reference V15 pitcher crossfit
# =============================================================================

def pitcher_half(
    frame,
    seed,
):
    pitcher = (
        frame["pitcher_id"]
        .fillna("__MISSING__")
        .astype(str)
    )

    pitcher_ids = np.array(
        sorted(
            pitcher.unique()
        )
    )

    rng = np.random.default_rng(
        int(seed)
    )

    rng.shuffle(
        pitcher_ids
    )

    group_a = set(
        pitcher_ids[
            : len(pitcher_ids) // 2
        ]
    )

    return np.where(
        pitcher.isin(
            group_a
        ).to_numpy(),
        0,
        1,
    ).astype(np.int8)


def crossfit_l2_variant(
    X,
    y,
    base_pred,
    frame,
):
    """
    Reference V15 방식:
    - pitcher 2-fold crossfit
    - ExtraTrees residual target = y - base_pred
    - 3 adapter seed consensus
    - correction shrink=0.50

    우리 adaptation:
    - R만 correction
    - F는 exact CONTROL 보호
    """
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    base_pred = np.asarray(
        base_pred,
        dtype=np.float64,
    )

    residual = (
        y - base_pred
    )

    gt = (
        frame[GAME_TYPE]
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )

    r_mask = gt != "F"

    seed_predictions = []
    seed_rows = []

    for adapter_seed in ADAPTER_SEEDS:
        split = pitcher_half(
            frame,
            adapter_seed,
        )

        correction = np.zeros(
            len(frame),
            dtype=np.float64,
        )

        for target_fold in [0, 1]:
            src = (
                (split != target_fold)
                & r_mask
            )

            tgt = (
                (split == target_fold)
                & r_mask
            )

            n_train = int(
                src.sum()
            )
            n_target = int(
                tgt.sum()
            )

            if (
                n_train <= ADAPTER_CFG[
                    "min_samples_leaf"
                ] * 2
                or n_target == 0
            ):
                raise RuntimeError(
                    "Insufficient R rows for ExtraTrees crossfit: "
                    f"seed={adapter_seed}, fold={target_fold}, "
                    f"train={n_train}, target={n_target}"
                )

            model = ExtraTreesRegressor(
                n_jobs=N_JOBS,
                random_state=(
                    16200
                    + int(
                        adapter_seed
                    )
                ),
                **ADAPTER_CFG,
            )

            model.fit(
                X.loc[src],
                residual[src],
            )

            correction[tgt] = (
                model.predict(
                    X.loc[tgt]
                )
            )

        pred = base_pred.copy()

        pred[r_mask] = np.clip(
            base_pred[r_mask]
            + ADAPTER_SHRINK
            * correction[r_mask],
            0.0,
            1.0,
        )

        # F exact protection
        pred[~r_mask] = (
            base_pred[~r_mask]
        )

        seed_predictions.append(
            pred
        )

        seed_rows.append({
            "adapter_seed": int(
                adapter_seed
            ),
            "brier": brier(
                y,
                pred,
            ),
            "bss": bss(
                y,
                pred,
            ),
            "gain_bss_vs_control": (
                bss(y, pred)
                - bss(
                    y,
                    base_pred,
                )
            ),
        })

    consensus = np.mean(
        np.column_stack(
            seed_predictions
        ),
        axis=1,
    )

    # F는 평균 후에도 base와 정확히 같아야 함.
    if not np.allclose(
        consensus[~r_mask],
        base_pred[~r_mask],
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            "F prediction changed in R-only L2"
        )

    return (
        consensus,
        pd.DataFrame(
            seed_rows
        ),
    )


# =============================================================================
# Load existing OOF without retraining CatBoost
# =============================================================================

def load_existing_oof():
    if not V026_OOF_PATH.exists():
        raise FileNotFoundError(
            f"기존 v026 OOF가 없습니다:\n{V026_OOF_PATH}\n\n"
            "이 스크립트는 CatBoost 재학습을 하지 않습니다. "
            "먼저 이미 실행한 v026 결과 폴더가 프로젝트의 validation/ 아래에 있는지 확인하세요."
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
        s
        for s in REQUESTED_BASE_SEEDS
        if s in available
    ]

    if not seeds:
        raise RuntimeError(
            f"Requested base seeds {REQUESTED_BASE_SEEDS} "
            f"not present in v026 OOF. available={available}"
        )

    return (
        oof.loc[
            oof["seed"].isin(
                seeds
            )
        ].copy(),
        seeds,
    )


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_one_base_seed(
    base_module,
    train,
    oof_seed,
    base_seed,
):
    fold_rows = []
    output_frames = []

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

        # Merge raw train rows strictly by row_id.
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
                f"OOF rows missing in train: {target_year}"
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

        # Target consistency.
        y = (
            part["y_true"]
            .to_numpy(
                dtype=np.float64
            )
        )

        train_y = (
            part[
                [ID]
            ]
            .merge(
                train[
                    [
                        ID,
                        TARGET,
                    ]
                ],
                on=ID,
                how="left",
                validate="one_to_one",
                sort=False,
            )[TARGET]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.allclose(
            y,
            train_y,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "OOF y_true != official train target"
            )

        control = (
            part["CONTROL"]
            .to_numpy(
                dtype=np.float64
            )
        )

        l1_pc = (
            part["FULL"]
            .to_numpy(
                dtype=np.float64
            )
        )

        # -------------------------------------------------------------
        # Previous-season prep, exactly as a future deployment-style
        # feature transformation would be frozen.
        # -------------------------------------------------------------
        source_year = (
            int(
                target_year
            )
            - 1
        )

        map_rows = train.loc[
            pd.to_numeric(
                train[SEASON],
                errors="raise",
            ).astype(int)
            .eq(
                source_year
            )
        ].copy()

        if len(map_rows) == 0:
            raise RuntimeError(
                f"No source season {source_year}"
            )

        prep = fit_adapter_prep(
            map_rows
        )

        core = transform_adapter_core(
            prep=prep,
            rows=raw,
            base_pred=control,
        )

        exact5 = build_exact5_for_target_year(
            base_module=base_module,
            train=train,
            target_rows=raw,
            target_year=target_year,
        )

        pc21 = build_pc21_for_target_year(
            base_module=base_module,
            train=train,
            target_rows=raw,
            target_year=target_year,
        )

        # C0:
        # residual adapter architecture present, PC absent.
        X_l2_base = pd.concat(
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

        # C1:
        # same adapter + PC.
        X_l2_pc = pd.concat(
            [
                X_l2_base,
                pc21.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        if (
            X_l2_base
            .columns
            .duplicated()
            .any()
            or X_l2_pc
            .columns
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                "Duplicate L2 feature columns"
            )

        if not np.isfinite(
            X_l2_pc.to_numpy(
                dtype=np.float64
            )
        ).all():
            raise RuntimeError(
                "Non-finite L2 feature matrix"
            )

        started = time.time()

        (
            l2_base,
            l2_base_seed_diag,
        ) = crossfit_l2_variant(
            X=X_l2_base,
            y=y,
            base_pred=control,
            frame=raw,
        )

        (
            l2_pc,
            l2_pc_seed_diag,
        ) = crossfit_l2_variant(
            X=X_l2_pc,
            y=y,
            base_pred=control,
            frame=raw,
        )

        elapsed = (
            time.time()
            - started
        )

        # Metrics
        A_bss = bss(y, control)
        B_bss = bss(y, l1_pc)
        C0_bss = bss(y, l2_base)
        C1_bss = bss(y, l2_pc)

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

            "A_control_brier": brier(
                y,
                control,
            ),
            "A_control_bss": A_bss,

            "B_l1_pc_brier": brier(
                y,
                l1_pc,
            ),
            "B_l1_pc_bss": B_bss,

            "C0_l2_base_brier": brier(
                y,
                l2_base,
            ),
            "C0_l2_base_bss": C0_bss,

            "C1_l2_pc_brier": brier(
                y,
                l2_pc,
            ),
            "C1_l2_pc_bss": C1_bss,

            # PC placement increments
            "l1_pc_increment_bss": (
                B_bss - A_bss
            ),
            "l2_pc_increment_bss": (
                C1_bss - C0_bss
            ),
            "l2_minus_l1_pc_increment_bss": (
                (C1_bss - C0_bss)
                - (B_bss - A_bss)
            ),

            # Final architecture comparisons
            "C1_minus_B_bss": (
                C1_bss - B_bss
            ),
            "C0_minus_A_bss": (
                C0_bss - A_bss
            ),

            "n_l2_base_features": int(
                X_l2_base.shape[1]
            ),
            "n_l2_pc_features": int(
                X_l2_pc.shape[1]
            ),
            "elapsed_seconds": float(
                elapsed
            ),
        }

        fold_rows.append(
            row
        )

        out = pd.DataFrame({
            ID: part[ID].to_numpy(),
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
            "A_CONTROL": control,
            "B_L1_PC": l1_pc,
            "C0_L2_BASE": l2_base,
            "C1_L2_PC": l2_pc,
        })

        output_frames.append(
            out
        )

        print(
            f"[base_seed={base_seed} | {target_year}] "
            f"A={A_bss:+.2f} | "
            f"B={B_bss:+.2f} "
            f"(L1 PC {B_bss-A_bss:+.2f}) | "
            f"C0={C0_bss:+.2f} | "
            f"C1={C1_bss:+.2f} "
            f"(L2 PC {C1_bss-C0_bss:+.2f}) | "
            f"C1-B={C1_bss-B_bss:+.2f}"
        )

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.concat(
            output_frames,
            ignore_index=True,
        ),
    )


def summarize_seed(
    frame,
    base_seed,
):
    part = frame.loc[
        frame["base_seed"]
        .eq(
            int(
                base_seed
            )
        )
    ].copy()

    y = part["y_true"].to_numpy(
        dtype=np.float64
    )

    A = part["A_CONTROL"].to_numpy(
        dtype=np.float64
    )
    B = part["B_L1_PC"].to_numpy(
        dtype=np.float64
    )
    C0 = part["C0_L2_BASE"].to_numpy(
        dtype=np.float64
    )
    C1 = part["C1_L2_PC"].to_numpy(
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

    A_bss = bss(y, A)
    B_bss = bss(y, B)
    C0_bss = bss(y, C0)
    C1_bss = bss(y, C1)

    l1_inc = (
        B_bss - A_bss
    )

    l2_inc = (
        C1_bss - C0_bss
    )

    return {
        "base_seed": int(
            base_seed
        ),

        "A_control_brier": brier(
            y,
            A,
        ),
        "A_control_bss": A_bss,

        "B_l1_pc_brier": brier(
            y,
            B,
        ),
        "B_l1_pc_bss": B_bss,

        "C0_l2_base_brier": brier(
            y,
            C0,
        ),
        "C0_l2_base_bss": C0_bss,

        "C1_l2_pc_brier": brier(
            y,
            C1,
        ),
        "C1_l2_pc_bss": C1_bss,

        "l1_pc_increment_bss": l1_inc,
        "l2_pc_increment_bss": l2_inc,

        # Positive => PC information is more useful inside L2
        # than directly inside CatBoost L1.
        "placement_advantage_l2_minus_l1_bss": (
            l2_inc - l1_inc
        ),

        # Positive => final L2-PC architecture beats current L1-PC champion.
        "final_C1_minus_B_bss": (
            C1_bss - B_bss
        ),

        # Residual adapter itself, without PC.
        "l2_base_gain_vs_control_bss": (
            C0_bss - A_bss
        ),

        "recent_C1_minus_B_bss": (
            bss(
                y[recent],
                C1[recent],
            )
            - bss(
                y[recent],
                B[recent],
            )
        ),

        "latest_2024_C1_minus_B_bss": (
            bss(
                y[latest],
                C1[latest],
            )
            - bss(
                y[latest],
                B[latest],
            )
        ),

        "recent_l1_pc_increment_bss": (
            bss(
                y[recent],
                B[recent],
            )
            - bss(
                y[recent],
                A[recent],
            )
        ),

        "recent_l2_pc_increment_bss": (
            bss(
                y[recent],
                C1[recent],
            )
            - bss(
                y[recent],
                C0[recent],
            )
        ),
    }


def build_placement_summary(
    seed_summary,
):
    l1 = (
        seed_summary[
            "l1_pc_increment_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    l2 = (
        seed_summary[
            "l2_pc_increment_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    placement = (
        seed_summary[
            "placement_advantage_l2_minus_l1_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    final_adv = (
        seed_summary[
            "final_C1_minus_B_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    recent_final = (
        seed_summary[
            "recent_C1_minus_B_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    latest_final = (
        seed_summary[
            "latest_2024_C1_minus_B_bss"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    l2_pc_positive = int(
        (l2 > 0).sum()
    )

    l2_location_wins = int(
        (placement > 0).sum()
    )

    final_l2_wins = int(
        (final_adv > 0).sum()
    )

    n = len(
        seed_summary
    )

    # Two different decisions:
    # 1) PC placement itself
    if (
        l2_location_wins == n
        and float(
            placement.mean()
        ) > 0
    ):
        placement_verdict = (
            "PC_PREFERS_L2"
        )
    elif (
        l2_location_wins
        >= max(
            2,
            n - 1,
        )
        and float(
            placement.mean()
        ) > 0
    ):
        placement_verdict = (
            "PC_LEANS_L2"
        )
    elif (
        l2_location_wins == 0
        and float(
            placement.mean()
        ) < 0
    ):
        placement_verdict = (
            "PC_PREFERS_L1"
        )
    else:
        placement_verdict = (
            "PLACEMENT_MIXED"
        )

    # 2) Which final system is currently stronger
    if (
        final_l2_wins == n
        and float(
            final_adv.mean()
        ) > 0
        and float(
            recent_final.mean()
        ) > 0
        and float(
            latest_final.mean()
        ) > 0
    ):
        final_verdict = (
            "L2_PC_FINAL_STRONGER"
        )
    elif (
        final_l2_wins
        >= max(
            2,
            n - 1,
        )
        and float(
            final_adv.mean()
        ) > 0
    ):
        final_verdict = (
            "L2_PC_FINAL_LEADS"
        )
    elif (
        final_l2_wins == 0
        and float(
            final_adv.mean()
        ) < 0
    ):
        final_verdict = (
            "KEEP_L1_PC"
        )
    else:
        final_verdict = (
            "FINAL_MIXED"
        )

    return pd.DataFrame(
        [
            {
                "n_base_seeds": int(
                    n
                ),

                "mean_l1_pc_increment_bss": float(
                    l1.mean()
                ),
                "std_l1_pc_increment_bss": float(
                    l1.std(
                        ddof=1
                    )
                    if n > 1
                    else 0.0
                ),

                "mean_l2_pc_increment_bss": float(
                    l2.mean()
                ),
                "std_l2_pc_increment_bss": float(
                    l2.std(
                        ddof=1
                    )
                    if n > 1
                    else 0.0
                ),

                "l2_pc_positive_seeds": int(
                    l2_pc_positive
                ),

                "mean_placement_advantage_l2_minus_l1_bss": float(
                    placement.mean()
                ),
                "l2_location_wins": int(
                    l2_location_wins
                ),

                "mean_final_C1_minus_B_bss": float(
                    final_adv.mean()
                ),
                "final_l2_wins": int(
                    final_l2_wins
                ),

                "mean_recent_C1_minus_B_bss": float(
                    recent_final.mean()
                ),

                "mean_2024_C1_minus_B_bss": float(
                    latest_final.mean()
                ),

                "placement_verdict": (
                    placement_verdict
                ),
                "final_verdict": (
                    final_verdict
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

    source_path = (
        find_source_path()
    )

    base_module = (
        import_module_from_path(
            "v021_pc_source",
            source_path,
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

    existing_oof, base_seeds = (
        load_existing_oof()
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
        "CATBOOST FITS : 0 "
        "(reuses v026 OOF)"
    )
    print(
        f"Base seeds    : {base_seeds}"
    )
    print(
        f"Adapter seeds : {ADAPTER_SEEDS}"
    )
    print(
        f"ExtraTrees    : {ADAPTER_CFG}"
    )
    print(
        f"Shrink        : {ADAPTER_SHRINK}"
    )
    print(
        "Routing       : R=adapter, F=CONTROL"
    )
    print(
        "Comparison    : "
        "A CONTROL | B L1-PC | C0 L2(no PC) | C1 L2(+PC)"
    )
    print(
        "=" * 116
    )

    fold_frames = []
    prediction_frames = []

    for base_seed in base_seeds:
        print()
        print(
            "#" * 116
        )
        print(
            f"BASE CATBOOST SEED {base_seed}"
        )
        print(
            "#" * 116
        )

        oof_seed = (
            existing_oof.loc[
                existing_oof[
                    "seed"
                ].eq(
                    int(
                        base_seed
                    )
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        fold_df, pred_df = (
            evaluate_one_base_seed(
                base_module=base_module,
                train=train,
                oof_seed=oof_seed,
                base_seed=base_seed,
            )
        )

        fold_frames.append(
            fold_df
        )
        prediction_frames.append(
            pred_df
        )

    fold_all = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    pred_all = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    seed_summary = pd.DataFrame(
        [
            summarize_seed(
                pred_all,
                seed,
            )
            for seed in base_seeds
        ]
    )

    placement_summary = (
        build_placement_summary(
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

    placement_summary.to_csv(
        PLACEMENT_SUMMARY_PATH,
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
        "existing_oof": str(
            V026_OOF_PATH
        ),
        "pc_source_script": str(
            source_path
        ),
        "base_seeds": [
            int(x)
            for x in base_seeds
        ],
        "adapter_seeds": (
            ADAPTER_SEEDS
        ),
        "adapter_cfg": (
            ADAPTER_CFG
        ),
        "adapter_shrink": float(
            ADAPTER_SHRINK
        ),
        "reference_alignment": {
            "extra_trees_cfg": (
                "V15_CF_CFG"
            ),
            "pitcher_crossfit": True,
            "adapter_seed_consensus": True,
            "residual_target": (
                "y - existing CatBoost CONTROL OOF"
            ),
            "differences": [
                (
                    "Current CatBoost Exact baseline "
                    "replaces reference base ensemble"
                ),
                (
                    "R-only routing protects F"
                ),
                (
                    "Current Exact-ASOF5 replaces "
                    "reference EXACT_ASOF_STATE block"
                ),
            ],
        },
        "comparisons": {
            "A": (
                "existing CONTROL"
            ),
            "B": (
                "existing FULL R-only PC in L1"
            ),
            "C0": (
                "ExtraTrees residual adapter "
                "without PC"
            ),
            "C1": (
                "same ExtraTrees residual adapter "
                "with PC21"
            ),
            "placement_metric": (
                "(C1-C0) BSS - (B-A) BSS"
            ),
        },
        "seed_summary": (
            seed_summary
            .to_dict(
                orient="records"
            )
        ),
        "placement_summary": (
            placement_summary
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
        "PC PLACEMENT | BASE-SEED SUMMARY"
    )
    print(
        "=" * 116
    )

    display_cols = [
        "base_seed",
        "l1_pc_increment_bss",
        "l2_pc_increment_bss",
        "placement_advantage_l2_minus_l1_bss",
        "l2_base_gain_vs_control_bss",
        "final_C1_minus_B_bss",
        "recent_C1_minus_B_bss",
        "latest_2024_C1_minus_B_bss",
    ]

    print(
        seed_summary[
            display_cols
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
        "=" * 116
    )
    print(
        "PC PLACEMENT | FINAL SUMMARY"
    )
    print(
        "=" * 116
    )

    print(
        placement_summary
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
        "Interpretation:"
    )
    print(
        "- L1 PC incremental = B - A"
    )
    print(
        "- L2 PC incremental = C1 - C0"
    )
    print(
        "- placement advantage > 0 "
        "=> PC 자체는 L2에 있을 때 더 큰 incremental value"
    )
    print(
        "- final C1-B > 0 "
        "=> 최종 시스템도 L2-PC가 현재 L1-PC보다 우수"
    )
    print(
        "- C0-A는 PC와 무관한 ExtraTrees residual adapter 자체 효과"
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
        f"- {PLACEMENT_SUMMARY_PATH}"
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

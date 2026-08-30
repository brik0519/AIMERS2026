#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v038_current_season_asof_state_ablation.py
==================================================

목적
----
현재 L1 champion:
    CatBoost + Exact-ASOF5 + HBEB + R-only PC FULL

에 다른 팀 GitHub들에서 반복적으로 유효했던
"통산 ASOF -> 현재 시즌 상태(current-season state)" 정보축을 추가한다.

Control은 재학습하지 않는다.
기존 v028 seed42 B_L1_PC OOF를 그대로 재사용한다.

Treatment만 CatBoost 재학습:
A) STATE_SUCCESS_EXT
   - current-season pitcher success의 smoothed/reliability 표현
   - current-season batter success
   - pitcher-vs-batter current state

B) STATE_COMPONENTS
   - current-season pitcher middle / ball / reverse / strike
   - current-season pitch mix
   - current-season batter middle

C) STATE_FULL
   - A + B

중요
----
현재 Exact5가 이미 제공하는
    exact_pitcher_season_n
    exact_pitcher_season_logn
    exact_pitcher_season_success_rate
    exact_pitcher_past_success_rate
    exact_pitcher_season_minus_past

는 그대로 유지한다.

이번 실험은 그 Exact5를 "교체"하지 않고,
아직 없는 current-state channel만 추가하는 one-factor Base A/B다.

정보 규칙
---------
각 row의 feature는:
- 그 row 자신의 공식 asof_* 값
- target season보다 이전의 labeled train history로 만든 상수

만 사용한다.

다른 validation/test row를 사용하지 않는다.

Component-rate boundary
-----------------------
middle/ball/reverse/strike/pitchmix/batter-middle은
이전 시즌 마지막 pitch의 subtype label을 직접 알 수 없으므로
reference 방식처럼 마지막 historical ASOF boundary를 사용한다.

즉:
    current window = cumulative count(row) - last historical pre-pitch count

이 window에는 boundary pitch 1개가 포함될 수 있다.
이는 target/test 다른 row를 보지 않고 component state를 구성하기 위한
reference-style bridge다.

검증
----
Expanding Window:
2019-20 -> 2021
2019-21 -> 2022
2019-22 -> 2023
2019-23 -> 2024

Control:
v028 seed42 B_L1_PC OOF reuse

Treatment routes:
GLOBAL
R_ONLY  : R=treatment / F=control
F_ONLY  : F=treatment / R=control

출력
----
validation/cb_reg_v038_current_season_asof_state_ablation/
    fold_results.csv
    oof_predictions.csv
    summary.csv
    state_feature_manifest.csv
    metadata.json
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v038_current_season_asof_state_ablation"

ROOT = Path(".")

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

CONTROL_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
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

# -------------------------------------------------------------------------
# ASOF rate specification
# -------------------------------------------------------------------------

SUCCESS_SPEC = {
    "pitcher": {
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "strengths": [
            40.0,
            80.0,
            160.0,
        ],
        "reliability_strength": 80.0,
    },
    "batter": {
        "id_col": "batter_id",
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "strengths": [
            40.0,
            80.0,
            160.0,
        ],
        "reliability_strength": 100.0,
    },
}

COMPONENT_SPECS = [
    # pitcher count-denominator state
    {
        "name": "pitcher_middle",
        "entity": "pitcher",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_middle_rate",
        "strength": 60.0,
    },
    {
        "name": "pitcher_ball",
        "entity": "pitcher",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_ball_rate",
        "strength": 60.0,
    },
    {
        "name": "pitcher_reverse",
        "entity": "pitcher",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_reverse_rate",
        "strength": 60.0,
    },
    {
        "name": "pitcher_strike",
        "entity": "pitcher",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_strike_rate",
        "strength": 60.0,
    },

    # pitch-mix state
    {
        "name": "pitcher_fastball",
        "entity": "mix",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_pitchmix_n",
        "rate_col": "asof_pitcher_fastball_rate",
        "strength": 60.0,
    },
    {
        "name": "pitcher_breaking",
        "entity": "mix",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_pitchmix_n",
        "rate_col": "asof_pitcher_breaking_rate",
        "strength": 60.0,
    },
    {
        "name": "pitcher_offspeed",
        "entity": "mix",
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_pitchmix_n",
        "rate_col": "asof_pitcher_offspeed_rate",
        "strength": 60.0,
    },

    # batter state
    {
        "name": "batter_middle",
        "entity": "batter",
        "id_col": "batter_id",
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_middle_rate",
        "strength": 80.0,
    },
]


SUCCESS_FEATURES = [
    "gseason__pitcher_rate_k40",
    "gseason__pitcher_rate_k80",
    "gseason__pitcher_rate_k160",
    "gseason__pitcher_reliability",
    "gseason__pitcher_minus_career",

    "gseason__batter_rate_k40",
    "gseason__batter_rate_k80",
    "gseason__batter_rate_k160",
    "gseason__batter_raw",
    "gseason__batter_log_n",
    "gseason__batter_reliability",
    "gseason__batter_minus_career",

    "gseason__pitcher_minus_batter_k80",
]


COMPONENT_FEATURES = [
    "gseason__pitcher_component_log_n",
    "gseason__pitcher_component_reliability",

    "gseason__pitcher_strike_rate",
    "gseason__pitcher_ball_rate",
    "gseason__pitcher_strike_ball_margin",
    "gseason__pitcher_middle_rate",
    "gseason__pitcher_reverse_rate",

    "gseason__pitcher_fastball_rate",
    "gseason__pitcher_breaking_rate",
    "gseason__pitcher_offspeed_rate",
    "gseason__pitcher_pitchmix_entropy",

    "gseason__batter_middle_rate",
    "gseason__batter_middle_minus_pitcher_middle",
]


TREATMENTS = {
    "STATE_SUCCESS_EXT": (
        SUCCESS_FEATURES
    ),
    "STATE_COMPONENTS": (
        COMPONENT_FEATURES
    ),
    "STATE_FULL": (
        SUCCESS_FEATURES
        + COMPONENT_FEATURES
    ),
}


# =============================================================================
# Helpers
# =============================================================================

def find_v021_source():
    preferred = [
        ROOT
        / "cb_reg_v021_pitcher_conditional_r_only_ablation.py",

        ROOT
        / "Exact_Pc_Feature_EX"
        / "cb_reg_v021_pitcher_conditional_r_only_ablation.py",
    ]

    for p in preferred:
        if p.exists():
            return p

    matches = sorted(
        p
        for p in ROOT.rglob(
            "cb_reg_v021_pitcher_conditional_r_only_ablation*.py"
        )
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and "validation" not in p.parts
    )

    if len(matches) == 1:
        print(
            f"[AUTO] v021 source found: {matches[0]}"
        )
        return matches[0]

    exact = [
        p
        for p in matches
        if p.name
        == "cb_reg_v021_pitcher_conditional_r_only_ablation.py"
    ]

    if len(exact) == 1:
        print(
            f"[AUTO] v021 source found: {exact[0]}"
        )
        return exact[0]

    if not matches:
        raise FileNotFoundError(
            "cb_reg_v021_pitcher_conditional_r_only_ablation*.py "
            "not found in project."
        )

    raise RuntimeError(
        "Multiple v021 source candidates found:\n"
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


def rounded_cumulative_count(
    n,
    rate,
):
    n = np.asarray(
        n,
        dtype=np.float64,
    )

    rate = np.asarray(
        rate,
        dtype=np.float64,
    )

    out = np.zeros(
        len(n),
        dtype=np.float64,
    )

    ok = (
        (n > 0)
        & np.isfinite(
            rate
        )
    )

    out[ok] = np.rint(
        n[ok]
        * rate[ok]
    )

    return out


# =============================================================================
# Control OOF
# =============================================================================

def load_control_oof():
    if not CONTROL_OOF_PATH.exists():
        raise FileNotFoundError(
            CONTROL_OOF_PATH
        )

    oof = pd.read_csv(
        CONTROL_OOF_PATH,
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
            oof.columns
        )
    )

    if missing:
        raise KeyError(
            f"Control OOF missing: {missing}"
        )

    oof = (
        oof.loc[
            pd.to_numeric(
                oof[
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

    oof[SEASON] = pd.to_numeric(
        oof[SEASON],
        errors="raise",
    ).astype(int)

    oof["y_true"] = pd.to_numeric(
        oof["y_true"],
        errors="raise",
    ).astype(np.float64)

    oof["control_pred"] = pd.to_numeric(
        oof["B_L1_PC"],
        errors="raise",
    ).astype(np.float64)

    oof = oof[
        [
            ID,
            SEASON,
            GAME_TYPE,
            "y_true",
            "control_pred",
        ]
    ].copy()

    if oof[ID].duplicated().any():
        raise RuntimeError(
            "Control seed42 OOF row_id duplicate"
        )

    return oof


# =============================================================================
# Current-season bank
# =============================================================================

def latest_season_prior(
    history,
    rate_col,
):
    if len(
        history
    ) == 0:
        return 0.5

    max_year = int(
        pd.to_numeric(
            history[SEASON],
            errors="raise",
        )
        .max()
    )

    s = pd.to_numeric(
        history.loc[
            pd.to_numeric(
                history[SEASON],
                errors="coerce",
            )
            .eq(
                max_year
            ),
            rate_col,
        ],
        errors="coerce",
    )

    if s.notna().any():
        return float(
            s.mean()
        )

    s2 = pd.to_numeric(
        history[
            rate_col
        ],
        errors="coerce",
    )

    if s2.notna().any():
        return float(
            s2.mean()
        )

    return 0.5


def build_success_bank(
    history,
    entity,
):
    spec = SUCCESS_SPEC[
        entity
    ]

    id_col = spec[
        "id_col"
    ]

    if len(
        history
    ) == 0:
        return {
            "n": {},
            "success": {},
            "prior": 0.5,
        }

    agg = (
        history.groupby(
            id_col,
            sort=False,
            dropna=False,
        )[TARGET]
        .agg(
            [
                "size",
                "sum",
            ]
        )
    )

    max_year = int(
        history[SEASON]
        .max()
    )

    prior = float(
        history.loc[
            history[SEASON].eq(
                max_year
            ),
            TARGET,
        ]
        .mean()
    )

    return {
        "n": (
            agg[
                "size"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),

        "success": (
            agg[
                "sum"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),

        "prior": (
            prior
            if np.isfinite(
                prior
            )
            else 0.5
        ),
    }


def build_component_bank(
    history,
):
    """
    각 rate별로 해당 denominator n이 최대인 historical row를
    pre-target boundary로 사용한다.
    """

    bank = {}

    for spec in COMPONENT_SPECS:
        name = spec[
            "name"
        ]

        id_col = spec[
            "id_col"
        ]

        n_col = spec[
            "n_col"
        ]

        rate_col = spec[
            "rate_col"
        ]

        prior = latest_season_prior(
            history,
            rate_col,
        )

        if len(
            history
        ) == 0:
            bank[
                name
            ] = {
                "base_n": {},
                "base_events": {},
                "prior": float(
                    prior
                ),
                "strength": float(
                    spec[
                        "strength"
                    ]
                ),
                "id_col": id_col,
                "n_col": n_col,
                "rate_col": rate_col,
            }

            continue

        tmp = history[
            [
                id_col,
                n_col,
                rate_col,
            ]
        ].copy()

        tmp[
            n_col
        ] = pd.to_numeric(
            tmp[
                n_col
            ],
            errors="coerce",
        ).fillna(
            0.0
        )

        tmp[
            rate_col
        ] = pd.to_numeric(
            tmp[
                rate_col
            ],
            errors="coerce",
        )

        # row with largest official cumulative n.
        idx = (
            tmp.groupby(
                id_col,
                sort=False,
                dropna=False,
            )[
                n_col
            ]
            .idxmax()
        )

        last = tmp.loc[
            idx
        ].copy()

        base_n = last[
            n_col
        ].to_numpy(
            np.float64
        )

        base_rate = last[
            rate_col
        ].to_numpy(
            np.float64
        )

        base_events = rounded_cumulative_count(
            base_n,
            base_rate,
        )

        ids = last[
            id_col
        ].tolist()

        bank[
            name
        ] = {
            "base_n": {
                k: float(
                    v
                )
                for k, v in zip(
                    ids,
                    base_n,
                )
            },

            "base_events": {
                k: float(
                    v
                )
                for k, v in zip(
                    ids,
                    base_events,
                )
            },

            "prior": float(
                prior
            ),

            "strength": float(
                spec[
                    "strength"
                ]
            ),

            "id_col": (
                id_col
            ),

            "n_col": (
                n_col
            ),

            "rate_col": (
                rate_col
            ),
        }

    return bank


def build_state_bank(
    available_history,
    target_year,
):
    hist = (
        available_history.loc[
            pd.to_numeric(
                available_history[
                    SEASON
                ],
                errors="raise",
            )
            .astype(int)
            .lt(
                int(
                    target_year
                )
            )
        ]
        .copy()
    )

    return {
        "target_year": int(
            target_year
        ),

        "pitcher_success": build_success_bank(
            hist,
            "pitcher",
        ),

        "batter_success": build_success_bank(
            hist,
            "batter",
        ),

        "components": build_component_bank(
            hist
        ),
    }


# =============================================================================
# Transform current-season state
# =============================================================================

def transform_success_state(
    rows,
    bank,
    entity,
    out,
):
    spec = SUCCESS_SPEC[
        entity
    ]

    sb = bank[
        f"{entity}_success"
    ]

    id_col = spec[
        "id_col"
    ]

    n_col = spec[
        "n_col"
    ]

    rate_col = spec[
        "rate_col"
    ]

    ids = rows[
        id_col
    ]

    hist_n = (
        ids.map(
            sb[
                "n"
            ]
        )
        .fillna(
            0.0
        )
        .to_numpy(
            np.float64
        )
    )

    hist_success = (
        ids.map(
            sb[
                "success"
            ]
        )
        .fillna(
            0.0
        )
        .to_numpy(
            np.float64
        )
    )

    asof_n = (
        pd.to_numeric(
            rows[
                n_col
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
        .to_numpy(
            np.float64
        )
    )

    career = pd.to_numeric(
        rows[
            rate_col
        ],
        errors="coerce",
    ).to_numpy(
        np.float64
    )

    prior = float(
        sb[
            "prior"
        ]
    )

    career = np.where(
        np.isfinite(
            career
        ),
        career,
        prior,
    )

    cumulative_success = rounded_cumulative_count(
        asof_n,
        career,
    )

    current_n = np.maximum(
        asof_n
        - hist_n,
        0.0,
    )

    current_success = np.clip(
        cumulative_success
        - hist_success,
        0.0,
        current_n,
    )

    raw = np.divide(
        current_success,
        current_n,
        out=np.full(
            len(
                rows
            ),
            prior,
            dtype=np.float64,
        ),
        where=(
            current_n
            > 0
        ),
    )

    prefix = (
        f"gseason__{entity}"
    )

    for k in spec[
        "strengths"
    ]:
        out[
            f"{prefix}_rate_k{int(k)}"
        ] = (
            (
                current_success
                + float(
                    k
                )
                * prior
            )
            / (
                current_n
                + float(
                    k
                )
            )
        ).astype(
            np.float32
        )

    out[
        f"{prefix}_raw"
    ] = raw.astype(
        np.float32
    )

    out[
        f"{prefix}_log_n"
    ] = np.log1p(
        current_n
    ).astype(
        np.float32
    )

    rk = float(
        spec[
            "reliability_strength"
        ]
    )

    out[
        f"{prefix}_reliability"
    ] = (
        current_n
        / (
            current_n
            + rk
        )
    ).astype(
        np.float32
    )

    out[
        f"{prefix}_minus_career"
    ] = (
        raw
        - career
    ).astype(
        np.float32
    )

    return {
        "current_n": (
            current_n
        ),
        "raw": (
            raw
        ),
        "k80": (
            out[
                f"{prefix}_rate_k80"
            ]
            .to_numpy(
                np.float64
            )
        ),
    }


def transform_component_state(
    rows,
    bank,
    out,
):
    comp_values = {}
    kind_windows = {}

    for spec in COMPONENT_SPECS:
        name = spec[
            "name"
        ]

        cb = bank[
            "components"
        ][
            name
        ]

        id_col = cb[
            "id_col"
        ]

        n_col = cb[
            "n_col"
        ]

        rate_col = cb[
            "rate_col"
        ]

        ids = rows[
            id_col
        ]

        base_n = (
            ids.map(
                cb[
                    "base_n"
                ]
            )
            .fillna(
                0.0
            )
            .to_numpy(
                np.float64
            )
        )

        base_events = (
            ids.map(
                cb[
                    "base_events"
                ]
            )
            .fillna(
                0.0
            )
            .to_numpy(
                np.float64
            )
        )

        n_now = (
            pd.to_numeric(
                rows[
                    n_col
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
            .to_numpy(
                np.float64
            )
        )

        rate_now = pd.to_numeric(
            rows[
                rate_col
            ],
            errors="coerce",
        ).to_numpy(
            np.float64
        )

        prior = float(
            cb[
                "prior"
            ]
        )

        rate_now = np.where(
            np.isfinite(
                rate_now
            ),
            rate_now,
            prior,
        )

        cumulative = rounded_cumulative_count(
            n_now,
            rate_now,
        )

        window_n = np.maximum(
            n_now
            - base_n,
            0.0,
        )

        window_events = np.clip(
            cumulative
            - base_events,
            0.0,
            window_n,
        )

        strength = float(
            cb[
                "strength"
            ]
        )

        smoothed = (
            window_events
            + strength
            * prior
        ) / (
            window_n
            + strength
        )

        comp_values[
            name
        ] = smoothed.astype(
            np.float64
        )

        kind_windows[
            name
        ] = (
            window_n
        )

    # Pitcher component sample/reliability.
    pitch_n = kind_windows[
        "pitcher_middle"
    ]

    out[
        "gseason__pitcher_component_log_n"
    ] = np.log1p(
        pitch_n
    ).astype(
        np.float32
    )

    out[
        "gseason__pitcher_component_reliability"
    ] = (
        pitch_n
        / (
            pitch_n
            + 60.0
        )
    ).astype(
        np.float32
    )

    strike = comp_values[
        "pitcher_strike"
    ]

    ball = comp_values[
        "pitcher_ball"
    ]

    middle = comp_values[
        "pitcher_middle"
    ]

    reverse = comp_values[
        "pitcher_reverse"
    ]

    out[
        "gseason__pitcher_strike_rate"
    ] = strike.astype(
        np.float32
    )

    out[
        "gseason__pitcher_ball_rate"
    ] = ball.astype(
        np.float32
    )

    out[
        "gseason__pitcher_strike_ball_margin"
    ] = (
        strike
        - ball
    ).astype(
        np.float32
    )

    out[
        "gseason__pitcher_middle_rate"
    ] = middle.astype(
        np.float32
    )

    out[
        "gseason__pitcher_reverse_rate"
    ] = reverse.astype(
        np.float32
    )

    fast = np.clip(
        comp_values[
            "pitcher_fastball"
        ],
        1e-6,
        1.0,
    )

    breaking = np.clip(
        comp_values[
            "pitcher_breaking"
        ],
        1e-6,
        1.0,
    )

    offspeed = np.clip(
        comp_values[
            "pitcher_offspeed"
        ],
        1e-6,
        1.0,
    )

    mix_sum = np.maximum(
        fast
        + breaking
        + offspeed,
        1e-6,
    )

    fast = (
        fast
        / mix_sum
    )

    breaking = (
        breaking
        / mix_sum
    )

    offspeed = (
        offspeed
        / mix_sum
    )

    out[
        "gseason__pitcher_fastball_rate"
    ] = fast.astype(
        np.float32
    )

    out[
        "gseason__pitcher_breaking_rate"
    ] = breaking.astype(
        np.float32
    )

    out[
        "gseason__pitcher_offspeed_rate"
    ] = offspeed.astype(
        np.float32
    )

    entropy = -(
        fast
        * np.log(
            np.clip(
                fast,
                1e-6,
                1.0,
            )
        )
        + breaking
        * np.log(
            np.clip(
                breaking,
                1e-6,
                1.0,
            )
        )
        + offspeed
        * np.log(
            np.clip(
                offspeed,
                1e-6,
                1.0,
            )
        )
    )

    out[
        "gseason__pitcher_pitchmix_entropy"
    ] = entropy.astype(
        np.float32
    )

    bmid = comp_values[
        "batter_middle"
    ]

    out[
        "gseason__batter_middle_rate"
    ] = bmid.astype(
        np.float32
    )

    out[
        "gseason__batter_middle_minus_pitcher_middle"
    ] = (
        bmid
        - middle
    ).astype(
        np.float32
    )


def build_current_state_for_rows(
    rows,
    available_history,
    scope,
):
    out = pd.DataFrame(
        index=rows.index
    )

    for col in (
        SUCCESS_FEATURES
        + COMPONENT_FEATURES
    ):
        out[
            col
        ] = np.nan

    seasons = pd.to_numeric(
        rows[
            SEASON
        ],
        errors="raise",
    ).astype(int)

    diagnostics = []

    for target_year in sorted(
        seasons.unique()
    ):
        mask = seasons.eq(
            int(
                target_year
            )
        )

        rr = rows.loc[
            mask
        ].copy()

        bank = build_state_bank(
            available_history=available_history,
            target_year=int(
                target_year
            ),
        )

        z = pd.DataFrame(
            index=rr.index
        )

        pstate = transform_success_state(
            rows=rr,
            bank=bank,
            entity="pitcher",
            out=z,
        )

        bstate = transform_success_state(
            rows=rr,
            bank=bank,
            entity="batter",
            out=z,
        )

        z[
            "gseason__pitcher_minus_batter_k80"
        ] = (
            pstate[
                "k80"
            ]
            - bstate[
                "k80"
            ]
        ).astype(
            np.float32
        )

        transform_component_state(
            rows=rr,
            bank=bank,
            out=z,
        )

        z = z.reindex(
            columns=(
                SUCCESS_FEATURES
                + COMPONENT_FEATURES
            )
        )

        z = z.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        # Current-state absent / new player:
        # smoothed features mostly finite already.
        # remaining missing values are neutral 0 for model input.
        z = z.fillna(
            0.0
        ).astype(
            np.float32
        )

        out.loc[
            rr.index,
            z.columns,
        ] = z

        diagnostics.append({
            "scope": str(
                scope
            ),
            "target_year": int(
                target_year
            ),
            "n_rows": int(
                len(
                    rr
                )
            ),
            "pitcher_current_n_mean": float(
                np.mean(
                    pstate[
                        "current_n"
                    ]
                )
            ),
            "pitcher_current_n_positive": float(
                np.mean(
                    pstate[
                        "current_n"
                    ]
                    > 0
                )
            ),
            "batter_current_n_mean": float(
                np.mean(
                    bstate[
                        "current_n"
                    ]
                )
            ),
            "batter_current_n_positive": float(
                np.mean(
                    bstate[
                        "current_n"
                    ]
                    > 0
                )
            ),
        })

    out = out.fillna(
        0.0
    ).astype(
        np.float32
    )

    if not np.isfinite(
        out.to_numpy(
            np.float64
        )
    ).all():
        raise RuntimeError(
            "Current-state block contains NaN/Inf."
        )

    return (
        out,
        pd.DataFrame(
            diagnostics
        ),
    )


# =============================================================================
# Build current L1 feature view
# =============================================================================

def build_fold_l1_view(
    src,
    train,
    base_features,
    cfg,
):
    train_seasons = list(
        cfg[
            "train_seasons"
        ]
    )

    val_year = int(
        cfg[
            "validation_season"
        ]
    )

    df_tr = (
        train.loc[
            train[
                SEASON
            ]
            .isin(
                train_seasons
            )
        ]
        .copy()
    )

    df_va = (
        train.loc[
            train[
                SEASON
            ]
            .eq(
                val_year
            )
        ]
        .copy()
    )

    if (
        len(
            df_tr
        )
        == 0
        or len(
            df_va
        )
        == 0
    ):
        raise RuntimeError(
            f"Empty fold {train_seasons} -> {val_year}"
        )

    # Exact-ASOF5
    pitcher_stats = (
        src.build_pitcher_season_stats(
            df_tr
        )
    )

    (
        tr_exact,
        _,
    ) = src.add_pitcher_exact_asof_features(
        df_tr,
        pitcher_stats,
    )

    (
        va_exact,
        _,
    ) = src.add_pitcher_exact_asof_features(
        df_va,
        pitcher_stats,
    )

    exact_features = list(
        base_features
    )

    for col in src.EXACT_FEATURES:
        if col not in exact_features:
            exact_features.append(
                col
            )

    # HBEB
    (
        tr_control,
        control_features,
        _,
    ) = src.apply_hbeb_pair(
        df_train=tr_exact,
        df_apply=tr_exact,
        base_features=exact_features,
    )

    (
        va_control,
        va_control_features,
        _,
    ) = src.apply_hbeb_pair(
        df_train=tr_exact,
        df_apply=va_exact,
        base_features=exact_features,
    )

    if (
        control_features
        != va_control_features
    ):
        raise RuntimeError(
            "HBEB train/validation feature mismatch"
        )

    # PC FULL 21
    (
        pc_tr,
        _,
    ) = src.build_pitcher_conditional_for_rows(
        rows=df_tr,
        available_history=df_tr,
        scope="state_train_pc",
    )

    (
        pc_va,
        _,
    ) = src.build_pitcher_conditional_for_rows(
        rows=df_va,
        available_history=df_tr,
        scope="state_validation_pc",
    )

    pc_tr = (
        src.mask_pitcher_conditional_to_r_only(
            pc_tr,
            df_tr,
        )
    )

    pc_va = (
        src.mask_pitcher_conditional_to_r_only(
            pc_va,
            df_va,
        )
    )

    tr_view = (
        tr_control.copy()
    )

    va_view = (
        va_control.copy()
    )

    for col in src.PC_FEATURES:
        tr_view[
            col
        ] = pc_tr[
            col
        ]

        va_view[
            col
        ] = pc_va[
            col
        ]

    base_l1_features = (
        list(
            control_features
        )
        + list(
            src.PC_FEATURES
        )
    )

    # New current-state
    (
        state_tr,
        diag_tr,
    ) = build_current_state_for_rows(
        rows=df_tr,
        available_history=df_tr,
        scope=f"fold{cfg['fold']}_train",
    )

    (
        state_va,
        diag_va,
    ) = build_current_state_for_rows(
        rows=df_va,
        available_history=df_tr,
        scope=f"fold{cfg['fold']}_validation",
    )

    for col in (
        SUCCESS_FEATURES
        + COMPONENT_FEATURES
    ):
        tr_view[
            col
        ] = state_tr[
            col
        ]

        va_view[
            col
        ] = state_va[
            col
        ]

    diagnostics = pd.concat(
        [
            diag_tr,
            diag_va,
        ],
        ignore_index=True,
    )

    return (
        df_tr,
        df_va,
        tr_view,
        va_view,
        base_l1_features,
        diagnostics,
    )


# =============================================================================
# Fit treatments
# =============================================================================

def fit_treatment(
    src,
    train_view,
    val_view,
    df_train,
    df_val,
    features,
):
    cat_cols = [
        c
        for c in src.CAT_COLS
        if c in features
    ]

    X_tr = src.prepare_features(
        train_view[
            features
        ],
        cat_cols,
    )

    X_va = src.prepare_features(
        val_view[
            features
        ],
        cat_cols,
    )

    y_tr = df_train[
        TARGET
    ].to_numpy(
        np.float64
    )

    y_va = df_val[
        TARGET
    ].to_numpy(
        np.float64
    )

    model = CatBoostRegressor(
        **src.MODEL_PARAMS
    )

    started = time.time()

    model.fit(
        X_tr,
        y_tr,
        eval_set=(
            X_va,
            y_va,
        ),
        cat_features=cat_cols,
        early_stopping_rounds=src.EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )

    elapsed = float(
        time.time()
        - started
    )

    pred = np.clip(
        np.asarray(
            model.predict(
                X_va
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    best_iteration = int(
        model.get_best_iteration()
    )

    return (
        pred,
        best_iteration,
        elapsed,
    )


# =============================================================================
# Evaluate
# =============================================================================

def align_control_for_fold(
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
            f"Control OOF alignment failed for {val_year}"
        )

    y_raw = aligned[
        TARGET
    ].to_numpy(
        np.float64
    )

    y_oof = aligned[
        "y_true"
    ].to_numpy(
        np.float64
    )

    if np.max(
        np.abs(
            y_raw
            - y_oof
        )
    ) > 1e-12:
        raise RuntimeError(
            f"Control y mismatch for {val_year}"
        )

    return aligned


def routed_prediction(
    control,
    treatment,
    game_type,
    route,
):
    control = np.asarray(
        control,
        dtype=np.float64,
    )

    treatment = np.asarray(
        treatment,
        dtype=np.float64,
    )

    gt = pd.Series(
        game_type
    ).fillna(
        "__MISSING__"
    ).astype(str).to_numpy()

    if route == "GLOBAL":
        return treatment.copy()

    out = control.copy()

    if route == "R_ONLY":
        mask = (
            gt != "F"
        )

    elif route == "F_ONLY":
        mask = (
            gt == "F"
        )

    else:
        raise ValueError(
            route
        )

    out[
        mask
    ] = treatment[
        mask
    ]

    return out


def summarize_oof(
    pred_df,
):
    rows = []

    for (
        treatment,
        route,
    ), g in pred_df.groupby(
        [
            "treatment",
            "route",
        ],
        sort=True,
    ):
        y = g[
            "y_true"
        ].to_numpy(
            np.float64
        )

        control = g[
            "control_pred"
        ].to_numpy(
            np.float64
        )

        pred = g[
            "pred"
        ].to_numpy(
            np.float64
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

        fold_gains = []

        for year, gy in g.groupby(
            SEASON,
            sort=True,
        ):
            yy = gy[
                "y_true"
            ].to_numpy(
                np.float64
            )

            cc = gy[
                "control_pred"
            ].to_numpy(
                np.float64
            )

            pp = gy[
                "pred"
            ].to_numpy(
                np.float64
            )

            fold_gains.append(
                bss(
                    yy,
                    pp,
                )
                - bss(
                    yy,
                    cc,
                )
            )

        rows.append({
            "treatment": str(
                treatment
            ),

            "route": str(
                route
            ),

            "pooled_gain_vs_control_bss": float(
                bss(
                    y,
                    pred,
                )
                - bss(
                    y,
                    control,
                )
            ),

            "recent_2023_2024_gain_bss": float(
                bss(
                    y[
                        recent
                    ],
                    pred[
                        recent
                    ],
                )
                - bss(
                    y[
                        recent
                    ],
                    control[
                        recent
                    ],
                )
            ),

            "gain_2024_bss": float(
                bss(
                    y[
                        latest
                    ],
                    pred[
                        latest
                    ],
                )
                - bss(
                    y[
                        latest
                    ],
                    control[
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

    out = pd.DataFrame(
        rows
    )

    out[
        "strong_candidate"
    ] = (
        (
            out[
                "pooled_gain_vs_control_bss"
            ]
            > 0
        )
        & (
            out[
                "recent_2023_2024_gain_bss"
            ]
            > 0
        )
        & (
            out[
                "gain_2024_bss"
            ]
            > 0
        )
        & (
            out[
                "fold_wins"
            ]
            >= 3
        )
    )

    return (
        out.sort_values(
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


# =============================================================================
# Main
# =============================================================================

def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"{OUT_DIR} already exists.\n"
            "기존 결과 보호를 위해 overwrite하지 않습니다."
        )

    source_path = find_v021_source()

    src = import_module(
        "pc_source_v021",
        source_path,
    )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    # All required ASOF columns must exist.
    train_header = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        nrows=0,
    )

    train_cols = set(
        train_header.columns
    )

    required_state_cols = {
        ID,
        TARGET,
        SEASON,
        GAME_TYPE,
        "pitcher_id",
        "batter_id",

        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_strike_rate",

        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",

        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
    }

    missing_state = sorted(
        required_state_cols
        - train_cols
    )

    if missing_state:
        raise KeyError(
            "Current-state required columns missing:\n"
            + "\n".join(
                f"- {c}"
                for c in missing_state
            )
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

    control_oof = load_control_oof()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "=" * 132
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 132
    )
    print(
        f"v021 source       : {source_path}"
    )
    print(
        f"Control           : v028 seed{CONTROL_SEED} B_L1_PC OOF (REUSE)"
    )
    print(
        "Control CatBoost fits: 0"
    )
    print(
        f"Treatments        : {list(TREATMENTS)}"
    )
    print(
        "Routes            : GLOBAL / R_ONLY / F_ONLY"
    )
    print(
        "=" * 132
    )

    fold_rows = []
    pred_frames = []
    diagnostics_all = []

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
            "=" * 132
        )
        print(
            f"FOLD {fold}: "
            f"{cfg['train_seasons']} -> {val_year}"
        )
        print(
            "=" * 132
        )

        (
            df_tr,
            df_va,
            tr_view,
            va_view,
            base_l1_features,
            diagnostics,
        ) = build_fold_l1_view(
            src=src,
            train=train,
            base_features=base_features,
            cfg=cfg,
        )

        diagnostics[
            "fold"
        ] = fold

        diagnostics_all.append(
            diagnostics
        )

        aligned = align_control_for_fold(
            control_oof=control_oof,
            df_val=df_va,
            val_year=val_year,
        )

        y = aligned[
            "y_true"
        ].to_numpy(
            np.float64
        )

        control = aligned[
            "control_pred"
        ].to_numpy(
            np.float64
        )

        gt = aligned[
            GAME_TYPE
        ].fillna(
            "__MISSING__"
        ).astype(str).to_numpy()

        control_bss = bss(
            y,
            control,
        )

        print(
            f"Control BSS: {control_bss:+.4f}"
        )

        for treatment_name, extra_features in TREATMENTS.items():
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
                    f"{treatment_name}: duplicate feature names"
                )

            (
                treatment_pred,
                best_iteration,
                elapsed,
            ) = fit_treatment(
                src=src,
                train_view=tr_view,
                val_view=va_view,
                df_train=df_tr,
                df_val=df_va,
                features=features,
            )

            print(
                f"{treatment_name:18s} "
                f"best_iter={best_iteration:4d} "
                f"global raw Δ="
                f"{bss(y, treatment_pred)-control_bss:+.4f} BSS"
            )

            for route in [
                "GLOBAL",
                "R_ONLY",
                "F_ONLY",
            ]:
                pred = routed_prediction(
                    control=control,
                    treatment=treatment_pred,
                    game_type=gt,
                    route=route,
                )

                gain = float(
                    bss(
                        y,
                        pred,
                    )
                    - control_bss
                )

                r_mask = (
                    gt != "F"
                )

                f_mask = (
                    gt == "F"
                )

                fold_rows.append({
                    "fold": fold,
                    "validation_year": (
                        val_year
                    ),
                    "treatment": (
                        treatment_name
                    ),
                    "route": (
                        route
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
                    "n_base_features": int(
                        len(
                            base_l1_features
                        )
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
                    "control_bss": float(
                        control_bss
                    ),
                    "model_bss": float(
                        bss(
                            y,
                            pred,
                        )
                    ),
                    "gain_vs_control_bss": (
                        gain
                    ),
                    "R_gain_vs_control_bss": (
                        float(
                            bss(
                                y[
                                    r_mask
                                ],
                                pred[
                                    r_mask
                                ],
                            )
                            - bss(
                                y[
                                    r_mask
                                ],
                                control[
                                    r_mask
                                ],
                            )
                        )
                        if r_mask.any()
                        else np.nan
                    ),
                    "F_gain_vs_control_bss": (
                        float(
                            bss(
                                y[
                                    f_mask
                                ],
                                pred[
                                    f_mask
                                ],
                            )
                            - bss(
                                y[
                                    f_mask
                                ],
                                control[
                                    f_mask
                                ],
                            )
                        )
                        if f_mask.any()
                        else np.nan
                    ),
                })

                pred_frames.append(
                    pd.DataFrame({
                        ID: (
                            df_va[
                                ID
                            ]
                            .to_numpy()
                        ),
                        SEASON: int(
                            val_year
                        ),
                        GAME_TYPE: (
                            gt
                        ),
                        "treatment": (
                            treatment_name
                        ),
                        "route": (
                            route
                        ),
                        "y_true": (
                            y
                        ),
                        "control_pred": (
                            control
                        ),
                        "treatment_raw_pred": (
                            treatment_pred
                        ),
                        "pred": (
                            pred
                        ),
                    })
                )

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = pd.concat(
        pred_frames,
        ignore_index=True,
    )

    summary = summarize_oof(
        pred_df
    )

    diagnostics_df = pd.concat(
        diagnostics_all,
        ignore_index=True,
    )

    manifest_rows = []

    for treatment_name, feats in TREATMENTS.items():
        for f in feats:
            manifest_rows.append({
                "treatment": treatment_name,
                "feature": f,
                "group": (
                    "SUCCESS_EXT"
                    if f in SUCCESS_FEATURES
                    else "COMPONENT"
                ),
            })

    manifest = pd.DataFrame(
        manifest_rows
    )

    fold_df.to_csv(
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

    summary.to_csv(
        OUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics_df.to_csv(
        OUT_DIR
        / "state_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest.to_csv(
        OUT_DIR
        / "state_feature_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 132
    )
    print(
        "FINAL SUMMARY"
    )
    print(
        "=" * 132
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    print()
    print(
        "=" * 132
    )
    print(
        "RECOMMENDED"
    )
    print(
        "=" * 132
    )

    strong = (
        summary.loc[
            summary[
                "strong_candidate"
            ]
        ]
    )

    if len(
        strong
    ):
        print(
            strong.to_string(
                index=False,
                float_format=lambda x: f"{x:+.4f}",
            )
        )
    else:
        print(
            "No strong candidate "
            "(requires pooled/recent/2024 positive and >=3/4 fold wins)."
        )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),

        "control_oof": str(
            CONTROL_OOF_PATH
        ),

        "control_seed": int(
            CONTROL_SEED
        ),

        "control_retrained": False,

        "treatments": {
            k: list(
                v
            )
            for k, v in TREATMENTS.items()
        },

        "state_method": (
            "current-season decomposition from row-local official ASOF "
            "+ historical train constants"
        ),

        "success_history": (
            "exact historical labeled count/sum by player"
        ),

        "component_history": (
            "last historical official cumulative ASOF boundary per player/rate"
        ),

        "component_boundary_note": (
            "Reference-style one-boundary-pitch bridge may remain because "
            "last prior-season subtype is not directly labeled."
        ),

        "routes": [
            "GLOBAL",
            "R_ONLY",
            "F_ONLY",
        ],

        "notes": [
            (
                "Existing Exact5 is retained."
            ),
            (
                "STATE_SUCCESS_EXT does not add pitcher raw/log_n because "
                "those substantially overlap Exact5 current pitcher success state."
            ),
            (
                "STATE_COMPONENTS tests current-season failure/pitchmix/batter-middle "
                "channels that are absent from Exact5."
            ),
            (
                "No validation/test row aggregation is used."
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
        "=" * 132
    )


if __name__ == "__main__":
    main()

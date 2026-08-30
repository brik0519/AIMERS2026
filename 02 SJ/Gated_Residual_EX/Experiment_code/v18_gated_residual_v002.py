# v18_gated_residual_v002.py
#
# 목적
# =============================================================================
# cb_reg_v003 OOF를 기반으로,
# V18-style 두 residual signal을 "gate"로 결합한다.
#
# parent:
#   pitcher_id × batter_hand
#
# detail:
#   pitcher_id × pressure_state × batter_hand
#
# 핵심:
# - 둘 다 같은 방향이면 correction 허용
# - parent만 있으면 작은 correction 허용
# - detail만 있으면 correction하지 않음
# - 둘이 반대 방향이면 correction하지 않음
# - v003 예측을 기본값으로 유지
#
# 검증:
#   discovery:   2021->2022, 2022->2023
#   confirmation: 2023->2024 (파라미터 선택에 사용하지 않음)
#
# 필요한 파일:
#   ./data/train.csv
#   ./validation/cb_reg_v003/oof_validation.npz
#
# 출력:
#   ./validation/v18_gated_residual_v002/
#     parameter_search.csv
#     selected_config.csv
#     transition_results.csv
#     summary.json

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths / constants
# =============================================================================

EXPERIMENT_NAME = "v18_gated_residual_v002"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"

BASE_EXPERIMENT = "cb_reg_v003"
OOF_PATH = (
    Path("./validation")
    / BASE_EXPERIMENT
    / "oof_validation.npz"
)

OUT_DIR = (
    Path("./validation")
    / EXPERIMENT_NAME
)

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"

PARENT_KEYS = [
    "pitcher_id",
    "batter_hand",
]

DETAIL_KEYS = [
    "pitcher_id",
    "pressure_state",
    "batter_hand",
]

TRANSITIONS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]

DISCOVERY_TRANSITIONS = [
    (2021, 2022),
    (2022, 2023),
]

CONFIRMATION_TRANSITION = (
    2023,
    2024,
)

# 이전 strict-transfer 실험에서 선택된 근처 범위를 기준으로,
# gate 자체만 검증하기 위해 작은 grid만 사용.
MIN_N_PARENT_GRID = [
    100,
    200,
]

MIN_N_DETAIL_GRID = [
    100,
    200,
]

SHRINK_PARENT_GRID = [
    300.0,
    700.0,
]

SHRINK_DETAIL_GRID = [
    100.0,
    300.0,
]

# parent만 있는 경우 correction을 얼마나 약하게 둘지
PARENT_ONLY_SCALE_GRID = [
    0.25,
    0.50,
]

# parent + detail이 같은 방향일 때
# detail을 얼마나 반영할지
DETAIL_WEIGHT_GRID = [
    0.25,
    0.50,
]

# 전체 correction 강도
ALPHA_GRID = [
    0.25,
    0.50,
]

CAP_GRID = [
    0.01,
    0.02,
]

MIN_ABS_EFFECT = 0.003

CENTER_RESIDUAL_BY_GAME_TYPE = True


# =============================================================================
# Metrics
# =============================================================================

def brier_score(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def metric_block(
    y_true,
    raw_pred,
    corrected_pred,
):
    raw = brier_score(
        y_true,
        raw_pred,
    )

    corrected = brier_score(
        y_true,
        corrected_pred,
    )

    return {
        "raw_brier": raw,
        "corrected_brier": corrected,
        "gain": float(
            raw - corrected
        ),
    }


# =============================================================================
# Data loading
# =============================================================================

def load_train():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    required = [
        ID_COL,
        TARGET_COL,
        SEASON_COL,
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    header = pd.read_csv(
        TRAIN_PATH,
        nrows=0,
        encoding="utf-8-sig",
    )

    missing = [
        c
        for c in required
        if c not in header.columns
    ]

    if missing:
        raise KeyError(
            "train.csv missing columns: "
            + ", ".join(missing)
        )

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=required,
        encoding="utf-8-sig",
        low_memory=False,
    )

    train[SEASON_COL] = (
        pd.to_numeric(
            train[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    balls = (
        pd.to_numeric(
            train["balls_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    strikes = (
        pd.to_numeric(
            train["strikes_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    train["pressure_state"] = np.where(
        (balls == 3)
        & (strikes == 2),
        "full",
        np.where(
            (balls == 3)
            | (strikes == 2),
            "high",
            "normal",
        ),
    )

    for col in [
        "game_type",
        "pitcher_id",
        "batter_hand",
        "pressure_state",
    ]:
        train[col] = (
            train[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return train


def load_oof():
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            OOF_PATH
        )

    with np.load(
        OOF_PATH,
        allow_pickle=True,
    ) as data:
        keys = set(
            data.files
        )

        required = {
            "row_id",
            "target",
            "validation_season",
        }

        missing = sorted(
            required - keys
        )

        if missing:
            raise KeyError(
                "OOF missing keys: "
                + ", ".join(missing)
            )

        if "prediction" in keys:
            pred = data[
                "prediction"
            ]
        elif "prediction_clip" in keys:
            pred = data[
                "prediction_clip"
            ]
        elif "prediction_raw" in keys:
            pred = np.clip(
                data[
                    "prediction_raw"
                ],
                0.0,
                1.0,
            )
        else:
            raise KeyError(
                "No prediction key in OOF."
            )

        oof = pd.DataFrame(
            {
                ID_COL: data[
                    "row_id"
                ],
                "oof_target": data[
                    "target"
                ].astype(
                    np.float64
                ),
                "prediction": np.clip(
                    np.asarray(
                        pred,
                        dtype=np.float64,
                    ),
                    0.0,
                    1.0,
                ),
                "validation_season": data[
                    "validation_season"
                ].astype(int),
            }
        )

    if oof[
        ID_COL
    ].duplicated().any():
        raise ValueError(
            "OOF row_id duplicated."
        )

    return oof


def build_frame():
    train = load_train()
    oof = load_oof()

    frame = oof.merge(
        train,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if frame[
        SEASON_COL
    ].isna().any():
        raise ValueError(
            "OOF rows missing from train."
        )

    if not np.array_equal(
        frame[
            SEASON_COL
        ]
        .astype(int)
        .to_numpy(),
        frame[
            "validation_season"
        ]
        .astype(int)
        .to_numpy(),
    ):
        raise ValueError(
            "Season mismatch."
        )

    frame[
        "residual"
    ] = (
        frame[
            TARGET_COL
        ].to_numpy(
            dtype=np.float64
        )
        - frame[
            "prediction"
        ].to_numpy(
            dtype=np.float64
        )
    )

    if (
        CENTER_RESIDUAL_BY_GAME_TYPE
    ):
        global_domain_residual = (
            frame
            .groupby(
                [
                    "validation_season",
                    "game_type",
                ],
                dropna=False,
            )[
                "residual"
            ]
            .transform(
                "mean"
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        frame[
            "conditional_residual"
        ] = (
            frame[
                "residual"
            ].to_numpy(
                dtype=np.float64
            )
            - global_domain_residual
        )
    else:
        frame[
            "conditional_residual"
        ] = frame[
            "residual"
        ].to_numpy(
            dtype=np.float64
        )

    return frame


# =============================================================================
# Correction table
# =============================================================================

def fit_table(
    source,
    key_cols,
    min_n,
    shrink_k,
):
    stats = (
        source
        .groupby(
            key_cols,
            dropna=False,
            sort=False,
        )[
            "conditional_residual"
        ]
        .agg(
            n="size",
            effect="mean",
        )
        .reset_index()
    )

    stats[
        "eligible"
    ] = (
        (
            stats["n"]
            >= int(min_n)
        )
        & (
            stats[
                "effect"
            ].abs()
            >= float(
                MIN_ABS_EFFECT
            )
        )
    )

    stats[
        "reliability"
    ] = (
        stats["n"]
        / (
            stats["n"]
            + float(
                shrink_k
            )
        )
    )

    stats[
        "shrunk_effect"
    ] = (
        stats[
            "effect"
        ]
        * stats[
            "reliability"
        ]
    )

    stats.loc[
        ~stats[
            "eligible"
        ],
        "shrunk_effect",
    ] = 0.0

    return (
        stats.loc[
            stats[
                "shrunk_effect"
            ]
            != 0.0,
            key_cols
            + [
                "n",
                "effect",
                "reliability",
                "shrunk_effect",
            ],
        ]
        .copy()
    )


def lookup_effect(
    target,
    table,
    key_cols,
):
    if table.empty:
        return np.zeros(
            len(target),
            dtype=np.float64,
        )

    return (
        target[
            key_cols
        ]
        .merge(
            table[
                key_cols
                + [
                    "shrunk_effect"
                ]
            ],
            on=key_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )[
            "shrunk_effect"
        ]
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )


# =============================================================================
# Gate
# =============================================================================

def build_gated_correction(
    parent_effect,
    detail_effect,
    parent_only_scale,
    detail_weight,
    alpha,
    cap,
):
    parent_effect = np.asarray(
        parent_effect,
        dtype=np.float64,
    )

    detail_effect = np.asarray(
        detail_effect,
        dtype=np.float64,
    )

    if len(
        parent_effect
    ) != len(
        detail_effect
    ):
        raise ValueError(
            "effect length mismatch"
        )

    parent_available = (
        parent_effect
        != 0.0
    )

    detail_available = (
        detail_effect
        != 0.0
    )

    same_sign = (
        parent_available
        & detail_available
        & (
            np.sign(
                parent_effect
            )
            == np.sign(
                detail_effect
            )
        )
    )

    parent_only = (
        parent_available
        & ~detail_available
    )

    # conflict = 둘 다 있는데 방향이 다름
    conflict = (
        parent_available
        & detail_available
        & ~same_sign
    )

    # detail만 있는 경우도 보정하지 않는다.
    detail_only = (
        ~parent_available
        & detail_available
    )

    correction = np.zeros(
        len(
            parent_effect
        ),
        dtype=np.float64,
    )

    # parent만 있을 때는 약한 보정.
    correction[
        parent_only
    ] = (
        parent_effect[
            parent_only
        ]
        * float(
            parent_only_scale
        )
    )

    # 둘이 같은 방향이면 parent를 중심으로 detail까지 섞는다.
    correction[
        same_sign
    ] = (
        (
            1.0
            - float(
                detail_weight
            )
        )
        * parent_effect[
            same_sign
        ]
        + float(
            detail_weight
        )
        * detail_effect[
            same_sign
        ]
    )

    correction *= float(
        alpha
    )

    correction = np.clip(
        correction,
        -float(cap),
        float(cap),
    )

    diagnostics = {
        "parent_available_rows": int(
            parent_available.sum()
        ),
        "detail_available_rows": int(
            detail_available.sum()
        ),
        "same_sign_rows": int(
            same_sign.sum()
        ),
        "parent_only_rows": int(
            parent_only.sum()
        ),
        "conflict_rows": int(
            conflict.sum()
        ),
        "detail_only_rows": int(
            detail_only.sum()
        ),
        "corrected_rows": int(
            (
                correction
                != 0.0
            ).sum()
        ),
    }

    return (
        correction,
        diagnostics,
    )


# =============================================================================
# Transition evaluation
# =============================================================================

def evaluate_transition(
    frame,
    source_year,
    target_year,
    min_n_parent,
    min_n_detail,
    shrink_parent,
    shrink_detail,
    parent_only_scale,
    detail_weight,
    alpha,
    cap,
):
    source = (
        frame.loc[
            frame[
                "validation_season"
            ]
            == int(
                source_year
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    target = (
        frame.loc[
            frame[
                "validation_season"
            ]
            == int(
                target_year
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    parent_table = fit_table(
        source=source,
        key_cols=PARENT_KEYS,
        min_n=min_n_parent,
        shrink_k=shrink_parent,
    )

    detail_table = fit_table(
        source=source,
        key_cols=DETAIL_KEYS,
        min_n=min_n_detail,
        shrink_k=shrink_detail,
    )

    parent_effect = lookup_effect(
        target=target,
        table=parent_table,
        key_cols=PARENT_KEYS,
    )

    detail_effect = lookup_effect(
        target=target,
        table=detail_table,
        key_cols=DETAIL_KEYS,
    )

    (
        correction,
        gate_diag,
    ) = build_gated_correction(
        parent_effect=parent_effect,
        detail_effect=detail_effect,
        parent_only_scale=parent_only_scale,
        detail_weight=detail_weight,
        alpha=alpha,
        cap=cap,
    )

    raw_pred = (
        target[
            "prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    corrected_pred = np.clip(
        raw_pred
        + correction,
        0.0,
        1.0,
    )

    y = (
        target[
            TARGET_COL
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    metrics = metric_block(
        y_true=y,
        raw_pred=raw_pred,
        corrected_pred=corrected_pred,
    )

    corrected_mask = (
        correction
        != 0.0
    )

    result = {
        "source_year": int(
            source_year
        ),
        "target_year": int(
            target_year
        ),
        "min_n_parent": int(
            min_n_parent
        ),
        "min_n_detail": int(
            min_n_detail
        ),
        "shrink_parent": float(
            shrink_parent
        ),
        "shrink_detail": float(
            shrink_detail
        ),
        "parent_only_scale": float(
            parent_only_scale
        ),
        "detail_weight": float(
            detail_weight
        ),
        "alpha": float(
            alpha
        ),
        "cap": float(
            cap
        ),
        "parent_groups": int(
            len(
                parent_table
            )
        ),
        "detail_groups": int(
            len(
                detail_table
            )
        ),
        "target_rows": int(
            len(
                target
            )
        ),
        "coverage": float(
            corrected_mask.mean()
        ),
        "mean_abs_correction": float(
            np.mean(
                np.abs(
                    correction
                )
            )
        ),
        **gate_diag,
        **metrics,
    }

    for domain in [
        "R",
        "F",
    ]:
        mask = (
            target[
                "game_type"
            ]
            .astype(str)
            .to_numpy()
            == domain
        )

        if np.any(
            mask
        ):
            block = metric_block(
                y_true=y[
                    mask
                ],
                raw_pred=raw_pred[
                    mask
                ],
                corrected_pred=corrected_pred[
                    mask
                ],
            )

            result[
                f"{domain}_gain"
            ] = float(
                block[
                    "gain"
                ]
            )

            result[
                f"{domain}_coverage"
            ] = float(
                corrected_mask[
                    mask
                ].mean()
            )
        else:
            result[
                f"{domain}_gain"
            ] = np.nan

            result[
                f"{domain}_coverage"
            ] = np.nan

    return result


# =============================================================================
# Discovery search
# =============================================================================

def run_search(
    frame,
):
    rows = []

    total = (
        len(
            MIN_N_PARENT_GRID
        )
        * len(
            MIN_N_DETAIL_GRID
        )
        * len(
            SHRINK_PARENT_GRID
        )
        * len(
            SHRINK_DETAIL_GRID
        )
        * len(
            PARENT_ONLY_SCALE_GRID
        )
        * len(
            DETAIL_WEIGHT_GRID
        )
        * len(
            ALPHA_GRID
        )
        * len(
            CAP_GRID
        )
    )

    print(
        f"Search configs: {total}"
    )

    counter = 0

    for min_n_parent in (
        MIN_N_PARENT_GRID
    ):
        for min_n_detail in (
            MIN_N_DETAIL_GRID
        ):
            for shrink_parent in (
                SHRINK_PARENT_GRID
            ):
                for shrink_detail in (
                    SHRINK_DETAIL_GRID
                ):
                    for parent_only_scale in (
                        PARENT_ONLY_SCALE_GRID
                    ):
                        for detail_weight in (
                            DETAIL_WEIGHT_GRID
                        ):
                            for alpha in (
                                ALPHA_GRID
                            ):
                                for cap in (
                                    CAP_GRID
                                ):
                                    counter += 1

                                    transition_rows = []

                                    for (
                                        source_year,
                                        target_year,
                                    ) in DISCOVERY_TRANSITIONS:
                                        result = (
                                            evaluate_transition(
                                                frame=frame,
                                                source_year=source_year,
                                                target_year=target_year,
                                                min_n_parent=min_n_parent,
                                                min_n_detail=min_n_detail,
                                                shrink_parent=shrink_parent,
                                                shrink_detail=shrink_detail,
                                                parent_only_scale=parent_only_scale,
                                                detail_weight=detail_weight,
                                                alpha=alpha,
                                                cap=cap,
                                            )
                                        )

                                        transition_rows.append(
                                            result
                                        )

                                    gains = [
                                        row[
                                            "gain"
                                        ]
                                        for row in transition_rows
                                    ]

                                    domain_gains = []

                                    for row in transition_rows:
                                        for col in [
                                            "R_gain",
                                            "F_gain",
                                        ]:
                                            value = row[
                                                col
                                            ]

                                            if np.isfinite(
                                                value
                                            ):
                                                domain_gains.append(
                                                    float(
                                                        value
                                                    )
                                                )

                                    summary = {
                                        "min_n_parent": int(
                                            min_n_parent
                                        ),
                                        "min_n_detail": int(
                                            min_n_detail
                                        ),
                                        "shrink_parent": float(
                                            shrink_parent
                                        ),
                                        "shrink_detail": float(
                                            shrink_detail
                                        ),
                                        "parent_only_scale": float(
                                            parent_only_scale
                                        ),
                                        "detail_weight": float(
                                            detail_weight
                                        ),
                                        "alpha": float(
                                            alpha
                                        ),
                                        "cap": float(
                                            cap
                                        ),
                                        "all_discovery_positive": bool(
                                            all(
                                                gain > 0.0
                                                for gain in gains
                                            )
                                        ),
                                        "worst_discovery_gain": float(
                                            np.min(
                                                gains
                                            )
                                        ),
                                        "mean_discovery_gain": float(
                                            np.mean(
                                                gains
                                            )
                                        ),
                                        "worst_discovery_domain_gain": float(
                                            np.min(
                                                domain_gains
                                            )
                                        ),
                                    }

                                    for row in transition_rows:
                                        year = int(
                                            row[
                                                "target_year"
                                            ]
                                        )

                                        summary[
                                            f"gain_{year}"
                                        ] = float(
                                            row[
                                                "gain"
                                            ]
                                        )

                                        summary[
                                            f"R_gain_{year}"
                                        ] = float(
                                            row[
                                                "R_gain"
                                            ]
                                        )

                                        summary[
                                            f"F_gain_{year}"
                                        ] = float(
                                            row[
                                                "F_gain"
                                            ]
                                        )

                                        summary[
                                            f"coverage_{year}"
                                        ] = float(
                                            row[
                                                "coverage"
                                            ]
                                        )

                                    rows.append(
                                        summary
                                    )

                                    if (
                                        counter % 50
                                        == 0
                                    ):
                                        print(
                                            f"  {counter}/{total}"
                                        )

    return pd.DataFrame(
        rows
    )


def select_config(
    search_df,
):
    robust = (
        search_df.loc[
            search_df[
                "all_discovery_positive"
            ]
        ]
        .copy()
    )

    if robust.empty:
        selected = (
            search_df.sort_values(
                by=[
                    "worst_discovery_gain",
                    "mean_discovery_gain",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
            .copy()
        )

        selected[
            "selection_status"
        ] = (
            "FAIL_NO_ALL_POSITIVE_CONFIG"
        )

        return selected

    # 1차: 두 transition 중 최악의 gain 최대화.
    # 2차: 평균 gain.
    # 3차: domain worst gain.
    # 동률이면 더 보수적인 alpha/cap을 선호.
    selected = (
        robust.sort_values(
            by=[
                "worst_discovery_gain",
                "mean_discovery_gain",
                "worst_discovery_domain_gain",
                "alpha",
                "cap",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[0]
        .copy()
    )

    selected[
        "selection_status"
    ] = "PASS_DISCOVERY"

    return selected


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 80
    )
    print(
        f"Base OOF: {OOF_PATH}"
    )

    frame = build_frame()

    print(
        f"OOF rows: {len(frame):,}"
    )

    # -------------------------------------------------------------------------
    # Discovery only
    # -------------------------------------------------------------------------
    search_df = run_search(
        frame
    )

    search_path = (
        OUT_DIR
        / "parameter_search.csv"
    )

    search_df.to_csv(
        search_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected = select_config(
        search_df
    )

    selected_config = {
        "min_n_parent": int(
            selected[
                "min_n_parent"
            ]
        ),
        "min_n_detail": int(
            selected[
                "min_n_detail"
            ]
        ),
        "shrink_parent": float(
            selected[
                "shrink_parent"
            ]
        ),
        "shrink_detail": float(
            selected[
                "shrink_detail"
            ]
        ),
        "parent_only_scale": float(
            selected[
                "parent_only_scale"
            ]
        ),
        "detail_weight": float(
            selected[
                "detail_weight"
            ]
        ),
        "alpha": float(
            selected[
                "alpha"
            ]
        ),
        "cap": float(
            selected[
                "cap"
            ]
        ),
        "selection_status": str(
            selected[
                "selection_status"
            ]
        ),
        "worst_discovery_gain": float(
            selected[
                "worst_discovery_gain"
            ]
        ),
        "mean_discovery_gain": float(
            selected[
                "mean_discovery_gain"
            ]
        ),
        "worst_discovery_domain_gain": float(
            selected[
                "worst_discovery_domain_gain"
            ]
        ),
    }

    selected_path = (
        OUT_DIR
        / "selected_config.csv"
    )

    pd.DataFrame(
        [
            selected_config
        ]
    ).to_csv(
        selected_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "FROZEN CONFIG — 2024 미사용"
    )
    print(
        "=" * 80
    )

    print(
        pd.DataFrame(
            [
                selected_config
            ]
        ).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Frozen config -> all transitions
    # -------------------------------------------------------------------------
    result_rows = []

    for (
        source_year,
        target_year,
    ) in TRANSITIONS:
        result = evaluate_transition(
            frame=frame,
            source_year=source_year,
            target_year=target_year,
            min_n_parent=selected_config[
                "min_n_parent"
            ],
            min_n_detail=selected_config[
                "min_n_detail"
            ],
            shrink_parent=selected_config[
                "shrink_parent"
            ],
            shrink_detail=selected_config[
                "shrink_detail"
            ],
            parent_only_scale=selected_config[
                "parent_only_scale"
            ],
            detail_weight=selected_config[
                "detail_weight"
            ],
            alpha=selected_config[
                "alpha"
            ],
            cap=selected_config[
                "cap"
            ],
        )

        result_rows.append(
            result
        )

    results_df = pd.DataFrame(
        result_rows
    )

    results_path = (
        OUT_DIR
        / "transition_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "STRICT NEXT-SEASON GATED RESULT"
    )
    print(
        "=" * 80
    )

    display_cols = [
        "source_year",
        "target_year",
        "coverage",
        "same_sign_rows",
        "parent_only_rows",
        "conflict_rows",
        "raw_brier",
        "corrected_brier",
        "gain",
        "R_gain",
        "F_gain",
    ]

    print(
        results_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    gain_map = {
        int(
            row[
                "target_year"
            ]
        ): float(
            row[
                "gain"
            ]
        )
        for _, row in (
            results_df.iterrows()
        )
    }

    discovery_pass = (
        selected_config[
            "selection_status"
        ]
        == "PASS_DISCOVERY"
        and gain_map[
            2022
        ] > 0.0
        and gain_map[
            2023
        ] > 0.0
    )

    confirmation_pass = (
        gain_map[
            2024
        ] > 0.0
    )

    strong_pass = (
        discovery_pass
        and confirmation_pass
    )

    summary = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base_experiment": (
            BASE_EXPERIMENT
        ),
        "parent_keys": (
            PARENT_KEYS
        ),
        "detail_keys": (
            DETAIL_KEYS
        ),
        "discovery_transitions": (
            DISCOVERY_TRANSITIONS
        ),
        "confirmation_transition": (
            CONFIRMATION_TRANSITION
        ),
        "selected_config": (
            selected_config
        ),
        "gains": (
            gain_map
        ),
        "discovery_pass": (
            discovery_pass
        ),
        "confirmation_pass": (
            confirmation_pass
        ),
        "strong_pass": (
            strong_pass
        ),
    }

    summary_path = (
        OUT_DIR
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 80
    )
    print(
        "DECISION"
    )
    print(
        "=" * 80
    )
    print(
        f"2021 -> 2022 : {gain_map[2022]:+.8f}"
    )
    print(
        f"2022 -> 2023 : {gain_map[2023]:+.8f}"
    )
    print(
        f"2023 -> 2024 : {gain_map[2024]:+.8f}  <- confirmation"
    )
    print(
        f"STRONG PASS  : {strong_pass}"
    )

    print()
    print(
        "Saved:"
    )
    print(
        f" - {search_path}"
    )
    print(
        f" - {selected_path}"
    )
    print(
        f" - {results_path}"
    )
    print(
        f" - {summary_path}"
    )


if __name__ == "__main__":
    main()

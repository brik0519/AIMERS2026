#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cb_reg_v040_shadow_rf_equivalence_audit.py
==========================================

목적
----
v039에서 확인된:

    Shadow Control pooled BSS - v028 Control pooled BSS = -5.85

의 원인이 R인지 F인지 재학습 없이 정확히 분해한다.

추가로 v038 R-only state treatment의 효과를
R domain 자체에서 직접 계산해서,
current-season state 개선이 control pipeline mismatch와 무관한지 확인한다.

재학습
------
NONE

입력
----
validation/cb_reg_v028_l1pc_l2_extratrees_no_pc/oof_predictions.csv

validation/cb_reg_v039_shadow_control_state_sanity/
    shadow_control_oof.csv

validation/cb_reg_v038_current_season_asof_state_ablation/
    oof_predictions.csv

출력
----
validation/cb_reg_v040_shadow_rf_equivalence_audit/
    rf_equivalence_by_year.csv
    rf_equivalence_summary.csv
    global_brier_decomposition.csv
    state_r_domain_by_year.csv
    state_r_domain_summary.csv
    metadata.json

실행
----
python cb_reg_v040_shadow_rf_equivalence_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v040_shadow_rf_equivalence_audit"

ROOT = Path(".")

V028_OOF_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v028_l1pc_l2_extratrees_no_pc"
    / "oof_predictions.csv"
)

V039_SHADOW_PATH = (
    ROOT
    / "validation"
    / "cb_reg_v039_shadow_control_state_sanity"
    / "shadow_control_oof.csv"
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
SEASON = "season"
GAME_TYPE = "game_type"

CONTROL_SEED = 42

STATE_TREATMENTS = [
    "STATE_SUCCESS_EXT",
    "STATE_FULL",
]

ROUTE = "R_ONLY"


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
# Load
# =============================================================================

def load_v028():
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
            f"v028 missing columns: {missing}"
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

    df["y_true"] = pd.to_numeric(
        df["y_true"],
        errors="raise",
    ).astype(
        np.float64
    )

    df["v028_pred"] = pd.to_numeric(
        df["B_L1_PC"],
        errors="raise",
    ).astype(
        np.float64
    )

    out = (
        df[
            [
                ID,
                SEASON,
                GAME_TYPE,
                "y_true",
                "v028_pred",
            ]
        ]
        .copy()
    )

    if out[ID].duplicated().any():
        raise RuntimeError(
            "v028 seed42 row_id duplicate"
        )

    return out


def load_shadow():
    if not V039_SHADOW_PATH.exists():
        raise FileNotFoundError(
            V039_SHADOW_PATH
        )

    df = pd.read_csv(
        V039_SHADOW_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        ID,
        SEASON,
        GAME_TYPE,
        "y_true",
        "shadow_control_pred",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            f"v039 shadow missing columns: {missing}"
        )

    df[SEASON] = (
        pd.to_numeric(
            df[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    for col in [
        "y_true",
        "shadow_control_pred",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        ).astype(
            np.float64
        )

    if df[ID].duplicated().any():
        raise RuntimeError(
            "shadow row_id duplicate"
        )

    return (
        df[
            [
                ID,
                SEASON,
                GAME_TYPE,
                "y_true",
                "shadow_control_pred",
            ]
        ]
        .copy()
    )


def load_v038_state():
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
            f"v038 missing columns: {missing}"
        )

    df = (
        df.loc[
            df["treatment"]
            .isin(
                STATE_TREATMENTS
            )
            & df["route"]
            .eq(
                ROUTE
            )
        ]
        .copy()
    )

    df[SEASON] = pd.to_numeric(
        df[SEASON],
        errors="raise",
    ).astype(int)

    df["y_true"] = pd.to_numeric(
        df["y_true"],
        errors="raise",
    ).astype(
        np.float64
    )

    df["treatment_raw_pred"] = pd.to_numeric(
        df["treatment_raw_pred"],
        errors="raise",
    ).astype(
        np.float64
    )

    return df


# =============================================================================
# Control R/F equivalence
# =============================================================================

def build_control_frame(
    v028,
    shadow,
):
    x = (
        shadow.merge(
            v028[
                [
                    ID,
                    "v028_pred",
                ]
            ],
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if x[
        "v028_pred"
    ].isna().any():
        raise RuntimeError(
            "v028-shadow alignment failed"
        )

    x["domain"] = np.where(
        x[GAME_TYPE]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .eq(
            "F"
        ),
        "F",
        "R",
    )

    x["abs_pred_diff"] = np.abs(
        x[
            "shadow_control_pred"
        ]
        - x[
            "v028_pred"
        ]
    )

    x["se_shadow"] = (
        x[
            "shadow_control_pred"
        ]
        - x[
            "y_true"
        ]
    ) ** 2

    x["se_v028"] = (
        x[
            "v028_pred"
        ]
        - x[
            "y_true"
        ]
    ) ** 2

    x["se_delta_shadow_minus_v028"] = (
        x[
            "se_shadow"
        ]
        - x[
            "se_v028"
        ]
    )

    return x


def summarize_rf_equivalence(
    x,
):
    rows = []

    for (
        year,
        domain,
    ), g in x.groupby(
        [
            SEASON,
            "domain",
        ],
        sort=True,
    ):
        y = g[
            "y_true"
        ].to_numpy(
            np.float64
        )

        s = g[
            "shadow_control_pred"
        ].to_numpy(
            np.float64
        )

        old = g[
            "v028_pred"
        ].to_numpy(
            np.float64
        )

        diff = np.abs(
            s - old
        )

        rows.append({
            "year": int(
                year
            ),
            "domain": str(
                domain
            ),
            "n": int(
                len(
                    g
                )
            ),
            "row_fraction_of_year": float(
                len(
                    g
                )
                / len(
                    x.loc[
                        x[
                            SEASON
                        ]
                        .eq(
                            year
                        )
                    ]
                )
            ),
            "max_abs_pred_diff": float(
                np.max(
                    diff
                )
            ),
            "mean_abs_pred_diff": float(
                np.mean(
                    diff
                )
            ),
            "p99_abs_pred_diff": float(
                np.quantile(
                    diff,
                    0.99,
                )
            ),
            "v028_brier": float(
                brier(
                    y,
                    old,
                )
            ),
            "shadow_brier": float(
                brier(
                    y,
                    s,
                )
            ),
            "shadow_minus_v028_brier": float(
                brier(
                    y,
                    s,
                )
                - brier(
                    y,
                    old,
                )
            ),
            "v028_bss": float(
                bss(
                    y,
                    old,
                )
            ),
            "shadow_bss": float(
                bss(
                    y,
                    s,
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

    return pd.DataFrame(
        rows
    )


def summarize_rf_overall(
    x,
):
    rows = []

    for domain, g in x.groupby(
        "domain",
        sort=True,
    ):
        y = g[
            "y_true"
        ].to_numpy(
            np.float64
        )

        s = g[
            "shadow_control_pred"
        ].to_numpy(
            np.float64
        )

        old = g[
            "v028_pred"
        ].to_numpy(
            np.float64
        )

        diff = np.abs(
            s - old
        )

        rows.append({
            "domain": str(
                domain
            ),
            "n": int(
                len(
                    g
                )
            ),
            "row_fraction": float(
                len(
                    g
                )
                / len(
                    x
                )
            ),
            "max_abs_pred_diff": float(
                np.max(
                    diff
                )
            ),
            "mean_abs_pred_diff": float(
                np.mean(
                    diff
                )
            ),
            "p99_abs_pred_diff": float(
                np.quantile(
                    diff,
                    0.99,
                )
            ),
            "v028_brier": float(
                brier(
                    y,
                    old,
                )
            ),
            "shadow_brier": float(
                brier(
                    y,
                    s,
                )
            ),
            "shadow_minus_v028_brier": float(
                brier(
                    y,
                    s,
                )
                - brier(
                    y,
                    old,
                )
            ),
            "v028_bss": float(
                bss(
                    y,
                    old,
                )
            ),
            "shadow_bss": float(
                bss(
                    y,
                    s,
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

    return pd.DataFrame(
        rows
    )


def build_global_brier_decomposition(
    x,
):
    """
    Global Brier delta는 row-weighted domain Brier delta의 정확한 합이므로
    R/F가 total mismatch에 얼마나 기여했는지 직접 분해한다.
    """

    global_delta = float(
        x[
            "se_delta_shadow_minus_v028"
        ]
        .mean()
    )

    rows = []

    for domain, g in x.groupby(
        "domain",
        sort=True,
    ):
        n_frac = float(
            len(
                g
            )
            / len(
                x
            )
        )

        domain_delta = float(
            g[
                "se_delta_shadow_minus_v028"
            ]
            .mean()
        )

        contribution = (
            n_frac
            * domain_delta
        )

        rows.append({
            "domain": str(
                domain
            ),
            "n": int(
                len(
                    g
                )
            ),
            "row_fraction": (
                n_frac
            ),
            "domain_brier_delta": (
                domain_delta
            ),
            "weighted_global_brier_contribution": (
                contribution
            ),
            "share_of_global_brier_delta": (
                float(
                    contribution
                    / global_delta
                )
                if abs(
                    global_delta
                )
                > 1e-18
                else np.nan
            ),
        })

    result = pd.DataFrame(
        rows
    )

    result.loc[
        len(
            result
        )
    ] = {
        "domain": "TOTAL",
        "n": int(
            len(
                x
            )
        ),
        "row_fraction": 1.0,
        "domain_brier_delta": (
            global_delta
        ),
        "weighted_global_brier_contribution": (
            global_delta
        ),
        "share_of_global_brier_delta": 1.0,
    }

    return result


# =============================================================================
# State effect directly on R
# =============================================================================

def build_state_r_audit(
    shadow,
    state,
):
    r_shadow = (
        shadow.loc[
            ~shadow[
                GAME_TYPE
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .eq(
                "F"
            )
        ]
        [
            [
                ID,
                SEASON,
                "y_true",
                "shadow_control_pred",
            ]
        ]
        .copy()
    )

    rows = []

    for treatment in STATE_TREATMENTS:
        t = (
            state.loc[
                state[
                    "treatment"
                ]
                .eq(
                    treatment
                )
                & ~state[
                    GAME_TYPE
                ]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
                .eq(
                    "F"
                ),
                [
                    ID,
                    "treatment_raw_pred",
                ],
            ]
            .copy()
        )

        x = r_shadow.merge(
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
                f"State alignment failed: {treatment}"
            )

        for year, g in x.groupby(
            SEASON,
            sort=True,
        ):
            y = g[
                "y_true"
            ].to_numpy(
                np.float64
            )

            c = g[
                "shadow_control_pred"
            ].to_numpy(
                np.float64
            )

            p = g[
                "treatment_raw_pred"
            ].to_numpy(
                np.float64
            )

            rows.append({
                "treatment": str(
                    treatment
                ),
                "year": int(
                    year
                ),
                "domain": "R",
                "n": int(
                    len(
                        g
                    )
                ),
                "shadow_bss": float(
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
                "gain_vs_shadow_bss": float(
                    bss(
                        y,
                        p,
                    )
                    - bss(
                        y,
                        c,
                    )
                ),
                "shadow_brier": float(
                    brier(
                        y,
                        c,
                    )
                ),
                "treatment_brier": float(
                    brier(
                        y,
                        p,
                    )
                ),
                "brier_improvement": float(
                    brier(
                        y,
                        c,
                    )
                    - brier(
                        y,
                        p,
                    )
                ),
            })

    return pd.DataFrame(
        rows
    )


def summarize_state_r(
    by_year,
):
    rows = []

    for treatment, g in by_year.groupby(
        "treatment",
        sort=True,
    ):
        gains = g[
            "gain_vs_shadow_bss"
        ].to_numpy(
            np.float64
        )

        recent = g.loc[
            g[
                "year"
            ]
            .isin(
                [
                    2023,
                    2024,
                ]
            ),
            "gain_vs_shadow_bss",
        ].to_numpy(
            np.float64
        )

        latest = g.loc[
            g[
                "year"
            ]
            .eq(
                2024
            ),
            "gain_vs_shadow_bss",
        ].to_numpy(
            np.float64
        )

        # pooled R OOF needs reconstruct from per-row source,
        # but mean fold gain is still useful here.
        rows.append({
            "treatment": str(
                treatment
            ),
            "mean_fold_gain_bss": float(
                np.mean(
                    gains
                )
            ),
            "min_fold_gain_bss": float(
                np.min(
                    gains
                )
            ),
            "fold_wins": int(
                np.sum(
                    gains > 0
                )
            ),
            "n_folds": int(
                len(
                    gains
                )
            ),
            "mean_recent_2023_2024_fold_gain_bss": float(
                np.mean(
                    recent
                )
            ),
            "gain_2024_bss": float(
                latest[
                    0
                ]
            )
            if len(
                latest
            )
            else np.nan,
        })

    return pd.DataFrame(
        rows
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

    v028 = load_v028()
    shadow = load_shadow()
    state = load_v038_state()

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
        "Retraining : NONE"
    )
    print(
        "Question   : Is v028-v039 Shadow mismatch concentrated in R or F?"
    )
    print(
        "=" * 132
    )

    control = build_control_frame(
        v028=v028,
        shadow=shadow,
    )

    by_year = summarize_rf_equivalence(
        control
    )

    overall = summarize_rf_overall(
        control
    )

    decomposition = (
        build_global_brier_decomposition(
            control
        )
    )

    state_r_by_year = build_state_r_audit(
        shadow=shadow,
        state=state,
    )

    state_r_summary = summarize_state_r(
        state_r_by_year
    )

    # Save
    by_year.to_csv(
        OUT_DIR
        / "rf_equivalence_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        OUT_DIR
        / "rf_equivalence_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decomposition.to_csv(
        OUT_DIR
        / "global_brier_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state_r_by_year.to_csv(
        OUT_DIR
        / "state_r_domain_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state_r_summary.to_csv(
        OUT_DIR
        / "state_r_domain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 132
    )
    print(
        "R/F CONTROL EQUIVALENCE BY YEAR"
    )
    print(
        "=" * 132
    )

    print(
        by_year.to_string(
            index=False,
            float_format=lambda x: f"{x:+.8f}",
        )
    )

    print()
    print(
        "=" * 132
    )
    print(
        "R/F CONTROL EQUIVALENCE OVERALL"
    )
    print(
        "=" * 132
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:+.8f}",
        )
    )

    print()
    print(
        "=" * 132
    )
    print(
        "GLOBAL BRIER MISMATCH DECOMPOSITION"
    )
    print(
        "=" * 132
    )

    print(
        decomposition.to_string(
            index=False,
            float_format=lambda x: f"{x:+.10f}",
        )
    )

    print()
    print(
        "=" * 132
    )
    print(
        "STATE EFFECT DIRECTLY INSIDE R"
    )
    print(
        "=" * 132
    )

    print(
        state_r_by_year.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print()
    print(
        "SUMMARY:"
    )

    print(
        state_r_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    # Interpretation helper
    r_row = (
        overall.loc[
            overall[
                "domain"
            ]
            .eq(
                "R"
            )
        ]
        .iloc[
            0
        ]
    )

    f_row = (
        overall.loc[
            overall[
                "domain"
            ]
            .eq(
                "F"
            )
        ]
        .iloc[
            0
        ]
    )

    print()
    print(
        "=" * 132
    )
    print(
        "DECISION HELPER"
    )
    print(
        "=" * 132
    )

    print(
        f"R shadow-v028 BSS delta: "
        f"{r_row['shadow_minus_v028_bss']:+.5f}"
    )

    print(
        f"F shadow-v028 BSS delta: "
        f"{f_row['shadow_minus_v028_bss']:+.5f}"
    )

    if (
        abs(
            float(
                r_row[
                    "shadow_minus_v028_bss"
                ]
            )
        )
        <
        abs(
            float(
                f_row[
                    "shadow_minus_v028_bss"
                ]
            )
        )
    ):
        print(
            "Mismatch is larger in F than R."
        )
    else:
        print(
            "Mismatch is NOT concentrated in F; R mismatch is at least as important."
        )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),
        "retraining": False,
        "v028_oof": str(
            V028_OOF_PATH
        ),
        "v039_shadow": str(
            V039_SHADOW_PATH
        ),
        "v038_oof": str(
            V038_OOF_PATH
        ),
        "control_seed": int(
            CONTROL_SEED
        ),
        "domain_definition": {
            "R": "game_type != F",
            "F": "game_type == F",
        },
        "notes": [
            (
                "Global Brier delta is decomposed exactly as row-weighted "
                "R/F Brier contributions."
            ),
            (
                "Subset BSS is reported only for same-domain model comparison; "
                "do not add R-BSS and F-BSS arithmetically."
            ),
            (
                "State R-domain audit compares frozen v038 treatment raw prediction "
                "against freshly trained v039 shadow control only on R rows."
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

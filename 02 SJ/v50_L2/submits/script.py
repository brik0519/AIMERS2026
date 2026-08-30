# script.py

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(".")
EXPERIMENT_NAME = "cb_reg_v050_v42_l2_all_r_shrink015_submit"

MODEL_CANDIDATES = [
    # 권장 제출 ZIP 구조
    ROOT / "model" / f"{EXPERIMENT_NAME}.pkl",

    # standalone train.py가 생성한 학습 직후 구조
    ROOT / "model" / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl",
]

TEST_CANDIDATES = [
    ROOT / "data" / "test.csv",
    ROOT / "open" / "test.csv",
]

SAMPLE_CANDIDATES = [
    ROOT / "data" / "sample_submission.csv",
    ROOT / "open" / "sample_submission.csv",
    ROOT / "sample_submission.csv",
]

OUTPUT_DIR = ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

DEFAULT_TARGET_COL = "control_success"
TARGET = DEFAULT_TARGET_COL



# =============================================================================
# Common
# =============================================================================

def find_first_existing(
    candidates,
    label: str,
    required: bool = True,
):
    for path in candidates:
        if path.exists():
            return path

    if required:
        raise FileNotFoundError(
            f"{label} not found. Expected one of:\n"
            + "\n".join(
                f"  - {p}"
                for p in candidates
            )
        )

    return None


def find_model_path() -> Path:
    return find_first_existing(
        MODEL_CANDIDATES,
        "Model artifact",
        required=True,
    )


def prepare_features(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


def assert_required_columns(
    df: pd.DataFrame,
    required,
    name: str,
):
    missing = sorted(
        set(required)
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"{name} required columns missing: {missing}"
        )


def rounded_cumulative_count(
    n,
    rate,
):
    """
    Must match train.py SUCCESS_FULL semantics exactly:
        cumulative success ~= round(n * rate)
    """
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
        (n > 0.0)
        & np.isfinite(rate)
    )

    out[ok] = np.rint(
        n[ok]
        * rate[ok]
    )

    return out


# =============================================================================
# Exact-ASOF5
# =============================================================================

def add_pitcher_exact_asof_features_test(
    test: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct current-season pitcher success state for each test row.

    Historical constants come only from frozen official train statistics
    stored in the artifact.

    NO information from another test row is used.
    """

    assert_required_columns(
        test,
        {
            SEASON_COL,
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
        },
        "Exact-ASOF test",
    )

    out = test.copy()

    exact_cols = [
        "exact_pitcher_season_n",
        "exact_pitcher_season_logn",
        "exact_pitcher_season_success_rate",
        "exact_pitcher_past_success_rate",
        "exact_pitcher_season_minus_past",
    ]

    for col in exact_cols:
        out[col] = np.nan

    row_seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = (
        pitcher_season_stats
        .copy()
    )

    assert_required_columns(
        stats,
        {
            "pitcher_id",
            SEASON_COL,
            "n",
            "success",
        },
        "pitcher_season_stats artifact",
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    # Only season values are used to select frozen train tables.
    # No test-row aggregation is performed.
    for target_year in sorted(
        row_seasons.unique()
    ):
        row_mask = (
            row_seasons
            == int(target_year)
        )

        rows = (
            out.loc[
                row_mask
            ]
        )

        hist = (
            stats.loc[
                stats[SEASON_COL]
                < int(target_year)
            ]
        )

        if len(hist) > 0:
            # This aggregation is over FROZEN TRAIN statistics only.
            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[
                    ["n", "success"]
                ]
                .sum()
            )

            hist_n = (
                rows[
                    "pitcher_id"
                ]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                rows[
                    "pitcher_id"
                ]
                .map(
                    hist_agg["success"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

        else:
            hist_n = np.zeros(
                len(rows),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(rows),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_n"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_rate = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        # Exact-ASOF5 follows train.py semantics exactly.
        asof_success = (
            asof_n
            * asof_rate
        )

        current_n = np.maximum(
            asof_n
            - hist_n,
            0.0,
        )

        current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                current_success
            ),
            np.clip(
                current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_current = (
            (current_n > 0.0)
            & np.isfinite(
                current_success
            )
        )

        current_rate[
            valid_current
        ] = (
            current_success[
                valid_current
            ]
            / current_n[
                valid_current
            ]
        )

        past_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_past = (
            hist_n > 0.0
        )

        past_rate[
            valid_past
        ] = (
            hist_success[
                valid_past
            ]
            / hist_n[
                valid_past
            ]
        )

        idx = rows.index

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = current_n

        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = np.log1p(
            current_n
        )

        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = current_rate

        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = past_rate

        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = (
            current_rate
            - past_rate
        )

    return out


# =============================================================================
# HB-EB
# =============================================================================

def apply_hbeb(
    df: pd.DataFrame,
    group_cols: list[str],
    n_col: str,
    rate_col: str,
    prior_df: pd.DataFrame,
    global_prior,
):
    """
    Apply frozen train-derived HB-EB prior to each row independently.
    """

    lookup = (
        df[
            group_cols
        ]
        .reset_index(
            drop=True
        )
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n
        * rate
    )

    denominator = (
        n
        + alpha
        + beta
    )

    return np.divide(
        success_count
        + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=(
            denominator
            > 0.0
        ),
    )


def add_hbeb_features(
    test: pd.DataFrame,
    hbeb_priors: dict,
) -> pd.DataFrame:
    out = test.copy()

    for (
        new_col,
        config,
    ) in hbeb_priors.items():
        out[
            new_col
        ] = apply_hbeb(
            df=out,
            group_cols=(
                config[
                    "group_cols"
                ]
            ),
            n_col=(
                config[
                    "n_col"
                ]
            ),
            rate_col=(
                config[
                    "rate_col"
                ]
            ),
            prior_df=(
                config[
                    "prior_df"
                ]
            ),
            global_prior=(
                config[
                    "global_prior"
                ]
            ),
        )

    return out


# =============================================================================
# PC FULL21
# =============================================================================

def add_pc_context_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    x = frame.copy()

    x[
        "v15_count"
    ] = (
        x[
            "balls_before"
        ]
        .astype(str)
        + "_"
        + x[
            "strikes_before"
        ]
        .astype(str)
    )

    x[
        "v15_inning_phase"
    ] = (
        pd.cut(
            pd.to_numeric(
                x["inning"],
                errors="coerce",
            ),
            [
                -np.inf,
                3,
                6,
                9,
                np.inf,
            ],
            labels=[
                "early",
                "middle",
                "late",
                "extra",
            ],
        )
        .astype(str)
    )

    x[
        "v15_pressure"
    ] = np.where(
        (
            x["balls_before"]
            == 3
        )
        & (
            x["strikes_before"]
            == 2
        ),
        "full",
        np.where(
            x[
                "balls_before"
            ]
            == 3,
            "three_ball",
            np.where(
                x[
                    "strikes_before"
                ]
                == 2,
                "two_strike",
                "normal",
            ),
        ),
    )

    x[
        "v15_runners"
    ] = np.where(
        (
            pd.to_numeric(
                x[
                    "num_runners_on"
                ],
                errors="coerce",
            )
            .fillna(0)
            > 0
        ),
        "on",
        "empty",
    )

    x[
        "v15_li_bucket"
    ] = (
        pd.cut(
            pd.to_numeric(
                x["li"],
                errors="coerce",
            ),
            [
                -np.inf,
                0.7,
                1.5,
                np.inf,
            ],
            labels=[
                "low",
                "mid",
                "high",
            ],
        )
        .astype(str)
    )

    return x


def pc_lookup(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    keys: list[str],
    field: str,
    default: float = 0.0,
):
    if table is None or len(table) == 0:
        return np.full(
            len(rows),
            default,
            dtype=np.float32,
        )

    if len(keys) == 1:
        s = (
            table
            .set_index(
                keys[0]
            )[field]
        )

        return (
            rows[
                keys[0]
            ]
            .map(s)
            .fillna(default)
            .to_numpy(
                dtype=np.float32
            )
        )

    idx = (
        pd.MultiIndex
        .from_frame(
            rows[
                keys
            ]
        )
    )

    s = (
        table
        .set_index(
            keys
        )[field]
    )

    return (
        s.reindex(
            idx
        )
        .fillna(default)
        .to_numpy(
            dtype=np.float32
        )
    )


def build_pc_test_features(
    test: pd.DataFrame,
    pc_tables_by_source_year: dict,
    pc_contexts,
    pc_features: list[str],
) -> pd.DataFrame:
    """
    For a target-season test row y:
        source table = y-1 labeled TRAIN season only.

    Every lookup uses only this row's keys.
    """

    required = {
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
    }

    assert_required_columns(
        test,
        required,
        "PC test",
    )

    r_all = (
        add_pc_context_columns(
            test.reset_index(
                drop=False
            )
            .rename(
                columns={
                    "index": "__orig_index",
                }
            )
        )
    )

    out = pd.DataFrame(
        0.0,
        index=np.arange(
            len(r_all)
        ),
        columns=pc_features,
        dtype=np.float32,
    )

    seasons = (
        pd.to_numeric(
            r_all[
                SEASON_COL
            ],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        target_mask = (
            seasons
            .eq(
                int(target_year)
            )
            .to_numpy()
        )

        target_pos = np.flatnonzero(
            target_mask
        )

        rr = (
            r_all.loc[
                target_mask
            ]
            .reset_index(
                drop=True
            )
        )

        source_year = (
            int(target_year)
            - 1
        )

        source_tables = (
            pc_tables_by_source_year
            .get(
                source_year
            )
        )

        if source_tables is None:
            continue

        for (
            ctx,
            _strength,
        ) in pc_contexts:
            table = (
                source_tables
                .get(ctx)
            )

            if table is None:
                continue

            keys = [
                "pitcher_id",
                ctx,
            ]

            prefix = (
                f"pc__{ctx}"
            )

            for field in (
                "rate",
                "reliability",
                "delta",
            ):
                col = (
                    f"{prefix}__{field}"
                )

                if col not in out.columns:
                    raise KeyError(
                        f"PC artifact feature mismatch: {col}"
                    )

                out.loc[
                    target_pos,
                    col,
                ] = pc_lookup(
                    table=table,
                    rows=rr,
                    keys=keys,
                    field=field,
                    default=0.0,
                )

    # PC is an R-only L1 feature block.
    game_type = (
        test[
            GAME_TYPE_COL
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    f_mask = (
        game_type
        == "F"
    )

    out.loc[
        f_mask,
        pc_features,
    ] = 0.0

    values = (
        out[
            pc_features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "PC test features contain NaN/Inf."
        )

    if f_mask.any():
        if not np.all(
            out.loc[
                f_mask,
                pc_features,
            ]
            .to_numpy(
                dtype=np.float64
            )
            == 0.0
        ):
            raise ValueError(
                "F PC masking failed."
            )

    out.index = test.index

    return out


# =============================================================================
# Current-season SUCCESS_FULL13
# =============================================================================

def _build_frozen_success_bank(
    success_history: dict,
    entity: str,
    target_year: int,
):
    """
    Aggregate only frozen TRAIN season-level statistics
    for seasons < target_year.
    """

    stats = (
        success_history[
            entity
        ]
        .copy()
    )

    id_col = (
        "pitcher_id"
        if entity == "pitcher"
        else "batter_id"
    )

    assert_required_columns(
        stats,
        {
            id_col,
            SEASON_COL,
            "n",
            "success",
        },
        f"{entity} success history artifact",
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    hist = (
        stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]
    )

    if len(hist) == 0:
        return {
            "n": {},
            "success": {},
            "prior": 0.5,
        }

    agg = (
        hist
        .groupby(
            id_col,
            sort=False,
            dropna=False,
        )[
            ["n", "success"]
        ]
        .sum()
    )

    max_hist_year = int(
        hist[SEASON_COL]
        .max()
    )

    season_prior = (
        success_history
        .get(
            "season_prior",
            {}
        )
    )

    prior = (
        season_prior
        .get(
            max_hist_year,
            0.5,
        )
    )

    prior = float(
        prior
    )

    if not np.isfinite(
        prior
    ):
        prior = 0.5

    return {
        "n": (
            agg[
                "n"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "success": (
            agg[
                "success"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "prior": prior,
    }


def _transform_success_state(
    rows: pd.DataFrame,
    bank: dict,
    entity: str,
    success_spec: dict,
    out: pd.DataFrame,
):
    spec = (
        success_spec[
            entity
        ]
    )

    id_col = (
        spec[
            "id_col"
        ]
    )

    n_col = (
        spec[
            "n_col"
        ]
    )

    rate_col = (
        spec[
            "rate_col"
        ]
    )

    assert_required_columns(
        rows,
        {
            id_col,
            n_col,
            rate_col,
        },
        f"{entity} SUCCESS state test",
    )

    ids = (
        rows[
            id_col
        ]
    )

    hist_n = (
        ids
        .map(
            bank[
                "n"
            ]
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    hist_success = (
        ids
        .map(
            bank[
                "success"
            ]
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    asof_n = (
        pd.to_numeric(
            rows[
                n_col
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    career = (
        pd.to_numeric(
            rows[
                rate_col
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    prior = float(
        bank[
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

    cumulative_success = (
        rounded_cumulative_count(
            asof_n,
            career,
        )
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
            len(rows),
            prior,
            dtype=np.float64,
        ),
        where=(
            current_n
            > 0.0
        ),
    )

    prefix = (
        f"gseason__{entity}"
    )

    for k in (
        spec[
            "strengths"
        ]
    ):
        out[
            f"{prefix}_rate_k{int(k)}"
        ] = (
            (
                current_success
                + float(k)
                * prior
            )
            / (
                current_n
                + float(k)
            )
        ).astype(
            np.float32
        )

    out[
        f"{prefix}_raw"
    ] = (
        raw.astype(
            np.float32
        )
    )

    out[
        f"{prefix}_log_n"
    ] = (
        np.log1p(
            current_n
        )
        .astype(
            np.float32
        )
    )

    reliability_strength = float(
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
            + reliability_strength
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
        "k80": (
            out[
                f"{prefix}_rate_k80"
            ]
            .to_numpy(
                dtype=np.float64
            )
        ),
    }


def build_success_test_features(
    test: pd.DataFrame,
    success_history: dict,
    success_spec: dict,
    success_features: list[str],
) -> pd.DataFrame:
    """
    Reconstruct current-season pitcher+batter success state.

    Historical terms are frozen train-derived statistics.
    Current terms use only this row's official cumulative asof_* values.
    """

    assert_required_columns(
        test,
        {
            SEASON_COL,
            "pitcher_id",
            "batter_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_batter_n",
            "asof_batter_success_rate",
        },
        "SUCCESS_FULL test",
    )

    out = pd.DataFrame(
        0.0,
        index=test.index,
        columns=success_features,
        dtype=np.float32,
    )

    seasons = (
        pd.to_numeric(
            test[
                SEASON_COL
            ],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        mask = (
            seasons
            .eq(
                int(target_year)
            )
        )

        rows = (
            test.loc[
                mask
            ]
            .copy()
        )

        z = pd.DataFrame(
            index=rows.index
        )

        pitcher_bank = (
            _build_frozen_success_bank(
                success_history=(
                    success_history
                ),
                entity="pitcher",
                target_year=int(
                    target_year
                ),
            )
        )

        batter_bank = (
            _build_frozen_success_bank(
                success_history=(
                    success_history
                ),
                entity="batter",
                target_year=int(
                    target_year
                ),
            )
        )

        pitcher_state = (
            _transform_success_state(
                rows=rows,
                bank=pitcher_bank,
                entity="pitcher",
                success_spec=(
                    success_spec
                ),
                out=z,
            )
        )

        batter_state = (
            _transform_success_state(
                rows=rows,
                bank=batter_bank,
                entity="batter",
                success_spec=(
                    success_spec
                ),
                out=z,
            )
        )

        z[
            "gseason__pitcher_minus_batter_k80"
        ] = (
            pitcher_state[
                "k80"
            ]
            - batter_state[
                "k80"
            ]
        ).astype(
            np.float32
        )

        missing = sorted(
            set(
                success_features
            )
            - set(
                z.columns
            )
        )

        if missing:
            raise KeyError(
                "SUCCESS_FULL feature generation mismatch: "
                f"{missing}"
            )

        z = (
            z[
                success_features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
            .astype(
                np.float32
            )
        )

        out.loc[
            rows.index,
            success_features,
        ] = (
            z[
                success_features
            ]
        )

    values = (
        out[
            success_features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "SUCCESS_FULL test features contain NaN/Inf."
        )

    return out





# =============================================================================
# L2 ExtraTrees residual adapter
# =============================================================================

ADAPTER_SEEDS = [
    42,
    2026,
    314,
]

EXPECTED_L2_CORE_FEATURES = 44
EXPECTED_L2_EXACT_FEATURES = 5
EXPECTED_L2_TOTAL_FEATURES = 49


def l2_numeric_series(
    frame: pd.DataFrame,
    col: str,
    default=np.nan,
) -> pd.Series:
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
        frame[col],
        errors="coerce",
    )


def add_l2_eng(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact old L2 six engineered features.
    Row-local only.
    """

    x = frame.copy()

    career = l2_numeric_series(
        x,
        "asof_pitcher_success_rate",
        0.5,
    )

    p1 = l2_numeric_series(
        x,
        "asof_pitcher_prev1_game_success_rate",
        np.nan,
    )

    p3 = l2_numeric_series(
        x,
        "asof_pitcher_prev3_game_success_rate",
        np.nan,
    )

    p5 = l2_numeric_series(
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
        0.5 * p1
        + 0.3 * p3
        + 0.2 * p5
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
        l2_numeric_series(
            x,
            "asof_pitcher_strike_rate",
            np.nan,
        )
        - l2_numeric_series(
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
        l2_numeric_series(
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
        l2_numeric_series(
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


def transform_l2_core(
    prep: dict,
    rows: pd.DataFrame,
    base_pred,
) -> pd.DataFrame:
    """
    Frozen 2024 adapter preprocessing only.

    core = raw numeric
           + ENG6
           + frozen category codes
           + base_pred

    No map/median is fit from test rows.
    """

    required_prep = {
        "numeric_cols",
        "med",
        "maps",
    }

    missing_prep = sorted(
        required_prep
        - set(
            prep.keys()
        )
    )

    if missing_prep:
        raise KeyError(
            f"L2 adapter_prep missing keys: {missing_prep}"
        )

    r = add_l2_eng(
        rows.reset_index(
            drop=True
        )
    )

    numeric_cols = list(
        prep[
            "numeric_cols"
        ]
    )

    missing_numeric = [
        c
        for c in numeric_cols
        if c not in r.columns
    ]

    if missing_numeric:
        raise KeyError(
            "L2 numeric input columns missing: "
            f"{missing_numeric}"
        )

    A = (
        r[
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
        .copy()
    )

    for col in numeric_cols:
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
        if col not in r.columns:
            raise KeyError(
                f"L2 categorical input column missing: {col}"
            )

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

    if (
        len(
            A
        )
        != len(
            rows
        )
    ):
        raise RuntimeError(
            "L2 core row count mismatch."
        )

    if not np.isfinite(
        A.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "L2 adapter core contains NaN/Inf."
        )

    if (
        A.shape[
            1
        ]
        != EXPECTED_L2_CORE_FEATURES
    ):
        raise ValueError(
            "L2 adapter core feature count mismatch: "
            f"{A.shape[1]} != {EXPECTED_L2_CORE_FEATURES}"
        )

    return A.reset_index(
        drop=True
    )


def build_l2_matrix(
    test: pd.DataFrame,
    transformed_l1: pd.DataFrame,
    l1_routed_pred,
    l2_state: dict,
) -> pd.DataFrame:
    """
    Exact 49-column L2 matrix.

    Exact5 is reused from the already row-local L1 transform.
    """

    required_l2_keys = {
        "adapter_prep",
        "adapter_feature_cols",
        "adapter_core_feature_cols",
        "exact_cols",
    }

    missing = sorted(
        required_l2_keys
        - set(
            l2_state.keys()
        )
    )

    if missing:
        raise KeyError(
            f"L2 artifact missing keys: {missing}"
        )

    core = (
        transform_l2_core(
            prep=(
                l2_state[
                    "adapter_prep"
                ]
            ),
            rows=test,
            base_pred=(
                l1_routed_pred
            ),
        )
    )

    exact_cols = list(
        l2_state[
            "exact_cols"
        ]
    )

    if (
        len(
            exact_cols
        )
        != EXPECTED_L2_EXACT_FEATURES
    ):
        raise ValueError(
            "L2 Exact5 schema size mismatch."
        )

    missing_exact = [
        col
        for col in exact_cols
        if col not in transformed_l1.columns
    ]

    if missing_exact:
        raise KeyError(
            f"L2 Exact5 columns missing: {missing_exact}"
        )

    exact = (
        transformed_l1[
            exact_cols
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

    expected_core_cols = list(
        l2_state[
            "adapter_core_feature_cols"
        ]
    )

    if list(
        core.columns
    ) != expected_core_cols:
        raise ValueError(
            "L2 core feature order mismatch."
        )

    X = pd.concat(
        [
            core,
            exact,
        ],
        axis=1,
    )

    expected_cols = list(
        l2_state[
            "adapter_feature_cols"
        ]
    )

    if (
        len(
            expected_cols
        )
        != EXPECTED_L2_TOTAL_FEATURES
    ):
        raise ValueError(
            "L2 artifact total feature count mismatch."
        )

    if list(
        X.columns
    ) != expected_cols:
        raise ValueError(
            "L2 total feature order mismatch."
        )

    if not np.isfinite(
        X.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "L2 matrix contains NaN/Inf."
        )

    return X


def predict_l2_correction(
    l2_state: dict,
    X_l2: pd.DataFrame,
):
    required = {
        "adapter_models",
        "adapter_seeds",
        "adapter_shrink",
        "selected_route",
    }

    missing = sorted(
        required
        - set(
            l2_state.keys()
        )
    )

    if missing:
        raise KeyError(
            f"L2 artifact required keys missing: {missing}"
        )

    selected_route = str(
        l2_state[
            "selected_route"
        ]
    )

    if selected_route != "ALL_R":
        raise ValueError(
            f"Unexpected L2 route: {selected_route}"
        )

    seeds = [
        int(
            x
        )
        for x in l2_state[
            "adapter_seeds"
        ]
    ]

    if seeds != ADAPTER_SEEDS:
        raise ValueError(
            f"L2 seed mismatch: {seeds}"
        )

    models = (
        l2_state[
            "adapter_models"
        ]
    )

    per_seed = []

    for seed in seeds:
        if seed not in models:
            raise KeyError(
                f"L2 model missing seed={seed}"
            )

        model = (
            models[
                seed
            ]
        )

        if hasattr(
            model,
            "feature_names_in_",
        ):
            if list(
                model.feature_names_in_
            ) != list(
                X_l2.columns
            ):
                raise ValueError(
                    f"L2 seed={seed} feature order mismatch."
                )

        corr = np.asarray(
            model.predict(
                X_l2
            ),
            dtype=np.float64,
        )

        if not np.isfinite(
            corr
        ).all():
            raise ValueError(
                f"L2 seed={seed} correction contains NaN/Inf."
            )

        per_seed.append(
            corr
        )

    corr_matrix = np.column_stack(
        per_seed
    )

    mean_corr = (
        corr_matrix.mean(
            axis=1
        )
    )

    shrink = float(
        l2_state[
            "adapter_shrink"
        ]
    )

    if not np.isclose(
        shrink,
        0.15,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Unexpected L2 shrink: {shrink}"
        )

    return (
        mean_corr,
        corr_matrix,
        shrink,
    )


def build_l1_masks(
    test: pd.DataFrame,
):
    assert_required_columns(
        test,
        {
            GAME_TYPE_COL,
        },
        "routing test",
    )

    gt = (
        test[
            GAME_TYPE_COL
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

    return {
        "F": (
            is_f
        ),
        "R": (
            ~is_f
        ),
    }

# =============================================================================
# Submission
# =============================================================================

def build_submission(
    test_ids,
    predictions,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            ID: test_ids,
            TARGET: predictions,
        }
    )

    if (
        pred_df[ID]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "prediction row_id duplicated."
        )

    # Evaluation server normally provides sample_submission.csv.
    # Search the same candidate locations used in previous submission code.
    sample_path = find_first_existing(
        SAMPLE_CANDIDATES,
        "sample_submission.csv",
        required=False,
    )

    if sample_path is not None:
        sample = pd.read_csv(
            sample_path,
            encoding="utf-8-sig",
        )

        assert_required_columns(
            sample,
            {ID},
            "sample_submission",
        )

        submission = (
            sample[
                [ID]
            ]
            .merge(
                pred_df,
                on=ID,
                how="left",
                validate="one_to_one",
                sort=False,
            )
        )

    else:
        submission = (
            pred_df.copy()
        )

    if (
        submission[
            TARGET
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "submission prediction contains NaN."
        )

    pred = (
        submission[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "submission prediction contains NaN/Inf."
        )

    if not (
        (
            pred >= 0.0
        )
        & (
            pred <= 1.0
        )
    ).all():
        raise ValueError(
            "submission prediction outside [0,1]."
        )

    return submission



# =============================================================================
# Main
# =============================================================================

def main():
    started = time.time()

    print(
        "=" * 108
    )
    print(
        f"Inference: {EXPERIMENT_NAME}"
    )
    print(
        "=" * 108
    )

    test_path = find_first_existing(
        TEST_CANDIDATES,
        "test.csv",
        required=True,
    )

    model_path = (
        find_model_path()
    )

    print(
        f"Model : {model_path}"
    )
    print(
        f"Test  : {test_path}"
    )

    artifact = (
        joblib.load(
            model_path
        )
    )

    required_artifact_keys = {
        "experiment_name",
        "control_model",
        "treatment_model",
        "control_features",
        "treatment_features",
        "control_cat_cols",
        "treatment_cat_cols",
        "pitcher_season_stats",
        "hbeb_priors",
        "pc_tables_by_source_year",
        "success_history",
        "pc_contexts",
        "success_spec",
        "pc_features",
        "success_features",
        "l2",
    }

    missing_artifact = sorted(
        required_artifact_keys
        - set(
            artifact.keys()
        )
    )

    if missing_artifact:
        raise KeyError(
            "Artifact required keys missing: "
            f"{missing_artifact}"
        )

    if (
        artifact[
            "experiment_name"
        ]
        != EXPERIMENT_NAME
    ):
        raise ValueError(
            "Artifact experiment_name mismatch: "
            f"{artifact['experiment_name']}"
        )

    control_model = (
        artifact[
            "control_model"
        ]
    )

    treatment_model = (
        artifact[
            "treatment_model"
        ]
    )

    control_features = list(
        artifact[
            "control_features"
        ]
    )

    treatment_features = list(
        artifact[
            "treatment_features"
        ]
    )

    control_cat_cols = list(
        artifact[
            "control_cat_cols"
        ]
    )

    treatment_cat_cols = list(
        artifact[
            "treatment_cat_cols"
        ]
    )

    pc_features = list(
        artifact[
            "pc_features"
        ]
    )

    success_features = list(
        artifact[
            "success_features"
        ]
    )

    l2_state = (
        artifact[
            "l2"
        ]
    )

    # -----------------------------------------------------------------
    # Load official test only.
    # train.csv is intentionally never loaded.
    # -----------------------------------------------------------------
    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    assert_required_columns(
        test,
        {
            ID,
            SEASON_COL,
            GAME_TYPE_COL,
        },
        "test.csv",
    )

    if (
        test[
            ID
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "test row_id duplicated."
        )

    test_ids = (
        test[
            ID
        ]
        .to_numpy()
    )

    print(
        f"Test rows: {len(test):,}"
    )

    # -----------------------------------------------------------------
    # 1) Exact-ASOF5
    # -----------------------------------------------------------------
    transformed = (
        add_pitcher_exact_asof_features_test(
            test=test,
            pitcher_season_stats=(
                artifact[
                    "pitcher_season_stats"
                ]
            ),
        )
    )

    # -----------------------------------------------------------------
    # 2) HB-EB
    # -----------------------------------------------------------------
    transformed = (
        add_hbeb_features(
            test=transformed,
            hbeb_priors=(
                artifact[
                    "hbeb_priors"
                ]
            ),
        )
    )

    # -----------------------------------------------------------------
    # 3) PC FULL21
    # -----------------------------------------------------------------
    pc = (
        build_pc_test_features(
            test=test,
            pc_tables_by_source_year=(
                artifact[
                    "pc_tables_by_source_year"
                ]
            ),
            pc_contexts=(
                artifact[
                    "pc_contexts"
                ]
            ),
            pc_features=(
                pc_features
            ),
        )
    )

    for col in pc_features:
        transformed[
            col
        ] = (
            pc[
                col
            ]
        )

    # -----------------------------------------------------------------
    # 4) SUCCESS_FULL13
    # -----------------------------------------------------------------
    success = (
        build_success_test_features(
            test=test,
            success_history=(
                artifact[
                    "success_history"
                ]
            ),
            success_spec=(
                artifact[
                    "success_spec"
                ]
            ),
            success_features=(
                success_features
            ),
        )
    )

    for col in success_features:
        transformed[
            col
        ] = (
            success[
                col
            ]
        )

    # -----------------------------------------------------------------
    # 5) Build exact L1 model matrices
    # -----------------------------------------------------------------
    missing_control = [
        col
        for col in control_features
        if col not in transformed.columns
    ]

    missing_treatment = [
        col
        for col in treatment_features
        if col not in transformed.columns
    ]

    if missing_control:
        raise KeyError(
            "CONTROL features missing: "
            f"{missing_control}"
        )

    if missing_treatment:
        raise KeyError(
            "TREATMENT features missing: "
            f"{missing_treatment}"
        )

    X_control = (
        prepare_features(
            transformed[
                control_features
            ],
            control_cat_cols,
        )
    )

    X_treatment = (
        prepare_features(
            transformed[
                treatment_features
            ],
            treatment_cat_cols,
        )
    )

    for name, model, matrix in [
        (
            "CONTROL",
            control_model,
            X_control,
        ),
        (
            "TREATMENT",
            treatment_model,
            X_treatment,
        ),
    ]:
        if hasattr(
            model,
            "feature_names_",
        ):
            if (
                list(
                    model.feature_names_
                )
                != list(
                    matrix.columns
                )
            ):
                raise ValueError(
                    f"{name} training/inference "
                    "feature order mismatch."
                )

    # -----------------------------------------------------------------
    # 6) L1 prediction + exact F/R routing
    # -----------------------------------------------------------------
    control_pred = np.clip(
        np.asarray(
            control_model.predict(
                X_control
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    treatment_pred = np.clip(
        np.asarray(
            treatment_model.predict(
                X_treatment
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    for name, pred in [
        (
            "CONTROL",
            control_pred,
        ),
        (
            "TREATMENT",
            treatment_pred,
        ),
    ]:
        if not np.isfinite(
            pred
        ).all():
            raise ValueError(
                f"{name} prediction contains NaN/Inf."
            )

    masks = (
        build_l1_masks(
            test
        )
    )

    l1_routed = (
        control_pred.copy()
    )

    l1_routed[
        masks[
            "R"
        ]
    ] = (
        treatment_pred[
            masks[
                "R"
            ]
        ]
    )

    if (
        masks[
            "F"
        ].any()
        and not np.array_equal(
            l1_routed[
                masks[
                    "F"
                ]
            ],
            control_pred[
                masks[
                    "F"
                ]
            ],
        )
    ):
        raise ValueError(
            "L1 F routing protection failed."
        )

    if (
        masks[
            "R"
        ].any()
        and not np.array_equal(
            l1_routed[
                masks[
                    "R"
                ]
            ],
            treatment_pred[
                masks[
                    "R"
                ]
            ],
        )
    ):
        raise ValueError(
            "L1 R routing failed."
        )

    # -----------------------------------------------------------------
    # 7) L2 49-feature matrix from frozen 2024 adapter state
    # -----------------------------------------------------------------
    X_l2 = (
        build_l2_matrix(
            test=test,
            transformed_l1=(
                transformed
            ),
            l1_routed_pred=(
                l1_routed
            ),
            l2_state=(
                l2_state
            ),
        )
    )

    (
        l2_corr,
        l2_corr_matrix,
        l2_shrink,
    ) = (
        predict_l2_correction(
            l2_state=(
                l2_state
            ),
            X_l2=X_l2,
        )
    )

    # -----------------------------------------------------------------
    # 8) Final L2 routing:
    #      ALL_R × 0.15
    #      F = L1 exactly protected
    # -----------------------------------------------------------------
    final_pred = (
        l1_routed.copy()
    )

    final_pred[
        masks[
            "R"
        ]
    ] = np.clip(
        l1_routed[
            masks[
                "R"
            ]
        ]
        + float(
            l2_shrink
        )
        * l2_corr[
            masks[
                "R"
            ]
        ],
        0.0,
        1.0,
    )

    if (
        masks[
            "F"
        ].any()
        and not np.array_equal(
            final_pred[
                masks[
                    "F"
                ]
            ],
            l1_routed[
                masks[
                    "F"
                ]
            ],
        )
    ):
        raise ValueError(
            "Final F protection failed."
        )

    if not np.isfinite(
        final_pred
    ).all():
        raise ValueError(
            "Final prediction contains NaN/Inf."
        )

    if not (
        (
            final_pred >= 0.0
        )
        & (
            final_pred <= 1.0
        )
    ).all():
        raise ValueError(
            "Final prediction outside [0,1]."
        )

    # -----------------------------------------------------------------
    # 9) submission.csv
    # -----------------------------------------------------------------
    submission = (
        build_submission(
            test_ids=test_ids,
            predictions=(
                final_pred
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = (
        time.time()
        - started
    )

    r_mask = (
        masks[
            "R"
        ]
    )

    f_mask = (
        masks[
            "F"
        ]
    )

    print()
    print(
        "=" * 108
    )
    print(
        "DONE"
    )
    print(
        "=" * 108
    )
    print(
        f"Rows         : {len(test):,}"
    )
    print(
        f"F protected  : {int(f_mask.sum()):,}"
    )
    print(
        f"R corrected  : {int(r_mask.sum()):,}"
    )
    print(
        f"L2 features  : {X_l2.shape[1]}"
    )
    print(
        f"L2 seeds     : {ADAPTER_SEEDS}"
    )
    print(
        f"L2 shrink    : {l2_shrink}"
    )
    print(
        f"L1 mean      : {l1_routed.mean():.6f}"
    )
    print(
        f"L2 corr mean : {l2_corr.mean():+.6f}"
    )
    if r_mask.any():
        print(
            f"R corr mean  : "
            f"{l2_corr[r_mask].mean():+.6f}"
        )
        print(
            f"R corr abs   : "
            f"{np.mean(np.abs(l2_corr[r_mask])):.6f}"
        )
    print(
        f"Final mean   : {final_pred.mean():.6f}"
    )
    print(
        f"Final min    : {final_pred.min():.6f}"
    )
    print(
        f"Final max    : {final_pred.max():.6f}"
    )
    print(
        f"Output       : {OUTPUT_PATH}"
    )
    print(
        f"Elapsed      : {elapsed:.1f}s"
    )
    print(
        "=" * 108
    )


if __name__ == "__main__":
    main()

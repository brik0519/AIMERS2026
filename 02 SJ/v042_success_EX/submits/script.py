# script.py

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# Paths / constants
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v042_success_full_r_only_submit"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

MODEL_ROOT = Path("./model")

# Official submission structure is simplest when the pkl is directly under model/.
# The nested path is also supported for convenient local smoke testing.
MODEL_CANDIDATES = [
    MODEL_ROOT / f"{EXPERIMENT_NAME}.pkl",
    MODEL_ROOT / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl",
]

OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"


# =============================================================================
# Common
# =============================================================================

def find_model_path() -> Path:
    existing = [
        p
        for p in MODEL_CANDIDATES
        if p.exists()
    ]

    if len(existing) == 0:
        raise FileNotFoundError(
            "Model artifact not found. Expected one of:\n"
            + "\n".join(
                f"  - {p}"
                for p in MODEL_CANDIDATES
            )
        )

    if len(existing) > 1:
        # Deterministic preference: direct model/ file.
        return MODEL_CANDIDATES[0]

    return existing[0]


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
            SEASON,
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
            out[SEASON],
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
            SEASON,
            "n",
            "success",
        },
        "pitcher_season_stats artifact",
    )

    stats[SEASON] = (
        pd.to_numeric(
            stats[SEASON],
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
                stats[SEASON]
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
        SEASON,
        GAME_TYPE,
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
                SEASON
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
            GAME_TYPE
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
            SEASON,
            "n",
            "success",
        },
        f"{entity} success history artifact",
    )

    stats[SEASON] = (
        pd.to_numeric(
            stats[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    hist = (
        stats.loc[
            stats[SEASON]
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
        hist[SEASON]
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
            SEASON,
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
                SEASON
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

    # Evaluation server provides sample_submission.csv.
    # Fallback keeps the code robust for local smoke tests.
    if SAMPLE_SUB_PATH.exists():
        sample = pd.read_csv(
            SAMPLE_SUB_PATH,
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
        "=" * 100
    )
    print(
        f"Inference: {EXPERIMENT_NAME}"
    )
    print(
        "=" * 100
    )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv not found: {TEST_PATH}"
        )

    model_path = (
        find_model_path()
    )

    print(
        f"Model: {model_path}"
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

    # ---------------------------------------------------------------------
    # Load official test only.
    # ---------------------------------------------------------------------
    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    assert_required_columns(
        test,
        {
            ID,
            SEASON,
            GAME_TYPE,
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

    # ---------------------------------------------------------------------
    # 1) Exact-ASOF5
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 2) HB-EB
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 3) PC FULL21
    # ---------------------------------------------------------------------
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
            pc[col]
        )

    # ---------------------------------------------------------------------
    # 4) SUCCESS_FULL13
    # ---------------------------------------------------------------------
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
            success[col]
        )

    # ---------------------------------------------------------------------
    # 5) Build exact model matrices
    # ---------------------------------------------------------------------
    missing_control = [
        c
        for c in control_features
        if c not in transformed.columns
    ]

    missing_treatment = [
        c
        for c in treatment_features
        if c not in transformed.columns
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

    # Hard feature-order guard.
    if hasattr(
        control_model,
        "feature_names_",
    ):
        if (
            list(
                control_model.feature_names_
            )
            != list(
                X_control.columns
            )
        ):
            raise ValueError(
                "CONTROL training/inference feature order mismatch."
            )

    if hasattr(
        treatment_model,
        "feature_names_",
    ):
        if (
            list(
                treatment_model.feature_names_
            )
            != list(
                X_treatment.columns
            )
        ):
            raise ValueError(
                "TREATMENT training/inference feature order mismatch."
            )

    # ---------------------------------------------------------------------
    # 6) Predict both branches
    # ---------------------------------------------------------------------
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

    if not np.isfinite(
        control_pred
    ).all():
        raise ValueError(
            "CONTROL prediction contains NaN/Inf."
        )

    if not np.isfinite(
        treatment_pred
    ).all():
        raise ValueError(
            "TREATMENT prediction contains NaN/Inf."
        )

    # ---------------------------------------------------------------------
    # 7) Official routing
    # ---------------------------------------------------------------------
    game_type = (
        test[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    r_mask = (
        game_type
        != "F"
    )

    f_mask = (
        game_type
        == "F"
    )

    final_pred = (
        control_pred.copy()
    )

    final_pred[
        r_mask
    ] = (
        treatment_pred[
            r_mask
        ]
    )

    # F must be exactly protected.
    if (
        f_mask.any()
        and not np.array_equal(
            final_pred[
                f_mask
            ],
            control_pred[
                f_mask
            ],
        )
    ):
        raise ValueError(
            "F routing protection failed."
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

    # ---------------------------------------------------------------------
    # 8) submission.csv
    # ---------------------------------------------------------------------
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

    print()
    print(
        "=" * 100
    )
    print(
        "DONE"
    )
    print(
        "=" * 100
    )
    print(
        f"Rows       : {len(test):,}"
    )
    print(
        f"R rows     : {int(r_mask.sum()):,}"
    )
    print(
        f"F rows     : {int(f_mask.sum()):,}"
    )
    print(
        f"Control μ  : {control_pred.mean():.6f}"
    )
    print(
        f"Treatment μ: {treatment_pred.mean():.6f}"
    )
    print(
        f"Final μ    : {final_pred.mean():.6f}"
    )
    print(
        f"Final min  : {final_pred.min():.6f}"
    )
    print(
        f"Final max  : {final_pred.max():.6f}"
    )
    print(
        f"Output     : {OUTPUT_PATH}"
    )
    print(
        f"Elapsed    : {elapsed:.1f}s"
    )
    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()

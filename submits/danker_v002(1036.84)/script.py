"""script.py — danker_v001 inference

Final architecture
------------------
L1 R branch:
    TREATMENT CatBoost
    = Control features + R-only Pitcher Conditional FULL

L1 F branch:
    0.75 * CONTROL CatBoost
    + 0.25 * DYNAMIC-ASOF residual CatBoost

Dynamic ASOF anchor:
    A_t = (1 - alpha_t) * A_prev + alpha_t * R_t
    alpha_t = lambda * n_t / (n_t + kappa)
    lambda = 1.0, kappa = 100.0

L2:
    ExtraTrees residual adapter, R only
    F rows are protected exactly at L1.

Evaluation-row independence
---------------------------
A test row uses only:
  * that row's own input / official ASOF values,
  * frozen statistics/tables/models stored in the train-only artifact.

No test-side groupby, rolling, lag, cumulative statistic, frequency, rank,
mean/distribution, or other-row lookup is used to create predictions.

Usage
-----
    python script.py
    python script.py --selfcheck

--selfcheck verifies that selected rows receive exactly the same predictions
when evaluated alone, in the full test, after shuffling, and after split/concat.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 경로 — 프로젝트 표준
# =============================================================================

EXPERIMENT_NAME = "danker_sj_v001"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =============================================================================
# 공통 상수
# =============================================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
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
# 기본 utility
# =============================================================================


def prepare_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def numeric_series(frame: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(
            np.full(len(frame), default, dtype=np.float64),
            index=frame.index,
        )
    return pd.to_numeric(frame[col], errors="coerce")


def add_eng(frame: pd.DataFrame) -> pd.DataFrame:
    """train.py L2 adapter와 동일한 row-local engineered features."""
    x = frame.copy()

    career = numeric_series(x, "asof_pitcher_success_rate", 0.5)
    p1 = numeric_series(x, "asof_pitcher_prev1_game_success_rate", np.nan)
    p3 = numeric_series(x, "asof_pitcher_prev3_game_success_rate", np.nan)
    p5 = numeric_series(x, "asof_pitcher_prev5_game_success_rate", np.nan)

    p1 = p1.fillna(career)
    p3 = p3.fillna(p1)
    p5 = p5.fillna(p3)

    x["eng_recent35"] = 0.5 * p1 + 0.3 * p3 + 0.2 * p5
    x["eng_recent_delta"] = x["eng_recent35"] - career
    x["eng_sb"] = (
        numeric_series(x, "asof_pitcher_strike_rate", np.nan)
        - numeric_series(x, "asof_pitcher_ball_rate", np.nan)
    )

    if "pitcher_hand" in x.columns and "batter_hand" in x.columns:
        x["eng_same"] = (
            x["pitcher_hand"]
            .astype(str)
            .eq(x["batter_hand"].astype(str))
            .astype(np.float32)
        )
    else:
        x["eng_same"] = 0.0

    x["eng_close"] = (
        numeric_series(x, "score_diff_pitcher_team", np.nan)
        .abs()
        .le(1)
        .astype(np.float32)
    )
    x["eng_late"] = (
        numeric_series(x, "inning", np.nan)
        .ge(7)
        .astype(np.float32)
    )
    return x


# =============================================================================
# Artifact load / validation
# =============================================================================


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    art = joblib.load(path)
    if not isinstance(art, dict):
        raise TypeError("MODEL_PATH artifact must be dict")

    if art.get("experiment_name") != EXPERIMENT_NAME:
        raise RuntimeError(
            f"artifact experiment={art.get('experiment_name')!r}; "
            f"expected={EXPERIMENT_NAME!r}"
        )

    required_top = {
        "control",
        "treatment",
        "dynamic",
        "shared_l1_state",
        "l1_routing",
        "l2",
        "l2_routing",
        "expected_target_year",
    }
    missing = sorted(required_top - set(art))
    if missing:
        raise KeyError(f"artifact required keys missing: {missing}")

    return art


def validate_test(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    expected_year = int(art["expected_target_year"])

    required = set(art["shared_l1_state"]["base_features"]) | {
        ID,
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }
    missing = sorted(required - set(test.columns))
    if missing:
        raise KeyError(f"test required columns missing: {missing}")

    season = pd.to_numeric(test[SEASON_COL], errors="raise").astype(int)
    invalid = ~season.eq(expected_year)
    if invalid.any():
        bad = sorted(season.loc[invalid].drop_duplicates().tolist())
        raise RuntimeError(
            f"evaluation season differs from expected {expected_year}: {bad}"
        )

    if test[ID].duplicated().any():
        raise RuntimeError("test row_id contains duplicates")

    return test.reset_index(drop=True)


# =============================================================================
# Exact-ASOF5 — frozen official train history + current row only
# =============================================================================


def add_pitcher_exact_asof_features(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    """Reconstruct current-season state for one fixed evaluation year.

    The historical aggregate comes only from artifact-stored official train
    statistics. No other evaluation row is referenced.
    """
    required = {
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise KeyError(f"Exact-ASOF required columns missing: {missing}")

    out = rows.copy()

    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL], errors="raise"
    ).astype(int)

    hist = stats.loc[stats[SEASON_COL] < int(target_year)]
    if len(hist) > 0:
        hist_agg = (
            hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]]
            .sum()
        )
        hist_n = (
            out["pitcher_id"]
            .map(hist_agg["n"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        hist_success = (
            out["pitcher_id"]
            .map(hist_agg["success"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    else:
        hist_n = np.zeros(len(out), dtype=np.float64)
        hist_success = np.zeros(len(out), dtype=np.float64)

    asof_n = (
        pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    asof_rate = pd.to_numeric(
        out["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    asof_success = asof_n * asof_rate
    raw_current_n = asof_n - hist_n
    current_n = np.maximum(raw_current_n, 0.0)
    raw_current_success = asof_success - hist_success

    current_success = np.where(
        np.isfinite(raw_current_success),
        np.clip(raw_current_success, 0.0, current_n),
        np.nan,
    )

    current_rate = np.full(len(out), np.nan, dtype=np.float64)
    valid_current = (current_n > 0.0) & np.isfinite(current_success)
    current_rate[valid_current] = (
        current_success[valid_current] / current_n[valid_current]
    )

    past_rate = np.full(len(out), np.nan, dtype=np.float64)
    valid_past = hist_n > 0.0
    past_rate[valid_past] = hist_success[valid_past] / hist_n[valid_past]

    out["exact_pitcher_season_n"] = current_n
    out["exact_pitcher_season_logn"] = np.log1p(current_n)
    out["exact_pitcher_season_success_rate"] = current_rate
    out["exact_pitcher_past_success_rate"] = past_rate
    out["exact_pitcher_season_minus_past"] = current_rate - past_rate

    return out


def exact5_matrix(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    transformed = add_pitcher_exact_asof_features(
        rows=rows,
        pitcher_season_stats=pitcher_season_stats,
        target_year=target_year,
    )
    exact = (
        transformed[EXACT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )
    if not np.isfinite(exact.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Exact-ASOF5 matrix contains NaN/Inf")
    return exact


# =============================================================================
# Frozen HB-EB transform
# =============================================================================


def apply_frozen_hbeb(
    df: pd.DataFrame,
    state: dict,
) -> np.ndarray:
    group_cols = list(state["group_cols"])
    n_col = state["n_col"]
    rate_col = state["rate_col"]
    prior_df = state["prior_df"]
    global_prior = tuple(state["global_prior"])

    lookup = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )
    if len(lookup) != len(df):
        raise RuntimeError("HB-EB frozen lookup changed row count")

    alpha = (
        lookup["alpha"]
        .fillna(float(global_prior[0]))
        .to_numpy(dtype=np.float64)
    )
    beta = (
        lookup["beta"]
        .fillna(float(global_prior[1]))
        .to_numpy(dtype=np.float64)
    )

    n = (
        pd.to_numeric(df[n_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    rate = (
        pd.to_numeric(df[rate_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.float64)
    )

    success = n * rate
    denominator = n + alpha + beta
    result = np.divide(
        success + alpha,
        denominator,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return result


def build_control_view(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    target_year = int(art["expected_target_year"])
    shared = art["shared_l1_state"]

    exact = add_pitcher_exact_asof_features(
        rows=test,
        pitcher_season_stats=shared["pitcher_season_stats"],
        target_year=target_year,
    )

    out = exact.copy()
    for new_col, state in shared["hbeb_priors"].items():
        out[new_col] = apply_frozen_hbeb(out, state)

    features = list(art["control"]["features"])
    missing = sorted(set(features) - set(out.columns))
    if missing:
        raise KeyError(f"control transformed columns missing: {missing}")

    return out


# =============================================================================
# Pitcher Conditional — frozen previous-season tables only
# =============================================================================


def _ctx_frame_v15(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()

    x["v15_count"] = (
        x["balls_before"].astype(str)
        + "_"
        + x["strikes_before"].astype(str)
    )
    x["v15_inning_phase"] = pd.cut(
        pd.to_numeric(x["inning"], errors="coerce"),
        [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype(str)
    x["v15_pressure"] = np.where(
        (x["balls_before"] == 3) & (x["strikes_before"] == 2),
        "full",
        np.where(
            x["balls_before"] == 3,
            "three_ball",
            np.where(x["strikes_before"] == 2, "two_strike", "normal"),
        ),
    )
    x["v15_runners"] = np.where(
        pd.to_numeric(x["num_runners_on"], errors="coerce").fillna(0) > 0,
        "on",
        "empty",
    )
    x["v15_li_bucket"] = pd.cut(
        pd.to_numeric(x["li"], errors="coerce"),
        [-np.inf, 0.7, 1.5, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)
    return x


def _pc_lookup(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    keys: list[str],
    field: str,
    default: float = 0.0,
) -> np.ndarray:
    if len(table) == 0:
        return np.full(len(rows), default, dtype=np.float32)

    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return (
            rows[keys[0]]
            .map(s)
            .fillna(default)
            .to_numpy(dtype=np.float32)
        )

    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(dtype=np.float32)


def build_frozen_pc(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    shared = art["shared_l1_state"]
    target_year = int(art["expected_target_year"])
    source_year = target_year - 1

    tables_by_year = shared["pc_tables_by_source_season"]
    if source_year not in tables_by_year:
        raise KeyError(
            f"PC source season {source_year} missing from artifact; "
            f"available={sorted(tables_by_year)}"
        )

    rows = _ctx_frame_v15(test.reset_index(drop=True))
    contexts = shared["pc_contexts"]
    fields = list(shared["pc_fields"])
    default = float(shared.get("pc_unseen_default", 0.0))

    pc_features = list(art["treatment"].get("pc_features", []))
    out = pd.DataFrame(
        0.0,
        index=np.arange(len(rows)),
        columns=pc_features,
        dtype=np.float32,
    )

    season_tables = tables_by_year[source_year]
    for spec in contexts:
        context = spec["context"]
        if context not in season_tables:
            raise KeyError(f"PC context {context!r} missing in artifact")
        table = season_tables[context]
        keys = ["pitcher_id", context]
        for field in fields:
            col = f"pc__{context}__{field}"
            if col not in out.columns:
                raise KeyError(f"PC feature {col!r} missing from treatment schema")
            out[col] = _pc_lookup(
                table=table,
                rows=rows,
                keys=keys,
                field=field,
                default=default,
            )

    # train.py treatment uses PC only for R; F is explicitly neutralized.
    f_mask = (
        test[GAME_TYPE_COL]
        .fillna("__MISSING__")
        .astype(str)
        .eq("F")
        .to_numpy()
    )
    if f_mask.any():
        out.loc[f_mask, pc_features] = 0.0

    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("frozen PITCHER_CONDITIONAL contains NaN/Inf")

    return out


# =============================================================================
# danker_sj_v001: SJ v042 SUCCESS_FULL 추론 (원문 이식)
# =============================================================================

SEASON = SEASON_COL          # SJ 코드 호환 별칭
GAME_TYPE = GAME_TYPE_COL

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


def build_treatment_view(
    control_view: pd.DataFrame,
    test: pd.DataFrame,
    art: dict,
) -> pd.DataFrame:
    out = control_view.copy()
    pc = build_frozen_pc(test, art)

    for col in pc.columns:
        if col in out.columns:
            raise RuntimeError(f"duplicate PC feature at inference: {col}")
        out[col] = pc[col].to_numpy()

    # --- danker_sj_v001: SUCCESS_FULL 13 ---
    shared = art["shared_l1_state"]
    success = build_success_test_features(
        test=test,
        success_history=shared["success_history"],
        success_spec=shared["success_spec"],
        success_features=list(shared["success_features"]),
    )
    for col in success.columns:
        if col in out.columns:
            raise RuntimeError(f"duplicate SUCCESS feature at inference: {col}")
        out[col] = success[col].to_numpy()

    features = list(art["treatment"]["features"])
    missing = sorted(set(features) - set(out.columns))
    if missing:
        raise KeyError(f"treatment transformed columns missing: {missing}")

    return out


# =============================================================================
# Dynamic ASOF anchor — frozen train history + current row only
# =============================================================================


def league_prev_for(
    target_year: int,
    season_league_means: dict,
    fallback_rate: float,
) -> float:
    prev = int(target_year) - 1

    # joblib preserves integer keys, but tolerate stringified keys defensively.
    if prev in season_league_means:
        return float(season_league_means[prev])
    if str(prev) in season_league_means:
        return float(season_league_means[str(prev)])

    older = []
    for year, value in season_league_means.items():
        try:
            y = int(year)
        except Exception:
            continue
        if y < int(target_year):
            older.append(float(value))

    if older:
        return float(np.mean(older))
    return float(fallback_rate)


def compute_prev_anchor(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    season_league_means: dict,
    target_year: int,
    anchor_strength: float,
    fallback_rate: float,
) -> np.ndarray:
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL], errors="raise"
    ).astype(int)

    hist = stats.loc[stats[SEASON_COL] < int(target_year)]
    if len(hist) > 0:
        agg = (
            hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]]
            .sum()
        )
        hist_n = (
            rows["pitcher_id"]
            .map(agg["n"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        hist_success = (
            rows["pitcher_id"]
            .map(agg["success"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    else:
        hist_n = np.zeros(len(rows), dtype=np.float64)
        hist_success = np.zeros(len(rows), dtype=np.float64)

    league_prev = league_prev_for(
        target_year=target_year,
        season_league_means=season_league_means,
        fallback_rate=fallback_rate,
    )

    anchor = (
        hist_success + float(anchor_strength) * league_prev
    ) / (
        hist_n + float(anchor_strength)
    )

    if not np.isfinite(anchor).all():
        raise RuntimeError("Previous anchor contains NaN/Inf")
    return np.clip(anchor, 0.0, 1.0)


def compute_dynamic_anchor(
    exact_view: pd.DataFrame,
    prev_anchor: np.ndarray,
    dynamic_lambda: float,
    dynamic_kappa: float,
) -> np.ndarray:
    prev_anchor = np.asarray(prev_anchor, dtype=np.float64)
    if len(prev_anchor) != len(exact_view):
        raise RuntimeError("Dynamic anchor length mismatch")

    n_t = (
        pd.to_numeric(
            exact_view["exact_pitcher_season_n"], errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    r_t = pd.to_numeric(
        exact_view["exact_pitcher_season_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    valid = (n_t > 0.0) & np.isfinite(r_t)
    reliability = np.zeros(len(exact_view), dtype=np.float64)
    reliability[valid] = n_t[valid] / (
        n_t[valid] + float(dynamic_kappa)
    )
    alpha = np.clip(
        float(dynamic_lambda) * reliability,
        0.0,
        1.0,
    )

    current_state = prev_anchor.copy()
    current_state[valid] = np.clip(r_t[valid], 0.0, 1.0)
    dynamic = (1.0 - alpha) * prev_anchor + alpha * current_state

    if not np.isfinite(dynamic).all():
        raise RuntimeError("Dynamic anchor contains NaN/Inf")
    return np.clip(dynamic, 0.0, 1.0)


# =============================================================================
# L2 adapter transform
# =============================================================================


def transform_adapter_core(
    prep: dict,
    rows: pd.DataFrame,
    base_pred: np.ndarray,
) -> pd.DataFrame:
    r = add_eng(rows.reset_index(drop=True))

    A = (
        r[prep["numeric_cols"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    for col in prep["numeric_cols"]:
        A[col] = (
            A[col]
            .fillna(float(prep["med"].get(col, 0.0)))
            .astype(np.float32)
        )

    for col, mapping in prep["maps"].items():
        A["code_" + col] = (
            r[col]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.float32)
        )

    A["base_pred"] = np.asarray(base_pred, dtype=np.float32)

    if not np.isfinite(A.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("adapter core contains NaN/Inf")
    return A.reset_index(drop=True)


# =============================================================================
# Prediction
# =============================================================================


def _predict_catboost(
    block: dict,
    view: pd.DataFrame,
    label: str,
) -> np.ndarray:
    features = list(block["features"])
    cat_cols = list(block["cat_cols"])
    model = block["model"]

    missing = sorted(set(features) - set(view.columns))
    if missing:
        raise KeyError(f"{label} inference features missing: {missing}")

    X = prepare_features(view[features], cat_cols)

    if hasattr(model, "feature_names_"):
        trained = list(model.feature_names_)
        if trained and trained != features:
            raise RuntimeError(f"{label} artifact feature order mismatch")

    pred = np.asarray(model.predict(X), dtype=np.float64)
    if len(pred) != len(view) or not np.isfinite(pred).all():
        raise RuntimeError(f"{label} prediction invalid")
    return pred


def predict(test: pd.DataFrame, art: dict) -> dict[str, np.ndarray]:
    """Predict every row independently using frozen artifact state."""
    test = test.reset_index(drop=True)
    n = len(test)
    target_year = int(art["expected_target_year"])

    # -----------------------------------------------------------------
    # L1 Control / Treatment views
    # -----------------------------------------------------------------
    control_view = build_control_view(test, art)
    treatment_view = build_treatment_view(control_view, test, art)

    pred_control = np.clip(
        _predict_catboost(art["control"], control_view, "CONTROL"),
        0.0,
        1.0,
    )
    pred_treatment = np.clip(
        _predict_catboost(art["treatment"], treatment_view, "TREATMENT"),
        0.0,
        1.0,
    )

    # -----------------------------------------------------------------
    # Dynamic-ASOF F expert
    # -----------------------------------------------------------------
    dyn = art["dynamic"]
    prev_anchor = compute_prev_anchor(
        rows=test,
        pitcher_season_stats=dyn["pitcher_season_stats"],
        season_league_means=dyn["season_league_means"],
        target_year=target_year,
        anchor_strength=float(dyn["anchor_strength"]),
        fallback_rate=float(dyn["anchor_fallback_rate"]),
    )
    dynamic_anchor = compute_dynamic_anchor(
        exact_view=control_view,
        prev_anchor=prev_anchor,
        dynamic_lambda=float(dyn["dynamic_lambda"]),
        dynamic_kappa=float(dyn["dynamic_kappa"]),
    )
    dynamic_correction = _predict_catboost(
        dyn,
        control_view,
        "DYNAMIC_ASOF",
    )
    pred_dynamic = np.clip(
        dynamic_anchor + dynamic_correction,
        0.0,
        1.0,
    )

    # -----------------------------------------------------------------
    # L1 routing
    # -----------------------------------------------------------------
    gt = (
        test[GAME_TYPE_COL]
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )
    r_mask = gt != "F"
    f_mask = gt == "F"

    blend_w = float(art["l1_routing"]["f_blend_weight"])
    pred_f_blend = np.clip(
        (1.0 - blend_w) * pred_control
        + blend_w * pred_dynamic,
        0.0,
        1.0,
    )

    pred_l1 = np.where(
        r_mask,
        pred_treatment,
        pred_f_blend,
    )
    pred_l1 = np.clip(pred_l1, 0.0, 1.0)

    # -----------------------------------------------------------------
    # L2 ExtraTrees residual adapter — R only
    # -----------------------------------------------------------------
    l2 = art["l2"]
    core = transform_adapter_core(
        prep=l2["adapter_prep"],
        rows=test,
        base_pred=pred_l1,
    )

    if list(core.columns) != list(l2["adapter_core_feature_cols"]):
        raise RuntimeError("L2 adapter core feature order mismatch")

    exact = exact5_matrix(
        rows=test,
        pitcher_season_stats=l2["pitcher_season_stats"],
        target_year=target_year,
    )
    if list(exact.columns) != list(l2["exact_cols"]):
        raise RuntimeError("L2 Exact-ASOF feature order mismatch")

    X_l2 = pd.concat(
        [core.reset_index(drop=True), exact.reset_index(drop=True)],
        axis=1,
    )

    if list(X_l2.columns) != list(l2["adapter_feature_cols"]):
        raise RuntimeError("L2 full feature order mismatch")
    if not np.isfinite(X_l2.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("L2 inference matrix contains NaN/Inf")

    corrections = []
    models = l2["adapter_models"]
    for seed in l2["adapter_seeds"]:
        key = int(seed)
        if key not in models:
            # Defensive support for stringified keys, though joblib normally preserves ints.
            if str(key) in models:
                key = str(key)
            else:
                raise KeyError(f"L2 model seed {seed} missing from artifact")
        corr = np.asarray(models[key].predict(X_l2), dtype=np.float64)
        if len(corr) != n or not np.isfinite(corr).all():
            raise RuntimeError(f"L2 correction invalid for seed={seed}")
        corrections.append(corr)

    if corrections:
        l2_correction = np.mean(np.column_stack(corrections), axis=1)
    else:
        l2_correction = np.zeros(n, dtype=np.float64)

    final = pred_l1.copy()
    final[r_mask] = np.clip(
        final[r_mask]
        + float(l2["adapter_shrink"]) * l2_correction[r_mask],
        0.0,
        1.0,
    )
    # F rows are explicitly protected.
    final[f_mask] = pred_l1[f_mask]

    if not np.isfinite(final).all():
        raise RuntimeError("final prediction contains NaN/Inf")
    if ((final < 0.0) | (final > 1.0)).any():
        raise RuntimeError("final prediction outside [0, 1]")

    return {
        "control": pred_control,
        "treatment": pred_treatment,
        "prev_anchor": prev_anchor,
        "dynamic_anchor": dynamic_anchor,
        "dynamic": pred_dynamic,
        "f_blend": pred_f_blend,
        "l1": pred_l1,
        "l2_correction": l2_correction,
        "final": final,
    }


# =============================================================================
# Row-independence selfcheck
# =============================================================================


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if len(a) else 0.0


def selfcheck(
    test: pd.DataFrame,
    art: dict,
    n_single: int = 30,
    seed: int = 0,
) -> dict:
    """Verify row prediction invariance.

    A: selected rows alone vs same rows inside full test
    B: shuffled full test vs original
    C: split/concat vs original

    Pass condition is exact max|difference| == 0.0 for final prediction and all
    major prediction-stage outputs.
    """
    keys = [
        "control",
        "treatment",
        "prev_anchor",
        "dynamic_anchor",
        "dynamic",
        "f_blend",
        "l1",
        "l2_correction",
        "final",
    ]

    full = predict(test, art)
    rng = np.random.default_rng(seed)

    selected = rng.choice(
        len(test),
        size=min(int(n_single), len(test)),
        replace=False,
    )
    selected = np.sort(selected)

    singles = {
        int(i): predict(test.iloc[[int(i)]].copy(), art)
        for i in selected
    }

    perm = rng.permutation(len(test))
    shuffled = predict(test.iloc[perm].copy(), art)

    cut = len(test) // 2
    left = predict(test.iloc[:cut].copy(), art)
    right = predict(test.iloc[cut:].copy(), art)

    result = {
        "experiment_name": EXPERIMENT_NAME,
        "n_test": int(len(test)),
        "n_single_rows": int(len(selected)),
    }

    all_pass = True
    for key in keys:
        a = max(
            abs(float(singles[int(i)][key][0]) - float(full[key][int(i)]))
            for i in selected
        ) if len(selected) else 0.0
        b = _max_abs(shuffled[key], full[key][perm])
        c = _max_abs(
            np.concatenate([left[key], right[key]]),
            full[key],
        )

        result[f"A_single_vs_full_{key}"] = float(a)
        result[f"B_shuffle_{key}"] = float(b)
        result[f"C_split_concat_{key}"] = float(c)

        key_pass = (a == 0.0) and (b == 0.0) and (c == 0.0)
        all_pass = all_pass and key_pass

        print(
            f"  {key:<14} "
            f"A={a:.3e}  B={b:.3e}  C={c:.3e}"
        )

    result["pass"] = bool(all_pass)
    return result


# =============================================================================
# Submission
# =============================================================================


def build_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    if ID not in sample.columns or TARGET not in sample.columns:
        raise KeyError(
            f"sample_submission must contain [{ID}, {TARGET}]"
        )
    if sample[ID].duplicated().any():
        raise RuntimeError("sample_submission row_id contains duplicates")

    pred_df = pd.DataFrame(
        {
            ID: test[ID].to_numpy(),
            TARGET: np.asarray(prediction, dtype=np.float64),
        }
    )

    out = (
        sample[[ID]]
        .merge(
            pred_df,
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if out[TARGET].isna().any():
        n_missing = int(out[TARGET].isna().sum())
        raise RuntimeError(
            f"submission prediction missing for {n_missing} row_id"
        )

    if len(out) != len(sample):
        raise RuntimeError("submission row count mismatch")

    return out[[ID, TARGET]]


# =============================================================================
# Main
# =============================================================================


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--test", default=str(TEST_PATH))
    parser.add_argument("--sample", default=str(SAMPLE_SUB_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--selfcheck-rows", type=int, default=30)
    args = parser.parse_args(argv)

    t0 = time.time()

    model_path = Path(args.model)
    test_path = Path(args.test)
    sample_path = Path(args.sample)
    output_path = Path(args.output)

    print("=" * 100)
    print(EXPERIMENT_NAME)
    print("=" * 100)
    print(f"Model      : {model_path}")
    print(f"Test       : {test_path}")
    print(f"Submission : {output_path}")
    print("Rule       : row-independent inference")
    print("=" * 100)

    art = load_artifact(model_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    test = validate_test(test, art)

    print("Inference...")
    outputs = predict(test, art)

    if args.selfcheck:
        print("Row-independence selfcheck...")
        result = selfcheck(
            test=test,
            art=art,
            n_single=args.selfcheck_rows,
            seed=0,
        )

        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        selfcheck_path = VALIDATION_DIR / "script_selfcheck.json"
        selfcheck_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Selfcheck saved: {selfcheck_path}")
        print(f"Selfcheck pass : {result['pass']}")

        if not result["pass"]:
            print("행 독립성 검사 실패 — submission 생성 금지")
            sys.exit(2)

    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    submission = build_submission(
        sample=sample,
        test=test,
        prediction=outputs["final"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved      : {output_path}")
    print(f"Rows       : {len(submission):,}")
    print(f"Elapsed    : {time.time() - t0:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
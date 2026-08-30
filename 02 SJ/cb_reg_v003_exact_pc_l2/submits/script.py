#script.py

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(".")
EXPERIMENT_NAME = "cb_reg_v003_exact_pc_l2"

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

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

PC_FIELDS = ("rate", "reliability", "delta")


# =============================================================================
# Basic helpers
# =============================================================================

def find_first_existing(candidates, label):
    for path in candidates:
        if path.exists():
            return path

    pretty = "\n".join(
        f"- {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"{label}를 찾지 못했습니다.\n확인 경로:\n{pretty}"
    )


def find_optional(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def prepare_features(df, cat_cols):
    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


def numeric_series(frame, col, default=np.nan):
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


# =============================================================================
# Exact-ASOF5
# =============================================================================

def add_pitcher_exact_asof_features(
    df,
    pitcher_season_stats,
):
    """
    Test row 자체의 official cumulative ASOF에서,
    artifact에 저장된 과거 completed-season train label 통계를 뺀다.

    다른 test row는 사용하지 않는다.
    """

    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 컬럼 없음: {missing}"
        )

    out = df.copy()

    for col in EXACT_FEATURES:
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = pitcher_season_stats.copy()

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        row_mask = (
            seasons == int(target_year)
        )

        rows = out.loc[
            row_mask
        ]

        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist) > 0:
            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[["n", "success"]]
                .sum()
            )

            hist_n = (
                rows["pitcher_id"]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                rows["pitcher_id"]
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
                rows["asof_pitcher_n"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_rate = (
            pd.to_numeric(
                rows["asof_pitcher_success_rate"],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_success = (
            asof_n
            * asof_rate
        )

        current_n = np.maximum(
            asof_n
            - hist_n,
            0.0,
        )

        raw_current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                raw_current_success
            ),
            np.clip(
                raw_current_success,
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
# Frozen HB-EB
# =============================================================================

def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    lookup = (
        df[group_cols]
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
        .clip(
            lower=0.0
        )
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
        n * rate
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
            denominator > 0.0
        ),
    )


def add_frozen_hbeb(
    df,
    hbeb_priors,
):
    out = df.copy()

    for new_col, info in hbeb_priors.items():
        out[new_col] = apply_hbeb(
            df=out,
            group_cols=list(
                info["group_cols"]
            ),
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=tuple(
                info["global_prior"]
            ),
        )

    return out


# =============================================================================
# Frozen PC lookup
# =============================================================================

def ctx_frame_v15(frame):
    x = frame.copy()

    x["v15_count"] = (
        x["balls_before"]
        .astype(str)
        + "_"
        + x["strikes_before"]
        .astype(str)
    )

    x["v15_inning_phase"] = pd.cut(
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
    ).astype(str)

    x["v15_pressure"] = np.where(
        (
            x["balls_before"] == 3
        )
        & (
            x["strikes_before"] == 2
        ),
        "full",
        np.where(
            x["balls_before"] == 3,
            "three_ball",
            np.where(
                x["strikes_before"] == 2,
                "two_strike",
                "normal",
            ),
        ),
    )

    x["v15_runners"] = np.where(
        pd.to_numeric(
            x["num_runners_on"],
            errors="coerce",
        )
        .fillna(0)
        > 0,
        "on",
        "empty",
    )

    x["v15_li_bucket"] = pd.cut(
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
    ).astype(str)

    return x


def pc_lookup(
    table,
    rows,
    keys,
    field,
    default=0.0,
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
            rows[keys[0]]
            .map(s)
            .fillna(default)
            .to_numpy(
                dtype=np.float32
            )
        )

    idx = pd.MultiIndex.from_frame(
        rows[keys]
    )

    s = (
        table
        .set_index(
            keys
        )[field]
    )

    return (
        s
        .reindex(idx)
        .fillna(default)
        .to_numpy(
            dtype=np.float32
        )
    )


def get_season_tables(
    tables_by_source_season,
    source_year,
):
    if source_year in tables_by_source_season:
        return tables_by_source_season[
            source_year
        ]

    # JSON-like/string-key artifact compatibility
    source_key = str(
        source_year
    )

    if source_key in tables_by_source_season:
        return tables_by_source_season[
            source_key
        ]

    return None


def build_frozen_pc(
    rows,
    shared_state,
):
    contexts = [
        x["context"]
        for x in shared_state[
            "pc_contexts"
        ]
    ]

    pc_features = [
        f"pc__{context}__{field}"
        for context in contexts
        for field in PC_FIELDS
    ]

    out = pd.DataFrame(
        0.0,
        index=rows.index,
        columns=pc_features,
        dtype=np.float32,
    )

    r_all = ctx_frame_v15(
        rows.copy()
    )

    seasons = (
        pd.to_numeric(
            r_all[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    tables_by_source_season = (
        shared_state[
            "pc_tables_by_source_season"
        ]
    )

    for target_year in sorted(
        seasons.unique()
    ):
        target_mask = (
            seasons
            .eq(
                int(target_year)
            )
        )

        rr = (
            r_all.loc[
                target_mask
            ]
            .copy()
        )

        source_year = (
            int(target_year)
            - 1
        )

        season_tables = get_season_tables(
            tables_by_source_season,
            source_year,
        )

        if season_tables is None:
            # 원 training code와 동일:
            # source season lookup이 없으면 neutral 0 vector.
            continue

        for context in contexts:
            table = season_tables.get(
                context
            )

            keys = [
                "pitcher_id",
                context,
            ]

            prefix = (
                f"pc__{context}"
            )

            for field in PC_FIELDS:
                out.loc[
                    target_mask,
                    f"{prefix}__{field}",
                ] = pc_lookup(
                    table=table,
                    rows=rr,
                    keys=keys,
                    field=field,
                    default=float(
                        shared_state.get(
                            "pc_unseen_default",
                            0.0,
                        )
                    ),
                )

    # R-only PC mask:
    # F rows = all zero.
    gt = (
        rows[
            GAME_TYPE_COL
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    f_mask = (
        gt == "F"
    )

    out.loc[
        f_mask,
        pc_features,
    ] = 0.0

    values = out.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "PC matrix contains NaN/Inf"
        )

    return out


# =============================================================================
# L2 adapter
# =============================================================================

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

    p1 = p1.fillna(
        career
    )

    p3 = p3.fillna(
        p1
    )

    p5 = p5.fillna(
        p3
    )

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
            .astype(
                np.float32
            )
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
        .astype(
            np.float32
        )
    )

    x["eng_late"] = (
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


def transform_adapter_core(
    prep,
    rows,
    base_pred,
):
    r = add_eng(
        rows
        .reset_index(
            drop=True
        )
    )

    numeric_cols = list(
        prep[
            "numeric_cols"
        ]
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

    med = prep[
        "med"
    ]

    for col in numeric_cols:
        A[col] = (
            A[col]
            .fillna(
                float(
                    med.get(
                        col,
                        0.0,
                    )
                )
            )
            .astype(
                np.float32
            )
        )

    for col, mapping in prep[
        "maps"
    ].items():
        A[
            "code_" + col
        ] = (
            r[col]
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
            .fillna(-1)
            .astype(
                np.float32
            )
        )

    A["base_pred"] = (
        np.asarray(
            base_pred,
            dtype=np.float32,
        )
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


def exact5_matrix(
    rows,
    pitcher_season_stats,
):
    transformed = (
        add_pitcher_exact_asof_features(
            rows,
            pitcher_season_stats,
        )
    )

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
        .fillna(0.0)
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
            "Exact-ASOF5 matrix contains NaN/Inf"
        )

    return exact


# =============================================================================
# Prediction
# =============================================================================

def validate_artifact(artifact):
    required = {
        "control",
        "treatment",
        "shared_l1_state",
        "l2",
        "l1_routing",
        "l2_routing",
    }

    missing = sorted(
        required - set(
            artifact.keys()
        )
    )

    if missing:
        raise KeyError(
            f"artifact missing keys: {missing}"
        )

    if artifact.get(
        "experiment_name"
    ) != EXPERIMENT_NAME:
        raise RuntimeError(
            "Wrong model artifact: "
            f"{artifact.get('experiment_name')}"
        )


def build_l1_views(
    test,
    artifact,
):
    shared = artifact[
        "shared_l1_state"
    ]

    # 1) Exact-ASOF5 from frozen train stats.
    base = add_pitcher_exact_asof_features(
        test,
        shared[
            "pitcher_season_stats"
        ],
    )

    # 2) Frozen HB-EB.
    control_view = add_frozen_hbeb(
        base,
        shared[
            "hbeb_priors"
        ],
    )

    control_info = artifact[
        "control"
    ]

    treatment_info = artifact[
        "treatment"
    ]

    control_features = list(
        control_info[
            "features"
        ]
    )

    treatment_features = list(
        treatment_info[
            "features"
        ]
    )

    missing_control = sorted(
        set(
            control_features
        )
        - set(
            control_view.columns
        )
    )

    if missing_control:
        raise KeyError(
            "control inference feature missing: "
            f"{missing_control}"
        )

    # 3) Frozen previous-season PC lookup.
    pc = build_frozen_pc(
        test,
        shared,
    )

    treatment_view = (
        control_view.copy()
    )

    for col in treatment_info[
        "pc_features"
    ]:
        if col not in pc.columns:
            raise KeyError(
                f"PC feature missing: {col}"
            )

        treatment_view[
            col
        ] = pc[
            col
        ]

    missing_treatment = sorted(
        set(
            treatment_features
        )
        - set(
            treatment_view.columns
        )
    )

    if missing_treatment:
        raise KeyError(
            "treatment inference feature missing: "
            f"{missing_treatment}"
        )

    return (
        control_view,
        treatment_view,
    )


def predict_l1(
    test,
    artifact,
):
    (
        control_view,
        treatment_view,
    ) = build_l1_views(
        test,
        artifact,
    )

    control_info = artifact[
        "control"
    ]

    treatment_info = artifact[
        "treatment"
    ]

    X_control = prepare_features(
        control_view[
            control_info[
                "features"
            ]
        ],
        control_info[
            "cat_cols"
        ],
    )

    X_treatment = prepare_features(
        treatment_view[
            treatment_info[
                "features"
            ]
        ],
        treatment_info[
            "cat_cols"
        ],
    )

    pred_control = np.asarray(
        control_info[
            "model"
        ].predict(
            X_control
        ),
        dtype=np.float64,
    )

    pred_treatment = np.asarray(
        treatment_info[
            "model"
        ].predict(
            X_treatment
        ),
        dtype=np.float64,
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

    r_mask = (
        gt != "F"
    )

    pred_l1 = np.where(
        r_mask,
        pred_treatment,
        pred_control,
    )

    pred_l1 = np.clip(
        pred_l1,
        0.0,
        1.0,
    )

    return (
        pred_l1,
        r_mask,
    )


def predict_l2(
    test,
    pred_l1,
    r_mask,
    artifact,
):
    l2 = artifact[
        "l2"
    ]

    core = transform_adapter_core(
        prep=l2[
            "adapter_prep"
        ],
        rows=test,
        base_pred=pred_l1,
    )

    exact = exact5_matrix(
        rows=test,
        pitcher_season_stats=l2[
            "pitcher_season_stats"
        ],
    )

    X = pd.concat(
        [
            core,
            exact,
        ],
        axis=1,
    )

    expected_cols = list(
        l2[
            "adapter_feature_cols"
        ]
    )

    missing = sorted(
        set(
            expected_cols
        )
        - set(
            X.columns
        )
    )

    extra = sorted(
        set(
            X.columns
        )
        - set(
            expected_cols
        )
    )

    if missing:
        raise KeyError(
            f"L2 feature missing: {missing}"
        )

    if extra:
        raise RuntimeError(
            f"Unexpected L2 features: {extra}"
        )

    # Training order exactly.
    X = X[
        expected_cols
    ]

    if not np.isfinite(
        X.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "L2 inference matrix contains NaN/Inf"
        )

    models = l2[
        "adapter_models"
    ]

    if isinstance(
        models,
        dict,
    ):
        model_list = list(
            models.values()
        )
    else:
        model_list = list(
            models
        )

    if len(model_list) == 0:
        raise RuntimeError(
            "No L2 adapter models"
        )

    corrections = []

    # IMPORTANT:
    # test 전체 prediction을 각 model에 넣는 것은 row별 독립적인
    # tree inference일 뿐, test-row aggregate가 아니다.
    for model in model_list:
        corr = np.asarray(
            model.predict(
                X
            ),
            dtype=np.float64,
        )

        corrections.append(
            corr
        )

    mean_corr = np.mean(
        np.vstack(
            corrections
        ),
        axis=0,
    )

    shrink = float(
        l2[
            "adapter_shrink"
        ]
    )

    pred_final = (
        np.asarray(
            pred_l1,
            dtype=np.float64,
        )
        .copy()
    )

    # R only.
    pred_final[
        r_mask
    ] = (
        pred_l1[
            r_mask
        ]
        + shrink
        * mean_corr[
            r_mask
        ]
    )

    # F is intentionally untouched.
    pred_final = np.clip(
        pred_final,
        0.0,
        1.0,
    )

    return (
        pred_final,
        mean_corr,
    )


# =============================================================================
# Submission
# =============================================================================

def make_submission(
    test,
    predictions,
):
    sample_path = find_optional(
        SAMPLE_CANDIDATES
    )

    if sample_path is not None:
        submission = pd.read_csv(
            sample_path,
            encoding="utf-8-sig",
        )

        if len(
            submission
        ) != len(
            test
        ):
            raise RuntimeError(
                "sample_submission/test row count mismatch: "
                f"{len(submission)} != {len(test)}"
            )

        if (
            ID in submission.columns
            and ID in test.columns
        ):
            # row order mismatch가 있으면 ID 기준으로 맞춘다.
            if not np.array_equal(
                submission[
                    ID
                ].to_numpy(),
                test[
                    ID
                ].to_numpy(),
            ):
                pred_map = pd.Series(
                    predictions,
                    index=test[
                        ID
                    ],
                )

                aligned = (
                    submission[
                        ID
                    ]
                    .map(
                        pred_map
                    )
                )

                if aligned.isna().any():
                    raise RuntimeError(
                        "sample_submission row_id alignment failed"
                    )

                predictions = (
                    aligned
                    .to_numpy(
                        dtype=np.float64
                    )
                )

        if DEFAULT_TARGET_COL in submission.columns:
            pred_col = DEFAULT_TARGET_COL

        else:
            non_id_cols = [
                c
                for c in submission.columns
                if c != ID
            ]

            if len(
                non_id_cols
            ) != 1:
                raise RuntimeError(
                    "submission prediction column을 "
                    "자동 결정할 수 없습니다: "
                    f"{submission.columns.tolist()}"
                )

            pred_col = (
                non_id_cols[0]
            )

        submission[
            pred_col
        ] = predictions

    else:
        if ID not in test.columns:
            raise KeyError(
                f"test missing {ID}"
            )

        submission = pd.DataFrame({
            ID: (
                test[
                    ID
                ]
                .to_numpy()
            ),
            DEFAULT_TARGET_COL: predictions,
        })

    return submission


def main():
    print(
        "=" * 108
    )

    print(
        EXPERIMENT_NAME
    )

    print(
        "=" * 108
    )

    model_path = find_first_existing(
        MODEL_CANDIDATES,
        "model artifact",
    )

    test_path = find_first_existing(
        TEST_CANDIDATES,
        "test.csv",
    )

    print(
        f"Model : {model_path}"
    )

    print(
        f"Test  : {test_path}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    artifact = joblib.load(
        model_path
    )

    validate_artifact(
        artifact
    )

    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if SEASON_COL not in test.columns:
        raise KeyError(
            f"test missing {SEASON_COL}"
        )

    test[
        SEASON_COL
    ] = (
        pd.to_numeric(
            test[
                SEASON_COL
            ],
            errors="raise",
        )
        .astype(int)
    )

    required = {
        ID,
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
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    }

    missing = sorted(
        required - set(
            test.columns
        )
    )

    if missing:
        raise KeyError(
            f"test required columns missing: {missing}"
        )

    expected_target_year = artifact.get(
        "expected_target_year"
    )

    if expected_target_year is not None:
        actual_seasons = sorted(
            test[
                SEASON_COL
            ]
            .unique()
            .tolist()
        )

        if actual_seasons != [
            int(
                expected_target_year
            )
        ]:
            print(
                "[WARN] test season differs from "
                f"artifact expected_target_year={expected_target_year}: "
                f"{actual_seasons}"
            )

    pred_l1, r_mask = predict_l1(
        test,
        artifact,
    )

    pred_final, mean_corr = predict_l2(
        test=test,
        pred_l1=pred_l1,
        r_mask=r_mask,
        artifact=artifact,
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

    f_mask = (
        gt == "F"
    )

    # Safety assertion:
    # F must be EXACTLY L1 after L2 routing.
    if f_mask.any():
        if not np.array_equal(
            pred_final[
                f_mask
            ],
            pred_l1[
                f_mask
            ],
        ):
            raise RuntimeError(
                "F protection violated: L2 changed F predictions"
            )

    submission = make_submission(
        test,
        pred_final,
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

    print(
        "-" * 108
    )

    print(
        f"Rows        : {len(test):,}"
    )

    print(
        f"R/F         : {int(r_mask.sum()):,} / {int(f_mask.sum()):,}"
    )

    print(
        f"L1 mean     : {float(np.mean(pred_l1)):.8f}"
    )

    print(
        f"L2 corr mean: {float(np.mean(mean_corr[r_mask])):+.8f}"
        if r_mask.any()
        else "L2 corr mean: N/A"
    )

    print(
        f"Final mean  : {float(np.mean(pred_final)):.8f}"
    )

    print(
        f"Final min/max: "
        f"{float(np.min(pred_final)):.8f} / "
        f"{float(np.max(pred_final)):.8f}"
    )

    print(
        f"Saved       : {OUTPUT_PATH}"
    )

    print(
        "=" * 108
    )


if __name__ == "__main__":
    main()

# cb_reg_v020_pitcher_conditional_ablation.py
# ============================================================
# Paired feature ablation
#
# Control
#   canonical cb_reg_v003 + Exact-ASOF 5 + HB-EB
#
# Treatment
#   Control + V15/V18 PITCHER_CONDITIONAL 21
#
# 목적
#   Residual correction을 완전히 제외한 상태에서
#   PITCHER_CONDITIONAL 자체의 incremental value만 측정한다.
#
# PITCHER_CONDITIONAL 원본 의미
#   바로 이전 labeled season에서 투수의 상황별 성공률을 만들고,
#   투수 자신의 이전 시즌 baseline으로 shrink한다.
#   7 contexts x (rate, reliability, delta) = 21 features.
#
# 대회 규칙
#   - test.csv는 header만 읽는다.
#   - Validation row의 conditional feature는 이전 시즌 train label만 사용한다.
#   - 같은 validation/test 시즌의 다른 row나 target은 절대 사용하지 않는다.
#   - Test row aggregation/rolling/frequency/distribution 사용 없음.
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

EXPERIMENT_NAME = "cb_reg_v020_pitcher_conditional_ablation"
DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
VALIDATION_DIR = Path("./validation") / EXPERIMENT_NAME
METADATA_PATH = VALIDATION_DIR / "metadata.json"
OOF_PATH = VALIDATION_DIR / "oof_predictions.csv"
COVERAGE_PATH = VALIDATION_DIR / "pitcher_conditional_coverage.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
]

EXPANDING_FOLDS = [
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {"fold": 4, "train_seasons": [2019, 2020, 2021, 2022, 2023], "validation_season": 2024},
]

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

HBEB_CONFIGS = [
    {
        "group_cols": ["pitcher_hand"],
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "new_col": "hbeb_pitcher_success_rate",
    },
    {
        "group_cols": ["batter_hand"],
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "new_col": "hbeb_batter_success_rate",
    },
]

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}
EARLY_STOPPING_ROUNDS = 80

# Reproduction guard: 이전 Exact-ASOF 실험의 pooled OOF.
EXPECTED_CONTROL_POOLED_BRIER = 0.2469124012
CONTROL_REPRO_TOL = 2e-7
EXPECTED_CONTROL_FEATURES = 52
EXPECTED_TREATMENT_FEATURES = 73

# V15 recovered notebook의 원본 context / shrink strength.
PC_CONTEXTS = [
    ("v15_count", 28.0),
    ("batter_hand", 35.0),
    ("v15_inning_phase", 35.0),
    ("base_state", 35.0),
    ("v15_pressure", 28.0),
    ("v15_runners", 35.0),
    ("v15_li_bucket", 35.0),
]
PC_PARENT_STRENGTH = 80.0
PC_FIELDS = ("rate", "reliability", "delta")
PC_FEATURES = [
    f"pc__{ctx}__{field}"
    for ctx, _ in PC_CONTEXTS
    for field in PC_FIELDS
]


def prepare_features(df, cat_cols):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return df


def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def calculate_bss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    brier = brier_score(y_true, y_pred)
    p = float(np.mean(y_true))
    baseline = p * (1.0 - p)

    if baseline <= 0.0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier / baseline
        )
    )


def build_pitcher_season_stats(df_train):
    """
    공식 train label만으로 pitcher × season 완결 통계를 만든다.

    Exact-ASOF 계산 시에는 target row의 season보다
    과거 season만 사용한다.
    """

    required = {
        "pitcher_id",
        SEASON_COL,
        TARGET,
    }

    missing = sorted(
        required - set(df_train.columns)
    )

    if missing:
        raise KeyError(
            f"pitcher season stats 필수 컬럼 없음: {missing}"
        )

    stats = (
        df_train
        .groupby(
            ["pitcher_id", SEASON_COL],
            dropna=False,
            sort=False,
        )[TARGET]
        .agg(
            n="size",
            success="sum",
        )
        .reset_index()
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats["n"] = (
        stats["n"]
        .astype(np.float64)
    )

    stats["success"] = (
        stats["success"]
        .astype(np.float64)
    )

    return stats


def add_pitcher_exact_asof_features(
    df,
    pitcher_season_stats,
):
    """
    해당 row의 official cumulative ASOF에서
    과거 완결 시즌 누적치를 빼서
    현재 시즌 시점 직전 상태를 복원한다.

    다른 validation/test row는 사용하지 않는다.
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

    diagnostics = []

    for target_year in sorted(
        seasons.unique()
    ):
        row_mask = (
            seasons == int(target_year)
        )

        rows = out.loc[row_mask]

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
                .map(hist_agg["n"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

            hist_success = (
                rows["pitcher_id"]
                .map(hist_agg["success"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
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
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64)
        )

        asof_rate = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(dtype=np.float64)
        )

        asof_success = (
            asof_n * asof_rate
        )

        raw_current_n = (
            asof_n - hist_n
        )

        current_n = np.maximum(
            raw_current_n,
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

        diagnostics.append(
            {
                "season": int(target_year),
                "n_rows": int(len(rows)),
                "negative_current_n": int(
                    (
                        raw_current_n
                        < -1e-8
                    ).sum()
                ),
                "current_rate_available": int(
                    np.isfinite(
                        current_rate
                    ).sum()
                ),
                "past_rate_available": int(
                    np.isfinite(
                        past_rate
                    ).sum()
                ),
            }
        )

    return (
        out,
        pd.DataFrame(diagnostics),
    )


def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    train subset만으로 HB-EB prior를 추정한다.
    """

    g_mu = float(
        df_train[rate_col].mean()
    )

    g_var = float(
        df_train[rate_col].var()
    )

    max_bernoulli_var = (
        g_mu * (1.0 - g_mu)
    )

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = (
            max_bernoulli_var
            * 0.5
        )

    g_m = (
        max_bernoulli_var
        / g_var
    ) - 1.0

    g_alpha = (
        g_mu * g_m
    )

    g_beta = (
        (1.0 - g_mu)
        * g_m
    )

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            ["mean", "var", "count"]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0
            - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(
            stats["var"]
        )
        & (
            stats["var"] > 0.0
        )
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"] >= 30
        )
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

    if valid_mask.any():
        mu = stats.loc[
            valid_mask,
            "mean",
        ]

        var = stats.loc[
            valid_mask,
            "var",
        ]

        m = (
            mu
            * (1.0 - mu)
            / var
        ) - 1.0

        stats.loc[
            valid_mask,
            "alpha",
        ] = (
            mu * m
        )

        stats.loc[
            valid_mask,
            "beta",
        ] = (
            (1.0 - mu)
            * m
        )

    prior_df = stats[
        group_cols
        + ["alpha", "beta"]
    ].copy()

    global_prior = (
        float(g_alpha),
        float(g_beta),
    )

    return (
        prior_df,
        global_prior,
    )


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
        .reset_index(drop=True)
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
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(dtype=np.float64)
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
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
        .to_numpy(dtype=np.float64)
    )

    success_count = (
        n * rate
    )

    denominator = (
        n + alpha + beta
    )

    return np.divide(
        success_count + alpha,
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


def apply_hbeb_pair(
    df_train,
    df_apply,
    base_features,
):
    """
    df_train만으로 prior를 만들고
    df_apply에 적용한다.
    """

    transformed = (
        df_apply.copy()
    )

    features = list(
        base_features
    )

    priors = {}

    for config in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config[
                "group_cols"
            ],
            rate_col=config[
                "rate_col"
            ],
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=config[
                "group_cols"
            ],
            n_col=config[
                "n_col"
            ],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            config["rate_col"]
            in features
        ):
            features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in features
        ):
            features.append(
                config["new_col"]
            )

        priors[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config[
                    "group_cols"
                ]
            ),
            "n_col": config[
                "n_col"
            ],
            "rate_col": config[
                "rate_col"
            ],
        }

    return (
        transformed,
        features,
        priors,
    )

# ============================================================
# V15/V18 PITCHER_CONDITIONAL — recovered notebook semantics
# ============================================================

def _ctx_frame_v15(frame):
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


def _pc_eb_table(history, keys, parent_keys=None, strength=40.0, parent_strength=80.0):
    # Recovered V15 code와 동일한 empirical-Bayes shrinkage 식.
    h = history
    if len(h) == 0:
        return pd.DataFrame(columns=keys + ["rate", "reliability", "delta", "log_n"])

    league = float(h[TARGET].mean())
    g = h.groupby(keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()

    if parent_keys is None:
        g["parent"] = league
    else:
        pg = h.groupby(parent_keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()
        pg["parent_rate"] = (
            pg["sum"] + parent_strength * league
        ) / (
            pg["size"] + parent_strength
        )
        g = g.merge(pg[parent_keys + ["parent_rate"]], on=parent_keys, how="left")
        g["parent"] = g["parent_rate"].fillna(league)

    g["rate"] = (
        g["sum"] + strength * g["parent"]
    ) / (
        g["size"] + strength
    )
    g["reliability"] = g["size"] / (g["size"] + strength)
    g["delta"] = g["rate"] - g["parent"]
    g["log_n"] = np.log1p(g["size"])
    return g[keys + ["rate", "reliability", "delta", "log_n"]]


def _pc_lookup(table, rows, keys, field, default=0.0):
    # 원본 V15의 unseen default=0.0을 그대로 유지.
    if len(table) == 0:
        return np.full(len(rows), default, dtype=np.float32)

    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)

    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)


def _pc_coverage(table, rows, keys):
    if len(rows) == 0 or len(table) == 0:
        return 0.0
    target_idx = pd.MultiIndex.from_frame(rows[keys].astype(str))
    table_idx = pd.MultiIndex.from_frame(table[keys].astype(str))
    return float(np.mean(target_idx.isin(table_idx)))


def build_pitcher_conditional_for_rows(rows, available_history, scope):
    """
    각 target season y의 PC는 오직 y-1 labeled season으로 생성한다.

    training rows:
      2020 row <- 2019 labels
      2021 row <- 2020 labels ...

    validation rows:
      2024 row <- fold-train의 2023 labels

    같은 target season label은 절대 사용하지 않는다.
    """
    required = {
        SEASON_COL, TARGET, "pitcher_id", "batter_hand", "base_state",
        "balls_before", "strikes_before", "inning", "num_runners_on", "li",
    }
    missing_rows = sorted((required - {TARGET}) - set(rows.columns))
    missing_hist = sorted(required - set(available_history.columns))
    if missing_rows:
        raise KeyError(f"PITCHER_CONDITIONAL row source missing: {missing_rows}")
    if missing_hist:
        raise KeyError(f"PITCHER_CONDITIONAL history source missing: {missing_hist}")

    r_all = _ctx_frame_v15(rows.reset_index(drop=False).rename(columns={"index": "__orig_index"}))
    h_all = _ctx_frame_v15(available_history.reset_index(drop=True))

    out = pd.DataFrame(0.0, index=np.arange(len(r_all)), columns=PC_FEATURES, dtype=np.float32)
    coverage_rows = []

    seasons = pd.to_numeric(r_all[SEASON_COL], errors="raise").astype(int)
    hist_seasons = pd.to_numeric(h_all[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(seasons.unique()):
        target_mask = seasons.eq(int(target_year)).to_numpy()
        target_pos = np.flatnonzero(target_mask)
        rr = r_all.loc[target_mask].reset_index(drop=True)
        source_year = int(target_year) - 1
        hh = h_all.loc[hist_seasons.eq(source_year)].reset_index(drop=True)

        # source season 부재(예: 2019)는 원본 lookup의 neutral 0 vector.
        if len(hh) == 0:
            for ctx, strength in PC_CONTEXTS:
                coverage_rows.append({
                    "scope": scope,
                    "target_season": int(target_year),
                    "source_season": int(source_year),
                    "context": ctx,
                    "strength": float(strength),
                    "n_rows": int(len(rr)),
                    "source_rows": 0,
                    "coverage": 0.0,
                })
            continue

        for ctx, strength in PC_CONTEXTS:
            table = _pc_eb_table(
                hh,
                ["pitcher_id", ctx],
                ["pitcher_id"],
                strength=float(strength),
                parent_strength=PC_PARENT_STRENGTH,
            )
            prefix = f"pc__{ctx}"
            keys = ["pitcher_id", ctx]
            for field in PC_FIELDS:
                out.loc[target_pos, f"{prefix}__{field}"] = _pc_lookup(
                    table, rr, keys, field, 0.0
                )

            coverage_rows.append({
                "scope": scope,
                "target_season": int(target_year),
                "source_season": int(source_year),
                "context": ctx,
                "strength": float(strength),
                "n_rows": int(len(rr)),
                "source_rows": int(len(hh)),
                "coverage": _pc_coverage(table, rr, keys),
            })

    # 원래 rows index로 복원.
    out.index = rows.index
    if list(out.columns) != PC_FEATURES:
        raise RuntimeError("PITCHER_CONDITIONAL feature order mismatch")
    if out.columns.duplicated().any():
        raise RuntimeError("PITCHER_CONDITIONAL duplicate columns")
    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("PITCHER_CONDITIONAL contains NaN/Inf")

    return out, pd.DataFrame(coverage_rows)


def fit_catboost_and_predict(train_df, val_df, features, cat_cols):
    X_train = prepare_features(train_df[features], cat_cols)
    X_val = prepare_features(val_df[features], cat_cols)
    y_train = train_df[TARGET].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET].to_numpy(dtype=np.float64)

    model = CatBoostRegressor(**MODEL_PARAMS)
    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )
    elapsed = time.time() - start
    pred = np.clip(np.asarray(model.predict(X_val), dtype=np.float64), 0.0, 1.0)
    best_iteration = int(model.get_best_iteration())
    return pred, best_iteration, float(elapsed)


def subset_metrics(y, pred, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    yy = np.asarray(y, dtype=np.float64)[mask]
    pp = np.asarray(pred, dtype=np.float64)[mask]
    return {
        "n": int(mask.sum()),
        "brier": brier_score(yy, pp),
        "bss": calculate_bss(yy, pp),
    }


def paired_summary(frame):
    y = frame["y_true"].to_numpy(np.float64)
    pc = frame["control_pred"].to_numpy(np.float64)
    pt = frame["treatment_pred"].to_numpy(np.float64)
    cb = brier_score(y, pc)
    tb = brier_score(y, pt)
    return {
        "n": int(len(frame)),
        "control_brier": cb,
        "treatment_brier": tb,
        "delta_brier_control_minus_treatment": float(cb - tb),
        "control_bss": calculate_bss(y, pc),
        "treatment_bss": calculate_bss(y, pt),
        "delta_bss_treatment_minus_control": float(calculate_bss(y, pt) - calculate_bss(y, pc)),
        "mean_abs_prediction_change": float(np.mean(np.abs(pt - pc))),
        "prediction_correlation": float(np.corrcoef(pc, pt)[0, 1]) if len(frame) > 1 else None,
    }


def run_paired_validation(train, base_features):
    fold_results = []
    oof_frames = []
    coverage_frames = []

    for cfg in EXPANDING_FOLDS:
        fold = int(cfg["fold"])
        train_seasons = list(cfg["train_seasons"])
        val_year = int(cfg["validation_season"])

        df_train = train.loc[train[SEASON_COL].isin(train_seasons)].copy()
        df_val = train.loc[train[SEASON_COL].eq(val_year)].copy()
        if len(df_train) == 0 or len(df_val) == 0:
            raise RuntimeError(f"Fold {fold}: empty train/validation")

        # Exact-ASOF — control과 treatment 공통.
        pitcher_season_stats = build_pitcher_season_stats(df_train)
        train_exact, train_exact_diag = add_pitcher_exact_asof_features(df_train, pitcher_season_stats)
        val_exact, val_exact_diag = add_pitcher_exact_asof_features(df_val, pitcher_season_stats)

        exact_features = list(base_features)
        for col in EXACT_FEATURES:
            if col not in exact_features:
                exact_features.append(col)

        # HB-EB — control과 treatment 공통.
        train_control, control_features, _ = apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )
        val_control, val_control_features, _ = apply_hbeb_pair(
            df_train=train_exact,
            df_apply=val_exact,
            base_features=exact_features,
        )
        if control_features != val_control_features:
            raise RuntimeError(f"Fold {fold}: control train/val feature order mismatch")
        if len(control_features) != EXPECTED_CONTROL_FEATURES:
            raise RuntimeError(
                f"Fold {fold}: control feature count {len(control_features)} != {EXPECTED_CONTROL_FEATURES}. "
                "canonical v019 schema/data를 확인하세요."
            )

        # Treatment only: prior-season PITCHER_CONDITIONAL 21개.
        pc_train, cov_train = build_pitcher_conditional_for_rows(
            rows=df_train,
            available_history=df_train,
            scope="train",
        )
        pc_val, cov_val = build_pitcher_conditional_for_rows(
            rows=df_val,
            available_history=df_train,
            scope="validation",
        )
        cov_train["fold"] = fold
        cov_train["validation_season"] = val_year
        cov_val["fold"] = fold
        cov_val["validation_season"] = val_year
        coverage_frames.extend([cov_train, cov_val])

        train_treatment = train_control.copy()
        val_treatment = val_control.copy()
        for col in PC_FEATURES:
            if col in train_treatment.columns or col in val_treatment.columns:
                raise RuntimeError(f"Duplicate PC feature: {col}")
            train_treatment[col] = pc_train[col]
            val_treatment[col] = pc_val[col]

        treatment_features = control_features + PC_FEATURES
        if len(treatment_features) != EXPECTED_TREATMENT_FEATURES:
            raise RuntimeError(
                f"Fold {fold}: treatment feature count {len(treatment_features)} != {EXPECTED_TREATMENT_FEATURES}"
            )
        if len(set(treatment_features)) != len(treatment_features):
            raise RuntimeError(f"Fold {fold}: duplicated treatment features")

        control_cat = [c for c in CAT_COLS if c in control_features]
        treatment_cat = [c for c in CAT_COLS if c in treatment_features]

        print("\n" + "=" * 88)
        print(f"Fold {fold}: {train_seasons} -> {val_year}")
        print(f"Control features   : {len(control_features)}")
        print(f"Treatment features : {len(treatment_features)} (+{len(PC_FEATURES)})")
        print("=" * 88)

        control_pred, control_iter, control_time = fit_catboost_and_predict(
            train_control, val_control, control_features, control_cat
        )
        treatment_pred, treatment_iter, treatment_time = fit_catboost_and_predict(
            train_treatment, val_treatment, treatment_features, treatment_cat
        )

        y_val = df_val[TARGET].to_numpy(np.float64)
        c_brier = brier_score(y_val, control_pred)
        t_brier = brier_score(y_val, treatment_pred)
        delta = c_brier - t_brier

        game_type = df_val[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
        r_mask = game_type != "F"
        f_mask = game_type == "F"

        fold_result = {
            "fold": fold,
            "train_seasons": train_seasons,
            "validation_season": val_year,
            "n_train": int(len(df_train)),
            "n_validation": int(len(df_val)),
            "control_features": len(control_features),
            "treatment_features": len(treatment_features),
            "control_brier": c_brier,
            "treatment_brier": t_brier,
            "delta_brier_control_minus_treatment": float(delta),
            "control_bss": calculate_bss(y_val, control_pred),
            "treatment_bss": calculate_bss(y_val, treatment_pred),
            "control_best_iteration": control_iter,
            "treatment_best_iteration": treatment_iter,
            "control_elapsed_seconds": control_time,
            "treatment_elapsed_seconds": treatment_time,
            "R": {
                "control": subset_metrics(y_val, control_pred, r_mask),
                "treatment": subset_metrics(y_val, treatment_pred, r_mask),
            },
            "F": {
                "control": subset_metrics(y_val, control_pred, f_mask),
                "treatment": subset_metrics(y_val, treatment_pred, f_mask),
            },
            "exact_asof_diagnostics": {
                "train": train_exact_diag.to_dict(orient="records"),
                "validation": val_exact_diag.to_dict(orient="records"),
            },
        }
        fold_results.append(fold_result)

        oof_frames.append(pd.DataFrame({
            ID: df_val[ID].to_numpy(),
            SEASON_COL: val_year,
            GAME_TYPE_COL: game_type,
            "y_true": y_val,
            "control_pred": control_pred,
            "treatment_pred": treatment_pred,
            "treatment_minus_control": treatment_pred - control_pred,
        }))

        print(
            f"RESULT | control={c_brier:.10f} | treatment={t_brier:.10f} | "
            f"delta(C-T)={delta:+.10f} | iter {control_iter}->{treatment_iter}"
        )

    oof = pd.concat(oof_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    return fold_results, oof, coverage


def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {VALIDATION_DIR}\n"
            "기존 실험을 보존하기 위해 폴더를 지우거나 EXPERIMENT_NAME을 바꾸세요."
        )
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns.tolist()

    required_train = {
        ID, TARGET, SEASON_COL, GAME_TYPE_COL, "pitcher_id", "batter_hand", "base_state",
        "balls_before", "strikes_before", "inning", "num_runners_on", "li",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_batter_n", "asof_batter_success_rate",
    }
    missing_train = sorted(required_train - set(train.columns))
    if missing_train:
        raise KeyError(f"train required columns missing: {missing_train}")

    base_features = [c for c in test_cols if c != ID]
    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError(
            "canonical v003의 asof_pitcher_pitchmix_n이 없습니다. "
            "잘못된 schema에서 실험하지 않도록 중단합니다."
        )

    # PC source가 test header에도 있어야 eventual row-local inference가 가능.
    required_pc_test = {
        SEASON_COL, "pitcher_id", "batter_hand", "base_state", "balls_before",
        "strikes_before", "inning", "num_runners_on", "li",
    }
    missing_pc_test = sorted(required_pc_test - set(test_cols))
    if missing_pc_test:
        raise KeyError(f"test header PC source missing: {missing_pc_test}")

    print("=" * 88)
    print(EXPERIMENT_NAME)
    print("Control   : cb_reg_v003 + Exact-ASOF + HB-EB")
    print("Treatment : Control + PITCHER_CONDITIONAL 21")
    print("Residual  : DISABLED")
    print("Test usage: header only")
    print("=" * 88)

    fold_results, oof, coverage = run_paired_validation(train, base_features)

    pooled = paired_summary(oof)
    recent = paired_summary(oof.loc[oof[SEASON_COL].isin([2023, 2024])])
    r_all = paired_summary(oof.loc[oof[GAME_TYPE_COL] != "F"])
    f_all = paired_summary(oof.loc[oof[GAME_TYPE_COL] == "F"])

    wins = sum(fr["delta_brier_control_minus_treatment"] > 0 for fr in fold_results)
    latest_delta = next(
        fr["delta_brier_control_minus_treatment"]
        for fr in fold_results
        if fr["validation_season"] == 2024
    )
    pooled_delta = pooled["delta_brier_control_minus_treatment"]
    recent_delta = recent["delta_brier_control_minus_treatment"]

    # Control reproduction is mandatory before Treatment interpretation.
    control_repro_error = abs(pooled["control_brier"] - EXPECTED_CONTROL_POOLED_BRIER)
    control_repro_pass = control_repro_error <= CONTROL_REPRO_TOL

    if wins == 4 and pooled_delta > 0 and recent_delta > 0:
        verdict = "STRONG_PASS"
    elif wins >= 3 and pooled_delta > 0 and latest_delta > 0:
        verdict = "PASS"
    elif pooled_delta > 0:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "purpose": "Incremental value of V15/V18 PITCHER_CONDITIONAL on Exact-ASOF CatBoost backbone",
        "control": "canonical cb_reg_v003 + Exact-ASOF 5 + HB-EB; no residual correction",
        "treatment": "control + PITCHER_CONDITIONAL 21",
        "residual_correction_enabled": False,
        "test_rows_used_for_feature_generation": False,
        "test_usage": "header only",
        "pitcher_conditional": {
            "source_policy": "exactly previous labeled season only",
            "parent": "pitcher_id previous-season baseline",
            "parent_strength": PC_PARENT_STRENGTH,
            "contexts": [{"context": c, "strength": s} for c, s in PC_CONTEXTS],
            "fields": list(PC_FIELDS),
            "feature_count": len(PC_FEATURES),
            "features": PC_FEATURES,
            "unseen_lookup_default": 0.0,
            "note": "Recovered V15/V18 notebook semantics; direct CatBoost use is this experiment's adaptation.",
        },
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "fold_results": fold_results,
        "pooled": pooled,
        "recent_2023_2024": recent,
        "R_pooled": r_all,
        "F_pooled": f_all,
        "fold_wins": int(wins),
        "latest_2024_delta": float(latest_delta),
        "verdict": verdict,
        "control_reproduction": {
            "expected_pooled_brier": EXPECTED_CONTROL_POOLED_BRIER,
            "actual_pooled_brier": pooled["control_brier"],
            "abs_error": float(control_repro_error),
            "tolerance": CONTROL_REPRO_TOL,
            "pass": bool(control_repro_pass),
        },
    }

    oof.to_csv(OOF_PATH, index=False, encoding="utf-8-sig")
    coverage.to_csv(COVERAGE_PATH, index=False, encoding="utf-8-sig")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 88)
    print("FINAL PAIRED ABLATION")
    print("=" * 88)
    for fr in fold_results:
        print(
            f"{fr['validation_season']} | "
            f"C={fr['control_brier']:.10f} | T={fr['treatment_brier']:.10f} | "
            f"delta(C-T)={fr['delta_brier_control_minus_treatment']:+.10f}"
        )
    print("-" * 88)
    print(f"Pooled C          : {pooled['control_brier']:.10f}")
    print(f"Pooled T          : {pooled['treatment_brier']:.10f}")
    print(f"Pooled delta C-T  : {pooled_delta:+.10f}")
    print(f"Recent 23-24 delta: {recent_delta:+.10f}")
    print(f"R pooled delta    : {r_all['delta_brier_control_minus_treatment']:+.10f}")
    print(f"F pooled delta    : {f_all['delta_brier_control_minus_treatment']:+.10f}")
    print(f"Fold wins         : {wins}/4")
    print(f"Verdict           : {verdict}")
    print(
        f"Control reproduce : {'PASS' if control_repro_pass else 'FAIL'} | "
        f"expected={EXPECTED_CONTROL_POOLED_BRIER:.10f} actual={pooled['control_brier']:.10f} "
        f"abs_error={control_repro_error:.3e}"
    )
    print(f"OOF               : {OOF_PATH}")
    print(f"Coverage          : {COVERAGE_PATH}")
    print(f"Metadata          : {METADATA_PATH}")
    print("=" * 88)

    if not control_repro_pass:
        raise RuntimeError(
            "Control reproduction failed. Treatment 결과를 해석하지 마세요. "
            "v019 Exact-ASOF baseline/schema 차이를 먼저 확인해야 합니다."
        )


if __name__ == "__main__":
    main()

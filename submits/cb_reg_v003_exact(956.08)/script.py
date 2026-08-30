#script.py

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPERIMENT_NAME = "cb_reg_v003_exact"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

MODEL_PATH = (
    Path("./model")
    / f"{EXPERIMENT_NAME}.pkl"
)

OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

EXACT_COLS = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


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


def add_pitcher_exact_asof_features_test(
    test,
    pitcher_season_stats,
):
    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required - set(test.columns)
    )
    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 test 컬럼 없음: {missing}"
        )

    out = test.copy()

    for col in EXACT_COLS:
        out[col] = np.nan

    test_season = pd.to_numeric(
        out[SEASON_COL],
        errors="raise",
    ).astype(int)

    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL],
        errors="raise",
    ).astype(int)

    # unique season 값만 순회할 뿐,
    # 같은 test season의 다른 row 값을 집계하지 않는다.
    for target_year in sorted(
        test_season.unique()
    ):
        mask = test_season == int(target_year)
        rows = out.loc[mask]

        # 오직 artifact에 저장된 train 완료 시즌 중
        # target_year보다 과거인 기록만 사용.
        hist = stats.loc[
            stats[SEASON_COL] < int(target_year)
        ]

        if len(hist):
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
                rows["asof_pitcher_success_rate"],
                errors="coerce",
            )
            .to_numpy(dtype=np.float64)
        )

        asof_success = asof_n * asof_rate

        current_n = np.maximum(
            asof_n - hist_n,
            0.0,
        )
        current_success = (
            asof_success - hist_success
        )

        current_success = np.where(
            np.isfinite(current_success),
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
            & np.isfinite(current_success)
        )

        current_rate[valid_current] = (
            current_success[valid_current]
            / current_n[valid_current]
        )

        past_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )
        valid_past = hist_n > 0.0

        past_rate[valid_past] = (
            hist_success[valid_past]
            / hist_n[valid_past]
        )

        idx = rows.index

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = current_n
        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = np.log1p(current_n)
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
        ] = current_rate - past_rate

    return out


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
        .fillna(global_prior[0])
        .to_numpy(dtype=np.float64)
    )
    beta = (
        lookup["beta"]
        .fillna(global_prior[1])
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
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.float64)
    )

    denominator = n + alpha + beta

    return np.divide(
        n * rate + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=denominator > 0.0,
    )


def build_submission(
    sample_submission,
    test_ids,
    predictions,
):
    pred_df = pd.DataFrame({
        ID: test_ids,
        TARGET: predictions,
    })

    if pred_df[ID].duplicated().any():
        raise ValueError(
            "prediction row_id 중복"
        )

    submission = (
        sample_submission[[ID]]
        .merge(
            pred_df,
            on=ID,
            how="left",
            validate="one_to_one",
        )
    )

    if submission[TARGET].isna().any():
        raise ValueError(
            "submission prediction NaN"
        )

    pred = submission[TARGET].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            "submission prediction NaN/Inf"
        )

    if not (
        (pred >= 0.0)
        & (pred <= 1.0)
    ).all():
        raise ValueError(
            "prediction outside [0,1]"
        )

    return submission


def main():
    started = time.time()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Artifact 없음: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if (
        artifact.get("experiment_name")
        != EXPERIMENT_NAME
    ):
        raise ValueError(
            "Artifact experiment_name 불일치"
        )

    model = artifact["model"]
    final_features = list(
        artifact["features"]
    )
    cat_cols = list(
        artifact["cat_cols"]
    )
    hbeb_priors = artifact[
        "hbeb_priors"
    ]
    pitcher_season_stats = artifact[
        "pitcher_season_stats"
    ]

    if "asof_pitcher_pitchmix_n" not in final_features:
        raise RuntimeError(
            "canonical v003의 asof_pitcher_pitchmix_n이 "
            "artifact에서 누락되었습니다."
        )

    for col in EXACT_COLS:
        if col not in final_features:
            raise RuntimeError(
                f"Exact-ASOF feature 누락: {col}"
            )

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )
    sample_sub = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    test_ids = test[ID].to_numpy()

    # 1) Exact-ASOF
    test = add_pitcher_exact_asof_features_test(
        test=test,
        pitcher_season_stats=pitcher_season_stats,
    )

    # 2) HB-EB
    for new_col, info in hbeb_priors.items():
        test[new_col] = apply_hbeb(
            df=test,
            group_cols=info["group_cols"],
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=info["global_prior"],
        )

    # 3) Feature schema check
    missing = [
        c for c in final_features
        if c not in test.columns
    ]
    if missing:
        raise KeyError(
            f"test 최종 feature 누락: {missing}"
        )

    X_test = prepare_features(
        test[final_features],
        cat_cols,
    )

    if hasattr(model, "feature_names_"):
        if (
            list(model.feature_names_)
            != list(X_test.columns)
        ):
            raise ValueError(
                "학습/추론 feature 순서 불일치"
            )

    # 4) Predict
    pred_raw = np.asarray(
        model.predict(X_test),
        dtype=np.float64,
    )

    if not np.isfinite(pred_raw).all():
        raise ValueError(
            "raw prediction NaN/Inf"
        )

    pred = np.clip(
        pred_raw,
        0.0,
        1.0,
    )

    # 5) Submission
    submission = build_submission(
        sample_submission=sample_sub,
        test_ids=test_ids,
        predictions=pred,
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

    print("=" * 80)
    print(f"Inference: {EXPERIMENT_NAME}")
    print(f"rows     : {len(test):,}")
    print(f"features : {len(final_features)}")
    print(f"pred min : {pred.min():.6f}")
    print(f"pred mean: {pred.mean():.6f}")
    print(f"pred max : {pred.max():.6f}")
    print(f"output   : {OUTPUT_PATH}")
    print(f"time     : {time.time() - started:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

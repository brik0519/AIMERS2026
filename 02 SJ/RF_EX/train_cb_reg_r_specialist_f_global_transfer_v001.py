# train_cb_reg_r_specialist_f_global_transfer_v001.py
# ============================================================
# R Specialist / F Global Temporal Transfer Check
#
# 목적
# ----
# 이전 transfer 실험에서:
# - R specialist는 2023 -> 2024로 강하게 transfer
# - F specialist는 2023 -> 2024에서 악화
#
# 따라서 이번 실험은:
#   R row만 R specialist 사용
#   F row는 Global model을 그대로 유지
#
# 비교
# ----
# Control:
#   모든 row -> Global model (iterations=3)
#
# Treatment:
#   R row -> R specialist (iterations=121)
#   F row -> Global model (iterations=3)
#
# 모든 iteration은 2023에서 이미 선택된 값을 고정한다.
# 2024는 Early Stopping/iteration 선택에 절대 사용하지 않는다.
#
# 유지 사항
# --------
# - CatBoostRegressor / RMSE
# - HB-EB
# - asof_pitcher_pitchmix_n 제외
# - seed / hyperparameter 동일
# - 2019~2023 train -> 2024 untouched validation
#
# 대회 규칙
# --------
# - test.csv는 header만 읽는다.
# - HB-EB prior는 train에서만 생성한다.
# - routing은 각 row 자신의 game_type만 사용한다.
# - 다른 test row / test 분포 / rolling / lag / group aggregate 사용 없음.
# ============================================================

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Feature 제외
# ============================================================

EXCLUDE_FEATURES = [
    "asof_pitcher_pitchmix_n",
]


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "cb_reg_r_specialist_f_global_transfer_v001"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME


# ============================================================
# 기본 설정
# ============================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

TRAIN_SEASONS = [
    2019,
    2020,
    2021,
    2022,
    2023,
]

VALIDATION_SEASON = 2024

R_GROUP = "R"
F_GROUP = "F"

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
]


# ============================================================
# 2023에서 이미 선택된 iteration
# ============================================================

GLOBAL_ITERATIONS = 3
R_SPECIALIST_ITERATIONS = 121


# ============================================================
# HB-EB
# ============================================================

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


# ============================================================
# CatBoost
# ============================================================

BASE_MODEL_PARAMS = {
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


# ============================================================
# JSON helper
# ============================================================

def to_python(obj):
    if isinstance(obj, dict):
        return {
            str(k): to_python(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            to_python(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    rate = pd.to_numeric(
        df_train[rate_col],
        errors="coerce",
    )

    g_mu = float(
        rate.mean()
    )

    g_var = float(
        rate.var()
    )

    max_bernoulli_var = (
        g_mu
        * (
            1.0 - g_mu
        )
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

    temp = (
        df_train[
            group_cols + [rate_col]
        ]
        .copy()
    )

    temp[rate_col] = pd.to_numeric(
        temp[rate_col],
        errors="coerce",
    )

    stats = (
        temp.groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            [
                "mean",
                "var",
                "count",
            ]
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

    stats["alpha"] = (
        g_alpha
    )

    stats["beta"] = (
        g_beta
    )

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
            * (
                1.0 - mu
            )
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
        + [
            "alpha",
            "beta",
        ]
    ].copy()

    return (
        prior_df,
        (
            float(g_alpha),
            float(g_beta),
        ),
    )


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    original_index = (
        df.index
    )

    lookup_df = (
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
        lookup_df["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup_df["beta"]
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

    eb_rate = np.divide(
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

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def fit_hbeb_priors(
    df_train,
):
    priors = {}

    for config in (
        HBEB_CONFIGS
    ):
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=(
                config[
                    "group_cols"
                ]
            ),
            rate_col=(
                config[
                    "rate_col"
                ]
            ),
        )

        priors[
            config["new_col"]
        ] = {
            "prior_df": (
                prior_df
            ),
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config[
                    "group_cols"
                ]
            ),
            "n_col": (
                config["n_col"]
            ),
            "rate_col": (
                config[
                    "rate_col"
                ]
            ),
        }

    return priors


def apply_hbeb_with_priors(
    df,
    base_features,
    priors,
):
    transformed = (
        df.copy()
    )

    current_features = (
        list(
            base_features
        )
    )

    for config in (
        HBEB_CONFIGS
    ):
        info = (
            priors[
                config["new_col"]
            ]
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=(
                config[
                    "group_cols"
                ]
            ),
            n_col=(
                config["n_col"]
            ),
            rate_col=(
                config[
                    "rate_col"
                ]
            ),
            prior_df=(
                info["prior_df"]
            ),
            global_prior=(
                info[
                    "global_prior"
                ]
            ),
        )

        if (
            config["rate_col"]
            in current_features
        ):
            current_features.remove(
                config[
                    "rate_col"
                ]
            )

        if (
            config["new_col"]
            not in current_features
        ):
            current_features.append(
                config[
                    "new_col"
                ]
            )

    return (
        transformed,
        current_features,
    )


# ============================================================
# Feature
# ============================================================

def prepare_features(
    df,
    cat_cols,
):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )

    return df


# ============================================================
# Metric
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    positive_rate = float(
        np.mean(
            y_true
        )
    )

    brier = float(
        np.mean(
            (
                y_pred - y_true
            ) ** 2
        )
    )

    baseline_brier = float(
        positive_rate
        * (
            1.0
            - positive_rate
        )
    )

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0
            * (
                1.0
                - brier
                / baseline_brier
            )
        )

        bss = float(
            max(
                0.0,
                bss_raw,
            )
        )

    return {
        "n": int(
            len(y_true)
        ),
        "positive_rate": (
            positive_rate
        ),
        "brier": (
            brier
        ),
        "baseline_brier": (
            baseline_brier
        ),
        "bss_raw": (
            bss_raw
        ),
        "bss": (
            bss
        ),
    }


def group_metrics(
    df,
    prediction,
    group_value,
):
    mask = (
        df[
            GROUP_COL
        ]
        .astype(str)
        .eq(group_value)
        .to_numpy()
    )

    if mask.sum() == 0:
        raise ValueError(
            f"Validation에 {group_value} row가 없습니다."
        )

    return calculate_metrics(
        df.loc[
            mask,
            TARGET,
        ].to_numpy(
            dtype=np.float64
        ),
        np.asarray(
            prediction,
            dtype=np.float64,
        )[mask],
    )


# ============================================================
# Fixed model
# ============================================================

def fit_fixed_model(
    X_train,
    y_train,
    cat_cols,
    iterations,
    name,
):
    params = {
        **BASE_MODEL_PARAMS,
        "iterations": int(
            iterations
        ),
    }

    model = CatBoostRegressor(
        **params
    )

    print()
    print(
        f"[{name}] "
        f"iterations={iterations}"
    )

    start = (
        time.time()
    )

    model.fit(
        X_train,
        y_train,
        cat_features=(
            cat_cols
        ),
    )

    elapsed = float(
        time.time()
        - start
    )

    return (
        model,
        elapsed,
    )


# ============================================================
# R specialist routing
# ============================================================

def r_specialist_f_global_prediction(
    global_model,
    r_model,
    X,
    group_series,
):
    """
    R row만 R specialist를 사용.
    F row는 Global prediction을 그대로 사용.
    각 row 자신의 game_type만 사용한다.
    """
    groups = (
        group_series
        .astype(str)
        .to_numpy()
    )

    r_mask = (
        groups == R_GROUP
    )

    f_mask = (
        groups == F_GROUP
    )

    if not (
        r_mask
        | f_mask
    ).all():
        bad = sorted(
            set(
                groups[
                    ~(
                        r_mask
                        | f_mask
                    )
                ]
            )
        )

        raise ValueError(
            f"예상하지 못한 game_type: {bad}"
        )

    # 기본은 Global prediction
    pred = np.asarray(
        global_model.predict(
            X
        ),
        dtype=np.float64,
    )

    # R row만 specialist로 교체
    if r_mask.any():
        pred[
            r_mask
        ] = np.asarray(
            r_model.predict(
                X.loc[
                    r_mask
                ]
            ),
            dtype=np.float64,
        )

    return pred


# ============================================================
# Main
# ============================================================

def main():

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    base_features = [
        c
        for c in test_cols
        if (
            c != ID
            and c not in EXCLUDE_FEATURES
        )
    ]

    if GROUP_COL not in base_features:
        raise ValueError(
            f"{GROUP_COL}이 feature에 없습니다."
        )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in base_features
    ]

    train = pd.read_csv(
        DATA_DIR
        / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + base_features
            + [TARGET]
        ),
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_mask = (
        train[
            SEASON_COL
        ].isin(
            TRAIN_SEASONS
        )
    )

    valid_mask = (
        train[
            SEASON_COL
        ]
        == VALIDATION_SEASON
    )

    df_train_raw = (
        train.loc[
            train_mask
        ]
        .copy()
    )

    df_valid_raw = (
        train.loc[
            valid_mask
        ]
        .copy()
    )

    if (
        len(df_train_raw) == 0
        or len(df_valid_raw) == 0
    ):
        raise ValueError(
            "Train/Validation split이 비었습니다."
        )

    train_groups = set(
        df_train_raw[
            GROUP_COL
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    valid_groups = set(
        df_valid_raw[
            GROUP_COL
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    expected = {
        R_GROUP,
        F_GROUP,
    }

    if train_groups != expected:
        raise ValueError(
            f"Train game_type 이상: {train_groups}"
        )

    if valid_groups != expected:
        raise ValueError(
            f"Validation game_type 이상: {valid_groups}"
        )

    # --------------------------------------------------------
    # HB-EB: 2019~2023 only
    # --------------------------------------------------------

    priors = (
        fit_hbeb_priors(
            df_train_raw
        )
    )

    (
        df_train,
        final_features,
    ) = apply_hbeb_with_priors(
        df=df_train_raw,
        base_features=base_features,
        priors=priors,
    )

    (
        df_valid,
        valid_features,
    ) = apply_hbeb_with_priors(
        df=df_valid_raw,
        base_features=base_features,
        priors=priors,
    )

    if (
        final_features
        != valid_features
    ):
        raise ValueError(
            "Train/Validation feature 불일치"
        )

    X_train = (
        prepare_features(
            df_train[
                final_features
            ],
            cat_cols,
        )
    )

    y_train = (
        df_train[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    X_valid = (
        prepare_features(
            df_valid[
                final_features
            ],
            cat_cols,
        )
    )

    y_valid = (
        df_valid[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    print("=" * 90)
    print(
        f"Experiment       : {EXPERIMENT_NAME}"
    )
    print(
        "Control          : "
        f"Global {GLOBAL_ITERATIONS} trees"
    )
    print(
        "Treatment        : "
        f"R->{R_SPECIALIST_ITERATIONS} trees, "
        f"F->Global {GLOBAL_ITERATIONS} trees"
    )
    print(
        f"Train            : {TRAIN_SEASONS[0]}~{TRAIN_SEASONS[-1]}"
    )
    print(
        f"Validation       : {VALIDATION_SEASON}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    (
        global_model,
        global_elapsed,
    ) = fit_fixed_model(
        X_train=X_train,
        y_train=y_train,
        cat_cols=cat_cols,
        iterations=(
            GLOBAL_ITERATIONS
        ),
        name="Global",
    )

    control_raw = np.asarray(
        global_model.predict(
            X_valid
        ),
        dtype=np.float64,
    )

    control_pred = np.clip(
        control_raw,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # R specialist
    # --------------------------------------------------------

    (
        r_model,
        r_elapsed,
    ) = fit_fixed_model(
        X_train=X_train,
        y_train=y_train,
        cat_cols=cat_cols,
        iterations=(
            R_SPECIALIST_ITERATIONS
        ),
        name="R Specialist",
    )

    treatment_raw = (
        r_specialist_f_global_prediction(
            global_model=(
                global_model
            ),
            r_model=(
                r_model
            ),
            X=X_valid,
            group_series=(
                df_valid[
                    GROUP_COL
                ]
            ),
        )
    )

    treatment_pred = np.clip(
        treatment_raw,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    control_all = (
        calculate_metrics(
            y_valid,
            control_pred,
        )
    )

    treatment_all = (
        calculate_metrics(
            y_valid,
            treatment_pred,
        )
    )

    control_r = (
        group_metrics(
            df_valid,
            control_pred,
            R_GROUP,
        )
    )

    treatment_r = (
        group_metrics(
            df_valid,
            treatment_pred,
            R_GROUP,
        )
    )

    control_f = (
        group_metrics(
            df_valid,
            control_pred,
            F_GROUP,
        )
    )

    treatment_f = (
        group_metrics(
            df_valid,
            treatment_pred,
            F_GROUP,
        )
    )

    result = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "train_seasons": (
            TRAIN_SEASONS
        ),
        "validation_season": (
            VALIDATION_SEASON
        ),
        "policy": (
            "R_specialist_F_global"
        ),
        "fixed_iterations": {
            "global": (
                GLOBAL_ITERATIONS
            ),
            "R_specialist": (
                R_SPECIALIST_ITERATIONS
            ),
        },
        "control": {
            "all": (
                control_all
            ),
            "R": (
                control_r
            ),
            "F": (
                control_f
            ),
        },
        "treatment": {
            "all": (
                treatment_all
            ),
            "R": (
                treatment_r
            ),
            "F": (
                treatment_f
            ),
        },
        "gain": {
            "overall_brier": float(
                control_all[
                    "brier"
                ]
                - treatment_all[
                    "brier"
                ]
            ),
            "R_brier": float(
                control_r[
                    "brier"
                ]
                - treatment_r[
                    "brier"
                ]
            ),
            "F_brier": float(
                control_f[
                    "brier"
                ]
                - treatment_f[
                    "brier"
                ]
            ),
            "overall_bss": float(
                treatment_all[
                    "bss"
                ]
                - control_all[
                    "bss"
                ]
            ),
        },
        "elapsed_seconds": {
            "global": (
                global_elapsed
            ),
            "R_specialist": (
                r_elapsed
            ),
        },
        "features": (
            final_features
        ),
        "categorical_features": (
            cat_cols
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "exclude_features": (
            EXCLUDE_FEATURES
        ),
        "base_model_params": (
            BASE_MODEL_PARAMS
        ),
    }

    # --------------------------------------------------------
    # Sanity:
    # F는 반드시 Control과 완전히 같아야 한다.
    # --------------------------------------------------------

    f_mask = (
        df_valid[
            GROUP_COL
        ]
        .astype(str)
        .eq(
            F_GROUP
        )
        .to_numpy()
    )

    if not np.allclose(
        control_pred[
            f_mask
        ],
        treatment_pred[
            f_mask
        ],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "F prediction이 Global과 다릅니다. "
            "Treatment는 F를 변경하면 안 됩니다."
        )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    pred_df = pd.DataFrame({
        ID: (
            df_valid[
                ID
            ].to_numpy()
        ),
        "target": (
            y_valid.to_numpy()
        ),
        "game_type": (
            df_valid[
                GROUP_COL
            ]
            .astype(str)
            .to_numpy()
        ),
        "control_global_prediction": (
            control_pred
        ),
        "treatment_r_specialist_f_global_prediction": (
            treatment_pred
        ),
        "validation_season": (
            VALIDATION_SEASON
        ),
    })

    pred_df.to_csv(
        VALIDATION_DIR
        / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(
                result
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    shutil.copy2(
        Path(
            __file__
        ).resolve(),
        MODEL_DIR
        / "train.py",
    )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "R Specialist / F Global Transfer Result"
    )
    print("=" * 90)

    print(
        "Control Global Brier : "
        f"{control_all['brier']:.10f}"
    )

    print(
        "Treatment Brier      : "
        f"{treatment_all['brier']:.10f}"
    )

    print(
        "Overall Brier Gain   : "
        f"{result['gain']['overall_brier']:+.10f}"
    )

    print(
        "Overall BSS Gain     : "
        f"{result['gain']['overall_bss']:+.4f}"
    )

    print()
    print(
        "R Control -> Treatment : "
        f"{control_r['brier']:.10f} "
        f"-> "
        f"{treatment_r['brier']:.10f}"
    )

    print(
        "R Brier Gain           : "
        f"{result['gain']['R_brier']:+.10f}"
    )

    print()
    print(
        "F Control -> Treatment : "
        f"{control_f['brier']:.10f} "
        f"-> "
        f"{treatment_f['brier']:.10f}"
    )

    print(
        "F Brier Gain           : "
        f"{result['gain']['F_brier']:+.10f}"
    )

    print()
    print(
        "※ F Gain은 정확히 0이어야 정상입니다."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()

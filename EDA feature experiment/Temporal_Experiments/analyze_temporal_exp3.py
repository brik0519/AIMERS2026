# analyze_temporal_exp3.py
"""
LG Aimers Phase2 - CatBoost Basic vs EXP3 Temporal Diagnostic

목적
----
1) Basic과 EXP3의 Expanding Window OOF Brier를 같은 row 기준으로 비교
2) CatBoost native SHAP으로 모델 내부 Feature 활용도를 비교
3) Basic vs EXP3의 row별 squared error 차이를 이용해 Residual Analysis 수행
4) 팀 EDA와 연결하여 Temporal Feature 값 구간별 실제 오차 개선/악화를 확인

주의
----
- SHAP은 Feature의 predictive usefulness를 직접 증명하지 않는다.
- SHAP은 '모델이 Feature를 얼마나 활용했는가'를 보는 진단 지표다.
- 실제 성능 차이는 OOF Brier / row별 squared error로 판단한다.
- SHAP 계산은 속도를 위해 각 Fold Validation에서 고정 seed로 일부 row를 샘플링한다.
- 외부 shap 패키지를 사용하지 않고 CatBoost의 native ShapValues를 사용한다.
"""

import gc
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_v001_ctr1_expanding_exp3"
BASIC_EXPERIMENT_NAME = "cb_v001_ctr1_expanding"
ANALYSIS_NAME = "shap_residual_v001"

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

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

VALID_SEASONS = [2021, 2022, 2023, 2024]

# SHAP은 진단용이므로 전체 Validation을 모두 계산하지 않고 고정 표본 사용.
# None으로 바꾸면 각 Fold 전체 Validation을 사용한다.
SHAP_SAMPLE_SIZE = 10_000
SHAP_RANDOM_SEED = 42


# =========================================================
# Temporal Feature
# =========================================================

TEMPORAL_FEATURES = [
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
    "success_trend_1_3",
    "success_trend_3_5",
    "middle_trend_1_3",
    "middle_trend_3_5",
]

TEMPORAL_GROUPS = {
    # 네 개의 기본 해석 그룹
    "success_level": [
        "recent1_success_vs_career",
        "recent3_success_vs_career",
        "recent5_success_vs_career",
    ],
    "middle_level": [
        "recent1_middle_vs_career",
        "recent3_middle_vs_career",
        "recent5_middle_vs_career",
    ],
    "success_trend": [
        "success_trend_1_3",
        "success_trend_3_5",
    ],
    "middle_trend": [
        "middle_trend_1_3",
        "middle_trend_3_5",
    ],

    # EDA와 연결해서 보기 위한 상위 묶음
    "success_all": [
        "recent1_success_vs_career",
        "recent3_success_vs_career",
        "recent5_success_vs_career",
        "success_trend_1_3",
        "success_trend_3_5",
    ],
    "middle_all": [
        "recent1_middle_vs_career",
        "recent3_middle_vs_career",
        "recent5_middle_vs_career",
        "middle_trend_1_3",
        "middle_trend_3_5",
    ],
    "level_all": [
        "recent1_success_vs_career",
        "recent3_success_vs_career",
        "recent5_success_vs_career",
        "recent1_middle_vs_career",
        "recent3_middle_vs_career",
        "recent5_middle_vs_career",
    ],
    "trend_all": [
        "success_trend_1_3",
        "success_trend_3_5",
        "middle_trend_1_3",
        "middle_trend_3_5",
    ],
}

SOURCE_ASOF_FEATURES = [
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]


def add_temporal_features(df):
    """기존 EXP3과 동일한 Temporal Feature 10개를 생성한다."""
    df = df.copy()

    df["recent1_success_vs_career"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )
    df["recent3_success_vs_career"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )
    df["recent5_success_vs_career"] = (
        df["asof_pitcher_prev5_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )
    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )
    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )
    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )
    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


def prepare_features(df, cat_cols):
    """학습 코드와 동일하게 범주형 결측값/타입만 처리한다."""
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


# =========================================================
# OOF / Residual
# =========================================================

def load_and_compare_oof(analysis_dir):
    """
    Basic과 EXP3 OOF를 row_id/season 기준으로 정확히 맞춘 뒤
    Fold/전체 Brier와 row별 squared error 차이를 저장한다.
    """
    basic_validation_dir = (
        VALIDATION_ROOT
        / BASIC_EXPERIMENT_NAME
    )

    basic_path = (
        basic_validation_dir
        / "oof_pred.csv"
    )
    exp3_path = (
        VALIDATION_DIR
        / "oof_pred.csv"
    )

    if not basic_path.exists():
        raise FileNotFoundError(
            f"Basic OOF가 없습니다: {basic_path}"
        )

    if not exp3_path.exists():
        raise FileNotFoundError(
            f"EXP3 OOF가 없습니다: {exp3_path}\n"
            "먼저 python train_expanding_exp3.py 를 실행하세요."
        )

    basic = pd.read_csv(
        basic_path,
        encoding="utf-8-sig",
    ).rename(
        columns={
            "prediction": "prediction_basic",
            "target": "target_basic",
            "fold": "fold_basic",
        }
    )

    exp3 = pd.read_csv(
        exp3_path,
        encoding="utf-8-sig",
    ).rename(
        columns={
            "prediction": "prediction_exp3",
            "target": "target_exp3",
            "fold": "fold_exp3",
        }
    )

    key_cols = [ID_COL, "season"]

    merged = basic.merge(
        exp3,
        on=key_cols,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        bad = merged.loc[
            merged["_merge"] != "both",
            key_cols + ["_merge"],
        ]
        raise AssertionError(
            "Basic/EXP3 OOF row 구성이 다릅니다.\n"
            + bad.head(10).to_string(index=False)
        )

    merged = merged.drop(columns="_merge")

    if not np.array_equal(
        merged["target_basic"].to_numpy(),
        merged["target_exp3"].to_numpy(),
    ):
        raise AssertionError(
            "Basic과 EXP3 OOF의 target이 일치하지 않습니다."
        )

    merged[TARGET_COL] = merged["target_basic"]
    merged["basic_squared_error"] = (
        merged["prediction_basic"]
        - merged[TARGET_COL]
    ) ** 2
    merged["exp3_squared_error"] = (
        merged["prediction_exp3"]
        - merged[TARGET_COL]
    ) ** 2

    # 양수 = EXP3 개선, 음수 = EXP3 악화
    merged["delta_squared_error"] = (
        merged["basic_squared_error"]
        - merged["exp3_squared_error"]
    )

    merged["prediction_shift"] = (
        merged["prediction_exp3"]
        - merged["prediction_basic"]
    )
    merged["abs_prediction_shift"] = (
        merged["prediction_shift"].abs()
    )

    comparison_rows = []

    for season in VALID_SEASONS:
        part = merged.loc[
            merged["season"] == season
        ]

        if part.empty:
            raise ValueError(
                f"OOF에 {season} Validation 데이터가 없습니다."
            )

        basic_brier = float(
            part["basic_squared_error"].mean()
        )
        exp3_brier = float(
            part["exp3_squared_error"].mean()
        )

        comparison_rows.append(
            {
                "scope": str(season),
                "n": int(len(part)),
                "basic_brier": basic_brier,
                "exp3_brier": exp3_brier,
                # 양수 = EXP3 개선
                "delta_brier_basic_minus_exp3": (
                    basic_brier - exp3_brier
                ),
                "mean_abs_prediction_shift": float(
                    part["abs_prediction_shift"].mean()
                ),
            }
        )

    basic_brier = float(
        merged["basic_squared_error"].mean()
    )
    exp3_brier = float(
        merged["exp3_squared_error"].mean()
    )

    comparison_rows.append(
        {
            "scope": "pooled_oof",
            "n": int(len(merged)),
            "basic_brier": basic_brier,
            "exp3_brier": exp3_brier,
            "delta_brier_basic_minus_exp3": (
                basic_brier - exp3_brier
            ),
            "mean_abs_prediction_shift": float(
                merged["abs_prediction_shift"].mean()
            ),
        }
    )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        analysis_dir / "brier_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged.to_csv(
        analysis_dir / "oof_residual_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return merged, comparison


# =========================================================
# Residual + Temporal EDA 연결
# =========================================================

def analyze_temporal_residuals(
    residual_df,
    train_with_temporal,
    analysis_dir,
):
    """
    각 Temporal Feature를 5분위로 나누어,
    해당 구간에서 Basic/EXP3 중 누가 더 낮은 Brier를 보이는지 확인한다.

    이 결과는 팀 EDA의 'Feature 값 구간별 target rate 차이'를
    실제 모델 error 차이와 연결하는 진단이다.
    """
    feature_frame = train_with_temporal[
        [
            ID_COL,
            "season",
            TARGET_COL,
            *TEMPORAL_FEATURES,
            "balls_before",
            "strikes_before",
        ]
    ].copy()

    merged = residual_df.merge(
        feature_frame,
        on=[ID_COL, "season"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_train"),
    )

    if merged[TEMPORAL_FEATURES].isna().all(axis=1).any():
        # 모든 Temporal이 NaN인 row가 존재할 수는 있으므로 오류로 보지는 않는다.
        # 단순히 전체 merge 실패 여부는 row_id로 확인한다.
        pass

    rows = []

    for feature in TEMPORAL_FEATURES:
        valid = merged.loc[
            merged[feature].notna()
        ].copy()

        if valid.empty:
            continue

        if valid[feature].nunique(dropna=True) < 2:
            continue

        try:
            valid["feature_bin"] = pd.qcut(
                valid[feature],
                q=5,
                duplicates="drop",
            )
        except ValueError:
            continue

        grouped = (
            valid.groupby(
                ["season", "feature_bin"],
                observed=True,
            )
            .agg(
                n=(ID_COL, "size"),
                feature_mean=(feature, "mean"),
                target_rate=(TARGET_COL, "mean"),
                basic_brier=(
                    "basic_squared_error",
                    "mean",
                ),
                exp3_brier=(
                    "exp3_squared_error",
                    "mean",
                ),
                mean_delta_squared_error=(
                    "delta_squared_error",
                    "mean",
                ),
                mean_abs_prediction_shift=(
                    "abs_prediction_shift",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped["feature"] = feature
        grouped["feature_bin"] = (
            grouped["feature_bin"]
            .astype(str)
        )
        grouped[
            "delta_brier_basic_minus_exp3"
        ] = (
            grouped["basic_brier"]
            - grouped["exp3_brier"]
        )

        rows.append(grouped)

    if rows:
        result = pd.concat(
            rows,
            ignore_index=True,
        )
    else:
        result = pd.DataFrame()

    result.to_csv(
        analysis_dir / "residual_by_temporal_bin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 팀의 경기상황 EDA와 연결해서 볼 수 있도록 count별 residual도 저장
    count_df = merged.copy()

    count_df["count_state"] = (
        count_df["balls_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
        + "_"
        + count_df["strikes_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    count_summary = (
        count_df.groupby(
            ["season", "count_state"],
            observed=True,
        )
        .agg(
            n=(ID_COL, "size"),
            target_rate=(TARGET_COL, "mean"),
            basic_brier=(
                "basic_squared_error",
                "mean",
            ),
            exp3_brier=(
                "exp3_squared_error",
                "mean",
            ),
            mean_delta_squared_error=(
                "delta_squared_error",
                "mean",
            ),
        )
        .reset_index()
    )

    count_summary[
        "delta_brier_basic_minus_exp3"
    ] = (
        count_summary["basic_brier"]
        - count_summary["exp3_brier"]
    )

    count_summary.to_csv(
        analysis_dir / "residual_by_count.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result, count_summary


# =========================================================
# SHAP
# =========================================================

def select_shap_sample(validation_df, valid_season):
    """
    같은 Validation row를 Basic/EXP3 양쪽에 사용하도록
    row 위치를 한 번만 고정해서 샘플링한다.
    """
    n = len(validation_df)

    if SHAP_SAMPLE_SIZE is None or SHAP_SAMPLE_SIZE >= n:
        return validation_df.copy()

    rng = np.random.default_rng(
        SHAP_RANDOM_SEED
        + int(valid_season)
    )

    positions = np.sort(
        rng.choice(
            n,
            size=SHAP_SAMPLE_SIZE,
            replace=False,
        )
    )

    return validation_df.iloc[
        positions
    ].copy()


def calculate_model_shap(
    model,
    X,
    features,
    cat_cols,
    experiment_name,
    valid_season,
):
    """CatBoost native ShapValues의 mean absolute SHAP을 계산한다."""
    model_feature_names = list(
        getattr(
            model,
            "feature_names_",
            [],
        )
    )

    if model_feature_names and model_feature_names != list(features):
        raise AssertionError(
            f"{experiment_name} {valid_season}: "
            "저장된 모델 Feature 순서와 분석 Feature 순서가 다릅니다."
        )

    pool = Pool(
        X,
        cat_features=cat_cols,
        feature_names=list(features),
    )

    shap_values = model.get_feature_importance(
        data=pool,
        type="ShapValues",
    )

    # 마지막 column은 Expected Value이므로 Feature importance에서 제외
    feature_shap = shap_values[:, :-1]

    if feature_shap.shape[1] != len(features):
        raise AssertionError(
            "SHAP Feature 수가 입력 Feature 수와 다릅니다."
        )

    mean_abs = np.mean(
        np.abs(feature_shap),
        axis=0,
    )

    return pd.DataFrame(
        {
            "experiment": experiment_name,
            "validation_season": int(valid_season),
            "feature": list(features),
            "mean_abs_shap": mean_abs,
            "n_shap_sample": int(len(X)),
        }
    )


def run_shap_analysis(
    train_with_temporal,
    raw_features,
    analysis_dir,
):
    """
    Basic과 EXP3의 동일 Validation 표본에서 SHAP을 계산한다.
    """
    basic_model_dir = (
        MODEL_DIR
        / BASIC_EXPERIMENT_NAME
    )
    exp3_model_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
    )

    if not basic_model_dir.exists():
        raise FileNotFoundError(
            f"Basic Fold 모델 폴더가 없습니다: {basic_model_dir}"
        )

    if not exp3_model_dir.exists():
        raise FileNotFoundError(
            f"EXP3 Fold 모델 폴더가 없습니다: {exp3_model_dir}"
        )

    basic_features = list(raw_features)
    exp3_features = (
        list(raw_features)
        + TEMPORAL_FEATURES
    )

    basic_cat_cols = [
        col for col in CAT_COLS
        if col in basic_features
    ]
    exp3_cat_cols = [
        col for col in CAT_COLS
        if col in exp3_features
    ]

    shap_frames = []

    for valid_season in VALID_SEASONS:
        validation = train_with_temporal.loc[
            train_with_temporal["season"]
            == valid_season
        ].copy()

        sample = select_shap_sample(
            validation,
            valid_season,
        )

        # 같은 row sample 사용
        X_basic = prepare_features(
            sample[basic_features],
            basic_cat_cols,
        )
        X_exp3 = prepare_features(
            sample[exp3_features],
            exp3_cat_cols,
        )

        basic_model_path = (
            basic_model_dir
            / f"fold_{valid_season}"
            / (
                f"{BASIC_EXPERIMENT_NAME}"
                f"_fold_{valid_season}.pkl"
            )
        )

        exp3_model_path = (
            exp3_model_dir
            / f"fold_{valid_season}"
            / (
                f"{EXPERIMENT_NAME}"
                f"_fold_{valid_season}.pkl"
            )
        )

        if not basic_model_path.exists():
            raise FileNotFoundError(
                f"Basic 모델이 없습니다: {basic_model_path}"
            )
        if not exp3_model_path.exists():
            raise FileNotFoundError(
                f"EXP3 모델이 없습니다: {exp3_model_path}"
            )

        print(
            f"[SHAP] {valid_season} "
            f"sample={len(sample):,}"
        )

        basic_model = joblib.load(
            basic_model_path
        )
        exp3_model = joblib.load(
            exp3_model_path
        )

        basic_shap = calculate_model_shap(
            model=basic_model,
            X=X_basic,
            features=basic_features,
            cat_cols=basic_cat_cols,
            experiment_name=BASIC_EXPERIMENT_NAME,
            valid_season=valid_season,
        )
        exp3_shap = calculate_model_shap(
            model=exp3_model,
            X=X_exp3,
            features=exp3_features,
            cat_cols=exp3_cat_cols,
            experiment_name=EXPERIMENT_NAME,
            valid_season=valid_season,
        )

        shap_frames.extend(
            [basic_shap, exp3_shap]
        )

        del (
            sample,
            X_basic,
            X_exp3,
            basic_model,
            exp3_model,
            basic_shap,
            exp3_shap,
        )
        gc.collect()

    shap = pd.concat(
        shap_frames,
        ignore_index=True,
    )

    shap.to_csv(
        analysis_dir / "shap_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate = (
        shap.groupby(
            ["experiment", "feature"],
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            std_abs_shap_across_folds=(
                "mean_abs_shap",
                "std",
            ),
            n_folds=(
                "validation_season",
                "nunique",
            ),
        )
    )

    aggregate.to_csv(
        analysis_dir / "shap_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # EXP3 Temporal 10개만 별도 저장
    temporal_shap = shap.loc[
        (shap["experiment"] == EXPERIMENT_NAME)
        & shap["feature"].isin(TEMPORAL_FEATURES)
    ].copy()

    temporal_shap.to_csv(
        analysis_dir / "shap_temporal_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    temporal_aggregate = aggregate.loc[
        (aggregate["experiment"] == EXPERIMENT_NAME)
        & aggregate["feature"].isin(TEMPORAL_FEATURES)
    ].sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    temporal_aggregate.to_csv(
        analysis_dir / "shap_temporal_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Temporal 그룹별 SHAP
    group_rows = []

    for valid_season in VALID_SEASONS:
        fold_temporal = temporal_shap.loc[
            temporal_shap["validation_season"]
            == valid_season
        ]

        for group_name, group_features in TEMPORAL_GROUPS.items():
            part = fold_temporal.loc[
                fold_temporal["feature"].isin(
                    group_features
                )
            ]

            group_rows.append(
                {
                    "validation_season": int(valid_season),
                    "group": group_name,
                    "n_features": int(len(part)),
                    "sum_mean_abs_shap": float(
                        part["mean_abs_shap"].sum()
                    ),
                    # Level(6개) vs Trend(4개)처럼 크기가 다른 그룹 비교용
                    "mean_abs_shap_per_feature": float(
                        part["mean_abs_shap"].mean()
                    ),
                }
            )

    group_summary = pd.DataFrame(
        group_rows
    )

    group_summary.to_csv(
        analysis_dir / "shap_temporal_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Temporal 원본 asof_*의 Basic vs EXP3 SHAP 비교
    source = shap.loc[
        shap["feature"].isin(
            SOURCE_ASOF_FEATURES
        )
    ].copy()

    source_pivot = source.pivot_table(
        index=[
            "validation_season",
            "feature",
        ],
        columns="experiment",
        values="mean_abs_shap",
    ).reset_index()

    if (
        BASIC_EXPERIMENT_NAME
        in source_pivot.columns
        and EXPERIMENT_NAME
        in source_pivot.columns
    ):
        source_pivot["exp3_minus_basic_shap"] = (
            source_pivot[EXPERIMENT_NAME]
            - source_pivot[BASIC_EXPERIMENT_NAME]
        )

    source_pivot.to_csv(
        analysis_dir / "shap_source_asof_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Temporal SHAP 간단한 bar plot
    if not temporal_aggregate.empty:
        plot_df = temporal_aggregate.sort_values(
            "mean_abs_shap",
            ascending=True,
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.barh(
            plot_df["feature"],
            plot_df["mean_abs_shap"],
        )
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(
            "EXP3 Temporal Feature SHAP "
            "(4-Fold Mean)"
        )
        fig.tight_layout()
        fig.savefig(
            analysis_dir
            / "shap_temporal_aggregate.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    return (
        shap,
        aggregate,
        temporal_aggregate,
        group_summary,
        source_pivot,
    )


# =========================================================
# Main
# =========================================================

def main():
    analysis_dir = (
        VALIDATION_DIR
        / "analysis"
        / ANALYSIS_NAME
    )

    if analysis_dir.exists():
        raise FileExistsError(
            f"Analysis already exists: {analysis_dir}\n"
            "기존 결과를 보존하고 ANALYSIS_NAME을 변경하세요."
        )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Raw Feature 목록
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    raw_features = [
        col for col in test_cols
        if col != ID_COL
    ]

    if len(raw_features) != 47:
        raise ValueError(
            f"Raw Feature 수가 예상(47)과 다릅니다: {len(raw_features)}"
        )

    # -----------------------------------------------------
    # 분석에 필요한 Train 데이터
    # SHAP 재현을 위해 Raw 47 전체가 필요
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[
            ID_COL,
            *raw_features,
            TARGET_COL,
        ],
    )

    if not train[ID_COL].is_unique:
        raise AssertionError(
            "train.csv row_id가 unique하지 않습니다."
        )

    train = add_temporal_features(
        train
    )

    # -----------------------------------------------------
    # 1. OOF Brier + Residual
    # -----------------------------------------------------

    residual_df, brier_comparison = (
        load_and_compare_oof(
            analysis_dir
        )
    )

    print()
    print("=" * 70)
    print("Basic vs EXP3 OOF Brier")
    print("=" * 70)
    print(
        brier_comparison.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 2. Temporal 값 구간별 Residual
    # -----------------------------------------------------

    analyze_temporal_residuals(
        residual_df=residual_df,
        train_with_temporal=train,
        analysis_dir=analysis_dir,
    )

    # -----------------------------------------------------
    # 3. SHAP
    # -----------------------------------------------------

    (
        _shap,
        _aggregate,
        temporal_aggregate,
        group_summary,
        source_pivot,
    ) = run_shap_analysis(
        train_with_temporal=train,
        raw_features=raw_features,
        analysis_dir=analysis_dir,
    )

    print()
    print("=" * 70)
    print("EXP3 Temporal SHAP - 4 Fold 평균")
    print("=" * 70)
    print(
        temporal_aggregate[
            [
                "feature",
                "mean_abs_shap",
                "std_abs_shap_across_folds",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("Temporal SHAP Group Summary")
    print("=" * 70)
    print(
        group_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("분석 완료")
    print("=" * 70)
    print(f"저장 위치: {analysis_dir}")
    print()
    print(
        "해석 주의: SHAP은 내부 활용도이며, "
        "Feature가 Brier를 개선했다는 증거가 아닙니다."
    )
    print(
        "실제 predictive usefulness는 "
        "brier_comparison.csv 및 residual 결과와 함께 해석하세요."
    )
    print(
        "또한 Basic vs EXP3의 원본 asof_* SHAP 변화는 "
        "정보 재분배를 '시사'할 수 있지만 인과 증거는 아닙니다."
    )


if __name__ == "__main__":
    main()

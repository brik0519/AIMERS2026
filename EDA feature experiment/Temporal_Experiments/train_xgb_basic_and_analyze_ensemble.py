# train_xgb_basic_and_analyze_ensemble.py
"""
LG Aimers Phase2
XGBoost Basic Expanding 학습 + Basic vs EXP3 Ensemble OOF 분석

목적
----
현재까지 확인한 것:
- CatBoost와 XGBoost는 Temporal 10개의 큰 사용 방향은 비슷하지만
  세부 SHAP 순위는 다르다.
- 이제 실제 prediction 수준에서 Temporal 추가가
  두 모델의 오류 구조와 앙상블 성능을 어떻게 바꾸는지 확인한다.

이번 스크립트가 하는 일
----------------------
1) XGBoost Basic (Raw 47) Expanding Window 4-Fold 학습
2) 기존 OOF 로드
   - CatBoost Basic
   - CatBoost EXP3
   - XGBoost EXP3
3) 새 XGBoost Basic OOF와 함께 4개 OOF 정렬/QA
4) 단일 모델 Brier 비교
5) CatBoost-XGBoost prediction correlation / disagreement 비교
6) 기존 고정 앙상블 가중치
      CatBoost 0.5655685
      XGBoost  0.4344315
   로 Basic Ensemble vs EXP3 Ensemble 비교
7) row-level ensemble residual 변화 분석
8) 시즌별 / Basic disagreement 구간별 / prediction shift 구간별 분석

중요한 해석 원칙
----------------
이 분석의 핵심 질문은:

A) Temporal이 각 단일 모델 자체를 개선했는가?
B) Temporal이 CatBoost와 XGBoost의 prediction 차이를 바꿨는가?
C) 그 변화가 실제 앙상블 Brier 개선으로 이어졌는가?

SHAP 차이만으로 C를 주장하지 않고,
실제 OOF prediction과 Brier로 검증한다.

누수 / 독립성
-------------
- XGBoost Basic에는 추가 파생 Feature가 없다.
- Expanding Window:
    2019~2020 -> 2021
    2019~2021 -> 2022
    2019~2022 -> 2023
    2019~2023 -> 2024
- Fold별 전처리는 Train season에만 fit한다.
- Test 데이터 내부 다른 row를 이용한 집계는 없다.
"""

import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "ensemble_basic_vs_exp3_oof_v001"

DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# Source Experiment 이름
# =========================================================

CB_BASIC_EXPERIMENT = "cb_v001_ctr1_expanding"
CB_EXP3_EXPERIMENT = "cb_v001_ctr1_expanding_exp3"

XGB_BASIC_EXPERIMENT = "xgb_basic_expanding_v001"
XGB_EXP3_EXPERIMENT = "xgb_exp3_expanding_shap_v001"


# =========================================================
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2019, 2020],
        "valid_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "valid_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "valid_season": 2024,
    },
]


# =========================================================
# 기존 최종 앙상블 가중치
# =========================================================

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315

if not np.isclose(
    CAT_WEIGHT + XGB_WEIGHT,
    1.0,
):
    raise ValueError(
        "Ensemble weight 합이 1이 아닙니다."
    )


# =========================================================
# XGBoost 기존 정책
# XGB EXP3와 동일
# =========================================================

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42

XGB_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# 공통 유틸
# =========================================================

def brier_metric(
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

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


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

    r = float(
        y_true.mean()
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    baseline_brier = float(
        r * (1.0 - r)
    )

    bss = float(
        max(
            0.0,
            100000.0
            * (
                1.0
                - brier / baseline_brier
            ),
        )
    )

    return {
        "positive_rate": r,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": bss,
        "prediction_mean": float(
            y_pred.mean()
        ),
        "prediction_std": float(
            y_pred.std()
        ),
    }


def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "입력 데이터에 Feature가 없습니다: "
            + ", ".join(missing)
        )

    X = df[features].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


def create_preprocessor(
    cat_cols,
    num_cols,
):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ],
        remainder="drop",
    )


def create_validation_model():
    return XGBClassifier(
        n_estimators=MAX_ITERATIONS,
        eval_metric=brier_metric,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **MODEL_PARAMS,
    )


# =========================================================
# XGBoost Basic 학습
# =========================================================

def train_xgb_basic(
    train,
    raw_features,
):
    """
    XGBoost Basic = Raw 47 only
    XGB EXP3와 동일한 설정 / 전처리 / Expanding Window를 사용한다.
    """
    model_dir = (
        MODEL_DIR
        / XGB_BASIC_EXPERIMENT
    )

    validation_dir = (
        VALIDATION_ROOT
        / XGB_BASIC_EXPERIMENT
    )

    if model_dir.exists():
        raise FileExistsError(
            f"기존 Model 폴더가 있습니다: {model_dir}"
        )

    if validation_dir.exists():
        raise FileExistsError(
            f"기존 Validation 폴더가 있습니다: {validation_dir}"
        )

    model_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    validation_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    cat_cols = [
        col
        for col in XGB_CAT_COLS
        if col in raw_features
    ]

    id_like_cols = [
        col
        for col in ID_LIKE_COLS
        if col in raw_features
    ]

    num_cols = [
        col
        for col in raw_features
        if col not in cat_cols
    ]

    fold_results = []
    oof_frames = []

    print()
    print("=" * 92)
    print("XGBoost Basic Expanding Window")
    print("=" * 92)
    print("XGBoost version  :", xgb.__version__)
    print("Raw Feature      :", len(raw_features))
    print("Categorical cols :", cat_cols)
    print("=" * 92)

    for fold_config in FOLDS:
        fold_number = int(
            fold_config["fold"]
        )

        train_seasons = list(
            fold_config[
                "train_seasons"
            ]
        )

        valid_season = int(
            fold_config[
                "valid_season"
            ]
        )

        # 시간축 QA
        assert (
            max(train_seasons)
            < valid_season
        )

        is_train = (
            train["season"]
            .isin(train_seasons)
        )

        is_valid = (
            train["season"]
            == valid_season
        )

        train_ids = train.loc[
            is_train,
            ID_COL,
        ]

        valid_ids = train.loc[
            is_valid,
            ID_COL,
        ]

        if not set(
            train_ids
        ).isdisjoint(
            set(valid_ids)
        ):
            raise AssertionError(
                f"Fold {fold_number}: "
                "Train/Validation row_id overlap"
            )

        train_df = train.loc[
            is_train
        ]

        valid_df = train.loc[
            is_valid
        ]

        y_train = train_df[
            TARGET_COL
        ].copy()

        y_valid = valid_df[
            TARGET_COL
        ].copy()

        X_train_raw = (
            prepare_raw_features(
                train_df,
                raw_features,
                id_like_cols,
            )
        )

        X_valid_raw = (
            prepare_raw_features(
                valid_df,
                raw_features,
                id_like_cols,
            )
        )

        # Fold Train 데이터에만 fit
        preprocessor = (
            create_preprocessor(
                cat_cols,
                num_cols,
            )
        )

        X_train = (
            preprocessor
            .fit_transform(
                X_train_raw
            )
        )

        X_valid = (
            preprocessor
            .transform(
                X_valid_raw
            )
        )

        model = (
            create_validation_model()
        )

        print()
        print("-" * 92)
        print(
            f"Fold {fold_number}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]}"
            f" -> {valid_season}"
        )
        print(
            "Train:",
            X_train.shape,
            "| Valid:",
            X_valid.shape,
        )
        print("-" * 92)

        start = time.time()

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_valid,
                    y_valid,
                )
            ],
            verbose=100,
        )

        elapsed = float(
            time.time()
            - start
        )

        valid_pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        metrics = (
            calculate_metrics(
                y_valid.to_numpy(),
                valid_pred,
            )
        )

        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        if (
            best_iteration is None
            or int(best_iteration) < 0
        ):
            final_iterations = int(
                MAX_ITERATIONS
            )
        else:
            final_iterations = int(
                best_iteration + 1
            )

        print(
            f"Brier={metrics['brier']:.8f} | "
            f"BSS={metrics['bss']:.2f} | "
            f"iter={final_iterations} | "
            f"time={elapsed:.1f}s"
        )

        # 저장
        fold_model_dir = (
            model_dir
            / f"fold_{valid_season}"
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        joblib.dump(
            model,
            fold_model_dir
            / (
                f"{XGB_BASIC_EXPERIMENT}"
                f"_fold_{valid_season}.pkl"
            ),
            compress=3,
        )

        joblib.dump(
            preprocessor,
            fold_model_dir
            / (
                f"{XGB_BASIC_EXPERIMENT}"
                f"_preprocessor_{valid_season}.pkl"
            ),
            compress=3,
        )

        fold_validation_dir = (
            validation_dir
            / f"fold_{valid_season}"
        )

        fold_validation_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_pred = pd.DataFrame(
            {
                ID_COL: (
                    valid_ids
                    .to_numpy()
                ),
                "season": int(
                    valid_season
                ),
                "target": (
                    y_valid
                    .to_numpy()
                ),
                "prediction": (
                    valid_pred
                ),
                "fold": int(
                    fold_number
                ),
            }
        )

        fold_pred.to_csv(
            fold_validation_dir
            / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fold_metadata = {
            "experiment_name": (
                XGB_BASIC_EXPERIMENT
            ),
            "fold": (
                fold_number
            ),
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "valid_season": (
                valid_season
            ),
            "n_train": int(
                is_train.sum()
            ),
            "n_valid": int(
                is_valid.sum()
            ),
            "raw_feature_count": int(
                len(raw_features)
            ),
            "cat_cols": (
                cat_cols
            ),
            "id_like_cols": (
                id_like_cols
            ),
            "model_params": (
                MODEL_PARAMS
            ),
            "max_iterations": (
                MAX_ITERATIONS
            ),
            "early_stopping_rounds": (
                EARLY_STOPPING_ROUNDS
            ),
            "best_iteration_zero_based": (
                None
                if best_iteration is None
                else int(
                    best_iteration
                )
            ),
            "final_iterations": (
                final_iterations
            ),
            "elapsed_seconds": (
                elapsed
            ),
            **metrics,
        }

        with (
            fold_validation_dir
            / "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                fold_metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        fold_results.append(
            fold_metadata
        )

        oof_frames.append(
            fold_pred
        )

        del (
            X_train_raw,
            X_valid_raw,
            X_train,
            X_valid,
            y_train,
            y_valid,
            model,
            preprocessor,
            valid_pred,
            fold_pred,
        )

        gc.collect()

    oof = pd.concat(
        oof_frames,
        axis=0,
        ignore_index=True,
    )

    if not oof[ID_COL].is_unique:
        raise AssertionError(
            "XGBoost Basic OOF row_id 중복"
        )

    oof.to_csv(
        validation_dir
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled = calculate_metrics(
        oof[
            "target"
        ].to_numpy(),
        oof[
            "prediction"
        ].to_numpy(),
    )

    metadata = {
        "experiment_name": (
            XGB_BASIC_EXPERIMENT
        ),
        "purpose": (
            "Raw47 XGBoost Basic Expanding Window"
        ),
        "raw_feature_count": int(
            len(raw_features)
        ),
        "fold_results": (
            fold_results
        ),
        "pooled_oof_metrics": (
            pooled
        ),
    }

    with (
        validation_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "[완료] XGBoost Basic pooled OOF Brier =",
        f"{pooled['brier']:.8f}",
    )

    return oof


# =========================================================
# OOF 로드 / QA
# =========================================================

def load_oof(
    experiment_name,
):
    path = (
        VALIDATION_ROOT
        / experiment_name
        / "oof_pred.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"OOF 파일 없음: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = {
        ID_COL,
        "season",
        "target",
        "prediction",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path} 필수 컬럼 누락: "
            f"{sorted(missing)}"
        )

    if not df[ID_COL].is_unique:
        raise AssertionError(
            f"{experiment_name}: row_id 중복"
        )

    return df


def align_oofs():
    """
    4개의 OOF를 row_id 기준으로 정확히 맞춘다.

    반환 컬럼
    ---------
    row_id, season, target,
    cb_basic, cb_exp3,
    xgb_basic, xgb_exp3
    """
    sources = {
        "cb_basic": (
            CB_BASIC_EXPERIMENT
        ),
        "cb_exp3": (
            CB_EXP3_EXPERIMENT
        ),
        "xgb_basic": (
            XGB_BASIC_EXPERIMENT
        ),
        "xgb_exp3": (
            XGB_EXP3_EXPERIMENT
        ),
    }

    loaded = {
        key: load_oof(
            experiment_name
        )
        for key, experiment_name
        in sources.items()
    }

    reference = (
        loaded["cb_basic"][
            [
                ID_COL,
                "season",
                "target",
            ]
        ]
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )

    for key, df in loaded.items():
        current = (
            df[
                [
                    ID_COL,
                    "season",
                    "target",
                ]
            ]
            .sort_values(ID_COL)
            .reset_index(drop=True)
        )

        if not reference.equals(
            current
        ):
            raise AssertionError(
                f"{key}: OOF row/season/target이 "
                "CatBoost Basic과 다릅니다."
            )

    merged = (
        reference
        .copy()
    )

    for key, df in loaded.items():
        pred = (
            df[
                [
                    ID_COL,
                    "prediction",
                ]
            ]
            .rename(
                columns={
                    "prediction": key
                }
            )
        )

        merged = merged.merge(
            pred,
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )

    if merged[
        [
            "cb_basic",
            "cb_exp3",
            "xgb_basic",
            "xgb_exp3",
        ]
    ].isna().any().any():
        raise AssertionError(
            "OOF merge 후 prediction 결측이 있습니다."
        )

    return merged


# =========================================================
# 분석용 파생
# =========================================================

def add_analysis_columns(
    df,
):
    result = df.copy()

    y = result[
        "target"
    ].to_numpy(
        dtype=np.float64
    )

    # -----------------------------------------------------
    # 단일 모델 squared error
    # -----------------------------------------------------

    for col in [
        "cb_basic",
        "cb_exp3",
        "xgb_basic",
        "xgb_exp3",
    ]:
        result[
            f"se_{col}"
        ] = (
            result[col] - y
        ) ** 2

    # -----------------------------------------------------
    # 모델별 Temporal 추가에 따른 prediction shift
    # -----------------------------------------------------

    result[
        "cb_temporal_shift"
    ] = (
        result[
            "cb_exp3"
        ]
        - result[
            "cb_basic"
        ]
    )

    result[
        "xgb_temporal_shift"
    ] = (
        result[
            "xgb_exp3"
        ]
        - result[
            "xgb_basic"
        ]
    )

    result[
        "cb_abs_temporal_shift"
    ] = np.abs(
        result[
            "cb_temporal_shift"
        ]
    )

    result[
        "xgb_abs_temporal_shift"
    ] = np.abs(
        result[
            "xgb_temporal_shift"
        ]
    )

    # -----------------------------------------------------
    # CB-XGB disagreement
    # -----------------------------------------------------

    result[
        "basic_disagreement"
    ] = np.abs(
        result[
            "cb_basic"
        ]
        - result[
            "xgb_basic"
        ]
    )

    result[
        "exp3_disagreement"
    ] = np.abs(
        result[
            "cb_exp3"
        ]
        - result[
            "xgb_exp3"
        ]
    )

    result[
        "disagreement_change"
    ] = (
        result[
            "exp3_disagreement"
        ]
        - result[
            "basic_disagreement"
        ]
    )

    # -----------------------------------------------------
    # 고정 가중치 Ensemble
    # -----------------------------------------------------

    result[
        "ensemble_basic"
    ] = (
        CAT_WEIGHT
        * result[
            "cb_basic"
        ]
        + XGB_WEIGHT
        * result[
            "xgb_basic"
        ]
    )

    result[
        "ensemble_exp3"
    ] = (
        CAT_WEIGHT
        * result[
            "cb_exp3"
        ]
        + XGB_WEIGHT
        * result[
            "xgb_exp3"
        ]
    )

    result[
        "se_ensemble_basic"
    ] = (
        result[
            "ensemble_basic"
        ]
        - y
    ) ** 2

    result[
        "se_ensemble_exp3"
    ] = (
        result[
            "ensemble_exp3"
        ]
        - y
    ) ** 2

    # positive -> EXP3 ensemble가 개선
    result[
        "delta_se_ensemble"
    ] = (
        result[
            "se_ensemble_basic"
        ]
        - result[
            "se_ensemble_exp3"
        ]
    )

    result[
        "ensemble_temporal_shift"
    ] = (
        result[
            "ensemble_exp3"
        ]
        - result[
            "ensemble_basic"
        ]
    )

    result[
        "ensemble_abs_temporal_shift"
    ] = np.abs(
        result[
            "ensemble_temporal_shift"
        ]
    )

    return result


# =========================================================
# 비교 summary
# =========================================================

def summarize_models(
    df,
):
    rows = []

    prediction_cols = {
        "cb_basic": "cb_basic",
        "cb_exp3": "cb_exp3",
        "xgb_basic": "xgb_basic",
        "xgb_exp3": "xgb_exp3",
        "ensemble_basic": (
            "ensemble_basic"
        ),
        "ensemble_exp3": (
            "ensemble_exp3"
        ),
    }

    for scope_name, scope_df in [
        ("2021", df.loc[
            df["season"] == 2021
        ]),
        ("2022", df.loc[
            df["season"] == 2022
        ]),
        ("2023", df.loc[
            df["season"] == 2023
        ]),
        ("2024", df.loc[
            df["season"] == 2024
        ]),
        ("pooled_oof", df),
    ]:
        y = scope_df[
            "target"
        ].to_numpy()

        for model_name, col in (
            prediction_cols.items()
        ):
            metrics = (
                calculate_metrics(
                    y,
                    scope_df[col]
                    .to_numpy(),
                )
            )

            rows.append(
                {
                    "scope": (
                        scope_name
                    ),
                    "model": (
                        model_name
                    ),
                    "n": int(
                        len(scope_df)
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


def summarize_pair_relationship(
    df,
):
    """
    Basic vs EXP3에서
    CB-XGB prediction 관계가 어떻게 변했는지 비교.
    """
    rows = []

    for scope_name, scope_df in [
        ("2021", df.loc[
            df["season"] == 2021
        ]),
        ("2022", df.loc[
            df["season"] == 2022
        ]),
        ("2023", df.loc[
            df["season"] == 2023
        ]),
        ("2024", df.loc[
            df["season"] == 2024
        ]),
        ("pooled_oof", df),
    ]:
        basic_corr = float(
            np.corrcoef(
                scope_df[
                    "cb_basic"
                ],
                scope_df[
                    "xgb_basic"
                ],
            )[0, 1]
        )

        exp3_corr = float(
            np.corrcoef(
                scope_df[
                    "cb_exp3"
                ],
                scope_df[
                    "xgb_exp3"
                ],
            )[0, 1]
        )

        rows.append(
            {
                "scope": scope_name,
                "n": int(
                    len(scope_df)
                ),
                "basic_cb_xgb_corr": (
                    basic_corr
                ),
                "exp3_cb_xgb_corr": (
                    exp3_corr
                ),
                "corr_change_exp3_minus_basic": (
                    exp3_corr
                    - basic_corr
                ),
                "basic_mean_abs_disagreement": float(
                    scope_df[
                        "basic_disagreement"
                    ].mean()
                ),
                "exp3_mean_abs_disagreement": float(
                    scope_df[
                        "exp3_disagreement"
                    ].mean()
                ),
                "disagreement_change": float(
                    scope_df[
                        "disagreement_change"
                    ].mean()
                ),
                "cb_mean_abs_temporal_shift": float(
                    scope_df[
                        "cb_abs_temporal_shift"
                    ].mean()
                ),
                "xgb_mean_abs_temporal_shift": float(
                    scope_df[
                        "xgb_abs_temporal_shift"
                    ].mean()
                ),
                "ensemble_mean_abs_temporal_shift": float(
                    scope_df[
                        "ensemble_abs_temporal_shift"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Ensemble identity 분석
# =========================================================

def summarize_ensemble_identity(
    df,
):
    """
    Brier ensemble identity:

    BS_E
      = w*BS_C + (1-w)*BS_X
        - w(1-w) * E[(p_C-p_X)^2]

    disagreement가 앙상블 Brier에 주는 수학적 기여를
    Basic / EXP3 각각 확인한다.
    """
    rows = []

    for scope_name, scope_df in [
        ("2021", df.loc[
            df["season"] == 2021
        ]),
        ("2022", df.loc[
            df["season"] == 2022
        ]),
        ("2023", df.loc[
            df["season"] == 2023
        ]),
        ("2024", df.loc[
            df["season"] == 2024
        ]),
        ("pooled_oof", df),
    ]:
        y = scope_df[
            "target"
        ].to_numpy()

        for variant in [
            "basic",
            "exp3",
        ]:
            cb_col = (
                f"cb_{variant}"
            )

            xgb_col = (
                f"xgb_{variant}"
            )

            ensemble_col = (
                f"ensemble_{variant}"
            )

            cb_bs = brier_metric(
                y,
                scope_df[
                    cb_col
                ],
            )

            xgb_bs = brier_metric(
                y,
                scope_df[
                    xgb_col
                ],
            )

            ensemble_bs = (
                brier_metric(
                    y,
                    scope_df[
                        ensemble_col
                    ],
                )
            )

            mean_sq_disagreement = float(
                np.mean(
                    (
                        scope_df[
                            cb_col
                        ]
                        - scope_df[
                            xgb_col
                        ]
                    ) ** 2
                )
            )

            weighted_single_bs = float(
                CAT_WEIGHT
                * cb_bs
                + XGB_WEIGHT
                * xgb_bs
            )

            diversity_term = float(
                CAT_WEIGHT
                * XGB_WEIGHT
                * mean_sq_disagreement
            )

            reconstructed = float(
                weighted_single_bs
                - diversity_term
            )

            rows.append(
                {
                    "scope": (
                        scope_name
                    ),
                    "variant": (
                        variant
                    ),
                    "cb_brier": (
                        cb_bs
                    ),
                    "xgb_brier": (
                        xgb_bs
                    ),
                    "weighted_single_brier": (
                        weighted_single_bs
                    ),
                    "mean_squared_disagreement": (
                        mean_sq_disagreement
                    ),
                    "diversity_term": (
                        diversity_term
                    ),
                    "ensemble_brier": (
                        ensemble_bs
                    ),
                    "reconstructed_ensemble_brier": (
                        reconstructed
                    ),
                    "identity_error": float(
                        ensemble_bs
                        - reconstructed
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Quantile 분석
# =========================================================

def quantile_bin(
    series,
    q,
    prefix,
):
    """
    동일 값 때문에 실제 bin 수가 줄어들 수 있어
    duplicates='drop' 사용.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    try:
        codes = pd.qcut(
            values,
            q=q,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return pd.Series(
            ["All"] * len(values),
            index=values.index,
        )

    result = pd.Series(
        index=values.index,
        dtype="object",
    )

    valid = codes.notna()

    result.loc[
        valid
    ] = (
        prefix
        + (
            codes.loc[
                valid
            ].astype(int)
            + 1
        ).astype(str)
    )

    result.loc[
        ~valid
    ] = "Missing"

    return result


def summarize_by_bins(
    df,
):
    """
    어떤 row 영역에서 EXP3 Ensemble 개선이 발생하는지 확인.

    1) Basic CB-XGB disagreement decile
    2) Temporal 추가로 disagreement가 얼마나 변했는지 decile
    3) Ensemble prediction shift magnitude decile
    """
    temp = df.copy()

    temp[
        "basic_disagreement_bin"
    ] = quantile_bin(
        temp[
            "basic_disagreement"
        ],
        q=10,
        prefix="D",
    )

    temp[
        "disagreement_change_bin"
    ] = quantile_bin(
        temp[
            "disagreement_change"
        ],
        q=10,
        prefix="DC",
    )

    temp[
        "ensemble_shift_bin"
    ] = quantile_bin(
        temp[
            "ensemble_abs_temporal_shift"
        ],
        q=10,
        prefix="S",
    )

    summaries = []

    for analysis_name, col in [
        (
            "basic_disagreement",
            "basic_disagreement_bin",
        ),
        (
            "disagreement_change",
            "disagreement_change_bin",
        ),
        (
            "ensemble_temporal_shift",
            "ensemble_shift_bin",
        ),
    ]:
        grouped = (
            temp
            .groupby(
                [
                    "season",
                    col,
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                n=(
                    ID_COL,
                    "size",
                ),
                mean_delta_se_ensemble=(
                    "delta_se_ensemble",
                    "mean",
                ),
                mean_basic_disagreement=(
                    "basic_disagreement",
                    "mean",
                ),
                mean_exp3_disagreement=(
                    "exp3_disagreement",
                    "mean",
                ),
                mean_disagreement_change=(
                    "disagreement_change",
                    "mean",
                ),
                mean_ensemble_abs_shift=(
                    "ensemble_abs_temporal_shift",
                    "mean",
                ),
            )
        )

        grouped.insert(
            0,
            "analysis",
            analysis_name,
        )

        grouped = grouped.rename(
            columns={
                col: "bin"
            }
        )

        summaries.append(
            grouped
        )

    return pd.concat(
        summaries,
        axis=0,
        ignore_index=True,
    )


# =========================================================
# 콘솔 출력
# =========================================================

def print_core_results(
    model_summary,
    pair_summary,
    identity_summary,
):
    print()
    print("=" * 116)
    print("단일 모델 + Ensemble OOF Brier")
    print("=" * 116)

    pivot = (
        model_summary
        .pivot(
            index="model",
            columns="scope",
            values="brier",
        )
        .reset_index()
    )

    desired = [
        "cb_basic",
        "cb_exp3",
        "xgb_basic",
        "xgb_exp3",
        "ensemble_basic",
        "ensemble_exp3",
    ]

    pivot[
        "__order"
    ] = (
        pivot[
            "model"
        ].map(
            {
                name: i
                for i, name
                in enumerate(
                    desired
                )
            }
        )
    )

    pivot = (
        pivot
        .sort_values(
            "__order"
        )
        .drop(
            columns="__order"
        )
    )

    cols = [
        "model",
        "2021",
        "2022",
        "2023",
        "2024",
        "pooled_oof",
    ]

    print(
        pivot[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )

    # Basic vs EXP3 delta
    lookup = (
        pivot
        .set_index(
            "model"
        )
    )

    print()
    print("[Temporal 추가에 따른 pooled OOF 변화]")
    print(
        "CatBoost  Basic - EXP3 =",
        f"{lookup.loc['cb_basic', 'pooled_oof'] - lookup.loc['cb_exp3', 'pooled_oof']:+.8f}",
    )
    print(
        "XGBoost   Basic - EXP3 =",
        f"{lookup.loc['xgb_basic', 'pooled_oof'] - lookup.loc['xgb_exp3', 'pooled_oof']:+.8f}",
    )
    print(
        "Ensemble  Basic - EXP3 =",
        f"{lookup.loc['ensemble_basic', 'pooled_oof'] - lookup.loc['ensemble_exp3', 'pooled_oof']:+.8f}",
    )

    print()
    print("=" * 116)
    print("CatBoost-XGBoost Prediction 관계")
    print("=" * 116)

    print(
        pair_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )

    print()
    print("=" * 116)
    print("Ensemble Brier Identity")
    print("=" * 116)

    pooled_identity = (
        identity_summary.loc[
            identity_summary[
                "scope"
            ]
            == "pooled_oof"
        ]
        .copy()
    )

    print(
        pooled_identity[
            [
                "variant",
                "cb_brier",
                "xgb_brier",
                "weighted_single_brier",
                "mean_squared_disagreement",
                "diversity_term",
                "ensemble_brier",
                "identity_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )

    print()
    print(
        "해석:"
    )
    print(
        "- Basic-EXP3 delta > 0 이면 EXP3가 개선."
    )
    print(
        "- corr가 내려가고 disagreement가 올라갔다고 해서 "
        "그 자체로 좋은 것은 아님."
    )
    print(
        "- 실제 판단은 ensemble_brier와 "
        "diversity_term 변화까지 같이 봐야 함."
    )
    print("=" * 116)


# =========================================================
# Main
# =========================================================

def main():
    # -----------------------------------------------------
    # 기존 Source 존재 확인
    # -----------------------------------------------------

    required_sources = [
        CB_BASIC_EXPERIMENT,
        CB_EXP3_EXPERIMENT,
        XGB_EXP3_EXPERIMENT,
    ]

    for experiment_name in (
        required_sources
    ):
        path = (
            VALIDATION_ROOT
            / experiment_name
            / "oof_pred.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"필요한 기존 OOF 없음: {path}"
            )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 분석 폴더가 있습니다: "
            f"{VALIDATION_DIR}"
        )

    # -----------------------------------------------------
    # Raw 47
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID_COL not in test_cols:
        raise ValueError(
            f"test.csv에 {ID_COL}가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID_COL
    ]

    if len(raw_features) != 47:
        raise ValueError(
            f"Raw Feature 수가 예상 47과 다릅니다: "
            f"{len(raw_features)}"
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR
        / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID_COL]
            + raw_features
            + [TARGET_COL]
        ),
    )

    if not train[ID_COL].is_unique:
        raise AssertionError(
            "train.csv row_id 중복"
        )

    if train[
        TARGET_COL
    ].isna().any():
        raise AssertionError(
            "train target 결측"
        )

    if "season" not in train.columns:
        raise ValueError(
            "train.csv에 season 컬럼이 없습니다."
        )

    # -----------------------------------------------------
    # XGB Basic 학습
    # -----------------------------------------------------

    train_xgb_basic(
        train=train,
        raw_features=raw_features,
    )

    # -----------------------------------------------------
    # Analysis 폴더 생성
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        VALIDATION_DIR
        / "train_xgb_basic_and_analyze_ensemble.py",
    )

    # -----------------------------------------------------
    # OOF align
    # -----------------------------------------------------

    aligned = align_oofs()

    analyzed = (
        add_analysis_columns(
            aligned
        )
    )

    analyzed.to_csv(
        VALIDATION_DIR
        / "oof_row_level_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Summaries
    # -----------------------------------------------------

    model_summary = (
        summarize_models(
            analyzed
        )
    )

    pair_summary = (
        summarize_pair_relationship(
            analyzed
        )
    )

    identity_summary = (
        summarize_ensemble_identity(
            analyzed
        )
    )

    bin_summary = (
        summarize_by_bins(
            analyzed
        )
    )

    model_summary.to_csv(
        VALIDATION_DIR
        / "model_brier_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_summary.to_csv(
        VALIDATION_DIR
        / "prediction_relationship_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    identity_summary.to_csv(
        VALIDATION_DIR
        / "ensemble_identity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bin_summary.to_csv(
        VALIDATION_DIR
        / "ensemble_delta_by_bin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "source_experiments": {
            "cb_basic": (
                CB_BASIC_EXPERIMENT
            ),
            "cb_exp3": (
                CB_EXP3_EXPERIMENT
            ),
            "xgb_basic": (
                XGB_BASIC_EXPERIMENT
            ),
            "xgb_exp3": (
                XGB_EXP3_EXPERIMENT
            ),
        },
        "ensemble_weights": {
            "catboost": (
                CAT_WEIGHT
            ),
            "xgboost": (
                XGB_WEIGHT
            ),
        },
        "xgb_model_params": (
            MODEL_PARAMS
        ),
        "xgb_max_iterations": (
            MAX_ITERATIONS
        ),
        "xgb_early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "validation_folds": (
            FOLDS
        ),
        "interpretation_note": (
            "Correlation/disagreement change alone does not prove "
            "ensemble improvement. Evaluate single-model Brier, "
            "ensemble Brier, and diversity term together."
        ),
    }

    with (
        VALIDATION_DIR
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Console
    # -----------------------------------------------------

    print_core_results(
        model_summary=(
            model_summary
        ),
        pair_summary=(
            pair_summary
        ),
        identity_summary=(
            identity_summary
        ),
    )

    print()
    print(
        "결과 저장:",
        VALIDATION_DIR,
    )
    print()
    print(
        "다음 파일을 우선 확인:"
    )
    print(
        "1) model_brier_summary.csv"
    )
    print(
        "2) prediction_relationship_summary.csv"
    )
    print(
        "3) ensemble_identity_summary.csv"
    )
    print(
        "4) ensemble_delta_by_bin.csv"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()

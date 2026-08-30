# script.py: cb_v004 추론 및 제출
#
# train.py(cb_v004)에서 생성된 단일 Artifact(model/cb_v004/cb_v004.pkl)를
# 로드하여 test.csv 각 행을 독립적으로 추론하고 output/submission.csv를
# 생성한다.
#
# cb_v004 train.py는 categorical 표현(CAT_FULL9/CAT_NOID7/CAT_MIN3 중 하나)과
# Temporal10 채택 여부를 실행마다 데이터로부터 탐색해 결정하므로, 이 파일은
# 그 결과를 하드코딩하지 않고 Artifact에 저장된 값(features/cat_cols/
# use_temporal_features)을 그대로 읽어 동일하게 재현한다.
#
# 규칙 준수:
# - test.csv는 이 파일에서만 읽으며, 각 행은 독립적으로 추론된다.
# - Temporal10(채택된 경우)은 해당 행의 asof_* 컬럼만으로 계산되는 결정론적
#   함수이며, test 내 다른 행이나 분포 통계를 전혀 참조하지 않는다.
# - test 행 간 groupby/rolling/rank/평균 등 어떤 형태의 집계도 사용하지 않는다.

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# 실험 설정
# =========================================================

EXPERIMENT_NAME = "ensemble_v005"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")

CB_EXPERIMENT_NAME = "cb_v002"
XGB_EXPERIMENT_NAME = "xgb_v002"
MLP_EXPERIMENT_NAME = "mlp_v003"

CB_MODEL_DIR = MODEL_ROOT / CB_EXPERIMENT_NAME
CB_MODEL_PATH = (
    CB_MODEL_DIR
    / f"{CB_EXPERIMENT_NAME}.pkl"
)

XGB_MODEL_DIR = MODEL_ROOT / XGB_EXPERIMENT_NAME
XGB_MODEL_PATH = (
    XGB_MODEL_DIR
    / f"{XGB_EXPERIMENT_NAME}.pkl"
)

MLP_MODEL_DIR = MODEL_ROOT / MLP_EXPERIMENT_NAME
MLP_MODEL_PATH = (
    MLP_MODEL_DIR
    / f"{MLP_EXPERIMENT_NAME}.pkl"
)


OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = (
    DATA_DIR
    / "sample_submission.csv"
)
OUTPUT_PATH = (
    OUTPUT_DIR
    / "submission.csv"
)

ID = "row_id"
TARGET = "control_success"

MODEL_NAMES = (
    CB_EXPERIMENT_NAME,
    XGB_EXPERIMENT_NAME,
    MLP_EXPERIMENT_NAME,
)


# =========================================================
# Device
# =========================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# Ensemble weight
# =========================================================
#
# ensemble_v004 train.py 실행 후 metadata.json의
# global_best.weights 값으로 반드시 교체한다.
#
# 평가 서버에서는 validation 결과 파일에 의존하지 않고,
# 아래 고정 weight만 사용한다.
#
ENSEMBLE_WEIGHTS = {
    "cb_v002": 0.857980818205651,
    "xgb_v002": 0.09536337329674678,
    "mlp_v003": 0.046655808497602244,
}


def load_ensemble_weights() -> dict[str, float]:
    weights = {}

    for model_name in MODEL_NAMES:
        raw_weight = ENSEMBLE_WEIGHTS.get(
            model_name
        )

        if raw_weight is None:
            raise ValueError(
                "ensemble_v005 weight가 아직 확정되지 않았습니다. "
                "train_ensemble_v005.py 실행 후 "
                "metadata.json의 global_best.weights를 "
                "ENSEMBLE_WEIGHTS에 입력하세요."
            )

        weight = float(
            raw_weight
        )

        if not np.isfinite(
            weight
        ):
            raise ValueError(
                f"{model_name} weight가 유한한 값이 아닙니다."
            )

        if weight < 0.0:
            raise ValueError(
                f"{model_name} weight가 음수입니다."
            )

        weights[
            model_name
        ] = weight

    weight_sum = float(
        sum(
            weights.values()
        )
    )

    if weight_sum <= 0.0:
        raise ValueError(
            "Ensemble weight 합이 0 이하입니다."
        )

    return {
        model_name:
            weight / weight_sum
        for model_name, weight
        in weights.items()
    }


# =========================================================
# CatBoost preprocessing / inference
# =========================================================

CB_CAT_COLS = [
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


def prepare_cb_features(
    df: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    missing_features = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "CatBoost 입력 Feature가 없습니다: "
            + ", ".join(
                missing_features
            )
        )

    X = df[
        features
    ].copy()

    for column in cat_cols:
        X[column] = (
            X[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    return X


def predict_catboost(
    test: pd.DataFrame,
) -> np.ndarray:
    if not CB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CatBoost 모델이 없습니다: "
            f"{CB_MODEL_PATH}"
        )

    model = joblib.load(
        CB_MODEL_PATH
    )

    features = [
        column
        for column in test.columns
        if column != ID
    ]

    cat_cols = [
        column
        for column in CB_CAT_COLS
        if column in features
    ]

    X_test = prepare_cb_features(
        test,
        features,
        cat_cols,
    )

    prediction = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    return np.asarray(
        prediction,
        dtype=np.float64,
    )


# =========================================================
# XGBoost preprocessing / inference
# =========================================================

def prepare_xgb_raw_features(
    df: pd.DataFrame,
    features: list[str],
    id_like_cols: list[str],
) -> pd.DataFrame:
    missing_features = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "XGBoost 입력 Feature가 없습니다: "
            + ", ".join(
                missing_features
            )
        )

    X = df[
        features
    ].copy()

    for column in id_like_cols:
        if column in X.columns:
            X[column] = (
                pd.to_numeric(
                    X[column],
                    errors="coerce",
                )
            )

    return X


def predict_xgboost(
    test: pd.DataFrame,
) -> np.ndarray:
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost 모델이 없습니다: "
            f"{XGB_MODEL_PATH}"
        )

    artifact = joblib.load(
        XGB_MODEL_PATH
    )

    required_keys = {
        "model",
        "preprocessor",
        "features",
        "id_like_cols",
    }

    missing = (
        required_keys
        - set(
            artifact.keys()
        )
    )

    if missing:
        raise ValueError(
            "XGBoost artifact key 누락: "
            + ", ".join(
                sorted(missing)
            )
        )

    model = artifact[
        "model"
    ]

    preprocessor = artifact[
        "preprocessor"
    ]

    features = artifact[
        "features"
    ]

    id_like_cols = artifact[
        "id_like_cols"
    ]

    X_raw = (
        prepare_xgb_raw_features(
            test,
            features,
            id_like_cols,
        )
    )

    X_test = (
        preprocessor.transform(
            X_raw
        )
    )

    prediction = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    return np.asarray(
        prediction,
        dtype=np.float64,
    )


# =========================================================
# MLP preprocessing
# =========================================================

def make_count_state(
    balls: pd.Series,
    strikes: pd.Series,
) -> pd.Series:
    balls_numeric = pd.to_numeric(
        balls,
        errors="coerce",
    )

    strikes_numeric = pd.to_numeric(
        strikes,
        errors="coerce",
    )

    missing = (
        balls_numeric.isna()
        | strikes_numeric.isna()
    )

    result = (
        balls_numeric
        .fillna(-1)
        .astype(np.int16)
        .astype(str)
        + "-"
        + strikes_numeric
        .fillna(-1)
        .astype(np.int16)
        .astype(str)
    )

    result = result.astype(
        object
    )

    result.loc[
        missing
    ] = None

    return result


def normalize_category_value(
    value: Any,
) -> str | None:
    if pd.isna(
        value
    ):
        return None

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        if (
            np.isfinite(
                value
            )
            and float(
                value
            ).is_integer()
        ):
            return str(
                int(
                    value
                )
            )

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return str(
            int(
                value
            )
        )

    return str(
        value
    )


def build_mlp_feature_frame(
    df: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> pd.DataFrame:
    X = df.copy()

    if (
        "count_state"
        in preprocessor[
            "categorical_columns"
        ]
    ):
        required = {
            "balls_before",
            "strikes_before",
        }

        missing = (
            required
            - set(
                X.columns
            )
        )

        if missing:
            raise ValueError(
                "MLP count_state 생성 컬럼 누락: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        X[
            "count_state"
        ] = make_count_state(
            X["balls_before"],
            X["strikes_before"],
        )

    required_features = (
        list(
            preprocessor[
                "categorical_columns"
            ]
        )
        + list(
            preprocessor[
                "numeric_columns"
            ]
        )
    )

    missing_features = [
        column
        for column in required_features
        if column not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "MLP 입력 Feature가 없습니다: "
            + ", ".join(
                missing_features
            )
        )

    return X[
        required_features
    ].copy()


def transform_mlp_categorical(
    series: pd.Series,
    mapping: dict[str, int],
    unknown_index: int,
    missing_index: int,
) -> np.ndarray:
    output = np.empty(
        len(
            series
        ),
        dtype=np.int64,
    )

    values = series.to_numpy(
        dtype=object,
        copy=False,
    )

    for index, raw_value in enumerate(
        values
    ):
        value = normalize_category_value(
            raw_value
        )

        if value is None:
            output[
                index
            ] = missing_index
        else:
            output[
                index
            ] = mapping.get(
                value,
                unknown_index,
            )

    return output


def transform_mlp_numeric(
    series: pd.Series,
    stats: dict[str, float],
) -> np.ndarray:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = (
        numeric
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            float(
                stats[
                    "median"
                ]
            )
        )
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )

    values = (
        values
        - float(
            stats[
                "mean"
            ]
        )
    ) / float(
        stats[
            "std"
        ]
    )

    values = values.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "MLP numeric transform 후 "
            "NaN/Inf가 발생했습니다."
        )

    return values


def transform_mlp_dataframe(
    df: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    feature_frame = (
        build_mlp_feature_frame(
            df,
            preprocessor,
        )
    )

    categorical_columns = (
        preprocessor[
            "categorical_columns"
        ]
    )

    numeric_columns = (
        preprocessor[
            "numeric_columns"
        ]
    )

    unknown_index = int(
        preprocessor[
            "unknown_index"
        ]
    )

    missing_index = int(
        preprocessor[
            "missing_index"
        ]
    )

    categorical_arrays = []

    for column in categorical_columns:
        encoded = (
            transform_mlp_categorical(
                feature_frame[
                    column
                ],
                preprocessor[
                    "category_maps"
                ][column],
                unknown_index,
                missing_index,
            )
        )

        categorical_arrays.append(
            encoded
        )

    if categorical_arrays:
        categorical = np.column_stack(
            categorical_arrays
        ).astype(
            np.int64,
            copy=False,
        )
    else:
        categorical = np.empty(
            (
                len(df),
                0,
            ),
            dtype=np.int64,
        )

    numeric_arrays = []

    for column in numeric_columns:
        transformed = (
            transform_mlp_numeric(
                feature_frame[
                    column
                ],
                preprocessor[
                    "numeric_stats"
                ][column],
            )
        )

        numeric_arrays.append(
            transformed
        )

    if numeric_arrays:
        numeric = np.column_stack(
            numeric_arrays
        ).astype(
            np.float32,
            copy=False,
        )
    else:
        numeric = np.empty(
            (
                len(df),
                0,
            ),
            dtype=np.float32,
        )

    return (
        numeric,
        categorical,
    )


# =========================================================
# MLP model
# =========================================================

class TabularMLP(
    nn.Module
):
    def __init__(
        self,
        cardinalities: list[int],
        embedding_dims: list[int],
        n_num_features: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        self.embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=int(
                            cardinality
                        ),
                        embedding_dim=int(
                            dim
                        ),
                        padding_idx=0,
                    )
                    for cardinality, dim
                    in zip(
                        cardinalities,
                        embedding_dims,
                    )
                ]
            )
        )

        embedding_total = int(
            sum(
                embedding_dims
            )
        )

        input_dim = (
            embedding_total
            + int(
                n_num_features
            )
        )

        if input_dim <= 0:
            raise ValueError(
                "MLP input feature가 없습니다."
            )

        layers = []

        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        prev_dim,
                        hidden_dim,
                    ),
                    nn.LayerNorm(
                        hidden_dim
                    ),
                    nn.SiLU(),
                    nn.Dropout(
                        dropout
                    ),
                ]
            )

            prev_dim = (
                hidden_dim
            )

        self.backbone = (
            nn.Sequential(
                *layers
            )
        )

        self.head = (
            nn.Linear(
                prev_dim,
                1,
            )
        )

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        parts = []

        if len(
            self.embeddings
        ) > 0:
            embedded = [
                embedding(
                    categorical[
                        :,
                        index,
                    ]
                )
                for index, embedding
                in enumerate(
                    self.embeddings
                )
            ]

            parts.append(
                torch.cat(
                    embedded,
                    dim=1,
                )
            )

        if numeric.shape[
            1
        ] > 0:
            parts.append(
                numeric
            )

        x = (
            parts[
                0
            ]
            if len(
                parts
            ) == 1
            else torch.cat(
                parts,
                dim=1,
            )
        )

        x = self.backbone(
            x
        )

        return (
            self.head(
                x
            )
            .squeeze(
                1
            )
        )


def create_mlp_model(
    model_config: dict[str, Any],
) -> TabularMLP:
    return TabularMLP(
        cardinalities=(
            model_config[
                "cardinalities"
            ]
        ),
        embedding_dims=(
            model_config[
                "embedding_dims"
            ]
        ),
        n_num_features=(
            model_config[
                "n_num_features"
            ]
        ),
        hidden_dims=(
            model_config[
                "hidden_dims"
            ]
        ),
        dropout=(
            model_config[
                "dropout"
            ]
        ),
    )


def predict_mlp(
    test: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MLP 모델이 없습니다: "
            f"{MLP_MODEL_PATH}"
        )

    bundle = joblib.load(
        MLP_MODEL_PATH
    )

    required_keys = {
        "model_state_dict",
        "model_config",
        "preprocessor",
    }

    missing = (
        required_keys
        - set(
            bundle.keys()
        )
    )

    if missing:
        raise ValueError(
            "MLP bundle key 누락: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    preprocessor = (
        bundle[
            "preprocessor"
        ]
    )

    numeric, categorical = (
        transform_mlp_dataframe(
            test,
            preprocessor,
        )
    )

    model = create_mlp_model(
        bundle[
            "model_config"
        ]
    ).to(
        device
    )

    model.load_state_dict(
        bundle[
            "model_state_dict"
        ]
    )

    model.eval()

    batch_size = 8192
    predictions = []

    with torch.inference_mode():
        for start in range(
            0,
            len(test),
            batch_size,
        ):
            end = min(
                start
                + batch_size,
                len(test),
            )

            categorical_batch = (
                torch.from_numpy(
                    categorical[
                        start:end
                    ]
                )
                .to(
                    device,
                    dtype=torch.long,
                )
            )

            numeric_batch = (
                torch.from_numpy(
                    numeric[
                        start:end
                    ]
                )
                .to(
                    device,
                    dtype=torch.float32,
                )
            )

            logits = model(
                categorical_batch,
                numeric_batch,
            )

            probability = (
                torch.sigmoid(
                    logits
                )
                .detach()
                .cpu()
                .numpy()
            )

            predictions.append(
                probability
            )

    return (
        np.concatenate(
            predictions
        )
        .astype(
            np.float64,
            copy=False,
        )
    )


# =========================================================
# Validation
# =========================================================

def validate_predictions(
    prediction: np.ndarray,
    n_expected: int,
    model_name: str,
) -> None:
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if len(
        prediction
    ) != n_expected:
        raise RuntimeError(
            f"{model_name}: 예측 수와 "
            "test 행 수가 다릅니다."
        )

    if not np.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            f"{model_name}: 예측에 "
            "NaN/Inf가 있습니다."
        )

    if not (
        (
            prediction
            >= 0.0
        )
        & (
            prediction
            <= 1.0
        )
    ).all():
        raise RuntimeError(
            f"{model_name}: 예측이 "
            "0~1 범위를 벗어났습니다."
        )


# =========================================================
# Main
# =========================================================

def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv가 없습니다: "
            f"{TEST_PATH}"
        )

    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(
            "sample_submission.csv가 없습니다: "
            f"{SAMPLE_SUB_PATH}"
        )

    weights = (
        load_ensemble_weights()
    )

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    submission = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    if ID not in test.columns:
        raise ValueError(
            f"test.csv에 {ID}가 없습니다."
        )

    if ID not in submission.columns:
        raise ValueError(
            f"sample_submission.csv에 "
            f"{ID}가 없습니다."
        )

    if TARGET not in submission.columns:
        raise ValueError(
            f"sample_submission.csv에 "
            f"{TARGET}가 없습니다."
        )

    if len(
        test
    ) != len(
        submission
    ):
        raise ValueError(
            "test.csv와 sample_submission.csv "
            "행 수가 다릅니다."
        )

    if not (
        test[
            ID
        ]
        .astype(
            str
        )
        .reset_index(
            drop=True
        )
        .equals(
            submission[
                ID
            ]
            .astype(
                str
            )
            .reset_index(
                drop=True
            )
        )
    ):
        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "row_id 순서가 다릅니다."
        )

    device = get_device()

    print("=" * 80)
    print("LG Aimers Ensemble v005 Inference")
    print("=" * 80)
    print(f"Rows       : {len(test):,}")
    print(f"Device     : {device}")
    print(f"Weights    : {weights}")
    print("=" * 80)

    # -----------------------------------------------------
    # 각 모델 독립 추론
    # -----------------------------------------------------

    cb_prediction = (
        predict_catboost(
            test
        )
    )

    xgb_prediction = (
        predict_xgboost(
            test
        )
    )

    mlp_prediction = (
        predict_mlp(
            test,
            device,
        )
    )

    validate_predictions(
        cb_prediction,
        len(
            test
        ),
        CB_EXPERIMENT_NAME,
    )

    validate_predictions(
        xgb_prediction,
        len(
            test
        ),
        XGB_EXPERIMENT_NAME,
    )

    validate_predictions(
        mlp_prediction,
        len(
            test
        ),
        MLP_EXPERIMENT_NAME,
    )

    print()
    print("[Single predictions]")

    print(
        f"{CB_EXPERIMENT_NAME:<10} "
        f"mean={cb_prediction.mean():.6f} "
        f"std={cb_prediction.std():.6f}"
    )

    print(
        f"{XGB_EXPERIMENT_NAME:<10} "
        f"mean={xgb_prediction.mean():.6f} "
        f"std={xgb_prediction.std():.6f}"
    )

    print(
        f"{MLP_EXPERIMENT_NAME:<10} "
        f"mean={mlp_prediction.mean():.6f} "
        f"std={mlp_prediction.std():.6f}"
    )

    # -----------------------------------------------------
    # 고정 weight ensemble
    #
    # weight는 공식 train의 OOF에서 미리 결정되었으며
    # test의 분포/다른 행/통계로 재조정하지 않는다.
    # -----------------------------------------------------

    prediction = (
        weights[
            CB_EXPERIMENT_NAME
        ]
        * cb_prediction
        + weights[
            XGB_EXPERIMENT_NAME
        ]
        * xgb_prediction
        + weights[
            MLP_EXPERIMENT_NAME
        ]
        * mlp_prediction
    )

    validate_predictions(
        prediction,
        len(
            test
        ),
        EXPERIMENT_NAME,
    )

    print()
    print("[Ensemble prediction]")

    print(
        f"mean={prediction.mean():.6f}"
    )

    print(
        f"std ={prediction.std():.6f}"
    )

    print(
        f"min ={prediction.min():.6f}"
    )

    print(
        f"max ={prediction.max():.6f}"
    )

    # -----------------------------------------------------
    # Submission
    # -----------------------------------------------------

    submission = (
        submission.copy()
    )

    submission[
        TARGET
    ] = prediction

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Submission 저장 완료")
    print("=" * 80)
    print(f"- {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
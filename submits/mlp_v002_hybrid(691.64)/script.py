# script.py: MLP mlp_v003 inference
# train.py에서 저장한 Final bundle을 로드하여
# test.csv 각 행을 독립적으로 추론하고 submission.csv를 생성한다.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# 실험 설정
# =========================================================

EXPERIMENT_NAME = "mlp_v003_hybrid"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"


# =========================================================
# Device
# =========================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# Feature Engineering
# =========================================================

def make_count_state(
    balls: pd.Series,
    strikes: pd.Series,
) -> pd.Series:
    """
    현재 행의 balls_before / strikes_before만 이용해 count_state 생성.

    다른 test 행이나 test 전체 통계는 사용하지 않는다.
    """

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

    result = result.astype(object)
    result.loc[missing] = None

    return result


def build_feature_frame(
    df: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> pd.DataFrame:
    """
    Final preprocessor가 기대하는 Feature만 생성한다.

    test 전체 평균/빈도/순위/누적 통계는 계산하지 않는다.
    """

    X = df.copy()

    if (
        "count_state"
        in preprocessor["categorical_columns"]
    ):
        required = {
            "balls_before",
            "strikes_before",
        }

        missing = (
            required
            - set(X.columns)
        )

        if missing:
            raise ValueError(
                "count_state 생성에 필요한 컬럼이 없습니다: "
                + ", ".join(
                    sorted(missing)
                )
            )

        X["count_state"] = (
            make_count_state(
                X["balls_before"],
                X["strikes_before"],
            )
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
            "입력 데이터에 학습 Feature가 없습니다: "
            + ", ".join(
                missing_features
            )
        )

    return X[
        required_features
    ].copy()


# =========================================================
# Preprocessor transform
# =========================================================

def normalize_category_value(
    value: Any,
) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(
        value,
        (float, np.floating),
    ):
        if (
            np.isfinite(value)
            and float(value).is_integer()
        ):
            return str(
                int(value)
            )

    if isinstance(
        value,
        (int, np.integer),
    ):
        return str(
            int(value)
        )

    return str(value)


def transform_categorical(
    series: pd.Series,
    mapping: dict[str, int],
    unknown_index: int,
    missing_index: int,
) -> np.ndarray:
    output = np.empty(
        len(series),
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
            output[index] = (
                missing_index
            )
        else:
            output[index] = (
                mapping.get(
                    value,
                    unknown_index,
                )
            )

    return output


def transform_numeric(
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
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(
            float(stats["median"])
        )
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )

    values = (
        values
        - float(stats["mean"])
    ) / float(
        stats["std"]
    )

    values = values.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Numeric transform 후 "
            "NaN/Inf가 발생했습니다."
        )

    return values


def transform_dataframe(
    df: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    feature_frame = (
        build_feature_frame(
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
        preprocessor["unknown_index"]
    )

    missing_index = int(
        preprocessor["missing_index"]
    )

    categorical_arrays = []

    for column in categorical_columns:
        mapping = (
            preprocessor[
                "category_maps"
            ][column]
        )

        encoded = (
            transform_categorical(
                feature_frame[column],
                mapping,
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
            (len(df), 0),
            dtype=np.int64,
        )

    numeric_arrays = []

    for column in numeric_columns:
        stats = (
            preprocessor[
                "numeric_stats"
            ][column]
        )

        transformed = (
            transform_numeric(
                feature_frame[column],
                stats,
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
            (len(df), 0),
            dtype=np.float32,
        )

    return (
        numeric,
        categorical,
    )


# =========================================================
# Model
# =========================================================

class TabularMLP(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        embedding_dims: list[int],
        n_num_features: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        if len(
            cardinalities
        ) != len(
            embedding_dims
        ):
            raise ValueError(
                "cardinalities and embedding_dims length mismatch"
            )

        self.embeddings = nn.ModuleList(
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

        layers: list[
            nn.Module
        ] = []

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

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(
            *layers
        )

        self.head = nn.Linear(
            prev_dim,
            1,
        )

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[
            torch.Tensor
        ] = []

        if len(
            self.embeddings
        ) > 0:
            embedded = [
                emb(
                    categorical[
                        :,
                        index,
                    ]
                )
                for index, emb
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
            parts[0]
            if len(parts) == 1
            else torch.cat(
                parts,
                dim=1,
            )
        )

        x = self.backbone(
            x
        )

        return self.head(
            x
        ).squeeze(
            1
        )


def create_model(
    model_config: dict[str, Any],
) -> TabularMLP:
    return TabularMLP(
        cardinalities=model_config[
            "cardinalities"
        ],
        embedding_dims=model_config[
            "embedding_dims"
        ],
        n_num_features=model_config[
            "n_num_features"
        ],
        hidden_dims=model_config[
            "hidden_dims"
        ],
        dropout=model_config[
            "dropout"
        ],
    )


# =========================================================
# Prediction 검증
# =========================================================

def validate_predictions(
    predictions: np.ndarray,
    n_expected: int,
) -> None:
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if len(
        predictions
    ) != n_expected:
        raise RuntimeError(
            "예측 수와 test 행 수가 다릅니다."
        )

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "예측값에 NaN 또는 무한대가 있습니다."
        )

    if not (
        (
            predictions
            >= 0.0
        )
        & (
            predictions
            <= 1.0
        )
    ).all():
        raise RuntimeError(
            "예측값이 0~1 범위를 벗어났습니다."
        )


# =========================================================
# Main
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # 파일 확인
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 파일을 찾을 수 없습니다: "
            f"{MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "test.csv를 찾을 수 없습니다: "
            f"{TEST_PATH}"
        )

    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(
            "sample_submission.csv를 찾을 수 없습니다: "
            f"{SAMPLE_SUB_PATH}"
        )

    # -----------------------------------------------------
    # Bundle 로드
    # -----------------------------------------------------

    bundle = joblib.load(
        MODEL_PATH
    )

    required_keys = {
        "model_state_dict",
        "model_config",
        "preprocessor",
    }

    missing_keys = (
        required_keys
        - set(
            bundle.keys()
        )
    )

    if missing_keys:
        raise ValueError(
            "MLP bundle에 필요한 key가 없습니다: "
            + ", ".join(
                sorted(
                    missing_keys
                )
            )
        )

    model_config = (
        bundle[
            "model_config"
        ]
    )

    preprocessor = (
        bundle[
            "preprocessor"
        ]
    )

    required_preprocessor_keys = {
        "categorical_columns",
        "numeric_columns",
        "cardinalities",
        "category_maps",
        "numeric_stats",
        "unknown_index",
        "missing_index",
        "variant",
    }

    missing_preprocessor = (
        required_preprocessor_keys
        - set(
            preprocessor.keys()
        )
    )

    if missing_preprocessor:
        raise ValueError(
            "preprocessor에 필요한 key가 없습니다: "
            + ", ".join(
                sorted(
                    missing_preprocessor
                )
            )
        )

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------

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
            "sample_submission.csv에 "
            f"{ID}가 없습니다."
        )

    if TARGET not in submission.columns:
        raise ValueError(
            "sample_submission.csv에 "
            f"{TARGET}가 없습니다."
        )

    if len(
        test
    ) != len(
        submission
    ):
        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "행 수가 일치하지 않습니다."
        )

    if not (
        test[ID]
        .astype(str)
        .reset_index(
            drop=True
        )
        .equals(
            submission[
                ID
            ]
            .astype(str)
            .reset_index(
                drop=True
            )
        )
    ):
        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "row_id 순서가 일치하지 않습니다."
        )

    # -----------------------------------------------------
    # Final preprocessor transform
    # -----------------------------------------------------

    numeric, categorical = (
        transform_dataframe(
            test,
            preprocessor,
        )
    )

    # -----------------------------------------------------
    # 모델 생성 / weight 로드
    # -----------------------------------------------------

    device = get_device()

    model = create_model(
        model_config
    ).to(
        device
    )

    model.load_state_dict(
        bundle[
            "model_state_dict"
        ]
    )

    model.eval()

    print("=" * 70)
    print("MLP mlp_v003 Inference")
    print("=" * 70)
    print(
        f"Model path           : "
        f"{MODEL_PATH}"
    )
    print(
        f"Device               : "
        f"{device}"
    )
    print(
        f"Rows                 : "
        f"{len(test):,}"
    )
    print(
        f"Numerical features   : "
        f"{numeric.shape[1]}"
    )
    print(
        f"Categorical features : "
        f"{categorical.shape[1]}"
    )
    print(
        f"Variant              : "
        f"{preprocessor.get('variant')}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # 추론
    #
    # Feature는 각 행 자체의 입력값과
    # train으로 fit된 preprocessor만 사용한다.
    # -----------------------------------------------------

    batch_size = 8192

    predictions = []

    with torch.inference_mode():
        for start in range(
            0,
            len(test),
            batch_size,
        ):
            end = min(
                start + batch_size,
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

    predictions = np.concatenate(
        predictions
    ).astype(
        np.float64,
        copy=False,
    )

    validate_predictions(
        predictions,
        len(test),
    )

    print()
    print("=" * 70)
    print("Inference 결과")
    print("=" * 70)
    print(
        f"Prediction mean      : "
        f"{predictions.mean():.6f}"
    )
    print(
        f"Prediction std       : "
        f"{predictions.std():.6f}"
    )
    print(
        f"Prediction min       : "
        f"{predictions.min():.6f}"
    )
    print(
        f"Prediction max       : "
        f"{predictions.max():.6f}"
    )

    # -----------------------------------------------------
    # Submission 저장
    # -----------------------------------------------------

    submission = (
        submission.copy()
    )

    submission[
        TARGET
    ] = predictions

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Submission 저장 완료")
    print("=" * 70)
    print(
        f"저장 경로            : "
        f"{OUTPUT_PATH}"
    )
    print(
        f"행 수                : "
        f"{len(submission)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

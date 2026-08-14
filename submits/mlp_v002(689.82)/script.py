"""Inference script for the processed PyTorch tabular MLP baseline.

The competition server provides the real hidden test.csv at inference time. Therefore
this script does NOT consume a locally precomputed final/test/*.npy artifact. Instead,
it loads the final preprocessor stored inside the trained model bundle and applies the
same deterministic transformation rules to the runtime test.csv before inference.

Output
------
./output/submission.csv
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "mlp_v002"
VARIANT = "processed"

ID = "row_id"
TARGET = "control_success"

BATCH_SIZE = 8192
NUM_WORKERS = 0

UNKNOWN_TOKEN = "__UNKNOWN__"
MISSING_TOKEN = "__MISSING__"

ALWAYS_DROP_COLUMNS = [
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "score_diff_home",
]
COUNT_COLUMNS = ["balls_before", "strikes_before"]


# =========================================================
# 경로
# =========================================================
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
# MLP model — train.py와 동일한 구조
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

        if len(cardinalities) != len(embedding_dims):
            raise ValueError(
                "cardinalities and embedding_dims length mismatch"
            )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=int(cardinality),
                    embedding_dim=int(dim),
                    padding_idx=0,
                )
                for cardinality, dim in zip(
                    cardinalities,
                    embedding_dims,
                )
            ]
        )

        embedding_total = int(sum(embedding_dims))
        input_dim = embedding_total + int(n_num_features)

        if input_dim <= 0:
            raise ValueError("MLP input feature가 없습니다.")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        # 추론에서는 state_dict로 덮어쓰지만 train.py와 동일한 정의를 유지한다.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=math.sqrt(5),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if len(self.embeddings) > 0:
            embedded = [
                emb(categorical[:, index])
                for index, emb in enumerate(self.embeddings)
            ]
            parts.append(torch.cat(embedded, dim=1))

        if numeric.shape[1] > 0:
            parts.append(numeric)

        if not parts:
            raise ValueError("MLP input feature가 없습니다.")

        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        x = self.backbone(x)
        return self.head(x).squeeze(1)


def create_model(config: dict[str, Any]) -> TabularMLP:
    required = {
        "cardinalities",
        "embedding_dims",
        "n_num_features",
        "hidden_dims",
        "dropout",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing model_config keys: {missing}")

    return TabularMLP(
        cardinalities=[int(x) for x in config["cardinalities"]],
        embedding_dims=[int(x) for x in config["embedding_dims"]],
        n_num_features=int(config["n_num_features"]),
        hidden_dims=[int(x) for x in config["hidden_dims"]],
        dropout=float(config["dropout"]),
    )


# =========================================================
# Final preprocessor — mlp_preprocess.py와 동일한 변환 규칙
# =========================================================
def add_count_state(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"balls_before", "strikes_before"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Count columns are missing: {missing}")

    frame = frame.copy()
    balls = pd.to_numeric(frame["balls_before"], errors="coerce")
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce")

    invalid = (
        balls.isna()
        | strikes.isna()
        | ~balls.isin([0, 1, 2, 3])
        | ~strikes.isin([0, 1, 2])
    )
    if invalid.any():
        examples = frame.loc[
            invalid,
            ["balls_before", "strikes_before"],
        ].head(5)
        raise ValueError(
            "Invalid ball/strike count detected:\n"
            + examples.to_string(index=False)
        )

    frame["count_state"] = (
        balls.astype("int8").astype(str)
        + "-"
        + strikes.astype("int8").astype(str)
    )
    return frame


def prepare_frame(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    frame = add_count_state(frame)

    drop_columns = list(ALWAYS_DROP_COLUMNS)
    if variant == "processed":
        drop_columns.extend(COUNT_COLUMNS)
    elif variant != "hybrid":
        raise ValueError(f"Unknown preprocessing variant: {variant!r}")

    missing = sorted(set(drop_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    return frame.drop(columns=drop_columns)


def categorical_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_TOKEN)


def validate_preprocessor(
    preprocessor: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    required = {
        "categorical_columns",
        "numeric_columns",
        "category_maps",
        "cardinalities",
        "numeric_stats",
        "unknown_index",
        "missing_index",
        "variant",
    }
    missing = sorted(required - set(preprocessor))
    if missing:
        raise ValueError(f"Missing preprocessor keys: {missing}")

    if preprocessor["variant"] != VARIANT:
        raise ValueError(
            "Variant mismatch: "
            f"script.py VARIANT={VARIANT!r}, "
            f"model preprocessor={preprocessor['variant']!r}"
        )

    categorical_columns = list(preprocessor["categorical_columns"])
    numeric_columns = list(preprocessor["numeric_columns"])

    expected_cardinalities = [
        int(preprocessor["cardinalities"][column])
        for column in categorical_columns
    ]
    model_cardinalities = [
        int(value) for value in model_config["cardinalities"]
    ]

    if expected_cardinalities != model_cardinalities:
        raise ValueError(
            "Model/preprocessor categorical cardinality mismatch"
        )

    if len(numeric_columns) != int(model_config["n_num_features"]):
        raise ValueError(
            "Model/preprocessor numeric feature count mismatch: "
            f"preprocessor={len(numeric_columns)}, "
            f"model={model_config['n_num_features']}"
        )


def validate_test_schema(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> None:
    required = set(preprocessor["categorical_columns"])
    required.update(preprocessor["numeric_columns"])
    required.add(ID)

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Runtime test is missing model features: {missing}"
        )

    if frame[ID].isna().any():
        raise ValueError("test.csv contains missing row_id")

    if frame[ID].astype(str).duplicated().any():
        raise ValueError("test.csv contains duplicate row_id")


def transform_categorical(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> np.ndarray:
    categorical_columns = list(preprocessor["categorical_columns"])

    if not categorical_columns:
        return np.empty((len(frame), 0), dtype=np.int64)

    columns: list[np.ndarray] = []
    unknown_index = int(preprocessor["unknown_index"])

    for column in categorical_columns:
        mapping = preprocessor["category_maps"][column]
        encoded = (
            categorical_strings(frame[column])
            .map(mapping)
            .fillna(unknown_index)
            .astype("int64")
            .to_numpy()
        )

        cardinality = int(preprocessor["cardinalities"][column])
        if len(encoded) > 0:
            min_value = int(np.min(encoded))
            max_value = int(np.max(encoded))
            if min_value < 0 or max_value >= cardinality:
                raise ValueError(
                    f"Categorical index out of range for {column}: "
                    f"min={min_value}, max={max_value}, "
                    f"cardinality={cardinality}"
                )

        columns.append(encoded)

    return np.column_stack(columns).astype(np.int64, copy=False)


def transform_numeric(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> np.ndarray:
    numeric_columns = list(preprocessor["numeric_columns"])

    if not numeric_columns:
        return np.empty((len(frame), 0), dtype=np.float32)

    columns: list[np.ndarray] = []

    for column in numeric_columns:
        stats = preprocessor["numeric_stats"][column]
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(float(stats["median"]))
            .astype("float64")
        )
        standardized = (
            values - float(stats["mean"])
        ) / float(stats["std"])
        columns.append(
            standardized.to_numpy(dtype=np.float32)
        )

    numeric = np.column_stack(columns).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite numeric value found in runtime test")

    return numeric


# =========================================================
# Runtime Dataset
# =========================================================
class InferenceDataset(Dataset):
    def __init__(
        self,
        categorical: np.ndarray,
        numeric: np.ndarray,
        row_ids: np.ndarray,
    ) -> None:
        if categorical.ndim != 2:
            raise ValueError("categorical must be 2-D")
        if numeric.ndim != 2:
            raise ValueError("numeric must be 2-D")

        lengths = {
            len(categorical),
            len(numeric),
            len(row_ids),
        }
        if len(lengths) != 1:
            raise ValueError(
                "Inference array length mismatch: "
                f"categorical={len(categorical)}, "
                f"numeric={len(numeric)}, "
                f"row_id={len(row_ids)}"
            )

        self.categorical = categorical
        self.numeric = numeric
        self.row_ids = row_ids

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "categorical": torch.from_numpy(
                self.categorical[index]
            ).long(),
            "numeric": torch.from_numpy(
                self.numeric[index]
            ).float(),
            "row_id": str(self.row_ids[index]),
        }


# =========================================================
# Model / Prediction
# =========================================================
def load_model_bundle(
    device: torch.device,
) -> tuple[TabularMLP, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    required = {
        "model_state_dict",
        "model_config",
        "preprocessor",
        "metadata",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Missing model bundle keys: {missing}")

    metadata = bundle["metadata"]
    bundle_experiment = metadata.get("experiment_name")
    if bundle_experiment != EXPERIMENT_NAME:
        raise ValueError(
            "Experiment mismatch: "
            f"script={EXPERIMENT_NAME!r}, "
            f"bundle={bundle_experiment!r}"
        )

    bundle_variant = metadata.get("variant")
    if bundle_variant != VARIANT:
        raise ValueError(
            "Variant mismatch: "
            f"script={VARIANT!r}, bundle={bundle_variant!r}"
        )

    model_config = bundle["model_config"]
    preprocessor = bundle["preprocessor"]
    validate_preprocessor(preprocessor, model_config)

    model = create_model(model_config)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, preprocessor, metadata, model_config


@torch.inference_mode()
def predict_mlp(
    model: nn.Module,
    dataset: InferenceDataset,
    device: torch.device,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    row_ids: list[str] = []
    predictions: list[np.ndarray] = []

    model.eval()

    for batch in loader:
        categorical = batch["categorical"].to(
            device,
            non_blocking=True,
        )
        numeric = batch["numeric"].to(
            device,
            non_blocking=True,
        )

        logits = model(categorical, numeric)
        probability = torch.sigmoid(logits)

        row_ids.extend([str(value) for value in batch["row_id"]])
        predictions.append(
            probability.detach().cpu().numpy()
        )

    prediction = np.concatenate(predictions).astype(
        np.float64,
        copy=False,
    )

    if len(prediction) != len(dataset):
        raise ValueError(
            "Prediction length mismatch: "
            f"pred={len(prediction)}, dataset={len(dataset)}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite MLP prediction detected")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("MLP prediction is outside [0, 1]")

    result = pd.DataFrame(
        {
            ID: np.asarray(row_ids, dtype=str),
            "prediction": prediction,
        }
    )

    if result[ID].duplicated().any():
        raise ValueError("Duplicate row_id detected in MLP predictions")

    return result


# =========================================================
# Submission
# =========================================================
def validate_submission_inputs(
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    for name, frame in [
        ("test.csv", test),
        ("sample_submission.csv", sample_submission),
    ]:
        if ID not in frame.columns:
            raise ValueError(f"{name} does not contain {ID!r}")

        if frame[ID].isna().any():
            raise ValueError(f"{name} contains missing row_id")

        if frame[ID].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate row_id")

    if TARGET not in sample_submission.columns:
        raise ValueError(
            f"sample_submission.csv does not contain {TARGET!r}"
        )

    test_ids = set(test[ID].astype(str))
    sample_ids = set(sample_submission[ID].astype(str))

    if test_ids != sample_ids:
        only_test = list(test_ids - sample_ids)[:5]
        only_sample = list(sample_ids - test_ids)[:5]
        raise ValueError(
            "test/sample_submission row_id sets differ. "
            f"only_test={only_test}, only_sample={only_sample}"
        )


def build_submission(
    sample_submission: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    sample = sample_submission.copy()
    sample[ID] = sample[ID].astype(str)

    pred = prediction_frame.copy()
    pred[ID] = pred[ID].astype(str)

    submission = (
        sample[[ID]]
        .merge(
            pred,
            on=ID,
            how="left",
            validate="one_to_one",
        )
        .rename(columns={"prediction": TARGET})
    )

    if submission[TARGET].isna().any():
        missing = submission.loc[
            submission[TARGET].isna(),
            ID,
        ].head(5).tolist()
        raise ValueError(
            "Missing predictions after row_id merge: "
            f"examples={missing}"
        )

    prediction = submission[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Submission contains non-finite predictions")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("Submission prediction is outside [0, 1]")

    return submission[[ID, TARGET]]


# =========================================================
# Main
# =========================================================
def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(
            f"Sample submission not found: {SAMPLE_SUB_PATH}"
        )

    device = get_device()

    print("=" * 70)
    print(EXPERIMENT_NAME)
    print("=" * 70)
    print(f"Variant       : {VARIANT}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Model Path    : {MODEL_PATH}")
    print(f"Test Path     : {TEST_PATH}")
    print("=" * 70)

    model, preprocessor, metadata, model_config = load_model_bundle(device)

    test_raw = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample_submission = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    validate_submission_inputs(test_raw, sample_submission)

    test = prepare_frame(test_raw, VARIANT)
    validate_test_schema(test, preprocessor)

    categorical = transform_categorical(test, preprocessor)
    numeric = transform_numeric(test, preprocessor)
    row_ids = test[ID].astype(str).to_numpy(dtype=str)

    # 저장된 Final model schema와 런타임 변환 결과가 정확히 일치하는지 검증한다.
    if numeric.shape[1] != int(model_config["n_num_features"]):
        raise ValueError(
            "Runtime numeric feature count mismatch: "
            f"array={numeric.shape[1]}, "
            f"model={model_config['n_num_features']}"
        )

    if categorical.shape[1] != len(model_config["cardinalities"]):
        raise ValueError(
            "Runtime categorical feature count mismatch: "
            f"array={categorical.shape[1]}, "
            f"model={len(model_config['cardinalities'])}"
        )

    dataset = InferenceDataset(
        categorical=categorical,
        numeric=numeric,
        row_ids=row_ids,
    )

    prediction_frame = predict_mlp(
        model,
        dataset,
        device,
    )

    # Runtime test의 row_id 집합과 MLP prediction의 row_id 집합을 다시 검증한다.
    if set(test_raw[ID].astype(str)) != set(prediction_frame[ID]):
        raise ValueError("Runtime test / prediction row_id sets differ")

    submission = build_submission(
        sample_submission,
        prediction_frame,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pred = submission[TARGET].to_numpy(dtype=np.float64)

    print()
    print("=" * 70)
    print("MLP inference 완료")
    print("=" * 70)
    print(f"Rows          : {len(submission):,}")
    print(f"Numerical     : {numeric.shape[1]}")
    print(f"Categorical   : {categorical.shape[1]}")
    print(f"Best Epoch    : {metadata.get('best_epoch')}")
    print(f"Prediction mean: {np.mean(pred):.6f}")
    print(f"Prediction std : {np.std(pred):.6f}")
    print(f"Prediction min : {np.min(pred):.6f}")
    print(f"Prediction max : {np.max(pred):.6f}")
    print(f"Output Path   : {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

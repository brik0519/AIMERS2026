"""Train a PyTorch tabular MLP from NumPy artifacts created by mlp_preprocess.py.

Workflow
--------
1) Validation preprocessing
   python mlp_preprocess.py --mode validation --variant processed

2) Validation training
   python train.py --mode validation

3) After selecting the variant, final preprocessing
   python mlp_preprocess.py --mode final --variant processed

4) Final training using the best validation epoch
   python train.py --mode final --best-epoch <BEST_EPOCH>

The script intentionally does not read train.csv/test.csv. Feature engineering and
preprocessing are owned by mlp_preprocess.py; this file only consumes the saved
NumPy arrays and preprocessor.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlp_dataset import TabularNpyDataset


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "mlp_v002"
VARIANT = "processed"  # "processed" or "hybrid"

ID = "row_id"
TARGET = "control_success"
VALIDATION_SEASON = 2024

SEED = 42
BATCH_SIZE = 4096
VALID_BATCH_SIZE = 4096
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIMS = [256, 256, 128]
DROPOUT = 0.15
NUM_WORKERS = 0


# =========================================================
# 경로
# =========================================================
DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

MLP_DATA_DIR = DATA_DIR / "mlp" / VARIANT
VALIDATION_DATA_DIR = MLP_DATA_DIR / "validation"
FINAL_DATA_DIR = MLP_DATA_DIR / "final"


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["validation", "final"],
        default="validation",
        help=(
            "validation: 2019-2023 train / 2024 validation; "
            "final: train on all preprocessed training rows"
        ),
    )
    parser.add_argument(
        "--best-epoch",
        type=int,
        default=None,
        help="Required in final mode. Use the best epoch from validation.",
    )
    return parser.parse_args()


# =========================================================
# Reproducibility / Device
# =========================================================
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        # mlp_v001과 동일한 속도 우선 설정.
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cpu")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# Metric
# =========================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    if not np.isfinite(y_true).all():
        raise ValueError("Non-finite target detected during metric calculation")
    if not np.isfinite(y_pred).all():
        raise ValueError("Non-finite prediction detected during metric calculation")

    y_pred = np.clip(y_pred, 0.0, 1.0)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0:
        bss = 0.0
    else:
        bss = max(
            0.0,
            100000.0 * (1.0 - brier / baseline_brier),
        )

    eps = 1e-7
    p = np.clip(y_pred, eps, 1.0 - eps)
    logloss = float(
        -np.mean(
            y_true * np.log(p)
            + (1.0 - y_true) * np.log(1.0 - p)
        )
    )

    return {
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": float(bss),
        "positive_rate": positive_rate,
        "logloss": logloss,
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


# =========================================================
# Preprocessor artifact
# =========================================================
def load_preprocessor(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {path}\n"
            "Run mlp_preprocess.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        preprocessor = json.load(file)

    required_keys = {
        "categorical_columns",
        "numeric_columns",
        "cardinalities",
        "category_maps",
        "numeric_stats",
        "unknown_index",
        "missing_index",
        "variant",
    }
    missing = sorted(required_keys - set(preprocessor))
    if missing:
        raise ValueError(
            f"Missing keys in preprocessor.json: {missing}"
        )

    if preprocessor["variant"] != VARIANT:
        raise ValueError(
            "Variant mismatch: "
            f"train.py VARIANT={VARIANT!r}, "
            f"preprocessor={preprocessor['variant']!r}"
        )

    return preprocessor


def validate_dataset_schema(
    dataset: TabularNpyDataset,
    preprocessor: dict[str, Any],
    split_name: str,
    require_target: bool,
) -> None:
    if dataset.numeric.ndim != 2:
        raise ValueError(
            f"{split_name}: numeric.npy must be 2-D, "
            f"got {dataset.numeric.shape}"
        )
    if dataset.categorical.ndim != 2:
        raise ValueError(
            f"{split_name}: categorical.npy must be 2-D, "
            f"got {dataset.categorical.shape}"
        )

    expected_numeric = len(preprocessor["numeric_columns"])
    expected_categorical = len(preprocessor["categorical_columns"])

    if dataset.numeric.shape[1] != expected_numeric:
        raise ValueError(
            f"{split_name}: numeric feature count mismatch: "
            f"array={dataset.numeric.shape[1]}, "
            f"preprocessor={expected_numeric}"
        )

    if dataset.categorical.shape[1] != expected_categorical:
        raise ValueError(
            f"{split_name}: categorical feature count mismatch: "
            f"array={dataset.categorical.shape[1]}, "
            f"preprocessor={expected_categorical}"
        )

    if require_target and dataset.targets is None:
        raise ValueError(f"{split_name}: target.npy is required")

    if dataset.targets is not None:
        targets = np.asarray(dataset.targets)
        if not np.isfinite(targets).all():
            raise ValueError(f"{split_name}: non-finite target detected")
        if not np.isin(targets, [0.0, 1.0]).all():
            values = np.unique(targets)[:10]
            raise ValueError(
                f"{split_name}: target must be binary 0/1; "
                f"examples={values.tolist()}"
            )

    for index, column in enumerate(preprocessor["categorical_columns"]):
        cardinality = int(preprocessor["cardinalities"][column])
        values = dataset.categorical[:, index]
        if len(values) == 0:
            continue
        min_value = int(np.min(values))
        max_value = int(np.max(values))
        if min_value < 0 or max_value >= cardinality:
            raise ValueError(
                f"{split_name}: categorical index out of range for "
                f"{column}: min={min_value}, max={max_value}, "
                f"cardinality={cardinality}"
            )


def print_unknown_ratios(
    dataset: TabularNpyDataset,
    preprocessor: dict[str, Any],
) -> None:
    columns = preprocessor["categorical_columns"]
    unknown_index = int(preprocessor["unknown_index"])

    if not columns:
        return

    print()
    print("[Validation unknown category ratio]")
    for index, column in enumerate(columns):
        values = dataset.categorical[:, index]
        ratio = float(np.mean(values == unknown_index))
        print(f"{column:<22}: {ratio:.6%}")


# =========================================================
# Dataset / Loader
# =========================================================
def make_loader(
    dataset: TabularNpyDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# =========================================================
# Model
# =========================================================
def choose_embedding_dim(n_categories: int) -> int:
    n_categories = max(1, int(n_categories))
    dim = int(round(4.0 * (n_categories ** 0.25)))
    return int(min(32, max(2, dim)))


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

        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        x = self.backbone(x)
        return self.head(x).squeeze(1)


def build_model_config(
    preprocessor: dict[str, Any],
) -> dict[str, Any]:
    categorical_columns = preprocessor["categorical_columns"]
    cardinalities = [
        int(preprocessor["cardinalities"][column])
        for column in categorical_columns
    ]
    embedding_dims = [
        choose_embedding_dim(cardinality)
        for cardinality in cardinalities
    ]

    return {
        "cardinalities": cardinalities,
        "embedding_dims": embedding_dims,
        "n_num_features": len(preprocessor["numeric_columns"]),
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": float(DROPOUT),
    }


def create_model(config: dict[str, Any]) -> TabularMLP:
    return TabularMLP(
        cardinalities=config["cardinalities"],
        embedding_dims=config["embedding_dims"],
        n_num_features=config["n_num_features"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )


# =========================================================
# Train / Evaluate
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        categorical = batch["categorical"].to(
            device,
            non_blocking=True,
        )
        numeric = batch["numeric"].to(
            device,
            non_blocking=True,
        )
        target = batch["target"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical, numeric)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        batch_n = int(target.shape[0])
        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

    return total_loss / max(1, total_n)


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    row_ids: list[str] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for batch in loader:
        if "target" not in batch:
            raise ValueError("Validation loader must contain target")

        categorical = batch["categorical"].to(
            device,
            non_blocking=True,
        )
        numeric = batch["numeric"].to(
            device,
            non_blocking=True,
        )

        logits = model(categorical, numeric)
        prediction = torch.sigmoid(logits)

        row_ids.extend([str(value) for value in batch["row_id"]])
        targets.append(
            batch["target"].detach().cpu().numpy()
        )
        predictions.append(
            prediction.detach().cpu().numpy()
        )

    return (
        np.asarray(row_ids, dtype=str),
        np.concatenate(targets).astype(np.float64, copy=False),
        np.concatenate(predictions).astype(np.float64, copy=False),
    )


# =========================================================
# Validation fit
# =========================================================
def fit_validation_model(
    train_dataset: TabularNpyDataset,
    valid_dataset: TabularNpyDataset,
    preprocessor: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    train_loader = make_loader(
        train_dataset,
        BATCH_SIZE,
        True,
        device,
    )
    valid_loader = make_loader(
        valid_dataset,
        VALID_BATCH_SIZE,
        False,
        device,
    )

    model_config = build_model_config(preprocessor)
    model = create_model(model_config).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=LEARNING_RATE * 0.05,
    )

    best_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_row_ids: np.ndarray | None = None
    best_targets: np.ndarray | None = None
    best_pred: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None

    patience_count = 0
    history: list[dict[str, Any]] = []

    print()
    print("=" * 70)
    print("MLP Validation 학습 시작")
    print("=" * 70)
    print(f"Experiment   : {EXPERIMENT_NAME}")
    print(f"Variant      : {VARIANT}")
    print(f"Device       : {device}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Max epochs   : {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Hidden dims  : {HIDDEN_DIMS}")
    print(f"Dropout      : {DROPOUT}")

    print()
    print("[Embedding schema]")
    for column, cardinality, dim in zip(
        preprocessor["categorical_columns"],
        model_config["cardinalities"],
        model_config["embedding_dims"],
    ):
        print(
            f"{column:<22} "
            f"cardinality={cardinality:<6} "
            f"dim={dim}"
        )

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        row_ids, y_valid, valid_pred = evaluate_loader(
            model,
            valid_loader,
            device,
        )
        metrics = calculate_metrics(y_valid, valid_pred)

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        seconds = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "val_brier": metrics["brier"],
                "val_bss": metrics["bss"],
                "val_logloss": metrics["logloss"],
                "learning_rate": current_lr,
                "seconds": seconds,
            }
        )

        improved = metrics["brier"] < best_brier - 1e-10
        if improved:
            best_brier = metrics["brier"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_row_ids = row_ids.copy()
            best_targets = y_valid.copy()
            best_pred = valid_pred.copy()
            best_metrics = dict(metrics)
            patience_count = 0
            marker = " *BEST*"
        else:
            patience_count += 1
            marker = ""

        print(
            f"Epoch {epoch:02d} | "
            f"train BCE={train_loss:.6f} | "
            f"val Brier={metrics['brier']:.8f} | "
            f"BSS={metrics['bss']:.2f} | "
            f"logloss={metrics['logloss']:.6f} | "
            f"lr={current_lr:.2e} | "
            f"{seconds:.1f}s{marker}"
        )

        if patience_count >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    elapsed = time.time() - start_time

    if (
        best_state is None
        or best_row_ids is None
        or best_targets is None
        or best_pred is None
        or best_metrics is None
        or best_epoch <= 0
    ):
        raise RuntimeError("No valid best checkpoint was produced")

    print()
    print("=" * 70)
    print("MLP Validation Best")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Brier Score      : {best_metrics['brier']:.8f}")
    print(f"Baseline Brier   : {best_metrics['baseline_brier']:.8f}")
    print(f"Validation Score : {best_metrics['bss']:.2f}")
    print(f"LogLoss          : {best_metrics['logloss']:.8f}")
    print(f"Prediction mean  : {best_metrics['prediction_mean']:.6f}")
    print(f"Prediction std   : {best_metrics['prediction_std']:.6f}")
    print(f"Prediction min   : {best_metrics['prediction_min']:.6f}")
    print(f"Prediction max   : {best_metrics['prediction_max']:.6f}")
    print(f"Total time       : {elapsed:.1f}s")

    return {
        "model_config": model_config,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "row_ids": best_row_ids,
        "targets": best_targets,
        "best_prediction": best_pred,
        "best_metrics": best_metrics,
        "history": history,
        "elapsed": elapsed,
    }


# =========================================================
# Save validation
# =========================================================
def save_validation(
    result: dict[str, Any],
    preprocessor: dict[str, Any],
) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)

    prediction_frame = pd.DataFrame(
        {
            ID: result["row_ids"],
            "target": result["targets"],
            "prediction": result["best_prediction"],
        }
    )
    prediction_frame.to_csv(
        VALIDATION_DIR / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR / "validation.npz",
        row_id=result["row_ids"],
        target=result["targets"],
        prediction=result["best_prediction"],
    )

    pd.DataFrame(result["history"]).to_csv(
        VALIDATION_DIR / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "TabularMLP",
        "variant": VARIANT,
        "validation_season": VALIDATION_SEASON,
        "validation": "2019-2023 Train / 2024 Validation",
        "categorical_columns": preprocessor["categorical_columns"],
        "numeric_columns": preprocessor["numeric_columns"],
        "n_features": (
            len(preprocessor["categorical_columns"])
            + len(preprocessor["numeric_columns"])
        ),
        "cardinalities": preprocessor["cardinalities"],
        "model_config": result["model_config"],
        "best_epoch": result["best_epoch"],
        "metrics": result["best_metrics"],
        "training_seconds": result["elapsed"],
        "preprocessing": {
            "source": "mlp_preprocess.py",
            "variant": VARIANT,
            "count_state": True,
            "raw_count_features_retained": VARIANT == "hybrid",
            "unknown_index": preprocessor["unknown_index"],
            "missing_index": preprocessor["missing_index"],
        },
        "training_config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "valid_batch_size": VALID_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "loss": "BCEWithLogitsLoss",
            "scheduler": "CosineAnnealingLR",
            "primary_metric": "Brier Score",
        },
    }

    with (VALIDATION_DIR / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# =========================================================
# Final fit
# =========================================================
def fit_final_model(
    train_dataset: TabularNpyDataset,
    preprocessor: dict[str, Any],
    best_epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    if best_epoch <= 0:
        raise ValueError("best_epoch must be >= 1")

    loader = make_loader(
        train_dataset,
        BATCH_SIZE,
        True,
        device,
    )

    model_config = build_model_config(preprocessor)
    model = create_model(model_config).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, best_epoch),
        eta_min=LEARNING_RATE * 0.05,
    )

    print()
    print("=" * 70)
    print("전체 데이터 재학습 시작")
    print("=" * 70)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Variant    : {VARIANT}")
    print(f"Epochs     : {best_epoch}")

    start_time = time.time()
    history: list[dict[str, Any]] = []

    for epoch in range(1, best_epoch + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        seconds = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "learning_rate": current_lr,
                "seconds": seconds,
            }
        )

        print(
            f"Epoch {epoch:02d}/{best_epoch:02d} | "
            f"train BCE={train_loss:.6f} | "
            f"lr={current_lr:.2e} | "
            f"{seconds:.1f}s"
        )

    elapsed = time.time() - start_time
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    return {
        "model_config": model_config,
        "state_dict": state_dict,
        "history": history,
        "elapsed": elapsed,
    }


def load_validation_best_epoch() -> int | None:
    metadata_path = VALIDATION_DIR / "metadata.json"
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    best_epoch = metadata.get("best_epoch")
    if best_epoch is None:
        return None

    return int(best_epoch)


def save_final_model(
    result: dict[str, Any],
    preprocessor: dict[str, Any],
    best_epoch: int,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        raise FileExistsError(f"Model already exists: {MODEL_PATH}")

    bundle = {
        "model_state_dict": result["state_dict"],
        "model_config": result["model_config"],
        "features": (
            preprocessor["categorical_columns"]
            + preprocessor["numeric_columns"]
        ),
        "cat_cols": preprocessor["categorical_columns"],
        "num_cols": preprocessor["numeric_columns"],
        "preprocessor": preprocessor,
        "metadata": {
            "experiment_name": EXPERIMENT_NAME,
            "model_type": "TabularMLP",
            "variant": VARIANT,
            "validation": "2019-2023 Train / 2024 Validation",
            "best_epoch": best_epoch,
            "training_config": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "hidden_dims": HIDDEN_DIMS,
                "dropout": DROPOUT,
                "loss": "BCEWithLogitsLoss",
                "scheduler": "CosineAnnealingLR",
                "primary_metric": "Brier Score",
            },
            "preprocessing": {
                "source": "mlp_preprocess.py",
                "variant": VARIANT,
                "count_state": True,
                "raw_count_features_retained": VARIANT == "hybrid",
                "unknown_index": preprocessor["unknown_index"],
                "missing_index": preprocessor["missing_index"],
            },
            "full_training_seconds": result["elapsed"],
        },
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)

    metadata_path = MODEL_DIR / f"{EXPERIMENT_NAME}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            bundle["metadata"],
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 현재 실행 중인 train.py를 모델과 함께 보관한다.
    source_path = Path(__file__).resolve()
    destination = MODEL_DIR / f"{EXPERIMENT_NAME}_train.py"
    shutil.copy2(source_path, destination)

    # Final preprocessing artifact도 모델과 함께 보관한다.
    preprocessor_copy = MODEL_DIR / f"{EXPERIMENT_NAME}_preprocessor.json"
    with preprocessor_copy.open("w", encoding="utf-8") as file:
        json.dump(preprocessor, file, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("Final model 저장 완료")
    print("=" * 70)
    print(f"Model Bundle : {MODEL_PATH}")
    print(f"Metadata     : {metadata_path}")
    print(f"Train Code   : {destination}")
    print(f"Preprocessor : {preprocessor_copy}")
    print(f"Best Epoch   : {best_epoch}")


# =========================================================
# Main
# =========================================================
def run_validation(device: torch.device) -> None:
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation already exists: {VALIDATION_DIR}"
        )

    preprocessor_path = VALIDATION_DATA_DIR / "preprocessor.json"
    preprocessor = load_preprocessor(preprocessor_path)

    train_dataset = TabularNpyDataset(
        VALIDATION_DATA_DIR / "train"
    )
    valid_dataset = TabularNpyDataset(
        VALIDATION_DATA_DIR / "valid"
    )

    validate_dataset_schema(
        train_dataset,
        preprocessor,
        "validation/train",
        require_target=True,
    )
    validate_dataset_schema(
        valid_dataset,
        preprocessor,
        "validation/valid",
        require_target=True,
    )

    print("=" * 70)
    print(EXPERIMENT_NAME)
    print("=" * 70)
    print(f"Mode          : validation")
    print(f"Variant       : {VARIANT}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Train rows    : {len(train_dataset):,}")
    print(f"Valid rows    : {len(valid_dataset):,}")
    print(f"Numerical     : {train_dataset.numeric.shape[1]}")
    print(f"Categorical   : {train_dataset.categorical.shape[1]}")
    print(
        "Features      : "
        f"{len(preprocessor['numeric_columns']) + len(preprocessor['categorical_columns'])}"
    )
    print("=" * 70)

    print_unknown_ratios(valid_dataset, preprocessor)

    result = fit_validation_model(
        train_dataset,
        valid_dataset,
        preprocessor,
        device,
    )
    save_validation(result, preprocessor)

    print()
    print("=" * 70)
    print("Validation 저장 완료")
    print("=" * 70)
    print(f"Validation Dir : {VALIDATION_DIR}")
    print(f"Best Epoch     : {result['best_epoch']}")
    print(
        "Validation Brier: "
        f"{result['best_metrics']['brier']:.8f}"
    )
    print(
        "Validation BSS  : "
        f"{result['best_metrics']['bss']:.2f}"
    )


def run_final(
    device: torch.device,
    best_epoch: int | None,
) -> None:
    if MODEL_PATH.exists():
        raise FileExistsError(f"Model already exists: {MODEL_PATH}")

    if best_epoch is None:
        best_epoch = load_validation_best_epoch()

    if best_epoch is None:
        raise ValueError(
            "Final mode requires --best-epoch, or an existing "
            f"{VALIDATION_DIR / 'metadata.json'}"
        )

    if best_epoch <= 0:
        raise ValueError("best_epoch must be >= 1")

    preprocessor_path = FINAL_DATA_DIR / "preprocessor.json"
    preprocessor = load_preprocessor(preprocessor_path)

    train_dataset = TabularNpyDataset(FINAL_DATA_DIR / "train")
    validate_dataset_schema(
        train_dataset,
        preprocessor,
        "final/train",
        require_target=True,
    )

    print("=" * 70)
    print(EXPERIMENT_NAME)
    print("=" * 70)
    print(f"Mode          : final")
    print(f"Variant       : {VARIANT}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Train rows    : {len(train_dataset):,}")
    print(f"Best epoch    : {best_epoch}")
    print(f"Numerical     : {train_dataset.numeric.shape[1]}")
    print(f"Categorical   : {train_dataset.categorical.shape[1]}")
    print("=" * 70)

    # Validation과 Final의 categorical vocabulary/cardinality가 달라질 수 있으므로
    # Final preprocessor로 모델 구조를 새로 생성한다.
    seed_everything()
    result = fit_final_model(
        train_dataset,
        preprocessor,
        best_epoch,
        device,
    )
    save_final_model(
        result,
        preprocessor,
        best_epoch,
    )



def main() -> None:
    args = parse_args()

    seed_everything()
    device = get_device()

    if args.mode == "validation":
        run_validation(device)
    else:
        run_final(device, args.best_epoch)


if __name__ == "__main__":
    main()

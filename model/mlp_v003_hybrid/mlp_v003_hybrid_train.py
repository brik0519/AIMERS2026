"""Train mlp_v003 from NumPy artifacts created by data/mlp/mlp_preprocess.py.

Workflow
--------
1) Expanding preprocessing
   python data/mlp/mlp_preprocess.py --mode expanding --variant processed

2) 4-Fold Expanding Window validation
   python train.py --mode expanding

3) Final preprocessing using official train only
   python data/mlp/mlp_preprocess.py --mode final --variant processed

4) Final training
   python train.py --mode final

The training script never derives features or statistics from test.csv.
It only consumes preprocessed NumPy arrays and preprocessor.json.
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "mlp_v003_hybrid"
VARIANT = "hybrid"  # "processed" or "hybrid"

ID = "row_id"
TARGET = "control_success"

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

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_ROOT / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

MLP_DATA_DIR = DATA_DIR / "mlp" / VARIANT
EXPANDING_DATA_DIR = MLP_DATA_DIR / "expanding"
FINAL_DATA_DIR = MLP_DATA_DIR / "final"

FINAL_EPOCH_STRATEGY = "latest_fold"

EXPANDING_FOLDS = [
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "validation_season": 2024,
    },
]


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["expanding", "final"],
        default="expanding",
        help=(
            "expanding: 4-Fold Expanding Window Validation; "
            "final: 전체 train artifact로 Final Model 학습"
        ),
    )
    parser.add_argument(
        "--best-epoch",
        type=int,
        default=None,
        help=(
            "final mode에서 선택적으로 지정. 생략하면 "
            "Expanding metadata의 latest_fold best epoch을 사용."
        ),
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
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")

# =========================================================
# Class
# =========================================================
class TabularNpyDataset(Dataset):
    """Memory-mapped numeric/categorical arrays with optional targets."""

    def __init__(self, split_dir: str | Path) -> None:
        self.split_dir = Path(split_dir)
        self.numeric = np.load(
            self.split_dir / "numeric.npy", mmap_mode="r"
        )
        self.categorical = np.load(
            self.split_dir / "categorical.npy", mmap_mode="r"
        )
        self.row_ids = np.load(
            self.split_dir / "row_id.npy", mmap_mode="r"
        )

        target_path = self.split_dir / "target.npy"
        self.targets = (
            np.load(target_path, mmap_mode="r")
            if target_path.exists()
            else None
        )

        lengths = {
            len(self.numeric),
            len(self.categorical),
            len(self.row_ids),
        }
        if self.targets is not None:
            lengths.add(len(self.targets))
        if len(lengths) != 1:
            raise ValueError(f"Array length mismatch in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.numeric)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "numeric": torch.tensor(
                self.numeric[index], dtype=torch.float32
            ),
            "categorical": torch.tensor(
                self.categorical[index], dtype=torch.long
            ),
            "row_id": str(self.row_ids[index]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(
                self.targets[index], dtype=torch.float32
            )
        return item


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
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0 * (1.0 - brier / baseline_brier)
        )
        bss = max(0.0, bss_raw)

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
        "bss_raw": float(bss_raw),
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
# Fold Validation 저장
# =========================================================
def save_fold_validation(
    result: dict[str, Any],
    preprocessor: dict[str, Any],
    fold_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation_season = int(fold_config["validation_season"])
    fold_dir = VALIDATION_DIR / f"fold_{validation_season}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    prediction_frame = pd.DataFrame(
        {
            ID: result["row_ids"],
            "target": result["targets"],
            "prediction": result["best_prediction"],
            "validation_season": validation_season,
        }
    )

    prediction_frame.to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        fold_dir / "validation.npz",
        row_id=result["row_ids"],
        target=result["targets"],
        prediction=result["best_prediction"],
        validation_season=np.full(
            len(result["row_ids"]),
            validation_season,
        ),
    )

    pd.DataFrame(result["history"]).to_csv(
        fold_dir / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = result["best_metrics"]
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "TabularMLP",
        "variant": VARIANT,
        "fold": int(fold_config["fold"]),
        "train_seasons": [
            int(season)
            for season in fold_config["train_seasons"]
        ],
        "validation_season": validation_season,
        "n_train": int(fold_config["n_train"]),
        "n_validation": int(fold_config["n_validation"]),
        "categorical_columns": preprocessor["categorical_columns"],
        "numeric_columns": preprocessor["numeric_columns"],
        "n_features": (
            len(preprocessor["categorical_columns"])
            + len(preprocessor["numeric_columns"])
        ),
        "cardinalities": preprocessor["cardinalities"],
        "model_config": result["model_config"],
        "best_epoch": int(result["best_epoch"]),
        "metrics": metrics,
        "training_seconds": float(result["elapsed"]),
        "preprocessing": {
            "source": "data/mlp/mlp_preprocess.py",
            "variant": VARIANT,
            "count_state": bool(preprocessor.get("count_state", True)),
            "raw_count_features_retained": bool(
                preprocessor.get(
                    "raw_count_features_retained",
                    VARIANT == "hybrid",
                )
            ),
            "unknown_index": int(preprocessor["unknown_index"]),
            "missing_index": int(preprocessor["missing_index"]),
            "fit_seasons": preprocessor.get("fit_seasons"),
            "fit_rows": preprocessor.get("fit_rows"),
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

    with (fold_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return prediction_frame, metadata


# =========================================================
# Expanding Window Validation
# =========================================================
def run_expanding_validation(
    device: torch.device,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation already exists: {VALIDATION_DIR}"
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    fold_metadata_list: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    print("=" * 70)
    print(EXPERIMENT_NAME)
    print("=" * 70)
    print("Mode          : expanding")
    print(f"Variant       : {VARIANT}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Data root     : {EXPANDING_DATA_DIR}")
    print("=" * 70)

    for base_config in EXPANDING_FOLDS:
        fold_config = dict(base_config)
        validation_season = int(
            fold_config["validation_season"]
        )
        fold_data_dir = (
            EXPANDING_DATA_DIR
            / f"fold_{validation_season}"
        )

        preprocessor = load_preprocessor(
            fold_data_dir / "preprocessor.json"
        )
        train_dataset = TabularNpyDataset(
            fold_data_dir / "train"
        )
        valid_dataset = TabularNpyDataset(
            fold_data_dir / "valid"
        )

        validate_dataset_schema(
            train_dataset,
            preprocessor,
            f"fold_{validation_season}/train",
            require_target=True,
        )
        validate_dataset_schema(
            valid_dataset,
            preprocessor,
            f"fold_{validation_season}/valid",
            require_target=True,
        )

        fold_config["n_train"] = len(train_dataset)
        fold_config["n_validation"] = len(valid_dataset)

        print()
        print("=" * 70)
        print(
            f"Fold {fold_config['fold']}: "
            f"{fold_config['train_seasons'][0]}"
            f"~{fold_config['train_seasons'][-1]} "
            f"-> {validation_season}"
        )
        print("=" * 70)
        print(f"Train rows    : {len(train_dataset):,}")
        print(f"Valid rows    : {len(valid_dataset):,}")
        print(f"Numerical     : {train_dataset.numeric.shape[1]}")
        print(f"Categorical   : {train_dataset.categorical.shape[1]}")

        print_unknown_ratios(
            valid_dataset,
            preprocessor,
        )

        # Fold별 초기화 조건을 동일하게 맞춘다.
        seed_everything(SEED)

        result = fit_validation_model(
            train_dataset,
            valid_dataset,
            preprocessor,
            device,
        )

        prediction_frame, fold_metadata = (
            save_fold_validation(
                result=result,
                preprocessor=preprocessor,
                fold_config=fold_config,
            )
        )

        fold_metadata_list.append(
            fold_metadata
        )
        oof_frames.append(
            prediction_frame
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    return fold_metadata_list, oof


# =========================================================
# Expanding Validation Summary
# =========================================================
def save_validation_summary(
    fold_metadata_list: list[dict[str, Any]],
    oof: pd.DataFrame,
) -> dict[str, Any]:
    briers = np.asarray(
        [
            fold["metrics"]["brier"]
            for fold in fold_metadata_list
        ],
        dtype=np.float64,
    )
    bss_values = np.asarray(
        [
            fold["metrics"]["bss"]
            for fold in fold_metadata_list
        ],
        dtype=np.float64,
    )
    bss_raw_values = np.asarray(
        [
            fold["metrics"]["bss_raw"]
            for fold in fold_metadata_list
        ],
        dtype=np.float64,
    )
    fold_sizes = np.asarray(
        [
            fold["n_validation"]
            for fold in fold_metadata_list
        ],
        dtype=np.float64,
    )

    recent_briers = [
        fold["metrics"]["brier"]
        for fold in fold_metadata_list
        if fold["validation_season"] in (2023, 2024)
    ]

    oof_metrics = calculate_metrics(
        oof["target"].to_numpy(),
        oof["prediction"].to_numpy(),
    )

    latest_fold = max(
        fold_metadata_list,
        key=lambda fold: fold["validation_season"],
    )

    summary = {
        "n_oof": int(len(oof)),
        "mean_fold_brier": float(np.mean(briers)),
        "std_fold_brier": float(np.std(briers)),
        "weighted_fold_brier": float(
            np.average(briers, weights=fold_sizes)
        ),
        "best_fold_brier": float(np.min(briers)),
        "worst_fold_brier": float(np.max(briers)),
        "fold_brier_range": float(
            np.max(briers) - np.min(briers)
        ),
        "recent_2023_2024_mean_brier": float(
            np.mean(recent_briers)
        ),
        "mean_fold_bss_raw": float(
            np.mean(bss_raw_values)
        ),
        "weighted_fold_bss_raw": float(
            np.average(
                bss_raw_values,
                weights=fold_sizes,
            )
        ),
        "mean_fold_bss": float(np.mean(bss_values)),
        "weighted_fold_bss": float(
            np.average(
                bss_values,
                weights=fold_sizes,
            )
        ),
        "pooled_oof_brier": float(
            oof_metrics["brier"]
        ),
        "pooled_oof_baseline_brier": float(
            oof_metrics["baseline_brier"]
        ),
        "pooled_oof_bss_raw": float(
            oof_metrics["bss_raw"]
        ),
        "pooled_oof_bss": float(
            oof_metrics["bss"]
        ),
        "best_epochs": {
            str(fold["validation_season"]): int(
                fold["best_epoch"]
            )
            for fold in fold_metadata_list
        },
        "latest_fold_best_epoch": int(
            latest_fold["best_epoch"]
        ),
    }

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR / "oof_validation.npz",
        row_id=oof[ID].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction=oof["prediction"].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "TabularMLP",
        "variant": VARIANT,
        "validation_method": "expanding_window",
        "final_epoch_strategy": FINAL_EPOCH_STRATEGY,
        "folds": fold_metadata_list,
        "summary": summary,
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
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return summary


def select_final_epoch(
    fold_metadata_list: list[dict[str, Any]],
) -> int:
    if FINAL_EPOCH_STRATEGY == "latest_fold":
        latest_fold = max(
            fold_metadata_list,
            key=lambda fold: fold["validation_season"],
        )
        return int(latest_fold["best_epoch"])

    raise ValueError(
        "Unsupported FINAL_EPOCH_STRATEGY: "
        f"{FINAL_EPOCH_STRATEGY}"
    )


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

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    summary = metadata.get(
        "summary",
        {}
    )
    best_epoch = summary.get(
        "latest_fold_best_epoch"
    )

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
            "validation": "4-Fold Expanding Window Validation",
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
def run_expanding(
    device: torch.device,
) -> None:
    (
        fold_metadata_list,
        oof,
    ) = run_expanding_validation(
        device
    )

    summary = save_validation_summary(
        fold_metadata_list,
        oof,
    )

    final_epoch = select_final_epoch(
        fold_metadata_list
    )

    print()
    print("=" * 70)
    print("MLP Expanding Window 결과 요약")
    print("=" * 70)

    for fold in fold_metadata_list:
        metrics = fold["metrics"]
        print(
            f"Valid {fold['validation_season']}: "
            f"Brier={metrics['brier']:.8f}, "
            f"BSS={metrics['bss']:.2f}, "
            f"epoch={fold['best_epoch']}"
        )

    print(
        "Fold 단순 평균 Brier : "
        f"{summary['mean_fold_brier']:.8f}"
    )
    print(
        "Fold 가중 평균 Brier : "
        f"{summary['weighted_fold_brier']:.8f}"
    )
    print(
        "Fold Brier 표준편차  : "
        f"{summary['std_fold_brier']:.8f}"
    )
    print(
        "Fold Brier 범위      : "
        f"{summary['fold_brier_range']:.8f}"
    )
    print(
        "2023~2024 평균 Brier : "
        f"{summary['recent_2023_2024_mean_brier']:.8f}"
    )
    print(
        "OOF pooled Brier     : "
        f"{summary['pooled_oof_brier']:.8f}"
    )
    print(
        "OOF pooled BSS       : "
        f"{summary['pooled_oof_bss']:.2f}"
    )
    print(
        "Final epoch strategy : "
        f"{FINAL_EPOCH_STRATEGY}"
    )
    print(
        "Final epoch          : "
        f"{final_epoch}"
    )
    print(
        "Validation 저장      : "
        f"{VALIDATION_DIR}"
    )
    print("=" * 70)


def run_final(
    device: torch.device,
    best_epoch: int | None,
) -> None:
    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Model already exists: {MODEL_PATH}"
        )

    if best_epoch is None:
        best_epoch = (
            load_validation_best_epoch()
        )

    if best_epoch is None:
        raise ValueError(
            "Final mode requires --best-epoch, "
            "or an existing Expanding metadata: "
            f"{VALIDATION_DIR / 'metadata.json'}"
        )

    if best_epoch <= 0:
        raise ValueError(
            "best_epoch must be >= 1"
        )

    preprocessor_path = (
        FINAL_DATA_DIR
        / "preprocessor.json"
    )
    preprocessor = load_preprocessor(
        preprocessor_path
    )

    train_dataset = TabularNpyDataset(
        FINAL_DATA_DIR / "train"
    )

    validate_dataset_schema(
        train_dataset,
        preprocessor,
        "final/train",
        require_target=True,
    )

    print("=" * 70)
    print(EXPERIMENT_NAME)
    print("=" * 70)
    print("Mode          : final")
    print(f"Variant       : {VARIANT}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )
    print(f"PyTorch       : {torch.__version__}")
    print(
        f"Train rows    : "
        f"{len(train_dataset):,}"
    )
    print(
        f"Best epoch    : "
        f"{best_epoch}"
    )
    print(
        f"Numerical     : "
        f"{train_dataset.numeric.shape[1]}"
    )
    print(
        f"Categorical   : "
        f"{train_dataset.categorical.shape[1]}"
    )
    print("=" * 70)

    # Final preprocessor의 cardinality로
    # 새로운 모델 구조를 생성한다.
    seed_everything(SEED)

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

    seed_everything(SEED)
    device = get_device()

    if args.mode == "expanding":
        run_expanding(device)
    else:
        run_final(
            device,
            args.best_epoch,
        )


if __name__ == "__main__":
    main()

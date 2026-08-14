"""Prepare leakage-safe NumPy arrays for a PyTorch tabular MLP.

Examples
--------
Validation experiment (fit on 2019-2023, transform 2024):
    python mlp_preprocess.py --mode validation --variant processed

Final training/inference (fit on all train rows, transform test):
    python mlp_preprocess.py --mode final --variant hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_DIR = Path("./data")
OUTPUT_ROOT = Path("./data/mlp")
TARGET = "control_success"
ID = "row_id"
VALIDATION_SEASON = 2024

BASE_CATEGORICAL_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "count_state",
]

ALWAYS_DROP_COLUMNS = [
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "score_diff_home",
]

COUNT_COLUMNS = ["balls_before", "strikes_before"]
UNKNOWN_TOKEN = "__UNKNOWN__"
MISSING_TOKEN = "__MISSING__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["validation", "final"],
        default="validation",
        help="validation: 2019-2023 fit / 2024 valid; final: all train fit / test",
    )
    parser.add_argument(
        "--variant",
        choices=["processed", "hybrid"],
        default="processed",
        help="processed drops raw counts; hybrid retains them as numeric inputs",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def add_count_state(frame: pd.DataFrame) -> pd.DataFrame:
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
            invalid, ["balls_before", "strikes_before"]
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

    missing = sorted(set(drop_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    return frame.drop(columns=drop_columns)


def categorical_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_TOKEN)


def fit_preprocessor(
    fit_frame: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> dict[str, Any]:
    category_maps: dict[str, dict[str, int]] = {}
    cardinalities: dict[str, int] = {}

    for column in categorical_columns:
        values = sorted(categorical_strings(fit_frame[column]).unique().tolist())
        mapping = {UNKNOWN_TOKEN: 0, MISSING_TOKEN: 1}
        for value in values:
            if value not in mapping:
                mapping[value] = len(mapping)
        category_maps[column] = mapping
        cardinalities[column] = len(mapping)

    numeric_stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        values = pd.to_numeric(fit_frame[column], errors="coerce")
        median = values.median()
        if pd.isna(median):
            median = 0.0
        filled = values.fillna(median).astype("float64")
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        numeric_stats[column] = {
            "median": float(median),
            "mean": mean,
            "std": std,
        }

    return {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "category_maps": category_maps,
        "cardinalities": cardinalities,
        "numeric_stats": numeric_stats,
        "unknown_index": 0,
        "missing_index": 1,
    }


def transform_categorical(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in preprocessor["categorical_columns"]:
        mapping = preprocessor["category_maps"][column]
        encoded = (
            categorical_strings(frame[column])
            .map(mapping)
            .fillna(preprocessor["unknown_index"])
            .astype("int64")
            .to_numpy()
        )
        columns.append(encoded)
    return np.column_stack(columns).astype(np.int64, copy=False)


def transform_numeric(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in preprocessor["numeric_columns"]:
        stats = preprocessor["numeric_stats"][column]
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(stats["median"])
            .astype("float64")
        )
        standardized = (values - stats["mean"]) / stats["std"]
        columns.append(standardized.to_numpy(dtype=np.float32))
    return np.column_stack(columns).astype(np.float32, copy=False)


def save_split(
    frame: pd.DataFrame,
    split_dir: Path,
    preprocessor: dict[str, Any],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    numeric = transform_numeric(frame, preprocessor)
    categorical = transform_categorical(frame, preprocessor)

    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite numeric value found in {split_dir.name}")

    np.save(split_dir / "numeric.npy", numeric, allow_pickle=False)
    np.save(split_dir / "categorical.npy", categorical, allow_pickle=False)

    row_ids = frame[ID].astype(str).to_numpy(dtype=str)

    np.save(
        split_dir / "row_id.npy",
        row_ids,
        allow_pickle=False,
    )
    if TARGET in frame.columns:
        target = frame[TARGET].to_numpy(dtype=np.float32)
        np.save(split_dir / "target.npy", target, allow_pickle=False)

    print(
        f"Saved {split_dir}: rows={len(frame):,}, "
        f"numeric={numeric.shape[1]}, categorical={categorical.shape[1]}"
    )


def determine_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [
        column for column in BASE_CATEGORICAL_COLUMNS if column in frame.columns
    ]
    excluded = {ID, TARGET, *categorical_columns}
    numeric_columns = [
        column for column in frame.columns if column not in excluded
    ]
    return categorical_columns, numeric_columns


def main() -> None:
    args = parse_args()
    train_path = args.raw_dir / "train.csv"
    test_path = args.raw_dir / "test.csv"

    train = prepare_frame(
        pd.read_csv(train_path, encoding="utf-8-sig"),
        args.variant,
    )

    output_dir = args.output_root / args.variant / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "validation":
        fit_frame = train.loc[train["season"] < VALIDATION_SEASON].copy()
        valid_frame = train.loc[train["season"] == VALIDATION_SEASON].copy()
        if fit_frame.empty or valid_frame.empty:
            raise ValueError("Validation split is empty; check the season column")
        splits = {"train": fit_frame, "valid": valid_frame}
        fit_description = "season < 2024"
    else:
        test = prepare_frame(
            pd.read_csv(test_path, encoding="utf-8-sig"),
            args.variant,
        )
        fit_frame = train
        splits = {"train": train, "test": test}
        fit_description = "all train rows"

    categorical_columns, numeric_columns = determine_features(fit_frame)
    preprocessor = fit_preprocessor(
        fit_frame,
        categorical_columns,
        numeric_columns,
    )
    preprocessor.update(
        {
            "mode": args.mode,
            "variant": args.variant,
            "fit_description": fit_description,
            "target": TARGET,
            "id": ID,
        }
    )

    with (output_dir / "preprocessor.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(preprocessor, file, ensure_ascii=False, indent=2)

    for split_name, split_frame in splits.items():
        save_split(
            split_frame,
            output_dir / split_name,
            preprocessor,
        )

    print(f"Preprocessor saved: {output_dir / 'preprocessor.json'}")


if __name__ == "__main__":
    main()

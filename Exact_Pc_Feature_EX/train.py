# train.py

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Config
# ============================================================

EXPERIMENT_NAME = "cb_reg_v003_exact_pc_full_seed42"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = Path("./model") / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

# Existing canonical control model.
CONTROL_EXPERIMENT = "cb_reg_v003_exact"
CONTROL_MODEL_PATH = (
    Path("./model")
    / CONTROL_EXPERIMENT
    / f"{CONTROL_EXPERIMENT}.pkl"
)

# Exact PC implementation that produced the accepted validation.
PC_SOURCE_PATH = Path(
    "./cb_reg_v021_pitcher_conditional_r_only_ablation.py"
)

# Existing validation result. We only read metadata; no fold retraining.
PC_VALIDATION_META_PATH = (
    Path("./validation")
    / "cb_reg_v021_pitcher_conditional_r_only_ablation"
    / "metadata.json"
)

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

RANDOM_SEED = 42

EXPECTED_CONTROL_FEATURES = 52
EXPECTED_TREATMENT_FEATURES = 73


# ============================================================
# Import exact validated v021 implementation
# ============================================================

def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"필수 파일 없음: {path}\n"
            "검증에 사용한 "
            "cb_reg_v021_pitcher_conditional_r_only_ablation.py를 "
            "프로젝트 루트에 두세요."
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Existing validation -> final iteration
# ============================================================

def load_final_iterations():
    """
    v021의 latest fold(2024) treatment best iteration을 재사용한다.

    v021 treatment_best_iteration은 CatBoost get_best_iteration()
    즉 zero-based index이므로 final full-train tree count = +1.
    """

    if not PC_VALIDATION_META_PATH.exists():
        raise FileNotFoundError(
            "기존 v021 validation metadata가 없습니다:\n"
            f"{PC_VALIDATION_META_PATH}\n\n"
            "이미 완료한 validation을 다시 돌리지 않기 위해 "
            "기존 metadata를 사용합니다."
        )

    metadata = json.loads(
        PC_VALIDATION_META_PATH.read_text(
            encoding="utf-8"
        )
    )

    fold_results = metadata.get(
        "fold_results",
        []
    )

    latest = [
        row
        for row in fold_results
        if int(
            row.get(
                "validation_season",
                -1,
            )
        ) == 2024
    ]

    if len(latest) != 1:
        raise RuntimeError(
            "v021 metadata에서 validation_season=2024인 "
            "fold를 정확히 1개 찾지 못했습니다."
        )

    best_iteration_zero_based = int(
        latest[0][
            "treatment_best_iteration"
        ]
    )

    if best_iteration_zero_based < 0:
        raise RuntimeError(
            "잘못된 treatment_best_iteration: "
            f"{best_iteration_zero_based}"
        )

    final_iterations = (
        best_iteration_zero_based + 1
    )

    return (
        final_iterations,
        best_iteration_zero_based,
        latest[0],
    )


# ============================================================
# PC lookup tables for future row-local inference
# ============================================================

def build_pc_tables_by_source_season(
    pc,
    train: pd.DataFrame,
):
    """
    script.py가 test row끼리 집계하지 않고 PC를 만들 수 있도록,
    공식 train의 각 completed season에서 lookup table을 미리 저장한다.

    target season y의 row는 source season y-1 table만 사용한다.

    예:
      2025 test row -> 2024 official train labels로 만든 table

    저장되는 table:
      pitcher_id x context
      -> rate / reliability / delta / log_n

    PC model에는 rate / reliability / delta만 들어가지만,
    원본 table을 재현성 목적으로 그대로 저장한다.
    """

    train_ctx = pc._ctx_frame_v15(
        train.reset_index(
            drop=True
        )
    )

    seasons = sorted(
        pd.to_numeric(
            train_ctx[SEASON_COL],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    result = {}

    for source_year in seasons:
        history = (
            train_ctx.loc[
                pd.to_numeric(
                    train_ctx[SEASON_COL],
                    errors="raise",
                )
                .astype(int)
                .eq(
                    int(source_year)
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        season_tables = {}

        for context, strength in pc.PC_CONTEXTS:
            table = pc._pc_eb_table(
                history=history,
                keys=[
                    "pitcher_id",
                    context,
                ],
                parent_keys=[
                    "pitcher_id",
                ],
                strength=float(
                    strength
                ),
                parent_strength=float(
                    pc.PC_PARENT_STRENGTH
                ),
            )

            season_tables[
                context
            ] = table

        result[
            int(source_year)
        ] = season_tables

    return result


# ============================================================
# Main
# ============================================================

def main():
    start_total = time.time()

    # --------------------------------------------------------
    # Preconditions
    # --------------------------------------------------------

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        CONTROL_MODEL_PATH,
        PC_SOURCE_PATH,
        PC_VALIDATION_META_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    if MODEL_DIR.exists():
        raise FileExistsError(
            "기존 model 보존을 위해 중단: "
            f"{MODEL_DIR}\n"
            "기존 폴더를 지우거나 EXPERIMENT_NAME을 변경하세요."
        )

    # --------------------------------------------------------
    # Exact validated PC implementation
    # --------------------------------------------------------

    pc = import_module_from_path(
        "pc_v021_source",
        PC_SOURCE_PATH,
    )

    if int(
        pc.MODEL_PARAMS.get(
            "random_seed",
            -1,
        )
    ) != RANDOM_SEED:
        raise RuntimeError(
            "v021 MODEL_PARAMS random_seed가 42가 아닙니다."
        )

    # --------------------------------------------------------
    # Use test only as schema/header.
    # --------------------------------------------------------

    test_cols = (
        pd.read_csv(
            TEST_PATH,
            encoding="utf-8-sig",
            nrows=0,
        )
        .columns
        .tolist()
    )

    base_features = [
        col
        for col in test_cols
        if col != ID
    ]

    if (
        "asof_pitcher_pitchmix_n"
        not in base_features
    ):
        raise RuntimeError(
            "canonical cb_reg_v003 feature "
            "'asof_pitcher_pitchmix_n'이 test schema에 없습니다."
        )

    required_pc_test_cols = {
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
    }

    missing_pc_test = sorted(
        required_pc_test_cols
        - set(
            test_cols
        )
    )

    if missing_pc_test:
        raise KeyError(
            "PC row-local inference에 필요한 test column 없음: "
            f"{missing_pc_test}"
        )

    # --------------------------------------------------------
    # Load train
    # --------------------------------------------------------

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_train = {
        ID,
        TARGET,
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    }

    missing_train = sorted(
        required_train
        - set(
            train.columns
        )
    )

    if missing_train:
        raise KeyError(
            f"train required columns missing: {missing_train}"
        )

    train[SEASON_COL] = (
        pd.to_numeric(
            train[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Existing CONTROL artifact
    # --------------------------------------------------------

    print("=" * 92)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print("CONTROL    : existing cb_reg_v003_exact (REUSE, no retraining)")
    print("TREATMENT  : CONTROL backbone + R-only PC FULL 21")
    print("Seed       : 42")
    print("Routing    : R=Treatment / F=CONTROL")
    print("Residual   : OFF")
    print("L2 / L3    : OFF")
    print("Validation : REUSE existing v021 metadata")
    print("Test usage : HEADER ONLY")
    print("=" * 92)

    control_artifact = joblib.load(
        CONTROL_MODEL_PATH
    )

    required_control_keys = {
        "model",
        "features",
        "cat_cols",
        "hbeb_priors",
        "pitcher_season_stats",
    }

    missing_control_keys = sorted(
        required_control_keys
        - set(
            control_artifact.keys()
        )
    )

    if missing_control_keys:
        raise KeyError(
            "기존 CONTROL artifact에 필요한 key가 없습니다: "
            f"{missing_control_keys}"
        )

    control_features = list(
        control_artifact[
            "features"
        ]
    )

    control_cat_cols = list(
        control_artifact[
            "cat_cols"
        ]
    )

    if len(
        control_features
    ) != EXPECTED_CONTROL_FEATURES:
        raise RuntimeError(
            "CONTROL feature count mismatch: "
            f"{len(control_features)} != {EXPECTED_CONTROL_FEATURES}"
        )

    if (
        "asof_pitcher_pitchmix_n"
        not in control_features
    ):
        raise RuntimeError(
            "기존 CONTROL artifact가 canonical v003_exact가 아닙니다."
        )

    control_seed = (
        control_artifact
        .get(
            "model_params",
            {},
        )
        .get(
            "random_seed",
            None,
        )
    )

    if (
        control_seed is not None
        and int(control_seed) != RANDOM_SEED
    ):
        raise RuntimeError(
            f"CONTROL random_seed={control_seed}; "
            "이번 LB A/B test는 seed42 고정이어야 합니다."
        )

    # --------------------------------------------------------
    # Existing v021 validation -> Treatment final iterations
    # --------------------------------------------------------

    (
        final_iterations,
        best_iteration_zero_based,
        latest_fold,
    ) = load_final_iterations()

    print()
    print(
        "Existing v021 latest fold:"
    )
    print(
        f"- validation season       : {latest_fold['validation_season']}"
    )
    print(
        f"- treatment best iteration: {best_iteration_zero_based} (0-based)"
    )
    print(
        f"- final tree count         : {final_iterations}"
    )

    # --------------------------------------------------------
    # Exact-ASOF
    # --------------------------------------------------------

    pitcher_season_stats = (
        pc.build_pitcher_season_stats(
            train
        )
    )

    (
        train_exact,
        exact_diag,
    ) = (
        pc.add_pitcher_exact_asof_features(
            train,
            pitcher_season_stats,
        )
    )

    exact_features = list(
        base_features
    )

    for col in pc.EXACT_FEATURES:
        if col not in exact_features:
            exact_features.append(
                col
            )

    # --------------------------------------------------------
    # HB-EB
    # --------------------------------------------------------

    (
        train_control_view,
        generated_control_features,
        treatment_hbeb_priors,
    ) = (
        pc.apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )
    )

    # Critical A/B guard:
    # PC treatment must sit on the exact same control feature backbone.
    if list(
        generated_control_features
    ) != list(
        control_features
    ):
        only_existing = [
            x
            for x in control_features
            if x not in generated_control_features
        ]

        only_generated = [
            x
            for x in generated_control_features
            if x not in control_features
        ]

        raise RuntimeError(
            "새 Treatment의 CONTROL backbone이 기존 "
            "cb_reg_v003_exact artifact와 다릅니다.\n"
            f"Only existing: {only_existing}\n"
            f"Only generated: {only_generated}\n"
            "순수 PC feature A/B test가 아니므로 중단합니다."
        )

    # --------------------------------------------------------
    # PITCHER_CONDITIONAL FULL 21
    # each train row y <- exactly previous labeled season y-1
    # --------------------------------------------------------

    (
        pc_train,
        pc_coverage,
    ) = (
        pc.build_pitcher_conditional_for_rows(
            rows=train,
            available_history=train,
            scope="final_train",
        )
    )

    # R-only feature activation.
    # F rows are zero vector during Treatment training.
    pc_train = (
        pc.mask_pitcher_conditional_to_r_only(
            pc_train,
            train,
        )
    )

    train_treatment = (
        train_control_view.copy()
    )

    for col in pc.PC_FEATURES:
        if col in train_treatment.columns:
            raise RuntimeError(
                f"Duplicate PC feature: {col}"
            )

        train_treatment[
            col
        ] = pc_train[
            col
        ]

    treatment_features = (
        list(
            generated_control_features
        )
        + list(
            pc.PC_FEATURES
        )
    )

    if len(
        treatment_features
    ) != EXPECTED_TREATMENT_FEATURES:
        raise RuntimeError(
            "Treatment feature count mismatch: "
            f"{len(treatment_features)} != {EXPECTED_TREATMENT_FEATURES}"
        )

    if len(
        set(
            treatment_features
        )
    ) != len(
        treatment_features
    ):
        raise RuntimeError(
            "Treatment feature duplicated"
        )

    treatment_cat_cols = [
        col
        for col in pc.CAT_COLS
        if col in treatment_features
    ]

    # --------------------------------------------------------
    # Final Treatment model: ONLY NEW CATBOOST FIT
    # --------------------------------------------------------

    X_full = (
        pc.prepare_features(
            train_treatment[
                treatment_features
            ],
            treatment_cat_cols,
        )
    )

    y_full = (
        train[TARGET]
        .astype(
            np.float64
        )
    )

    final_params = {
        **pc.MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
        "random_seed": RANDOM_SEED,
    }

    treatment_model = (
        CatBoostRegressor(
            **final_params
        )
    )

    print()
    print("=" * 92)
    print("Final Treatment training")
    print("=" * 92)
    print(f"Train rows : {len(train):,}")
    print(f"Features   : {len(treatment_features)}")
    print(f"PC features: {len(pc.PC_FEATURES)}")
    print(f"Iterations : {final_iterations}")
    print(f"Seed       : {RANDOM_SEED}")
    print("CONTROL fit: 0")
    print("NEW fit    : 1 Treatment CatBoost")
    print("=" * 92)

    fit_start = time.time()

    treatment_model.fit(
        X_full,
        y_full,
        cat_features=(
            treatment_cat_cols
        ),
    )

    fit_elapsed = (
        time.time()
        - fit_start
    )

    if hasattr(
        treatment_model,
        "feature_names_",
    ):
        if list(
            treatment_model.feature_names_
        ) != list(
            treatment_features
        ):
            raise RuntimeError(
                "Treatment model feature order mismatch"
            )

    # --------------------------------------------------------
    # Freeze PC lookup tables for row-local inference.
    # No test rows are used.
    # --------------------------------------------------------

    pc_tables_by_source_season = (
        build_pc_tables_by_source_season(
            pc=pc,
            train=train,
        )
    )

    latest_train_season = int(
        train[SEASON_COL]
        .max()
    )

    if (
        latest_train_season
        not in pc_tables_by_source_season
    ):
        raise RuntimeError(
            "latest train season PC table missing"
        )

    # --------------------------------------------------------
    # Save ONE routed artifact
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "Routed CatBoostRegressor"
        ),
        "random_seed": (
            RANDOM_SEED
        ),

        # ----------------------------------------------
        # CONTROL: already trained model + frozen state
        # ----------------------------------------------
        "control": {
            "experiment_name": (
                CONTROL_EXPERIMENT
            ),
            "retrained": False,
            "artifact": (
                control_artifact
            ),
        },

        # ----------------------------------------------
        # TREATMENT
        # ----------------------------------------------
        "treatment": {
            "model": (
                treatment_model
            ),
            "features": (
                treatment_features
            ),
            "cat_cols": (
                treatment_cat_cols
            ),
            "control_backbone_features": (
                generated_control_features
            ),
            "hbeb_priors": (
                treatment_hbeb_priors
            ),
            "pitcher_season_stats": (
                pitcher_season_stats
            ),
            "exact_features": list(
                pc.EXACT_FEATURES
            ),
            "pc_features": list(
                pc.PC_FEATURES
            ),
            "pc_contexts": [
                {
                    "context": context,
                    "strength": float(
                        strength
                    ),
                }
                for context, strength
                in pc.PC_CONTEXTS
            ],
            "pc_fields": list(
                pc.PC_FIELDS
            ),
            "pc_parent_strength": float(
                pc.PC_PARENT_STRENGTH
            ),
            "pc_unseen_default": 0.0,
            "pc_tables_by_source_season": (
                pc_tables_by_source_season
            ),
            "final_iterations": int(
                final_iterations
            ),
            "model_params": (
                final_params
            ),
        },

        # ----------------------------------------------
        # Deployment policy
        # ----------------------------------------------
        "routing": {
            "column": (
                GAME_TYPE_COL
            ),
            "treatment_rule": (
                "game_type != 'F'"
            ),
            "control_rule": (
                "game_type == 'F'"
            ),
            "R_model": (
                "treatment"
            ),
            "F_model": (
                "control"
            ),
        },

        "prediction_policy": (
            "np.clip(raw_prediction, 0.0, 1.0)"
        ),

        "feature_policy": {
            "test_rows_used_during_training": False,
            "test_usage_during_training": (
                "header_only"
            ),
            "exact_asof": (
                "official row ASOF minus completed prior-season train labels"
            ),
            "pc_source": (
                "exactly previous labeled season only"
            ),
            "pc_activation": (
                "R only; F vector=0 in treatment model"
            ),
        },
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    final_meta = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "purpose": (
            "Public LB checkpoint for R-only PC FULL on "
            "canonical cb_reg_v003_exact"
        ),
        "seed": (
            RANDOM_SEED
        ),
        "control": {
            "experiment": (
                CONTROL_EXPERIMENT
            ),
            "retrained": False,
            "path": str(
                CONTROL_MODEL_PATH
            ),
            "n_features": int(
                len(
                    control_features
                )
            ),
        },
        "treatment": {
            "new_fit_count": 1,
            "n_features": int(
                len(
                    treatment_features
                )
            ),
            "n_pc_features": int(
                len(
                    pc.PC_FEATURES
                )
            ),
            "final_iterations": int(
                final_iterations
            ),
            "best_iteration_source": (
                "existing v021 2024 expanding-window fold"
            ),
            "best_iteration_zero_based": int(
                best_iteration_zero_based
            ),
            "fit_elapsed_seconds": float(
                fit_elapsed
            ),
        },
        "routing": (
            "R=treatment / F=existing control"
        ),
        "residual": False,
        "l2": False,
        "l3": False,
        "multi_seed": False,
        "pc": {
            "parent_strength": float(
                pc.PC_PARENT_STRENGTH
            ),
            "contexts": [
                {
                    "context": context,
                    "strength": float(
                        strength
                    ),
                }
                for context, strength
                in pc.PC_CONTEXTS
            ],
            "fields": list(
                pc.PC_FIELDS
            ),
            "features": list(
                pc.PC_FEATURES
            ),
            "source_policy": (
                "target season y uses exactly source season y-1 labels"
            ),
            "unseen_default": 0.0,
        },
        "train_seasons": sorted(
            [
                int(x)
                for x in train[
                    SEASON_COL
                ]
                .unique()
            ]
        ),
        "latest_train_season": int(
            latest_train_season
        ),
        "n_train": int(
            len(
                train
            )
        ),
        "test_usage": (
            "header only"
        ),
        "exact_asof_diagnostics": (
            exact_diag.to_dict(
                orient="records"
            )
        ),
        "pc_training_coverage": (
            pc_coverage.to_dict(
                orient="records"
            )
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "total_elapsed_seconds": float(
            time.time()
            - start_total
        ),
    }

    with open(
        MODEL_DIR / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Reproducibility files.
    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR / "train.py",
        )
    except Exception:
        pass

    try:
        shutil.copy2(
            PC_SOURCE_PATH,
            MODEL_DIR
            / "cb_reg_v021_pitcher_conditional_r_only_ablation.py",
        )
    except Exception:
        pass

    print()
    print("=" * 92)
    print("Final routed model saved")
    print("=" * 92)
    print(f"Artifact     : {MODEL_PATH}")
    print(f"CONTROL fits : 0")
    print(f"Treatment fits: 1")
    print(f"Seed         : {RANDOM_SEED}")
    print(
        f"Features     : "
        f"{len(control_features)} CONTROL / "
        f"{len(treatment_features)} Treatment"
    )
    print(
        f"PC           : {len(pc.PC_FEATURES)} features, R-only"
    )
    print(
        f"Routing      : R=Treatment / F=CONTROL"
    )
    print(
        f"Final trees  : {final_iterations}"
    )
    print(
        f"Fit elapsed  : {fit_elapsed:.1f}s"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()

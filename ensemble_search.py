# ensemble_search_v002.py
#
# LG Aimers Phase2 Validation ensemble 분석 확장 버전
#
# 추가 기능:
# 1. Prediction correlation
# 2. Residual correlation
# 3. Pair grid search
# 4. Pair exact optimum (Brier / squared error 기준 해석적 최적해)
# 5. Pair 비교 요약 저장
# 6. Triple grid search 유지
# 7. 전체 최적 ensemble prediction 저장
#
# 현재 주요 모델:
# - CatBoostClassifier
# - CatBoostRegressor
# - XGBClassifier

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 설정
# =========================================================

VALIDATION_ROOT = Path("./validation")

OUTPUT_ROOT = Path("./validation")
EXPERIMENT_NAME = "ensemble_v002"

OUTPUT_DIR = OUTPUT_ROOT / EXPERIMENT_NAME

MODEL_CONFIG = {
    "cb_cls": {
        "path": VALIDATION_ROOT / "cb_v001" / "pred.csv",
        "prediction_candidates": [
            "prediction",
        ],
    },
    "cb_reg": {
        "path": VALIDATION_ROOT / "cb_reg_v001" / "pred.csv",
        "prediction_candidates": [
            "prediction_clip",
            "prediction_raw",
            "prediction",
        ],
    },
    "xgb_cls": {
        "path": VALIDATION_ROOT / "xgb_v001" / "pred.csv",
        "prediction_candidates": [
            "prediction",
        ],
    },
}

PAIR_WEIGHT_STEP = 0.01
TRIPLE_WEIGHT_STEP = 0.02


# =========================================================
# Metric
# =========================================================

def calculate_brier_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true와 y_pred 길이가 다릅니다."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "prediction에 NaN 또는 Inf가 있습니다."
        )

    r = float(
        np.mean(y_true)
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    baseline_brier = float(
        r * (1.0 - r)
    )

    if baseline_brier <= 0:
        raise ValueError(
            "baseline_brier가 0 이하입니다."
        )

    bss = float(
        max(
            0.0,
            100000.0 * (
                1.0
                - brier / baseline_brier
            ),
        )
    )

    return (
        brier,
        baseline_brier,
        bss,
    )


# =========================================================
# Prediction load
# =========================================================

def detect_prediction_column(
    df,
    candidates,
    model_name,
):
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"{model_name}: prediction 컬럼을 찾지 못했습니다.\n"
        f"columns={list(df.columns)}\n"
        f"candidates={candidates}"
    )


def load_prediction(
    model_name,
    config,
):
    path = Path(
        config["path"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction 파일이 없습니다: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required_cols = {
        "row_id",
        "target",
    }

    missing_cols = (
        required_cols
        - set(df.columns)
    )

    if missing_cols:
        raise ValueError(
            f"{model_name}: 필수 컬럼 누락: {missing_cols}"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: row_id 중복이 있습니다."
        )

    pred_col = (
        detect_prediction_column(
            df=df,
            candidates=config[
                "prediction_candidates"
            ],
            model_name=model_name,
        )
    )

    out = df[
        [
            "row_id",
            "target",
            pred_col,
        ]
    ].copy()

    out = out.rename(
        columns={
            pred_col: model_name,
        }
    )

    out[model_name] = pd.to_numeric(
        out[model_name],
        errors="raise",
    )

    if not np.isfinite(
        out[model_name].to_numpy()
    ).all():
        raise ValueError(
            f"{model_name}: prediction에 NaN 또는 Inf가 있습니다."
        )

    print(
        f"[로드] {model_name:<10} "
        f"| rows={len(out):,} "
        f"| column={pred_col} "
        f"| path={path}"
    )

    return out


# =========================================================
# Merge / validation
# =========================================================

def merge_predictions(
    model_config,
):
    merged = None
    expected_n = None

    for model_name, config in (
        model_config.items()
    ):
        df = load_prediction(
            model_name,
            config,
        )

        if expected_n is None:
            expected_n = len(df)
        elif len(df) != expected_n:
            raise ValueError(
                "모델별 validation row 수가 다릅니다. "
                f"expected={expected_n}, "
                f"{model_name}={len(df)}"
            )

        df = df.rename(
            columns={
                "target":
                    f"target__{model_name}"
            }
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df,
                on="row_id",
                how="inner",
                validate="one_to_one",
            )

    if merged is None:
        raise ValueError(
            "로드된 모델이 없습니다."
        )

    if len(merged) != expected_n:
        raise ValueError(
            "모델 간 row_id 집합이 다릅니다. "
            f"expected={expected_n}, merged={len(merged)}"
        )

    target_cols = [
        c
        for c in merged.columns
        if c.startswith("target__")
    ]

    base_target = (
        merged[target_cols[0]]
        .to_numpy()
    )

    for col in target_cols[1:]:
        other_target = (
            merged[col]
            .to_numpy()
        )

        if not np.array_equal(
            base_target,
            other_target,
        ):
            mismatch = int(
                np.sum(
                    base_target
                    != other_target
                )
            )

            raise ValueError(
                f"target 불일치: "
                f"{target_cols[0]} vs {col}, "
                f"mismatch={mismatch}"
            )

    merged["target"] = (
        base_target
    )

    merged = merged.drop(
        columns=target_cols
    )

    merged = merged.sort_values(
        "row_id"
    ).reset_index(
        drop=True
    )

    return merged


# =========================================================
# Single model summary
# =========================================================

def evaluate_single_models(
    df,
    model_names,
):
    records = []

    y_true = (
        df["target"]
        .to_numpy(
            dtype=np.float64
        )
    )

    baseline_brier = None

    for model_name in model_names:
        pred = (
            df[model_name]
            .to_numpy(
                dtype=np.float64
            )
        )

        (
            brier,
            this_baseline_brier,
            bss,
        ) = calculate_brier_metrics(
            y_true,
            pred,
        )

        if baseline_brier is None:
            baseline_brier = (
                this_baseline_brier
            )

        records.append({
            "model": model_name,
            "brier": brier,
            "bss": bss,
            "prediction_mean":
                float(np.mean(pred)),
            "prediction_std":
                float(np.std(pred)),
            "prediction_min":
                float(np.min(pred)),
            "prediction_max":
                float(np.max(pred)),
        })

    result = pd.DataFrame(
        records
    ).sort_values(
        "brier"
    ).reset_index(
        drop=True
    )

    return (
        result,
        baseline_brier,
    )


# =========================================================
# Correlation
# =========================================================

def calculate_correlations(
    df,
    model_names,
):
    """
    Prediction correlation과 residual correlation 계산.

    residual = prediction - target
    """

    prediction_corr = (
        df[model_names]
        .corr(
            method="pearson"
        )
    )

    residual_df = pd.DataFrame(
        index=df.index
    )

    y_true = (
        df["target"]
        .to_numpy(
            dtype=np.float64
        )
    )

    for model_name in model_names:
        residual_df[model_name] = (
            df[model_name]
            .to_numpy(
                dtype=np.float64
            )
            - y_true
        )

    residual_corr = (
        residual_df
        .corr(
            method="pearson"
        )
    )

    return (
        prediction_corr,
        residual_corr,
    )


# =========================================================
# Pair grid search
# =========================================================

def search_pair_grid(
    df,
    model_a,
    model_b,
    step,
):
    y_true = (
        df["target"]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_a = (
        df[model_a]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_b = (
        df[model_b]
        .to_numpy(
            dtype=np.float64
        )
    )

    n_steps = int(
        round(
            1.0 / step
        )
    )

    weights = (
        np.arange(
            n_steps + 1,
            dtype=np.float64,
        )
        * step
    )

    records = []
    best = None

    for w_a in weights:
        w_a = float(
            np.clip(
                w_a,
                0.0,
                1.0,
            )
        )

        w_b = float(
            1.0 - w_a
        )

        pred = (
            w_a * pred_a
            + w_b * pred_b
        )

        pred = np.clip(
            pred,
            0.0,
            1.0,
        )

        (
            brier,
            baseline_brier,
            bss,
        ) = calculate_brier_metrics(
            y_true,
            pred,
        )

        record = {
            "model_a": model_a,
            "model_b": model_b,
            "weight_a": w_a,
            "weight_b": w_b,
            "brier": brier,
            "bss": bss,
        }

        records.append(
            record
        )

        if (
            best is None
            or brier < best["brier"]
        ):
            best = {
                **record,
                "prediction":
                    pred.copy(),
                "baseline_brier":
                    baseline_brier,
            }

    result = pd.DataFrame(
        records
    ).sort_values(
        "brier"
    ).reset_index(
        drop=True
    )

    return (
        result,
        best,
    )


# =========================================================
# Pair exact optimum
# =========================================================

def search_pair_exact(
    df,
    model_a,
    model_b,
):
    """
    Brier Score를 최소화하는 2-model convex ensemble의
    exact optimum weight 계산.

    ensemble:
        p = w * p_a + (1-w) * p_b

    unconstrained optimum:
        w* = sum((p_a-p_b)(y-p_b))
             / sum((p_a-p_b)^2)

    최종적으로 convex ensemble 조건을 위해 [0,1]로 clip.
    """

    y_true = (
        df["target"]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_a = (
        df[model_a]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_b = (
        df[model_b]
        .to_numpy(
            dtype=np.float64
        )
    )

    delta = (
        pred_a
        - pred_b
    )

    denominator = float(
        np.dot(
            delta,
            delta,
        )
    )

    if denominator <= 0:
        # 두 prediction이 완전히 같다면 모든 weight가 동일.
        w_unconstrained = 0.5
    else:
        numerator = float(
            np.dot(
                delta,
                y_true - pred_b,
            )
        )

        w_unconstrained = (
            numerator
            / denominator
        )

    w_a = float(
        np.clip(
            w_unconstrained,
            0.0,
            1.0,
        )
    )

    w_b = float(
        1.0 - w_a
    )

    pred = (
        w_a * pred_a
        + w_b * pred_b
    )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    (
        brier,
        baseline_brier,
        bss,
    ) = calculate_brier_metrics(
        y_true,
        pred,
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "weight_a_unconstrained":
            float(w_unconstrained),
        "weight_a": w_a,
        "weight_b": w_b,
        "brier": brier,
        "bss": bss,
        "baseline_brier":
            baseline_brier,
        "prediction":
            pred.copy(),
    }


# =========================================================
# Triple grid search
# =========================================================

def search_triple_grid(
    df,
    model_a,
    model_b,
    model_c,
    step,
):
    y_true = (
        df["target"]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_a = (
        df[model_a]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_b = (
        df[model_b]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_c = (
        df[model_c]
        .to_numpy(
            dtype=np.float64
        )
    )

    n_steps = int(
        round(
            1.0 / step
        )
    )

    records = []
    best = None

    for i in range(
        n_steps + 1
    ):
        for j in range(
            n_steps - i + 1
        ):
            k = (
                n_steps
                - i
                - j
            )

            w_a = (
                i / n_steps
            )

            w_b = (
                j / n_steps
            )

            w_c = (
                k / n_steps
            )

            pred = (
                w_a * pred_a
                + w_b * pred_b
                + w_c * pred_c
            )

            pred = np.clip(
                pred,
                0.0,
                1.0,
            )

            (
                brier,
                baseline_brier,
                bss,
            ) = calculate_brier_metrics(
                y_true,
                pred,
            )

            record = {
                "model_a": model_a,
                "model_b": model_b,
                "model_c": model_c,
                "weight_a":
                    float(w_a),
                "weight_b":
                    float(w_b),
                "weight_c":
                    float(w_c),
                "brier":
                    brier,
                "bss":
                    bss,
            }

            records.append(
                record
            )

            if (
                best is None
                or brier < best["brier"]
            ):
                best = {
                    **record,
                    "prediction":
                        pred.copy(),
                    "baseline_brier":
                        baseline_brier,
                }

    result = pd.DataFrame(
        records
    ).sort_values(
        "brier"
    ).reset_index(
        drop=True
    )

    return (
        result,
        best,
    )


# =========================================================
# Pair comparison
# =========================================================

def build_pair_comparison(
    pair_grid_best,
    pair_exact_best,
    prediction_corr,
    residual_corr,
):
    records = []

    exact_lookup = {
        (
            x["model_a"],
            x["model_b"],
        ): x
        for x in pair_exact_best
    }

    for grid in pair_grid_best:
        key = (
            grid["model_a"],
            grid["model_b"],
        )

        exact = (
            exact_lookup[key]
        )

        model_a = (
            grid["model_a"]
        )

        model_b = (
            grid["model_b"]
        )

        records.append({
            "model_a": model_a,
            "model_b": model_b,

            "prediction_correlation":
                float(
                    prediction_corr.loc[
                        model_a,
                        model_b,
                    ]
                ),

            "residual_correlation":
                float(
                    residual_corr.loc[
                        model_a,
                        model_b,
                    ]
                ),

            "grid_weight_a":
                float(
                    grid["weight_a"]
                ),
            "grid_weight_b":
                float(
                    grid["weight_b"]
                ),
            "grid_brier":
                float(
                    grid["brier"]
                ),
            "grid_bss":
                float(
                    grid["bss"]
                ),

            "exact_weight_a_unconstrained":
                float(
                    exact[
                        "weight_a_unconstrained"
                    ]
                ),
            "exact_weight_a":
                float(
                    exact["weight_a"]
                ),
            "exact_weight_b":
                float(
                    exact["weight_b"]
                ),
            "exact_brier":
                float(
                    exact["brier"]
                ),
            "exact_bss":
                float(
                    exact["bss"]
                ),

            "exact_brier_gain_vs_grid":
                float(
                    grid["brier"]
                    - exact["brier"]
                ),

            "exact_bss_gain_vs_grid":
                float(
                    exact["bss"]
                    - grid["bss"]
                ),
        })

    result = pd.DataFrame(
        records
    ).sort_values(
        "exact_brier"
    ).reset_index(
        drop=True
    )

    return result


# =========================================================
# Save
# =========================================================

def save_outputs(
    df,
    single_summary,
    prediction_corr,
    residual_corr,
    pair_grid_all,
    pair_grid_best,
    pair_exact_best,
    pair_comparison,
    triple_result,
    triple_best,
    global_best,
):
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Ensemble experiment already exists: {OUTPUT_DIR}\n"
            "EXPERIMENT_NAME을 변경하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    df.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    single_summary.to_csv(
        OUTPUT_DIR
        / "single_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_corr.to_csv(
        OUTPUT_DIR
        / "prediction_correlation.csv",
        encoding="utf-8-sig",
    )

    residual_corr.to_csv(
        OUTPUT_DIR
        / "residual_correlation.csv",
        encoding="utf-8-sig",
    )

    pair_grid_all.to_csv(
        OUTPUT_DIR
        / "pair_grid_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        pair_grid_best
    ).sort_values(
        "brier"
    ).to_csv(
        OUTPUT_DIR
        / "pair_grid_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    exact_save_records = []

    for item in pair_exact_best:
        exact_save_records.append({
            k: v
            for k, v in item.items()
            if k not in {
                "prediction",
                "baseline_brier",
            }
        })

    pd.DataFrame(
        exact_save_records
    ).sort_values(
        "brier"
    ).to_csv(
        OUTPUT_DIR
        / "pair_exact_optimum.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_comparison.to_csv(
        OUTPUT_DIR
        / "pair_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    triple_result.to_csv(
        OUTPUT_DIR
        / "triple_grid_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_pred_df = pd.DataFrame({
        "row_id":
            df["row_id"],
        "target":
            df["target"],
        "prediction":
            global_best[
                "prediction"
            ],
    })

    best_pred_df.to_csv(
        OUTPUT_DIR
        / "best_prediction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "best_prediction.npz",
        row_id=df[
            "row_id"
        ].to_numpy(),
        target=df[
            "target"
        ].to_numpy(),
        prediction=global_best[
            "prediction"
        ],
    )

    metadata = {
        "experiment_name":
            EXPERIMENT_NAME,
        "n_validation":
            int(len(df)),
        "pair_weight_step":
            PAIR_WEIGHT_STEP,
        "triple_weight_step":
            TRIPLE_WEIGHT_STEP,
        "single_models":
            single_summary.to_dict(
                orient="records"
            ),
        "pair_comparison":
            pair_comparison.to_dict(
                orient="records"
            ),
        "triple_best": {
            k: v
            for k, v in (
                triple_best.items()
            )
            if k != "prediction"
        },
        "global_best": {
            k: v
            for k, v in (
                global_best.items()
            )
            if k != "prediction"
        },
    }

    with open(
        OUTPUT_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 76)
    print("Validation Ensemble Search V2")
    print("=" * 76)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    print()
    print("[1] Prediction 로드")

    df = merge_predictions(
        MODEL_CONFIG
    )

    model_names = list(
        MODEL_CONFIG.keys()
    )

    print()
    print(
        f"Validation rows: {len(df):,}"
    )

    print(
        f"Positive rate  : "
        f"{df['target'].mean():.6f}"
    )

    # -----------------------------------------------------
    # Single
    # -----------------------------------------------------

    print()
    print("[2] Single model 성능")

    (
        single_summary,
        baseline_brier,
    ) = evaluate_single_models(
        df,
        model_names,
    )

    print()
    print(
        single_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print()
    print(
        f"Baseline Brier: "
        f"{baseline_brier:.8f}"
    )

    # -----------------------------------------------------
    # Correlations
    # -----------------------------------------------------

    print()
    print("[3] Prediction / Residual correlation")

    (
        prediction_corr,
        residual_corr,
    ) = calculate_correlations(
        df,
        model_names,
    )

    print()
    print("[Prediction correlation]")
    print(
        prediction_corr.to_string(
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("[Residual correlation]")
    print(
        residual_corr.to_string(
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # -----------------------------------------------------
    # Pair grid / exact
    # -----------------------------------------------------

    print()
    print("[4] Pair ensemble grid + exact optimum")

    pair_grid_results = []
    pair_grid_best = []
    pair_exact_best = []

    for (
        model_a,
        model_b,
    ) in combinations(
        model_names,
        2,
    ):
        (
            grid_result,
            grid_best,
        ) = search_pair_grid(
            df=df,
            model_a=model_a,
            model_b=model_b,
            step=PAIR_WEIGHT_STEP,
        )

        exact_best = (
            search_pair_exact(
                df=df,
                model_a=model_a,
                model_b=model_b,
            )
        )

        pair_grid_results.append(
            grid_result
        )

        pair_grid_best.append({
            k: v
            for k, v in grid_best.items()
            if k not in {
                "prediction",
                "baseline_brier",
            }
        })

        pair_exact_best.append(
            exact_best
        )

        print()
        print(
            f"{model_a} + {model_b}"
        )

        print(
            "  [Grid]"
        )

        print(
            f"  weights: "
            f"{model_a}={grid_best['weight_a']:.4f}, "
            f"{model_b}={grid_best['weight_b']:.4f}"
        )

        print(
            f"  Brier={grid_best['brier']:.10f}"
        )

        print(
            f"  BSS={grid_best['bss']:.4f}"
        )

        print(
            "  [Exact]"
        )

        print(
            f"  unconstrained "
            f"{model_a} weight="
            f"{exact_best['weight_a_unconstrained']:.8f}"
        )

        print(
            f"  weights: "
            f"{model_a}={exact_best['weight_a']:.8f}, "
            f"{model_b}={exact_best['weight_b']:.8f}"
        )

        print(
            f"  Brier={exact_best['brier']:.10f}"
        )

        print(
            f"  BSS={exact_best['bss']:.4f}"
        )

        print(
            f"  gain vs grid: "
            f"Brier "
            f"{grid_best['brier'] - exact_best['brier']:.12f}, "
            f"BSS "
            f"{exact_best['bss'] - grid_best['bss']:.6f}"
        )

        print(
            f"  pred corr="
            f"{prediction_corr.loc[model_a, model_b]:.6f}"
        )

        print(
            f"  residual corr="
            f"{residual_corr.loc[model_a, model_b]:.6f}"
        )

    pair_grid_all = pd.concat(
        pair_grid_results,
        ignore_index=True,
    ).sort_values(
        "brier"
    ).reset_index(
        drop=True
    )

    pair_comparison = (
        build_pair_comparison(
            pair_grid_best=
                pair_grid_best,
            pair_exact_best=
                pair_exact_best,
            prediction_corr=
                prediction_corr,
            residual_corr=
                residual_corr,
        )
    )

    # -----------------------------------------------------
    # Triple
    # -----------------------------------------------------

    print()
    print("[5] Triple ensemble grid")

    if len(model_names) != 3:
        raise ValueError(
            "현재 triple grid는 3개 모델 기준입니다."
        )

    (
        model_a,
        model_b,
        model_c,
    ) = model_names

    (
        triple_result,
        triple_best,
    ) = search_triple_grid(
        df=df,
        model_a=model_a,
        model_b=model_b,
        model_c=model_c,
        step=TRIPLE_WEIGHT_STEP,
    )

    print()
    print(
        f"weights: "
        f"{model_a}="
        f"{triple_best['weight_a']:.2f}, "
        f"{model_b}="
        f"{triple_best['weight_b']:.2f}, "
        f"{model_c}="
        f"{triple_best['weight_c']:.2f}"
    )

    print(
        f"Brier="
        f"{triple_best['brier']:.10f}"
    )

    print(
        f"BSS="
        f"{triple_best['bss']:.4f}"
    )

    # -----------------------------------------------------
    # Global best
    # -----------------------------------------------------

    candidates = []

    for exact in (
        pair_exact_best
    ):
        candidates.append({
            **exact,
            "ensemble_type":
                "pair_exact",
        })

    candidates.append({
        **triple_best,
        "ensemble_type":
            "triple_grid",
    })

    global_best = min(
        candidates,
        key=lambda x:
            x["brier"],
    )

    print()
    print("=" * 76)
    print("Global Best Ensemble")
    print("=" * 76)

    print(
        f"Type  : "
        f"{global_best['ensemble_type']}"
    )

    if (
        global_best[
            "ensemble_type"
        ]
        == "pair_exact"
    ):
        print(
            f"Models: "
            f"{global_best['model_a']} "
            f"+ {global_best['model_b']}"
        )

        print(
            f"Weights: "
            f"{global_best['weight_a']:.8f}, "
            f"{global_best['weight_b']:.8f}"
        )

    else:
        print(
            f"Models: "
            f"{global_best['model_a']} "
            f"+ {global_best['model_b']} "
            f"+ {global_best['model_c']}"
        )

        print(
            f"Weights: "
            f"{global_best['weight_a']:.4f}, "
            f"{global_best['weight_b']:.4f}, "
            f"{global_best['weight_c']:.4f}"
        )

    print(
        f"Brier : "
        f"{global_best['brier']:.10f}"
    )

    print(
        f"BSS   : "
        f"{global_best['bss']:.4f}"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_outputs(
        df=df,
        single_summary=
            single_summary,
        prediction_corr=
            prediction_corr,
        residual_corr=
            residual_corr,
        pair_grid_all=
            pair_grid_all,
        pair_grid_best=
            pair_grid_best,
        pair_exact_best=
            pair_exact_best,
        pair_comparison=
            pair_comparison,
        triple_result=
            triple_result,
        triple_best=
            triple_best,
        global_best=
            global_best,
    )

    print()
    print("[저장 완료]")

    for name in [
        "validation_predictions.csv",
        "single_model_summary.csv",
        "prediction_correlation.csv",
        "residual_correlation.csv",
        "pair_grid_search.csv",
        "pair_grid_best.csv",
        "pair_exact_optimum.csv",
        "pair_comparison.csv",
        "triple_grid_search.csv",
        "best_prediction.csv",
        "best_prediction.npz",
        "metadata.json",
    ]:
        print(
            f"- {OUTPUT_DIR / name}"
        )

    print("=" * 76)


if __name__ == "__main__":
    main()

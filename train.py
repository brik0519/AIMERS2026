# ensemble_search.py
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 경로 / 설정
# =========================================================
EXPERIMENT_NAME = "ensemble_v009"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
OUTPUT_DIR = VALIDATION_DIR

MODEL_CONFIG = {
    "cb_v002": {
        "path": VALIDATION_ROOT / "cb_v002" / "oof_pred.csv",
        "prediction_candidates": ["prediction"],
    },
    "xgb_v002": {
        "path": VALIDATION_ROOT / "xgb_v002" / "oof_pred.csv",
        "prediction_candidates": ["prediction"],
    },
    "mlp_v003_hybrid": {
        "path": VALIDATION_ROOT / "mlp_v003_hybrid" / "oof_pred.csv",
        "prediction_candidates": ["prediction"],
    },
}

BASELINE_PAIR = ("cb_v002", "xgb_v002")
INCREMENTAL_MODEL = "mlp_v003_hybrid"
TRIPLE_MODELS = (*BASELINE_PAIR, INCREMENTAL_MODEL)

# Hybrid MLP OOF만 신규 사용한다.
# CatBoost/XGBoost는 cb_v002, xgb_v002를 그대로 유지한다.

PAIR_WEIGHT_STEP = 0.01
TRIPLE_WEIGHT_STEP = 0.02
EPS = 1e-12

# 2023은 weight optimization에서만 제외한다.
# 최종 고정 weight 평가는 2021~2024 전체 OOF에서 다시 수행한다.
EXCLUDED_OPTIMIZATION_SEASONS = [2023]

# =========================================================
# Metric
# =========================================================
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true와 y_pred 길이가 다릅니다.")

    if not np.isfinite(y_true).all():
        raise ValueError("target에 NaN 또는 Inf가 있습니다.")

    if not np.isfinite(y_pred).all():
        raise ValueError("prediction에 NaN 또는 Inf가 있습니다.")

    target_mean = float(y_true.mean())
    prediction_mean = float(y_pred.mean())

    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(target_mean * (1.0 - target_mean))

    if baseline_brier <= 0:
        raise ValueError("baseline_brier가 0 이하입니다.")

    raw_skill = float(
        100000.0 * (1.0 - brier / baseline_brier)
    )
    competition_bss = float(max(0.0, raw_skill))

    p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    logloss = float(
        -np.mean(
            y_true * np.log(p)
            + (1.0 - y_true) * np.log(1.0 - p)
        )
    )

    return {
        "brier": brier,
        "baseline_brier": baseline_brier,
        "raw_skill": raw_skill,
        "competition_bss": competition_bss,
        "logloss": logloss,
        "target_mean": target_mean,
        "prediction_mean": prediction_mean,
        "bias": float(prediction_mean - target_mean),
        "prediction_std": float(y_pred.std()),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
    }


# =========================================================
# Prediction load / merge
# =========================================================
def detect_prediction_column(df, candidates, model_name):
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"{model_name}: prediction 컬럼을 찾지 못했습니다. "
        f"columns={list(df.columns)}, candidates={candidates}"
    )


def load_prediction(model_name, config):
    path = Path(config["path"])

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction 파일이 없습니다: {path}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    required_cols = {
        "row_id",
        "target",
        "validation_season",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name}: 필수 컬럼 누락: {missing}"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: row_id 중복이 있습니다."
        )

    pred_col = detect_prediction_column(
        df,
        config["prediction_candidates"],
        model_name,
    )

    out = df[
        [
            "row_id",
            "target",
            "validation_season",
            pred_col,
        ]
    ].copy()

    out = out.rename(
        columns={pred_col: model_name}
    )

    out["target"] = pd.to_numeric(
        out["target"],
        errors="raise",
    )

    out["validation_season"] = pd.to_numeric(
        out["validation_season"],
        errors="raise",
    ).astype(int)

    out[model_name] = pd.to_numeric(
        out[model_name],
        errors="raise",
    )

    if not np.isfinite(
        out[model_name].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            f"{model_name}: prediction에 NaN 또는 Inf가 있습니다."
        )

    print(
        f"[로드] {model_name:<10} | rows={len(out):,} "
        f"| seasons={sorted(out['validation_season'].unique().tolist())} "
        f"| column={pred_col} | path={path}"
    )

    return out


def merge_predictions(model_config):
    merged = None
    expected_n = None
    row_id_sets = {}

    for model_name, config in model_config.items():
        df = load_prediction(
            model_name,
            config,
        )

        if expected_n is None:
            expected_n = len(df)
        elif len(df) != expected_n:
            raise ValueError(
                "모델별 OOF row 수가 다릅니다. "
                f"expected={expected_n}, {model_name}={len(df)}"
            )

        row_id_sets[model_name] = set(
            df["row_id"]
        )

        df = df.rename(
            columns={
                "target": f"target__{model_name}",
                "validation_season":
                    f"validation_season__{model_name}",
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

    model_names = list(
        model_config.keys()
    )

    base_set = row_id_sets[
        model_names[0]
    ]

    for model_name in model_names[1:]:
        if (
            row_id_sets[model_name]
            != base_set
        ):
            raise ValueError(
                "row_id 집합 불일치: "
                f"{model_names[0]} vs {model_name}"
            )

    if len(merged) != expected_n:
        raise ValueError(
            f"merge row 수 불일치: "
            f"expected={expected_n}, merged={len(merged)}"
        )

    target_cols = [
        column
        for column in merged.columns
        if column.startswith("target__")
    ]

    season_cols = [
        column
        for column in merged.columns
        if column.startswith(
            "validation_season__"
        )
    ]

    base_target = merged[
        target_cols[0]
    ].to_numpy()

    for column in target_cols[1:]:
        other = merged[
            column
        ].to_numpy()

        if not np.array_equal(
            base_target,
            other,
        ):
            mismatch = int(
                np.sum(
                    base_target != other
                )
            )
            raise ValueError(
                "target 불일치: "
                f"{target_cols[0]} vs {column}, "
                f"mismatch={mismatch}"
            )

    base_season = merged[
        season_cols[0]
    ].to_numpy()

    for column in season_cols[1:]:
        other = merged[
            column
        ].to_numpy()

        if not np.array_equal(
            base_season,
            other,
        ):
            mismatch = int(
                np.sum(
                    base_season != other
                )
            )
            raise ValueError(
                "validation_season 불일치: "
                f"{season_cols[0]} vs {column}, "
                f"mismatch={mismatch}"
            )

    merged["target"] = (
        base_target
    )

    merged["validation_season"] = (
        base_season.astype(int)
    )

    merged = merged.drop(
        columns=(
            target_cols
            + season_cols
        )
    )

    return (
        merged
        .sort_values(
            [
                "validation_season",
                "row_id",
            ]
        )
        .reset_index(drop=True)
    )


# =========================================================
# Single / correlation
# =========================================================
def evaluate_single_models(df, model_names):
    y = df["target"].to_numpy(dtype=np.float64)

    rows = []

    for model_name in model_names:
        pred = df[model_name].to_numpy(dtype=np.float64)

        rows.append({
            "model": model_name,
            **calculate_metrics(y, pred),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("brier")
        .reset_index(drop=True)
    )


def calculate_correlations(df, model_names):
    prediction_corr = df[model_names].corr(
        method="pearson"
    )

    y = df["target"].to_numpy(dtype=np.float64)

    residual_df = pd.DataFrame({
        model_name:
            df[model_name].to_numpy(dtype=np.float64) - y
        for model_name in model_names
    })

    residual_corr = residual_df.corr(
        method="pearson"
    )

    return prediction_corr, residual_corr


# =========================================================
# Fold analysis
# =========================================================
def evaluate_single_models_by_fold(
    df,
    model_names,
):
    rows = []

    for season in sorted(
        df["validation_season"].unique()
    ):
        fold_df = df[
            df["validation_season"] == season
        ]

        fold_summary = (
            evaluate_single_models(
                fold_df,
                model_names,
            )
        )

        fold_summary.insert(
            0,
            "validation_season",
            int(season),
        )

        rows.append(
            fold_summary
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def calculate_fold_correlations(
    df,
    model_names,
):
    prediction_rows = []
    residual_rows = []

    for season in sorted(
        df["validation_season"].unique()
    ):
        fold_df = df[
            df["validation_season"] == season
        ]

        prediction_corr, residual_corr = (
            calculate_correlations(
                fold_df,
                model_names,
            )
        )

        for model_a, model_b in combinations(
            model_names,
            2,
        ):
            prediction_rows.append({
                "validation_season":
                    int(season),
                "model_a": model_a,
                "model_b": model_b,
                "correlation": float(
                    prediction_corr.loc[
                        model_a,
                        model_b,
                    ]
                ),
            })

            residual_rows.append({
                "validation_season":
                    int(season),
                "model_a": model_a,
                "model_b": model_b,
                "correlation": float(
                    residual_corr.loc[
                        model_a,
                        model_b,
                    ]
                ),
            })

    return (
        pd.DataFrame(
            prediction_rows
        ),
        pd.DataFrame(
            residual_rows
        ),
    )


def search_fold_exact_optimum(
    df,
    model_names,
):
    rows = []

    model_a, model_b, model_c = (
        model_names
    )

    for season in sorted(
        df["validation_season"].unique()
    ):
        fold_df = df[
            df["validation_season"] == season
        ]

        pair_results = []

        for pair_a, pair_b in combinations(
            model_names,
            2,
        ):
            result = search_pair_exact(
                fold_df,
                pair_a,
                pair_b,
            )

            pair_results.append({
                "validation_season":
                    int(season),
                "ensemble_type":
                    "pair_exact",
                "models":
                    f"{pair_a}+{pair_b}",
                "weights": {
                    pair_a:
                        float(
                            result[
                                "weight_a"
                            ]
                        ),
                    pair_b:
                        float(
                            result[
                                "weight_b"
                            ]
                        ),
                },
                "brier":
                    float(
                        result[
                            "brier"
                        ]
                    ),
                "raw_skill":
                    float(
                        result[
                            "raw_skill"
                        ]
                    ),
                "competition_bss":
                    float(
                        result[
                            "competition_bss"
                        ]
                    ),
            })

        triple = search_triple_exact(
            fold_df,
            model_a,
            model_b,
            model_c,
        )

        triple_record = {
            "validation_season":
                int(season),
            "ensemble_type":
                "triple_exact",
            "models":
                (
                    f"{model_a}+"
                    f"{model_b}+"
                    f"{model_c}"
                ),
            "weights": {
                model_a:
                    float(
                        triple[
                            "weight_a"
                        ]
                    ),
                model_b:
                    float(
                        triple[
                            "weight_b"
                        ]
                    ),
                model_c:
                    float(
                        triple[
                            "weight_c"
                        ]
                    ),
            },
            "brier":
                float(
                    triple[
                        "brier"
                    ]
                ),
            "raw_skill":
                float(
                    triple[
                        "raw_skill"
                    ]
                ),
            "competition_bss":
                float(
                    triple[
                        "competition_bss"
                    ]
                ),
        }

        rows.extend(
            pair_results
        )
        rows.append(
            triple_record
        )

    result = pd.DataFrame(
        rows
    )

    result["weights"] = result[
        "weights"
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return result


def apply_fixed_weights(
    df,
    weights,
):
    prediction = np.zeros(
        len(df),
        dtype=np.float64,
    )

    for model_name, weight in (
        weights.items()
    ):
        prediction += (
            float(weight)
            * df[
                model_name
            ].to_numpy(
                dtype=np.float64
            )
        )

    return prediction


def evaluate_fixed_weights_by_fold(
    df,
    weights,
    label,
):
    rows = []

    for season in sorted(
        df["validation_season"].unique()
    ):
        fold_df = df[
            df["validation_season"] == season
        ]

        prediction = (
            apply_fixed_weights(
                fold_df,
                weights,
            )
        )

        metrics = (
            calculate_metrics(
                fold_df[
                    "target"
                ].to_numpy(
                    dtype=np.float64
                ),
                prediction,
            )
        )

        rows.append({
            "label": label,
            "validation_season":
                int(season),
            "n_validation":
                int(
                    len(fold_df)
                ),
            **metrics,
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# Pair
# =========================================================
def pair_exact_from_arrays(y, pred_a, pred_b):
    y = np.asarray(y, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)

    delta = pred_a - pred_b
    denom = float(np.dot(delta, delta))

    if denom <= EPS:
        w_unconstrained = 0.5
    else:
        w_unconstrained = float(
            np.dot(delta, y - pred_b) / denom
        )

    w_a = float(np.clip(w_unconstrained, 0.0, 1.0))
    w_b = float(1.0 - w_a)

    pred = w_a * pred_a + w_b * pred_b

    return {
        "weight_a_unconstrained":
            w_unconstrained,
        "weight_a":
            w_a,
        "weight_b":
            w_b,
        **calculate_metrics(y, pred),
        "prediction":
            pred.copy(),
    }


def search_pair_exact(df, model_a, model_b):
    y = df["target"].to_numpy(dtype=np.float64)

    result = pair_exact_from_arrays(
        y,
        df[model_a].to_numpy(dtype=np.float64),
        df[model_b].to_numpy(dtype=np.float64),
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        **result,
    }


def search_pair_grid(
    df,
    model_a,
    model_b,
    step,
):
    y = df["target"].to_numpy(dtype=np.float64)
    pred_a = df[model_a].to_numpy(dtype=np.float64)
    pred_b = df[model_b].to_numpy(dtype=np.float64)

    n_steps = int(round(1.0 / step))

    rows = []
    best = None

    for i in range(n_steps + 1):
        w_a = float(i / n_steps)
        w_b = float(1.0 - w_a)

        pred = w_a * pred_a + w_b * pred_b

        record = {
            "model_a": model_a,
            "model_b": model_b,
            "weight_a": w_a,
            "weight_b": w_b,
            **calculate_metrics(y, pred),
        }

        rows.append(record)

        if (
            best is None
            or record["brier"] < best["brier"]
        ):
            best = {
                **record,
                "prediction": pred.copy(),
            }

    return (
        pd.DataFrame(rows)
        .sort_values("brier")
        .reset_index(drop=True),
        best,
    )


# =========================================================
# Triple grid
# =========================================================
def search_triple_grid(
    df,
    model_a,
    model_b,
    model_c,
    step,
):
    y = df["target"].to_numpy(dtype=np.float64)

    pa = df[model_a].to_numpy(dtype=np.float64)
    pb = df[model_b].to_numpy(dtype=np.float64)
    pc = df[model_c].to_numpy(dtype=np.float64)

    n_steps = int(round(1.0 / step))

    rows = []
    best = None

    for i in range(n_steps + 1):
        for j in range(n_steps - i + 1):
            k = n_steps - i - j

            wa = float(i / n_steps)
            wb = float(j / n_steps)
            wc = float(k / n_steps)

            pred = wa * pa + wb * pb + wc * pc

            record = {
                "model_a": model_a,
                "model_b": model_b,
                "model_c": model_c,
                "weight_a": wa,
                "weight_b": wb,
                "weight_c": wc,
                **calculate_metrics(y, pred),
            }

            rows.append(record)

            if (
                best is None
                or record["brier"] < best["brier"]
            ):
                best = {
                    **record,
                    "prediction": pred.copy(),
                }

    return (
        pd.DataFrame(rows)
        .sort_values("brier")
        .reset_index(drop=True),
        best,
    )


# =========================================================
# Triple exact convex optimum
# =========================================================
def search_triple_exact(
    df,
    model_a,
    model_b,
    model_c,
):
    """
    1) unconstrained solution이 simplex 내부면 채택.
    2) simplex 외부면 AB / AC / BC 세 edge의
       pair exact optimum 계산.
    3) 가장 Brier가 작은 edge solution을 채택.
    """

    y = df["target"].to_numpy(dtype=np.float64)

    pa = df[model_a].to_numpy(dtype=np.float64)
    pb = df[model_b].to_numpy(dtype=np.float64)
    pc = df[model_c].to_numpy(dtype=np.float64)

    # pc를 baseline으로 둔다.
    design = np.column_stack([
        pa - pc,
        pb - pc,
    ])

    target = y - pc

    solution, _, _, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    wa_u = float(solution[0])
    wb_u = float(solution[1])
    wc_u = float(1.0 - wa_u - wb_u)

    pred_u = (
        wa_u * pa
        + wb_u * pb
        + wc_u * pc
    )

    unconstrained_metrics = calculate_metrics(
        y,
        pred_u,
    )

    inside = (
        wa_u >= -EPS
        and wb_u >= -EPS
        and wc_u >= -EPS
        and wa_u <= 1.0 + EPS
        and wb_u <= 1.0 + EPS
        and wc_u <= 1.0 + EPS
    )

    if inside:
        weights = np.array(
            [
                max(0.0, wa_u),
                max(0.0, wb_u),
                max(0.0, wc_u),
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()

        wa, wb, wc = map(float, weights)

        pred = (
            wa * pa
            + wb * pb
            + wc * pc
        )

        return {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
            "solution_type": "interior",
            "inside_simplex": True,
            "selected_edge": None,
            "weight_a_unconstrained": wa_u,
            "weight_b_unconstrained": wb_u,
            "weight_c_unconstrained": wc_u,
            "unconstrained_brier":
                unconstrained_metrics["brier"],
            "weight_a": wa,
            "weight_b": wb,
            "weight_c": wc,
            **calculate_metrics(y, pred),
            "prediction": pred.copy(),
        }

    # simplex 밖: 세 edge exact optimum
    edges = []

    # AB edge: C = 0
    ab = pair_exact_from_arrays(y, pa, pb)
    edges.append({
        "edge": f"{model_a}+{model_b}",
        "weight_a": ab["weight_a"],
        "weight_b": ab["weight_b"],
        "weight_c": 0.0,
        "brier": ab["brier"],
        "prediction": ab["prediction"],
    })

    # AC edge: B = 0
    ac = pair_exact_from_arrays(y, pa, pc)
    edges.append({
        "edge": f"{model_a}+{model_c}",
        "weight_a": ac["weight_a"],
        "weight_b": 0.0,
        "weight_c": ac["weight_b"],
        "brier": ac["brier"],
        "prediction": ac["prediction"],
    })

    # BC edge: A = 0
    bc = pair_exact_from_arrays(y, pb, pc)
    edges.append({
        "edge": f"{model_b}+{model_c}",
        "weight_a": 0.0,
        "weight_b": bc["weight_a"],
        "weight_c": bc["weight_b"],
        "brier": bc["brier"],
        "prediction": bc["prediction"],
    })

    best_edge = min(
        edges,
        key=lambda x: x["brier"],
    )

    wa = float(best_edge["weight_a"])
    wb = float(best_edge["weight_b"])
    wc = float(best_edge["weight_c"])

    pred = best_edge["prediction"]

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "solution_type": "edge",
        "inside_simplex": False,
        "selected_edge": best_edge["edge"],
        "weight_a_unconstrained": wa_u,
        "weight_b_unconstrained": wb_u,
        "weight_c_unconstrained": wc_u,
        "unconstrained_brier":
            unconstrained_metrics["brier"],
        "weight_a": wa,
        "weight_b": wb,
        "weight_c": wc,
        **calculate_metrics(y, pred),
        "prediction": pred.copy(),
    }


# =========================================================
# Pair comparison / MLP incremental value
# =========================================================
def build_pair_comparison(
    pair_grid_best,
    pair_exact_best,
    prediction_corr,
    residual_corr,
    single_summary,
):
    single = (
        single_summary
        .set_index("model")
        .to_dict(orient="index")
    )

    exact_lookup = {
        (x["model_a"], x["model_b"]): x
        for x in pair_exact_best
    }

    rows = []

    for grid in pair_grid_best:
        a = grid["model_a"]
        b = grid["model_b"]
        exact = exact_lookup[(a, b)]

        best_single_brier = min(
            single[a]["brier"],
            single[b]["brier"],
        )

        best_single_raw_skill = max(
            single[a]["raw_skill"],
            single[b]["raw_skill"],
        )

        rows.append({
            "model_a": a,
            "model_b": b,
            "prediction_correlation":
                float(prediction_corr.loc[a, b]),
            "residual_correlation":
                float(residual_corr.loc[a, b]),
            "grid_weight_a": grid["weight_a"],
            "grid_weight_b": grid["weight_b"],
            "grid_brier": grid["brier"],
            "grid_raw_skill": grid["raw_skill"],
            "exact_weight_a_unconstrained":
                exact["weight_a_unconstrained"],
            "exact_weight_a": exact["weight_a"],
            "exact_weight_b": exact["weight_b"],
            "exact_brier": exact["brier"],
            "exact_raw_skill": exact["raw_skill"],
            "exact_competition_bss":
                exact["competition_bss"],
            "exact_logloss": exact["logloss"],
            "exact_bias": exact["bias"],
            "exact_brier_gain_vs_grid":
                grid["brier"] - exact["brier"],
            "exact_raw_skill_gain_vs_grid":
                exact["raw_skill"] - grid["raw_skill"],
            "exact_brier_gain_vs_best_single":
                best_single_brier - exact["brier"],
            "exact_raw_skill_gain_vs_best_single":
                exact["raw_skill"]
                - best_single_raw_skill,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("exact_brier")
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        np.arange(1, len(result) + 1),
    )

    return result


def find_pair_exact(
    pair_exact_best,
    model_a,
    model_b,
):
    wanted = {model_a, model_b}

    for item in pair_exact_best:
        if {
            item["model_a"],
            item["model_b"],
        } == wanted:
            return item

    raise ValueError(
        f"pair exact 결과 없음: {model_a}+{model_b}"
    )


def evaluate_incremental_value(
    baseline_pair,
    triple_exact,
    incremental_model,
):
    weights = {
        triple_exact["model_a"]:
            triple_exact["weight_a"],
        triple_exact["model_b"]:
            triple_exact["weight_b"],
        triple_exact["model_c"]:
            triple_exact["weight_c"],
    }

    incremental_weight = float(weights[incremental_model])

    delta_brier = float(
        triple_exact["brier"]
        - baseline_pair["brier"]
    )

    delta_raw_skill = float(
        triple_exact["raw_skill"]
        - baseline_pair["raw_skill"]
    )

    if (
        delta_brier < -EPS
        and incremental_weight > 1e-8
    ):
        decision = "GO"
    elif (
        delta_brier < -EPS
        and incremental_weight <= 1e-8
    ):
        decision = "NO-GO_INCREMENTAL_WEIGHT_ZERO"
    elif (
        abs(delta_brier) <= EPS
        and incremental_weight > 1e-8
    ):
        decision = "CONDITIONAL_GO"
    else:
        decision = "NO-GO"

    return {
        "baseline_pair":
            f"{baseline_pair['model_a']}+"
            f"{baseline_pair['model_b']}",
        "baseline_pair_brier":
            float(baseline_pair["brier"]),
        "baseline_pair_raw_skill":
            float(baseline_pair["raw_skill"]),
        "triple_brier":
            float(triple_exact["brier"]),
        "triple_raw_skill":
            float(triple_exact["raw_skill"]),
        "delta_brier_triple_minus_baseline":
            delta_brier,
        "delta_raw_skill_triple_minus_baseline":
            delta_raw_skill,
        "incremental_model":
            incremental_model,
        "incremental_weight":
            incremental_weight,
        "decision":
            decision,
    }


# =========================================================
# Save
# =========================================================
def strip_prediction(item):
    return {
        k: v
        for k, v in item.items()
        if k != "prediction"
    }


def json_safe(obj):
    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_outputs(
    full_df,
    search_df,
    full_single_summary,
    optimization_single_summary,
    fold_single_summary,
    prediction_corr,
    residual_corr,
    fold_prediction_corr,
    fold_residual_corr,
    pair_grid_all,
    pair_grid_best,
    pair_exact_best,
    pair_comparison,
    triple_grid_result,
    triple_grid_best,
    triple_exact,
    incremental_value,
    fold_exact_optimum,
    fold_ensemble_performance,
    optimization_best,
    full_oof_metrics,
    full_prediction,
):
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Ensemble experiment already exists: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # OOF source / optimization subset
    # -----------------------------------------------------
    full_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    search_df.to_csv(
        OUTPUT_DIR / "optimization_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_single_summary.to_csv(
        OUTPUT_DIR / "single_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    optimization_single_summary.to_csv(
        OUTPUT_DIR / "optimization_single_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_single_summary.to_csv(
        OUTPUT_DIR / "fold_single_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Correlation
    # optimization correlation은 2023 제외 OOF 기준.
    # Fold correlation은 전체 OOF 기준.
    # -----------------------------------------------------
    prediction_corr.to_csv(
        OUTPUT_DIR / "prediction_correlation.csv",
        encoding="utf-8-sig",
    )

    residual_corr.to_csv(
        OUTPUT_DIR / "residual_correlation.csv",
        encoding="utf-8-sig",
    )

    fold_prediction_corr.to_csv(
        OUTPUT_DIR / "fold_prediction_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_residual_corr.to_csv(
        OUTPUT_DIR / "fold_residual_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Pair / Triple optimization
    # -----------------------------------------------------
    pair_grid_all.to_csv(
        OUTPUT_DIR / "pair_grid_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        pair_grid_best
    ).sort_values(
        "brier"
    ).to_csv(
        OUTPUT_DIR / "pair_grid_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_exact_df = pd.DataFrame([
        strip_prediction(item)
        for item in pair_exact_best
    ]).sort_values(
        "brier"
    ).reset_index(drop=True)

    pair_exact_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(pair_exact_df) + 1,
        ),
    )

    pair_exact_df.to_csv(
        OUTPUT_DIR / "pair_exact_optimum.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_comparison.to_csv(
        OUTPUT_DIR / "pair_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    triple_grid_result.to_csv(
        OUTPUT_DIR / "triple_grid_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        strip_prediction(
            triple_grid_best
        )
    ]).to_csv(
        OUTPUT_DIR / "triple_grid_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        strip_prediction(
            triple_exact
        )
    ]).to_csv(
        OUTPUT_DIR / "triple_exact_optimum.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        incremental_value
    ]).to_csv(
        OUTPUT_DIR / "incremental_value.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Temporal diagnostics
    # -----------------------------------------------------
    fold_exact_optimum.to_csv(
        OUTPUT_DIR / "fold_exact_optimum.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_ensemble_performance.to_csv(
        OUTPUT_DIR / "fold_ensemble_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 새 weight를 전체 2021~2024 OOF에 고정 적용한 결과
    # -----------------------------------------------------
    pd.DataFrame({
        "row_id": full_df["row_id"],
        "target": full_df["target"],
        "validation_season":
            full_df["validation_season"],
        "prediction": full_prediction,
    }).to_csv(
        OUTPUT_DIR / "best_prediction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        OUTPUT_DIR / "best_prediction.npz",
        row_id=full_df[
            "row_id"
        ].to_numpy(),
        target=full_df[
            "target"
        ].to_numpy(),
        validation_season=full_df[
            "validation_season"
        ].to_numpy(),
        prediction=full_prediction,
    )

    optimization_seasons = sorted(
        [
            int(value)
            for value in search_df[
                "validation_season"
            ].unique()
        ]
    )

    evaluation_seasons = sorted(
        [
            int(value)
            for value in full_df[
                "validation_season"
            ].unique()
        ]
    )

    metadata = {
        "experiment_name":
            EXPERIMENT_NAME,
        "validation_method":
            "expanding_oof",
        "optimization_policy":
            "exclude_validation_season",
        "optimization_seasons":
            optimization_seasons,
        "excluded_optimization_seasons": [
            int(value)
            for value
            in EXCLUDED_OPTIMIZATION_SEASONS
        ],
        "evaluation_seasons":
            evaluation_seasons,
        "n_optimization":
            int(len(search_df)),
        "n_evaluation":
            int(len(full_df)),
        "optimization_target_mean":
            float(
                search_df[
                    "target"
                ].mean()
            ),
        "evaluation_target_mean":
            float(
                full_df[
                    "target"
                ].mean()
            ),
        "pair_weight_step":
            PAIR_WEIGHT_STEP,
        "triple_weight_step":
            TRIPLE_WEIGHT_STEP,
        "models":
            list(
                MODEL_CONFIG.keys()
            ),
        "source_models": {
            model_name:
                str(
                    config[
                        "path"
                    ]
                )
            for model_name, config
            in MODEL_CONFIG.items()
        },
        "single_models_full_oof":
            full_single_summary.to_dict(
                orient="records"
            ),
        "single_models_optimization":
            optimization_single_summary.to_dict(
                orient="records"
            ),
        "pair_comparison":
            pair_comparison.to_dict(
                orient="records"
            ),
        "triple_grid_best":
            strip_prediction(
                triple_grid_best
            ),
        "triple_exact":
            strip_prediction(
                triple_exact
            ),
        "incremental_value":
            incremental_value,
        "global_best": {
            "ensemble_type":
                optimization_best[
                    "ensemble_type"
                ],
            "models":
                optimization_best[
                    "models"
                ],
            "weights":
                optimization_best[
                    "weights"
                ],
            "optimization_brier":
                float(
                    optimization_best[
                        "brier"
                    ]
                ),
            "optimization_raw_skill":
                float(
                    optimization_best[
                        "raw_skill"
                    ]
                ),
            "optimization_competition_bss":
                float(
                    optimization_best[
                        "competition_bss"
                    ]
                ),
            "full_oof_brier":
                float(
                    full_oof_metrics[
                        "brier"
                    ]
                ),
            "full_oof_raw_skill":
                float(
                    full_oof_metrics[
                        "raw_skill"
                    ]
                ),
            "full_oof_competition_bss":
                float(
                    full_oof_metrics[
                        "competition_bss"
                    ]
                ),
        },
        "fold_ensemble_performance":
            fold_ensemble_performance.to_dict(
                orient="records"
            ),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                metadata
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Main
# =========================================================
def main():

    print("=" * 80)
    print("LG Aimers Phase2 Ensemble v005 Search")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. 전체 OOF 로드
    # -----------------------------------------------------
    print(
        "\n[1] Expanding OOF 로드 / alignment audit"
    )

    full_df = merge_predictions(
        MODEL_CONFIG
    )

    model_names = list(
        MODEL_CONFIG.keys()
    )

    full_seasons = sorted(
        full_df[
            "validation_season"
        ].unique().tolist()
    )

    print(
        f"Full OOF rows        : "
        f"{len(full_df):,}"
    )

    print(
        f"Full positive rate   : "
        f"{full_df['target'].mean():.6f}"
    )

    print(
        f"Evaluation seasons   : "
        f"{full_seasons}"
    )

    print(
        "PASS: row_id / target / "
        "validation_season alignment"
    )

    # -----------------------------------------------------
    # 2. 2023은 optimization에서만 제외
    # -----------------------------------------------------
    search_df = full_df[
        ~full_df[
            "validation_season"
        ].isin(
            EXCLUDED_OPTIMIZATION_SEASONS
        )
    ].copy()

    optimization_seasons = sorted(
        search_df[
            "validation_season"
        ].unique().tolist()
    )

    if len(search_df) == 0:
        raise ValueError(
            "Optimization OOF가 비어 있습니다."
        )

    if (
        set(
            EXCLUDED_OPTIMIZATION_SEASONS
        )
        & set(
            optimization_seasons
        )
    ):
        raise RuntimeError(
            "제외 시즌이 optimization data에 "
            "남아 있습니다."
        )

    print()
    print(
        "[2] Optimization subset"
    )

    print(
        f"Excluded seasons     : "
        f"{EXCLUDED_OPTIMIZATION_SEASONS}"
    )

    print(
        f"Optimization seasons : "
        f"{optimization_seasons}"
    )

    print(
        f"Optimization rows    : "
        f"{len(search_df):,}"
    )

    # -----------------------------------------------------
    # 3. Single model
    # -----------------------------------------------------
    print(
        "\n[3] Single model 비교"
    )

    full_single_summary = (
        evaluate_single_models(
            full_df,
            model_names,
        )
    )

    optimization_single_summary = (
        evaluate_single_models(
            search_df,
            model_names,
        )
    )

    fold_single_summary = (
        evaluate_single_models_by_fold(
            full_df,
            model_names,
        )
    )

    display_cols = [
        "model",
        "brier",
        "raw_skill",
        "competition_bss",
        "bias",
        "prediction_std",
    ]

    print(
        "\n[전체 2021~2024 OOF]"
    )

    print(
        full_single_summary[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print(
        "\n[2023 제외 Optimization OOF]"
    )

    print(
        optimization_single_summary[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    # -----------------------------------------------------
    # 4. Correlation
    #
    # Weight 탐색용 correlation은 search_df.
    # Fold별 diagnostic은 full_df.
    # -----------------------------------------------------
    print(
        "\n[4] Optimization Prediction / Residual correlation"
    )

    (
        prediction_corr,
        residual_corr,
    ) = calculate_correlations(
        search_df,
        model_names,
    )

    (
        fold_prediction_corr,
        fold_residual_corr,
    ) = calculate_fold_correlations(
        full_df,
        model_names,
    )

    print(
        "\n[Prediction correlation]"
    )

    print(
        prediction_corr.to_string(
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\n[Residual correlation]"
    )

    print(
        residual_corr.to_string(
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    # -----------------------------------------------------
    # 5. Pair Search - search_df only
    # -----------------------------------------------------
    print(
        "\n[5] Pair grid + exact optimum "
        "(2023 제외)"
    )

    pair_grid_results = []
    pair_grid_best = []
    pair_exact_best = []

    for model_a, model_b in combinations(
        model_names,
        2,
    ):
        (
            grid_result,
            grid_best,
        ) = search_pair_grid(
            search_df,
            model_a,
            model_b,
            PAIR_WEIGHT_STEP,
        )

        exact_best = (
            search_pair_exact(
                search_df,
                model_a,
                model_b,
            )
        )

        pair_grid_results.append(
            grid_result
        )

        pair_grid_best.append(
            strip_prediction(
                grid_best
            )
        )

        pair_exact_best.append(
            exact_best
        )

        print()
        print(
            f"{model_a} + {model_b}"
        )

        print(
            "Grid : "
            f"{grid_best['weight_a']:.4f} / "
            f"{grid_best['weight_b']:.4f} | "
            f"Brier={grid_best['brier']:.10f}"
        )

        print(
            "Exact: "
            f"{exact_best['weight_a']:.8f} / "
            f"{exact_best['weight_b']:.8f} | "
            f"Brier={exact_best['brier']:.10f}"
        )

    pair_grid_all = (
        pd.concat(
            pair_grid_results,
            ignore_index=True,
        )
        .sort_values(
            "brier"
        )
        .reset_index(
            drop=True
        )
    )

    pair_comparison = (
        build_pair_comparison(
            pair_grid_best,
            pair_exact_best,
            prediction_corr,
            residual_corr,
            optimization_single_summary,
        )
    )

    # -----------------------------------------------------
    # 6. Triple Search - search_df only
    # -----------------------------------------------------
    print(
        "\n[6] Triple grid "
        "(2023 제외)"
    )

    (
        model_a,
        model_b,
        model_c,
    ) = TRIPLE_MODELS

    (
        triple_grid_result,
        triple_grid_best,
    ) = search_triple_grid(
        search_df,
        model_a,
        model_b,
        model_c,
        TRIPLE_WEIGHT_STEP,
    )

    print(
        "Grid weights: "
        f"{model_a}="
        f"{triple_grid_best['weight_a']:.4f}, "
        f"{model_b}="
        f"{triple_grid_best['weight_b']:.4f}, "
        f"{model_c}="
        f"{triple_grid_best['weight_c']:.4f}"
    )

    print(
        f"Brier="
        f"{triple_grid_best['brier']:.10f}"
    )

    print(
        "\n[7] Triple exact convex optimum "
        "(2023 제외)"
    )

    triple_exact = (
        search_triple_exact(
            search_df,
            model_a,
            model_b,
            model_c,
        )
    )

    print(
        "Exact weights: "
        f"{model_a}="
        f"{triple_exact['weight_a']:.10f}, "
        f"{model_b}="
        f"{triple_exact['weight_b']:.10f}, "
        f"{model_c}="
        f"{triple_exact['weight_c']:.10f}"
    )

    print(
        f"Brier="
        f"{triple_exact['brier']:.10f} | "
        f"RawSkill="
        f"{triple_exact['raw_skill']:.6f} | "
        f"BSS="
        f"{triple_exact['competition_bss']:.6f}"
    )

    # -----------------------------------------------------
    # 8. MLP incremental value - search_df 기준
    # -----------------------------------------------------
    print(
        f"\n[8] {INCREMENTAL_MODEL} "
        "incremental value"
    )

    baseline_pair = (
        find_pair_exact(
            pair_exact_best,
            BASELINE_PAIR[0],
            BASELINE_PAIR[1],
        )
    )

    incremental_value = (
        evaluate_incremental_value(
            baseline_pair,
            triple_exact,
            INCREMENTAL_MODEL,
        )
    )

    for key, value in (
        incremental_value.items()
    ):
        print(
            f"{key}: {value}"
        )

    # -----------------------------------------------------
    # 9. Optimization global best
    # -----------------------------------------------------
    global_candidates = []

    for _, row in (
        optimization_single_summary
        .iterrows()
    ):
        model = row["model"]

        global_candidates.append({
            "ensemble_type":
                "single",
            "models":
                model,
            "brier":
                float(
                    row["brier"]
                ),
            "raw_skill":
                float(
                    row[
                        "raw_skill"
                    ]
                ),
            "competition_bss":
                float(
                    row[
                        "competition_bss"
                    ]
                ),
            "weights": {
                model: 1.0
            },
            "prediction":
                search_df[
                    model
                ].to_numpy(
                    dtype=np.float64
                ),
        })

    for item in pair_exact_best:
        global_candidates.append({
            "ensemble_type":
                "pair_exact",
            "models":
                (
                    f"{item['model_a']}+"
                    f"{item['model_b']}"
                ),
            "brier":
                item["brier"],
            "raw_skill":
                item[
                    "raw_skill"
                ],
            "competition_bss":
                item[
                    "competition_bss"
                ],
            "weights": {
                item["model_a"]:
                    item[
                        "weight_a"
                    ],
                item["model_b"]:
                    item[
                        "weight_b"
                    ],
            },
            "prediction":
                item[
                    "prediction"
                ],
        })

    global_candidates.append({
        "ensemble_type":
            "triple_exact",
        "models":
            (
                f"{model_a}+"
                f"{model_b}+"
                f"{model_c}"
            ),
        "brier":
            triple_exact[
                "brier"
            ],
        "raw_skill":
            triple_exact[
                "raw_skill"
            ],
        "competition_bss":
            triple_exact[
                "competition_bss"
            ],
        "weights": {
            model_a:
                triple_exact[
                    "weight_a"
                ],
            model_b:
                triple_exact[
                    "weight_b"
                ],
            model_c:
                triple_exact[
                    "weight_c"
                ],
        },
        "prediction":
            triple_exact[
                "prediction"
            ],
    })

    optimization_best = min(
        global_candidates,
        key=lambda item:
            item["brier"],
    )

    print(
        "\n[9] Optimization global best"
    )

    print(
        f"Type    : "
        f"{optimization_best['ensemble_type']}"
    )

    print(
        f"Models  : "
        f"{optimization_best['models']}"
    )

    print(
        f"Weights : "
        f"{optimization_best['weights']}"
    )

    print(
        f"Brier   : "
        f"{optimization_best['brier']:.10f}"
    )

    # -----------------------------------------------------
    # 10. 새 고정 weight를 전체 OOF에 재적용
    # -----------------------------------------------------
    full_prediction = (
        apply_fixed_weights(
            full_df,
            optimization_best[
                "weights"
            ],
        )
    )

    full_oof_metrics = (
        calculate_metrics(
            full_df[
                "target"
            ].to_numpy(
                dtype=np.float64
            ),
            full_prediction,
        )
    )

    print(
        "\n[10] Full 2021~2024 OOF 재평가"
    )

    print(
        f"Optimization Brier : "
        f"{optimization_best['brier']:.10f}"
    )

    print(
        f"Full OOF Brier     : "
        f"{full_oof_metrics['brier']:.10f}"
    )

    print(
        f"Full OOF BSS       : "
        f"{full_oof_metrics['competition_bss']:.6f}"
    )

    # -----------------------------------------------------
    # 11. Fold robustness
    # -----------------------------------------------------
    fold_exact_optimum = (
        search_fold_exact_optimum(
            full_df,
            model_names,
        )
    )

    fold_ensemble_performance = (
        evaluate_fixed_weights_by_fold(
            full_df,
            optimization_best[
                "weights"
            ],
            label=(
                "exclude_2023_fixed_weights"
            ),
        )
    )

    print(
        "\n[11] Fold temporal robustness"
    )

    print(
        fold_ensemble_performance[
            [
                "validation_season",
                "brier",
                "raw_skill",
                "competition_bss",
                "bias",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    # -----------------------------------------------------
    # 12. Save
    # -----------------------------------------------------
    save_outputs(
        full_df=(
            full_df
        ),
        search_df=(
            search_df
        ),
        full_single_summary=(
            full_single_summary
        ),
        optimization_single_summary=(
            optimization_single_summary
        ),
        fold_single_summary=(
            fold_single_summary
        ),
        prediction_corr=(
            prediction_corr
        ),
        residual_corr=(
            residual_corr
        ),
        fold_prediction_corr=(
            fold_prediction_corr
        ),
        fold_residual_corr=(
            fold_residual_corr
        ),
        pair_grid_all=(
            pair_grid_all
        ),
        pair_grid_best=(
            pair_grid_best
        ),
        pair_exact_best=(
            pair_exact_best
        ),
        pair_comparison=(
            pair_comparison
        ),
        triple_grid_result=(
            triple_grid_result
        ),
        triple_grid_best=(
            triple_grid_best
        ),
        triple_exact=(
            triple_exact
        ),
        incremental_value=(
            incremental_value
        ),
        fold_exact_optimum=(
            fold_exact_optimum
        ),
        fold_ensemble_performance=(
            fold_ensemble_performance
        ),
        optimization_best=(
            optimization_best
        ),
        full_oof_metrics=(
            full_oof_metrics
        ),
        full_prediction=(
            full_prediction
        ),
    )

    print(
        "\n[저장 완료]"
    )

    print(
        f"- {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

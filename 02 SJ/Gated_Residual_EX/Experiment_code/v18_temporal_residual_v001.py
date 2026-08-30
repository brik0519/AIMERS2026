# v18_temporal_residual_v001.py
#
# cb_reg_v003 OOF를 재사용해 참고깃허브 V18의 핵심인
# strict next-season conditional residual transfer를 검증한다.
#
# 공개 개발일지에서 확인되는 V18 핵심 조건:
#   1) pitcher_id x batter_hand
#   2) pitcher_id x pressure x batter_hand
#   3) 2022->2023, 2023->2024처럼 다음 시즌 전이에서 반복되는 신호를 우선
#
# 주의: 공개 개발일지에는 pressure의 정확한 정의와 최종 결합 수식은 없음.
# 여기서는 프로젝트용 pressure_state를 다음처럼 정의한다.
#   full   : 3-2
#   high   : balls==3 또는 strikes==2
#   normal : 그 외
# 따라서 이는 V18 공개 설명을 최대한 충실히 재현한 우리 검증 구현이다.
#
# 필요 파일:
#   ./data/train.csv
#   ./validation/cb_reg_v003/oof_validation.npz
#
# 실행:
#   python v18_temporal_residual_v001.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 설정
# =============================================================================

EXPERIMENT_NAME = "v18_temporal_residual_v001"
TRAIN_PATH = Path("./data/train.csv")
OOF_PATH = Path("./validation/cb_reg_v003/oof_validation.npz")
OUT_DIR = Path("./validation") / EXPERIMENT_NAME

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"

SCHEMAS = {
    "pitcher_x_batter_hand": ["pitcher_id", "batter_hand"],
    "pitcher_x_pressure_x_batter_hand": [
        "pitcher_id",
        "pressure_state",
        "batter_hand",
    ],
}

# 2024는 절대 파라미터 선택에 사용하지 않음.
DISCOVERY_TRANSITIONS = [(2021, 2022), (2022, 2023)]
CONFIRM_TRANSITION = (2023, 2024)
ALL_TRANSITIONS = DISCOVERY_TRANSITIONS + [CONFIRM_TRANSITION]

# 작은 사전 고정 grid.
MIN_N_GRID = [50, 100, 200]
SHRINK_K_GRID = [100.0, 300.0, 700.0]
ALPHA_GRID = [0.25, 0.50, 0.75]
CAP_GRID = [0.01, 0.02]
MIN_ABS_EFFECT = 0.003

# 2023 F 전체 level shift가 특정 pitcher residual로 섞이는 것을 줄이기 위한 보호 장치.
CENTER_BY_SEASON_GAME_TYPE = True


# =============================================================================
# Metric
# =============================================================================

def brier(y_true, pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean((y_true - pred) ** 2))


def metric_gain(y_true, raw_pred, corrected_pred):
    raw = brier(y_true, raw_pred)
    corr = brier(y_true, corrected_pred)
    return {
        "raw_brier": raw,
        "corrected_brier": corr,
        "gain": raw - corr,  # 양수면 개선
    }


# =============================================================================
# 데이터
# =============================================================================

def load_train():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    required = [
        ID_COL,
        TARGET_COL,
        SEASON_COL,
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    header = pd.read_csv(TRAIN_PATH, nrows=0, encoding="utf-8-sig")
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise KeyError("train.csv missing columns: " + ", ".join(missing))

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=required,
        encoding="utf-8-sig",
        low_memory=False,
    )

    train[SEASON_COL] = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)

    balls = pd.to_numeric(train["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(train["strikes_before"], errors="coerce").fillna(0).astype(int)

    train["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where((balls == 3) | (strikes == 2), "high", "normal"),
    )

    for col in ["game_type", "pitcher_id", "batter_hand", "pressure_state"]:
        train[col] = train[col].fillna("__MISSING__").astype(str)

    return train


def load_oof():
    if not OOF_PATH.exists():
        raise FileNotFoundError(OOF_PATH)

    with np.load(OOF_PATH, allow_pickle=True) as data:
        keys = set(data.files)
        required = {"row_id", "target", "validation_season"}
        missing = sorted(required - keys)
        if missing:
            raise KeyError("OOF missing keys: " + ", ".join(missing))

        if "prediction" in keys:
            pred = data["prediction"]
        elif "prediction_clip" in keys:
            pred = data["prediction_clip"]
        elif "prediction_raw" in keys:
            pred = np.clip(data["prediction_raw"], 0.0, 1.0)
        else:
            raise KeyError("OOF has no prediction/prediction_clip/prediction_raw")

        oof = pd.DataFrame(
            {
                ID_COL: data["row_id"],
                "oof_target": data["target"].astype(np.float64),
                "prediction": np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0),
                "validation_season": data["validation_season"].astype(int),
            }
        )

    if oof[ID_COL].duplicated().any():
        raise ValueError("OOF row_id is not unique")

    return oof


def build_frame():
    train = load_train()
    oof = load_oof()

    frame = oof.merge(
        train,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if frame[SEASON_COL].isna().any():
        raise ValueError("Some OOF rows were not found in train.csv")

    if not np.array_equal(
        frame[SEASON_COL].astype(int).to_numpy(),
        frame["validation_season"].astype(int).to_numpy(),
    ):
        raise ValueError("validation_season != train season")

    if np.max(
        np.abs(
            frame[TARGET_COL].to_numpy(dtype=np.float64)
            - frame["oof_target"].to_numpy(dtype=np.float64)
        )
    ) > 1e-12:
        raise ValueError("OOF target != train target")

    frame["residual"] = (
        frame[TARGET_COL].to_numpy(dtype=np.float64)
        - frame["prediction"].to_numpy(dtype=np.float64)
    )

    if CENTER_BY_SEASON_GAME_TYPE:
        group_mean = (
            frame.groupby(
                ["validation_season", "game_type"],
                dropna=False,
            )["residual"]
            .transform("mean")
            .to_numpy(dtype=np.float64)
        )
        frame["conditional_residual"] = frame["residual"].to_numpy() - group_mean
    else:
        frame["conditional_residual"] = frame["residual"].to_numpy(dtype=np.float64)

    needed = {s for pair in ALL_TRANSITIONS for s in pair}
    available = set(frame["validation_season"].unique().tolist())
    missing = sorted(needed - available)
    if missing:
        raise ValueError(f"OOF missing seasons {missing}; available={sorted(available)}")

    return frame


# =============================================================================
# Source season -> frozen correction table
# =============================================================================

def fit_table(source, key_cols, min_n, shrink_k, alpha, cap):
    stats = (
        source.groupby(key_cols, dropna=False, sort=False)["conditional_residual"]
        .agg(n="size", effect="mean")
        .reset_index()
    )

    stats["reliability"] = stats["n"] / (stats["n"] + float(shrink_k))
    stats["correction"] = stats["effect"] * stats["reliability"] * float(alpha)
    stats["correction"] = np.clip(stats["correction"], -float(cap), float(cap))

    eligible = (
        (stats["n"] >= int(min_n))
        & (stats["effect"].abs() >= float(MIN_ABS_EFFECT))
    )
    stats.loc[~eligible, "correction"] = 0.0

    return stats.loc[
        stats["correction"] != 0.0,
        key_cols + ["n", "effect", "reliability", "correction"],
    ].copy()


def lookup_correction(target, table, key_cols):
    if table.empty:
        return np.zeros(len(target), dtype=np.float64)

    return (
        target[key_cols]
        .merge(
            table[key_cols + ["correction"]],
            on=key_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )["correction"]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


# =============================================================================
# Transition 평가
# =============================================================================

def evaluate_transition(
    frame,
    schema_name,
    source_year,
    target_year,
    min_n,
    shrink_k,
    alpha,
    cap,
):
    key_cols = SCHEMAS[schema_name]

    source = frame.loc[
        frame["validation_season"] == int(source_year)
    ].copy().reset_index(drop=True)

    target = frame.loc[
        frame["validation_season"] == int(target_year)
    ].copy().reset_index(drop=True)

    table = fit_table(
        source,
        key_cols,
        min_n,
        shrink_k,
        alpha,
        cap,
    )

    correction = lookup_correction(target, table, key_cols)
    raw_pred = target["prediction"].to_numpy(dtype=np.float64)
    corrected_pred = np.clip(raw_pred + correction, 0.0, 1.0)
    y = target[TARGET_COL].to_numpy(dtype=np.float64)

    result = {
        "schema": schema_name,
        "source_year": int(source_year),
        "target_year": int(target_year),
        "min_n": int(min_n),
        "shrink_k": float(shrink_k),
        "alpha": float(alpha),
        "cap": float(cap),
        "source_groups": int(len(table)),
        "target_rows": int(len(target)),
        "corrected_rows": int(np.sum(correction != 0.0)),
        "coverage": float(np.mean(correction != 0.0)),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        **metric_gain(y, raw_pred, corrected_pred),
    }

    # R/F는 선택에는 쓰지 않고 진단용으로만 기록.
    game_type = target["game_type"].astype(str).to_numpy()
    for domain in ["R", "F"]:
        mask = game_type == domain
        if not np.any(mask):
            result[f"{domain}_gain"] = np.nan
            result[f"{domain}_coverage"] = np.nan
            continue

        dm = metric_gain(y[mask], raw_pred[mask], corrected_pred[mask])
        result[f"{domain}_gain"] = float(dm["gain"])
        result[f"{domain}_coverage"] = float(np.mean(correction[mask] != 0.0))

    return result


# =============================================================================
# Discovery grid: 2024를 보지 않고 config 선택
# =============================================================================

def run_search(frame):
    rows = []

    for schema_name in SCHEMAS:
        print("\n" + "=" * 80)
        print(f"DISCOVERY SEARCH: {schema_name}")
        print("=" * 80)

        for min_n in MIN_N_GRID:
            for shrink_k in SHRINK_K_GRID:
                for alpha in ALPHA_GRID:
                    for cap in CAP_GRID:
                        r22 = evaluate_transition(
                            frame, schema_name, 2021, 2022,
                            min_n, shrink_k, alpha, cap,
                        )
                        r23 = evaluate_transition(
                            frame, schema_name, 2022, 2023,
                            min_n, shrink_k, alpha, cap,
                        )

                        gains = [r22["gain"], r23["gain"]]
                        rows.append(
                            {
                                "schema": schema_name,
                                "min_n": int(min_n),
                                "shrink_k": float(shrink_k),
                                "alpha": float(alpha),
                                "cap": float(cap),
                                "all_discovery_positive": bool(all(g > 0 for g in gains)),
                                "worst_discovery_gain": float(min(gains)),
                                "mean_discovery_gain": float(np.mean(gains)),
                                "gain_2022": float(r22["gain"]),
                                "gain_2023": float(r23["gain"]),
                                "coverage_2022": float(r22["coverage"]),
                                "coverage_2023": float(r23["coverage"]),
                                "R_gain_2022": float(r22["R_gain"]),
                                "F_gain_2022": float(r22["F_gain"]),
                                "R_gain_2023": float(r23["R_gain"]),
                                "F_gain_2023": float(r23["F_gain"]),
                            }
                        )

    return pd.DataFrame(rows)


def select_config(search_df, schema_name):
    block = search_df.loc[search_df["schema"] == schema_name].copy()
    passed = block.loc[block["all_discovery_positive"]].copy()

    if passed.empty:
        selected = block.sort_values(
            ["worst_discovery_gain", "mean_discovery_gain"],
            ascending=[False, False],
        ).iloc[0].copy()
        selected["selection_status"] = "FAIL_NO_ALL_POSITIVE_CONFIG"
        return selected

    # 평균 하나가 아니라 discovery 중 최악 gain을 최대화.
    # 동률이면 더 보수적인 보정 선택.
    selected = passed.sort_values(
        [
            "worst_discovery_gain",
            "mean_discovery_gain",
            "alpha",
            "cap",
            "min_n",
            "shrink_k",
        ],
        ascending=[False, False, True, True, False, False],
    ).iloc[0].copy()
    selected["selection_status"] = "PASS_DISCOVERY"
    return selected


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(EXPERIMENT_NAME)
    print("=" * 80)
    print(f"Train : {TRAIN_PATH}")
    print(f"OOF   : {OOF_PATH}")

    frame = build_frame()
    print(f"OOF rows: {len(frame):,}")
    print("Seasons :", sorted(frame["validation_season"].unique().tolist()))

    # 1) Discovery-only grid search
    search_df = run_search(frame)
    search_df.to_csv(
        OUT_DIR / "parameter_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2) 2024를 보기 전에 schema별 config freeze
    configs = {}
    config_rows = []

    for schema_name in SCHEMAS:
        s = select_config(search_df, schema_name)
        cfg = {
            "schema": schema_name,
            "min_n": int(s["min_n"]),
            "shrink_k": float(s["shrink_k"]),
            "alpha": float(s["alpha"]),
            "cap": float(s["cap"]),
            "selection_status": str(s["selection_status"]),
            "worst_discovery_gain": float(s["worst_discovery_gain"]),
            "mean_discovery_gain": float(s["mean_discovery_gain"]),
        }
        configs[schema_name] = cfg
        config_rows.append(cfg)

    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(
        OUT_DIR / "selected_configs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("FROZEN CONFIGS — 2024 미사용")
    print("=" * 80)
    print(config_df.to_string(index=False))

    # 3) Frozen config로 세 transition 전부 평가
    results = []

    for schema_name, cfg in configs.items():
        for source_year, target_year in ALL_TRANSITIONS:
            r = evaluate_transition(
                frame=frame,
                schema_name=schema_name,
                source_year=source_year,
                target_year=target_year,
                min_n=cfg["min_n"],
                shrink_k=cfg["shrink_k"],
                alpha=cfg["alpha"],
                cap=cfg["cap"],
            )
            r["selection_status"] = cfg["selection_status"]
            results.append(r)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUT_DIR / "transition_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("STRICT NEXT-SEASON RESULT")
    print("=" * 80)
    print(
        results_df[
            [
                "schema",
                "source_year",
                "target_year",
                "coverage",
                "raw_brier",
                "corrected_brier",
                "gain",
                "R_gain",
                "F_gain",
            ]
        ].to_string(index=False)
    )

    # 4) 최종 판정
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment": "cb_reg_v003",
        "schemas": SCHEMAS,
        "pressure_definition": {
            "full": "3-2",
            "high": "balls==3 or strikes==2",
            "normal": "otherwise",
            "note": "public V18 log does not expose exact pressure formula",
        },
        "discovery_transitions": DISCOVERY_TRANSITIONS,
        "confirmation_transition": CONFIRM_TRANSITION,
        "selected_configs": configs,
        "decisions": {},
    }

    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    for schema_name, cfg in configs.items():
        block = results_df.loc[results_df["schema"] == schema_name]
        gain_map = {
            int(row["target_year"]): float(row["gain"])
            for _, row in block.iterrows()
        }

        discovery_pass = (
            cfg["selection_status"] == "PASS_DISCOVERY"
            and gain_map[2022] > 0
            and gain_map[2023] > 0
        )
        confirmation_pass = gain_map[2024] > 0
        strong_pass = discovery_pass and confirmation_pass

        summary["decisions"][schema_name] = {
            "gain_2022": gain_map[2022],
            "gain_2023": gain_map[2023],
            "gain_2024_confirmation": gain_map[2024],
            "discovery_pass": discovery_pass,
            "confirmation_pass": confirmation_pass,
            "strong_pass": strong_pass,
        }

        print(f"\n[{schema_name}]")
        print(f"2021 -> 2022 : {gain_map[2022]:+.8f}")
        print(f"2022 -> 2023 : {gain_map[2023]:+.8f}")
        print(f"2023 -> 2024 : {gain_map[2024]:+.8f}  <- confirmation")
        print(f"STRONG PASS  : {strong_pass}")

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n판정 기준")
    print("- 2021->22, 2022->23 둘 다 + : discovery PASS")
    print("- config freeze 후 2023->24도 + : STRONG PASS")
    print("- 2024를 본 뒤 alpha/min_n/shrink/cap을 재조정하지 않음")

    print("\nSaved:")
    print(f"- {OUT_DIR / 'parameter_search.csv'}")
    print(f"- {OUT_DIR / 'selected_configs.csv'}")
    print(f"- {OUT_DIR / 'transition_results.csv'}")
    print(f"- {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

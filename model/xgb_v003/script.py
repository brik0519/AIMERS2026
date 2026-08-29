# script.py
# xgb_v002_fe_v001_exact4_success_only 추론 및 제출
#
# EDA feature experiment/Combined_Feature_Experiment/
# xgb_v002_fe_v001_exact4_success_only_train.py에서 생성된 artifact를 로드해
# test.csv를 추론하고 output/submission.csv를 만든다. 이 모델은 gated
# residual correction을 적용하지 않는다(Stage-1 예측을 그대로 제출) — raw47
# 대비 성능 개선은 순수하게 feature engineering(Exact-ASOF4 + situational5 +
# temporal10 + BBEB2(success_rate만, 나머지 asof rate는 raw))에서만 온다.
#
# pkl 안의 pitcher_season_stats(DataFrame)는 pandas 버전 간 pickle 비호환을
# 피하기 위해 순수 dict로 저장돼 있다 — dataframe_from_portable()로 이
# 프로세스의 pandas로 다시 만들어 쓴다(cb_reg_v019_asof_residual_platoon_only
# 최초 제출 시 겪은 StringDtype 에러에 대한 대응, 애초에 이 형식으로 저장했다).
# bbeb_priors는 스칼라 값(alpha/beta 등)만 담고 있어 DataFrame이 없으므로
# 별도 변환이 필요 없다.

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPERIMENT_NAME = "xgb_v002_fe_v001_exact4_success_only"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"


# ============================================================
# 공통
# ============================================================

def dataframe_from_portable(portable):
    """train 쪽 dataframe_to_portable()로 저장된 dict를 DataFrame으로
    복원한다. 이 프로세스에 설치된 pandas로 새로 만드는 것이므로 학습
    환경의 pandas 버전과 달라도 문제가 없다."""
    return pd.DataFrame(portable["data"], columns=portable["columns"])


def prepare_raw_features(df, features, id_like_cols):
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError("입력 데이터에 학습 Feature가 없습니다: " + ", ".join(missing_features))

    X = df[features].copy()
    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


# ============================================================
# 상황(situational) / Temporal feature (train 쪽과 완전히 동일, row-wise)
# ============================================================

def add_situational_features(df):
    df = df.copy()

    balls = pd.to_numeric(df["balls_before"], errors="coerce")
    strikes = pd.to_numeric(df["strikes_before"], errors="coerce")

    df["count_balance"] = np.where(
        balls > strikes, "batter_ahead",
        np.where(balls < strikes, "pitcher_ahead", "even"),
    )

    df["full_count"] = ((balls == 3) & (strikes == 2)).astype(int)

    df["scoring_position"] = (
        (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)
    ).astype(int)

    is_pitcher_home = df["top_bottom"] == "T"
    df["pitcher_win_expectancy"] = np.where(
        is_pitcher_home, df["home_win_expectancy"], df["away_win_expectancy"]
    )

    df["platoon"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)

    return df


def add_temporal_features(df):
    df = df.copy()

    df["recent1_success_vs_career"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["recent3_success_vs_career"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["recent5_success_vs_career"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]

    df["recent1_middle_vs_career"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["recent3_middle_vs_career"] = df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["recent5_middle_vs_career"] = df["asof_pitcher_prev5_game_middle_rate"] - df["asof_pitcher_middle_rate"]

    df["success_trend_1_3"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev3_game_success_rate"]
    df["success_trend_3_5"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]

    df["middle_trend_1_3"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_prev3_game_middle_rate"]
    df["middle_trend_3_5"] = df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_prev5_game_middle_rate"]

    return df


# ============================================================
# Exact-ASOF 4개 (season_logn 제외, train 쪽과 완전히 동일 로직)
# ============================================================

def add_pitcher_exact_asof_features_test(test, pitcher_season_stats):
    required = {"season", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise KeyError(f"Exact-ASOF 필수 test 컬럼 없음: {missing}")

    out = test.copy()

    exact_cols = [
        "exact_pitcher_season_n", "exact_pitcher_season_success_rate",
        "exact_pitcher_past_success_rate", "exact_pitcher_season_minus_past",
    ]
    for col in exact_cols:
        out[col] = np.nan

    test_season = pd.to_numeric(out["season"], errors="raise").astype(int)

    stats = pitcher_season_stats.copy()
    stats["season"] = pd.to_numeric(stats["season"], errors="raise").astype(int)

    for target_year in sorted(test_season.unique()):
        row_mask = test_season == int(target_year)
        rows = out.loc[row_mask]

        hist = stats.loc[stats["season"] < int(target_year)]

        if len(hist) > 0:
            hist_agg = hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]].sum()
            hist_n = rows["pitcher_id"].map(hist_agg["n"]).fillna(0.0).to_numpy(dtype=np.float64)
            hist_success = rows["pitcher_id"].map(hist_agg["success"]).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            hist_n = np.zeros(len(rows), dtype=np.float64)
            hist_success = np.zeros(len(rows), dtype=np.float64)

        asof_n = pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
        asof_rate = pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce").to_numpy(dtype=np.float64)
        asof_success = asof_n * asof_rate

        current_n = np.maximum(asof_n - hist_n, 0.0)
        current_success = asof_success - hist_success
        current_success = np.where(np.isfinite(current_success), np.clip(current_success, 0.0, current_n), np.nan)

        current_rate = np.full(len(rows), np.nan, dtype=np.float64)
        valid_current = (current_n > 0.0) & np.isfinite(current_success)
        current_rate[valid_current] = current_success[valid_current] / current_n[valid_current]

        past_rate = np.full(len(rows), np.nan, dtype=np.float64)
        valid_past = hist_n > 0.0
        past_rate[valid_past] = hist_success[valid_past] / hist_n[valid_past]

        idx = rows.index
        out.loc[idx, "exact_pitcher_season_n"] = current_n
        out.loc[idx, "exact_pitcher_season_success_rate"] = current_rate
        out.loc[idx, "exact_pitcher_past_success_rate"] = past_rate
        out.loc[idx, "exact_pitcher_season_minus_past"] = current_rate - past_rate

    return out


# ============================================================
# BBEB (개인 단위, pitcher_success/batter_success 2개만 — 나머지 asof rate는
# raw 그대로 model input에 들어간다. train 쪽 compute_eb_rate와 동일 로직)
# ============================================================

def reconstruct_counts(df, n_col, rate_col):
    n_raw = pd.to_numeric(df[n_col], errors="coerce").to_numpy(dtype=float)
    rate = pd.to_numeric(df[rate_col], errors="coerce").to_numpy(dtype=float)

    valid = (
        np.isfinite(n_raw) & np.isfinite(rate)
        & (n_raw >= 0.0) & (rate >= 0.0) & (rate <= 1.0)
    )

    n = np.full(len(df), np.nan, dtype=float)
    success = np.full(len(df), np.nan, dtype=float)
    failure = np.full(len(df), np.nan, dtype=float)

    n_valid = np.rint(n_raw[valid])
    success_valid = np.clip(np.rint(n_valid * rate[valid]), 0.0, n_valid)
    failure_valid = n_valid - success_valid

    n[valid] = n_valid
    success[valid] = success_valid
    failure[valid] = failure_valid

    return {"valid": valid, "n": n, "success": success, "failure": failure}


def compute_eb_rate(df, n_col, rate_col, prior_state):
    alpha = float(prior_state["alpha"])
    beta = float(prior_state["beta"])

    counts = reconstruct_counts(df, n_col, rate_col)
    valid = counts["valid"]
    success = counts["success"]
    failure = counts["failure"]

    prior_mean = alpha / (alpha + beta)
    eb_rate = np.full(len(df), prior_mean, dtype=float)

    if np.any(valid):
        posterior_alpha = alpha + success[valid]
        posterior_beta = beta + failure[valid]
        eb_rate[valid] = posterior_alpha / (posterior_alpha + posterior_beta)

    return eb_rate


def apply_bbeb_features(df, bbeb_feature_specs, bbeb_priors):
    df = df.copy()

    for spec in bbeb_feature_specs:
        prior_state = bbeb_priors[spec["name"]]
        df[f"bbeb_{spec['name']}_rate"] = compute_eb_rate(df, spec["n_col"], spec["rate_col"], prior_state)

    df["eb_pitcher_batter_gap"] = df["bbeb_pitcher_success_rate"] - df["bbeb_batter_success_rate"]

    return df


# ============================================================
# Submission
# ============================================================

def build_submission(sample_submission, test_ids, predictions):
    pred_df = pd.DataFrame({ID: test_ids, TARGET: predictions})

    if pred_df[ID].duplicated().any():
        raise ValueError("prediction row_id 중복")

    submission = sample_submission[[ID]].merge(pred_df, on=ID, how="left", validate="one_to_one")

    if submission[TARGET].isna().any():
        raise ValueError("submission prediction NaN")

    pred = submission[TARGET].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError("submission prediction NaN/Inf")

    if not ((pred >= 0.0) & (pred <= 1.0)).all():
        raise ValueError("prediction outside [0,1]")

    return submission


# ============================================================
# Main
# ============================================================

def main():
    start = time.time()

    print("=" * 90)
    print(f"Inference: {EXPERIMENT_NAME}")
    print("=" * 90)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Artifact 없음: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if artifact.get("model_name") != EXPERIMENT_NAME:
        raise ValueError("Artifact model_name 불일치")

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    final_features = artifact["features"]
    id_like_cols = artifact["id_like_cols"]

    bbeb_feature_specs = artifact["bbeb_feature_specs"]
    bbeb_priors = artifact["bbeb_priors"]
    pitcher_season_stats = dataframe_from_portable(artifact["pitcher_season_stats"])

    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    test_ids = test[ID].to_numpy()

    print(f"test rows: {len(test):,}")

    # --------------------------------------------------------
    # 1) 상황(situational) + Temporal (row-wise, 다른 행 참조 없음)
    # --------------------------------------------------------

    test = add_situational_features(test)
    test = add_temporal_features(test)

    # --------------------------------------------------------
    # 2) Exact-ASOF 4개
    # --------------------------------------------------------

    test = add_pitcher_exact_asof_features_test(test=test, pitcher_season_stats=pitcher_season_stats)

    # --------------------------------------------------------
    # 3) BBEB (success_rate 2개만, 나머지 asof rate는 raw 그대로 유지)
    # --------------------------------------------------------

    test = apply_bbeb_features(test, bbeb_feature_specs, bbeb_priors)

    # --------------------------------------------------------
    # 4) Feature -> preprocessor -> prediction (correction 없음, Stage-1 그대로 제출)
    # --------------------------------------------------------

    missing_features = [c for c in final_features if c not in test.columns]
    if missing_features:
        raise KeyError(f"test 최종 feature 누락: {missing_features}")

    X_test_raw = prepare_raw_features(test, final_features, id_like_cols)
    X_test = preprocessor.transform(X_test_raw)

    pred_raw = np.asarray(model.predict_proba(X_test)[:, 1], dtype=np.float64)
    if not np.isfinite(pred_raw).all():
        raise ValueError("raw prediction NaN/Inf")

    final_pred = np.clip(pred_raw, 0.0, 1.0)

    # --------------------------------------------------------
    # 5) Submission
    # --------------------------------------------------------

    submission = build_submission(sample_submission=sample_sub, test_ids=test_ids, predictions=final_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Final pred mean: {final_pred.mean():.6f}")
    print(f"Final pred min : {final_pred.min():.6f}")
    print(f"Final pred max : {final_pred.max():.6f}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()

# ============================================================
# CatBoost Raw Feature Baseline
# ============================================================

import os
import time

import joblib
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# 설정
# ============================================================

DATA_DIR = "./data"

ID = "row_id"
TARGET = "control_success"

MODEL_PATH = "./model/rf.pkl"


# ============================================================
# 데이터 로드
# ============================================================

# test.csv의 헤더만 읽어 입력 Feature를 결정한다.
test_cols = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="utf-8-sig",
    nrows=0,
).columns

FEATURES = [c for c in test_cols if c != ID]

# CatBoost에서 범주형으로 처리할 컬럼 후보
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
]

# 실제 test.csv에 존재하는 컬럼만 사용
CAT_COLS = [c for c in CAT_COLS if c in FEATURES]

train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="utf-8-sig",
    usecols=FEATURES + [TARGET],
)

print("=" * 60)
print("데이터 로드 완료")
print("train shape:", train.shape)
print("Feature 수:", len(FEATURES))
print("범주형 Feature 수:", len(CAT_COLS))
print("범주형 Feature:", CAT_COLS)
print("시즌:", train["season"].min(), "~", train["season"].max())
print(f"제구 성공률: {train[TARGET].mean():.6f}")
print("=" * 60)


# ============================================================
# 범주형 전처리
# ============================================================

def prepare_features(df):
    """
    CatBoost 입력용 DataFrame을 생성한다.

    수치형 결측값은 CatBoost가 직접 처리한다.
    범주형 결측값은 고정 문자열로 바꾸고 문자열 dtype으로 통일한다.
    """
    df = df.copy()

    for col in CAT_COLS:
        df[col] = df[col].fillna("__MISSING__").astype(str)

    return df


# ============================================================
# 2024 시즌 Validation 분할
# ============================================================

is_val = train["season"] == 2024

X_train = prepare_features(
    train.loc[~is_val, FEATURES]
)
y_train = train.loc[~is_val, TARGET]

X_val = prepare_features(
    train.loc[is_val, FEATURES]
)
y_val = train.loc[is_val, TARGET]

print("학습 데이터:", X_train.shape)
print("검증 데이터:", X_val.shape)


# ============================================================
# Validation 모델
# ============================================================

model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.07,
    depth=8,
    loss_function="Logloss",
    l2_leaf_reg=10.0,
    random_strength=1.0,
    random_seed=42,
    thread_count=-1,
    verbose=100,
    allow_writing_files=False,
)


# ============================================================
# Validation 학습
# ============================================================

print("=" * 60)
print("Validation 모델 학습 시작")
print("=" * 60)

start_time = time.time()

model.fit(
    X_train,
    y_train,
    cat_features=CAT_COLS,
    eval_set=(X_val, y_val),
    early_stopping_rounds=80,
    use_best_model=True,
)

elapsed = time.time() - start_time

print(f"Validation 모델 학습 완료: {elapsed:.1f}초")


# ============================================================
# Brier Skill Score 계산
# ============================================================

val_pred = model.predict_proba(X_val)[:, 1]

r = y_val.mean()

brier = ((val_pred - y_val) ** 2).mean()
baseline_brier = r * (1 - r)

score = max(
    0,
    100000 * (1 - brier / baseline_brier),
)

print("=" * 60)
print("Validation 결과")
print("=" * 60)
print(f"Brier Score          : {brier:.8f}")
print(f"Baseline Brier       : {baseline_brier:.8f}")
print(f"Validation Score     : {score:.2f}")
print(f"Target mean          : {y_val.mean():.6f}")
print(f"Prediction mean      : {val_pred.mean():.6f}")
print(f"Prediction std       : {val_pred.std():.6f}")
print(f"Prediction min       : {val_pred.min():.6f}")
print(f"Prediction max       : {val_pred.max():.6f}")


# ============================================================
# 최적 iteration 결정
# ============================================================

best_iteration = model.get_best_iteration()

if best_iteration is None or best_iteration < 0:
    final_iterations = 800
else:
    # CatBoost iteration index는 0부터 시작하므로 1을 더한다.
    final_iterations = best_iteration + 1

print(f"Best iteration       : {best_iteration}")
print(f"Final iterations     : {final_iterations}")


# ============================================================
# 전체 데이터 재학습
# ============================================================

X_full = prepare_features(train[FEATURES])
y_full = train[TARGET]

final_model = CatBoostClassifier(
    iterations=final_iterations,
    learning_rate=0.07,
    depth=8,
    loss_function="Logloss",
    l2_leaf_reg=10.0,
    random_strength=1.0,
    random_seed=42,
    thread_count=-1,
    verbose=100,
    allow_writing_files=False,
)

print("=" * 60)
print("전체 데이터 재학습 시작")
print("=" * 60)

start_time = time.time()

final_model.fit(
    X_full,
    y_full,
    cat_features=CAT_COLS,
)

elapsed = time.time() - start_time

print(f"전체 데이터 재학습 완료: {elapsed:.1f}초")


# ============================================================
# 모델 저장
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True,
)

joblib.dump(
    final_model,
    MODEL_PATH,
    compress=3,
)

print("=" * 60)
print(f"모델 저장 완료: {MODEL_PATH}")
print("=" * 60)
# %% [markdown]
# # [Baseline] RandomForest — 학습
# 
# KBO 투구 하나가 **제구 성공** 투구일 확률을 예측하는 베이스라인입니다.
# 
# - **입력**: `test.csv` 의 47개 컬럼 (경기 상황, 투수·타자의 직전까지 누적 기록 등)
# - **출력**: 제구 성공 확률 (0 이상 1 이하의 실수)
# - **평가지표**: Brier Skill Score
# 
# `trackman_history.csv` 는 이 베이스라인에서 사용하지 않습니다. 2019~2024 과거 로그
# 179만 행이 그대로 남아 있으니 직접 활용해 보세요.
# 
# 이 노트북은 모델을 **학습**하여 `./model/rf.pkl` 로 저장합니다. 저장한 모델은
# 추론용 `script.py` 와 함께 `baseline_submit.zip` 으로 묶어 제출합니다.

# %% [markdown]
# ## 1. 라이브러리 불러오기
# 
# 데이터 처리(pandas)와 모델 학습(scikit-learn)에 필요한 라이브러리를 불러옵니다.
# `joblib` 은 학습한 모델을 파일로 저장할 때 사용합니다.

# %%
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

# %% [markdown]
# ## 2. 데이터 불러오기
# 
# `train.csv` 는 2019~2024 시즌이고 평가 데이터는 2025 시즌입니다.
# 
# 사용할 피처 목록은 `test.csv` 가 정합니다. `train.csv` 에만 있는 컬럼을 학습에 넣으면
# 평가 시점에 그 컬럼이 없어 추론이 실패하기 때문입니다.

# %%
test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns
FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                    encoding="utf-8-sig", usecols=FEATURES + [TARGET])

print("train:", train.shape, "| 피처:", len(FEATURES),
      f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)})")
print("시즌:", train["season"].min(), "~", train["season"].max())
print(f"제구 성공률: {train[TARGET].mean():.4f}")

# %% [markdown]
# ## 3. 전처리 정의
# 
# 범주형 3개(`top_bottom`, `game_type`, `base_state`)는 정수로 바꾸고, 수치형 44개의
# 결측값은 중앙값으로 채웁니다.
# 
# 두 변환을 `ColumnTransformer` 로 묶어 모델 파이프라인 안에 넣습니다. 이렇게 하면
# 추론할 때도 같은 변환이 자동으로 따라가므로, 전처리를 빠뜨려 생기는 실수를 막을 수
# 있습니다. `handle_unknown="use_encoded_value"` 는 학습 때 보지 못한 범주가 평가
# 데이터에 나타나면 -1 로 처리하라는 뜻입니다.

# %%
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
])

# %% [markdown]
# ## 4. 모델 정의와 학습
# 
# 트리 깊이를 10, 잎 노드의 최소 샘플 수를 200으로 제한해 얕게 두었습니다. 학습이
# 1분 안에 끝나고 모델 파일도 4MB 정도로 가볍습니다. `random_state` 를 고정했고 GPU 를
# 사용하지 않으므로, 같은 데이터와 같은 패키지 버전이면 어느 환경에서 실행해도 결과가
# 같습니다.

# %%
model = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=-1,
        random_state=42,
    )),
])

# 2024 시즌을 검증용으로 떼어 두고 2019~2023 으로 학습합니다.
is_val = train["season"] == 2024
X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
print("train:", len(X_train), "| val:", len(X_val))

t = time.time()
model.fit(X_train, y_train)
print(f"학습 완료 :: {time.time() - t:.1f}s")

# %% [markdown]
# ## 5. 검증 — Brier Skill Score
# 
# 학습 데이터에서 떼어 둔 2024 시즌으로 검증 점수를 계산합니다.
# 
# Brier 는 예측 확률과 실제값(0/1) 차이의 제곱 평균이고, 이를 상수 예측의 Brier 인
# `r(1-r)` 로 나누어 Brier Skill Score 를 구합니다. 검증 분할 방식은 참가자가 자유롭게
# 바꿀 수 있습니다.

# %%
val_pred = model.predict_proba(X_val)[:, 1]

r = y_val.mean()
brier = ((val_pred - y_val) ** 2).mean()
baseline_brier = r * (1 - r)
score = max(0, 100000 * (1 - brier / baseline_brier))

print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
print(f"Validation Score: {score:.2f}")

# %% [markdown]
# ## 6. 전체 데이터로 재학습 & 모델 저장
# 
# 검증으로 성능을 확인했으니 이제 전체 학습 데이터로 다시 학습합니다.
# 
# 학습한 파이프라인을 `./model/rf.pkl` 로 저장합니다. 이 파일을 추론용 `script.py`,
# `requirements.txt` 와 함께 `baseline_submit.zip` 으로 묶으면 제출 준비가 끝납니다.

# %%
t = time.time()
model.fit(train[FEATURES], train[TARGET])
print(f"재학습 완료 :: {time.time() - t:.1f}s")

os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/rf.pkl", compress=3)
print("저장 완료: ./model/rf.pkl")



# %% [markdown]
# # [Baseline] RandomForest — 추론
# 
# 학습 노트북이 저장한 모델(`./model/rf.pkl`)을 불러와 평가 데이터
# (`./data/test.csv`)를 예측하고, 제출 파일(`./output/submission.csv`)을 만듭니다.
# 
# **코드 제출 방식** — 이 대회는 결과 CSV 가 아니라 코드를 제출합니다. 아래 구조의 zip 을
# 제출하면 평가 서버가 zip 을 풀고 `script.py` 를 그대로 실행하여 채점합니다.
# 
# ```
# baseline_submit.zip
# ├── model/
# │   └── rf.pkl                # 학습 노트북이 저장한 모델
# ├── script.py               # 추론 코드 (서버가 실행)
# └── requirements.txt        # 필요한 라이브러리
# ```
# 
# 이 노트북의 코드 셀을 순서대로 이어 붙인 것이 `script.py` 입니다. 함께 배포된
# `[Baseline_Inference]_RandomForest (추론).py` 를 `script.py` 로 이름만 바꿔 넣어도 됩니다.
# 
# 배포된 `test.csv` 는 제출 형식 확인용 5행 샘플입니다. 실제 평가 데이터 245,789행은
# 평가 서버에만 있으며, 서버가 같은 경로(`./data/test.csv`)로 넣어 줍니다.

# %% [markdown]
# ## 1. 라이브러리 불러오기
# 
# `pandas` 로 데이터를 읽고 결과를 씁니다. `joblib` 으로 학습 때 저장한 모델(`.pkl`)을
# 불러옵니다.

# %%
import os

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# %% [markdown]
# ## 2. 데이터 로드와 전처리
# 
# 추론 입력은 반드시 **학습 때와 똑같은 방식**으로 만들어야 합니다. 학습에서 `row_id` 만
# 빼고 나머지 컬럼을 그대로 사용했으므로 여기서도 동일하게 골라냅니다.
# 
# 범주형 인코딩과 결측 대치는 모델 파일 안의 파이프라인이 함께 수행하므로 이 코드에서는
# 컬럼만 선택합니다. 파이프라인 밖에서 피처를 만들었다면 그 코드도 여기에 있어야 평가
# 서버에서 같은 입력이 됩니다.

# %%
# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def build_features(df):
    """모델 입력 추출 — 학습 때와 동일하게 row_id만 빼고 전부 사용.

    범주형 인코딩(top_bottom, game_type, base_state)과 결측 대치는
    모델 파일 안의 파이프라인이 함께 수행하므로 여기서는 컬럼만 선택.
    """
    return df.drop(columns=[ID_COL])

# %% [markdown]
# ## 3. 제출 파일 생성 유틸
# 
# 제출 파일은 `sample_submission.csv` 와 **같은 row_id 순서, 같은 컬럼**이어야 합니다.
# 모델 예측을 `row_id` 기준으로 `sample_submission` 에 채워 넣습니다. 예측에 없는
# `row_id` 는 기존 placeholder 값을 유지합니다.

# %%
# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")

# %% [markdown]
# ## 4. 추론 실행
# 
# 학습 노트북이 저장한 파이프라인을 불러와 평가 데이터를 예측하고
# `./output/submission.csv` 를 생성합니다. 파이프라인이 전처리와 분류를 함께 수행하므로
# 데이터를 그대로 넣으면 됩니다.
# 
# `predict_proba(X)[:, 1]` 은 제구 성공(1) 확률입니다. 제출값은 0 이상 1 이하의 실수여야
# 하므로 0/1 로 반올림하지 않습니다.

# %%
# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # rf.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "rf.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()



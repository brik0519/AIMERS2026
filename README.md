# AIMERS2026

본 프로젝트는 각 머신러닝 실험을 다음 세 단계로 관리합니다.

```.bash
train.py
    ↓
모델 학습 및 저장
    ↓
script.py
    ↓
저장된 모델을 이용한 inference
    ↓
package.py
    ↓
제출에 필요한 파일 수집 및 ZIP 생성
```
각 파일의 역할은 명확하게 분리합니다.

|파일|역할|
|---|--|
|train.py	|모델 학습, Validation, 최종 모델 저장|
|script.py	|저장된 모델을 로드하여 Test 데이터 추론|
|package.py	|모델, script.py, requirements.txt를 모아 제출 ZIP 생성|

train.py가 하나의 실험을 정의하고 모델을 생성하면, script.py는 해당 모델의 학습 규약을 따라 inference를 수행합니다.

마지막으로 package.py는 제출에 필요한 파일들을 정해진 구조로 묶습니다.

만약 새로운 모델을 테스트해보고 싶다면,
model/base_v001 경로의 train.py와
validation/base_v001 경로의 validation.py를
생성형 AI에게 주고 새로운 모델을 적용해달라 요청하면 됩니다.

이후 
```
python train.py
↓
python script.py
↓
python package.py
```
순서대로 실행하면 자동으로 압축된 제출물 파일이
submits 하위 경로에 들어 있습니다.
그대로 업로드하시면 됩니다.

> [!CAUTION]
> 아직 앙상블과 관련된 테스트 구조 및 코드 구조는 완성되지 않았습니다.
>
> 단일 모델로 테스트 하는 경우에만 유효합니다!

## 1. Project Directory Structure

현재 프로젝트는 다음과 같은 구조를 사용합니다.
```.bash
.
├── baseline/
│   ├── Baseline_Inference.ipynb
│   ├── Baseline_Train.ipynb
│   └── requirements.txt
│
├── data/
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── trackman_history.csv
│   └── train.csv
│
├── model/
│   ├── cb_reg_v001/
│   ├── cb_v001/
│   ├── ensemble_v001/
│   ├── et_v001/
│   ├── lgb_v001/
│   ├── rf_v001/
│   └── xgb_v001/
│
├── output/
│   └── submission.csv
│
├── submits/
│   ├── baseline(549.64)/
│   ├── cb_v001(783.26)/
│   ├── ensemble_v001(807.34)/
│   ├── et_v001(510.49)/
│   ├── lgb_v002(748.22)/
│   └── xgb_v001(759.36)/
│
├── validation/
│   ├── cb_reg_v001/
│   ├── cb_v001/
│   ├── ensemble_v001/
│   ├── et_v001/
│   ├── lgb_v001/
│   ├── rf_v001/
│   └── xgb_v001/
│
├── package.py
├── README.md
├── requirements.txt
├── script.py
└── train.py
```

각 디렉터리는 다음 역할을 가집니다.


### data/

학습 및 추론에 사용하는 원본 데이터입니다.

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── trackman_history.csv
```

train.py는 주로 train.csv와 test.csv를 사용합니다.

script.py는 실제 제출 inference 과정에서 다음 파일을 사용합니다.

test.csv
sample_submission.csv


### model/

각 실험의 최종 학습 모델을 저장합니다.
```
model/
├── cb_v001/
├── xgb_v001/
├── lgb_v001/
└── ...
```

예를 들어

EXPERIMENT_NAME = "cb_v001"

이면 train.py는 다음 경로를 생성합니다.

```
model/
└── cb_v001/
    ├── cb_v001.pkl
    └── train.py
```

cb_v001.pkl은 전체 학습 데이터를 이용하여 최종적으로 다시 학습한 모델입니다.

같이 저장되는 train.py는 해당 모델을 생성한 학습 코드를 보존하기 위한 용도입니다.


### validation/

각 실험의 Validation 결과를 저장합니다.

예를 들어

```
validation/
└── cb_v001/
    ├── pred.csv
    ├── validation.npz
    └── metadata.json
```

이 생성됩니다.

각 파일의 역할은 다음과 같습니다.

#### pred.csv

Validation row별 실제 target과 prediction을 저장합니다.

row_id,target,prediction
...

앙상블 분석에서도 이 파일을 사용합니다.

#### validation.npz

Validation 결과를 NumPy 형식으로 압축 저장합니다.

row_id
target
prediction

#### metadata.json

실험에 대한 수치 정보를 저장합니다.

예를 들어 다음과 같은 정보가 포함됩니다.

```
experiment_name
model_type
model_params
validation_season
n_validation
positive_rate
brier
baseline_brier
bss
prediction_mean
prediction_std
prediction_min
prediction_max
best_iteration
final_iterations
```

따라서 validation/은 단순히 score를 기록하는 디렉터리가 아니라, 이후 모델 비교·앙상블·Error Analysis를 위한 실험 결과 저장소입니다.



### output/

script.py가 inference 결과를 저장하는 위치입니다.

```
output/
└── submission.csv
```

최종적으로 대회 평가 시스템에서 필요한 예측 결과가 이 위치에 생성됩니다.


### submits/

package.py가 생성한 제출 패키지를 관리합니다.

예를 들어

```
submits/
└── cb_v001/
    ├── model/
    │   └── cb_v001.pkl
    ├── script.py
    └── requirements.txt
```

와 같은 구조가 만들어집니다.

Leaderboard 제출 이후에는 실험 결과를 쉽게 확인하기 위해 다음과 같이 점수를 디렉터리 이름에 기록할 수 있습니다.

```
cb_v001(783.26)
xgb_v001(759.36)
ensemble_v001(807.34)
```

이 디렉터리는 제출 당시 사용한 모델과 inference 코드를 함께 보존하는 역할도 합니다.


## 2. train.py

train.py는 하나의 머신러닝 실험 전체를 담당합니다.

CatBoost를 예로 들면 다음과 같습니다.
```
DATA_DIR = Path("./data")
ID = "row_id"
TARGET = "control_success"
EXPERIMENT_NAME = "cb_v001"
MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
```

이 값들이 현재 실험의 데이터셋 위치와 저장 위치를 결정합니다.



## 3. EXPERIMENT_NAME: 수행한 모델 실험의 약칭(구분명)

현재 단일 모델 실험에서 가장 중요한 변수는

EXPERIMENT_NAME = "cb_v001"

입니다.

이 하나의 값으로 모델과 Validation 결과가 연결됩니다.

```
EXPERIMENT_NAME
      │
      ├── model/cb_v001/
      │
      └── validation/cb_v001/
```

모델 파일 이름 역시 동일한 실험명을 사용합니다.

```
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = (
    MODEL_DIR
    / f"{EXPERIMENT_NAME}.pkl"
)
```

따라서 최종 결과는

```
cb_v001
│
├── model/
│   └── cb_v001/
│       └── cb_v001.pkl
│
└── validation/
    └── cb_v001/
        ├── pred.csv
        ├── validation.npz
        └── metadata.json
```

이 됩니다.

이 구조를 통해 특정 Validation 결과가 어떤 최종 모델과 연결되는지를 실험 이름만으로 추적할 수 있습니다.


## 4. Validation
Validation 이후
전체 데이터로 최종 모델을 다시 학습합니다.

2019 ~ 2023 Train
+
2024 Validation
        ↓
전체 Train 데이터
        ↓
Final Model

## 5. 최종 모델 저장

전체 데이터 재학습이 완료되면

```
joblib.dump(
    final_model,
    MODEL_PATH,
    compress=3,
)
```
을 통해 모델을 저장합니다.

예를 들어

EXPERIMENT_NAME = "cb_v001"

이면
```
model/
└── cb_v001/
    ├── cb_v001.pkl
    └── train.py
```
이 생성됩니다.

여기서

cb_v001.pkl

이 실제 script.py에서 사용해야 하는 inference model입니다.

## 5. train.py에서 script.py로 전달되는 정보

중요한 점은 script.py가 train.py를 직접 import하지 않는다는 것입니다.

즉 다음과 같은 자동 연결은 존재하지 않습니다.

train.py 변수
     ↓
Python import
     ↓
script.py

대신 script.py는 train.py에서 정의된 학습 규약을 동일하게 구현해야 합니다.

논리적으로 전달되는 주요 정보는 다음과 같습니다.

|train.py|script.py에서 대응되어야 하는 내용|
|--|--|
|ID = "row_id"|	ID_COL = "row_id"|
|TARGET = "control_success"	|TARGET_COL = "control_success"|
|features	|Test에서 row_id를 제외한 입력 Feature|
|CAT_COLS	|동일한 CatBoost 범주형 Feature|
|prepare_features()	|동일한 inference 전처리|
|MODEL_PATH	|실제 제출 패키지 내부 모델 경로|
|predict_proba()[:, 1]	|동일한 확률 예측 방식|

즉 데이터 흐름은 다음과 같습니다.
```
train.py
│
├── ID = row_id
├── TARGET = control_success
├── Feature 정의
├── Categorical Feature 정의
├── 전처리 규칙
└── 학습된 *.pkl
        │
        ▼
script.py
│
├── 동일한 ID/TARGET 규약
├── 동일한 Feature 구성
├── 동일한 전처리
├── *.pkl 로드
└── predict_proba()[:, 1]
        │
        ▼
output/submission.csv
```


## 6. script.py에서 package.py로의 연결

package.py에서는

SCRIPT_PATH = Path("./script.py")

로 현재 프로젝트 루트의 script.py를 선택하고

shutil.copy2(
    SCRIPT_PATH,
    SUBMIT_DIR / "script.py",
)

로 제출 디렉터리에 복사합니다.

## 7. train.py에서 package.py로 직접 전달되는 변수

package.py는 train.py를 직접 import합니다.

import train

그리고 다음 두 값을 직접 사용합니다.

EXPERIMENT_NAME = train.EXPERIMENT_NAME
MODEL_PATH = Path(
    train.MODEL_PATH
)

따라서 현재 단일 모델 구조에서 train.py → package.py로 직접 연결되는 변수는 정확히 두 개입니다.

train.EXPERIMENT_NAME
train.MODEL_PATH

⸻

EXPERIMENT_NAME

train.py:

EXPERIMENT_NAME = "cb_v001"

package.py:

EXPERIMENT_NAME = train.EXPERIMENT_NAME

따라서 제출 디렉터리는 자동으로

submits/cb_v001/

가 됩니다.

⸻

MODEL_PATH

train.py:

MODEL_PATH = (
    MODEL_DIR
    / f"{EXPERIMENT_NAME}.pkl"
)

결과:

model/cb_v001/cb_v001.pkl

package.py:

MODEL_PATH = Path(
    train.MODEL_PATH
)

따라서 package.py가 별도로 모델 경로를 다시 작성할 필요가 없습니다.

⸻

17. 전체 변수 연결 관계

단일 CatBoost 실험을 기준으로 전체 연결 관계를 정리하면 다음과 같습니다.

```
train.py
│
├── EXPERIMENT_NAME = "cb_v001"
│       │
│       ├───────────────┐
│       ▼               ▼
│ model/cb_v001/   validation/cb_v001/
│       │
│       ▼
│ cb_v001.pkl
│       │
│       │
│       └──────────────────────────┐
│                                  │
├── ID = "row_id"                  │
│       │                          │
│       └──── logical sync ──► script.py
│                                  │
├── TARGET = "control_success"     │
│       │                          │
│       └──── logical sync ──► script.py
│                                  │
├── Feature 규칙                   │
│       └──── logical sync ──► script.py
│                                  │
├── CAT_COLS                       │
│       └──── logical sync ──► script.py
│                                  │
├── prepare_features()             │
│       └──── logical sync ──► script.py
│                                  │
│                                  │
└── MODEL_PATH ────────────────────┼────► package.py
                                   │
script.py                          │
│                                  │
├── 모델 로드 ◄────────────────────┘
├── Test inference
└── output/submission.csv
package.py
│
├── train.EXPERIMENT_NAME
├── train.MODEL_PATH
├── ./script.py
└── ./requirements.txt
        │
        ▼
submits/cb_v001/
├── model/
│   └── cb_v001.pkl
├── script.py
└── requirements.txt
        │
        ▼
submits/cb_v001.zip
```


## 8. package.py

package.py의 책임은 단순합니다.

"현재 실험에 필요한 제출 파일을 모아 ZIP 파일을 생성한다."

다음과 같은 모델 평가 기능은 수행하지 않습니다.

모델 성능 검증
Inference 결과 검증
환경 재현성 검증
ZIP 용량 검증

패키징에 필요한 파일이 존재하는지만 확인합니다.

## 9. Packaging Input

현재 package.py가 필요로 하는 입력은 세 가지입니다.

1. train.MODEL_PATH
2. ./script.py
3. ./requirements.txt

코드에서는
```
MODEL_PATH = Path(
    train.MODEL_PATH
)
SCRIPT_PATH = Path(
    "./script.py"
)
REQUIREMENTS_PATH = Path(
    "./requirements.txt"
)
```

로 정의합니다.

## 10. Packaging 과정

package.py를 실행하면

```
python package.py
```

다음 순서로 동작합니다.

```
train.py import
        ↓
EXPERIMENT_NAME 확인
MODEL_PATH 확인
        ↓
requirements.txt 내용 출력
        ↓
동일한 Submit 디렉터리 존재 여부 확인
        ↓
필수 파일 존재 여부 확인
        ↓
submits/{EXPERIMENT_NAME}/ 생성
        ↓
model/ 생성
        ↓
모델 복사
script.py 복사
requirements.txt 복사
        ↓
ZIP 생성
```

## 11. Packaging 결과

예를 들어

EXPERIMENT_NAME = "cb_v001"

이면 다음 디렉터리가 생성됩니다.

```
submits/
└── cb_v001/
    ├── model/
    │   └── cb_v001.pkl
    ├── script.py
    └── requirements.txt
```

그리고 동일한 내용을 가진 ZIP 파일이 생성됩니다.

```
submits/
└── cb_v001.zip
```

ZIP 내부 구조는 다음과 같습니다.

```
model/
└── cb_v001.pkl
script.py
requirements.txt
```

실험 이름을 감싸는 추가 상위 디렉터리는 ZIP 내부에 포함되지 않습니다.

또한 requirements.txt 내부에 어떤 패키지가
들어있는지를 package.py 실행 시 표시해주는 기능이
들어있습니다.

만약 본인이 사용한 패키지가 뜨지 않는 경우,
requirements.txt를 수정 후 다시 실행해주세요.

## 12. Experiment Lifecycle

하나의 단일 모델 실험은 다음 lifecycle을 가집니다.

예를 들어

cb_v001

실험을 수행한다고 가정합니다.

### 1. 실험 정의

EXPERIMENT_NAME = "cb_v001"

### 2. Validation

2019~2023 → Train

2024 → Validation

#### 결과:

```
validation/cb_v001/
├── pred.csv
├── validation.npz
└── metadata.json
```

### 3. 전체 데이터 재학습

Validation에서 얻은 최적 iteration을 이용하여 전체 데이터를 다시 학습합니다.

결과:
```
model/cb_v001/
├── cb_v001.pkl
└── train.py
```

4. Inference 규약 반영

script.py에서
```
ID
TARGET
Feature
Categorical Feature
전처리
모델 경로
Prediction 방식
```

을 해당 모델에 맞게 사용합니다.

5. Packaging
```
python package.py
```

결과:

```
submits/cb_v001/
submits/cb_v001.zip
```

6. Leaderboard 기록

제출 후 필요하면 결과를 디렉터리 이름에 기록합니다.

예:

cb_v001 → cb_v001(783.26)

따라서

실험명
→ Validation
→ Model
→ Submission
→ Leaderboard

를 하나의 흐름으로 추적할 수 있습니다.

## 13. 전체 프로젝트 데이터 흐름

최종적으로 프로젝트 전체 흐름은 다음과 같습니다.

                     data/
                       │
              ┌────────┴────────┐
              │                 │
          train.csv          test.csv
              │                 │
              ▼                 │
           train.py             │
              │                 │
      ┌───────┴────────┐        │
      │                │        │
      ▼                ▼        │
 validation/         model/     │
      │                │        │
      │                └────┐   │
      │                     ▼   ▼
      │                   script.py
      │                     │
      │                     ▼
      │              output/submission.csv
      │
      │
      └── 모델 비교 / 앙상블 분석 등에 활용
 train.py ── EXPERIMENT_NAME ──┐
                               │
 train.py ── MODEL_PATH ───────┤
                               ▼
                           package.py
                               ▲
                               │
                ┌──────────────┴────────────┐
                │                           │
            script.py               requirements.txt
                │                           │
                └──────────────┬────────────┘
                               ▼
                     submits/{experiment}/
                               │
                               ▼
                     submits/{experiment}.zip


## 14. 핵심 설계 원칙

현재 단일 모델 Pipeline에서는 EXPERIMENT_NAME을 중심으로 실험 결과를 관리합니다.

EXPERIMENT_NAME
       │
       ├── validation/
       ├── model/
       └── submits/

이를 통해

어떤 Validation 결과가 어떤 모델을 만들었고, 어떤 모델이 어떤 제출 파일에 사용되었는가

를 추적할 수 있습니다.

각 프로그램의 책임도 분리합니다.

train.py
→ 학습과 Validation
script.py
→ Inference
package.py
→ Packaging

특히 package.py는 단순히

필요한 파일 확인
        ↓
파일 복사
        ↓
제출 디렉터리 생성
        ↓
ZIP 생성

만을 담당하도록 유지합니다.

## 15. 현재 단일 모델 기준 정리

현재 CatBoost 실험을 예로 들면 전체 연결은 다음 한 줄로 정리할 수 있습니다.

cb_v001
→ validation/cb_v001/
→ model/cb_v001/cb_v001.pkl
→ script.py에서 해당 모델 추론
→ package.py
→ submits/cb_v001.zip

이 구조를 기반으로 CatBoost, XGBoost, LightGBM 등 각 단일 모델 실험을 독립적으로 관리합니다.

앙상블은 하나의 제출이 여러 개의 모델 artifact를 사용한다는 점에서 단일 모델과 구조가 다르므로, 별도의 제출 구조로 확장하여 관리할 예정에 있습니다.
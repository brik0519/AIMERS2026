# Temporal Feature Engineering Experiment

LG Aimers KBO 제구 성공 확률 예측에서 **Temporal Feature**와 **Count State**를 추가했을 때 성능이 어떻게 변하는지 확인한 실험입니다.

초기 실험은 기존 CatBoost + XGBoost Ensemble에 Feature를 추가하는 방식으로 진행했습니다. 이후 Feature Engineering 효과를 더 직접적으로 확인하기 위해 **단일 모델을 고정한 상태에서 Basic과 EXP3을 비교**하는 추가 실험을 진행했습니다.

---

## 1. Experiment Goal

초기 Feature Engineering 실험에서는 기존 **CatBoost + XGBoost Ensemble**을 유지한 상태에서 Feature 구성만 변경했습니다.

실험 구성:

- Basic
- EXP1 = Temporal + Count
- EXP3 = Temporal only
- EXP4 = Count only

이 과정에서 EXP3이 가장 높은 Public LB를 기록했습니다.

다만 Ensemble에서는 CatBoost와 XGBoost의 Feature 활용 방식과 두 모델을 결합하는 효과가 함께 반영되므로, 성능 변화가 Feature 자체에서 발생한 것인지 구분하기 어렵습니다.

따라서 추가 실험에서는 **모델을 하나로 고정하고 Feature만 변경**했습니다.

- CatBoost Basic vs CatBoost + EXP3
- XGBoost Basic vs XGBoost + EXP3

---

## 2. Feature Configuration

| Experiment | Feature 구성 | Feature 수 |
|---|---|---:|
| Basic | Raw | 47 |
| EXP1 | Raw + Temporal 10 + Count State 1 | 58 |
| EXP3 | Raw + Temporal 10 | 57 |
| EXP4 | Raw + Count State 1 | 48 |

---

## 3. EXP3 Temporal Features

기존 `asof_*` Feature를 제거하지 않고, 투수의 **최근 기록과 해당 시점까지의 누적 기록 간 차이**, 그리고 **최근 구간 간 변화량**을 추가했습니다.

### Recent vs Cumulative — 6 Features

```text
recent1_success_vs_career
recent3_success_vs_career
recent5_success_vs_career

recent1_middle_vs_career
recent3_middle_vs_career
recent5_middle_vs_career
```

예:

```text
recent3_success_vs_career
= 최근 3경기 success rate
- 해당 투구 시점까지의 누적 success rate
```

변수명에는 `career`가 포함되어 있지만 실제 의미는 선수의 최종 커리어 기록이 아니라 **해당 시점까지 계산된 as-of 누적값**입니다.

### Trend — 4 Features

```text
success_trend_1_3
success_trend_3_5

middle_trend_1_3
middle_trend_3_5
```

예:

```text
success_trend_1_3
= 최근 1경기 success rate
- 최근 3경기 success rate
```

---

## 4. Count State

기존 `balls_before`, `strikes_before`는 그대로 유지하면서 두 값을 결합한 범주형 Feature를 추가했습니다.

예:

```text
balls_before = 3
strikes_before = 2

→ count_state = "3_2"
```

---

## 5. Initial Ensemble Feature Experiment

기존 CatBoost + XGBoost Ensemble의 모델 구조와 가중치를 유지하고 Feature만 변경했습니다.

- CatBoost weight: `0.5655685`
- XGBoost weight: `0.4344315`

### Local Validation

시간 순서를 유지하여 다음 두 구간에서 검증했습니다.

- 2019~2022 → 2023 Validation
- 2019~2023 → 2024 Validation

| Model | 2023 Brier ↓ | 2024 Brier ↓ |
|---|---:|---:|
| Basic | 0.253595 | 0.248270 |
| EXP1 | 0.253491 | **0.248239** |
| EXP2 | 0.253458 | 0.248277 |
| EXP3 | **0.253373** | 0.248326 |
| EXP4 | 0.253585 | 0.248268 |

### Public Leaderboard

| Experiment | Feature | Public LB |
|---|---|---:|
| Basic | Raw 47 | **829.4690** |
| EXP1 | Temporal + Count | 819.8670 |
| **EXP3** | **Temporal only** | **840.9992** |
| EXP4 | Count only | 818.2141 |

초기 Ensemble 실험에서는 EXP3이 가장 높은 Public LB를 기록했습니다.

다만 이 결과만으로 Temporal Feature 자체의 효과라고 단정하지 않고, 단일 모델 실험으로 추가 확인했습니다.

---

## 6. CatBoost Single-Model Verification

기존 CatBoost Basic과 동일한 모델 구조와 설정을 유지한 상태에서 Temporal Feature 10개만 추가했습니다.

| Model | Public LB |
|---|---:|
| CatBoost Basic | **783.26** |
| CatBoost + EXP3 | 779.299 |

현재 CatBoost 설정에서는 EXP3 적용 후 Public LB 개선이 확인되지 않았습니다.

2024 Local Validation에서도:

| Model | 2024 Brier ↓ | Validation Score |
|---|---:|---:|
| CatBoost Basic | **0.24831851** | **595.83** |
| CatBoost + EXP3 | 0.24836432 | 577.49 |

로 나타났습니다.

---

## 7. XGBoost Single-Model Verification

기존 XGBoost Basic의 전처리와 하이퍼파라미터를 유지하고 Temporal Feature 10개만 추가했습니다.

유지한 설정:

- `OrdinalEncoder`
- Categorical Feature 정책
- ID Feature numeric 처리
- XGBoost 하이퍼파라미터
- Brier Score 기반 Early Stopping
- 2019~2023 Train / 2024 Validation

| Model | Public LB |
|---|---:|
| XGBoost Basic | **759.36** |
| XGBoost + EXP3 | 714.143 |

현재 XGBoost 설정에서도 EXP3 적용 후 Public LB 개선이 확인되지 않았습니다.

---

## 8. Overall Result

| Model / Experiment | Public LB |
|---|---:|
| XGBoost Basic | **759.36** |
| XGBoost + EXP3 | 714.143 |
| CatBoost Basic | **783.26** |
| CatBoost + EXP3 | 779.299 |
| Ensemble Basic | 829.469 |
| EXP1 Ensemble | 819.867 |
| **EXP3 Ensemble** | **840.999** |
| EXP4 Ensemble | 818.214 |

단순 비교하면:

```text
CatBoost
Basic > EXP3

XGBoost
Basic > EXP3

Ensemble
EXP3 > Basic
```

---

## 9. Current Interpretation

단일 모델을 고정하고 Feature 효과를 다시 확인한 결과, **CatBoost와 XGBoost 모두 EXP3 적용 후 Basic 대비 Public LB 개선이 확인되지 않았습니다.**

반면 초기 Ensemble 실험에서는 EXP3 적용 후 Public LB가 상승했습니다.

현재 실험만으로는 **왜 단일 모델과 Ensemble에서 결과가 다르게 나타났는지 정확한 원인을 판단하기 어렵습니다.**

가능한 요인으로는 다음과 같은 부분을 생각해볼 수 있습니다.

- Temporal Feature가 CatBoost와 XGBoost의 예측에 서로 다른 영향을 주었을 가능성
- 두 모델을 결합하는 과정에서 예측이 보완적으로 작용했을 가능성
- 모델 구조 및 Feature 활용 방식의 차이
- 전처리 방식의 차이
- Ensemble weight와의 조합

다만 현재는 이를 구분하기 위한 추가 실험을 진행하지 않았기 때문에 특정 원인으로 단정하지 않습니다.

> 현재 확인된 사실은 EXP3이 CatBoost와 XGBoost 단일 모델에서는 Basic 대비 성능 개선을 보이지 않았지만, 기존 Ensemble 실험에서는 성능 상승이 관찰되었다는 점입니다. 이 차이가 발생한 이유는 추가 검증이 필요합니다.

---

## 10. Count State

Count State가 포함된 EXP1과 EXP4는 현재 Ensemble 실험에서 Basic보다 낮은 Public LB를 기록했습니다.

```text
Basic Ensemble : 829.469
EXP1           : 819.867
EXP4           : 818.214
```

따라서 **현재 Feature 구성과 모델 설정에서는 Count State 추가에 따른 성능 개선을 확인하지 못했습니다.**

이는 Count State 자체가 항상 불필요하다는 의미가 아니라, 현재 실험 범위에서는 개선 효과가 관찰되지 않았다는 의미입니다.

---

## 11. Directory Structure

```text
Temporal Feature Experiment/
├── README.md
│
├── ensemble/
│   ├── basic/
│   │   └── train.py
│   └── exp/
│       └── train.py
│
├── exp3_catboost/
│   └── train.py
│
└── exp3_xgboost/
    └── train.py
```

### File Description

- `ensemble/basic/train.py`
  - 기존 CatBoost + XGBoost Basic Ensemble 학습 코드

- `ensemble/exp/train.py`
  - Ensemble Feature 실험 코드
  - `exp1`: Temporal + Count
  - `exp3`: Temporal only
  - `exp4`: Count only

- `exp3_catboost/train.py`
  - CatBoost 단독 모델에서 EXP3 Temporal Feature 효과를 확인하는 코드

- `exp3_xgboost/train.py`
  - XGBoost 단독 모델에서 EXP3 Temporal Feature 효과를 확인하는 코드

---

## 12. Conclusion

이번 실험에서는 초기 Ensemble 기준 EXP3이 가장 높은 Public LB를 기록했습니다.

하지만 Feature Engineering 효과를 단일 모델 기준으로 다시 확인했을 때 CatBoost와 XGBoost 모두 Basic 대비 개선이 확인되지 않았습니다.

따라서 현재 단계에서는:

- EXP3의 단일 모델 개선 효과는 확인되지 않음
- Ensemble에서는 EXP3 적용 후 성능 상승이 관찰됨
- 단일 모델과 Ensemble 결과 차이의 정확한 원인은 아직 불명확함
- Count State는 현재 구성에서 개선 효과가 확인되지 않음

으로 정리합니다.

추가 실험이 필요한지, 또는 다른 해석이 가능한지는 팀 논의 후 결정합니다.

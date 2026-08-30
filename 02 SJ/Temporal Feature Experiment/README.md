# Temporal Feature Experiment

LG Aimers KBO 제구 성공 확률 예측에서 **투수의 최근 성적 변화(Temporal Feature)** 와 **볼카운트 상태(Count State)** 가 기존 CatBoost + XGBoost 앙상블의 성능에 어떤 영향을 주는지 확인한 실험입니다.

이번 실험은 최종 모델 확정을 위한 것이 아니라, 기존에 조사한 KBO/xStrikes 관련 아이디어를 실제 데이터에 간단히 적용해 **추후 Feature Engineering에서 활용할 만한 방향을 탐색**하기 위해 진행했습니다.

## Baseline

기본 모델은 CatBoost와 XGBoost의 Soft Voting Ensemble입니다.

- CatBoost weight: `0.5655685`
- XGBoost weight: `0.4344315`
- Raw Feature: 47개
- Public LB: **829.4690**

## Experiment

### EXP-1 — Temporal + Count

기존 47개 Feature에 아래 11개 Feature를 추가합니다.

- Temporal Feature 10개
  - 최근 1/3/5경기 success rate와 career success rate의 차이
  - 최근 1/3/5경기 middle rate와 career middle rate의 차이
  - 최근 1↔3경기, 3↔5경기 success/middle 변화량
- Count State 1개
  - `balls_before`와 `strikes_before`를 결합한 범주형 Feature

총 58개 Feature를 사용합니다.

### EXP-2 — Recent Form State

EXP-1에 `recent3_success_vs_career`를 기반으로 한 상태 Feature를 추가합니다.

- `decline`
- `improve_or_equal`
- `missing`

2023 Validation에서는 개선되었지만 2024에서 재현되지 않아 최종 제출 후보에서는 제외했습니다.

### EXP-3 — Temporal Only

기존 47개 Feature에 **Temporal Feature 10개만 추가**합니다.

총 57개 Feature를 사용하며, 이번 실험에서 가장 높은 Public LB를 기록했습니다.

### EXP-4 — Count Only

기존 47개 Feature에 `count_state` 1개만 추가합니다.

총 48개 Feature를 사용합니다.

## Validation Result

시간 순서를 유지하기 위해 다음 두 구간에서 검증했습니다.

- 2019~2022 → 2023 Validation
- 2019~2023 → 2024 Validation

| Model | 2023 Brier ↓ | 2024 Brier ↓ |
|---|---:|---:|
| Basic | 0.253595 | 0.248270 |
| EXP-1 | 0.253491 | **0.248239** |
| EXP-2 | 0.253458 | 0.248277 |
| EXP-3 | **0.253373** | 0.248326 |
| EXP-4 | 0.253585 | 0.248268 |

## Public Leaderboard Result

| Model | Feature | Public LB |
|---|---|---:|
| Basic | Raw 47 | 829.4690 |
| EXP-1 | Temporal + Count | 819.8670 |
| **EXP-3** | **Temporal Only** | **840.9992** |
| EXP-4 | Count Only | 818.2141 |

**Best: EXP-3 — 840.9992**

Baseline 대비 약 **+11.53점** 개선되었습니다.

## Conclusion

이번 실험에서는 **Temporal Feature만 추가한 EXP-3이 가장 좋은 성능**을 보였습니다.

특히 다음 정보가 추후 Feature Engineering에서 다시 활용할 가치가 있는 것으로 확인되었습니다.

- 투수의 최근 성적과 장기 평균의 차이
- 최근 1/3/5경기 사이의 성적 변화
- 최근 컨디션의 상승/하락 추세

반면 `count_state`는 단독 및 Temporal Feature와의 결합 모두 Public LB에서 성능이 하락했습니다.

따라서 이번 결과는 **EXP-3을 최종 모델로 확정한다는 의미가 아니라, 이후 Feature Engineering에서 Temporal 계열 Feature를 우선적으로 검토할 근거**로 사용합니다.

## Directory Structure

```text
Temporal Feature Experiment/
├── basic_ensenble/
│   ├── basic_ensenble_train.py
│   └── submit/
│       ├── model/
│       │   └── rf.pkl
│       ├── script.py
│       └── requirements.txt
│
├── exp_code/
│   ├── error_analysis.py
│   ├── error_analysis_v001/
│   ├── exp1_core_fe.py
│   ├── exp1_core_fe_v001/
│   ├── exp1_core_fe_2024.py
│   ├── exp1_core_fe_2024_v001/
│   ├── exp2_recent_form.py
│   ├── exp2_recent_form_v001/
│   ├── exp3_temporal_only.py
│   ├── exp3_temporal_only_v001/
│   ├── exp4_count_only.py
│   └── exp4_count_only_v001/
│
├── exp_train.py
│
├── exp1_s/
│   ├── model/ensemble.pkl
│   ├── script.py
│   └── requirements.txt
│
├── exp3_s/
│   ├── model/ensemble.pkl
│   ├── script.py
│   └── requirements.txt
│
└── exp4_s/
    ├── model/ensemble.pkl
    ├── script.py
    └── requirements.txt
```

## File Description

- `basic_ensenble/basic_ensenble_train.py`
  - 기존 CatBoost + XGBoost Baseline 학습 코드

- `exp_code/error_analysis.py`
  - Baseline의 2023 Validation 오차를 count, base state, LI, inning, recent-vs-career 기준으로 분석

- `exp_code/exp1_core_fe.py`
  - Temporal 10개 + Count State 1개를 추가한 EXP-1의 2023 Validation

- `exp_code/exp1_core_fe_2024.py`
  - EXP-1의 2024 Validation 재검증

- `exp_code/exp2_recent_form.py`
  - Recent Form State Feature 추가 효과 검증

- `exp_code/exp3_temporal_only.py`
  - Temporal Feature 10개만 사용하는 EXP-3 검증

- `exp_code/exp4_count_only.py`
  - Count State 1개만 사용하는 EXP-4 검증

- `exp_code/*_v001/`
  - 각 실험의 `summary.csv`, validation prediction 및 Feature 목록 저장 폴더

- `exp_train.py`
  - EXP-1 / EXP-3 / EXP-4를 2019~2024 전체 데이터로 재학습하는 코드
  - `--mode exp1`, `--mode exp3`, `--mode exp4`로 실행

- `exp1_s/`, `exp3_s/`, `exp4_s/`
  - 각 실험의 실제 제출용 모델 및 inference 파일
  - `model/ensemble.pkl`, `script.py`, `requirements.txt`로 구성

## Final Training

프로젝트 루트에 `./data/train.csv`, `./data/test.csv`가 있는 상태에서 실행합니다.

```bash
python exp_train.py --mode exp1
python exp_train.py --mode exp3
python exp_train.py --mode exp4
```

각 모델은 전체 2019~2024 데이터를 사용해 학습됩니다.

## Requirements

```text
scikit-learn==1.8.0
joblib==1.5.3
pandas==2.3.3
catboost
xgboost
```

## Note

- 모든 실험에서 모델 구조와 앙상블 가중치는 동일하게 유지하고 Feature 구성만 변경했습니다.
- 추가 Feature는 제공된 `asof_*` 정보와 현재 행의 볼/스트라이크 정보만 사용합니다.
- Test row 간 aggregation, rolling, frequency 등은 사용하지 않습니다.
- `__MACOSX/`, `.DS_Store`는 macOS에서 자동 생성된 파일로 실험과 무관하며 삭제해도 됩니다.

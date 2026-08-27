# Combined Feature Experiment (cb_reg_v002_fe_v001)

`model/cb_reg_v002`(raw asof_* 47개, CatBoostRegressor RMSE)를 기준으로,
다음을 결합한 새 feature set을 검증하는 실험이다.

1. **BBEB(Beta-Binomial Empirical Bayes) 7개** — `asof_pitcher_success/middle/ball/
   strike/reverse_rate`, `asof_batter_success/middle_rate`를 제거하고
   `bbeb_<name>_rate`로 교체. Prior fit 방식은 `Reliability Feature Experiment/
   cb_reg_v002_eb_experiments/cb_reg_v002_eb_replace_v001_train.py`(지금까지
   가장 좋은 성과를 낸 변형, pooled_oof_clipped_brier=0.24716861)와 동일하게
   **그 fold의 train 구간 전체(expanding)**로 fit한다. pitch-mix 3개(fastball/
   breaking/offspeed)는 이번 실험에서는 raw 그대로 유지한다.
2. **신규 상황(situational) feature 5개** — `count_balance`, `full_count`,
   `scoring_position`, `pitcher_win_expectancy`, `platoon`.
3. **`eb_pitcher_batter_gap`** — `bbeb_pitcher_success_rate - bbeb_batter_success_rate`.
4. **Temporal 10개** — `Temporal Feature Experiment/exp_code/exp3_temporal_only.py`의
   EXP-3 공식과 동일(raw `asof_pitcher_success_rate`/`asof_pitcher_middle_rate`를
   계산에만 사용하고 최종 feature 목록에는 넣지 않음).

총 57개 feature, CatBoostRegressor(RMSE), Expanding Window Validation
(validation seasons 2021~2024)을 사용한다.

## pitcher_win_expectancy 정의

`home_win_expectancy`/`away_win_expectancy`를 "현재 투수 팀 관점"으로 통일한다.
투수 팀이 홈/원정인지는 리포지토리에 기존 구현이 없어 `top_bottom`과
`score_diff_home`/`score_diff_pitcher_team`을 실제 train.csv로 교차검증해서
확인했다: `top_bottom=='T'`(초)일 때 `score_diff_home == score_diff_pitcher_team`이
100% 일치 → **투수 팀 = 홈팀**. `'B'`(말)일 때는 반대 → **투수 팀 = 원정팀**.
(야구 규칙과도 일치: 홈팀은 말에 공격한다.)

## platoon 정의

`pitcher_hand`/`batter_hand`는 `{1, 2}` 정수 코드이고 L/R 매핑 문서가 없어,
원본 코드를 그대로 이어붙인다(`"1_1"`, `"1_2"`, `"2_1"`, `"2_2"`).

## Leakage 안전성

- BBEB prior는 항상 해당 fold의 train 구간 전체에서만 fit한다.
- 상황/Temporal feature는 그 행 자신의 raw 값만 사용하는 row-wise 산술이라
  fold와 무관하게 미리 한 번만 계산해도 leakage가 없다.
- test.csv 적용 시에는 train.csv 전체로 fit한 최종 BBEB prior를 얼려서 사용한다.

## 실행

```bash
python "EDA feature experiment/Combined_Feature_Experiment/cb_reg_v002_fe_v001_train.py"
```

저장소 루트에서 실행한다 (`./data`, `./model`, `./validation` 상대 경로 사용).

## 비교 방법

`model/cb_reg_v002`는 재학습하지 않고 기존 `validation/cb_reg_v002/metadata.json`의
`summary.pooled_oof_clipped_brier`를 그대로 baseline으로 재사용한다. 본 스크립트
실행 마지막에 새 실험의 `pooled_oof_clipped_brier`와 나란히 비교해서 출력한다.

## 결과

- **pooled_oof_clipped_brier = 0.24711557** (baseline `cb_reg_v002` 0.24718539
  대비 개선, BSS +27.9)
- **Public LB = 838.3809069808** (baseline LB 829.22 대비 +9.16로, 내부 OOF
  개선 방향이 실제 리더보드에서도 확인됨)
- Row/Feature QA 전부 통과(row 수·순서·target 불변, `full_count`/
  `scoring_position` 정의 검증).
- `Reliability Feature Experiment/cb_reg_v002_eb_experiments/
  cb_reg_v002_eb_replace_v001`(BBEB 단독, OOF 0.247169)보다도 개선.

### Seed 재현성 검증

동일 feature로 `random_seed`만 7 / 2024로 바꿔 재학습한 결과, pooled OOF
Brier는 각각 0.24711160 / 0.24712531(원본 seed=42: 0.24711557)이었다.
3-seed 표준편차는 5.8e-06(BSS 환산 약 2.31)로, baseline(`cb_reg_v002`)과의
격차(6.8e-05)가 이 seed 노이즈의 약 11.8배에 달해 baseline 대비 개선이
seed 운이 아님을 확인했다.

## 산출물

- `model/cb_reg_v002_fe_v001/` — 최종 모델, `final_metadata.json`,
  `feature_manifest.json`(raw/BBEB/situation/temporal feature 목록 및 개수),
  `bbeb_priors.json`(전체 train으로 fit한 최종 prior), `test_features_preview.csv`
- `validation/cb_reg_v002_fe_v001/` — fold별 결과, `metadata.json`(pooled Brier 등),
  `qa_report.json`(Row QA + Feature QA + 특수 검증 결과)

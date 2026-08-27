# Combined Feature Experiment — fe_v002 계열 (conditional shrinkage ablation)

`cb_reg_v002_fe_v001`(BBEB7 + situational5 + eb_pitcher_batter_gap +
Temporal10, 57 feature, pooled_oof_clipped_brier=**0.24711557**, Public
LB=838.38, 별도 브랜치 `experiment/situational_feature_v001`)를 기준으로,
투수x조건(platoon/count) 조건부 shrinkage feature를 추가했을 때 성능이
어떻게 변하는지 추적하는 ablation 시리즈다. 첫 시도(fe_v002)가
baseline보다 악화되어, 그 원인을 하나씩 격리해가며 검증한 16개 실험을
모아 정리했다.

이 폴더의 모든 `*_train.py`는 저장소 루트에서 실행하며(`./data`, `./model`,
`./validation` 상대 경로 사용), 실행마다 `model/<EXPERIMENT_NAME>/`와
`validation/<EXPERIMENT_NAME>/`에 산출물을 남긴다.

```bash
python "EDA feature experiment/Combined_Feature_Experiment/<스크립트명>.py"
```

비교 지표는 전부 **pooled OOF clipped Brier score**(`validation/<name>/
metadata.json`의 `summary.pooled_oof_clipped_brier`, **낮을수록 좋음**)이다.

## 실험 결과 요약 (Brier 오름차순 = 좋은 순)

| 순위 | 스크립트 | pooled_oof_clipped_brier | vs fe_v001 |
|---|---|---|---|
| — | `cb_reg_v002_fe_v001` (baseline, 별도 브랜치) | 0.24711557 | — |
| 1 | `cb_reg_v002_fe_v002j_no_game_month_train.py` | 0.24710723 | **-0.0000083** (근소 개선) |
| 2 | `cb_reg_v002_fe_v002k_no_player_id_train.py` | 0.24711503 | -0.0000005 (거의 동일) |
| 3 | `cb_reg_v002_fe_v002n_hbeb_success_train.py` | 0.24712091 | +0.0000053 (거의 동일) |
| 4 | `cb_reg_v002_fe_v002h2_count_bbeb_delta_only_train.py` | 0.24715609 | +0.0000405 |
| 5 | `cb_reg_v002_fe_v002i2_platoon_bbeb_delta_only_train.py` | 0.24735449 | +0.0002389 |
| 6 | `cb_reg_v002_fe_v002l_count_platoon_bbeb_delta_train.py` | 0.24741517 | +0.0002996 |
| 7 | `cb_reg_v002_fe_v002m_no_id_count_platoon_bbeb_delta_train.py` | 0.24743825 | +0.0003227 |
| 8 | `cb_reg_v002_fe_v002h_count_bbeb_train.py` | 0.24766133 | +0.0005457 |
| 9 | `cb_reg_v002_fe_v002e_delta_only_train.py` | 0.24793149 | +0.0008159 |
| 10 | `cb_reg_v002_fe_v002c_count_only_train.py` | 0.24801057 | +0.0008950 |
| 11 | `cb_reg_v002_fe_v002i_platoon_bbeb_train.py` | 0.24852347 | +0.0014079 |
| 12 | `cb_reg_v002_fe_v002_mingroupn10_train.py` | 0.24870141 | +0.0015858 |
| 13 | `cb_reg_v002_fe_v002_mingroupn50_train.py` | 0.24872978 | +0.0016142 |
| 14 | `cb_reg_v002_fe_v002_train.py` | 0.24880612 | +0.0016905 (악화) |
| 15 | `cb_reg_v002_fe_v002b_platoon_only_train.py` | 0.24881300 | +0.0016974 (악화) |
| 16 | `cb_reg_v002_fe_v002_mingroupn30_train.py` | 0.24892446 | +0.0018089 (최악) |

**결론**: platoon/count를 GITS 스타일 고정-strength 계층 shrinkage로 추가하면
거의 항상 baseline보다 악화된다. 원인은 대부분 `platoon_rate`가 이미
top feature인 `bbeb_pitcher_success_rate`와 corr=0.76으로 겹치기
때문(ablation E/B/C로 격리 확인)이며, 우리 BBEB(MLE) 방식 + delta만 사용하면
(h2, i2) 손해를 크게 줄일 수 있지만 그래도 baseline을 능가하진 못한다.
반대로 **`game_month` 하나만 제거(j)**하는 가장 단순한 ablation이 유일하게
baseline을 (근소하게) 앞섰다. 이 시리즈에서 채택 가능한 최선의 카드는
`fe_v002j`(game_month 제거)와 `fe_v002h2`(count_bbeb_delta only) 두 개이며,
나머지는 모두 baseline 대비 이득이 없어 drop 상태다.

---

## 실험별 상세

### fe_v002 — 본실험, GITS 스타일 platoon/count shrinkage 추가 (61개)

fe_v001(57 feature) 기준으로 **Platoon/Count 조건부 EB feature 4개**를
추가(총 61개). GITS(brik0519/GITS, jiyunjung0/Baseball 비공식 clean-room
재구성) 저장소의 `common_features.py`(build_platoon_lookup /
build_count_lookup) 아이디어를 이식: "투수 통산 rate → 투수x조건 rate"
2단계 계층적 shrinkage를 고정 strength(platoon=80, count=120)로 적용.

```
parent_rate = (sum + strength*global_rate) / (n + strength)   # 투수 통산
cond_rate   = (sum + strength*parent_rate) / (n + strength)   # 투수x조건
cond_delta  = cond_rate - parent_rate
```

- `platoon_rate`/`platoon_delta`: 투수 x 타자손(batter_hand) 조건
- `count_rate`/`count_delta`: 투수 x 볼카운트 조건

결과: **0.24880612** — baseline보다 악화. 이 실패를 원인 분리하기 위해
아래 ablation B~N이 이어진다.

### `cb_reg_v002_fe_v002_mingroupn10/30/50_train.py` — Step 4: 표본 부족 대응
fe_v002가 악화된 원인이 표본 부족(pitcher x condition 그룹이 너무 작음)일
가능성을 검증. pitcher x condition 그룹의 (LOO 반영) 표본 수가
`MIN_GROUP_N` 미만이면 `group_rate` 대신 `parent_rate`(투수 통산)로 hard
fallback(delta=0). `MIN_GROUP_N`=10/30/50 세 버전 비교. 그 외 로직은 fe_v002와
동일(61 feature, platoon+count 둘 다 포함).

결과: 0.24870141 / 0.24892446 / 0.24872978 — 셋 다 fe_v002보다 약간 낫거나
비슷하지만, 여전히 baseline보다 크게 악화. 표본 부족만으로는 설명되지 않음.

### `cb_reg_v002_fe_v002b_platoon_only_train.py` — Ablation B
fe_v002의 원인 분리: **platoon 관련 2개(platoon_rate, platoon_delta)만**
fe_v001에 추가, count는 제외.

결과: **0.24881300** — fe_v002 전체 추가(0.24880612)와 거의 동일한 악화폭.
즉 fe_v002 악화의 대부분이 platoon에서 온다는 것을 확인.

### `cb_reg_v002_fe_v002c_count_only_train.py` — Ablation C
**count 관련 2개(count_rate, count_delta)만** fe_v001에 추가, platoon은 제외.

결과: **0.24801057** — B(platoon only)보다는 덜 나쁘지만 여전히 baseline보다
악화.

### `cb_reg_v002_fe_v002e_delta_only_train.py` — Ablation E
B/C 결과 분석으로 `platoon_rate`/`count_rate`가 이미 top feature인
`bbeb_pitcher_success_rate`와 corr=0.76으로 강하게 겹친다는 걸 확인. rate
계열을 모델 입력에서 빼고 **delta 계열만(platoon_delta, count_delta, corr
< 0.08로 독립적)** 넣으면 collinearity로 인한 악화가 사라지는지 검증.

결과: **0.24793149** — 개선되긴 했지만 여전히 baseline보다 악화. rate/delta
문제만으로는 fe_v002 악화를 다 설명하지 못함.

### BBEB(MLE) 방식으로 shrinkage 교체 — H, H2, I, I2

ablation 시리즈(B/C/E/mingroupn10~50)에서 GITS 스타일 고정-strength
계층 shrinkage로는 platoon/count 둘 다 계속 악화됨을 확인했고, `platoon`은
fe_v001에 이미 범주형 feature(pitcher_hand_batter_hand concat, 57개 중
importance 4위)로 존재해 이중 중복 가능성도 있었다. 그래서 "shrinkage
방식 자체를 GITS 고정 strength → 우리 BBEB(MLE)로 바꾸면 나아지는가"를
격리해서 검증한 시리즈.

#### `cb_reg_v002_fe_v002h_count_bbeb_train.py`
platoon 계열은 아예 넣지 않고, `count_bbeb_rate`/`count_bbeb_delta`만
추가: `(pitcher_id, balls_before, strikes_before)` 조합을 group으로 삼아
`fit_beta_binomial_prior_from_summary()`로 alpha/beta를 MLE 추정한 뒤 그
prior로 각 행의 conjugate posterior를 계산(기존 BBEB의 target 방식과 동일).

결과: **0.24766133** — GITS 방식의 ablation C(0.24801057)보다 fe_v001과의
격차가 크게 좁혀짐. shrinkage 방식 교체가 유효함을 확인.

#### `cb_reg_v002_fe_v002h2_count_bbeb_delta_only_train.py`
h와 동일한 count_bbeb 계산이지만, **delta만** 최종 feature에 포함(rate 제외).

결과: **0.24715609** — baseline(0.24711557)과 거의 같은 수준까지 좁혀짐.
h2가 이 시리즈 중 baseline에 가장 근접한 결과.

#### `cb_reg_v002_fe_v002i_platoon_bbeb_train.py`
h와 같은 처방을 platoon에 적용: 기존 범주형 `platoon` 컬럼은 최종 feature
목록에서 제거하고, 대신 `(pitcher_id, batter_hand)` 조합을 group으로 하는
`platoon_bbeb_rate`/`platoon_bbeb_delta`를 BBEB(MLE)로 계산해 추가.

결과: **0.24852347** — h(count_bbeb)만큼 개선되지 않음. platoon은 shrinkage
방식을 바꿔도 여전히 손해가 큰 편.

#### `cb_reg_v002_fe_v002i2_platoon_bbeb_delta_only_train.py`
i와 동일한 platoon_bbeb 계산이지만 **delta만** 포함(rate 제외).

결과: **0.24735449** — i보다는 크게 개선되지만, h2(count_bbeb delta only)만큼
baseline에 근접하지는 못함.

### 결합 및 추가 ablation — L, M

#### `cb_reg_v002_fe_v002l_count_platoon_bbeb_delta_train.py`
h2와 i2에서 확인한 "BBEB(MLE) + delta only" 최선의 처방을 **count와
platoon 둘 다** 결합(`count_bbeb_delta` + `platoon_bbeb_delta`, 총 59개).

결과: **0.24741517** — h2 단독(0.24715609)보다 오히려 나빠짐. 두 delta를
같이 넣는 것이 단독으로 넣는 것보다 이득이 없고 약간의 간섭이 있는 것으로
보임.

#### `cb_reg_v002_fe_v002m_no_id_count_platoon_bbeb_delta_train.py`
l과 동일한 feature 구성에 더해, `pitcher_id`/`batter_id`/
`pitcher_team_id`/`batter_team_id`를 **모델 입력 feature에서 제외**(K의
player-ID 제거를 재현·결합). BBEB/count_bbeb/platoon_bbeb의 그룹핑 키로는
계속 필요하므로 데이터 로딩 단계에서는 유지.

결과: **0.24743825** — l과 거의 동일(차이 미미). player ID 제거가 이
조합에서는 뚜렷한 효과가 없음.

### 단일 feature 제거 ablation — J, K

#### `cb_reg_v002_fe_v002j_no_game_month_train.py`
fe_v001(57 feature)에서 **`game_month` 딱 1개만 제거**(56 feature). GITS
자료가 "game_month가 조건부 feature와 결합 시 CatBoost에서 해롭다"고
경고했는데, fe_v001을 포함한 ablation 시리즈 B~I가 전부 game_month를 그대로
두고 있었기에, 그 경고가 fe_v001 단독(조건부 feature 없이)에도 적용되는지
가장 단순하게 확인.

결과: **0.24710723** — **이 시리즈 전체에서 유일하게 baseline을 앞선 실험**
(근소하지만). game_month 제거가 조건부 feature 유무와 무관하게 약간의
이득이 있을 가능성.

#### `cb_reg_v002_fe_v002k_no_player_id_train.py`
fe_v001에서 raw player ID categorical 4개(`pitcher_id`, `batter_id`,
`pitcher_team_id`, `batter_team_id`)를 전부 제거(57→53 feature). GITS 자료가
"raw player ID categorical이 해로웠다"(자기네 파이프라인 기준 LGBM+ID:
637→173, CatBoost+ID: 770→617 BSS 급락)고 보고한 걸 이 파이프라인에서 검증.
`pitcher_id`/`batter_id`는 BBEB 7개 중 5개의 그룹핑 키로 여전히 필요하므로
모델 입력 feature 목록에서만 제외.

결과: **0.24711503** — baseline과 거의 동일(차이 무시할 수준). 이
파이프라인에서는 player ID의 해로운 영향이 GITS 보고만큼 크지 않음.

### 대안 shrinkage 방식 — N

#### `cb_reg_v002_fe_v002n_hbeb_success_train.py`
fe_v001을 베이스로, `pitcher_success`/`batter_success` 2개 feature의 보정
방식만 `cb_reg_v003`의 HBEB(손잡이 2-group, method-of-moments prior, 실제
LB 840.80으로 프로젝트 전체 1위)로 교체. `cb_reg_v003`이 OOF는 fe_v001보다
낮은데 실LB는 더 높다는 OOF-LB 역전 관측을 근거로, "이 보정 방식이 실제로는
더 잘 일반화되는 것 아닌가"를 검증하기 위해 나머지 구성(situational 5 +
`eb_pitcher_batter_gap` + Temporal 10 + 나머지 BBEB 5개)은 그대로 유지한 채
pitcher_success/batter_success만 격리해서 교체.

1. **BBEB 5개**(`asof_pitcher_middle/ball/strike/reverse_rate`,
   `asof_batter_middle_rate`) — fe_v001과 완전히 동일한 로직.
2. **HBEB 2개**(`hbeb_pitcher_success_rate`, `hbeb_batter_success_rate`) —
   `model/cb_reg_v003/train.py`의 `compute_hbeb_priors()`/`apply_hbeb()`를
   그대로 재사용(method-of-moments prior, group=pitcher_hand/batter_hand).
3. `eb_pitcher_batter_gap`을 `hbeb_*` 기준으로 재정의.
4. Temporal 10개는 fe_v001과 동일(raw 컬럼 참조는 hbeb 교체와 무관).

결과: **0.24712091** — OOF 기준으로는 baseline과 거의 동일(근소하게 악화).
OOF만으로는 HBEB가 우수하다는 가설을 지지하지 못하며, 실LB 검증이 별도로
필요함(OOF-LB 역전이 이 격리 실험에도 재현되는지는 미확인).

---

## 공통 정의

fe_v001에서 이어받은 정의로, 이 시리즈의 모든 스크립트에도 동일하게
적용된다.

### pitcher_win_expectancy 정의
`home_win_expectancy`/`away_win_expectancy`를 "현재 투수 팀 관점"으로
통일한다. 투수 팀이 홈/원정인지는 리포지토리에 기존 구현이 없어
`top_bottom`과 `score_diff_home`/`score_diff_pitcher_team`을 실제
train.csv로 교차검증해서 확인했다: `top_bottom=='T'`(초)일 때
`score_diff_home == score_diff_pitcher_team`이 100% 일치 → **투수 팀 =
홈팀**. `'B'`(말)일 때는 반대 → **투수 팀 = 원정팀**(야구 규칙과도 일치:
홈팀은 말에 공격한다).

### platoon 정의
`pitcher_hand`/`batter_hand`는 `{1, 2}` 정수 코드이고 L/R 매핑 문서가 없어,
원본 코드를 그대로 이어붙인다(`"1_1"`, `"1_2"`, `"2_1"`, `"2_2"`).

### Leakage 안전성 (모든 실험 공통)
- BBEB/HBEB/count_bbeb/platoon_bbeb prior는 항상 해당 fold의 train 구간
  전체(expanding)에서만 fit한다.
- 상황/Temporal feature는 그 행 자신의 raw 값만 사용하는 row-wise 산술이라
  fold와 무관하게 미리 한 번만 계산해도 leakage가 없다.
- test.csv 적용 시에는 train.csv 전체로 fit한 최종 prior를 얼려서 사용한다.

### 산출물 (모든 실험 공통)
- `model/<EXPERIMENT_NAME>/` — 최종 모델, `final_metadata.json`,
  `feature_manifest.json`(feature 목록 및 개수), prior JSON,
  `test_features_preview.csv`
- `validation/<EXPERIMENT_NAME>/` — fold별 결과, `metadata.json`(pooled
  Brier 등 `summary`), `qa_report.json`(Row QA + Feature QA + 특수 검증 결과)

### 비교 방법
`cb_reg_v002_fe_v001`(별도 브랜치 `experiment/situational_feature_v001`)은
재학습하지 않고 기존 `validation/cb_reg_v002_fe_v001/metadata.json`의
`summary.pooled_oof_clipped_brier`를 그대로 baseline으로 재사용한다. 각
스크립트는 실행 마지막에 새 실험의 `pooled_oof_clipped_brier`와 나란히
비교해서 출력한다.

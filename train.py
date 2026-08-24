"""
CatBoost 환경에서 Trackman feature 최종 확인용 handoff 스크립트.
preprocess_trackman.py 실행 결과(data/trackman/*)와 이 대화에서 만든
build_deviation_features.py 결과가 이미 존재한다고 가정한다.
Question 1("CatBoost에서도 개선이 없는가")만 확인하는 목적이며,
그 외 실험 설계(4-fold expanding window, feature 정의)는 HGB 검증과 동일하게 유지한다.
"""
import pandas as pd, numpy as np
from catboost import CatBoostClassifier

cols = ['season','pitcher_id','batter_id','pitcher_hand','batter_hand','game_type',
        'balls_before','strikes_before','outs_before','num_runners_on',
        'score_diff_pitcher_team','li','asof_pitcher_success_rate','asof_batter_success_rate','control_success']
df = pd.read_csv('data/train.csv', usecols=cols)
df['same_hand'] = (df.pitcher_hand==df.batter_hand).astype(int)
df['is_F'] = (df.game_type=='F').astype(int)

BASE_FEATS = ['balls_before','strikes_before','outs_before','num_runners_on',
              'score_diff_pitcher_team','li','asof_pitcher_success_rate','asof_batter_success_rate','same_hand','is_F']
PHYS_MEAN = ['tm_rel_speed_mean','tm_spin_rate_mean','tm_induced_vert_break_mean','tm_rel_height_mean','tm_rel_side_mean']
PHYS_STD  = ['tm_rel_speed_std','tm_spin_rate_std','tm_induced_vert_break_std','tm_rel_height_std','tm_rel_side_std']
REL_FEATS = ['tm_history_available','tm_n']

def brier(y, p): return np.mean((p-y)**2)

def fit_eval(tr, va, feats):
    m = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
                            loss_function='Logloss', verbose=False, random_seed=0)
    m.fit(tr[feats], tr.control_success)
    p = m.predict_proba(va[feats])[:,1]
    return brier(va.control_success.values, p)

rows=[]
for val_season in [2021,2022,2023,2024]:
    tr = df[df.season.isin(range(2019,val_season))].copy()
    va = df[df.season==val_season].copy()
    dev = pd.read_csv(f'data/trackman/trackman_pitcher_profiles_fold_{val_season}.csv')
    keep = ['pitcher_id','target_season']+REL_FEATS+PHYS_MEAN+PHYS_STD
    dev = dev[keep].rename(columns={'target_season':'season'})
    tr2 = tr.merge(dev, on=['pitcher_id','season'], how='left')
    va2 = va.merge(dev, on=['pitcher_id','season'], how='left')
    for c in REL_FEATS:
        tr2[c]=tr2[c].fillna(0); va2[c]=va2[c].fillna(0)
    bs0 = fit_eval(tr2, va2, BASE_FEATS)
    bs1 = fit_eval(tr2, va2, BASE_FEATS+REL_FEATS+PHYS_MEAN+PHYS_STD)
    rows.append(dict(val_season=val_season, baseline=bs0, with_trackman=bs1, delta=bs1-bs0))
    print(val_season, bs0, bs1, bs1-bs0)
print(pd.DataFrame(rows))
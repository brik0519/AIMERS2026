from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'model'
DATA=ROOT/'data'
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
META=joblib.load(MODEL/'preprocess.joblib')
TARGET='control_success'

# Row-wise features: only official current-row inputs. No target aggregation and no other prediction/test row is used.
def _n(frame,c,default=np.nan):
    if c in frame:return pd.to_numeric(frame[c],errors='coerce').to_numpy(np.float64)
    return np.full(len(frame),default,np.float64)

def build_self_state(frame):
    n=len(frame);z=pd.DataFrame(index=np.arange(n))
    pn=np.maximum(np.nan_to_num(_n(frame,'asof_pitcher_n',0),nan=0.0),0);bn=np.maximum(np.nan_to_num(_n(frame,'asof_batter_n',0),nan=0.0),0)
    career=_n(frame,'asof_pitcher_success_rate',.5);career=np.where(np.isfinite(career),career,.5)
    p1=_n(frame,'asof_pitcher_prev1_game_success_rate');p3=_n(frame,'asof_pitcher_prev3_game_success_rate');p5=_n(frame,'asof_pitcher_prev5_game_success_rate')
    p1=np.where(np.isfinite(p1),p1,career);p3=np.where(np.isfinite(p3),p3,p1);p5=np.where(np.isfinite(p5),p5,p3)
    bsucc=_n(frame,'asof_batter_success_rate',.5);bsucc=np.where(np.isfinite(bsucc),bsucc,.5)
    middle=_n(frame,'asof_pitcher_middle_rate',0);reverse=_n(frame,'asof_pitcher_reverse_rate',0);ball=_n(frame,'asof_pitcher_ball_rate',0);strike=_n(frame,'asof_pitcher_strike_rate',0)
    fast=_n(frame,'asof_pitcher_fastball_rate',1/3);brk=_n(frame,'asof_pitcher_breaking_rate',1/3);off=_n(frame,'asof_pitcher_offspeed_rate',1/3)
    fast=np.where(np.isfinite(fast),fast,1/3);brk=np.where(np.isfinite(brk),brk,1/3);off=np.where(np.isfinite(off),off,1/3);sm=np.maximum(fast+brk+off,1e-6);fast/=sm;brk/=sm;off/=sm
    balls=np.nan_to_num(_n(frame,'balls_before',0));strikes=np.nan_to_num(_n(frame,'strikes_before',0));inning=np.nan_to_num(_n(frame,'inning',1));score=np.nan_to_num(_n(frame,'score_diff_pitcher_team',0));li=np.maximum(np.nan_to_num(_n(frame,'li',0)),0)
    top=frame['top_bottom'].astype(str).to_numpy();hwe=np.nan_to_num(_n(frame,'home_win_expectancy',50),nan=50)/100.;awe=np.nan_to_num(_n(frame,'away_win_expectancy',50),nan=50)/100.;pwe=np.where(top=='T',hwe,awe)
    ph=frame['pitcher_hand'].astype(str).to_numpy();bh=frame['batter_hand'].astype(str).to_numpy()
    z['self__log_pitcher_n']=np.log1p(pn).astype(np.float32);z['self__pitcher_reliability']=(pn/(pn+120)).astype(np.float32)
    z['self__career_success']=career.astype(np.float32);z['self__prev1_success']=p1.astype(np.float32);z['self__prev3_success']=p3.astype(np.float32);z['self__prev5_success']=p5.astype(np.float32)
    z['self__recent_blend']=(.50*p1+.30*p3+.20*p5).astype(np.float32);z['self__prev1_minus_prev5']=(p1-p5).astype(np.float32);z['self__recent_minus_career']=(.5*p1+.3*p3+.2*p5-career).astype(np.float32)
    middle_safe=np.nan_to_num(middle,nan=0.0);m1=_n(frame,'asof_pitcher_prev1_game_middle_rate');m3=_n(frame,'asof_pitcher_prev3_game_middle_rate');m5=_n(frame,'asof_pitcher_prev5_game_middle_rate');m1=np.where(np.isfinite(m1),m1,middle_safe);m3=np.where(np.isfinite(m3),m3,m1);m5=np.where(np.isfinite(m5),m5,m3);z['self__recent_middle_blend']=(.50*m1+.30*m3+.20*m5).astype(np.float32);z['self__recent_middle_delta']=(m1-m5).astype(np.float32)
    z['self__log_batter_n']=np.log1p(bn).astype(np.float32);z['self__batter_reliability']=(bn/(bn+120)).astype(np.float32);z['self__batter_success']=bsucc.astype(np.float32);z['self__pitcher_minus_batter']=(career-bsucc).astype(np.float32)
    z['self__middle']=np.nan_to_num(middle,nan=0).astype(np.float32);z['self__reverse']=np.nan_to_num(reverse,nan=0).astype(np.float32);z['self__strike_ball_margin']=np.nan_to_num(strike-ball,nan=0).astype(np.float32)
    entropy=-(fast*np.log(np.clip(fast,1e-6,1))+brk*np.log(np.clip(brk,1e-6,1))+off*np.log(np.clip(off,1e-6,1)))
    z['self__pitchmix_entropy']=entropy.astype(np.float32);z['self__offspeed']=off.astype(np.float32);z['self__fast_off_logratio']=np.log(np.clip(fast,1e-6,1)/np.clip(off,1e-6,1)).astype(np.float32);z['self__break_off_logratio']=np.log(np.clip(brk,1e-6,1)/np.clip(off,1e-6,1)).astype(np.float32)
    z['self__count_pressure']=(.7*(balls==3)+1.1*((balls==3)&(strikes==2))+.25*(strikes==2)).astype(np.float32);z['self__full_count']=((balls==3)&(strikes==2)).astype(np.float32)
    z['self__inning']=inning.astype(np.float32);z['self__late']=(inning>=7).astype(np.float32);z['self__score_abs']=np.abs(score).astype(np.float32);z['self__li_log1p']=np.log1p(li).astype(np.float32)
    z['self__pitcher_we']=np.clip(pwe,0,1).astype(np.float32);z['self__win_uncertainty']=(1-2*np.abs(np.clip(pwe,0,1)-.5)).astype(np.float32);z['self__same_hand']=(ph==bh).astype(np.float32)
    pressure=z['self__count_pressure'].to_numpy(np.float64);z['self__career_x_pressure']=(career*pressure).astype(np.float32);z['self__recent_x_pressure']=((.5*p1+.3*p3+.2*p5)*pressure).astype(np.float32);z['self__offspeed_x_same_hand']=(off*(ph==bh)).astype(np.float32)
    return z

# ------------------------------------------------------------------
# Global-ASOF-aligned current-season decomposition.
# Official asof_n is global across R/F, so historical subtraction MUST
# use all official history, not one routed domain.
# ------------------------------------------------------------------
P_COMPONENT_RATES=['asof_pitcher_reverse_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']
B_COMPONENT_RATES=['asof_batter_middle_rate']

def _rounded_cumulative_count(n,rate):
    n=np.asarray(n,np.float64);rate=np.asarray(rate,np.float64);out=np.zeros(len(n),np.float64);ok=(n>0)&np.isfinite(rate);out[ok]=np.rint(n[ok]*rate[ok]);return out

def build_global_season_bank(history,cutoff):
    h=history.loc[history.season.astype(int)<int(cutoff)].copy()
    prior=float(h.loc[h.season.eq(h.season.max()),TARGET].mean()) if len(h) else .5
    pg=h.groupby('pitcher_id',sort=False)[TARGET].agg(['size','sum']) if len(h) else pd.DataFrame(columns=['size','sum'])
    bg=h.groupby('batter_id',sort=False)[TARGET].agg(['size','sum']) if len(h) else pd.DataFrame(columns=['size','sum'])
    bank={'cutoff':int(cutoff),'prior':prior,'pitcher_n':pg['size'].to_dict(),'pitcher_s':pg['sum'].to_dict(),'batter_n':bg['size'].to_dict(),'batter_s':bg['sum'].to_dict(),'source_max_season':int(h.season.max()) if len(h) else None}

    # Component-rate lineage: asof_n * rate is a rounded cumulative count. The last
    # historical row stores counts BEFORE that last pitch. We therefore use that
    # pre-last boundary as the baseline. The resulting window contains exactly one
    # boundary pitch + current-season pitches, avoiding any need to inspect other test rows.
    prev=h.loc[h.season.eq(h.season.max())] if len(h) else h
    bank['component_priors']={c:float(pd.to_numeric(prev[c],errors='coerce').mean()) if c in prev and pd.to_numeric(prev[c],errors='coerce').notna().any() else 0.5 for c in P_COMPONENT_RATES+B_COMPONENT_RATES}
    if len(h):
        pidx=h.groupby('pitcher_id',sort=False)['asof_pitcher_n'].idxmax();plast=h.loc[pidx,['pitcher_id','asof_pitcher_n']+P_COMPONENT_RATES].set_index('pitcher_id')
        pbase_n=pd.to_numeric(plast['asof_pitcher_n'],errors='coerce').fillna(0).astype(np.float64);bank['pitcher_component_base_n']=pbase_n.to_dict();bank['pitcher_component_base']={}
        for c in P_COMPONENT_RATES:
            rr=pd.to_numeric(plast[c],errors='coerce');cnt=np.rint(pbase_n*rr);bank['pitcher_component_base'][c]=cnt.to_dict()
        bidx=h.groupby('batter_id',sort=False)['asof_batter_n'].idxmax();blast=h.loc[bidx,['batter_id','asof_batter_n']+B_COMPONENT_RATES].set_index('batter_id')
        bbase_n=pd.to_numeric(blast['asof_batter_n'],errors='coerce').fillna(0).astype(np.float64);bank['batter_component_base_n']=bbase_n.to_dict();bank['batter_component_base']={}
        for c in B_COMPONENT_RATES:
            rr=pd.to_numeric(blast[c],errors='coerce');cnt=np.rint(bbase_n*rr);bank['batter_component_base'][c]=cnt.to_dict()
    else:
        bank['pitcher_component_base_n']={};bank['pitcher_component_base']={c:{} for c in P_COMPONENT_RATES};bank['batter_component_base_n']={};bank['batter_component_base']={c:{} for c in B_COMPONENT_RATES}
    return bank

def transform_global_season(bank,rows):
    r=rows.reset_index(drop=True);z=pd.DataFrame(index=np.arange(len(r)));prior=float(bank['prior'])
    def one(ent,ncol,rcol,prefix,kn=80.0):
        ids=r[ent];hn=ids.map(bank[prefix+'_n']).fillna(0).to_numpy(np.float64);hs=ids.map(bank[prefix+'_s']).fillna(0).to_numpy(np.float64)
        an=np.maximum(np.nan_to_num(_n(r,ncol,0),nan=0),0);ar=_n(r,rcol,prior);ar=np.where(np.isfinite(ar),ar,prior)
        # Exact integer reconstruction: official rates are rounded, but n*rate is always
        # within <0.01 of the underlying integer count in the released train.
        cumulative_success=_rounded_cumulative_count(an,ar)
        sn=np.maximum(an-hn,0.0);ss=np.clip(cumulative_success-hs,0.0,sn)
        raw=np.divide(ss,sn,out=np.full(len(r),prior,np.float64),where=sn>0)
        for k in [40.,80.,160.]:z[f'gseason__{prefix}_rate_k{int(k)}']=((ss+k*prior)/(sn+k)).astype(np.float32)
        z[f'gseason__{prefix}_raw']=raw.astype(np.float32);z[f'gseason__{prefix}_log_n']=np.log1p(sn).astype(np.float32);z[f'gseason__{prefix}_reliability']=(sn/(sn+kn)).astype(np.float32);z[f'gseason__{prefix}_minus_career']=(raw-ar).astype(np.float32)
    one('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','pitcher',80.)
    one('batter_id','asof_batter_n','asof_batter_success_rate','batter',100.)
    z['gseason__pitcher_minus_batter_k80']=(z['gseason__pitcher_rate_k80']-z['gseason__batter_rate_k80']).astype(np.float32)

    # Exact cumulative component counts with one-pitch boundary bridge.
    ids=r['pitcher_id'];an=np.maximum(np.nan_to_num(_n(r,'asof_pitcher_n',0),nan=0),0);base_n=ids.map(bank['pitcher_component_base_n']).fillna(0).to_numpy(np.float64);win_n=np.maximum(an-base_n,0.0)
    comp={}
    for c in P_COMPONENT_RATES:
        pr=float(bank['component_priors'].get(c,0.5));rr=_n(r,c,pr);rr=np.where(np.isfinite(rr),rr,pr);cum=_rounded_cumulative_count(an,rr);base=ids.map(bank['pitcher_component_base'].get(c,{})).fillna(0).to_numpy(np.float64);wc=np.clip(cum-base,0.0,win_n);comp[c]=((wc+60.0*pr)/(win_n+60.0)).astype(np.float32)
    z['gseason__pitcher_component_log_n']=np.log1p(win_n).astype(np.float32);z['gseason__pitcher_component_reliability']=(win_n/(win_n+60.)).astype(np.float32)
    z['gseason__pitcher_strike_rate']=comp['asof_pitcher_strike_rate'];z['gseason__pitcher_ball_rate']=comp['asof_pitcher_ball_rate'];z['gseason__pitcher_strike_ball_margin']=(comp['asof_pitcher_strike_rate']-comp['asof_pitcher_ball_rate']).astype(np.float32)
    z['gseason__pitcher_middle_rate']=comp['asof_pitcher_middle_rate'];z['gseason__pitcher_reverse_rate']=comp['asof_pitcher_reverse_rate']
    fast=np.clip(comp['asof_pitcher_fastball_rate'],1e-6,1);brk=np.clip(comp['asof_pitcher_breaking_rate'],1e-6,1);off=np.clip(comp['asof_pitcher_offspeed_rate'],1e-6,1);sm=np.maximum(fast+brk+off,1e-6);fast=fast/sm;brk=brk/sm;off=off/sm
    z['gseason__pitcher_fastball_rate']=fast.astype(np.float32);z['gseason__pitcher_breaking_rate']=brk.astype(np.float32);z['gseason__pitcher_offspeed_rate']=off.astype(np.float32);z['gseason__pitcher_pitchmix_entropy']=(-(fast*np.log(fast)+brk*np.log(brk)+off*np.log(off))).astype(np.float32)

    bids=r['batter_id'];ban=np.maximum(np.nan_to_num(_n(r,'asof_batter_n',0),nan=0),0);bbase_n=bids.map(bank['batter_component_base_n']).fillna(0).to_numpy(np.float64);bwin=np.maximum(ban-bbase_n,0.0);c='asof_batter_middle_rate';pr=float(bank['component_priors'].get(c,0.5));rr=_n(r,c,pr);rr=np.where(np.isfinite(rr),rr,pr);cum=_rounded_cumulative_count(ban,rr);base=bids.map(bank['batter_component_base'].get(c,{})).fillna(0).to_numpy(np.float64);wc=np.clip(cum-base,0.0,bwin);bmid=((wc+80.*pr)/(bwin+80.)).astype(np.float32)
    z['gseason__batter_middle_rate']=bmid;z['gseason__batter_middle_minus_pitcher_middle']=(bmid-z['gseason__pitcher_middle_rate']).astype(np.float32)
    return z

def global_season_features_for_training(frame):
    pieces=[]
    for year in sorted(frame.season.astype(int).unique()):
        idx=frame.index[frame.season.astype(int).eq(year)];bank=build_global_season_bank(frame.loc[frame.season.astype(int)<year],year);part=transform_global_season(bank,frame.loc[idx]);part.index=idx;pieces.append(part)
    return pd.concat(pieces).sort_index().reindex(frame.index).reset_index(drop=True)


print('Exact-ASOF row-state functions loaded')


def add_eng(frame):
    x=frame.copy()
    x['eng_recent35']=(.5*pd.to_numeric(x.asof_pitcher_prev1_game_success_rate,errors='coerce')+.3*pd.to_numeric(x.asof_pitcher_prev3_game_success_rate,errors='coerce')+.2*pd.to_numeric(x.asof_pitcher_prev5_game_success_rate,errors='coerce'))
    x['eng_recent_delta']=x.eng_recent35-pd.to_numeric(x.asof_pitcher_success_rate,errors='coerce')
    x['eng_sb']=pd.to_numeric(x.asof_pitcher_strike_rate,errors='coerce')-pd.to_numeric(x.asof_pitcher_ball_rate,errors='coerce')
    x['eng_same']=(x.pitcher_hand.astype(str)==x.batter_hand.astype(str)).astype(np.float32)
    x['eng_close']=(pd.to_numeric(x.score_diff_pitcher_team,errors='coerce').abs()<=1).astype(np.float32)
    x['eng_late']=(pd.to_numeric(x.inning,errors='coerce')>=7).astype(np.float32)
    return x

def encode_with_maps(frame,maps):
    out=pd.DataFrame(index=np.arange(len(frame)))
    for c,mp in maps.items():
        out['code_'+c]=frame[c].astype('string').fillna('<NA>').astype(str).map(mp).fillna(-1).astype(np.float32)
    return out

def transform_base(test):
    rawnum=META['rawnum']
    selfx=build_self_state(test)
    gx=transform_global_season(META['target_global_bank'],test)
    B0=pd.concat([test[rawnum].apply(pd.to_numeric,errors='coerce').reset_index(drop=True),selfx.reset_index(drop=True),gx.reset_index(drop=True)],axis=1)

    cb=encode_with_maps(test,META['exact_cat_maps'])
    B_exact=pd.concat([B0,cb],axis=1)
    B_exact=B_exact.reindex(columns=META['exact_feature_cols'])
    m1=joblib.load(MODEL/'base_exact_lgb.joblib')
    p1=m1.predict_proba(B_exact)[:,1]

    B=B0.reindex(columns=META['ridge_feature_cols']).replace([np.inf,-np.inf],np.nan)
    med=pd.Series(META['ridge_med'])
    B=B.fillna(med).to_numpy(np.float32)
    mu=np.asarray(META['ridge_mu'],dtype=np.float32);sd=np.asarray(META['ridge_sd'],dtype=np.float32)
    B=np.clip((B-mu)/sd,-8,8)
    m2=joblib.load(MODEL/'base_ridge.joblib')
    p2=np.clip(m2.predict(B),.001,.999)

    t=add_eng(test)
    Bnum=t[META['trend_num_cols']].apply(pd.to_numeric,errors='coerce').reset_index(drop=True)
    cb=encode_with_maps(t,META['trend_cat_maps'])
    Bt=pd.concat([Bnum,cb],axis=1).astype(np.float32).reindex(columns=META['trend_feature_cols'])
    m3=joblib.load(MODEL/'base_trend_lgb.joblib')
    prior=META['trend_prior']
    bt=np.array([float(prior[d]) for d in test.domain3.to_numpy()],dtype=np.float64)
    p3=np.clip(bt+m3.predict(Bt),.001,.999)
    return np.clip((np.asarray(p1,float)+np.asarray(p2,float)+np.asarray(p3,float))/3.0,0,1)

def transform_adapter(rows,base_pred):
    prep=META['adapter_prep'];r=add_eng(rows.reset_index(drop=True));A=r[prep['cols']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    for c in prep['cols']:
        A[c]=A[c].fillna(float(prep['med'].get(c,0.))).astype(np.float32)
    for c,mp in prep['maps'].items():
        A['code_'+c]=r[c].astype('string').fillna('<NA>').astype(str).map(mp).fillna(-1).astype(np.float32)
    A['base_pred']=np.asarray(base_pred,dtype=np.float32)
    A=A.reindex(columns=META['adapter_base_feature_cols'])
    if not np.isfinite(A.to_numpy(float)).all():
        raise RuntimeError('non-finite adapter input')
    return A


# ---------------- V15 selected challenger blocks ----------------
def _ctx_frame_v15(frame):
    x=frame.copy()
    x['v15_count']=x['balls_before'].astype(str)+'_'+x['strikes_before'].astype(str)
    x['v15_inning_phase']=pd.cut(pd.to_numeric(x['inning'],errors='coerce'),[-np.inf,3,6,9,np.inf],labels=['early','middle','late','extra']).astype(str)
    x['v15_pressure']=np.where((x['balls_before']==3)&(x['strikes_before']==2),'full',np.where(x['balls_before']==3,'three_ball',np.where(x['strikes_before']==2,'two_strike','normal')))
    x['v15_runners']=np.where(pd.to_numeric(x['num_runners_on'],errors='coerce').fillna(0)>0,'on','empty')
    x['v15_li_bucket']=pd.cut(pd.to_numeric(x['li'],errors='coerce'),[-np.inf,.7,1.5,np.inf],labels=['low','mid','high']).astype(str)
    return x

def _lookup_v15(table,rows,keys,field,default=0.0):
    if len(keys)==1:
        s=table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)
    idx=pd.MultiIndex.from_frame(rows[keys])
    s=table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)

def transform_pitcher_conditional(rows):
    r=_ctx_frame_v15(rows.reset_index(drop=True));parts=[]
    for spec in META['pitcher_conditional_specs']:
        t=spec['table'];ctx=spec['ctx'];prefix=spec['prefix'];z=pd.DataFrame(index=np.arange(len(r)))
        keys=['pitcher_id',ctx]
        for f in ('rate','reliability','delta'):
            z[prefix+'__'+f]=_lookup_v15(t,r,keys,f,0.0)
        parts.append(z)
    out=pd.concat(parts,axis=1).astype(np.float32)
    return out.reindex(columns=META['pitcher_conditional_cols'])

def transform_exact_adapter_block(rows):
    sx=build_self_state(rows).add_prefix('exact__')
    gx=transform_global_season(META['target_global_bank'],rows).add_prefix('exact__')
    out=pd.concat([sx.reset_index(drop=True),gx.reset_index(drop=True)],axis=1)
    out=out.replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
    return out.reindex(columns=META['exact_adapter_cols'])

def transform_v15_adapter(rows,base_pred):
    base=transform_adapter(rows,base_pred)
    pc=transform_pitcher_conditional(rows)
    ex=transform_exact_adapter_block(rows)
    X=pd.concat([base.reset_index(drop=True),pc.reset_index(drop=True),ex.reset_index(drop=True)],axis=1)
    X=X.reindex(columns=META['adapter_feature_cols'])
    if X.isna().any().any() or not np.isfinite(X.to_numpy(float)).all():
        raise RuntimeError('non-finite V15 adapter input')
    return X


# ---------------- V18 strict-temporal stable conditional correction ----------------
def _v18_pressure_frame(rows):
    r=rows.copy()
    balls=pd.to_numeric(r['balls_before'],errors='coerce').fillna(0).astype(int)
    strikes=pd.to_numeric(r['strikes_before'],errors='coerce').fillna(0).astype(int)
    r['v18_pressure']=np.where((balls==3)&(strikes==2),'full',np.where(balls==3,'3ball',np.where(strikes==2,'2strike','normal')))
    r['v18_pressure_bhand']=r['v18_pressure'].astype(str)+'|'+r['batter_hand'].astype(str)
    return r

def transform_v18_stable(rows):
    r=_v18_pressure_frame(rows.reset_index(drop=True));parts=[]
    for spec in META['v18_stable_specs']:
        t=spec['table'];ctx=spec['ctx'];prefix=spec['prefix'];z=pd.DataFrame(index=np.arange(len(r)))
        de=_lookup_v15(t,r,['pitcher_id',ctx],'delta',0.0).astype(np.float32)
        re=_lookup_v15(t,r,['pitcher_id',ctx],'reliability',0.0).astype(np.float32)
        z[prefix+'__delta']=de
        z[prefix+'__weighted_delta']=(de*re).astype(np.float32)
        z[prefix+'__reliability']=re
        parts.append(z)
    z=pd.concat(parts,axis=1)
    balls=pd.to_numeric(r['balls_before'],errors='coerce').fillna(0).astype(int).to_numpy()
    strikes=pd.to_numeric(r['strikes_before'],errors='coerce').fillna(0).astype(int).to_numpy()
    for b in range(4): z[f'ctx__ball_{b}']=(balls==b).astype(np.float32)
    for s in range(3): z[f'ctx__strike_{s}']=(strikes==s).astype(np.float32)
    for b in range(4):
        for s in range(3): z[f'ctx__count_{b}_{s}']=((balls==b)&(strikes==s)).astype(np.float32)
    z=z.reindex(columns=META['v18_stable_cols']).replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
    if not np.isfinite(z.to_numpy(float)).all(): raise RuntimeError('non-finite V18 stable features')
    return z

def main():
    test=pd.read_csv(DATA/'test.csv',encoding='utf-8-sig',low_memory=False)
    if TARGET in test.columns: raise RuntimeError('test.csv unexpectedly contains target')
    required=['row_id','season','pitcher_id','batter_id','game_type','pitcher_team_id','batter_team_id']
    missing=[c for c in required if c not in test.columns]
    if missing: raise RuntimeError(f'missing test columns: {missing}')
    test['season']=pd.to_numeric(test['season'],errors='raise').astype(int)
    test['game_type']=test['game_type'].astype('string').fillna('<NA>').astype(str)
    expected=int(META['expected_year'])
    if bool((test.season!=expected).any()): raise RuntimeError(f'evaluation season differs from expected {expected}')
    if bool((~test.game_type.isin(['R','F'])).any()): raise RuntimeError('unexpected game_type')
    anchor=int(META['anchor_team_id'])
    test['domain3']=np.where(test.game_type.eq('F'),'F',np.where((test.pitcher_team_id==anchor)|(test.batter_team_id==anchor),'R_ANCHOR','R_CORE'))

    base=transform_base(test)
    X=transform_v15_adapter(test,base)
    corr=np.zeros(len(test),dtype=np.float64)
    seeds=META['adapter_seeds']
    for seed in seeds:
        c=np.zeros(len(test),dtype=np.float64)
        for d in ['R_CORE','R_ANCHOR','F']:
            mask=test.domain3.eq(d).to_numpy()
            if not mask.any(): continue
            m=joblib.load(MODEL/f'adapter_seed{int(seed)}_{d}.joblib')
            c[mask]=m.predict(X.loc[mask])
        corr+=c/len(seeds)
    final=np.clip(base+float(META['adapter_shrink'])*corr,0,1)
    if META.get('v18_stable_enabled',False):
        Xs=transform_v18_stable(test)
        sm=joblib.load(MODEL/META['v18_stable_model_file'])
        arr=Xs.to_numpy(np.float32);mu=np.asarray(META['v18_stable_mu'],dtype=np.float32);sd=np.asarray(META['v18_stable_sd'],dtype=np.float32);arr=np.clip((arr-mu)/sd,-8,8)
        sc=np.asarray(sm.predict(arr),dtype=np.float64)
        core=test.domain3.eq(META.get('v18_route_domain','R_CORE')).to_numpy()
        final[core]=np.clip(final[core]+float(META['v18_stable_gamma'])*sc[core],0,1)
    if len(final)!=len(test) or not np.isfinite(final).all() or (final<0).any() or (final>1).any():
        raise RuntimeError('invalid prediction vector')
    out=pd.DataFrame({'row_id':test.row_id.to_numpy(),'control_success':final})
    out.to_csv(OUT/'submission.csv',index=False)
    print('V18_TEMPORAL_STABLE_INFERENCE_OK',len(out),float(final.min()),float(final.max()),float(final.mean()))

if __name__=='__main__':
    main()

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
rng = np.random.default_rng(0)

df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv', low_memory=False)
df['t']=pd.to_datetime(df['created_date'],errors='coerce'); df=df.sort_values('t').reset_index(drop=True)
df['lat']=pd.to_numeric(df['latitude'],errors='coerce'); df['lon']=pd.to_numeric(df['longitude'],errors='coerce')
df['hour']=df['t'].dt.hour.fillna(0).astype(int); df['weekday']=df['t'].dt.weekday.fillna(0).astype(int)
y=df['requires_road_closure'].astype(int).values; N=len(df); base=y.mean()
cats=['event_type','event_cause','veh_type','corridor','police_station','zone']
nums=['lat','lon','hour','weekday']
for c in cats: df[c]=df[c].astype(str).fillna('NA')
codes=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit_transform(df[cats]).astype(int)
X=pd.DataFrame(codes,columns=cats)
for c in nums: X[c]=df[c].astype(float).values
cat_idx=list(range(len(cats)))
planned=(df['event_type']=='planned').values

def oof_for(make, cv, fitkw=lambda tr:{}):
    oof=np.full(N,np.nan); per=[]
    for tr,te in cv:
        m=make(); m.fit(X.iloc[tr],y[tr],**fitkw(tr)); p=m.predict_proba(X.iloc[te])[:,1]; oof[te]=p
        per.append((roc_auc_score(y[te],p),average_precision_score(y[te],p)))
    return oof,np.array(per)
def pooled(oof):
    c=~np.isnan(oof); return roc_auc_score(y[c],oof[c]),average_precision_score(y[c],oof[c])
def bdelta(yv,pa,pb,fn,n=2000):
    s=[];k=len(yv)
    for _ in range(n):
        b=rng.integers(0,k,k)
        if yv[b].sum() in (0,k):continue
        s.append(fn(yv[b],pa[b])-fn(yv[b],pb[b]))
    s=np.array(s);return s.mean(),np.percentile(s,[2.5,97.5]),2*min((s<=0).mean(),(s>=0).mean())

rcv=list(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y))
tcv=list(TimeSeriesSplit(5).split(X))
mk_cat=lambda:CatBoostClassifier(iterations=400,depth=6,learning_rate=0.05,verbose=0,random_seed=0,cat_features=cat_idx,thread_count=-1)
mk_lgb=lambda:LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,random_state=0,n_jobs=-1,verbose=-1)
mk_xgb=lambda:XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,tree_method='hist',random_state=0,n_jobs=-1,eval_metric='logloss')
mk_hgb=lambda:HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,max_depth=6,categorical_features=cat_idx,random_state=0)
mk_rf =lambda:RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=0,class_weight='balanced')
mk_et =lambda:ExtraTreesClassifier(n_estimators=600,n_jobs=-1,random_state=0,class_weight='balanced')
lgb_kw=lambda tr:{'categorical_feature':cat_idx}

def lr_oof(cv):
    oof=np.full(N,np.nan);per=[]
    ct=ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore',min_frequency=10),cats),('n',StandardScaler(),nums)])
    for tr,te in cv:
        pipe=Pipeline([('ct',ct),('lr',LogisticRegression(max_iter=2000,class_weight='balanced'))])
        pipe.fit(X.iloc[tr],y[tr]);p=pipe.predict_proba(X.iloc[te])[:,1];oof[te]=p
        per.append((roc_auc_score(y[te],p),average_precision_score(y[te],p)))
    return oof,np.array(per)

models=[('CatBoost',mk_cat,{}),('LightGBM',mk_lgb,{'fitkw':lgb_kw}),('XGBoost',mk_xgb,{}),
        ('HistGBM',mk_hgb,{}),('RandomForest',mk_rf,{}),('ExtraTrees',mk_et,{})]
oofs={'random':{},'temporal':{}}; tab={}
for name,make,extra in models:
    for tag,cv in [('random',rcv),('temporal',tcv)]:
        oof,per=oof_for(make,cv,extra.get('fitkw',lambda tr:{})); a,p=pooled(oof)
        oofs[tag][name]=oof; tab[(name,tag)]=(a,p,per[:,0].std())
for tag,cv in [('random',rcv),('temporal',tcv)]:
    oof,per=lr_oof(cv);a,p=pooled(oof);oofs[tag]['Logistic']=oof;tab[('Logistic',tag)]=(a,p,per[:,0].std())

def regime_oof(cv):
    oof=np.full(N,np.nan)
    for tr,te in cv:
        for reg in [True,False]:
            trr=tr[planned[tr]==reg]; tee=te[planned[te]==reg]
            if len(tee)==0: continue
            if len(trr)<20: oof[tee]=y[tr].mean(); continue
            m=mk_cat(); m.fit(X.iloc[trr],y[trr]); oof[tee]=m.predict_proba(X.iloc[tee])[:,1]
    return oof
for tag,cv in [('random',rcv),('temporal',tcv)]:
    oof=regime_oof(cv);a,p=pooled(oof);oofs[tag]['Regime-split']=oof;tab[('Regime-split',tag)]=(a,p,np.nan)

print(f'N={N} base={base:.4f} pos={y.sum()}\n')
print('MODEL TOURNAMENT (pooled OOF AUC / PR-AUC)')
print(f'{"model":14s} | {"RANDOM-CV":>16s} | {"TEMPORAL-CV":>16s}')
for n in ['CatBoost','LightGBM','XGBoost','HistGBM','RandomForest','ExtraTrees','Logistic','Regime-split']:
    ra,rp,_=tab[(n,'random')]; ta,tp,_=tab[(n,'temporal')]
    print(f'{n:14s} | {ra:.3f} / {rp:.3f}  | {ta:.3f} / {tp:.3f}')

fam=['CatBoost','LightGBM','XGBoost','HistGBM','ExtraTrees']
print('\nENSEMBLES (family: '+', '.join(fam)+')')
for tag,cv in [('random',rcv),('temporal',tcv)]:
    M=np.column_stack([oofs[tag][f] for f in fam]); cov=~np.isnan(M).any(1); yv=y[cov]
    ranks=np.column_stack([pd.Series(oofs[tag][f][cov]).rank().values for f in fam]); blend=ranks.mean(1)
    stk=cross_val_predict(LogisticRegression(max_iter=1000,class_weight='balanced'),M[cov],yv,
                          cv=StratifiedKFold(5,shuffle=True,random_state=1),method='predict_proba')[:,1]
    best=max(fam,key=lambda f:roc_auc_score(yv,oofs[tag][f][cov]))
    ba=roc_auc_score(yv,oofs[tag][best][cov]); bp=average_precision_score(yv,oofs[tag][best][cov])
    print(f' [{tag}] best={best} {ba:.3f}/{bp:.3f} | rank-blend {roc_auc_score(yv,blend):.3f}/{average_precision_score(yv,blend):.3f} | logit-stack {roc_auc_score(yv,stk):.3f}/{average_precision_score(yv,stk):.3f}')
    if tag=='temporal':
        d,ci,pv=bdelta(yv,blend/len(blend),oofs[tag][best][cov],roc_auc_score)
        d2,ci2,pv2=bdelta(yv,stk,oofs[tag][best][cov],roc_auc_score)
        print(f'   sig: rank-blend vs best ΔAUC{d:+.4f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] p={pv:.3f} | logit-stack vs best ΔAUC{d2:+.4f} CI[{ci2[0]:+.3f},{ci2[1]:+.3f}] p={pv2:.3f}')
np.save('oof_temporal_cat.npy', oofs['temporal']['CatBoost'])

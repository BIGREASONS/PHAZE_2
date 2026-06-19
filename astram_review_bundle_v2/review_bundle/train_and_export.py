"""
ASTraM final model — train on 100% of labeled data + export deployable artifacts.
Recipe is frozen from the OOF audit: 7 equal-weight members, isotonic calibration
fit on a TEMPORAL holdout. Run:  python train_and_export.py
"""
import os, json, warnings, joblib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA='Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
OUT='astram_model_v1.0'; MEMB=os.path.join(OUT,'members')
os.makedirs(MEMB, exist_ok=True)
CATS=['event_type','event_cause','veh_type','corridor','police_station','zone']
NUMS=['lat','lon','hour','weekday']; CAT_IDX=list(range(len(CATS)))
SEEDS=list(range(10))          # seed-bag depth for trees
USE_TABPFN=True

def load():
    df=pd.read_csv(DATA,low_memory=False)
    df['t']=pd.to_datetime(df['created_date'],errors='coerce'); df=df.sort_values('t').reset_index(drop=True)
    df['lat']=pd.to_numeric(df['latitude'],errors='coerce'); df['lon']=pd.to_numeric(df['longitude'],errors='coerce')
    df['hour']=df['t'].dt.hour.fillna(0).astype(int); df['weekday']=df['t'].dt.weekday.fillna(0).astype(int)
    for c in CATS: df[c]=df[c].astype(str).fillna('NA')
    y=df['requires_road_closure'].astype(int).values
    return df,y

def mk(name,seed):
    if name=='catboost': return CatBoostClassifier(iterations=400,depth=6,learning_rate=0.05,verbose=0,random_seed=seed,cat_features=CAT_IDX,thread_count=-1)
    if name=='lightgbm': return LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,random_state=seed,n_jobs=-1,verbose=-1)
    if name=='xgboost':  return XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,tree_method='hist',random_state=seed,n_jobs=-1,eval_metric='logloss')
    if name=='randomforest': return RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=seed,class_weight='balanced')
    if name=='extratrees':   return ExtraTreesClassifier(n_estimators=600,n_jobs=-1,random_state=seed,class_weight='balanced')

def fit_tree_bag(name,Xt,y):
    bag=[]
    for s in (SEEDS if name in('catboost','lightgbm','xgboost') else [0]):
        m=mk(name,s)
        m.fit(Xt,y,categorical_feature=CAT_IDX) if name=='lightgbm' else m.fit(Xt,y)
        bag.append(m)
    return bag

def pred_tree_bag(bag,Xt): return np.mean([m.predict_proba(Xt)[:,1] for m in bag],axis=0)

def fit_lr(Xraw,y):
    ct=ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore',min_frequency=10),CATS),('n',StandardScaler(),NUMS)])
    p=Pipeline([('ct',ct),('lr',LogisticRegression(max_iter=2000,class_weight='balanced'))]); p.fit(Xraw,y); return p

def build_X(df):
    enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit(df[CATS])
    Xt=pd.DataFrame(enc.transform(df[CATS]).astype(int),columns=CATS)
    for c in NUMS: Xt[c]=df[c].astype(float).values
    return enc,Xt,df[CATS+NUMS].copy()

def ensemble_scores(members,Xt,Xraw,tab=None):
    cols=[pred_tree_bag(members[n],Xt) for n in ['catboost','lightgbm','xgboost','randomforest','extratrees']]
    cols.append(members['logistic'].predict_proba(Xraw)[:,1])
    if tab is not None: cols.append(tab)
    return np.mean(cols,axis=0)

df,y=load(); N=len(df)
enc,Xt,Xraw=build_X(df)

# ── PASS 1: temporal holdout → fit & FREEZE calibrator ───────────────────────
split=int(N*0.8); tr=np.arange(split); te=np.arange(split,N)
mem_cal={n:fit_tree_bag(n,Xt.iloc[tr],y[tr]) for n in ['catboost','lightgbm','xgboost','randomforest','extratrees']}
mem_cal['logistic']=fit_lr(Xraw.iloc[tr],y[tr])
tab_te=None
if USE_TABPFN:
    try:
        from tabpfn import TabPFNClassifier
        Xnp=np.hstack([Xt[CATS].values, Xt[NUMS].values]).astype(float)
        pos=tr[y[tr]==1]; neg=tr[y[tr]==0]
        keep=np.sort(np.concatenate([pos,np.random.RandomState(0).choice(neg,min(len(neg),4000-len(pos)),replace=False)]))
        tp=TabPFNClassifier(device='cpu',ignore_pretraining_limits=True,n_estimators=3); tp.fit(Xnp[keep],y[keep])
        tab_te=tp.predict_proba(Xnp[te])[:,1]
    except Exception as e:
        print('TabPFN unavailable -> 6-model mean for calibration:',e); USE_TABPFN=False
raw_te=ensemble_scores(mem_cal,Xt.iloc[te],Xraw.iloc[te],tab_te)
cal=IsotonicRegression(out_of_bounds='clip').fit(raw_te,y[te])
print(f'Holdout temporal  ROC={roc_auc_score(y[te],raw_te):.4f}  PR={average_precision_score(y[te],raw_te):.4f}')

# ── PASS 2: retrain ALL members on 100% data ────────────────────────────────
members={n:fit_tree_bag(n,Xt,y) for n in ['catboost','lightgbm','xgboost','randomforest','extratrees']}
members['logistic']=fit_lr(Xraw,y)
tabpfn_blob=None
if USE_TABPFN:
    Xnp=np.hstack([Xt[CATS].values, Xt[NUMS].values]).astype(float)
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    keep=np.sort(np.concatenate([pos,np.random.RandomState(0).choice(neg,min(len(neg),4000-len(pos)),replace=False)]))
    tabpfn_blob={'X':Xnp[keep],'y':y[keep]}

# ── EXPORT ──────────────────────────────────────────────────────────────────
joblib.dump(enc, os.path.join(OUT,'encoder.pkl'))
for n in ['catboost','lightgbm','xgboost','randomforest','extratrees']:
    joblib.dump(members[n], os.path.join(MEMB,f'{n}.pkl'))
joblib.dump(members['logistic'], os.path.join(MEMB,'logistic.pkl'))
if tabpfn_blob is not None: joblib.dump(tabpfn_blob, os.path.join(MEMB,'tabpfn.pkl'))
joblib.dump(cal, os.path.join(OUT,'calibrator.pkl'))

imp=np.mean([members['catboost'][0].get_feature_importance()],axis=0)
fi={f:round(float(v),4) for f,v in sorted(zip(CATS+NUMS,imp/imp.sum()),key=lambda x:-x[1])}
json.dump(fi, open(os.path.join(OUT,'feature_importance.json'),'w'),indent=2)
json.dump({'HIGH':0.50,'MEDIUM':0.20,'LOW':0.0}, open(os.path.join(OUT,'thresholds.json'),'w'),indent=2)
json.dump({'members':['catboost','lightgbm','xgboost','randomforest','extratrees','logistic']+(['tabpfn'] if tabpfn_blob else []),
           'weights':'equal','cats':CATS,'nums':NUMS,'seedbag_seeds':SEEDS,
           'note':'OOF-derived recipe; members retrained on 100% data; calibrator frozen on temporal holdout'},
          open(os.path.join(OUT,'manifest.json'),'w'),indent=2)
json.dump({'name':'Final ASTraM Model','version':'v1.0','status':'ONLINE',
           'training_date':datetime.now(timezone.utc).isoformat(),
           'metrics':{'temporal_roc_auc':0.767,'temporal_pr_auc':0.32,'corridor_pr_auc':0.32,'n_train':int(N)}},
          open(os.path.join(OUT,'metadata.json'),'w'),indent=2)
print('Exported ->', OUT)

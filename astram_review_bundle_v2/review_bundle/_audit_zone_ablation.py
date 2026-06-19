import numpy as np, pandas as pd, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA='Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
SEED=0
df=pd.read_csv(DATA,low_memory=False)
df['t']=pd.to_datetime(df['created_date'],errors='coerce'); df=df.sort_values('t').reset_index(drop=True)
df['lat']=pd.to_numeric(df['latitude'],errors='coerce'); df['lon']=pd.to_numeric(df['longitude'],errors='coerce')
df['hour']=df['t'].dt.hour.fillna(0).astype(int); df['weekday']=df['t'].dt.weekday.fillna(0).astype(int)
y=df['requires_road_closure'].astype(int).values; N=len(df)
corridor=df['corridor'].astype(str).fillna('NA').values
nums=['lat','lon','hour','weekday']

SCHEMES={'temporal':list(TimeSeriesSplit(5).split(df)),
         'corridor':list(GroupKFold(5).split(df,y,groups=corridor))}

def build_X(cats):
    for c in cats: df[c]=df[c].astype(str).fillna('NA')
    codes=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit_transform(df[cats]).astype(int)
    Xt=pd.DataFrame(codes,columns=cats)
    for c in nums: Xt[c]=df[c].astype(float).values
    return Xt, df[cats+nums].copy(), list(range(len(cats)))

def mk(name,cat_idx):
    if name=='CatBoost': return CatBoostClassifier(iterations=400,depth=6,learning_rate=0.05,verbose=0,random_seed=SEED,cat_features=cat_idx,thread_count=-1)
    if name=='LightGBM': return LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,random_state=SEED,n_jobs=-1,verbose=-1)
    if name=='XGBoost': return XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,tree_method='hist',random_state=SEED,n_jobs=-1,eval_metric='logloss')
    if name=='RandomForest': return RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=SEED,class_weight='balanced')
    if name=='ExtraTrees': return ExtraTreesClassifier(n_estimators=600,n_jobs=-1,random_state=SEED,class_weight='balanced')
TREES=['CatBoost','LightGBM','XGBoost','RandomForest','ExtraTrees']

def tree_oof(name,scheme,Xt,cat_idx):
    oof=np.full(N,np.nan)
    for tr,te in SCHEMES[scheme]:
        m=mk(name,cat_idx)
        if name=='LightGBM': m.fit(Xt.iloc[tr],y[tr],categorical_feature=cat_idx)
        else: m.fit(Xt.iloc[tr],y[tr])
        oof[te]=m.predict_proba(Xt.iloc[te])[:,1]
    return oof

def lr_oof(scheme,Xraw,cats):
    oof=np.full(N,np.nan)
    ct=ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore',min_frequency=10),cats),('n',StandardScaler(),nums)])
    for tr,te in SCHEMES[scheme]:
        pipe=Pipeline([('ct',ct),('lr',LogisticRegression(max_iter=2000,class_weight='balanced'))])
        pipe.fit(Xraw.iloc[tr],y[tr]); oof[te]=pipe.predict_proba(Xraw.iloc[te])[:,1]
    return oof

CONFIGS={
 'WITH_zone':['event_type','event_cause','veh_type','corridor','police_station','zone'],
 'NO_zone':['event_type','event_cause','veh_type','corridor','police_station'],
 'NO_zone_latlon':['event_type','event_cause','veh_type','corridor','police_station'],  # also drop lat/lon nums
}

results={}
for cfg,cats in CONFIGS.items():
    drop_latlon = (cfg=='NO_zone_latlon')
    nums=['hour','weekday'] if drop_latlon else ['lat','lon','hour','weekday']
    Xt,Xraw,cat_idx=build_X(cats)
    for scheme in SCHEMES:
        oofs={}
        t0=time.time()
        for nm in TREES: oofs[nm]=tree_oof(nm,scheme,Xt,cat_idx)
        oofs['Logistic']=lr_oof(scheme,Xraw,cats)
        M=np.column_stack([oofs[k] for k in oofs]); mask=~np.isnan(M).any(1); idx=np.where(mask)[0]; yv=y[idx]
        per={}
        for k in oofs:
            per[k]=[round(roc_auc_score(yv,oofs[k][idx]),4),round(average_precision_score(yv,oofs[k][idx]),4)]
        meanb=np.mean([oofs[k][idx] for k in oofs],axis=0)
        per['MEAN6']=[round(roc_auc_score(yv,meanb),4),round(average_precision_score(yv,meanb),4)]
        results[f'{cfg}|{scheme}']={'n':len(idx),'per':per,'sec':round(time.time()-t0,1)}
        print(f'[{cfg:14s} {scheme:8s}] n={len(idx)} MEAN6 ROC={per["MEAN6"][0]} PR={per["MEAN6"][1]} CatBoost PR={per["CatBoost"][1]} ({results[f"{cfg}|{scheme}"]["sec"]}s)',flush=True)

json.dump(results,open('_zone_ablation_results.json','w'),indent=2)

print('\n================ SUMMARY: MEAN-of-6 trainable models ================')
print(f'{"scheme":9s} | {"WITH zone":>16s} | {"NO zone":>16s} | {"NO zone+latlon":>16s}')
print(f'{"":9s} | {"ROC":>7s} {"PR":>8s} | {"ROC":>7s} {"PR":>8s} | {"ROC":>7s} {"PR":>8s}')
for scheme in SCHEMES:
    w=results[f'WITH_zone|{scheme}']['per']['MEAN6']; n=results[f'NO_zone|{scheme}']['per']['MEAN6']; nl=results[f'NO_zone_latlon|{scheme}']['per']['MEAN6']
    print(f'{scheme:9s} | {w[0]:7.4f} {w[1]:8.4f} | {n[0]:7.4f} {n[1]:8.4f} | {nl[0]:7.4f} {nl[1]:8.4f}')
print('\n================ SUMMARY: CatBoost (key single) ================')
for scheme in SCHEMES:
    w=results[f'WITH_zone|{scheme}']['per']['CatBoost']; n=results[f'NO_zone|{scheme}']['per']['CatBoost']; nl=results[f'NO_zone_latlon|{scheme}']['per']['CatBoost']
    print(f'{scheme:9s} | WITH {w} | NO_zone {n} | NO_zone_latlon {nl}')

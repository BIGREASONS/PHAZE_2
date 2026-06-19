import sys, io, warnings, time, os
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, torch
torch.set_num_threads(16)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from tabpfn import TabPFNClassifier

df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv', low_memory=False)
df['t']=pd.to_datetime(df['created_date'],errors='coerce'); df=df.sort_values('t').reset_index(drop=True)
df['lat']=pd.to_numeric(df['latitude'],errors='coerce'); df['lon']=pd.to_numeric(df['longitude'],errors='coerce')
df['hour']=df['t'].dt.hour.fillna(0).astype(int); df['weekday']=df['t'].dt.weekday.fillna(0).astype(int)
y=df['requires_road_closure'].astype(int).values; N=len(df)
cats=['event_type','event_cause','veh_type','corridor','police_station','zone']; nums=['lat','lon','hour','weekday']
for c in cats: df[c]=df[c].astype(str).fillna('NA')
codes=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit_transform(df[cats]).astype(int)
X=np.hstack([codes, df[nums].values.astype(float)]); cat_idx=list(range(len(cats)))

def run(cv,tag):
    oof=np.full(N,np.nan); per=[]; t0=time.time()
    for i,(tr,te) in enumerate(cv):
        if len(tr)>4000:   # cap CPU context; keep class balance
            pos=tr[y[tr]==1]; neg=tr[y[tr]==0]
            negk=np.random.RandomState(i).choice(neg, 4000-len(pos), replace=False)
            tr=np.sort(np.concatenate([pos,negk]))
        clf=TabPFNClassifier(device='cpu', ignore_pretraining_limits=True, n_estimators=3)
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]; oof[te]=p
        per.append((roc_auc_score(y[te],p),average_precision_score(y[te],p)))
        print(f'  [{tag}] fold{i} auc={per[-1][0]:.3f} ({time.time()-t0:.0f}s)',flush=True)
    c=~np.isnan(oof); per=np.array(per)
    print(f'[{tag}] TabPFN pooled AUC={roc_auc_score(y[c],oof[c]):.3f} PR={average_precision_score(y[c],oof[c]):.3f} (foldstd {per[:,0].std():.3f})')
    np.save(f'oof_tabpfn_{tag}.npy', oof); return oof

print(f'N={N} base={y.mean():.4f}',flush=True)
run(list(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)),'random')
run(list(TimeSeriesSplit(5).split(X)),'temporal')

# stack: does TabPFN add to the CatBoost OOF?
try:
    catoof=np.load('oof_temporal_cat.npy'); toof=np.load('oof_tabpfn_temporal.npy')
    c=(~np.isnan(catoof))&(~np.isnan(toof)); yv=y[c]
    blend=(pd.Series(catoof[c]).rank().values+pd.Series(toof[c]).rank().values)/2
    print(f'\n[temporal] CatBoost {roc_auc_score(yv,catoof[c]):.3f} | TabPFN {roc_auc_score(yv,toof[c]):.3f} | Cat+TabPFN blend {roc_auc_score(yv,blend):.3f}')
except Exception as e: print('blend skipped',e)

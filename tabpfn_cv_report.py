import sys, io, os, time, warnings
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, torch
torch.set_num_threads(16)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from tabpfn import TabPFNClassifier

DATA='Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df=pd.read_csv(DATA,low_memory=False)
df['t']=pd.to_datetime(df['created_date'],errors='coerce'); df=df.sort_values('t').reset_index(drop=True)
df['lat']=pd.to_numeric(df['latitude'],errors='coerce'); df['lon']=pd.to_numeric(df['longitude'],errors='coerce')
df['hour']=df['t'].dt.hour.fillna(0).astype(int); df['weekday']=df['t'].dt.weekday.fillna(0).astype(int)
y=df['requires_road_closure'].astype(int).values; N=len(df); base=y.mean()
cats=['event_type','event_cause','veh_type','corridor','police_station','zone']; nums=['lat','lon','hour','weekday']
for c in cats: df[c]=df[c].astype(str).fillna('NA')
codes=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit_transform(df[cats]).astype(int)
X=np.hstack([codes, df[nums].values.astype(float)])
groups=df['corridor'].values

def run(cv,tag):
    oof=np.full(N,np.nan); per=[]; t0=time.time()
    for i,(tr,te) in enumerate(cv):
        if len(tr)>4000:                      # cap CPU context, preserve all positives
            pos=tr[y[tr]==1]; neg=tr[y[tr]==0]
            negk=np.random.RandomState(i).choice(neg, max(1,4000-len(pos)), replace=False)
            tr=np.sort(np.concatenate([pos,negk]))
        clf=TabPFNClassifier(device='cpu', ignore_pretraining_limits=True, n_estimators=3)
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]; oof[te]=p
        per.append((roc_auc_score(y[te],p),average_precision_score(y[te],p)))
        print(f'  [{tag}] fold{i} auc={per[-1][0]:.3f} pr={per[-1][1]:.3f} ({time.time()-t0:.0f}s)',flush=True)
    c=~np.isnan(oof); per=np.array(per)
    A=roc_auc_score(y[c],oof[c]); P=average_precision_score(y[c],oof[c])
    print(f'[{tag}] TabPFN pooled AUC={A:.4f} PR={P:.4f}  foldmeanAUC={per[:,0].mean():.3f}±{per[:,0].std():.3f}',flush=True)
    np.save(f'oof_tabpfn_{tag}.npy', oof); return A,P

print(f'N={N} base={base:.4f} pos={y.sum()}',flush=True)
res={}
res['temporal']=run(list(TimeSeriesSplit(5).split(X)),'temporal')
res['corridor']=run(list(GroupKFold(5).split(X,y,groups=groups)),'corridor')
# random already on disk
roof=np.load('oof_tabpfn_random.npy')
res['random']=(roc_auc_score(y,roof),average_precision_score(y,roof))
print('\n=== TabPFN CV REPORT ===')
for k in ['random','temporal','corridor']:
    print(f'  {k:9s}  AUC={res[k][0]:.4f}  PR-AUC={res[k][1]:.4f}')

import numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata

DATA='Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df=pd.read_csv(DATA,low_memory=False)
df['t']=pd.to_datetime(df['created_date'],errors='coerce')
df=df.sort_values('t').reset_index(drop=True)
y=df['requires_road_closure'].astype(int).values
N=len(y)

def load(f): return np.load(f) if os.path.exists(f) else None
bases=['CatBoost','LightGBM','XGBoost','RandomForest','ExtraTrees','Logistic']

def get(scheme):
    cols={}
    for b in bases:
        a=load(f'oof_{scheme}_{b}.npy')
        if a is not None: cols[b]=a
    t=load(f'oof_tabpfn_{scheme}.npy')
    if t is not None: cols['TabPFN']=t
    M=np.column_stack([cols[k] for k in cols]); mask=~np.isnan(M).any(1)
    return cols, np.where(mask)[0]

def rmean(arrs):
    return np.mean([rankdata(a)/len(a) for a in arrs],axis=0)

def evalb(name, scheme, parts, rank=False):
    cols,idx=get(scheme); yv=y[idx]
    arrs=[cols[p][idx] for p in parts]
    blend = rmean(arrs) if rank else np.mean(arrs,axis=0)
    return roc_auc_score(yv,blend), average_precision_score(yv,blend)

cands={
 'CatBoost only':['CatBoost'],
 'TabPFN only':['TabPFN'],
 'mean(Cat,TabPFN)':['CatBoost','TabPFN'],
 'rank-mean(Cat,TabPFN)':['CatBoost','TabPFN'],
 'mean(Cat,TabPFN,Logistic)':['CatBoost','TabPFN','Logistic'],
 'mean(Cat,TabPFN,RF)':['CatBoost','TabPFN','RandomForest'],
 'mean(all trees)':['CatBoost','LightGBM','XGBoost','RandomForest','ExtraTrees'],
 'mean(all 7)':['CatBoost','LightGBM','XGBoost','RandomForest','ExtraTrees','Logistic','TabPFN'],
 'rank-mean(all 7)':bases+['TabPFN'],
}
for scheme in ['temporal','corridor','random']:
    print(f"\n=== {scheme.upper()} ===")
    print(f"{'candidate':30s} {'ROC':>8s} {'PR':>8s}")
    rows=[]
    for name,parts in cands.items():
        rank = name.startswith('rank')
        roc,pr=evalb(name,scheme,parts,rank=rank)
        rows.append((name,roc,pr))
    for name,roc,pr in sorted(rows,key=lambda r:-r[2]):
        print(f"{name:30s} {roc:8.4f} {pr:8.4f}")

# Robustness summary: min PR across temporal+corridor (worst-case shift)
print("\n=== WORST-CASE (min PR over temporal & corridor) ===")
rows=[]
for name,parts in cands.items():
    rank=name.startswith('rank')
    prs=[evalb(name,s,parts,rank=rank)[1] for s in ['temporal','corridor']]
    rocs=[evalb(name,s,parts,rank=rank)[0] for s in ['temporal','corridor']]
    rows.append((name,min(prs),np.mean(prs),min(rocs),np.mean(rocs)))
print(f"{'candidate':30s} {'minPR':>8s} {'meanPR':>8s} {'minROC':>8s} {'meanROC':>8s}")
for name,mn,mp,mr,mrr in sorted(rows,key=lambda r:-r[1]):
    print(f"{name:30s} {mn:8.4f} {mp:8.4f} {mr:8.4f} {mrr:8.4f}")

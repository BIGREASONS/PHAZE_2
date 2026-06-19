import numpy as np, pandas as pd, os, json
from sklearn.metrics import roc_auc_score, average_precision_score

DATA='Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df=pd.read_csv(DATA,low_memory=False)
df['t']=pd.to_datetime(df['created_date'],errors='coerce')
df=df.sort_values('t').reset_index(drop=True)
y=df['requires_road_closure'].astype(int).values
N=len(y)
print(f"N={N} pos={y.sum()} base_rate={y.mean():.4f}")

# temporal block boundary used in adversarial test (last 20%)
split=int(N*0.8)
print(f"train rate(first80%)={y[:split].mean():.4f}  test rate(last20%)={y[split:].mean():.4f}")

def load(f):
    return np.load(f) if os.path.exists(f) else None

bases=['CatBoost','LightGBM','XGBoost','RandomForest','ExtraTrees','Logistic']
def scheme_table(scheme, extra_tab=True):
    cols={}
    for b in bases:
        a=load(f'oof_{scheme}_{b}.npy')
        if a is not None: cols[b]=a
    tab=load(f'oof_tabpfn_{scheme}.npy')
    if tab is not None: cols['TabPFN']=tab
    # shared joint coverage mask across all loaded models
    M=np.column_stack([cols[k] for k in cols])
    mask=~np.isnan(M).any(1)
    idx=np.where(mask)[0]
    yv=y[idx]
    print(f"\n=== {scheme.upper()}  shared-eval rows={len(idx)}/{N}  pos={yv.sum()} rate={yv.mean():.4f} ===")
    print(f"{'model':14s} {'ROC':>8s} {'PR':>8s}")
    rows=[]
    for k in cols:
        a=cols[k][idx]
        roc=roc_auc_score(yv,a); pr=average_precision_score(yv,a)
        rows.append((k,roc,pr))
    for k,roc,pr in sorted(rows,key=lambda r:-r[2]):
        print(f"{k:14s} {roc:8.4f} {pr:8.4f}")
    return idx,yv,cols

for s in ['random','temporal','corridor']:
    scheme_table(s)

# Optuna / seedbag / stack are random-CV only -> evaluate on full N (they have full coverage)
print("\n=== RANDOM-CV extras (full coverage) ===")
print(f"{'model':22s} {'ROC':>8s} {'PR':>8s}")
extra={
 'seedbag_CatBoost':'oof_seedbag_CatBoost.npy','seedbag_LightGBM':'oof_seedbag_LightGBM.npy',
 'seedbag_XGBoost':'oof_seedbag_XGBoost.npy','seedbag_RandomForest':'oof_seedbag_RandomForest.npy',
 'seedbag_ExtraTrees':'oof_seedbag_ExtraTrees.npy',
 'optuna_catboost':'oof_optuna_catboost.npy','optuna_lgbm':'oof_optuna_lgbm.npy','optuna_xgb':'oof_optuna_xgb.npy',
 'stack_l2_lr':'oof_stack_l2_lr.npy','stack_l2_ridge':'oof_stack_l2_ridge.npy',
 'stack_l2_lgbm_meta':'oof_stack_l2_lgbm_meta.npy','stack_l3':'oof_stack_l3.npy',
}
res={}
for name,f in extra.items():
    a=load(f)
    if a is None: continue
    m=~np.isnan(a)
    roc=roc_auc_score(y[m],a[m]); pr=average_precision_score(y[m],a[m])
    res[name]=(roc,pr,m.mean())
for name in extra:
    if name in res:
        roc,pr,cov=res[name]
        print(f"{name:22s} {roc:8.4f} {pr:8.4f}  cov={cov:.2f}")

# CRITICAL: how do random-CV stack/optuna models do under a TRUE temporal holdout?
# We can't get temporal OOF for them, but we CAN test the random-CV OOF on last-20% block
# only as a sanity check of optimism (random OOF on late rows is still in-distribution OOF,
# so this is NOT a temporal generalization test). Skip to avoid misleading.

# Check oof_temporal_cat.npy vs oof_temporal_CatBoost.npy
a=load('oof_temporal_cat.npy'); b=load('oof_temporal_CatBoost.npy')
if a is not None and b is not None:
    print(f"\noof_temporal_cat vs CatBoost identical? {np.allclose(np.nan_to_num(a),np.nan_to_num(b))}  "
          f"cat_cov={(~np.isnan(a)).mean():.2f} CatBoost_cov={(~np.isnan(b)).mean():.2f}")

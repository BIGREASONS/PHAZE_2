import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

F = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(F, low_memory=False).reset_index(drop=True)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)
y = df['requires_road_closure'].astype(int).values
base = y.mean()

# T=0-legal features only (set at creation)
df['hour'] = df['t'].dt.hour
df['weekday'] = df['t'].dt.weekday
cat = ['event_type','event_cause','veh_type','corridor','police_station','zone']
num = ['latitude','longitude','hour','weekday']
for c in cat: df[c] = df[c].astype(str).fillna('NA')
X = df[cat+num].copy()

try:
    from catboost import CatBoostClassifier, Pool
    def fit_pred(Xtr,ytr,Xte):
        m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                               loss_function='Logloss', verbose=0, random_seed=0,
                               cat_features=cat)
        m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1]
    backend='CatBoost'
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X[cat])
    def fit_pred(Xtr,ytr,Xte):
        Xtr=Xtr.copy(); Xte=Xte.copy()
        Xtr[cat]=enc.transform(Xtr[cat]); Xte[cat]=enc.transform(Xte[cat])
        m=HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=6)
        m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1]
    backend='HistGBM'
print(f'backend={backend}  N={len(df)}  base={base:.4f}\n')

def scores(splits, name):
    a,p=[],[]
    for tr,te in splits:
        pr=fit_pred(X.iloc[tr],y[tr],X.iloc[te])
        a.append(roc_auc_score(y[te],pr)); p.append(average_precision_score(y[te],pr))
    print(f'{name:28s} AUC={np.mean(a):.3f}±{np.std(a):.3f}   PR-AUC={np.mean(p):.3f}±{np.std(p):.3f}  (lift x{np.mean(p)/base:.1f})')

# 1) random 5-fold (what you report)
scores(list(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)), 'Random 5-fold')
# 2) spatial: GroupKFold by corridor
scores(list(GroupKFold(5).split(X,y,groups=df['corridor'])), 'GroupKFold by corridor')
# 3) temporal forward holdout: train past -> test future (expanding, 4 folds)
n=len(df); idx=np.arange(n); tsplits=[]
for k in range(1,5):
    cut=int(n*k/5); tsplits.append((idx[:cut], idx[cut:int(n*(k+1)/5)] if k<4 else idx[cut:]))
scores(tsplits, 'Temporal forward-chain')

# AUC decomposition: easy (cause is near-deterministic no-closure) vs hard remainder
cause_rate = df.groupby('event_cause')['requires_road_closure'].transform('mean')
easy = (cause_rate<0.02).values
pr = np.zeros(n)
for tr,te in StratifiedKFold(5,shuffle=True,random_state=1).split(X,y):
    pr[te]=fit_pred(X.iloc[tr],y[tr],X.iloc[te])
print(f'\nDecomposition (random CV oof preds):')
print(f'  EASY subset (cause closure-rate<2%): {easy.sum()} rows, {y[easy].sum()} closures, base={y[easy].mean():.3f}')
print(f'  HARD subset (remainder):             {(~easy).sum()} rows, {y[~easy].sum()} closures, base={y[~easy].mean():.3f}')
print(f'  AUC overall={roc_auc_score(y,pr):.3f}   AUC on HARD-only={roc_auc_score(y[~easy],pr[~easy]):.3f}')

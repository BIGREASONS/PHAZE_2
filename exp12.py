import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
import h3
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
rng = np.random.default_rng(0)

F = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(F, low_memory=False)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)
df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
y = df['requires_road_closure'].astype(int).values
N = len(df); base = y.mean()

# features
base_cat = ['event_type','event_cause','veh_type','corridor','police_station','zone']
for c in base_cat: df[c] = df[c].astype(str).fillna('NA')
df['hour'] = df['t'].dt.hour.fillna(0).astype(int)
df['weekday'] = df['t'].dt.weekday.fillna(0).astype(int)
df['cause_x_hour'] = df['event_cause'] + '_' + df['hour'].astype(str)
for r in (7,8,9):
    df[f'h3_{r}'] = [h3.latlng_to_cell(a,b,r) for a,b in zip(df['lat'],df['lon'])]
# causal spatial density: prior incidents (time order) within radius
coords = np.radians(df[['lat','lon']].values)
bt = BallTree(coords, metric='haversine')
for R,col in [(500,'density_500m'),(1000,'density_1000m')]:
    rad = R/6371000.0
    nbrs = bt.query_radius(coords, r=rad)
    df[col] = [int((nb < i).sum()) for i,nb in enumerate(nbrs)]  # only earlier (=prior in time) rows

def fit_oof(features, cats, mask=None, splits=5):
    """temporal OOF preds over a (optionally masked) time-sorted frame."""
    idx = np.arange(N) if mask is None else np.where(mask)[0]
    sub = df.iloc[idx].reset_index(drop=True); ys = y[idx]
    tss = TimeSeriesSplit(n_splits=splits)
    oof_p = np.full(len(idx), np.nan); per=[]
    for tr,te in tss.split(sub):
        m = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05,
                               loss_function='Logloss', verbose=0, random_seed=0,
                               cat_features=[c for c in cats if c in features])
        m.fit(sub.iloc[tr][features], ys[tr])
        p = m.predict_proba(sub.iloc[te][features])[:,1]; oof_p[te]=p
        per.append((roc_auc_score(ys[te],p), average_precision_score(ys[te],p)))
    cov = ~np.isnan(oof_p)
    return oof_p, ys, cov, np.array(per)

def boot_ci(yv, pv, fn, n=2000):
    s=[]; k=len(yv)
    for _ in range(n):
        b=rng.integers(0,k,k)
        if yv[b].sum() in (0,k): continue
        s.append(fn(yv[b],pv[b]))
    return np.percentile(s,[2.5,97.5])

def boot_delta(yv,pa,pb,fn,n=2000):
    s=[]; k=len(yv)
    for _ in range(n):
        b=rng.integers(0,k,k)
        if yv[b].sum() in (0,k): continue
        s.append(fn(yv[b],pa[b])-fn(yv[b],pb[b]))
    s=np.array(s); p=2*min((s<=0).mean(),(s>=0).mean())
    return s.mean(), np.percentile(s,[2.5,97.5]), p

num_cols=['lat','lon','hour','weekday','density_500m','density_1000m']
print(f'N={N}  base={base:.4f}  pos={y.sum()}\n')
print('='*94)
print('EXPERIMENT 2 — feature ladder under TimeSeriesSplit(5), temporal OOF')
print('='*94)
configs = [
 ('Base',        base_cat+['lat','lon']),
 ('+Hour',       base_cat+['lat','lon','hour']),
 ('+Weekday',    base_cat+['lat','lon','hour','weekday']),
 ('+Cause×Hour', base_cat+['lat','lon','hour','weekday','cause_x_hour']),
 ('+H3',         base_cat+['lat','lon','hour','weekday','cause_x_hour','h3_7','h3_8','h3_9']),
 ('+Density',    base_cat+['lat','lon','hour','weekday','cause_x_hour','h3_7','h3_8','h3_9','density_500m','density_1000m']),
]
cats_all = base_cat+['cause_x_hour','h3_7','h3_8','h3_9']
store={}
print(f'{"config":13s} {"AUC mean±std":>16s} {"AUC 95%CI":>16s} {"PR mean±std":>16s} {"PR 95%CI":>16s}')
for name,feats in configs:
    oof,ys,cov,per = fit_oof(feats, cats_all)
    yv,pv = ys[cov], oof[cov]; store[name]=(yv,pv)
    aci=boot_ci(yv,pv,roc_auc_score); pci=boot_ci(yv,pv,average_precision_score)
    a=per[:,0]; pr=per[:,1]
    print(f'{name:13s} {a.mean():.3f}±{a.std():.3f}     [{aci[0]:.3f},{aci[1]:.3f}]  {pr.mean():.3f}±{pr.std():.3f}    [{pci[0]:.3f},{pci[1]:.3f}]')

print('\nSIGNIFICANCE (paired bootstrap on common OOF rows)')
yv0,pv0=store['Base']
prev='Base'
print(f'{"comparison":24s} {"ΔAUC":>8s} {"95%CI":>18s} {"p":>7s}   {"ΔPR":>8s} {"95%CI":>18s} {"p":>7s}')
for name,_ in configs[1:]:
    yv,pv=store[name]
    dA,ciA,pA=boot_delta(yv,pv,pv0,roc_auc_score)
    dP,ciP,pP=boot_delta(yv,pv,pv0,average_precision_score)
    print(f'{name+" vs Base":24s} {dA:+.4f} [{ciA[0]:+.3f},{ciA[1]:+.3f}] {pA:6.3f}   {dP:+.4f} [{ciP[0]:+.3f},{ciP[1]:+.3f}] {pP:6.3f}')

print('\n'+'='*94)
print('EXPERIMENT 1 — Planned vs Unplanned (regime-specific temporal CV, event_type dropped within regime)')
print('='*94)
reg_feats = [c for c in base_cat if c!='event_type']+['lat','lon','hour','weekday']
print(f'{"regime":12s} {"n":>6s} {"closures":>9s} {"base":>6s} {"AUC mean±std":>15s} {"AUC 95%CI":>16s} {"PR mean±std":>15s} {"PR 95%CI":>16s}')
for reg in ['planned','unplanned']:
    mask = (df['event_type']==reg).values
    oof,ys,cov,per = fit_oof(reg_feats, base_cat, mask=mask)
    yv,pv=ys[cov],oof[cov]
    aci=boot_ci(yv,pv,roc_auc_score); pci=boot_ci(yv,pv,average_precision_score)
    a=per[:,0]; pr=per[:,1]
    print(f'{reg:12s} {mask.sum():6d} {int(y[mask].sum()):9d} {y[mask].mean():6.3f} {a.mean():.3f}±{a.std():.3f}   [{aci[0]:.3f},{aci[1]:.3f}]  {pr.mean():.3f}±{pr.std():.3f}   [{pci[0]:.3f},{pci[1]:.3f}]')

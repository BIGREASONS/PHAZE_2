"""
Distribution-shift stress test for the GridSight AI road-closure stack.

Honest design (the whole point of the exercise):
  * Base-model OOF is generated INSIDE each CV scheme. A random-CV OOF re-split by
    time would be leaky, so trees + Logistic are refit within Temporal / Corridor /
    Random folds. Only TabPFN (slow) is loaded from its per-scheme cached OOF.
  * Meta-learner is cross-fit with a fold partition that mirrors the scheme, so the
    rows it scores were never in its own training fold.
  * Every table metric for a scheme is computed on ONE shared evaluation mask
    (the meta pooled-OOF coverage) so standalone vs stack is apples-to-apples.

No new features, no embeddings, no H3/OSM/external data. Inputs = the same
6 categoricals + 4 numerics used everywhere else, plus cached TabPFN OOF.
"""
import sys, io, os, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
SEED = 0

# ----------------------------------------------------------------------------- data
df = pd.read_csv(DATA, low_memory=False)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)          # <-- identical order to cached TabPFN OOF
df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
df['hour'] = df['t'].dt.hour.fillna(0).astype(int)
df['weekday'] = df['t'].dt.weekday.fillna(0).astype(int)
y = df['requires_road_closure'].astype(int).values
N = len(df); base = y.mean()
corridor = df['corridor'].astype(str).fillna('NA').values

cats = ['event_type','event_cause','veh_type','corridor','police_station','zone']
nums = ['lat','lon','hour','weekday']
for c in cats: df[c] = df[c].astype(str).fillna('NA')
codes = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(df[cats]).astype(int)
Xtree = pd.DataFrame(codes, columns=cats)
for c in nums: Xtree[c] = df[c].astype(float).values
Xraw = df[cats + nums].copy()                            # for the LR one-hot pipeline
cat_idx = list(range(len(cats)))

# ----------------------------------------------------------------------------- schemes
SCHEMES = {
    'random'  : list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtree, y)),
    'temporal': list(TimeSeriesSplit(5).split(Xtree)),
    'corridor': list(GroupKFold(5).split(Xtree, y, groups=corridor)),
}

# ----------------------------------------------------------------------------- base models
mk = {
 'CatBoost'    : lambda: CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, verbose=0,
                                            random_seed=SEED, cat_features=cat_idx, thread_count=-1),
 'LightGBM'    : lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                        random_state=SEED, n_jobs=-1, verbose=-1),
 'XGBoost'     : lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                       tree_method='hist', random_state=SEED, n_jobs=-1, eval_metric='logloss'),
 'RandomForest': lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED, class_weight='balanced'),
 'ExtraTrees'  : lambda: ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=SEED, class_weight='balanced'),
}
fitkw = {'LightGBM': lambda tr: {'categorical_feature': cat_idx}}
BASE_ORDER = ['CatBoost','LightGBM','XGBoost','RandomForest','ExtraTrees','Logistic','TabPFN']

def tree_oof(name, scheme):
    f = f'oof_{scheme}_{name}.npy'
    if os.path.exists(f):
        o = np.load(f)
        if o.shape[0] == N: return o
    oof = np.full(N, np.nan)
    for tr, te in SCHEMES[scheme]:
        m = mk[name](); m.fit(Xtree.iloc[tr], y[tr], **fitkw.get(name, lambda tr: {})(tr))
        oof[te] = m.predict_proba(Xtree.iloc[te])[:, 1]
    np.save(f, oof); return oof

def lr_oof(scheme):
    f = f'oof_{scheme}_Logistic.npy'
    if os.path.exists(f):
        o = np.load(f)
        if o.shape[0] == N: return o
    oof = np.full(N, np.nan)
    ct = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore', min_frequency=10), cats),
                            ('n', StandardScaler(), nums)])
    for tr, te in SCHEMES[scheme]:
        pipe = Pipeline([('ct', ct), ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))])
        pipe.fit(Xraw.iloc[tr], y[tr]); oof[te] = pipe.predict_proba(Xraw.iloc[te])[:, 1]
    np.save(f, oof); return oof

print(f'N={N} base={base:.4f} pos={y.sum()}', flush=True)
print('Generating / loading within-scheme base OOF ...', flush=True)
OOF = {s: {} for s in SCHEMES}
for s in SCHEMES:
    t0 = time.time()
    for name in mk: OOF[s][name] = tree_oof(name, s)
    OOF[s]['Logistic'] = lr_oof(s)
    OOF[s]['TabPFN'] = np.load(f'oof_tabpfn_{s}.npy')          # per-scheme cached (slow model)
    cov = ~np.isnan(np.column_stack([OOF[s][m] for m in BASE_ORDER])).any(1)
    print(f'  [{s:8s}] base OOF ready, joint coverage={cov.sum()}/{N}  ({time.time()-t0:.0f}s)', flush=True)

# ----------------------------------------------------------------------------- meta CV (mirrors scheme)
def meta_folds(scheme, idx):
    """fold list over positions of `idx` (already restricted to base-covered, time-sorted)."""
    if scheme == 'random':
        return list(StratifiedKFold(5, shuffle=True, random_state=1).split(idx, y[idx]))
    if scheme == 'corridor':
        return list(GroupKFold(5).split(idx, y[idx], groups=corridor[idx]))
    return list(TimeSeriesSplit(5).split(idx))                # temporal: forward-chained meta

def fit_meta(kind, Xtr, ytr, Xte):
    if kind == 'Logistic':
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    if kind == 'Ridge':
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)      # regress 0/1; AUC/AP are rank metrics
        return m.predict(sc.transform(Xte))
    if kind == 'LightGBM':
        m = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                           random_state=SEED, n_jobs=-1, verbose=-1).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

META = ['Logistic','Ridge','LightGBM']

def stack_eval(scheme, cols):
    """Return pooled (auc,ap), per-fold std, eval-mask. cols = base models fed to meta."""
    base_cov = ~np.isnan(np.column_stack([OOF[scheme][m] for m in cols])).any(1)
    idx = np.where(base_cov)[0]                               # base-covered positions, time-sorted
    B = np.column_stack([OOF[scheme][m][idx] for m in cols])
    yv = y[idx]
    out = {}
    for kind in META:
        pred = np.full(len(idx), np.nan); per = []
        for tr, te in meta_folds(scheme, idx):
            p = fit_meta(kind, B[tr], yv[tr], B[te]); pred[te] = p
            per.append((roc_auc_score(yv[te], p), average_precision_score(yv[te], p)))
        cov = ~np.isnan(pred); per = np.array(per)
        out[kind] = dict(auc=roc_auc_score(yv[cov], pred[cov]),
                         ap=average_precision_score(yv[cov], pred[cov]),
                         auc_sd=per[:,0].std(), ap_sd=per[:,1].std(),
                         eval_idx=idx[cov])
    return out

def standalone_on(scheme, eval_idx):
    """score each base model on a fixed eval index set."""
    res = {}
    yv = y[eval_idx]
    for m in BASE_ORDER:
        p = OOF[scheme][m][eval_idx]
        res[m] = (roc_auc_score(yv, p), average_precision_score(yv, p))
    return res

# ----------------------------------------------------------------------------- run all schemes
RESULT = {}
for s in SCHEMES:
    full = stack_eval(s, BASE_ORDER)                         # WITH TabPFN
    notab = stack_eval(s, [m for m in BASE_ORDER if m != 'TabPFN'])  # WITHOUT TabPFN
    # shared eval mask = Logistic-with-TabPFN coverage (same fold structure for all metas)
    emask = full['Logistic']['eval_idx']
    RESULT[s] = dict(full=full, notab=notab, emask=emask, standalone=standalone_on(s, emask))

# ----------------------------------------------------------------------------- TABLES
def best_standalone(s):
    sa = RESULT[s]['standalone']
    name = max(sa, key=lambda m: sa[m][1])                   # best by PR-AUC
    return name, sa[name]

print('\n' + '='*78)
print('DISTRIBUTION-SHIFT STRESS TEST  (all metrics per-scheme on shared eval mask)')
print('='*78)
for s in SCHEMES:
    print(f'  [{s:8s}] eval rows used = {len(RESULT[s]["emask"])}')

print('\nTABLE A — Best standalone model under each CV scheme')
print(f'{"scheme":10s} | {"best model":13s} | {"ROC-AUC":>8s} | {"PR-AUC":>8s}')
for s in SCHEMES:
    n,(a,p) = best_standalone(s)
    print(f'{s:10s} | {n:13s} | {a:8.4f} | {p:8.4f}')

print('\nTABLE B — Stack performance under each CV scheme (meta on all 7 base models)')
print(f'{"scheme":10s} | {"meta":9s} | {"ROC-AUC":>8s} | {"PR-AUC":>8s} | {"AUC sd":>7s} | {"AP sd":>7s}')
for s in SCHEMES:
    for k in META:
        r = RESULT[s]['full'][k]
        print(f'{s:10s} | {k:9s} | {r["auc"]:8.4f} | {r["ap"]:8.4f} | {r["auc_sd"]:7.4f} | {r["ap_sd"]:7.4f}')

def best_meta(s):
    return max(META, key=lambda k: RESULT[s]['full'][k]['ap'])

print('\nTABLE C — Lift of stack (best meta by PR-AUC) over best standalone')
print(f'{"scheme":10s} | {"best meta":9s} | {"dAUC":>8s} | {"dPR-AUC":>8s}')
for s in SCHEMES:
    bn,(ba,bp) = best_standalone(s); bm = best_meta(s); r = RESULT[s]['full'][bm]
    print(f'{s:10s} | {bm:9s} | {r["auc"]-ba:+8.4f} | {r["ap"]-bp:+8.4f}')

print('\nTABLE D — TabPFN contribution (stack PR-AUC with vs without TabPFN)')
print(f'{"scheme":10s} | {"meta":9s} | {"AP w/o":>8s} | {"AP w/":>8s} | {"dAP":>8s}')
keep_votes = {}
for s in SCHEMES:
    for k in META:
        wo = RESULT[s]['notab'][k]['ap']; w = RESULT[s]['full'][k]['ap']
        print(f'{s:10s} | {k:9s} | {wo:8.4f} | {w:8.4f} | {w-wo:+8.4f}')
    # per-scheme decision uses best meta
    bm = best_meta(s)
    keep_votes[s] = RESULT[s]['full'][bm]['ap'] - RESULT[s]['notab'][bm]['ap']

# ----------------------------------------------------------------------------- VERDICT
print('\n' + '='*78)
print('VERDICT')
print('='*78)
shift = ['temporal','corridor']
lifts = {}
for s in shift:
    bn,(ba,bp) = best_standalone(s); bm = best_meta(s); lifts[s] = RESULT[s]['full'][bm]['ap'] - bp
survives = all(lifts[s] > 0 for s in shift)
for s in shift:
    bn,(ba,bp) = best_standalone(s); bm = best_meta(s); r = RESULT[s]['full'][bm]
    sd = r['ap_sd']
    tag = 'beats' if lifts[s] > 0 else 'does NOT beat'
    print(f'  {s:8s}: stack({bm}) PR-AUC={r["ap"]:.4f} vs best standalone({bn}) {bp:.4f}'
          f'  -> lift {lifts[s]:+.4f}  (fold AP sd {sd:.3f})  [{tag}]')
print()
if survives:
    print('  >>> (A) STACK SURVIVES DISTRIBUTION SHIFT')
    print('      Stack PR-AUC exceeds best standalone under BOTH temporal and corridor CV.')
else:
    print('  >>> (B) STACK ONLY WORKS UNDER RANDOM CV')
    print('      Under at least one shifted scheme the stack does not beat the best single model.')
print(f'\n  TabPFN dAP by scheme (best meta): ' +
      ', '.join(f'{s}={keep_votes[s]:+.4f}' for s in SCHEMES))

# persist machine-readable summary
summary = {s: {
    'eval_rows': int(len(RESULT[s]['emask'])),
    'best_standalone': dict(model=best_standalone(s)[0], auc=best_standalone(s)[1][0], ap=best_standalone(s)[1][1]),
    'stack': {k: {kk: float(vv) for kk, vv in RESULT[s]['full'][k].items() if kk != 'eval_idx'} for k in META},
    'tabpfn_dAP': {k: float(RESULT[s]['full'][k]['ap'] - RESULT[s]['notab'][k]['ap']) for k in META},
} for s in SCHEMES}
json.dump(summary, open('stack_shift_summary.json','w'), indent=2)
print('\nsaved: oof_{scheme}_{model}.npy, stack_shift_summary.json')

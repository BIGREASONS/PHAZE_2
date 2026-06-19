import pandas as pd, numpy as np
pd.set_option('display.width', 200)

F = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(F, low_memory=False)
y = df['requires_road_closure'].astype(int)
N = len(df)
base = y.mean()
print('='*70)
print(f'N={N}  base rate P(closure)={base:.4f}  pos={y.sum()} neg={(1-y).sum()}')

# ---- 1. Is closure just `priority` in disguise? ----
print('\n[1] CLOSURE vs PRIORITY (the discarded target)')
print(pd.crosstab(df['priority'], df['requires_road_closure'], margins=True))
print('event_type values:', df['event_type'].unique())
print(pd.crosstab(df['event_type'], df['requires_road_closure'], normalize='index'))

# ---- 2. Determinism trap: does an intake lookup table predict closure? ----
print('\n[2] DETERMINISM TRAP')
for keys in [['corridor'], ['event_cause'], ['event_type','event_cause','corridor']]:
    g = df.groupby(keys)['requires_road_closure'].agg(['mean','count'])
    # fraction of ROWS sitting in a near-deterministic group (mean<0.05 or >0.95) with support>=10
    det = g[(g['count']>=10) & ((g['mean']<0.05)|(g['mean']>0.95))]
    rows_in_det = df.merge(det.reset_index()[keys], on=keys, how='inner').shape[0] if len(det) else 0
    print(f'  by {keys}: {len(g)} groups; {rows_in_det}/{N} rows ({rows_in_det/N:.0%}) in near-deterministic group (|p|>.95, n>=10)')

# ---- 3. Timestamp provenance: is the record (and its text) mutated after T=0? ----
print('\n[3] TIMESTAMP PROVENANCE')
for c in ['created_date','modified_datetime','closed_datetime','start_datetime']:
    df[c+'_dt'] = pd.to_datetime(df[c], errors='coerce')
gap_mod = (df['modified_datetime_dt'] - df['created_date_dt']).dt.total_seconds()/60
print(f'  modified - created (min): median={gap_mod.median():.1f}  %edited(>1min)={ (gap_mod>1).mean():.1%}  %edited(>60min)={(gap_mod>60).mean():.1%}')
# For closed rows, is the LAST modification at closure time (not creation)?
m = df['closed_datetime_dt'].notna()
gap_modclose = (df.loc[m,'modified_datetime_dt'] - df.loc[m,'closed_datetime_dt']).dt.total_seconds().abs()/60
print(f'  rows with closed_datetime: {m.sum()} ({m.mean():.0%})')
print(f'    |modified - closed| (min): median={gap_modclose.median():.1f}  %within 1min={ (gap_modclose<1).mean():.1%}  %within 60min={(gap_modclose<60).mean():.1%}')
print(f'    => if modified clusters at closure time, the row state (incl. description) is POST-T=0')
print('  closure rate by whether closed_datetime exists:')
print(df.assign(has_closed=m).groupby('has_closed')['requires_road_closure'].mean())

# ---- 4. Text leakage with PROPER base rate + coverage ----
print('\n[4] TEXT LEAKAGE')
kws = ['closed','closure','cleared','resolved','removed','done','reopened','restored','shifted','towed','normal']
d = df['description'].fillna('').str.lower()
has_text = df['description'].notna()
print(f'  description present: {has_text.mean():.0%}; P(closure|text)={y[has_text].mean():.3f} vs P(closure|no text)={y[~has_text].mean():.3f}')
kw = d.apply(lambda s: any(k in s for k in kws))
print(f'  keyword coverage={kw.mean():.1%}  base={base:.3f}')
print(f'  P(closure | keyword)   = {y[kw].mean():.3f}   (n={kw.sum()})')
print(f'  P(closure | no keyword)= {y[~kw].mean():.3f}   (n={(~kw).sum()})')
# does TEXT LENGTH alone leak (independent of keywords)?
length = d.str.len()
print('  closure rate by description-length quartile (proxy for "how much happened"):')
ql = pd.qcut(length[has_text], 4, duplicates='drop')
print(y[has_text].groupby(ql, observed=True).mean())

# ---- 5. status field ----
print('\n[5] STATUS (a lifecycle field) vs closure')
print(df.groupby('status')['requires_road_closure'].agg(['mean','count']))

# ---- 6. temporal range (is temporal CV feasible?) ----
print('\n[6] TIME RANGE')
print('  created_date:', df['created_date_dt'].min(), '->', df['created_date_dt'].max())

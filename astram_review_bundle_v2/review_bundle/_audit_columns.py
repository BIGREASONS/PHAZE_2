import pandas as pd, numpy as np, warnings, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings('ignore')
def asc(x): return str(x).encode('ascii','replace').decode()
pd.set_option('display.width',200)
DATA='Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df=pd.read_csv(DATA,low_memory=False)
N=len(df)
y=df['requires_road_closure'].astype(str).str.upper().isin(['TRUE','1','YES']).astype(int).values
print(f"N={N}  target_rate={y.mean():.4f}  pos={y.sum()}")

USED={'event_type','event_cause','veh_type','corridor','police_station','zone',
      'latitude','longitude','created_date','requires_road_closure'}

def nullish(s):
    s2=s.astype(str).str.strip()
    return ~s2.isin(['','NULL','null','None','nan','NaN','NA','0','0.0','FALSE'])

print(f"\n{'column':24s} {'used':4s} {'dtype':10s} {'nunq':>7s} {'fill%':>6s} {'meanY|present':>12s} {'meanY|absent':>12s}  sample")
for c in df.columns:
    present=nullish(df[c]).values
    fill=present.mean()*100
    nun=df[c].nunique(dropna=True)
    yp = y[present].mean() if present.sum()>0 else float('nan')
    ya = y[~present].mean() if (~present).sum()>0 else float('nan')
    samp=[asc(v)[:22] for v in df.loc[present,c].dropna().unique()[:3]]
    used='YES' if c in USED else '-'
    print(f"{c:24s} {used:4s} {str(df[c].dtype):10s} {nun:7d} {fill:6.1f} {yp:12.4f} {ya:12.4f}  {samp}")

# target rate by key low-card categoricals (unused candidates)
for c in ['priority','status','direction','cargo_material','reason_breakdown','authenticated','veh_type','event_type']:
    if c in df.columns and df[c].nunique()<=25:
        print(f"\n-- target rate by {c} (top by count) --")
        g=df.groupby(df[c].astype(str)).agg(n=('id','size'))
        g['rate']=df.groupby(df[c].astype(str)).apply(lambda d: y[d.index].mean())
        print(g.sort_values('n',ascending=False).head(12).to_string())

# numeric candidates
for c in ['age_of_truck','endlatitude','endlongitude','veh_no']:
    if c in df.columns:
        v=pd.to_numeric(df[c],errors='coerce')
        print(f"\n-- {c}: numeric_nonnull={v.notna().mean()*100:.1f}%  min={v.min()} max={v.max()} median={v.median()}")

# text length features
for c in ['description','comment','address','end_address','route_path','meta_data','map_file']:
    if c in df.columns:
        L=df[c].astype(str).str.strip().replace({'NULL':'','nan':''}).str.len()
        nonempty=(L>0).mean()*100
        if nonempty>0:
            yhas=y[(L>0).values].mean(); ynot=y[(L==0).values].mean() if (L==0).any() else float('nan')
            print(f"\n-- TEXT {c}: nonempty={nonempty:.1f}%  meanlen={L[L>0].mean():.0f}  meanY|has={yhas:.4f} meanY|empty={ynot:.4f}")

# timestamp coverage at T=0?
for c in ['start_datetime','end_datetime','created_date','modified_datetime','closed_datetime','resolved_datetime']:
    if c in df.columns:
        t=pd.to_datetime(df[c],errors='coerce',utc=True)
        print(f"\n-- TS {c}: parse%={t.notna().mean()*100:.1f}  min={t.min()} max={t.max()}")

# duration: end-start, resolved-created (POST-resolution -> leakage check)
cd=pd.to_datetime(df['created_date'],errors='coerce',utc=True)
for a,b in [('start_datetime','end_datetime'),('created_date','closed_datetime'),('created_date','resolved_datetime')]:
    ta=pd.to_datetime(df[a],errors='coerce',utc=True); tb=pd.to_datetime(df[b],errors='coerce',utc=True)
    dur=(tb-ta).dt.total_seconds()/60
    ok=dur.notna()&(dur>=0)
    if ok.sum()>0:
        print(f"\n-- DUR {b}-{a}: avail%={ok.mean()*100:.1f}  median_min={dur[ok].median():.0f}  meanY|long(>median)={y[(ok)&(dur>dur[ok].median())].mean():.4f} meanY|short={y[(ok)&(dur<=dur[ok].median())].mean():.4f}")

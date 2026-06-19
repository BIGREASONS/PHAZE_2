import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
F = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
CACHE = 'osm_features.csv'
df = pd.read_csv(F, low_memory=False)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)
df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')

MAJOR = {'motorway','trunk','primary','secondary','motorway_link','trunk_link','primary_link','secondary_link'}
def first(h): return h[0] if isinstance(h,list) else h

if not os.path.exists(CACHE):
    import osmnx as ox, geopandas as gpd
    ox.settings.use_cache=True; ox.settings.requests_timeout=900; ox.settings.log_console=False
    n,s = df['lat'].max()+0.01, df['lat'].min()-0.01
    e,w = df['lon'].max()+0.01, df['lon'].min()-0.01
    print('downloading drive graph...', flush=True)
    try:
        G = ox.graph_from_bbox((w,s,e,n), network_type='drive')        # osmnx>=2.0 bbox=(left,bottom,right,top)
    except TypeError:
        G = ox.graph_from_bbox(north=n,south=s,east=e,west=w,network_type='drive')
    print('graph:', len(G.nodes),'nodes', len(G.edges),'edges', flush=True)
    X=df['lon'].values; Y=df['lat'].values
    ne = ox.distance.nearest_edges(G, X=X, Y=Y)
    rt=[]
    for u,v,k in ne:
        rt.append(first(G.get_edge_data(u,v,k).get('highway','unknown')))
    df['road_type']=rt
    nn = ox.distance.nearest_nodes(G, X=X, Y=Y)
    df['intersection_degree']=[G.nodes[x].get('street_count', G.degree(x)) for x in nn]
    # distance to nearest MAJOR road (projected metres)
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
    edges['hw']=edges['highway'].apply(first)
    maj = edges[edges['hw'].isin(MAJOR)][['geometry']].copy()
    crs_m = 32643  # UTM 43N (Bangalore)
    maj_m = maj.to_crs(crs_m)
    pts = gpd.GeoDataFrame(df.index.to_frame(name='ridx'),
                           geometry=gpd.points_from_xy(df['lon'],df['lat']), crs=4326).to_crs(crs_m)
    j = gpd.sjoin_nearest(pts, maj_m, how='left', distance_col='d')
    df['distance_to_major_road'] = j.groupby('ridx')['d'].min().reindex(df.index).values
    df[['road_type','intersection_degree','distance_to_major_road']].to_csv(CACHE, index=False)
    print('cached', CACHE, flush=True)
else:
    o = pd.read_csv(CACHE)
    for c in o.columns: df[c]=o[c].values
    print('loaded cached OSM features', flush=True)

# ---- model: Base vs Base+OSM under TimeSeriesSplit ----
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
rng=np.random.default_rng(0)
y=df['requires_road_closure'].astype(int).values; N=len(df); base=y.mean()
base_cat=['event_type','event_cause','veh_type','corridor','police_station','zone']
for c in base_cat: df[c]=df[c].astype(str).fillna('NA')
df['road_type']=df['road_type'].astype(str).fillna('NA')
df['intersection_degree']=pd.to_numeric(df['intersection_degree'],errors='coerce').fillna(0)
df['distance_to_major_road']=pd.to_numeric(df['distance_to_major_road'],errors='coerce').fillna(-1)
print('road_type top:', df['road_type'].value_counts().head(6).to_dict())
print('intersection_degree dist:', df['intersection_degree'].describe()[['mean','min','max']].to_dict())
print('dist_to_major(m):', df['distance_to_major_road'].describe()[['mean','50%','max']].to_dict())

def fit_oof(feats,cats):
    tss=TimeSeriesSplit(5); oof=np.full(N,np.nan); per=[]
    for tr,te in tss.split(df):
        m=CatBoostClassifier(iterations=250,depth=6,learning_rate=0.05,verbose=0,random_seed=0,
                             cat_features=[c for c in cats if c in feats])
        m.fit(df.iloc[tr][feats],y[tr]); p=m.predict_proba(df.iloc[te][feats])[:,1]; oof[te]=p
        per.append((roc_auc_score(y[te],p),average_precision_score(y[te],p)))
    cov=~np.isnan(oof); return oof,cov,np.array(per)
def bdelta(yv,pa,pb,fn,n=2000):
    s=[];k=len(yv)
    for _ in range(n):
        b=rng.integers(0,k,k)
        if yv[b].sum() in (0,k):continue
        s.append(fn(yv[b],pa[b])-fn(yv[b],pb[b]))
    s=np.array(s);return s.mean(),np.percentile(s,[2.5,97.5]),2*min((s<=0).mean(),(s>=0).mean())
def bci(yv,pv,fn,n=2000):
    s=[];k=len(yv)
    for _ in range(n):
        b=rng.integers(0,k,k)
        if yv[b].sum() in (0,k):continue
        s.append(fn(yv[b],pv[b]))
    return np.percentile(s,[2.5,97.5])

osm=['road_type','intersection_degree','distance_to_major_road']
print('\n'+'='*90)
print('EXPERIMENT 3 — OSM micro-test under TimeSeriesSplit(5)')
print('='*90)
res={}
for name,feats in [('Base',base_cat+['lat','lon']),('Base+OSM',base_cat+['lat','lon']+osm)]:
    oof,cov,per=fit_oof(feats,base_cat+['road_type']); res[name]=(y[cov],oof[cov])
    a,pr=per[:,0],per[:,1]; aci=bci(y[cov],oof[cov],roc_auc_score); pci=bci(y[cov],oof[cov],average_precision_score)
    print(f'{name:9s} AUC={a.mean():.3f}±{a.std():.3f} CI[{aci[0]:.3f},{aci[1]:.3f}]  PR={pr.mean():.3f}±{pr.std():.3f} CI[{pci[0]:.3f},{pci[1]:.3f}]')
yv,_=res['Base']
dA,ciA,pA=bdelta(res['Base+OSM'][0],res['Base+OSM'][1],res['Base'][1],roc_auc_score)
dP,ciP,pP=bdelta(res['Base+OSM'][0],res['Base+OSM'][1],res['Base'][1],average_precision_score)
print(f'\nΔ(OSM): AUC {dA:+.4f} CI[{ciA[0]:+.3f},{ciA[1]:+.3f}] p={pA:.3f}   PR {dP:+.4f} CI[{ciP[0]:+.3f},{ciP[1]:+.3f}] p={pP:.3f}')

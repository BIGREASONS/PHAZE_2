import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
import h3
# import osmnx as ox
# import networkx as nx

from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

# Preprocessing
df['target_closure'] = df['requires_road_closure'].fillna(False).astype(int)
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce')

df['hour'] = df['start_datetime'].dt.hour.fillna(df['created_date'].dt.hour).fillna(0).astype(int)
df['weekday'] = df['start_datetime'].dt.weekday.fillna(df['created_date'].dt.weekday).fillna(0).astype(int)

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = df['latitude'].fillna(df['latitude'].median())
df['longitude'] = df['longitude'].fillna(df['longitude'].median())

base_features = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'hour', 'weekday', 'latitude', 'longitude']
cat_features = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'hour', 'weekday']

for c in cat_features:
    df[c] = df[c].fillna('missing').astype(str)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.metrics import roc_auc_score

def evaluate(X, y, cat_cols):
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = CatBoostClassifier(iterations=200, cat_features=cat_cols, verbose=0, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_te)[:, 1]
        try:
            scores.append(roc_auc_score(y_te, preds))
        except:
            scores.append(0.5)
    return np.mean(scores)

y = df['target_closure']

print("\n--- EXP 1: H3 Ablation ---")
def get_h3(lat, lon, res):
    try: return h3.geo_to_h3(lat, lon, res)
    except: return 'missing'

df['h3_7'] = df.apply(lambda r: get_h3(r['latitude'], r['longitude'], 7), axis=1)
df['h3_8'] = df.apply(lambda r: get_h3(r['latitude'], r['longitude'], 8), axis=1)
df['h3_9'] = df.apply(lambda r: get_h3(r['latitude'], r['longitude'], 9), axis=1)

base_score = evaluate(df[base_features], y, cat_features)
print(f"Base AUC: {base_score:.4f}")

score_h3_7 = evaluate(df[base_features + ['h3_7']], y, cat_features + ['h3_7'])
print(f"Base + H3(7) AUC: {score_h3_7:.4f}")

score_h3_8 = evaluate(df[base_features + ['h3_8']], y, cat_features + ['h3_8'])
print(f"Base + H3(8) AUC: {score_h3_8:.4f}")

score_h3_9 = evaluate(df[base_features + ['h3_9']], y, cat_features + ['h3_9'])
print(f"Base + H3(9) AUC: {score_h3_9:.4f}")


print("\n--- EXP 3: Spatial Density ---")
rad = np.deg2rad(df[['latitude', 'longitude']].values)
tree = BallTree(rad, metric='haversine')
# 500m = 0.5/6371, 1km = 1.0/6371
df['density_500m'] = tree.query_radius(rad, r=0.5/6371, count_only=True) - 1
df['density_1km'] = tree.query_radius(rad, r=1.0/6371, count_only=True) - 1

# OOF historical closure rate for H3_8
df['hist_closure_rate'] = 0.0
for train_idx, test_idx in kf.split(df):
    X_tr, X_te = df.iloc[train_idx], df.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    rate_map = y_tr.groupby(X_tr['h3_8']).mean().to_dict()
    df.loc[test_idx, 'hist_closure_rate'] = X_te['h3_8'].map(rate_map).fillna(y_tr.mean())

spatial_features = base_features + ['h3_8', 'density_500m', 'density_1km', 'hist_closure_rate']
spatial_cat = cat_features + ['h3_8']
score_spatial = evaluate(df[spatial_features], y, spatial_cat)
print(f"Base + H3(8) + Density + HistRate AUC: {score_spatial:.4f}")


print("\n--- EXP 4: Text Audit ---")
print("Null count in description:", df['description'].isna().sum(), "/", len(df))
print("Null count in comment:", df['comment'].isna().sum(), "/", len(df))

# Check words implying leakage
leak_words = ['clear', 'resolv', 'clos', 'finish', 'remov', 'done']
desc = df['description'].fillna('').str.lower()
leaks = desc.str.contains('|'.join(leak_words))
print(f"Rows with suspicious 'after-the-fact' words in description: {leaks.sum()} ({leaks.mean()*100:.1f}%)")
if leaks.sum() > 0:
    print("Example suspicious descriptions:")
    print(df[leaks]['description'].head(3).tolist())


print("\n--- EXP 5: Closure Distribution ---")
print("Top 5 Event Types by Closure Rate:")
print(df.groupby('event_type')['target_closure'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(5))

print("\nTop 5 Event Causes by Closure Rate:")
print(df.groupby('event_cause')['target_closure'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(5))


print("\n--- EXP 2: OSMnx Road Hierarchy ---")
try:
    print("Downloading OSM graph for bounding box... this might take a minute.")
    # Pad bbox slightly
    north, south = df['latitude'].max() + 0.01, df['latitude'].min() - 0.01
    east, west = df['longitude'].max() + 0.01, df['longitude'].min() - 0.01
    G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='drive')
    print("Graph downloaded. Mapping incidents to nearest edges...")
    edges = ox.distance.nearest_edges(G, X=df['longitude'].values, Y=df['latitude'].values)
    
    road_types = []
    for u, v, key in edges:
        edge_data = G.get_edge_data(u, v, key)
        hw = edge_data.get('highway', 'unknown')
        if isinstance(hw, list):
            hw = hw[0]
        road_types.append(hw)
        
    df['osm_highway'] = road_types
    print("Road hierarchy distribution:")
    print(df['osm_highway'].value_counts().head(10))
    
    osm_features = base_features + ['osm_highway']
    osm_cat = cat_features + ['osm_highway']
    score_osm = evaluate(df[osm_features], y, osm_cat)
    print(f"Base + OSM Hierarchy AUC: {score_osm:.4f}")
    
except Exception as e:
    print("OSMnx failed:", e)

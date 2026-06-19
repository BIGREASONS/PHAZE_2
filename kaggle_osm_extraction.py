import pandas as pd
import numpy as np
import osmnx as ox
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# 1. Load Data
df = pd.read_csv('/kaggle/input/astram-dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
df['target_closure'] = df['requires_road_closure'].fillna(False).astype(int)
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(12.9716)
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(77.5946)

# 2. OSMnx Graph
north, south = df['latitude'].max() + 0.01, df['latitude'].min() - 0.01
east, west = df['longitude'].max() + 0.01, df['longitude'].min() - 0.01
G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='drive')
edges = ox.distance.nearest_edges(G, X=df['longitude'].values, Y=df['latitude'].values)

hw_list = []
for u, v, key in edges:
    hw = G.get_edge_data(u, v, key).get('highway', 'unknown')
    hw_list.append(hw[0] if isinstance(hw, list) else hw)
    
df['osm_highway'] = hw_list

# 3. Model
base_cat = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone']
for c in base_cat: df[c] = df[c].fillna('missing').astype(str)
df['osm_highway'] = df['osm_highway'].astype(str)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
def eval_model(features, cat_cols):
    aucs = []
    for tr, te in kf.split(df):
        m = CatBoostClassifier(iterations=200, cat_features=cat_cols, verbose=0, random_state=42)
        m.fit(df.iloc[tr][features], df.iloc[tr]['target_closure'])
        aucs.append(roc_auc_score(df.iloc[te]['target_closure'], m.predict_proba(df.iloc[te][features])[:, 1]))
    return np.mean(aucs)

base_feat = base_cat + ['latitude', 'longitude']
print("Base AUC:", eval_model(base_feat, base_cat))
print("Base + OSM AUC:", eval_model(base_feat + ['osm_highway'], base_cat + ['osm_highway']))
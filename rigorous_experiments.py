import pandas as pd
import numpy as np
import h3
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import BallTree
from catboost import CatBoostClassifier
import json
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

# Prep Target
df['target_closure'] = df['requires_road_closure'].fillna(False).astype(int)
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = df['latitude'].fillna(df['latitude'].median())
df['longitude'] = df['longitude'].fillna(df['longitude'].median())

# Base categorical cleaning
base_cat = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone']
for c in base_cat:
    df[c] = df[c].fillna('missing').astype(str)

base_features = base_cat + ['latitude', 'longitude']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def eval_cv(X, y, cat_features, model_name="CatBoost"):
    fold_aucs = []
    importances = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = CatBoostClassifier(iterations=200, cat_features=cat_features, verbose=0, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_te)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y_te, preds))
        except:
            fold_aucs.append(0.5)
        importances.append(model.get_feature_importance())
    return fold_aucs, np.mean(importances, axis=0)

y = df['target_closure']

print("--- EXP 1: H3 Ablation ---")
def get_h3(lat, lon, res):
    try: return h3.geo_to_h3(lat, lon, res)
    except: return 'missing'

df['h3_7'] = df.apply(lambda r: get_h3(r['latitude'], r['longitude'], 7), axis=1)
df['h3_8'] = df.apply(lambda r: get_h3(r['latitude'], r['longitude'], 8), axis=1)
df['h3_9'] = df.apply(lambda r: get_h3(r['latitude'], r['longitude'], 9), axis=1)

base_auc, base_imp = eval_cv(df[base_features], y, base_cat)
h3_7_auc, _ = eval_cv(df[base_features + ['h3_7']], y, base_cat + ['h3_7'])
h3_8_auc, h3_8_imp = eval_cv(df[base_features + ['h3_8']], y, base_cat + ['h3_8'])
h3_9_auc, _ = eval_cv(df[base_features + ['h3_9']], y, base_cat + ['h3_9'])

h3_res = pd.DataFrame({
    'Model': ['Base', 'Base + H3(7)', 'Base + H3(8)', 'Base + H3(9)'],
    'Features': [str(base_features), str(base_features+['h3_7']), str(base_features+['h3_8']), str(base_features+['h3_9'])],
    'Folds': 5,
    'Seed': 42,
    'Fold_1_AUC': [base_auc[0], h3_7_auc[0], h3_8_auc[0], h3_9_auc[0]],
    'Fold_2_AUC': [base_auc[1], h3_7_auc[1], h3_8_auc[1], h3_9_auc[1]],
    'Fold_3_AUC': [base_auc[2], h3_7_auc[2], h3_8_auc[2], h3_9_auc[2]],
    'Fold_4_AUC': [base_auc[3], h3_7_auc[3], h3_8_auc[3], h3_9_auc[3]],
    'Fold_5_AUC': [base_auc[4], h3_7_auc[4], h3_8_auc[4], h3_9_auc[4]],
    'Mean_AUC': [np.mean(base_auc), np.mean(h3_7_auc), np.mean(h3_8_auc), np.mean(h3_9_auc)],
    'Std_AUC': [np.std(base_auc), np.std(h3_7_auc), np.std(h3_8_auc), np.std(h3_9_auc)]
})
h3_res.to_csv('h3_ablation_results.csv', index=False)

h3_imp_df = pd.DataFrame({'feature': base_features + ['h3_8'], 'importance': h3_8_imp})
h3_imp_df.to_csv('h3_feature_importance.csv', index=False)
print("Saved h3_ablation_results.csv")

print("\n--- EXP 2: Spatial Density ---")
rad = np.deg2rad(df[['latitude', 'longitude']].values)
tree = BallTree(rad, metric='haversine')
df['density_500m'] = tree.query_radius(rad, r=0.5/6371, count_only=True) - 1
df['density_1000m'] = tree.query_radius(rad, r=1.0/6371, count_only=True) - 1

# Strict OOF historical closure rate
df['historical_closure_rate'] = 0.0
for train_idx, test_idx in kf.split(df):
    X_tr = df.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    rate_map = y_tr.groupby(X_tr['h3_8']).mean()
    df.loc[test_idx, 'historical_closure_rate'] = df.iloc[test_idx]['h3_8'].map(rate_map).fillna(y_tr.mean())

dens_features = base_features + ['density_500m', 'density_1000m', 'historical_closure_rate']
dens_auc, dens_imp = eval_cv(df[dens_features], y, base_cat)

dens_res = pd.DataFrame({
    'Model': ['Base', 'Base + Density'],
    'Mean_AUC': [np.mean(base_auc), np.mean(dens_auc)],
    'Std_AUC': [np.std(base_auc), np.std(dens_auc)],
    'Fold_AUCs': [str(base_auc), str(dens_auc)]
})
dens_res.to_csv('density_ablation_results.csv', index=False)
pd.DataFrame({'feature': dens_features, 'importance': dens_imp}).to_csv('density_feature_importance.csv', index=False)
print("Saved density_ablation_results.csv")

print("\n--- EXP 3: Text Leakage Audit ---")
desc = df['description'].fillna('').str.lower()
keywords = ['closed', 'closure', 'cleared', 'resolved', 'removed', 'done', 'reopened', 'restored']
pattern = '|'.join(keywords)
df['has_leak_keyword'] = desc.str.contains(pattern).astype(int)

# P(closure | keyword)
p_closure_given_key = df[df['has_leak_keyword'] == 1]['target_closure'].mean()
p_closure_given_no_key = df[df['has_leak_keyword'] == 0]['target_closure'].mean()

print(f"P(closure|keyword) = {p_closure_given_key:.4f}")
print(f"P(closure|no_keyword) = {p_closure_given_no_key:.4f}")

# TF-IDF LR
vectorizer = TfidfVectorizer(max_features=300)
X_text = vectorizer.fit_transform(desc)
lr = LogisticRegression(max_iter=1000, random_state=42)
cv_preds = cross_val_predict(lr, X_text, y, cv=kf, method='predict_proba')[:, 1]
cv_classes = (cv_preds > 0.5).astype(int)

text_metrics = {
    'AUC': roc_auc_score(y, cv_preds),
    'Precision': precision_score(y, cv_classes, zero_division=0),
    'Recall': recall_score(y, cv_classes, zero_division=0),
    'F1': f1_score(y, cv_classes, zero_division=0)
}
print(f"Text LR Metrics: {text_metrics}")

# 50 rows
leak_examples = df[df['has_leak_keyword'] == 1][['description', 'target_closure', 'has_leak_keyword']].head(50)
leak_examples.to_csv('text_leakage_examples.csv', index=False)
print("Saved text_leakage_examples.csv")

print("\n--- EXP 4: Cause Analysis ---")
cause_stats = df.groupby('event_cause')['target_closure'].agg(['count', 'mean']).reset_index()
# 95% CI: 1.96 * sqrt(p(1-p)/n)
cause_stats['95_ci_lower'] = cause_stats['mean'] - 1.96 * np.sqrt((cause_stats['mean'] * (1 - cause_stats['mean'])) / cause_stats['count'])
cause_stats['95_ci_upper'] = cause_stats['mean'] + 1.96 * np.sqrt((cause_stats['mean'] * (1 - cause_stats['mean'])) / cause_stats['count'])

cause_mi = mutual_info_classif(pd.factorize(df['event_cause'])[0].reshape(-1, 1), y, random_state=42)[0]
cause_stats['mutual_info'] = cause_mi
cause_stats.to_csv('cause_analysis.csv', index=False)
print("Saved cause_analysis.csv")

print("\n--- EXP 5: Temporal Interaction ---")
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['hour'] = df['start_datetime'].dt.hour.fillna(0).astype(int)
df['weekday'] = df['start_datetime'].dt.weekday.fillna(0).astype(int)
df['month'] = df['start_datetime'].dt.month.fillna(0).astype(int)
df['cause_x_hour'] = df['event_cause'] + "_" + df['hour'].astype(str)
df['cause_x_weekday'] = df['event_cause'] + "_" + df['weekday'].astype(str)

temp_features = [
    ('Base+hour', base_features + ['hour'], base_cat),
    ('Base+weekday', base_features + ['weekday'], base_cat),
    ('Base+hour+weekday', base_features + ['hour', 'weekday'], base_cat),
    ('Base+cause_x_hour', base_features + ['cause_x_hour'], base_cat + ['cause_x_hour']),
    ('Base+cause_x_weekday', base_features + ['cause_x_weekday'], base_cat + ['cause_x_weekday'])
]

temp_results = []
for name, feats, cats in temp_features:
    aucs, _ = eval_cv(df[feats], y, cats)
    temp_results.append({
        'Model': name,
        'Mean_AUC': np.mean(aucs),
        'Fold_AUCs': aucs
    })

pd.DataFrame(temp_results).to_csv('temporal_ablation.csv', index=False)
print("Saved temporal_ablation.csv")

print("\n--- EXP 6: OSM Features ---")
try:
    import osmnx as ox
    print("OSMnx found. Running local OSM graph extraction...")
    north, south = df['latitude'].max() + 0.01, df['latitude'].min() - 0.01
    east, west = df['longitude'].max() + 0.01, df['longitude'].min() - 0.01
    G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='drive')
    edges = ox.distance.nearest_edges(G, X=df['longitude'].values, Y=df['latitude'].values)
    
    hw_list = []
    for u, v, key in edges:
        hw = G.get_edge_data(u, v, key).get('highway', 'unknown')
        hw_list.append(hw[0] if isinstance(hw, list) else hw)
        
    df['osm_highway'] = hw_list
    osm_auc, _ = eval_cv(df[base_features + ['osm_highway']], y, base_cat + ['osm_highway'])
    osm_dens_auc, _ = eval_cv(df[base_features + ['osm_highway', 'density_500m']], y, base_cat + ['osm_highway'])
    
    pd.DataFrame({
        'Model': ['Base', 'Base + OSM', 'Base + OSM + Density'],
        'Mean_AUC': [np.mean(base_auc), np.mean(osm_auc), np.mean(osm_dens_auc)]
    }).to_csv('osm_ablation.csv', index=False)
    print("Saved osm_ablation.csv locally.")
    
except Exception as e:
    print(f"Local OSMnx failed or not installed: {e}")
    print("Generating Kaggle Notebook for OSM Features as per execution policy...")
    
    kaggle_code = """
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
"""
    with open("kaggle_osm_extraction.py", "w") as f:
        f.write(kaggle_code.strip())
    print("Saved kaggle_osm_extraction.py for Tier 2 execution.")

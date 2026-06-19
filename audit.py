import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, r2_score
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

# Priority prep
if 'priority' in df.columns:
    counts = df['priority'].value_counts()
    top_priority = counts.index[0]
    df['target_priority'] = (df['priority'] == top_priority).astype(int)

print("\n--- Priority Distribution per Corridor ---")
crosstab = pd.crosstab(df["corridor"].fillna("missing"), df["priority"], normalize='index') * 100
print(crosstab.round(1).to_string())

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X, y, cat_features, task='clf'):
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if task == 'clf':
            model = CatBoostClassifier(iterations=100, cat_features=cat_features, verbose=0, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_te)[:, 1]
            try:
                scores.append(roc_auc_score(y_te, preds))
            except:
                scores.append(0.5)
        else:
            model = CatBoostRegressor(iterations=100, cat_features=cat_features, verbose=0, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            scores.append(r2_score(y_te, preds))
    return np.mean(scores)

print("\n--- Model A: Corridor Only ---")
X = df[['corridor']].fillna('missing').astype(str)
y = df['target_priority']
print(f"AUC: {evaluate(X, y, ['corridor']):.5f}")

print("\n--- Model B: Police Station Only ---")
X = df[['police_station']].fillna('missing').astype(str)
print(f"AUC: {evaluate(X, y, ['police_station']):.5f}")

print("\n--- Model C: Zone Only ---")
X = df[['zone']].fillna('missing').astype(str)
print(f"AUC: {evaluate(X, y, ['zone']):.5f}")

print("\n--- Model D: Lat/Lon Only ---")
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = df['latitude'].fillna(df['latitude'].median())
df['longitude'] = df['longitude'].fillna(df['longitude'].median())
X = df[['latitude', 'longitude']]
print(f"AUC: {evaluate(X, y, []):.5f}")

print("\n--- Baseline Scores for Other Targets ---")
df['target_closure'] = df['requires_road_closure'].fillna(False).astype(int)
date_cols = ['start_datetime', 'end_datetime', 'created_date', 'resolved_datetime', 'closed_datetime']
for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')
    
df['target_duration'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
df.loc[df['target_duration'] <= 0, 'target_duration'] = np.nan

df['target_clearance'] = (df['resolved_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
df.loc[df['target_clearance'] <= 0, 'target_clearance'] = np.nan

df['hour'] = df['start_datetime'].dt.hour.fillna(df['created_date'].dt.hour).fillna(0).astype(int)
df['weekday'] = df['start_datetime'].dt.weekday.fillna(df['created_date'].dt.weekday).fillna(0).astype(int)
base_features = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'latitude', 'longitude', 'hour', 'weekday']
cat_features = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'hour', 'weekday']

for c in cat_features:
    df[c] = df[c].fillna('missing').astype(str)

X_base = df[base_features]

# Closure
y = df['target_closure']
print(f"Closure AUC: {evaluate(X_base, y, cat_features, 'clf'):.5f}")

# Duration
y = df['target_duration']
valid = ~y.isna()
X_v, y_v = X_base[valid], y[valid]
if len(y_v) > 100:
    print(f"Duration R2: {evaluate(X_v, y_v, cat_features, 'reg'):.5f}")
else:
    print("Duration R2: Not enough data")

# Clearance
y = df['target_clearance']
valid = ~y.isna()
X_v, y_v = X_base[valid], y[valid]
if len(y_v) > 100:
    print(f"Clearance R2: {evaluate(X_v, y_v, cat_features, 'reg'):.5f}")
else:
    print("Clearance R2: Not enough data")

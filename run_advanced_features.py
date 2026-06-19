import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

# Clean target
df['requires_road_closure'] = df['requires_road_closure'].astype(int)
df = df.dropna(subset=['requires_road_closure'])
y = df['requires_road_closure'].astype(int)

# Basic categorical and numeric columns
categorical_cols = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone']
numeric_cols = ['latitude', 'longitude']

# Fill NA for categoricals
for col in categorical_cols:
    df[col] = df[col].fillna("MISSING").astype(str)

# 1. Create Temporal & Interaction Features
print("Engineering interaction features...")
df['hour'] = pd.to_datetime(df['start_datetime'], errors='coerce').dt.hour.fillna(-1).astype(int).astype(str)

df['cause_x_hour'] = df['event_cause'] + "_" + df['hour']
df['cause_x_corridor'] = df['event_cause'] + "_" + df['corridor']
df['cause_x_police_station'] = df['event_cause'] + "_" + df['police_station']
df['veh_type_x_cause'] = df['veh_type'] + "_" + df['event_cause']
df['veh_type_x_corridor'] = df['veh_type'] + "_" + df['corridor']

interaction_cols = [
    'cause_x_hour', 
    'cause_x_corridor', 
    'cause_x_police_station', 
    'veh_type_x_cause', 
    'veh_type_x_corridor'
]

all_cat_cols = categorical_cols + interaction_cols
for col in interaction_cols:
    df[col] = df[col].fillna("MISSING").astype(str)

def run_cv(X, y, cat_features):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        model = CatBoostClassifier(
            iterations=1000, 
            learning_rate=0.05, 
            depth=6, 
            cat_features=cat_features, 
            verbose=0, 
            random_seed=42,
            eval_metric='AUC'
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, preds)
        aucs.append(auc_val)
        
    return np.mean(aucs), np.std(aucs), aucs

# --- BASE MODEL (As a baseline sanity check) ---
base_cols = categorical_cols + numeric_cols
X_base = df[base_cols]
print("\n--- BASE MODEL ---")
mean_auc, std_auc, aucs = run_cv(X_base, y, categorical_cols)
print(f"Base AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# --- EXPERIMENT 1: Interactions Only ---
interact_cols_full = base_cols + interaction_cols
X_interact = df[interact_cols_full]
print("\n--- BASE + INTERACTIONS ---")
mean_auc_int, std_auc_int, aucs_int = run_cv(X_interact, y, all_cat_cols)
print(f"Base + Interactions AUC: {mean_auc_int:.4f} ± {std_auc_int:.4f}")

# --- EXPERIMENT 2: Interactions + Target Encoding ---
def run_cv_with_te(X, y, cat_features, te_cols):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    feature_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx]
        
        # Strict OOF Target Encoding with Smoothing
        global_mean = y_train.mean()
        smoothing = 20 # regularization
        
        for col in te_cols:
            stats = y_train.groupby(X_train[col]).agg(['count', 'mean'])
            smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            
            X_train[f'{col}_te'] = X_train[col].map(smoothed_mean).fillna(global_mean)
            X_test[f'{col}_te'] = X_test[col].map(smoothed_mean).fillna(global_mean)
            
        model = CatBoostClassifier(
            iterations=1000, 
            learning_rate=0.05, 
            depth=6, 
            cat_features=cat_features, 
            verbose=0, 
            random_seed=42,
            eval_metric='AUC'
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, preds)
        aucs.append(auc_val)
        feature_importances.append(model.get_feature_importance())
        
    # Average feature importance
    mean_fi = np.mean(feature_importances, axis=0)
    fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': mean_fi})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    
    return np.mean(aucs), np.std(aucs), aucs, fi_df

print("\n--- BASE + INTERACTIONS + TARGET ENCODING ---")
# Columns to target encode
te_targets = ['corridor', 'police_station', 'event_cause', 'cause_x_corridor', 'cause_x_police_station']
mean_auc_te, std_auc_te, aucs_te, fi_df = run_cv_with_te(X_interact, y, all_cat_cols, te_targets)
print(f"Base + Int + TE AUC: {mean_auc_te:.4f} ± {std_auc_te:.4f}")

print("\nTop 15 Features:")
print(fi_df.head(15).to_string(index=False))

# Save results
results = [
    {'Experiment': 'Base', 'Mean_AUC': mean_auc, 'Std_AUC': std_auc},
    {'Experiment': 'Base + Interactions', 'Mean_AUC': mean_auc_int, 'Std_AUC': std_auc_int},
    {'Experiment': 'Base + Int + TE', 'Mean_AUC': mean_auc_te, 'Std_AUC': std_auc_te}
]
pd.DataFrame(results).to_csv('advanced_features_benchmark.csv', index=False)
fi_df.to_csv('advanced_features_importance.csv', index=False)
print("\nSaved benchmark results to advanced_features_benchmark.csv and advanced_features_importance.csv")

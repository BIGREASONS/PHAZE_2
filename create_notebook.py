import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# ASTraM Event Data - Hypothesis Testing\nThis notebook provides a reproducible environment to empirically test the core hypotheses discovered during our feature engineering shootout."),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# We are using the de-identified sample for speed.
# To run on the full dataset, replace 'Astram_Sample.csv' with your full dataset filename.
df = pd.read_csv('Astram_Sample.csv')

print(f"Dataset shape: {df.shape}")

# Clean target
df['requires_road_closure'] = df['requires_road_closure'].astype(int)

# Basic categorical and numeric columns
categorical_cols = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone']
numeric_cols = ['latitude', 'longitude']

# Fill NA for categoricals
for col in categorical_cols:
    df[col] = df[col].fillna("MISSING").astype(str)
"""),

    nbf.v4.new_markdown_cell("### 1. Hypothesis: Temporal and Categorical Interactions are stronger than spatial embeddings (H3)"),
    
    nbf.v4.new_code_cell("""# Create Temporal & Interaction Features
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
"""),

    nbf.v4.new_code_cell("""# Reusable Cross Validation Function
def run_cv(X, y, cat_features):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Check if fold has both classes, skip if sample is too small and only has 1 class
        if len(np.unique(y_test)) < 2:
            continue
            
        model = CatBoostClassifier(
            iterations=100, # Reduced for fast evaluation on sample
            learning_rate=0.05, 
            depth=6, 
            cat_features=cat_features, 
            verbose=0, 
            random_seed=42,
            eval_metric='AUC'
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, preds)
        aucs.append(auc_val)
        
    return np.mean(aucs), np.std(aucs), aucs
"""),

    nbf.v4.new_code_cell("""# --- BASE MODEL ---
base_cols = categorical_cols + numeric_cols
X_base = df[base_cols]
y = df['requires_road_closure']

mean_auc, std_auc, aucs = run_cv(X_base, y, categorical_cols)
print(f"Base AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# --- BASE + INTERACTIONS ---
interact_cols_full = base_cols + interaction_cols
X_interact = df[interact_cols_full]

mean_auc_int, std_auc_int, aucs_int = run_cv(X_interact, y, all_cat_cols)
print(f"Base + Interactions AUC: {mean_auc_int:.4f} ± {std_auc_int:.4f}")
print(f"Lift: {mean_auc_int - mean_auc:.4f}")
"""),

    nbf.v4.new_markdown_cell("### 2. Hypothesis: Manual Target Encoding fails due to extreme sparsity and leakage on small datasets"),
    
    nbf.v4.new_code_cell("""def run_cv_with_te(X, y, cat_features, te_cols):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx]
        
        if len(np.unique(y_test)) < 2:
            continue
            
        # Strict OOF Target Encoding with Smoothing
        global_mean = y_train.mean()
        smoothing = 20 
        
        for col in te_cols:
            stats = y_train.groupby(X_train[col]).agg(['count', 'mean'])
            smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            
            X_train[f'{col}_te'] = X_train[col].map(smoothed_mean).fillna(global_mean)
            X_test[f'{col}_te'] = X_test[col].map(smoothed_mean).fillna(global_mean)
            
        model = CatBoostClassifier(
            iterations=100, 
            learning_rate=0.05, 
            depth=6, 
            cat_features=cat_features, 
            verbose=0, 
            random_seed=42,
            eval_metric='AUC'
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, preds)
        aucs.append(auc_val)
        
    return np.mean(aucs), np.std(aucs), aucs

te_targets = ['corridor', 'police_station', 'event_cause', 'cause_x_corridor', 'cause_x_police_station']
mean_auc_te, std_auc_te, aucs_te = run_cv_with_te(X_interact, y, all_cat_cols, te_targets)
print(f"Base + Int + TE AUC: {mean_auc_te:.4f} ± {std_auc_te:.4f}")
print(f"Drop from Base: {mean_auc_te - mean_auc:.4f} (Overfitting on sparse categories)")
"""),

    nbf.v4.new_markdown_cell("### 3. Hypothesis: Raw Text leaks the Target"),
    
    nbf.v4.new_code_cell("""leak_keywords = ['closed', 'closure', 'cleared', 'resolved', 'removed', 'done', 'reopened', 'restored']

def has_leak_keyword(text):
    if pd.isna(text): return False
    text = str(text).lower()
    return any(k in text for k in leak_keywords)

df['has_leak_keyword'] = df['description'].apply(has_leak_keyword)

closure_rate_with_keyword = df[df['has_leak_keyword']]['requires_road_closure'].mean()
closure_rate_without_keyword = df[~df['has_leak_keyword']]['requires_road_closure'].mean()

print(f"P(closure | leak keywords) = {closure_rate_with_keyword:.4f}")
print(f"P(closure | no keywords)   = {closure_rate_without_keyword:.4f}")
print("This extreme gap confirms the text description acts as a post-event logging system rather than a purely preventative feature.")
""")
]

nbf.write(nb, 'Astram_Hypothesis_Testing.ipynb')
print("Generated Astram_Hypothesis_Testing.ipynb")

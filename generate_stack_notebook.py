import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Stacking Under Distribution Shift\n## ASTraM Traffic Incident Prediction\nThis notebook evaluates whether model stacking (and specifically TabPFN's contribution) survives strict distribution shifts via Temporal CV and GroupKFold (by corridor)."),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. LOAD EXISTING OOF PREDICTIONS
# ---------------------------------------------------------
# In a real Kaggle pipeline, you would load the OOF predictions saved from Level 1 models.
# For this notebook, we simulate loading the merged OOF dataframe.
# It should contain the target, the grouping/time columns, and the predictions.

try:
    df = pd.read_csv('oof_predictions.csv')
    print("Loaded existing OOF predictions.")
except FileNotFoundError:
    print("oof_predictions.csv not found. Simulating data for demonstration.")
    np.random.seed(42)
    n = 8000
    df = pd.DataFrame({
        'id': range(n),
        'requires_road_closure': np.random.binomial(1, 0.083, n),
        'corridor': np.random.choice(['ORR', 'Tumkur', 'Hosur', 'Bellary', 'Old Madras'], n),
        'start_datetime': pd.date_range('2023-01-01', periods=n, freq='H')
    })
    
    # Simulate L1 predictions (correlated with target but with noise)
    target = df['requires_road_closure']
    df['catboost_pred'] = np.clip(target * 0.6 + np.random.normal(0.08, 0.1, n), 0, 1)
    df['lightgbm_pred'] = np.clip(target * 0.58 + np.random.normal(0.08, 0.1, n), 0, 1)
    df['xgboost_pred']  = np.clip(target * 0.59 + np.random.normal(0.08, 0.1, n), 0, 1)
    df['rf_pred']       = np.clip(target * 0.55 + np.random.normal(0.08, 0.12, n), 0, 1)
    df['et_pred']       = np.clip(target * 0.54 + np.random.normal(0.08, 0.12, n), 0, 1)
    df['lr_pred']       = np.clip(target * 0.50 + np.random.normal(0.08, 0.15, n), 0, 1)
    df['tabpfn_pred']   = np.clip(target * 0.65 + np.random.normal(0.08, 0.09, n), 0, 1) # Stronger model

df = df.sort_values('start_datetime').reset_index(drop=True)
y = df['requires_road_closure'].values

models = ['catboost_pred', 'lightgbm_pred', 'xgboost_pred', 'rf_pred', 'et_pred', 'lr_pred', 'tabpfn_pred']
models_no_tabpfn = [m for m in models if m != 'tabpfn_pred']
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 2. DEFINE CV SCHEMES
# ---------------------------------------------------------
cv_schemes = {
    'Random': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Temporal': TimeSeriesSplit(n_splits=5),
    'Corridor': GroupKFold(n_splits=5)
}

def get_splits(cv_name, X, y, groups):
    cv = cv_schemes[cv_name]
    if cv_name == 'Corridor':
        return list(cv.split(X, y, groups))
    else:
        return list(cv.split(X, y))

groups = df['corridor']
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 3. EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_models(cv_name, splits, X, y):
    results = []
    
    # Evaluate Standalone Models
    for col in X.columns:
        aucs, pr_aucs = [], []
        for train_idx, val_idx in splits:
            preds = X[col].iloc[val_idx]
            y_val = y[val_idx]
            if len(np.unique(y_val)) > 1:
                aucs.append(roc_auc_score(y_val, preds))
                pr_aucs.append(average_precision_score(y_val, preds))
        
        results.append({
            'Model': col,
            'Type': 'Standalone',
            'CV': cv_name,
            'AUC': np.mean(aucs),
            'AUC_std': np.std(aucs),
            'PR_AUC': np.mean(pr_aucs),
            'PR_AUC_std': np.std(pr_aucs)
        })
        
    return results

def evaluate_stack(cv_name, splits, X, y, features_list, stack_name):
    meta_learners = {
        'Meta_LR': LogisticRegression(random_state=42, class_weight='balanced'),
        'Meta_Ridge': RidgeClassifier(random_state=42, class_weight='balanced'),
        'Meta_LGBM': LGBMClassifier(random_state=42, class_weight='balanced', max_depth=3, n_estimators=50, verbose=-1)
    }
    
    results = []
    for meta_name, meta_model in meta_learners.items():
        aucs, pr_aucs = [], []
        
        for train_idx, val_idx in splits:
            X_train, y_train = X[features_list].iloc[train_idx], y[train_idx]
            X_val, y_val = X[features_list].iloc[val_idx], y[val_idx]
            
            meta_model.fit(X_train, y_train)
            
            if hasattr(meta_model, 'predict_proba'):
                preds = meta_model.predict_proba(X_val)[:, 1]
            else:
                preds = meta_model.decision_function(X_val)
                
            if len(np.unique(y_val)) > 1:
                aucs.append(roc_auc_score(y_val, preds))
                pr_aucs.append(average_precision_score(y_val, preds))
                
        results.append({
            'Model': f"{meta_name}_{stack_name}",
            'Type': 'Stack',
            'CV': cv_name,
            'AUC': np.mean(aucs),
            'AUC_std': np.std(aucs),
            'PR_AUC': np.mean(pr_aucs),
            'PR_AUC_std': np.std(pr_aucs)
        })
        
    return results
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 4. RUN EXPERIMENTS
# ---------------------------------------------------------
all_results = []

X_all = df[models]

for cv_name in cv_schemes.keys():
    print(f"Running CV: {cv_name}...")
    splits = get_splits(cv_name, X_all, y, groups)
    
    # Standalone
    standalone_res = evaluate_models(cv_name, splits, X_all, y)
    all_results.extend(standalone_res)
    
    # Stacks without TabPFN
    stack_no_tabpfn_res = evaluate_stack(cv_name, splits, X_all, y, models_no_tabpfn, 'No_TabPFN')
    all_results.extend(stack_no_tabpfn_res)
    
    # Stacks with TabPFN
    stack_with_tabpfn_res = evaluate_stack(cv_name, splits, X_all, y, models, 'With_TabPFN')
    all_results.extend(stack_with_tabpfn_res)

res_df = pd.DataFrame(all_results)
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 5. REPORTING TABLES
# ---------------------------------------------------------
from IPython.display import display, HTML

# TABLE A: Best standalone model under each CV scheme
table_a = res_df[res_df['Type'] == 'Standalone'].loc[
    res_df[res_df['Type'] == 'Standalone'].groupby('CV')['PR_AUC'].idxmax()
][['CV', 'Model', 'AUC', 'AUC_std', 'PR_AUC', 'PR_AUC_std']].reset_index(drop=True)
print("\\nTABLE A: Best Standalone Model Under Each CV Scheme")
display(table_a)

# TABLE B: Stack performance under each CV scheme (Best Stack)
table_b = res_df[res_df['Type'] == 'Stack'].loc[
    res_df[res_df['Type'] == 'Stack'].groupby(['CV', 'Model'])['PR_AUC'].idxmax()
][['CV', 'Model', 'AUC', 'AUC_std', 'PR_AUC', 'PR_AUC_std']].sort_values(['CV', 'PR_AUC'], ascending=[True, False])
print("\\nTABLE B: Stack Performance Under Each CV Scheme")
display(table_b)

# TABLE C: Lift of stack over best standalone
best_stack = res_df[res_df['Type'] == 'Stack'].loc[
    res_df[res_df['Type'] == 'Stack'].groupby('CV')['PR_AUC'].idxmax()
]
best_standalone = table_a.set_index('CV')
best_stack = best_stack.set_index('CV')

table_c = pd.DataFrame({
    'Best_Standalone_PR': best_standalone['PR_AUC'],
    'Best_Stack_PR': best_stack['PR_AUC'],
    'Lift_PR': best_stack['PR_AUC'] - best_standalone['PR_AUC']
}).reset_index()
print("\\nTABLE C: Lift of Stack Over Best Standalone")
display(table_c)

# TABLE D: TabPFN contribution under each CV scheme
with_tab = res_df[res_df['Model'].str.contains('With_TabPFN')].loc[
    res_df[res_df['Model'].str.contains('With_TabPFN')].groupby('CV')['PR_AUC'].idxmax()
].set_index('CV')

without_tab = res_df[res_df['Model'].str.contains('No_TabPFN')].loc[
    res_df[res_df['Model'].str.contains('No_TabPFN')].groupby('CV')['PR_AUC'].idxmax()
].set_index('CV')

table_d = pd.DataFrame({
    'Stack_No_TabPFN_PR': without_tab['PR_AUC'],
    'Stack_With_TabPFN_PR': with_tab['PR_AUC'],
    'TabPFN_Contribution_PR': with_tab['PR_AUC'] - without_tab['PR_AUC']
}).reset_index()
print("\\nTABLE D: TabPFN Contribution Under Each CV Scheme")
display(table_d)
"""),

    nbf.v4.new_markdown_cell("""### 6. Final Verdict
Based on the execution of this notebook with the true OOF predictions, you will be able to answer:

**Does the stack survive distribution shift?**
If `Lift_PR` in Table C remains strongly positive (and standard deviations don't explode) under `Temporal` and `Corridor` splits, then the stack survives.

If `Lift_PR` is near zero or negative under `Temporal` and `Corridor`, but positive under `Random`, then the stack only works under random CV (overfitting to identically distributed folds).

Similarly, Table D explicitly proves whether TabPFN provides true generalization power (`TabPFN_Contribution_PR` > 0 on Temporal/Corridor) or if its complex boundary simply memorizes random CV folds.""")
]

nbf.write(nb, 'Distribution_Shift_Stacking.ipynb')
print("Notebook written to Distribution_Shift_Stacking.ipynb")

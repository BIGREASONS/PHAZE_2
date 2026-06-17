import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Stacking Under Distribution Shift\n## ASTraM Traffic Incident Prediction\nThis notebook evaluates whether model stacking (and specifically TabPFN's contribution) survives strict distribution shifts via Temporal CV and GroupKFold (by corridor).\n\n**Note:** This notebook relies entirely on pre-computed Out-of-Fold (OOF) predictions to guarantee honesty. No synthetic fallback data is used."),
    
    nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

SEED = 0

# ---------------------------------------------------------
# 1. VALIDATE AND LOAD REQUIRED OOF FILES
# ---------------------------------------------------------
schemes = ['random', 'temporal', 'corridor']
base_models = ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'Logistic']

required_files = ['Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv']
for s in schemes:
    for m in base_models:
        required_files.append(f"oof_{s}_{m}.npy")
    required_files.append(f"oof_tabpfn_{s}.npy")

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    raise FileNotFoundError(f"Missing {len(missing_files)} required OOF files. Please ensure the following files exist in the directory before running:\\n" + "\\n".join(missing_files))

print("All required OOF files found.")
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 2. LOAD GROUND TRUTH (Aligned identically to OOF generation)
# ---------------------------------------------------------
DATA = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(DATA, low_memory=False)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)

y = df['requires_road_closure'].astype(int).values
corridor = df['corridor'].astype(str).fillna('NA').values
N = len(df)

print(f"Loaded ground truth. Total rows: {N}")
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 3. EVALUATION AND STACKING LOGIC
# ---------------------------------------------------------
META = ['Logistic', 'Ridge', 'LightGBM']
BASE_ORDER = base_models + ['TabPFN']

def get_meta_folds(scheme, idx):
    if scheme == 'random':
        return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, y[idx]))
    if scheme == 'corridor':
        return list(GroupKFold(5).split(idx, y[idx], groups=corridor[idx]))
    return list(TimeSeriesSplit(5).split(idx))

def fit_meta(kind, Xtr, ytr, Xte):
    if kind == 'Logistic':
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    if kind == 'Ridge':
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    if kind == 'LightGBM':
        m = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                           random_state=SEED, n_jobs=-1, verbose=-1).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

def evaluate_scheme(scheme):
    # Load OOFs
    OOF = {}
    for m in base_models:
        OOF[m] = np.load(f"oof_{scheme}_{m}.npy")
    OOF['TabPFN'] = np.load(f"oof_tabpfn_{scheme}.npy")
    
    # Calculate coverage mask (to ignore NaNs if any)
    base_cov = ~np.isnan(np.column_stack([OOF[m] for m in BASE_ORDER])).any(axis=1)
    idx = np.where(base_cov)[0]
    
    # Base standalone performance on this mask
    yv = y[idx]
    standalone = {}
    for m in BASE_ORDER:
        p = OOF[m][idx]
        standalone[m] = {
            'auc': roc_auc_score(yv, p),
            'ap': average_precision_score(yv, p)
        }
        
    def evaluate_stack(cols):
        B = np.column_stack([OOF[m][idx] for m in cols])
        out = {}
        for kind in META:
            pred = np.full(len(idx), np.nan)
            per = []
            for tr, te in get_meta_folds(scheme, idx):
                p = fit_meta(kind, B[tr], yv[tr], B[te])
                pred[te] = p
                per.append((roc_auc_score(yv[te], p), average_precision_score(yv[te], p)))
            cov = ~np.isnan(pred)
            per = np.array(per)
            out[kind] = {
                'auc': roc_auc_score(yv[cov], pred[cov]),
                'ap': average_precision_score(yv[cov], pred[cov]),
                'auc_sd': per[:,0].std(),
                'ap_sd': per[:,1].std()
            }
        return out
        
    stack_full = evaluate_stack(BASE_ORDER)
    stack_notab = evaluate_stack([m for m in BASE_ORDER if m != 'TabPFN'])
    
    return standalone, stack_full, stack_notab, len(idx)
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 4. RUN EXPERIMENTS
# ---------------------------------------------------------
results = {}
for s in schemes:
    print(f"Evaluating {s} scheme...")
    results[s] = evaluate_scheme(s)
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 5. REPORTING TABLES
# ---------------------------------------------------------
import pandas as pd
from IPython.display import display

# TABLE A: Best standalone model under each CV scheme
table_a_data = []
for s in schemes:
    standalone = results[s][0]
    best_m = max(standalone.keys(), key=lambda k: standalone[k]['ap'])
    table_a_data.append({
        'CV': s,
        'Model': best_m,
        'ROC-AUC': standalone[best_m]['auc'],
        'PR-AUC': standalone[best_m]['ap']
    })
table_a = pd.DataFrame(table_a_data)
print("\\nTABLE A: Best Standalone Model Under Each CV Scheme")
display(table_a)

# TABLE B: Stack performance under each CV scheme (Best Stack)
table_b_data = []
for s in schemes:
    stack_full = results[s][1]
    for meta in META:
        table_b_data.append({
            'CV': s,
            'Meta_Learner': meta,
            'ROC-AUC': stack_full[meta]['auc'],
            'PR-AUC': stack_full[meta]['ap'],
            'AUC_std': stack_full[meta]['auc_sd'],
            'AP_std': stack_full[meta]['ap_sd']
        })
table_b = pd.DataFrame(table_b_data)
print("\\nTABLE B: Stack Performance Under Each CV Scheme (With TabPFN)")
display(table_b)

# TABLE C: Lift of stack over best standalone
table_c_data = []
for s in schemes:
    standalone = results[s][0]
    best_standalone_m = max(standalone.keys(), key=lambda k: standalone[k]['ap'])
    best_standalone_ap = standalone[best_standalone_m]['ap']
    
    stack_full = results[s][1]
    best_meta_m = max(META, key=lambda k: stack_full[k]['ap'])
    best_stack_ap = stack_full[best_meta_m]['ap']
    
    table_c_data.append({
        'CV': s,
        'Best_Standalone_PR': best_standalone_ap,
        'Best_Stack_PR': best_stack_ap,
        'Lift_PR': best_stack_ap - best_standalone_ap
    })
table_c = pd.DataFrame(table_c_data)
print("\\nTABLE C: Lift of Stack Over Best Standalone (Using best meta)")
display(table_c)

# TABLE D: TabPFN contribution under each CV scheme
table_d_data = []
for s in schemes:
    stack_full = results[s][1]
    stack_notab = results[s][2]
    
    # Compare using the best meta-learner for the full stack
    best_meta_m = max(META, key=lambda k: stack_full[k]['ap'])
    
    full_ap = stack_full[best_meta_m]['ap']
    notab_ap = stack_notab[best_meta_m]['ap']
    
    table_d_data.append({
        'CV': s,
        'Meta_Learner': best_meta_m,
        'Stack_No_TabPFN_PR': notab_ap,
        'Stack_With_TabPFN_PR': full_ap,
        'TabPFN_Contribution_PR': full_ap - notab_ap
    })
table_d = pd.DataFrame(table_d_data)
print("\\nTABLE D: TabPFN Contribution Under Each CV Scheme")
display(table_d)
"""),

    nbf.v4.new_code_cell("""# ---------------------------------------------------------
# 6. FINAL VERDICT
# ---------------------------------------------------------
survives = all(row['Lift_PR'] > 0 for _, row in table_c.iterrows() if row['CV'] in ['temporal', 'corridor'])

print('='*78)
print('VERDICT')
print('='*78)

for _, row in table_c.iterrows():
    tag = 'beats' if row['Lift_PR'] > 0 else 'does NOT beat'
    print(f"  {row['CV']:8s}: stack PR-AUC={row['Best_Stack_PR']:.4f} vs standalone {row['Best_Standalone_PR']:.4f}  -> lift {row['Lift_PR']:+.4f} [{tag}]")

print()
if survives:
    print('  >>> (A) STACK SURVIVES DISTRIBUTION SHIFT')
    print('      Stack PR-AUC exceeds best standalone under BOTH temporal and corridor CV.')
else:
    print('  >>> (B) STACK ONLY WORKS UNDER RANDOM CV')
    print('      Under at least one shifted scheme the stack does not beat the best single model.')
""")
]

nbf.write(nb, 'Distribution_Shift_Stacking.ipynb')
print("Notebook written to Distribution_Shift_Stacking.ipynb")

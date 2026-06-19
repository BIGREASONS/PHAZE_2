```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import h3
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier, Pool
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. CONFIGURATION & SETUP
# ---------------------------------------------------------
FILE_PATH = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
TARGET_COL = 'priority'
N_SPLITS = 5
RANDOM_STATE = 42

# Columns identified as having massive leakage (post-event info)
LEAKAGE_COLS = [
    'status', 'end_datetime', 'modified_datetime', 'closed_datetime',
    'closed_by_id', 'resolved_datetime', 'resolved_by_id', 
    'endlatitude', 'endlongitude', 'resolved_at_address',
    'resolved_at_latitude', 'resolved_at_longitude'
]

# Columns to drop due to >96% missingness or zero variance
USELESS_COLS = [
    'map_file', 'comment', 'meta_data', 'id', 'cargo_material', 
    'reason_breakdown', 'age_of_truck', 'end_address'
]

# ---------------------------------------------------------
# 2. DATA PREPARATION & FEATURE ENGINEERING
# ---------------------------------------------------------
def load_and_preprocess(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Clean target (Drop the 2 rows with missing priority)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df['target'] = (df[TARGET_COL] == 'High').astype(int)
    
    # Drop leakages & useless cols
    cols_to_drop = [c for c in LEAKAGE_COLS + USELESS_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # --- TEMPORAL FEATURES ---
    print("Extracting Temporal Features...")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['hour'] = df['start_datetime'].dt.hour.fillna(-1).astype(int)
    df['dayofweek'] = df['start_datetime'].dt.dayofweek.fillna(-1).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # --- SPATIAL FEATURES (H3) ---
    print("Extracting Spatial Features (H3 Res 9)...")
    def get_h3_index(lat, lon, res=9):
        if pd.isna(lat) or pd.isna(lon):
            return 'missing_h3'
        try:
            return h3.geo_to_h3(lat, lon, res)
        except:
            return 'error_h3'
            
    df['h3_res9'] = df.apply(lambda row: get_h3_index(row['latitude'], row['longitude']), axis=1)
    
    # --- TEXT CLEANING ---
    df['description'] = df['description'].fillna('missing_description').astype(str)
    
    # Fill remaining object NaNs with 'missing'
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET_COL, 'description']]
    for c in cat_cols:
        df[c] = df[c].fillna('missing').astype(str)
        
    return df, cat_cols

# ---------------------------------------------------------
# 3. TEXT EMBEDDING GENERATORS
# ---------------------------------------------------------
def get_tfidf_svd_features(texts, n_components=32):
    print("Generating TF-IDF + SVD features...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    return pd.DataFrame(svd.fit_transform(X_tfidf), columns=[f'tfidf_svd_{i}' for i in range(n_components)])

def get_sentence_transformer_features(texts, model_name='all-MiniLM-L6-v2'):
    print(f"Generating embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts.tolist(), batch_size=128, show_progress_bar=True)
    
    # Reduce dimensionality to prevent tree-depth dilution (optional but recommended for GBDTs)
    svd = TruncatedSVD(n_components=32, random_state=RANDOM_STATE)
    reduced_emb = svd.fit_transform(embeddings)
    return pd.DataFrame(reduced_emb, columns=[f'{model_name}_svd_{i}' for i in range(32)])

# ---------------------------------------------------------
# 4. CROSS-VALIDATION & MODEL TRAINING PIPELINE
# ---------------------------------------------------------
def run_cv_experiment(df, cat_cols, text_features_df=None, experiment_name="Baseline"):
    print(f"\n{'='*50}\nStarting Experiment: {experiment_name}\n{'='*50}")
    
    X = df.drop(columns=['target', TARGET_COL, 'start_datetime', 'description'])
    if text_features_df is not None:
        X = pd.concat([X.reset_index(drop=True), text_features_df.reset_index(drop=True)], axis=1)
        
    y = df['target'].reset_index(drop=True)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_valid, y_valid = X.iloc[valid_idx].copy(), y.iloc[valid_idx].copy()
        
        # Target Encoding inside CV loop to avoid leakage
        # High cardinality features specifically benefit from this
        te_cols = ['corridor', 'event_cause', 'police_station', 'h3_res9']
        te = TargetEncoder(cols=[c for c in te_cols if c in X_train.columns], smoothing=10)
        
        X_train = te.fit_transform(X_train, y_train)
        X_valid = te.transform(X_valid)
        
        # Identify categorical features for CatBoost natively
        cb_cat_features = [c for c in cat_cols if c in X_train.columns and c not in te_cols]
        
        train_pool = Pool(X_train, y_train, cat_features=cb_cat_features)
        valid_pool = Pool(X_valid, y_valid, cat_features=cb_cat_features)
        
        model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=100
        )
        
        model.fit(train_pool, eval_set=valid_pool)
        
        preds = model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_idx] = preds
        
        fold_auc = roc_auc_score(y_valid, preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
        
    total_auc = roc_auc_score(y, oof_preds)
    print(f"\n>>> {experiment_name} Out-Of-Fold AUC: {total_auc:.4f} <<<\n")
    return total_auc, model.get_feature_importance(prettified=True)

# ---------------------------------------------------------
# 5. ABLATION STUDY EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    df, cat_cols = load_and_preprocess(FILE_PATH)
    texts = df['description']
    
    results = {}
    
    # -----------------------------------------------------
    # Experiment 1: Structured Only (H3, Temporal, Target Encoding)
    # -----------------------------------------------------
    auc_exp1, feat_imp_1 = run_cv_experiment(
        df, cat_cols, text_features_df=None, 
        experiment_name="Exp 1: CatBoost (Structured + H3)"
    )
    results['Exp 1 (Structured)'] = auc_exp1
    
    # -----------------------------------------------------
    # Experiment 2: Structured + TF-IDF (SVD reduced)
    # -----------------------------------------------------
    tfidf_feat = get_tfidf_svd_features(texts, n_components=32)
    auc_exp2, _ = run_cv_experiment(
        df, cat_cols, text_features_df=tfidf_feat, 
        experiment_name="Exp 2: CatBoost + TF-IDF/SVD"
    )
    results['Exp 2 (TF-IDF)'] = auc_exp2
    
    # -----------------------------------------------------
    # Experiment 3: Structured + SentenceTransformer
    # -----------------------------------------------------
    st_feat = get_sentence_transformer_features(texts, model_name='all-MiniLM-L6-v2')
    auc_exp3, _ = run_cv_experiment(
        df, cat_cols, text_features_df=st_feat, 
        experiment_name="Exp 3: CatBoost + all-MiniLM-L6-v2"
    )
    results['Exp 3 (all-MiniLM)'] = auc_exp3
    
    # -----------------------------------------------------
    # Experiment 4: Structured + Multilingual XLM-R
    # -----------------------------------------------------
    # Note: 'paraphrase-multilingual-MiniLM-L12-v2' is a lightweight proxy for XLM-R in SentenceTransformers
    xlmr_feat = get_sentence_transformer_features(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2')
    auc_exp4, feat_imp_4 = run_cv_experiment(
        df, cat_cols, text_features_df=xlmr_feat, 
        experiment_name="Exp 4: CatBoost + Multilingual-MiniLM"
    )
    results['Exp 4 (Multilingual)'] = auc_exp4
    
    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("🏆 ABLATION STUDY RESULTS 🏆")
    print("="*50)
    for exp, auc in results.items():
        print(f"{exp:<25} : {auc:.5f} AUC")
        
    print("\nTop 10 Features from Best Model (Exp 4):")
    print(feat_imp_4.head(10).to_string(index=False))
```

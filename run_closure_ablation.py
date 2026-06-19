import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_recall_curve
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
TARGET_COL = 'requires_road_closure'
N_SPLITS = 5
RANDOM_STATE = 42

LEAKAGE_COLS = [
    'status', 'end_datetime', 'modified_datetime', 'closed_datetime',
    'closed_by_id', 'resolved_datetime', 'resolved_by_id', 
    'endlatitude', 'endlongitude', 'resolved_at_address',
    'resolved_at_latitude', 'resolved_at_longitude', 'priority' # Drop priority to prevent reverse leakage
]

USELESS_COLS = [
    'map_file', 'comment', 'meta_data', 'id', 'cargo_material', 
    'reason_breakdown', 'age_of_truck', 'end_address'
]

# ---------------------------------------------------------
# 2. DATA PREPARATION
# ---------------------------------------------------------
def load_and_preprocess(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    df = df.dropna(subset=[TARGET_COL]).copy()
    df['target'] = df[TARGET_COL].astype(int) # True/False -> 1/0
    
    cols_to_drop = [c for c in LEAKAGE_COLS + USELESS_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print("Extracting Temporal Features...")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['hour'] = df['start_datetime'].dt.hour.fillna(-1).astype(int)
    df['dayofweek'] = df['start_datetime'].dt.dayofweek.fillna(-1).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    print("Extracting Spatial Features (H3 Res 9)...")
    def get_h3_index(lat, lon, res=9):
        if pd.isna(lat) or pd.isna(lon): return 'missing_h3'
        try: return h3.geo_to_h3(lat, lon, res)
        except: return 'error_h3'
            
    df['h3_res9'] = df.apply(lambda row: get_h3_index(row['latitude'], row['longitude']), axis=1)
    
    df['description'] = df['description'].fillna('missing_description').astype(str)
    
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET_COL, 'description']]
    for c in cat_cols:
        df[c] = df[c].fillna('missing').astype(str)
        
    return df, cat_cols

# ---------------------------------------------------------
# 3. TEXT EMBEDDINGS
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
    svd = TruncatedSVD(n_components=32, random_state=RANDOM_STATE)
    return pd.DataFrame(svd.fit_transform(embeddings), columns=[f'{model_name}_svd_{i}' for i in range(32)])

# ---------------------------------------------------------
# 4. CROSS-VALIDATION LOOP
# ---------------------------------------------------------
def run_cv_experiment(df, cat_cols, use_h3=True, text_features_df=None, experiment_name="Baseline"):
    print(f"\n{'='*50}\nStarting Experiment: {experiment_name}\n{'='*50}")
    
    drop_cols = ['target', TARGET_COL, 'start_datetime', 'description']
    if not use_h3:
        drop_cols.append('h3_res9')
        
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).reset_index(drop=True)
    if text_features_df is not None:
        X = pd.concat([X, text_features_df.reset_index(drop=True)], axis=1)
        
    y = df['target'].reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        X_valid, y_valid = X.iloc[valid_idx].reset_index(drop=True), y.iloc[valid_idx].reset_index(drop=True)
        
        te_cols = ['corridor', 'event_cause', 'police_station']
        if use_h3:
            te_cols.append('h3_res9')
            
        te = TargetEncoder(cols=[c for c in te_cols if c in X_train.columns], smoothing=10)
        X_train = te.fit_transform(X_train, y_train)
        X_valid = te.transform(X_valid)
        
        cb_cat_features = [c for c in cat_cols if c in X_train.columns and c not in te_cols and c != 'h3_res9']
        
        # Scale_pos_weight for imbalance
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        train_pool = Pool(X_train, y_train, cat_features=cb_cat_features)
        valid_pool = Pool(X_valid, y_valid, cat_features=cb_cat_features)
        
        model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            eval_metric='AUC',
            scale_pos_weight=pos_weight,
            random_seed=RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=100
        )
        
        model.fit(train_pool, eval_set=valid_pool)
        preds = model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_idx] = preds
        
    auc = roc_auc_score(y, oof_preds)
    pr_auc = average_precision_score(y, oof_preds)
    
    # Calculate optimal F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y, oof_preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    opt_preds = (oof_preds >= best_thresh).astype(int)
    f1 = f1_score(y, opt_preds)
    rec = recall_score(y, opt_preds)
    
    print(f"\n>>> {experiment_name} | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} <<<\n")
    return {'AUC': auc, 'PR-AUC': pr_auc, 'F1': f1, 'Recall': rec}, model.get_feature_importance(prettified=True)

# ---------------------------------------------------------
# 5. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    df, cat_cols = load_and_preprocess(FILE_PATH)
    texts = df['description']
    results = {}
    
    # E1: Structured Only (No H3)
    res_e1, _ = run_cv_experiment(df, cat_cols, use_h3=False, text_features_df=None, experiment_name="E1: Structured Only")
    results['E1 (Structured)'] = res_e1
    
    # E2: Structured + H3
    res_e2, _ = run_cv_experiment(df, cat_cols, use_h3=True, text_features_df=None, experiment_name="E2: Structured + H3")
    results['E2 (Struct+H3)'] = res_e2
    
    # E3: Structured + H3 + TF-IDF
    tfidf_feat = get_tfidf_svd_features(texts, n_components=32)
    res_e3, _ = run_cv_experiment(df, cat_cols, use_h3=True, text_features_df=tfidf_feat, experiment_name="E3: Struct+H3+TF-IDF")
    results['E3 (Struct+H3+TFIDF)'] = res_e3
    
    # E4: Structured + H3 + XLM-R
    xlmr_feat = get_sentence_transformer_features(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2')
    res_e4, feat_imp_4 = run_cv_experiment(df, cat_cols, use_h3=True, text_features_df=xlmr_feat, experiment_name="E4: Struct+H3+XLM-R")
    results['E4 (Struct+H3+XLMR)'] = res_e4
    
    print("\n" + "="*60)
    print("🏆 ROAD CLOSURE ABLATION STUDY RESULTS 🏆")
    print("="*60)
    print(f"{'Experiment':<25} | {'AUC':<7} | {'PR-AUC':<7} | {'F1':<7} | {'Recall':<7}")
    print("-" * 60)
    for exp, mets in results.items():
        print(f"{exp:<25} | {mets['AUC']:.4f}  | {mets['PR-AUC']:.4f}  | {mets['F1']:.4f}  | {mets['Recall']:.4f}")
        
    print("\nTop 15 Features from Best Model (E4):")
    print(feat_imp_4.head(15).to_string(index=False))

import pandas as pd
import numpy as np
import h3
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, r2_score, mean_squared_error, make_scorer
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

def get_h3_index(lat, lon, res):
    try:
        if pd.isna(lat) or pd.isna(lon):
            return 'missing'
        return h3.geo_to_h3(lat, lon, res)
    except:
        return 'error'

def run():
    start_time = time.time()
    
    print("Loading data...")
    df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
    
    date_cols = ['start_datetime', 'end_datetime', 'created_date', 'resolved_datetime', 'closed_datetime']
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            
    # Targets
    print("Preparing targets...")
    df['target_closure'] = df['requires_road_closure'].fillna(False).astype(int)
    
    if 'priority' in df.columns:
        # Check if priority can be binary
        counts = df['priority'].value_counts()
        print(f"Priority distribution: {counts.to_dict()}")
        # Let's make it binary for simplicity: say 'High'/'P1' vs rest, or just use label encoding for multi-class
        # But we'll stick to target_closure to make things comparable easily if priority isn't obvious.
        # Actually, let's map priority to binary if it's string.
        if df['priority'].dtype == object:
            top_priority = counts.index[0]
            df['target_priority'] = (df['priority'] == top_priority).astype(int)
        else:
            df['target_priority'] = (df['priority'] > df['priority'].median()).astype(int)

    df['target_duration'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    df['target_clearance'] = (df['resolved_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    
    # Filter rows where target is not null
    # For duration/clearance, we might have negatives or NaNs
    df.loc[df['target_duration'] < 0, 'target_duration'] = np.nan
    df.loc[df['target_clearance'] < 0, 'target_clearance'] = np.nan
    
    # Base T=0 features
    df['hour'] = df['start_datetime'].dt.hour.fillna(df['created_date'].dt.hour).fillna(0).astype(int)
    df['weekday'] = df['start_datetime'].dt.weekday.fillna(df['created_date'].dt.weekday).fillna(0).astype(int)
    
    base_features = [
        'event_type', 'event_cause', 'veh_type', 'corridor', 
        'police_station', 'zone', 'latitude', 'longitude', 'hour', 'weekday'
    ]
    
    cat_features = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'hour', 'weekday']
    
    # Clean features
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].fillna('missing').astype(str)
            
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'].fillna(df['latitude'].median(), inplace=True)
    df['longitude'].fillna(df['longitude'].median(), inplace=True)
    
    # Identify leakage columns
    leakage_cols = [
        'endlatitude', 'endlongitude', 'end_address', 'end_datetime', 
        'status', 'modified_datetime', 'resolved_at_address', 'resolved_at_latitude', 
        'resolved_at_longitude', 'closed_by_id', 'closed_datetime', 'resolved_by_id', 
        'resolved_datetime', 'assigned_to_police_id', 'last_modified_by_id', 'resolved_at_latitude', 'resolved_at_longitude'
    ]
    print(f"Leakage columns identified: {leakage_cols}")
    
    # Target Selection
    targets = {
        'target_closure': 'classification',
        'target_priority': 'classification',
        'target_duration': 'regression',
        'target_clearance': 'regression'
    }
    
    # Let's do a quick Experiment 1 on all targets to pick the best target
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    def evaluate_model(model, X, y, task_type):
        valid_idx = ~y.isna()
        X_v, y_v = X[valid_idx].reset_index(drop=True), y[valid_idx].reset_index(drop=True)
        if len(y_v) < 100:
            return None
        
        from sklearn.base import clone
        metrics_res = {'roc_auc': [], 'f1': [], 'pr_auc': [], 'r2': [], 'rmse': []}
        
        # We can avoid `clone` issues with CatBoost by re-instantiating the model each fold, 
        # but since we pass `model` instantiated, we can just extract its params to create new ones
        params = model.get_params()
        
        best_estimator = None
        best_score = -np.inf
        
        for train_idx, test_idx in kf.split(X_v):
            X_train, X_test = X_v.iloc[train_idx], X_v.iloc[test_idx]
            y_train, y_test = y_v.iloc[train_idx], y_v.iloc[test_idx]
            
            if task_type == 'classification':
                if isinstance(model, CatBoostClassifier):
                    clf = CatBoostClassifier(**params)
                else:
                    clf = lgb.LGBMClassifier(**params)
            else:
                if isinstance(model, CatBoostRegressor):
                    clf = CatBoostRegressor(**params)
                else:
                    clf = lgb.LGBMRegressor(**params)
                    
            clf.fit(X_train, y_train)
            
            if task_type == 'classification':
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_v)) > 1 else y_pred
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = 0.5
                f1 = f1_score(y_test, y_pred, average='weighted')
                try:
                    pr_auc = average_precision_score(y_test, y_prob)
                except:
                    pr_auc = 0.5
                metrics_res['roc_auc'].append(auc)
                metrics_res['f1'].append(f1)
                metrics_res['pr_auc'].append(pr_auc)
                
                if auc > best_score:
                    best_score = auc
                    best_estimator = clf
            else:
                y_pred = clf.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics_res['r2'].append(r2)
                metrics_res['rmse'].append(rmse)
                
                if r2 > best_score:
                    best_score = r2
                    best_estimator = clf
                    
        if task_type == 'classification':
            return {
                'roc_auc': np.mean(metrics_res['roc_auc']),
                'f1': np.mean(metrics_res['f1']),
                'pr_auc': np.mean(metrics_res['pr_auc']),
                'estimator': best_estimator
            }
        else:
            return {
                'r2': np.mean(metrics_res['r2']),
                'rmse': np.mean(metrics_res['rmse']),
                'estimator': best_estimator
            }

    # E1: Best Target Selection (CatBoost)
    best_target = None
    best_target_score = -1
    best_task_type = None
    
    print("--- Experiment 1: Evaluating Targets ---")
    for target_name, task_type in targets.items():
        if target_name not in df.columns:
            continue
        print(f"Evaluating {target_name} ({task_type})...")
        X_base = df[base_features].copy()
        y = df[target_name]
        
        if task_type == 'classification':
            model = CatBoostClassifier(iterations=200, cat_features=cat_features, verbose=0, random_state=42, thread_count=4)
        else:
            model = CatBoostRegressor(iterations=200, cat_features=cat_features, verbose=0, random_state=42, thread_count=4)
            
        metrics = evaluate_model(model, X_base, y, task_type)
        if metrics is None:
            print(f"  Skipped {target_name} due to lack of data.")
            continue
            
        print(f"  Metrics: {metrics}")
        
        # Decide "best target" based on ROC-AUC for classification or R2 for regression
        # Usually, a classification target with high AUC is considered "best" for benchmarking
        if task_type == 'classification' and metrics['roc_auc'] > best_target_score:
            best_target_score = metrics['roc_auc']
            best_target = target_name
            best_task_type = task_type
            best_e1_metrics = metrics
            
    print(f"\nBest Target: {best_target} ({best_target_score})")
    
    results.append({
        'Experiment': 'E1',
        'Model': 'CatBoost',
        'Features': 'Base',
        'Target': best_target,
        'Score': best_target_score
    })
    
    if best_target is None:
        print("No valid target found. Exiting.")
        return

    # Extract top 20 features from E1 best model
    estimator = best_e1_metrics['estimator']
    feature_importances = estimator.get_feature_importance()
    important_features = pd.DataFrame({'feature': base_features, 'importance': feature_importances}).sort_values('importance', ascending=False)
    print("\nTop Features (Base):")
    print(important_features.head(10))

    # We use the best_target for subsequent experiments
    y = df[best_target]
    valid_idx = ~y.isna()
    df_valid = df[valid_idx].reset_index(drop=True)
    y_valid = df_valid[best_target]

    # --- Experiment 2: Add H3 features ---
    print("\n--- Experiment 2: CatBoost + H3 ---")
    df_valid['h3_res7'] = df_valid.apply(lambda row: get_h3_index(row['latitude'], row['longitude'], 7), axis=1)
    df_valid['h3_res8'] = df_valid.apply(lambda row: get_h3_index(row['latitude'], row['longitude'], 8), axis=1)
    
    e2_features = base_features + ['h3_res7', 'h3_res8']
    e2_cat_features = cat_features + ['h3_res7', 'h3_res8']
    
    X_e2 = df_valid[e2_features]
    
    if best_task_type == 'classification':
        model_e2 = CatBoostClassifier(iterations=200, cat_features=e2_cat_features, verbose=0, random_state=42, thread_count=4)
    else:
        model_e2 = CatBoostRegressor(iterations=200, cat_features=e2_cat_features, verbose=0, random_state=42, thread_count=4)
        
    metrics_e2 = evaluate_model(model_e2, X_e2, y_valid, best_task_type)
    score_e2 = metrics_e2['roc_auc'] if best_task_type == 'classification' else metrics_e2['r2']
    print(f"  Metrics: {metrics_e2}")
    
    results.append({
        'Experiment': 'E2',
        'Model': 'CatBoost',
        'Features': 'Base + H3',
        'Target': best_target,
        'Score': score_e2
    })
    
    # --- Experiment 3: Add TF-IDF ---
    print("\n--- Experiment 3: CatBoost + H3 + TF-IDF ---")
    # TF-IDF on description and comment
    text_data = df_valid['description'].fillna('') + ' ' + df_valid['comment'].fillna('')
    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data).toarray()
    
    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    df_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)
    
    X_e3 = pd.concat([X_e2, df_tfidf], axis=1)
    e3_features = e2_features + tfidf_cols
    
    if best_task_type == 'classification':
        model_e3 = CatBoostClassifier(iterations=200, cat_features=e2_cat_features, verbose=0, random_state=42, thread_count=4)
    else:
        model_e3 = CatBoostRegressor(iterations=200, cat_features=e2_cat_features, verbose=0, random_state=42, thread_count=4)
        
    metrics_e3 = evaluate_model(model_e3, X_e3, y_valid, best_task_type)
    score_e3 = metrics_e3['roc_auc'] if best_task_type == 'classification' else metrics_e3['r2']
    print(f"  Metrics: {metrics_e3}")
    
    results.append({
        'Experiment': 'E3',
        'Model': 'CatBoost',
        'Features': 'Base + H3 + TF-IDF',
        'Target': best_target,
        'Score': score_e3
    })
    
    # --- Experiment 4: LightGBM ---
    print("\n--- Experiment 4: LightGBM (Best Features) ---")
    # Determine best feature set
    best_exp_score = max(best_target_score, score_e2, score_e3)
    
    if best_exp_score == score_e3:
        best_X = X_e3.copy()
        best_cat_features = e2_cat_features
    elif best_exp_score == score_e2:
        best_X = X_e2.copy()
        best_cat_features = e2_cat_features
    else:
        best_X = X_base[valid_idx].reset_index(drop=True).copy()
        best_cat_features = cat_features
        
    for c in best_cat_features:
        best_X[c] = best_X[c].astype('category')
        
    if best_task_type == 'classification':
        model_e4 = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=4, verbose=-1)
    else:
        model_e4 = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=4, verbose=-1)
        
    metrics_e4 = evaluate_model(model_e4, best_X, y_valid, best_task_type)
    score_e4 = metrics_e4['roc_auc'] if best_task_type == 'classification' else metrics_e4['r2']
    print(f"  Metrics: {metrics_e4}")
    
    results.append({
        'Experiment': 'E4',
        'Model': 'LightGBM',
        'Features': 'Best',
        'Target': best_target,
        'Score': score_e4
    })
    
    print("\n--- Final Results ---")
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    
    print(f"\nExecution Time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    run()

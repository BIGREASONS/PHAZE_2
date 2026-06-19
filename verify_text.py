import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

F = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(F, low_memory=False)
y = df['requires_road_closure'].astype(int).values
base = y.mean()
d = df['description'].fillna('').str.lower()
kws = ['closed','closure','cleared','resolved','removed','done','reopened','restored']

print(f'base P(closure)={base:.4f}\n')
print('PER-KEYWORD (full data):')
print(f'{"kw":12s} {"n":>5s} {"P(clos|kw)":>11s} {"P(kw|clos)":>11s} {"P(kw|noclos)":>12s}')
for k in kws:
    has = d.str.contains(k, regex=False)
    n = has.sum()
    pck = y[has.values].mean() if n else float('nan')
    pkc = has.values[y==1].mean()
    pkn = has.values[y==0].mean()
    print(f'{k:12s} {n:5d} {pck:11.3f} {pkc:11.3f} {pkn:12.3f}')

anykw = d.apply(lambda s: any(k in s for k in kws))
print(f'\nANY keyword: n={anykw.sum()}  P(closure|kw)={y[anykw.values].mean():.3f}  P(kw|closure)={anykw.values[y==1].mean():.3f}')

# Reproduce on the 500-row sample to see if their 34.5% is sampling noise
print('\n--- 500-row SAMPLE (random_state=42) ---')
s = df.sample(500, random_state=42)
ys = s['requires_road_closure'].astype(int).values
ds = s['description'].fillna('').str.lower()
aks = ds.apply(lambda x: any(k in x for k in kws))
print(f'sample base={ys.mean():.3f}  n_kw={aks.sum()}  P(closure|kw)={ys[aks.values].mean() if aks.sum() else float("nan"):.3f}')

# Does TF-IDF actually hit ~0.738, and is it leakage or just re-encoding cause/type?
print('\n--- TF-IDF logistic, 5-fold stratified, on description ---')
mask = df['description'].notna().values
X = df.loc[mask,'description'].fillna('')
yt = y[mask]
vec = TfidfVectorizer(min_df=5, max_features=5000, ngram_range=(1,2))
Xv = vec.fit_transform(X)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(LogisticRegression(max_iter=1000, class_weight='balanced'),
                      Xv, yt, cv=cv, method='predict_proba')[:,1]
print(f'  n={mask.sum()} base={yt.mean():.3f}  AUC={roc_auc_score(yt,p):.3f}  PR-AUC={average_precision_score(yt,p):.3f}')

# top tokens driving it -> is it leakage words, or cause/location words?
lr = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xv, yt)
names = np.array(vec.get_feature_names_out())
top = np.argsort(lr.coef_[0])[-25:][::-1]
print('  top + tokens:', ', '.join(names[top]))

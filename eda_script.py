import pandas as pd
import numpy as np
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
print(f"Row count: {df.shape[0]}")
print(f"Column count: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes.value_counts())

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nCardinality of categorical columns:")
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")

print("\nText columns (potential):")
for col in cat_cols:
    if df[col].nunique() > 100:
        print(f"{col} (n_unique: {df[col].nunique()})")

print("\nTemporal columns (potential):")
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        print(col)

print("\nSpatial columns (potential):")
for col in df.columns:
    if 'lat' in col.lower() or 'lon' in col.lower():
        print(col)

print("\nTarget Candidates:")
print(df[['status', 'priority', 'requires_road_closure', 'event_cause', 'event_type']].nunique())

print("\nTarget Distributions:")
if 'priority' in df.columns:
    print("Priority:\n", df['priority'].value_counts(dropna=False))
if 'status' in df.columns:
    print("Status:\n", df['status'].value_counts(dropna=False))
if 'requires_road_closure' in df.columns:
    print("Requires Road Closure:\n", df['requires_road_closure'].value_counts(dropna=False))


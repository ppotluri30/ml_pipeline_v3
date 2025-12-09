#!/usr/bin/env python3
"""Test pandas datetime parsing to diagnose the timestamp collapse issue"""

import pandas as pd
from datetime import datetime, timedelta

# Simulate what Locust sends
base = datetime(2025, 11, 8, 1, 0, 0)
timestamps = [(base + timedelta(minutes=i*2)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(30)]

print("=" * 80)
print("INPUT TIMESTAMPS")
print("=" * 80)
print(f"Total: {len(timestamps)}")
print(f"Unique: {len(set(timestamps))}")
print(f"First 5: {timestamps[:5]}")
print()

# Test 1: Default parsing (what current code does)
print("=" * 80)
print("TEST 1: pd.to_datetime with errors='coerce' (CURRENT CODE)")
print("=" * 80)
df1 = pd.DataFrame({"time": timestamps})
idx1 = pd.to_datetime(df1["time"], errors="coerce")
print(f"Parsed unique: {idx1.nunique()}")
print(f"Parsed dtype: {idx1.dtype}")
print(f"First 5: {idx1.head().tolist()}")
print(f"Has NaT: {idx1.isna().any()}")
print()

# Test 2: Explicit format
print("=" * 80)
print("TEST 2: pd.to_datetime with explicit format")
print("=" * 80)
df2 = pd.DataFrame({"time": timestamps})
try:
    idx2 = pd.to_datetime(df2["time"], format="%Y-%m-%dT%H:%M:%S", errors="raise")
    print(f"Parsed unique: {idx2.nunique()}")
    print(f"Parsed dtype: {idx2.dtype}")
    print(f"First 5: {idx2.head().tolist()}")
    print(f"Has NaT: {idx2.isna().any()}")
except Exception as e:
    print(f"ERROR: {e}")
print()

# Test 3: With UTC flag
print("=" * 80)
print("TEST 3: pd.to_datetime with utc=True")
print("=" * 80)
df3 = pd.DataFrame({"time": timestamps})
idx3 = pd.to_datetime(df3["time"], utc=True, errors="coerce")
print(f"Parsed unique: {idx3.nunique()}")
print(f"Parsed dtype: {idx3.dtype}")
print(f"First 5: {idx3.head().tolist()}")
print(f"Has NaT: {idx3.isna().any()}")
print(f"Has timezone: {idx3.dt.tz is not None}")
print()

# Test 4: Check if sort_index causes issues
print("=" * 80)
print("TEST 4: After sort_index()")
print("=" * 80)
df4 = pd.DataFrame({"value": range(30)})
df4.index = idx1
df4_sorted = df4.sort_index()
print(f"Before sort unique: {df4.index.nunique()}")
print(f"After sort unique: {df4_sorted.index.nunique()}")
print()

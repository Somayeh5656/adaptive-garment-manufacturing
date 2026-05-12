"""
merge_trends_sales.py
---------------------
Merges the multi‑modal trend signals (visual + Google Trends) with the aligned
real H&M sales data.  The result is a unified daily dataset containing both
features and target.

Inputs (in data/):
  - daily_trend_signals.csv          (from build_timeseries.py)
  - hm_daily_sales_aligned.csv       (from align_hm_sales.py)

Output (in data/):
  - daily_trend_signals_with_real_sales.csv
"""

import pandas as pd
import os

# ------------------------------------------------------------
# 1. Paths – all data in ../data/ relative to this script
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

TRENDS_FILE = os.path.join(DATA_DIR, 'daily_trend_signals.csv')
SALES_FILE  = os.path.join(DATA_DIR, 'hm_daily_sales_aligned.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'daily_trend_signals_with_real_sales.csv')

# ------------------------------------------------------------
# 2. Load the two datasets
# ------------------------------------------------------------
trends = pd.read_csv(TRENDS_FILE, parse_dates=['date'])
real_sales = pd.read_csv(SALES_FILE, parse_dates=['date'])

# ------------------------------------------------------------
# 3. Merge on date (inner join – only days present in both)
# ------------------------------------------------------------
merged = trends.merge(real_sales, on='date', how='inner')

print(f"Merged dataset has {len(merged)} rows and {len(merged.columns)} columns.")
print(merged.head())

# ------------------------------------------------------------
# 4. Save the enriched dataset
# ------------------------------------------------------------
merged.to_csv(OUTPUT_FILE, index=False)
print(f"Saved merged dataset to {OUTPUT_FILE}")
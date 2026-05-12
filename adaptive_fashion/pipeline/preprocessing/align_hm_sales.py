"""
align_hm_sales.py
-----------------
Aligns the daily H&M sales data with the social media trend window.

- Shifts all sales dates forward by 6 years so they overlap with the period
  covered by Instagram, Pinterest, and Google Trends (2025‑11‑15 to 2026‑02‑15).
- Filters the shifted sales to only those 93 days.
- Saves the result to data/hm_daily_sales_aligned.csv.
"""

import pandas as pd
import os

# ------------------------------------------------------------
# 1. Paths – all data lives in ../data/ relative to this script
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

RAW_SALES_CSV = os.path.join(DATA_DIR, 'hm_daily_sales_raw.csv')
ALIGNED_SALES_CSV = os.path.join(DATA_DIR, 'hm_daily_sales_aligned.csv')

# ------------------------------------------------------------
# 2. Load raw aggregated sales
# ------------------------------------------------------------
df = pd.read_csv(RAW_SALES_CSV, parse_dates=['date'])

# ------------------------------------------------------------
# 3. Shift dates forward by 6 years
#    This places the sales data into the same timeframe as the
#    social media trend signals (originally 2018‑2020 → 2024‑2026).
# ------------------------------------------------------------
df['date'] = df['date'] + pd.DateOffset(years=6)

# ------------------------------------------------------------
# 4. Filter to the exact trend window (2025‑11‑15 to 2026‑02‑15)
# ------------------------------------------------------------
start_date = "2025-11-15"
end_date = "2026-02-15"
df_aligned = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

print(f"Aligned {len(df_aligned)} days within the trend window.")
print(df_aligned.head())

# ------------------------------------------------------------
# 5. Save the aligned sales
# ------------------------------------------------------------
df_aligned.to_csv(ALIGNED_SALES_CSV, index=False)
print(f"Saved aligned sales to {ALIGNED_SALES_CSV}")
"""
build_timeseries.py
--------------------
Converts the per‑image YOLOS detections into daily counts, merges with
Google Trends data, and saves the unified multi‑modal daily time series.

All input files are read from ../data/ and the output is written there.
"""

import pandas as pd
import ast
import os

# ------------------------------------------------------------
# 1. Define paths relative to this script (located in pipeline/preprocessing/)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

DETECTIONS_CSV = os.path.join(DATA_DIR, 'clothing_detections.csv')
TRENDS_CSV     = os.path.join(DATA_DIR, 'google_trends_latest.csv')
OUTPUT_CSV     = os.path.join(DATA_DIR, 'daily_trend_signals.csv')

# ------------------------------------------------------------
# 2. Load YOLOS detections
# ------------------------------------------------------------
df = pd.read_csv(DETECTIONS_CSV)
df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', utc=True)

# Shift Pinterest dates +6 years to align with the Instagram/Trends window
pinterest_mask = df['source'] == 'pinterest'
df.loc[pinterest_mask, 'datetime'] = df.loc[pinterest_mask, 'datetime'] + pd.DateOffset(years=6)

df['date'] = df['datetime'].dt.date   # keep only the date part

# ------------------------------------------------------------
# 3. Expand the clothing list into separate rows, then pivot to daily counts
# ------------------------------------------------------------
df['clothing'] = df['clothing'].apply(ast.literal_eval)

rows = []
for _, row in df.iterrows():
    for item in row['clothing']:
        rows.append({
            'date': row['date'],
            'source': row['source'],
            'item': item
        })

df_items = pd.DataFrame(rows)

daily = df_items.groupby(['date', 'item']).size().reset_index(name='count')
pivot = daily.pivot(index='date', columns='item', values='count').fillna(0)
pivot.columns = ['detect_' + str(col) for col in pivot.columns]  # e.g. detect_dress

# ------------------------------------------------------------
# 4. Load Google Trends (fixed filename)
# ------------------------------------------------------------
trends = pd.read_csv(TRENDS_CSV)
trends['date'] = pd.to_datetime(trends['date']).dt.date
trends.set_index('date', inplace=True)
trends.columns = ['trend_' + str(col) for col in trends.columns]  # e.g. trend_cargo pants

# ------------------------------------------------------------
# 5. Merge the two modalities (outer join) and fill missing values with 0
# ------------------------------------------------------------
combined = trends.join(pivot, how='outer').fillna(0)
combined.reset_index(inplace=True)
combined.rename(columns={'index': 'date'}, inplace=True)

# ------------------------------------------------------------
# 6. Save the unified daily trend signals
# ------------------------------------------------------------
combined.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Daily trend signals saved to {OUTPUT_CSV}")
print(combined.head())
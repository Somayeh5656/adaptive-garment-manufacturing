"""
prepare_tft_data_with_nlp.py
----------------------------
Prepares the final multi‑modal dataset for training the Temporal Fusion
Transformer (TFT).  It reads the enriched file that already contains
visual counts, Google Trends, NLP scores, and real sales, then adds the
required TFT columns (time_idx, group_id) and saves the result.

Input:  data/daily_trend_signals_with_nlp_full.csv
Output: data/tft_data_with_nlp.csv
"""

import pandas as pd
import os

# ------------------------------------------------------------
# 1. Paths – input and output inside ../data/
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

INPUT_FILE = os.path.join(DATA_DIR, 'daily_trend_signals_with_nlp_full.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'tft_data_with_nlp.csv')

# ------------------------------------------------------------
# 2. Load the enriched dataset
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE, parse_dates=['date'])

# ------------------------------------------------------------
# 3. Sort chronologically and add TFT‑required columns
# ------------------------------------------------------------
df.sort_values('date', inplace=True)
df['time_idx'] = range(len(df))         # integer time index (0, 1, 2, ...)
df['group_id'] = 'garment'              # single time‑series group

# ------------------------------------------------------------
# 4. Save the prepared data
# ------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print(f"Prepared TFT data saved to {OUTPUT_FILE}")
print("Columns:", df.columns.tolist())
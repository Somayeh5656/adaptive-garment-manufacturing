"""
extract_hm_sales.py
-------------------
Loads the H&M article metadata and transactions, filters for fashion items,
and aggregates daily sales. All input CSVs (articles.csv, transactions_train.csv)
must be placed in the data/ folder. The output (hm_daily_sales_raw.csv) is saved
to the same folder.
"""

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1. Paths – all data in ../data/ relative to this script
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

ARTICLES_CSV    = os.path.join(DATA_DIR, 'articles.csv')
TRANSACTIONS_CSV = os.path.join(DATA_DIR, 'transactions_train.csv')
OUTPUT_CSV      = os.path.join(DATA_DIR, 'hm_daily_sales_raw.csv')

# ------------------------------------------------------------
# 2. Load articles and identify fashion items
# ------------------------------------------------------------
articles = pd.read_csv(
    ARTICLES_CSV,
    usecols=['article_id', 'product_group_name'],
    dtype={'article_id': str}
)

# Normalise article_id: strip and pad to 10 characters
articles['article_id'] = articles['article_id'].str.strip().str.zfill(10)

# Quick overview of product groups (optional)
print("Unique product_group_name values (first 20):")
print(articles['product_group_name'].value_counts().head(20))

# Define which product groups are considered fashion garments
fashion_groups = [
    'Garment Upper body',
    'Garment Lower body',
    'Garment Full body',
    'Underwear',
    'Nightwear',
    'Swimwear',
    'Socks & Tights'
]

pattern = '|'.join(fashion_groups)
fashion_mask = articles['product_group_name'].str.contains(pattern, na=False, case=False)
fashion_articles = articles.loc[fashion_mask, 'article_id'].tolist()
fashion_set = set(fashion_articles)
print(f"Found {len(fashion_set)} fashion article IDs.")

# ------------------------------------------------------------
# 3. Quick diagnostic: check overlap on a small transaction sample
# ------------------------------------------------------------
print("\n--- DIAGNOSTIC ---")
sample_trans = pd.read_csv(
    TRANSACTIONS_CSV,
    nrows=10000,
    usecols=['article_id'],
    dtype={'article_id': str}
)
sample_trans['article_id'] = sample_trans['article_id'].str.strip().str.zfill(10)

print("First 10 article_id from articles (cleaned):", list(fashion_set)[:10])
print("First 10 article_id from transactions sample (cleaned):",
      sample_trans['article_id'].head(10).tolist())

overlap = sample_trans['article_id'].isin(fashion_set).sum()
print(f"Overlap in first 10k rows: {overlap}")

if overlap == 0:
    print("WARNING: No overlap found. Check fashion categories or ID formats.")
else:
    print("Overlap found – proceeding with full processing.")

# ------------------------------------------------------------
# 4. Process full transactions in chunks
# ------------------------------------------------------------
chunk_size = 500000
daily_sales = {}   # date -> total sales

for i, chunk in enumerate(pd.read_csv(
        TRANSACTIONS_CSV,
        chunksize=chunk_size,
        usecols=['t_dat', 'article_id', 'price'],
        dtype={'article_id': str, 'price': 'float32'},
        parse_dates=['t_dat'])):
    print(f"Processing chunk {i}...")
    chunk['article_id'] = chunk['article_id'].str.strip().str.zfill(10)
    fashion_chunk = chunk[chunk['article_id'].isin(fashion_set)]
    if fashion_chunk.empty:
        print(f"  No fashion items in chunk {i}")
        continue
    print(f"  Found {len(fashion_chunk)} fashion transactions")
    grouped = fashion_chunk.groupby(fashion_chunk['t_dat'].dt.date)['price'].sum()
    for date, value in grouped.items():
        daily_sales[date] = daily_sales.get(date, 0) + value

# ------------------------------------------------------------
# 5. Convert to DataFrame and save
# ------------------------------------------------------------
if daily_sales:
    sales_df = pd.DataFrame(list(daily_sales.items()), columns=['date', 'sales'])
    sales_df = sales_df.sort_values('date').reset_index(drop=True)
    print(f"Aggregated {len(sales_df)} days of sales.")
    print(sales_df.head(10))
else:
    print("No sales data aggregated – check the filters.")
    sales_df = pd.DataFrame(columns=['date', 'sales'])

sales_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved hm_daily_sales_raw.csv to {OUTPUT_CSV}")
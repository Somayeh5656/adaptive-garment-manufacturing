#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nlp_trends.py
-------------
Uses a zero‑shot classification model (BART‑large‑MNLI) to analyse
Instagram captions. Candidate labels are dynamically taken from the garment
categories detected by YOLOS (from clothing_detections.csv), ensuring that
textual signals are perfectly aligned with the visual ones.

The script:
1. Loads clothing_detections.csv to obtain unique garment categories.
2. Loads Instagram posts (instagram_posts_with_dates.csv) and keeps captions.
3. Classifies each caption with the garment labels (multi‑label).
4. Aggregates daily average scores per category.
5. Merges the NLP scores with the existing trend + sales dataset
   (daily_trend_signals_with_real_sales.csv).

Output: data/daily_trend_signals_with_nlp_yolos.csv
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import logging
import sys
import ast
import os

# ----------------------------------------------------------------------
# CONFIGURATION – all filenames point into ../data/
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

CLOTHING_DETECTIONS_FILE = os.path.join(DATA_DIR, 'clothing_detections.csv')
INSTAGRAM_POSTS_FILE     = os.path.join(DATA_DIR, 'instagram_posts_with_dates.csv')
TRENDS_FILE              = os.path.join(DATA_DIR, 'daily_trend_signals_with_real_sales.csv')
OUTPUT_FILE              = os.path.join(DATA_DIR, 'daily_trend_signals_with_nlp_yolos.csv')

# Zero‑shot model (can be swapped for a smaller one to speed up)
MODEL_NAME = "facebook/bart-large-mnli"
DEVICE = -1   # -1 for CPU, 0 for first GPU (if available)

# ----------------------------------------------------------------------
# SETUP LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# STEP 1 – EXTRACT UNIQUE GARMENT NAMES FROM YOLOS DETECTIONS
# ----------------------------------------------------------------------
logger.info("Loading clothing detections from YOLOS...")
df_cloth = pd.read_csv(CLOTHING_DETECTIONS_FILE)

# The 'clothing' column contains a string representation of a list, e.g. "['bag, wallet', 'belt']"
df_cloth['clothing'] = df_cloth['clothing'].apply(ast.literal_eval)

# Collect all garment names
all_garments = set()
for clothes_list in df_cloth['clothing']:
    for item in clothes_list:
        all_garments.add(item)

candidate_labels = sorted(all_garments)
logger.info(f"Found {len(candidate_labels)} unique garment categories from YOLOS.")
print("First 10 categories:", candidate_labels[:10])

# ----------------------------------------------------------------------
# STEP 2 – LOAD INSTAGRAM POSTS (with captions)
# ----------------------------------------------------------------------
logger.info("Loading Instagram posts with captions...")
df_ig = pd.read_csv(INSTAGRAM_POSTS_FILE, parse_dates=['datetime'])

# Keep only rows with non‑empty alt_text (captions)
df_ig = df_ig.dropna(subset=['alt_text'])
df_ig = df_ig[df_ig['alt_text'] != ""]
logger.info(f"Loaded {len(df_ig)} posts with non‑empty captions.")

if len(df_ig) == 0:
    logger.error("No valid captions found. Exiting.")
    sys.exit(1)

# ----------------------------------------------------------------------
# STEP 3 – LOAD ZERO‑SHOT CLASSIFIER
# ----------------------------------------------------------------------
logger.info(f"Loading zero‑shot classification model: {MODEL_NAME}")
try:
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=DEVICE,
    )
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# STEP 4 – CLASSIFY EACH CAPTION
# ----------------------------------------------------------------------
logger.info("Classifying captions (this may take a while)...")
results = []   # each element: dict with 'date' and scores for each garment

for idx, row in tqdm(df_ig.iterrows(), total=len(df_ig), desc="Classifying"):
    caption = row['alt_text']
    date = row['datetime'].date()
    try:
        output = classifier(caption, candidate_labels, multi_label=True)
        result_row = {'date': date}
        for label, score in zip(output['labels'], output['scores']):
            # Clean label to be a valid DataFrame column name
            clean_label = label.replace(' ', '_').replace(',', '_')
            result_row[clean_label] = score
        results.append(result_row)
    except Exception as e:
        logger.warning(f"Error at index {idx}: {e}. Skipping this post.")
        continue

if len(results) == 0:
    logger.error("No classifications succeeded. Exiting.")
    sys.exit(1)

logger.info(f"Classified {len(results)} posts.")

# ----------------------------------------------------------------------
# STEP 5 – AGGREGATE DAILY SCORES (mean per garment per day)
# ----------------------------------------------------------------------
logger.info("Aggregating daily scores...")
df_scores = pd.DataFrame(results)

# Cleaned labels (same as during classification)
cleaned_labels = [label.replace(' ', '_').replace(',', '_') for label in candidate_labels]

daily_nlp = df_scores.groupby('date')[cleaned_labels].mean().reset_index()
daily_nlp['date'] = pd.to_datetime(daily_nlp['date'])
daily_nlp = daily_nlp.sort_values('date').reset_index(drop=True)

logger.info(f"Aggregated to {len(daily_nlp)} days of NLP signals.")
print("Sample of daily NLP signals:")
print(daily_nlp.head())

# ----------------------------------------------------------------------
# STEP 6 – MERGE WITH EXISTING TREND + SALES DATASET
# ----------------------------------------------------------------------
logger.info("Loading existing daily trend signals (with real sales)...")
trends = pd.read_csv(TRENDS_FILE, parse_dates=['date'])

merged = pd.merge(trends, daily_nlp, on='date', how='inner')
logger.info(f"Merged dataset has {len(merged)} rows and {len(merged.columns)} columns.")

# ----------------------------------------------------------------------
# STEP 7 – SAVE THE ENRICHED DATASET
# ----------------------------------------------------------------------
merged.to_csv(OUTPUT_FILE, index=False)
logger.info(f"Saved enriched dataset to {OUTPUT_FILE}")

print("\n✅ NLP analysis complete. New file created:", OUTPUT_FILE)
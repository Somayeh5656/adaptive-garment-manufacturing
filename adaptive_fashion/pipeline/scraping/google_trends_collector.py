"""
google_trends_collector.py
--------------------------
Fetches daily Google Trends search interest for predefined fashion keywords
and saves the resulting CSV into the `data/` folder with a fixed name
(google_trends_latest.csv) so downstream scripts can find it easily.
"""

import os
from pytrends.request import TrendReq
import pandas as pd

# ------------------------------------------------------------
# 1. Initialise the pytrends API client
# ------------------------------------------------------------
# hl='en-US' : results in US English
# tz=360    : timezone offset in minutes (360 = US/Pacific)
pytrends = TrendReq(hl='en-US', tz=360)

# ------------------------------------------------------------
# 2. Build the search payload
# ------------------------------------------------------------
keywords = ["cargo pants", "crop top", "baggy jeans"]
pytrends.build_payload(
    keywords,
    cat=0,                # all categories
    timeframe='today 3-m',  # last 3 months
    geo='',               # worldwide
    gprop=''              # web search (default)
)

# ------------------------------------------------------------
# 3. Retrieve interest‑over‑time data
# ------------------------------------------------------------
data = pytrends.interest_over_time()

if not data.empty:
    # Drop the 'isPartial' column (often True for the current day)
    data = data.drop(columns=['isPartial'])

    # Ensure the data/ directory exists (go one level up from scraping/ to pipeline/data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Fixed filename so other scripts always know where to find it
    output_filename = 'google_trends_latest.csv'
    output_path = os.path.join(output_dir, output_filename)

    # Save the CSV
    data.to_csv(output_path)
    print(f"Saved Google Trends data to {output_path}")
    print(data.head())
else:
    print("No data returned – check network or query parameters.")
"""
pinterest_api.py
----------------
Collects fashion-related pins from Pinterest using the official API.
Reads credentials from a .env file in the pipeline root (pipeline/.env).
Saves the collected pins to data/pinterest_pins.csv.
"""

import requests
import pandas as pd
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 1. Load environment variables from pipeline/.env
# ----------------------------------------------------------------------
# This script is in pipeline/scraping/, so go one level up to pipeline/
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(script_dir, '..')
env_path = os.path.join(pipeline_dir, '.env')
load_dotenv(dotenv_path=env_path)

ACCESS_TOKEN = os.getenv("PINTEREST_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError(
        "Missing PINTEREST_ACCESS_TOKEN environment variable. "
        "Make sure pipeline/.env exists and contains the token."
    )

# ----------------------------------------------------------------------
# 2. Define helper function to search pins
# ----------------------------------------------------------------------
def search_pins(keyword, page_size=50):
    """
    Searches public pins for a given keyword.
    Returns a list of pin dictionaries (id, title, description, etc.).
    """
    url = "https://api.pinterest.com/v5/search/pins"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    params = {"query": keyword, "page_size": page_size}

    pins = []
    page_count = 0

    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        data = response.json()

        # Debug: print structure of the first pin (only on the first page)
        if page_count == 0 and data.get("items"):
            print(f"\n[DEBUG] First pin structure for keyword '{keyword}':")
            print(json.dumps(data["items"][0], indent=2)[:2000])

        for item in data.get("items", []):
            # Extract the best available image URL
            media_images = item.get("media", {}).get("images", {})
            if media_images:
                # Priority: originals → 600x → first available resolution
                img_obj = (
                    media_images.get("originals") or
                    media_images.get("600x") or
                    next(iter(media_images.values()), {})
                )
                image_url = img_obj.get("url") if img_obj else None
            else:
                image_url = None

            pin = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "created_at": item.get("created_at"),
                "image_url": image_url,
                "dominant_color": item.get("dominant_color"),
                "link": item.get("link"),
                "keyword_searched": keyword
            }
            pins.append(pin)

        # Pagination: if there's a bookmark, load next page
        bookmark = data.get("bookmark")
        if bookmark:
            params["bookmark"] = bookmark
            time.sleep(0.5)
            page_count += 1
            print(f"   Fetched page {page_count}, total pins so far: {len(pins)}")
        else:
            break

    return pins

# ----------------------------------------------------------------------
# 3. Main – search for multiple keywords and save to data/
# ----------------------------------------------------------------------
# Keywords chosen for the thesis proof-of-concept.
keywords = ["cargo pants 2026", "crop top 2026", "baggy jeans 2026"]
all_pins = []

for kw in keywords:
    print(f"\n🔍 Searching Pinterest for '{kw}' ...")
    pins = search_pins(kw, page_size=100)   # maximum page size
    print(f"   Found {len(pins)} pins for '{kw}'")
    if pins and pins[0].get("image_url"):
        print(f"   Sample image URL: {pins[0]['image_url'][:100]}...")
    all_pins.extend(pins)
    time.sleep(1)   # be respectful to the API

# Build DataFrame
df = pd.DataFrame(all_pins)

# Ensure the data/ directory exists
data_dir = os.path.join(pipeline_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(data_dir, 'pinterest_pins.csv')
df.to_csv(output_path, index=False)

print(f"\n✅ Saved {len(df)} pins to {output_path}")
print("\n📊 Summary by keyword:")
print(df['keyword_searched'].value_counts())

valid_images = df['image_url'].notna().sum()
print(f"\n🖼️ Pins with valid image URLs: {valid_images} / {len(df)} "
      f"({valid_images/len(df)*100:.1f}%)")
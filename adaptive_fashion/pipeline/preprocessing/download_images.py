"""
download_images.py
------------------
Downloads Instagram and Pinterest images listed in the CSVs produced by the
scraping scripts.  Images are stored under data/images/instagram/ and
data/images/pinterest/ so they are found by detect_fashion.py.

Expected input files (inside data/):
    - instagram_posts_with_dates.csv
    - pinterest_pins.csv
"""

import pandas as pd
import requests
import os
import time

# ------------------------------------------------------------
# 1. Setup paths (relative to this script, goes to data/images/)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

INSTAGRAM_IMG_DIR = os.path.join(IMAGES_DIR, 'instagram')
PINTEREST_IMG_DIR = os.path.join(IMAGES_DIR, 'pinterest')

# Create the image folders if they don't exist
os.makedirs(INSTAGRAM_IMG_DIR, exist_ok=True)
os.makedirs(PINTEREST_IMG_DIR, exist_ok=True)

# Paths to the CSV files
INSTAGRAM_CSV = os.path.join(DATA_DIR, 'instagram_posts_with_dates.csv')
PINTEREST_CSV = os.path.join(DATA_DIR, 'pinterest_pins.csv')

def download_image(url, folder, filename):
    """
    Saves an image from a URL to folder/filename.
    Returns True on success, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(folder, filename), 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        return False
    return False

# ------------------------------------------------------------
# 2. Download Instagram images
# ------------------------------------------------------------
df_ig = pd.read_csv(INSTAGRAM_CSV)
for idx, row in df_ig.iterrows():
    url = row['image_url']
    if pd.isna(url):
        continue
    filename = f"ig_{idx}.jpg"
    print(f"Downloading Instagram {filename} ...")
    success = download_image(url, INSTAGRAM_IMG_DIR, filename)
    if success:
        print("  OK")
    else:
        print("  FAILED")
    time.sleep(1)   # be polite to the server

# ------------------------------------------------------------
# 3. Download Pinterest images
# ------------------------------------------------------------
df_pin = pd.read_csv(PINTEREST_CSV)
for idx, row in df_pin.iterrows():
    url = row['image_url']
    if pd.isna(url):
        continue
    filename = f"pin_{idx}.jpg"
    print(f"Downloading Pinterest {filename} ...")
    success = download_image(url, PINTEREST_IMG_DIR, filename)
    if success:
        print("  OK")
    else:
        print("  FAILED")
    time.sleep(1)

print("✅ Done downloading images.")
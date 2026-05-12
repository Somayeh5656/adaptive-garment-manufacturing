"""
detect_fashion.py
-----------------
Runs YOLOS (fine‑tuned on Fashionpedia) on the downloaded Instagram and Pinterest images.
Filters out profile pictures (Instagram), then for each image outputs a list of
clothing categories.  Results are saved to data/clothing_detections.csv.

Expected input files (inside data/):
    - instagram_posts_with_dates.csv (from selenium_instagram2.py)
    - pinterest_pins.csv              (from pinterest_api.py)
    - images/instagram/ig_<idx>.jpg   (downloaded images)
    - images/pinterest/pin_<idx>.jpg  (downloaded images)
"""

import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import pandas as pd
import os

# ------------------------------------------------------------
# 1. Load YOLOS model and processor (Fashionpedia fine‑tuned)
# ------------------------------------------------------------
processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")

def detect_clothing(image_path, threshold=0.3):
    """
    Runs object detection on a single image.
    Returns a list of detected clothing category names (e.g., ['dress', 'belt']).
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    detected = []
    for score, label in zip(results["scores"], results["labels"]):
        category = model.config.id2label[label.item()]
        detected.append(category)
    return detected

# ------------------------------------------------------------
# 2. Paths setup (relative to this script, target data/ folder)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')               # pipeline/data/
IMAGES_DIR = os.path.join(DATA_DIR, 'images')                   # pipeline/data/images/

INSTAGRAM_CSV = os.path.join(DATA_DIR, 'instagram_posts_with_dates.csv')
PINTEREST_CSV = os.path.join(DATA_DIR, 'pinterest_pins.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'clothing_detections.csv')

# ------------------------------------------------------------
# 3. Process Instagram images
# ------------------------------------------------------------
df_ig = pd.read_csv(INSTAGRAM_CSV)
results = []

# The alt text of a profile picture (Finnish text). We skip these.
profile_alt = "Käyttäjän thesis202656 profiilikuva"

for idx, row in df_ig.iterrows():
    # Skip profile‑picture rows
    if row['alt_text'] == profile_alt:
        continue

    img_path = os.path.join(IMAGES_DIR, 'instagram', f'ig_{idx}.jpg')
    if not os.path.exists(img_path):
        continue

    print(f"Processing Instagram {img_path} ...")
    try:
        clothes = detect_clothing(img_path)
        results.append({
            "source": "instagram",
            "index": idx,
            "datetime": row['datetime'],
            "clothing": clothes,
            "alt": row['alt_text']
        })
    except Exception as e:
        print(f"Error: {e}")

# ------------------------------------------------------------
# 4. Process Pinterest images
# ------------------------------------------------------------
df_pin = pd.read_csv(PINTEREST_CSV)
for idx, row in df_pin.iterrows():
    img_path = os.path.join(IMAGES_DIR, 'pinterest', f'pin_{idx}.jpg')
    if not os.path.exists(img_path):
        continue

    print(f"Processing Pinterest {img_path} ...")
    try:
        clothes = detect_clothing(img_path)
        results.append({
            "source": "pinterest",
            "index": idx,
            "datetime": row['created_at'],      # timestamp from API
            "clothing": clothes,
            "description": row['description']
        })
    except Exception as e:
        print(f"Error: {e}")

# ------------------------------------------------------------
# 5. Save all detection results
# ------------------------------------------------------------
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Detection results saved to {OUTPUT_CSV} (total rows: {len(results)})")
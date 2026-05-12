"""
selenium_instagram2.py
-----------------------
Scrapes Instagram posts from a given hashtag page using Selenium.
Requires a manual login step (the script pauses for 60 seconds).
Extracts post URLs, image URLs, alt text (captions), and publication dates.
Saves the data to pipeline/data/instagram_posts_with_dates.csv.
"""

import csv
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ----------------------------------------------------------------------
# CONFIGURATION – adjust these values as needed
# ----------------------------------------------------------------------
HASHTAG = "fashion"                     # target hashtag
SCROLL_COUNT = 5                        # how many times to scroll (loads more posts)
MAX_POSTS = 50                          # maximum number of posts to process
CSV_FILENAME = "instagram_posts_with_dates.csv"   # output file name (saved in ../data)

# ----------------------------------------------------------------------
# 1. Start the browser and perform manual login
# ----------------------------------------------------------------------
driver = webdriver.Edge()               # can be changed to Chrome, Firefox, etc.
driver.get("https://www.instagram.com/accounts/login/")
print("Please log in manually within 60 seconds...")
time.sleep(60)                          # give time to complete the login

# ----------------------------------------------------------------------
# 2. Navigate to the hashtag page and scroll to load posts
# ----------------------------------------------------------------------
driver.get(f"https://www.instagram.com/explore/tags/{HASHTAG}/")
time.sleep(5)

for i in range(SCROLL_COUNT):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    print(f"Scroll {i+1}/{SCROLL_COUNT}")
    time.sleep(3)

# ----------------------------------------------------------------------
# 3. Collect all unique post URLs
# ----------------------------------------------------------------------
post_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/p/')]")
post_urls = set()
for link in post_links:
    href = link.get_attribute('href')
    if href:
        post_urls.add(href)

print(f"Found {len(post_urls)} unique post URLs.")

# Limit to the desired number of posts
post_urls = list(post_urls)[:MAX_POSTS]
print(f"Will process {len(post_urls)} posts.")

# ----------------------------------------------------------------------
# 4. Visit each post and extract metadata
# ----------------------------------------------------------------------
data = []
for idx, post_url in enumerate(post_urls, 1):
    print(f"Processing post {idx}/{len(post_urls)}: {post_url}")

    # Open the post in a new tab
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get(post_url)
    time.sleep(2)

    try:
        # Wait for the <time> element that contains the post date
        time_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "time"))
        )
        post_datetime = time_element.get_attribute('datetime')  # ISO 8601 format

        # Locate the main post image (the one with alt text and hosted on cdninstagram.com)
        img_element = driver.find_element(By.XPATH,
            "//img[@alt and contains(@src, 'cdninstagram.com')]")
        img_url = img_element.get_attribute('src')
        alt_text = img_element.get_attribute('alt')   # often contains the caption

        data.append([post_url, img_url, alt_text, post_datetime])
        print(f"  Date: {post_datetime}")
    except Exception as e:
        print(f"  Error: {e}")
        data.append([post_url, "", "", ""])

    # Close the tab and return to the main window
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(1)   # pause between posts

# ----------------------------------------------------------------------
# 5. Clean up and save the CSV
# ----------------------------------------------------------------------
driver.quit()

# Determine the output path: pipeline/data/CSV_FILENAME
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, CSV_FILENAME)

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['post_url', 'image_url', 'alt_text', 'datetime'])
    writer.writerows(data)

print(f"\nDone! Saved {len(data)} posts to {output_path}")
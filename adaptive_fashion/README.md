# Adaptive Garment Manufacturing – AI Pipeline

This is the **data‑driven pipeline** portion of the thesis **Adaptive Garment Manufacturing: A Multi‑Modal AI Pipeline for Trend Prediction and Digital Twin Simulation**.

The pipeline:
- Scrapes social media (Instagram, Pinterest) and Google Trends
- Detects fashion items in images using YOLOS
- Extracts textual sentiment from captions with a zero‑shot BART model
- Builds a multi‑modal time‑series dataset
- Trains a Temporal Fusion Transformer (TFT) to forecast 30‑day demand
- Runs a SimPy digital twin of a garment factory
- Trains a reinforcement learning agent (PPO) to optimise inventory reorder points
- Integrates with a robotic cutting cell (separate ROS 2 workspace)

---

## Prerequisites

- **Python 3.12** (or compatible ≥3.10)
- **pip** virtual environment (`venv`)
- **Git LFS** (if you want to store large model files, otherwise they are excluded)
- **Chrome/Edge** for Instagram scraping (Selenium uses a local browser)

Optional but recommended:
- NVIDIA GPU for faster training (the pipeline works on CPU as well)

---

## Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/yourusername/adaptive-garment-manufacturing.git
cd adaptive-garment-manufacturing/pipeline

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements_pipeline.txt
```

If you plan to run the Instagram scraper, ensure you have a modern browser installed.

---

## Configuration & Secrets

The Pinterest API and OAuth credentials must **not** be committed.  
Create a `.env` file in the `pipeline/` directory (copy from `.env.example`):

```bash
cp .env.example .env
```

Fill in your own tokens. Example `.env.example`:

```
PINTEREST_ACCESS_TOKEN=your_access_token
PINTEREST_CLIENT_ID=your_client_id
PINTEREST_CLIENT_SECRET=your_client_secret
PINTEREST_AUTH_CODE=your_oauth_code
```

All scripts that use these secrets load them via `python-dotenv`.

---

## Data

### Obtaining raw data

1. **Instagram** – run `scraping/selenium_instagram2.py` after manually logging in. The script saves `instagram_posts_with_dates.csv`.
2. **Pinterest** – use `scraping/pinterest_api.py` (requires valid `.env`). Output: `pinterest_pins.csv`.
3. **Google Trends** – `scraping/google_trends_collector.py` automatically downloads CSVs.
4. **H&M sales** – download `articles.csv` and `transactions_train.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and place them in `data/`. Run `preprocessing/extract_hm_sales.py` and `align_hm_sales.py` to produce `hm_daily_sales_aligned.csv`.

> The pipeline includes all preprocessing scripts; you do **not** need to run every step manually – see “End‑to‑End Run” below.

---

## Running the Pipeline

### Step‑by‑step (for development)

All scripts are in subfolders. Run them **from the `pipeline/` directory**.

1. **Download images** (after scraping)
   ```bash
   python preprocessing/download_images.py
   ```

2. **Visual trend detection (YOLOS)**
   ```bash
   python preprocessing/detect_fashion.py    # creates clothing_detections.csv
   ```

3. **Build daily time series of visual counts + Google Trends**
   ```bash
   python preprocessing/build_timeseries.py   # -> daily_trend_signals.csv
   ```

4. **Textual trend detection (zero‑shot NLP)**
   ```bash
   python preprocessing/nlp_trends.py         # -> daily_trend_signals_with_nlp_yolos.csv
   ```

5. **Merge with real sales**
   ```bash
   python preprocessing/merge_trends_sales.py   # -> daily_trend_signals_with_real_sales.csv
   python preprocessing/merge_nlp_full.py       # -> daily_trend_signals_with_nlp_full.csv
   ```

6. **Prepare for TFT**
   ```bash
   python forecasting/prepare_tft_data_with_nlp.py   # -> tft_data_with_nlp.csv
   ```

7. **Train TFT forecasting model**
   ```bash
   python forecasting/train_tft_with_nlp.py           # saves tft_model_nlp.ckpt
   ```

8. **Generate forecast**
   ```bash
   python forecasting/forecast_tft_nlp.py             # -> forecast_tft_nlp.csv
   ```

9. **Run the SimPy digital twin (baseline)**
   ```bash
   python simulation/baseline.py
   ```

10. **Train the RL inventory agent (PPO)**
    ```bash
    python simulation/train_rl.py
    ```

11. **Evaluate the integrated system (with robotics)**
    ```bash
    python simulation/evaluate_rl.py
    ```

### End‑to‑End (automated)

To run the entire pipeline from scratch (except manual Instagram login), execute the scripts in the order above. You can create a shell script that chains them, but note that some steps require the previous outputs.

---

## Integration with the Robotic Cutting Cell

The RL agent (`garment_env.py`) expects the ROS 2 cutting service to be available.  
Make sure the robotics workspace is built and the cutting service is running before you call `evaluate_rl.py`.  
See the [robotics README](../robotics/README.md) for details.

---

## Folder Structure (pipeline/)

```
pipeline/
├── scraping/           # Instagram, Pinterest, Google Trends
├── preprocessing/      # image download, YOLOS, NLP, merging, alignment
├── forecasting/        # TFT training and forecast generation
├── simulation/         # SimPy digital twin, PPO RL agent, evaluation
├── data/               # (optional) raw and intermediate data files
├── .env      # template for secrets
├── requirements_pipeline.txt
└── README.md
```

---

## Notes

- The Instagram scraper requires manual login; after that it scrolls #fashion and extracts posts.
- Pinterest limits free API calls; the collected pins are a proof‑of‑concept.
- The H&M sales dates are shifted forward 6 years to align with the trend period – this is intentional.
- The trained TFT and PPO models are **not** included in the repository due to size. Run the training scripts to regenerate them.


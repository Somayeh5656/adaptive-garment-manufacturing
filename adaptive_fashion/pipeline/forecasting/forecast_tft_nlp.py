"""
forecast_tft_nlp.py
-------------------
Loads a trained Temporal Fusion Transformer (TFT) checkpoint and generates
a 30‑day demand forecast using the last 30 days of the multi‑modal dataset.

Inputs:
  - data/tft_model_nlp.ckpt   (trained model from train_tft_with_nlp.py)
  - data/tft_data_with_nlp.csv (the same data used for training)

Output:
  - data/forecast_tft_nlp.csv  (daily median forecast for next 30 days)
"""

import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# 1. Paths – all data in ../data/ relative to this script
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

MODEL_CHECKPOINT = os.path.join(DATA_DIR, 'tft_model_nlp.ckpt')
INPUT_CSV        = os.path.join(DATA_DIR, 'tft_data_with_nlp.csv')
OUTPUT_CSV       = os.path.join(DATA_DIR, 'forecast_tft_nlp.csv')

# ------------------------------------------------------------
# 2. Load the trained TFT model
# ------------------------------------------------------------
tft = TemporalFusionTransformer.load_from_checkpoint(MODEL_CHECKPOINT)

# ------------------------------------------------------------
# 3. Load and prepare the data (same file used for training)
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
df = df.rename(columns={'date': 'time'})   # TFT expects column named 'time'

# ------------------------------------------------------------
# 4. Use the last 30 days as encoder input
# ------------------------------------------------------------
max_encoder_length = 30
last_data = df.iloc[-max_encoder_length:].copy()
last_time_idx = last_data['time_idx'].max()
last_date = last_data['time'].max()

# ------------------------------------------------------------
# 5. Create a future dataframe for the next 30 days
# ------------------------------------------------------------
future_df = pd.DataFrame({
    'time': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30),
    'time_idx': np.arange(last_time_idx + 1, last_time_idx + 31),
    'group_id': 'garment',
    'sales': 0       # placeholder – not used during prediction
})

# Copy the last known values of all other features (trends, NLP, etc.)
last_vals = last_data.iloc[-1].to_dict()
for col in df.columns:
    if col not in future_df.columns and col not in ['time', 'time_idx', 'group_id', 'sales']:
        future_df[col] = last_vals[col]

# Combine history and future
predict_df = pd.concat([last_data, future_df], ignore_index=True)

# ------------------------------------------------------------
# 6. Generate quantile predictions
# ------------------------------------------------------------
predictions = tft.predict(predict_df, mode="quantiles")

# Median forecast (index 3 out of 7 quantiles)
median_forecast = predictions[0, :, 3].detach().cpu().numpy()

# ------------------------------------------------------------
# 7. Save the forecast CSV
# ------------------------------------------------------------
forecast_df = pd.DataFrame({
    'ds': future_df['time'],
    'yhat': median_forecast
})
forecast_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Forecast saved to {OUTPUT_CSV}")

# ------------------------------------------------------------
# Optional: quick plot
# ------------------------------------------------------------
plt.plot(forecast_df['ds'], forecast_df['yhat'])
plt.title("TFT Forecast with NLP signals")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()
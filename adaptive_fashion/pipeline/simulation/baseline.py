"""
baseline.py
-----------
Runs the garment factory digital twin with a fixed inventory reorder point
(baseline policy). The forecast is read from data/forecast_tft_nlp.csv.
"""

import os
from garment_env import GarmentFactoryEnv

# ------------------------------------------------------------
# Paths – forecast CSV is in ../data/ relative to this script
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FORECAST_CSV = os.path.join(DATA_DIR, 'forecast_tft_nlp.csv')

# ------------------------------------------------------------
# Create environment with fixed reorder point (no RL)
# ------------------------------------------------------------
env_fixed = GarmentFactoryEnv(FORECAST_CSV, sim_days=30, fixed_reorder_point=10000)

obs, _ = env_fixed.reset()
done = False
total_reward = 0

while not done:
    # Dummy action – the fixed reorder point overrides it
    obs, reward, done, _, _ = env_fixed.step([0])
    total_reward += reward

print(f"Baseline total garments: {env_fixed.cumulative_output}")
print(f"Baseline total reward (profit): {total_reward:.2f}")
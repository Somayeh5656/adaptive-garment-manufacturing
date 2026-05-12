"""
evaluate_rl.py
--------------
Evaluates the trained PPO inventory agent in the full garment factory
digital twin environment (including the robotic cutting cell, if available).
Generates and saves key performance plots.

Inputs (from ../data/):
  - forecast_tft_nlp.csv
  - ppo_garment_factory.zip    (or *3M, adjust MODEL_NAME accordingly)

Outputs (in ../plots/):
  - cumulative_production.pdf
  - fabric_inventory.pdf
  - queue_lengths.pdf
  - daily_waste.pdf
  - waste_histogram.pdf
  - daily_production.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import rclpy
import os

from garment_env import GarmentFactoryEnv
from stable_baselines3 import PPO

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
PLOTS_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

FORECAST_CSV = os.path.join(DATA_DIR, 'forecast_tft_nlp.csv')
MODEL_NAME = "ppo_garment_factory_3M"          # adjust if your model is named differently
MODEL_PATH = os.path.join(DATA_DIR, MODEL_NAME)

# ------------------------------------------------------------
# 2. Load trained PPO agent
# ------------------------------------------------------------
print("Loading PPO model...")
model = PPO.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}.zip")

# ------------------------------------------------------------
# 3. Create the environment (robotic cutter will be initialised)
# ------------------------------------------------------------
print("Creating environment...")
env = GarmentFactoryEnv(FORECAST_CSV, sim_days=30)
print(f"Environment created. Forecast for day 0: {env.forecast[0]:.1f}")

# ------------------------------------------------------------
# 4. Run one evaluation episode
# ------------------------------------------------------------
obs, _ = env.reset()
done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(f"  Day {step+1}: reward = {reward:.2f}")
    step += 1

print("\nSimulation finished.")
print(f"Total garments produced: {env.cumulative_output}")
print(f"Final fabric inventory: {env.fabric_level:.1f} m")
if env.cumulative_output > 0:
    avg_waste = (env.total_waste / env.cumulative_output) * 100
    print(f"Average waste per garment: {avg_waste:.2f}%")
else:
    print("No garments produced, waste statistics unavailable.")

# ------------------------------------------------------------
# 5. Generate and save plots
# ------------------------------------------------------------

# 5.1 Cumulative Production vs. Forecast
plt.figure(figsize=(8, 5))
ideal_cumulative = np.cumsum(env.forecast)
ideal_time = np.arange(1, len(env.forecast) + 1)
plt.plot(ideal_time, ideal_cumulative, 'y--', linewidth=2, label='Ideal Forecast')
plt.step(env.time_stamps, env.output_values, where='post',
         color='green', linewidth=2, label='Actual Output')
plt.xlabel('Time (days)')
plt.ylabel('Garments Produced')
plt.title('Cumulative Production vs. Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cumulative_production.pdf'), dpi=300)
plt.close()

# 5.2 Fabric Inventory Over Time
plt.figure(figsize=(8, 5))
inv_time, inv_level = zip(*env.fabric_log)
plt.step(inv_time, inv_level, where='post', color='blue', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Fabric (meters)')
plt.title('Fabric Inventory')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fabric_inventory.pdf'), dpi=300)
plt.close()

# 5.3 Queue Lengths (Bottleneck Analysis)
plt.figure(figsize=(8, 5))
if env.cutting_queue_log:
    cut_q_time, cut_q_len = zip(*env.cutting_queue_log)
    plt.plot(cut_q_time, cut_q_len, label='Cutting', linewidth=1.5)
if env.sewing_queue_log:
    sew_q_time, sew_q_len = zip(*env.sewing_queue_log)
    plt.plot(sew_q_time, sew_q_len, label='Sewing', linewidth=1.5)
if env.inspection_queue_log:
    ins_q_time, ins_q_len = zip(*env.inspection_queue_log)
    plt.plot(ins_q_time, ins_q_len, label='Inspection', linewidth=1.5)
plt.xlabel('Time (days)')
plt.ylabel('Queue Length (waiting jobs)')
plt.title('Queue Lengths at Workstations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'queue_lengths.pdf'), dpi=300)
plt.close()

# 5.4 Daily Cutting Waste
if hasattr(env, 'daily_waste_list') and env.daily_waste_list:
    plt.figure(figsize=(8, 5))
    days = np.arange(1, len(env.daily_waste_list) + 1)
    plt.bar(days, env.daily_waste_list, color='red', alpha=0.7)
    plt.xlabel('Day')
    plt.ylabel('Waste Fraction')
    plt.title('Daily Cutting Waste')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'daily_waste.pdf'), dpi=300)
    plt.close()
else:
    print("Daily waste data not available – skipping waste plot.")

# 5.5 Waste Distribution Histogram
if hasattr(env, 'daily_waste_list') and env.daily_waste_list:
    plt.figure(figsize=(8, 5))
    plt.hist(env.daily_waste_list, bins=15, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel('Waste Fraction')
    plt.ylabel('Frequency')
    plt.title('Distribution of Daily Cutting Waste')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'waste_histogram.pdf'), dpi=300)
    plt.close()

# 5.6 Daily Production
if hasattr(env, 'daily_output_list') and env.daily_output_list:
    plt.figure(figsize=(8, 5))
    days = np.arange(1, len(env.daily_output_list) + 1)
    plt.plot(days, env.daily_output_list, 'g-', linewidth=2)
    plt.xlabel('Day')
    plt.ylabel('Garments Produced')
    plt.title('Daily Production')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'daily_production.pdf'), dpi=300)
    plt.close()

print(f"\nPlots saved to {PLOTS_DIR}")

# ------------------------------------------------------------
# 6. Clean ROS 2 shutdown
# ------------------------------------------------------------
try:
    rclpy.shutdown()
except Exception:
    pass
"""
train_rl.py
-----------
Trains a Proximal Policy Optimisation (PPO) agent to learn an optimal
inventory reorder point for the garment factory digital twin.
The forecast is read from data/forecast_tft_nlp.csv and the trained model
is saved to data/ppo_garment_factory.zip.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from garment_env import GarmentFactoryEnv

# ------------------------------------------------------------
# Paths – forecast CSV is in ../data/ relative to this script
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FORECAST_CSV = os.path.join(DATA_DIR, 'forecast_tft_nlp.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'ppo_garment_factory')

# ------------------------------------------------------------
# Create the environment
# ------------------------------------------------------------
env = GarmentFactoryEnv(FORECAST_CSV, sim_days=30)
check_env(env)   # verify the environment follows the Gymnasium API

# ------------------------------------------------------------
# Define policy network architecture
# ------------------------------------------------------------
policy_kwargs = dict(net_arch=[64, 64])

# ------------------------------------------------------------
# Create the PPO agent
# ------------------------------------------------------------
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs
)

# ------------------------------------------------------------
# Train the agent
# ------------------------------------------------------------
print("Starting training...")
model.learn(total_timesteps=100)   # adjust the number of timesteps as needed
model.save(MODEL_PATH)

print(f"Training complete. Model saved as {MODEL_PATH}.zip")
#!/usr/bin/env python3
"""
train_vision_cutting_rl.py
----------------------------
Trains a Soft Actor‑Critic (SAC) agent on the kinematic VisionCuttingEnv.
The agent learns to move a virtual tool to the fabric position with minimal
steps and waste.

Training uses randomised fabric positions so the policy generalises.
The final model is saved to ../models/vision_cutting_agent_final.zip
(relative to this script).
"""

import rclpy
import sys
import os
import time

# Make sure the fabric_cell_control package is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from fabric_cell_control.vision_cutting_env import VisionCuttingEnv


def main():
    # ------------------------------------------------------------
    # 1. Initialise a ROS 2 node (required by the environment)
    # ------------------------------------------------------------
    rclpy.init()
    node = rclpy.create_node('vision_cutting_training')

    # ------------------------------------------------------------
    # 2. Create the kinematic environment with random fabric
    # ------------------------------------------------------------
    env = VisionCuttingEnv(node, max_steps=50, randomize_fabric=True)
    time.sleep(1)   # brief pause for ROS to settle

    # ------------------------------------------------------------
    # 3. Create the SAC agent
    # ------------------------------------------------------------
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto_0.1',   # automatic entropy tuning
        gamma=0.98             # slightly lower discount to prioritise speed
    )

    # ------------------------------------------------------------
    # 4. Checkpoint callback – saves intermediate models
    # ------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./vision_cutting_models/',
        name_prefix='vision_cutting_agent'
    )

    # ------------------------------------------------------------
    # 5. Train the agent
    # ------------------------------------------------------------
    print("Starting training for vision‑guided cutting (random fabric positions)...")
    model.learn(total_timesteps=50000, callback=checkpoint_callback)

    # ------------------------------------------------------------
    # 6. Save the final model into ../models/
    # ------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'vision_cutting_agent_final')
    model.save(model_path)      # Stable‑Baselines3 appends .zip automatically
    print(f"Training complete! Model saved as {model_path}.zip")

    # ------------------------------------------------------------
    # 7. Shutdown ROS 2 cleanly
    # ------------------------------------------------------------
    rclpy.shutdown()


if __name__ == '__main__':
    main()
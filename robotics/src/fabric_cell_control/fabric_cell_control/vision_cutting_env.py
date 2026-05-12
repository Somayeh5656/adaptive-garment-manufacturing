#!/usr/bin/env python3
"""
vision_cutting_env.py
-----------------------
Gymnasium environment for vision‑guided robotic cutting.

This is a **kinematic** environment – the tool position is updated directly
from actions without simulating Gazebo physics. It can operate in two modes:
  1. randomize_fabric=True  – fabric position is randomised (used for training)
  2. randomize_fabric=False – fabric pose comes from a ROS 2 topic (used for service)

When used inside CuttingService, the node is shared externally.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
import threading
import time
import gymnasium as gym
from gymnasium import spaces


class VisionCuttingEnv(gym.Env):
    """
    Kinematic cutting environment for training/evaluating a SAC agent.

    State (6D): [tool_x, tool_y, tool_z, fabric_x, fabric_y, fabric_z]
    Action (3D): [dx, dy, dz] velocity in metres per step (max 0.05 m/step)
    """

    def __init__(self, node=None, max_steps=50, randomize_fabric=False, fast_mode=False):
        super().__init__()
        self.fast_mode = fast_mode  # skip sleep delays for fast evaluation

        # ROS 2 node handling
        if node is not None:
            self.node = node
            self.external_node = True
        else:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node('vision_cutting_env')
            self.external_node = False
            self._ros_spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self._ros_spin_thread.start()

        # Action space: small 3D velocity
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05], dtype=np.float32),
            dtype=np.float32
        )
        # Observation space: tool and fabric positions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.randomize_fabric = randomize_fabric
        self.max_steps = max_steps
        self.step_count = 0
        self.cut_complete = False

        # Default positions (overwritten in reset)
        self.fabric_position = np.array([0.5, 0.0, 0.05])
        self.fabric_received = False
        self.tool_position = np.array([0.5, 0.0, 0.1])

        # Subscribe to real fabric pose only if not randomizing
        if not self.randomize_fabric:
            self.fabric_sub = self.node.create_subscription(
                Pose, '/fabric_pose', self._fabric_callback, 10)
        else:
            self.fabric_received = True   # always "received" when random

    def _spin_ros(self):
        """Background ROS spinner for standalone usage."""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _fabric_callback(self, msg):
        """Receive fabric pose from perception node."""
        self.fabric_position = np.array(
            [msg.position.x, msg.position.y, msg.position.z]
        )
        self.fabric_received = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_fabric:
            # Randomise fabric within a realistic workspace
            self.fabric_position = np.array([
                self.np_random.uniform(0.2, 0.8),
                self.np_random.uniform(-0.3, 0.3),
                self.np_random.uniform(0.02, 0.12)
            ])
            self.fabric_received = True
        else:
            # Wait for the perception node to publish a pose
            timeout = 5.0
            start = time.time()
            while not self.fabric_received and (time.time() - start) < timeout:
                if not self.external_node:
                    rclpy.spin_once(self.node, timeout_sec=0.1)
                if not self.fast_mode:
                    time.sleep(0.01)
            if not self.fabric_received:
                self.fabric_position = np.array([0.5, 0.0, 0.05])  # fallback

        # Randomise tool start position
        self.tool_position = np.array([
            self.np_random.uniform(0.1, 0.7),
            self.np_random.uniform(-0.2, 0.2),
            self.np_random.uniform(0.05, 0.2)
        ], dtype=np.float32)

        self.step_count = 0
        self.cut_complete = False
        return self._get_obs(), {}

    def step(self, action):
        """Apply a 3D velocity action and compute reward."""
        self.tool_position += action

        reward = self._compute_reward(action)

        dist = np.linalg.norm(self.tool_position - self.fabric_position)
        if dist < 0.02:   # within 2 cm is considered a successful cut
            self.cut_complete = True
            reward += 50.0

        self.step_count += 1
        terminated = self.step_count >= self.max_steps or self.cut_complete
        truncated = False
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Return the 6‑dimensional observation vector."""
        return np.concatenate(
            [self.tool_position, self.fabric_position], dtype=np.float32
        )

    def _compute_reward(self, action):
        """
        Reward: distance penalty (encourages reaching the fabric)
               + small energy/step penalty to favour efficient movements.
        """
        dist = np.linalg.norm(self.tool_position - self.fabric_position)
        dist_reward = -dist * 20.0
        energy_penalty = -np.sum(np.square(action)) * 2.0
        step_penalty = -0.1
        return dist_reward + energy_penalty + step_penalty

    def close(self):
        if not self.external_node:
            self.node.destroy_node()
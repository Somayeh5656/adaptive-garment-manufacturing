#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import math
import gymnasium as gym
from gymnasium import spaces
from tf2_ros import TransformListener, Buffer

class GazeboCuttingEnv(gym.Env):
    """Gymnasium environment for cutting in Gazebo."""
    
    def __init__(self, node=None):
        super().__init__()
        
        # Handle node creation
        if node is not None:
            self.node = node
            self.external_node = True
        else:
            # Create a new node and spin it in a background thread
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node('gazebo_cutting_env')
            self.external_node = False
            self._ros_spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self._ros_spin_thread.start()
        
        # Action space: [v_shoulder, v_elbow] in rad/s
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: tool (x,y,z) and fabric (x,y,z)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # ROS2 publishers/subscribers
        self.pub_vel_cmd = self.node.create_publisher(
            Float64MultiArray,
            '/joint_velocity_controller/commands',
            10
        )
        self.fabric_sub = self.node.create_subscription(
            Pose, '/fabric_pose', self._fabric_callback, 10
        )
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10
        )
        
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
        # Internal state
        self.fabric_position = np.array([0.5, 0.0, 0.05])
        self.tool_position = np.array([0.5, 0.0, 0.1])
        self.step_count = 0
        self.max_steps = 50
        self.cut_complete = False
        
        # Flag to indicate fabric pose received
        self.fabric_received = False
        
    def _spin_ros(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
    
    def _fabric_callback(self, msg):
        self.fabric_position = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.fabric_received = True
    
    def _joint_callback(self, msg):
        # Not needed for direct use
        pass
    
    def _update_tool_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'tool_link', rclpy.time.Time())
            self.tool_position = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except:
            pass
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Wait for fabric pose (timeout)
        start = time.time()
        while not self.fabric_received and (time.time() - start) < 5.0:
            if not self.external_node:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.1)
        
        # Send zero velocity to stop robot
        cmd = Float64MultiArray()
        cmd.data = [0.0, 0.0]
        self.pub_vel_cmd.publish(cmd)
        time.sleep(0.5)
        
        self._update_tool_pose()
        self.step_count = 0
        self.cut_complete = False
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        # action: [v_shoulder, v_elbow]
        cmd = Float64MultiArray()
        cmd.data = [float(action[0]), float(action[1])]
        self.pub_vel_cmd.publish(cmd)
        
        # Let simulation run
        time.sleep(0.05)
        
        # Process any pending callbacks
        if not self.external_node:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        self._update_tool_pose()
        reward = self._compute_reward(action)
        
        dist = np.linalg.norm(self.tool_position - self.fabric_position)
        if dist < 0.02:
            self.cut_complete = True
            reward += 50.0
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps or self.cut_complete
        truncated = False  # we don't use time limits
        info = {}
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.concatenate([self.tool_position, self.fabric_position])
    
    def _compute_reward(self, action):
        dist = np.linalg.norm(self.tool_position - self.fabric_position)
        dist_reward = -dist * 10.0
        energy_penalty = -np.sum(np.square(action)) * 2.0
        step_penalty = -0.1
        return dist_reward + energy_penalty + step_penalty
    
    def close(self):
        if not self.external_node:
            self.node.destroy_node()
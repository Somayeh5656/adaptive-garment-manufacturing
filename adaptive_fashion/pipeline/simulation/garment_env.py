"""
garment_env.py
--------------
SimPy‑based digital twin of a garment factory wrapped as a Gymnasium
environment for reinforcement learning.  Supports both a fixed reorder
point (baseline) and a PPO agent that learns the optimal reorder point.

The environment communicates with a ROS 2 robotic cutting service to
obtain realistic cutting times and waste fractions.  By default, the
service is called **once per day in batch mode**, dramatically reducing
simulation time (≈175× speedup).  The old one‑garment‑at‑a‑time code is
preserved in a comment block for reference or GPU‑heavy setups.
"""

import gymnasium as gym
from gymnasium import spaces
import simpy
import numpy as np
import random
import pandas as pd
import rclpy
from rclpy.node import Node
from fabric_cell_interfaces.srv import ExecuteCut
import threading
import time

# ======================================================================
# Global ROS 2 helpers – initialised once per process
# ======================================================================
_ros_initialized = False
_ros_node = None
_cutting_client = None


def init_ros_client():
    """
    Initialise the ROS 2 context, create a single Node and a Service Client
    for /execute_cut.  A background daemon thread keeps the ROS event loop
    alive without blocking the main simulation.
    """
    global _ros_initialized, _ros_node, _cutting_client
    if not _ros_initialized:
        rclpy.init(args=None)
        _ros_node = Node('garment_env_client')
        _cutting_client = _ros_node.create_client(ExecuteCut, '/execute_cut')
        _ros_initialized = True
        threading.Thread(target=_spin_ros, daemon=True).start()
    return _cutting_client


def _spin_ros():
    """Continuously spin the global ROS node so callbacks are processed."""
    while rclpy.ok():
        rclpy.spin_once(_ros_node, timeout_sec=0.01)


# ======================================================================
# RoboticCuttingStation – wrapper around the ROS 2 cutting service
# ======================================================================
class RoboticCuttingStation:
    """
    Handles communication with the /execute_cut ROS 2 service.
    Falls back to a fast heuristic (0.5 s cycle time, 15 % waste) if the
    service is not available or times out.
    """

    def __init__(self):
        self.client = init_ros_client()
        print("Waiting for /execute_cut service...")
        if self.client.wait_for_service(timeout_sec=10.0):
            print("✅ Successfully connected to ROS 2 Cutting Service!")
        else:
            print("⚠️ ROS 2 service not available, will use fallback")

    def request_cut(self, fabric_type: str = "cotton", batch_size: int = 1):
        """
        Send a batch request to the cutting service.

        Returns
        -------
        (cycle_time_seconds, avg_waste_fraction) or the fallback values.
        """
        try:
            if not self.client.service_is_ready():
                print("Service not ready, using fallback")
                return 0.5, 0.15

            req = ExecuteCut.Request()
            req.fabric_type = fabric_type
            req.batch_size = int(batch_size)

            future = self.client.call_async(req)
            start = time.time()
            while not future.done():
                time.sleep(0.01)
                if time.time() - start > 60.0:
                    print("Cutting service timeout, using fallback")
                    return 0.5, 0.15

            response = future.result()
            if response.success:
                return response.total_cycle_time, response.avg_waste_percentage
            else:
                print("Cutting failed, using fallback")
                return 0.5, 0.15

        except Exception as e:
            print(f"ROS service error: {e} – using fallback")
            return 0.5, 0.15


# ======================================================================
# GarmentFactoryEnv – the main Gymnasium environment
# ======================================================================
class GarmentFactoryEnv(gym.Env):
    """
    Discrete‑event simulation of a three‑station garment factory (cutting,
    sewing, inspection) with inventory management.

    Observations (6‑dim vector):
        [fabric_level, forecast_today, queue_cutting, queue_sewing,
         queue_inspection, day_index]

    Actions (1‑dim continuous [-1, 1]):
        Mapped linearly to a reorder point in [0, 50000] metres of fabric.

    Reward = daily profit / 1000  (revenue – holding – reorder – stockout).
    """

    def __init__(self, forecast_csv: str, sim_days: int = 30,
                 fixed_reorder_point: float = None):
        super().__init__()

        # ----------------------------------------------------------------
        # Load and align the daily demand forecast
        # ----------------------------------------------------------------
        forecast_df = pd.read_csv(forecast_csv, parse_dates=['ds'])
        forecast_df = forecast_df.sort_values('ds')
        self.forecast = forecast_df['yhat'].tolist()
        self.sim_days = min(sim_days, len(self.forecast))

        # ----------------------------------------------------------------
        # Station parameters (times in days, capacities are machine counts)
        # ----------------------------------------------------------------
        self.CUTTING_TIME = 0.005          # fallback when robotics disabled
        self.SEWING_TIME = 0.02
        self.INSPECTION_TIME = 0.005
        self.CUTTING_CAPACITY = 10
        self.SEWING_CAPACITY = 30          # sewing is the bottleneck
        self.INSPECTION_CAPACITY = 8

        # ----------------------------------------------------------------
        # Inventory parameters
        # ----------------------------------------------------------------
        self.INITIAL_FABRIC = 50000        # metres
        self.FABRIC_PER_GARMENT = 2.5      # metres per garment
        self.REORDER_QUANTITY = 50000      # fixed order quantity
        self.LEAD_TIME = 3                 # days until delivery

        # ----------------------------------------------------------------
        # RL‑specific configuration
        # ----------------------------------------------------------------
        self.fixed_reorder_point = fixed_reorder_point  # None → RL active

        # Action space: normalized value that gets mapped to [0, 50000]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,),
                                       dtype=np.float32)

        # Observation space (all non‑negative, clipped to inf for safety)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,),
                                            dtype=np.float32)

        # ----------------------------------------------------------------
        # Robotics integration
        # ----------------------------------------------------------------
        self.robotic_cutter = RoboticCuttingStation()
        self.total_waste = 0.0
        self.daily_waste = 0.0
        self.daily_waste_list = []         # recorded once per day
        self.daily_output_list = []        # recorded once per day

        # ----------------------------------------------------------------
        # Seeding for reproducibility
        # ----------------------------------------------------------------
        random.seed(42)
        np.random.seed(42)

        # Attributes initialised in reset()
        self.env = None
        self.cutting = None
        self.sewing = None
        self.inspection = None
        self.fabric_level = None
        self.cumulative_output = None
        self.daily_output = None
        self.day = None
        self.reorder_point = None
        self.order_pending = None
        self.order_placed_this_day = None
        self.cutting_queue_log = None
        self.sewing_queue_log = None
        self.inspection_queue_log = None
        self.fabric_log = None
        self.time_stamps = None
        self.output_values = None

    # ------------------------------------------------------------------
    # Gymnasium API: reset()
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a fresh SimPy environment and resources each episode
        self.env = simpy.Environment()
        self.cutting = simpy.Resource(self.env, capacity=self.CUTTING_CAPACITY)
        self.sewing = simpy.Resource(self.env, capacity=self.SEWING_CAPACITY)
        self.inspection = simpy.Resource(self.env,
                                         capacity=self.INSPECTION_CAPACITY)

        # State
        self.fabric_level = self.INITIAL_FABRIC
        self.cumulative_output = 0
        self.daily_output = 0
        self.daily_waste = 0.0
        self.day = 0
        self.order_pending = False
        self.order_placed_this_day = False

        # Logs
        self.cutting_queue_log = []
        self.sewing_queue_log = []
        self.inspection_queue_log = []
        self.fabric_log = [(0, self.fabric_level)]
        self.time_stamps = [0]
        self.output_values = [0]
        self.total_waste = 0.0

        # Start background SimPy processes
        self.env.process(self._job_generator())
        self.env.process(self._inventory_manager())

        # Initial reorder point
        self.reorder_point = (self.fixed_reorder_point
                              if self.fixed_reorder_point is not None
                              else 10000)

        time.sleep(0.5)          # let ROS initialise if needed
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Gymnasium API: step()
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Parameters
        ----------
        action : ndarray, shape (1,)
            Value in [-1, 1]; converted to reorder point if RL is active.

        Returns
        -------
        obs, reward, done, truncated, info
        """
        if self.fixed_reorder_point is None:
            # Map [-1, 1] → [0, 50000]
            self.reorder_point = float((action[0] + 1) / 2 * 50000)

        self.order_placed_this_day = False
        self.env.run(until=self.day + 1)          # simulate one full day

        obs = self._get_obs()
        reward = self._compute_reward()

        # Record daily KPIs, then reset the daily accumulators
        self.daily_output_list.append(self.daily_output)
        self.daily_waste_list.append(self.daily_waste)
        self.daily_output = 0

        done = (self.day >= self.sim_days - 1)
        self.day += 1
        return obs, reward, done, False, {}

    # ------------------------------------------------------------------
    # Observation vector
    # ------------------------------------------------------------------
    def _get_obs(self):
        forecast_today = (self.forecast[self.day]
                          if self.day < len(self.forecast) else 0)
        return np.array([
            self.fabric_level,
            forecast_today,
            len(self.cutting.queue),
            len(self.sewing.queue),
            len(self.inspection.queue),
            self.day
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward function (daily profit, scaled by 1/1000)
    # ------------------------------------------------------------------
    def _compute_reward(self):
        revenue = self.daily_output * 50              # 50 € per garment
        holding_cost = self.fabric_level * 0.1        # 0.1 € per metre
        reorder_cost = 1000 if self.order_placed_this_day else 0
        stockout_penalty = 5000 if self.fabric_level < 0 else 0

        profit = revenue - holding_cost - reorder_cost - stockout_penalty
        return profit / 1000.0

    # ==================================================================
    # JOB GENERATOR (batch mode – default)
    # ==================================================================
    # The **old per‑garment generator** (fast but unrealistic when
    # robotics are involved) is preserved at the bottom of this file for
    # reference.  Replace `_job_generator` with that version if you want
    # to compare the two approaches or if you have enough GPU time to
    # call the ROS service for every single garment.

    def _job_generator(self):
        """
        Daily batch job generator.
        - Calls the ROS cutting service ONCE per day with the full daily
          demand as `batch_size`.
        - The service samples up to 5 real cut paths (in the kinematic
          simulation) and extrapolates to the whole batch.
        - Garments are then fed into the sewing/inspection queues.
        - Advances the SimPy clock by exactly 1 day after each batch.
        """
        day = 0
        while day < len(self.forecast):
            daily_demand = int(self.forecast[day])

            if daily_demand > 0:
                # --------------------------------------------------------
                # 1. Call robotic cutting service exactly once today
                # --------------------------------------------------------
                total_cycle_time_sec, avg_waste = self.robotic_cutter.request_cut(
                    fabric_type="cotton",
                    batch_size=daily_demand
                )

                total_fabric_needed = daily_demand * self.FABRIC_PER_GARMENT

                # --------------------------------------------------------
                # 2. Consume fabric and record waste
                # --------------------------------------------------------
                if self.fabric_level >= total_fabric_needed:
                    self.fabric_level -= total_fabric_needed
                    self.total_waste += (avg_waste * daily_demand)
                    self.daily_waste += (avg_waste * daily_demand)
                else:
                    print(f"Day {day}: Stockout! Could not process batch.")

                # --------------------------------------------------------
                # 3. Launch the rest of the factory pipeline for each piece
                # --------------------------------------------------------
                per_unit_cutting_time = total_cycle_time_sec / daily_demand
                for _ in range(daily_demand):
                    self.env.process(
                        self._garment_process_after_cutting(per_unit_cutting_time)
                    )

                self._log_fabric()

            # ------------------------------------------------------------
            # 4. Advance one full day
            # ------------------------------------------------------------
            yield self.env.timeout(1.0)
            day += 1

    # ==================================================================
    # GARMENT PROCESS (post‑cutting)
    # ==================================================================
    def _garment_process_after_cutting(self, cutting_time_per_unit: float):
        """
        Process one garment that has already been "cut" by the batch service.
        `cutting_time_per_unit` is the extrapolated average cutting time in
        seconds; it is converted to days before occupying the cutting resource
        (this keeps queue statistics meaningful).
        """
        cutting_time_days = cutting_time_per_unit / 86400.0

        # Cutting (resource occupied for the extrapolated time)
        with self.cutting.request() as req:
            yield req
            yield self.env.timeout(cutting_time_days)

        # Sewing
        with self.sewing.request() as req:
            yield req
            yield self.env.timeout(self.SEWING_TIME)

        # Inspection
        with self.inspection.request() as req:
            yield req
            yield self.env.timeout(self.INSPECTION_TIME)

        # Garment finished
        self.cumulative_output += 1
        self.daily_output += 1
        self._log_queues()
        self.time_stamps.append(self.env.now)
        self.output_values.append(self.cumulative_output)

    # ==================================================================
    # DEPRECATED: per‑garment process (kept for reference / GPU setups)
    # ==================================================================
    # This version called the ROS service **once per garment**, which
    # accurately simulated individual variability but was ~175× slower.
    # If you have a GPU‑accelerated setup and want to use it, comment out
    # the batch `_job_generator` and `_garment_process_after_cutting`,
    # then uncomment the two sections below.
    #
    # def _job_generator(self):
    #     """Original per‑garment Poisson generator."""
    #     day = 0
    #     while day < len(self.forecast):
    #         rate = self.forecast[day]
    #         iat = random.expovariate(rate) if rate > 0 else 1.0
    #         yield self.env.timeout(iat)
    #         self.env.process(self._garment_process(
    #             f"Garment_{self.env.now:.4f}"))
    #         current_day = int(self.env.now)
    #         if current_day >= len(self.forecast):
    #             break
    #         if current_day != day:
    #             day = current_day
    #
    # def _garment_process(self, name):
    #     """
    #     Original per‑garment process (cuts one piece at a time via ROS).
    #     """
    #     if self.fabric_level < self.FABRIC_PER_GARMENT:
    #         return                         # out of fabric → abort
    #     self.fabric_level -= self.FABRIC_PER_GARMENT
    #     self._log_fabric()
    #
    #     # Call ROS cutting service for this single piece
    #     cycle_time_sec, waste = self.robotic_cutter.request_cut()
    #     cutting_time_days = cycle_time_sec / 86400.0
    #     self.total_waste += waste
    #     self.daily_waste += waste
    #
    #     # Occupied resources
    #     with self.cutting.request() as req:
    #         yield req
    #         yield self.env.timeout(cutting_time_days)
    #     with self.sewing.request() as req:
    #         yield req
    #         yield self.env.timeout(self.SEWING_TIME)
    #     with self.inspection.request() as req:
    #         yield req
    #         yield self.env.timeout(self.INSPECTION_TIME)
    #
    #     self.cumulative_output += 1
    #     self.daily_output += 1
    #     self.time_stamps.append(self.env.now)
    #     self.output_values.append(self.cumulative_output)
    #     self._log_queues()

    # ==================================================================
    # INVENTORY MANAGER (continuous review policy)
    # ==================================================================
    def _inventory_manager(self):
        """
        Background SimPy process that monitors the fabric level and places
        an order when it falls below the current reorder point.  The order
        arrives after LEAD_TIME days.
        """
        while True:
            if not self.order_pending and self.fabric_level < self.reorder_point:
                self.order_pending = True
                self.order_placed_this_day = True
                yield self.env.timeout(self.LEAD_TIME)
                self.fabric_level += self.REORDER_QUANTITY
                self.order_pending = False
                self._log_fabric()
            else:
                yield self.env.timeout(0.1)   # check again soon

    # ==================================================================
    # LOGGING HELPERS
    # ==================================================================
    def _log_fabric(self):
        self.fabric_log.append((self.env.now, self.fabric_level))

    def _log_queues(self):
        self.cutting_queue_log.append((self.env.now, len(self.cutting.queue)))
        self.sewing_queue_log.append((self.env.now, len(self.sewing.queue)))
        self.inspection_queue_log.append((self.env.now,
                                          len(self.inspection.queue)))
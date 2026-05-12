import simpy
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load forecast data
# ------------------------------------------------------------
forecast_df = pd.read_csv("forecast_tft_nlp.csv", parse_dates=['ds'])
forecast_df = forecast_df.sort_values('ds')          # ensure chronological order
forecast = forecast_df['yhat'].tolist()
SIM_DAYS = len(forecast)                              # 30 days
print(f"Loaded forecast for {SIM_DAYS} days.")

# ------------------------------------------------------------
# 2. Simulation parameters
# ------------------------------------------------------------
# Station times (days per garment)
CUTTING_TIME = 0.005
SEWING_TIME = 0.02
INSPECTION_TIME = 0.005

# Station capacities (parallel machines)
CUTTING_CAPACITY = 10
SEWING_CAPACITY = 30          # increased to reduce bottleneck
INSPECTION_CAPACITY = 8

# Inventory parameters
INITIAL_FABRIC = 50000          # meters
FABRIC_PER_GARMENT = 2.5        # meters per garment
REORDER_POINT = 10000            # reorder when fabric drops below this
REORDER_QUANTITY = 50000         # amount to reorder
LEAD_TIME = 3                    # days to receive new fabric

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# 3. Global variables for tracking
# ------------------------------------------------------------
cumulative_output = 0            # total garments finished
time_stamps = [0]                # for cumulative output plot
output_values = [0]

fabric_level = INITIAL_FABRIC    # current fabric stock
fabric_log = [(0, fabric_level)] # time series of inventory

order_pending = False            # is there an outstanding order?

# NEW: Queue length logs
cutting_queue_log = []   # (time, queue length)
sewing_queue_log = []
inspection_queue_log = []

# ------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------
def log_fabric(env):
    """Record current fabric level at this time."""
    fabric_log.append((env.now, fabric_level))

def inventory_manager(env):
    """Monitor fabric and reorder when needed."""
    global fabric_level, order_pending
    while True:
        if not order_pending and fabric_level < REORDER_POINT:
            order_pending = True
            print(f"Time {env.now:.2f}: Reordering fabric. Level: {fabric_level:.0f}")
            yield env.timeout(LEAD_TIME)                 # wait for delivery
            fabric_level += REORDER_QUANTITY
            order_pending = False
            print(f"Time {env.now:.2f}: Fabric delivered. New level: {fabric_level:.0f}")
            log_fabric(env)
        else:
            # Check again after a short interval
            yield env.timeout(0.1)

def garment_process(env, name, cutting, sewing, inspection):
    """Flow of one garment through all stations."""
    global fabric_level, cumulative_output

    # --- Check fabric availability ---
    if fabric_level < FABRIC_PER_GARMENT:
        print(f"Time {env.now:.2f}: Out of fabric! {name} aborted.")
        return
    fabric_level -= FABRIC_PER_GARMENT
    log_fabric(env)

    # --- Cutting ---
    with cutting.request() as req:
        yield req
        yield env.timeout(CUTTING_TIME)

    # --- Sewing ---
    with sewing.request() as req:
        yield req
        yield env.timeout(SEWING_TIME)

    # --- Inspection ---
    with inspection.request() as req:
        yield req
        yield env.timeout(INSPECTION_TIME)

    # Garment finished
    cumulative_output += 1
    time_stamps.append(env.now)
    output_values.append(cumulative_output)

def job_generator(env, forecast, cutting, sewing, inspection):
    """Generate garments according to the daily forecast."""
    day = 0
    while day < len(forecast):
        rate = forecast[day]                     # jobs per day
        iat = random.expovariate(rate)           # interarrival time
        yield env.timeout(iat)

        # Start a new garment
        env.process(garment_process(env, f"Garment_{env.now:.4f}",
                                    cutting, sewing, inspection))

        # Check if day changed
        current_day = int(env.now)
        if current_day >= len(forecast):
            break
        if current_day != day:
            day = current_day

# NEW: Monitor queue lengths
def monitor_queues(env, cutting, sewing, inspection):
    """Record queue lengths at regular intervals."""
    while True:
        cutting_queue_log.append((env.now, len(cutting.queue)))
        sewing_queue_log.append((env.now, len(sewing.queue)))
        inspection_queue_log.append((env.now, len(inspection.queue)))
        yield env.timeout(0.1)   # sample every 0.1 days

# ------------------------------------------------------------
# 5. Create SimPy environment and start processes
# ------------------------------------------------------------
env = simpy.Environment()

# Resources (stations)
cutting = simpy.Resource(env, capacity=CUTTING_CAPACITY)
sewing = simpy.Resource(env, capacity=SEWING_CAPACITY)
inspection = simpy.Resource(env, capacity=INSPECTION_CAPACITY)

# Start the generator, inventory manager, and queue monitor
env.process(job_generator(env, forecast, cutting, sewing, inspection))
env.process(inventory_manager(env))
env.process(monitor_queues(env, cutting, sewing, inspection))

print(f"Starting simulation for {SIM_DAYS} days...")
env.run(until=SIM_DAYS)

print(f"Simulation finished. Total garments produced: {cumulative_output}")

# ------------------------------------------------------------
# 6. Plot results
# ------------------------------------------------------------
# Ideal cumulative forecast
ideal_cumulative = np.cumsum(forecast)
ideal_time = np.arange(1, SIM_DAYS + 1)

plt.figure(figsize=(14, 8))

# Cumulative production vs forecast
plt.subplot(2,2,1)
plt.plot(ideal_time, ideal_cumulative, 'y--', linewidth=2, label='Ideal Forecast')
plt.step(time_stamps, output_values, where='post', color='green', linewidth=2,
         label='Actual Output')
plt.xlabel('Time (days)')
plt.ylabel('Garments Produced')
plt.title('Cumulative Production')
plt.legend()
plt.grid(True)

# Fabric inventory over time
plt.subplot(2,2,2)
inv_time, inv_level = zip(*fabric_log)
plt.step(inv_time, inv_level, where='post', color='blue', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Fabric (meters)')
plt.title('Fabric Inventory')
plt.grid(True)

# Queue lengths over time
plt.subplot(2,2,3)
q_time, q_len = zip(*cutting_queue_log)
plt.plot(q_time, q_len, label='Cutting', linewidth=1.5)
q_time, q_len = zip(*sewing_queue_log)
plt.plot(q_time, q_len, label='Sewing', linewidth=1.5)
q_time, q_len = zip(*inspection_queue_log)
plt.plot(q_time, q_len, label='Inspection', linewidth=1.5)
plt.xlabel('Time (days)')
plt.ylabel('Queue Length (waiting jobs)')
plt.title('Queue Lengths')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# run_simulation.py
from garment_env import GarmentFactoryEnv

# Create environment with fixed reorder point (no RL)
env = GarmentFactoryEnv("forecast_tft_nlp.csv", sim_days=30, fixed_reorder_point=10000)

# Run one episode
obs, _ = env.reset()
done = False
while not done:
    # Dummy action (ignored because fixed_reorder_point is set)
    obs, reward, done, _, _ = env.step([0])
    print(f"Day {env.day}: reward={reward:.2f}")

# After simulation, print final stats
print(f"\n--- Simulation finished ---")
print(f"Total garments produced: {env.cumulative_output}")
print(f"Final fabric inventory: {env.fabric_level:.1f} m")
print(f"Total waste accumulated: {env.total_waste:.2%}")   # waste percentage over all garments
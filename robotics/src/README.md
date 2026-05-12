```markdown
# fashion_robotics_ws – Robotic Cutting Cell

Part of the **Adaptive Garment Manufacturing** thesis.

This folder (`robotics`) is a ROS 2 workspace. It contains four packages:

- `fabric_cell_description` – URDF, world, controller config
- `fabric_cell_bringup` – launch file for Gazebo simulation
- `fabric_cell_control` – perception (YOLOS) and cutting service (RL)
- `fabric_cell_interfaces` – custom ROS service definition

## Prerequisites

- Ubuntu 24.04
- ROS 2 Jazzy (desktop)
- Gazebo Harmonic (`ros-jazzy-ros-gz-sim`)
- `ros-jazzy-ros-gz-bridge`
- `ros-jazzy-ros2-control`, `ros-jazzy-ros2-controllers`, `ros-jazzy-gz-ros2-control`
- Python 3.12

## Setup

1. **Install system packages** (ROS 2, Gazebo, etc.):
   ```bash
   sudo apt install ros-jazzy-desktop ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
                    ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-gz-ros2-control xvfb
   ```

2. **Get the code**:
   ```bash
   git clone https://github.com/yourusername/adaptive-garment-manufacturing.git
   cd adaptive-garment-manufacturing/robotics
   ```

3. **Set up a Python virtual environment** (with access to system site‑packages so that `rclpy` is available):
   ```bash
   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   pip install -r requirements_ros.txt
   ```

4. **Build the workspace**:
   ```bash
   source /opt/ros/jazzy/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```
   (Add the last two source lines to your `~/.bashrc` if you want them automatically available.)

## Run the Full Simulation

Start the main simulation with the launch file:

```bash
# if you haven't already sourced the environment
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source venv/bin/activate

ros2 launch fabric_cell_bringup gazebo_simulation.launch.py
```

In **separate terminals** (with the same environment sourced):

- **Perception node**:
  ```bash
  ros2 run fabric_cell_control perception_node
  ```
- **Cutting service**:
  ```bash
  ros2 run fabric_cell_control cutting_service
  ```
- **Quick test of the service**:
  ```bash
  ros2 service call /execute_cut fabric_cell_interfaces/srv/ExecuteCut "{fabric_type: 'cotton', batch_size: 1}"
  ```

## Training the Cutting Agent (if model not provided)

If the pre‑trained model `vision_cutting_agent_final.zip` is missing, train it:

```bash
cd robotics   # workspace root
source venv/bin/activate
python3 src/fabric_cell_control/scripts/train_vision_cutting_rl.py
```

After training, the model file is created in the same directory. The cutting service expects it there; adjust the path in `cutting_service.py` if you move it.

## Notes

- The Gazebo robot does **not** move due to headless VM limitations; only the camera and kinematic simulation are used.
- For the full manufacturing digital twin, see the `pipeline/` folder in the parent repository.

## Troubleshooting

- **Missing plugin** – ensure `ros-jazzy-gz-ros2-control` is installed.
- **Segfault/OpenGL** – launch Gazebo with `xvfb-run -a` if running headless, or use the headless rendering flag already in the launch file.
- **Service not available** – verify that the cutting service node is running and all environments are sourced.
```
-------------------------

ssh -Y ubuntu@eunoia-vm
source /opt/ros/humble/setup.bash
source ~/fashion_robotics_ws/install/setup.bash
ros2 launch fabric_cell_bringup gazebo_simulation.launch.py

ssh -Y ubuntu@eunoia-vm
source /opt/ros/humble/setup.bash
source ~/fashion_robotics_ws/install/setup.bash
ros2 topic list

ssh -Y ubuntu@eunoia-vm
ros2 run fabric_cell_control cutting_service

ssh -Y ubuntu@eunoia-vm
ros2 run fabric_cell_control perception_node

ssh -Y ubuntu@eunoia-vm
cd ~/adaptive_fashion
source ~/adaptive_fashion/venv/bin/activate
python evaluate_rl.py



cd ~/fashion_robotics_ws
rm -rf build install log
colcon build --symlink-install
source install/setup.bash


source ~/ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/fashion_robotics_ws/install/setup.bash
cd ~/fashion_robotics_ws/src/fabric_cell_control/scripts
python train_vision_cutting_rl.py



--------
cutting service 

cd ~/fashion_robotics_ws
rm -rf build/fabric_cell_control install/fabric_cell_control
colcon build --packages-select fabric_cell_control --symlink-install
source install/setup.bash

ros2 run fabric_cell_control cutting_service
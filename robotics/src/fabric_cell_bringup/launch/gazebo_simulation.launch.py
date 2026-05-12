"""
Launch file for the robotic cutting cell simulation.
Starts Gazebo (headless), the robot state publisher, spawns the robot,
loads the two ros2_control controllers, and bridges camera and joint topics.

Important:
  - The ros2_control_node (controller_manager) must already be running
    before the spawner nodes are executed.  This is normally done by the
    companion launch file (gazebo_control.launch.py) or manually.
  - Controller names match the entries in controller.yaml.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # --- Paths ---
    pkg_gazebo_ros = get_package_share_directory('ros_gz_sim')
    pkg_fabric_cell = get_package_share_directory('fabric_cell_description')
    world_path = os.path.join(pkg_fabric_cell, 'worlds', 'cutting_cell.world')
    urdf_path = os.path.join(pkg_fabric_cell, 'urdf', 'cutting_cell.urdf')

    # --- 1. Gazebo with headless rendering (no GUI) ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': world_path + ' --headless-rendering'
        }.items()
    )

    # --- 2. Robot state publisher ---
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # --- 3. Spawn the robot in Gazebo ---
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'cutting_robot',
            '-x', '0.5',
            '-y', '0.0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # --- 4. Load controllers (must match controller.yaml) ---
    #      The ros2_control_node itself is assumed to be running.
    load_joint_state_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],     # corrected name
        output='screen'
    )
    load_joint_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_velocity_controller'],
        output='screen'
    )

    # --- 5. ROS–Gazebo bridges ---
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image'],
        output='screen'
    )
    joint_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/cutting_cell/model/cutting_robot/joint_state'
            '@sensor_msgs/msg/JointState[gz.msgs.Model'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        load_joint_state_controller,
        load_joint_velocity_controller,
        camera_bridge,
        joint_bridge
    ])
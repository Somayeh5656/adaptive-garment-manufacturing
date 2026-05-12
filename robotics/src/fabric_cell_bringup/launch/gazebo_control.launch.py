"""
Launch file for the Gazebo simulation of the robotic cutting cell.
Starts Gazebo (headless via xvfb), the robot state publisher, spawns the
robot, sets up ros2_control, and bridges the camera and joint topics.

Environment variables force software rendering – remove them if you have
a proper GPU.
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package paths
    pkg_fabric_cell = get_package_share_directory('fabric_cell_description')
    world_path = os.path.join(pkg_fabric_cell, 'worlds', 'cutting_cell.world')
    urdf_path = os.path.join(pkg_fabric_cell, 'urdf', 'cutting_cell.urdf')

    # --- Environment variables for software rendering (headless VM) ---
    # Remove or comment these if you have a real GPU.
    set_ld = SetEnvironmentVariable('LD_LIBRARY_PATH', '/opt/ros/jazzy/lib:$LD_LIBRARY_PATH')
    set_gazebo_plugin = SetEnvironmentVariable('GAZEBO_PLUGIN_PATH', '/opt/ros/jazzy/lib:$GAZEBO_PLUGIN_PATH')
    set_sw = SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1')
    set_gallium = SetEnvironmentVariable('GALLIUM_DRIVER', 'llvmpipe')
    set_qt = SetEnvironmentVariable('QT_QPA_PLATFORM', 'offscreen')

    # --- Gazebo server (headless, via xvfb) ---
    gazebo = ExecuteProcess(
        cmd=['xvfb-run', '-a', 'gz', 'sim', '-s', '-r', world_path],
        output='screen',
        name='gazebo'
    )

    # --- Robot state publisher ---
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        arguments=[urdf_path],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # --- Spawn the robot in Gazebo ---
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'cutting_robot',
                   '-x', '0.5', '-y', '0.0', '-z', '0.1'],
        output='screen'
    )

    # --- Controller manager (ros2_control) ---
    cm = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Spawn the two controllers
    spawn_joint_state = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )
    spawn_vel = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_velocity_controller'],
        output='screen'
    )

    # --- ROS‑Gazebo bridges ---
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image'],
        output='screen'
    )
    joint_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/world/cutting_cell/model/cutting_robot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model'],
        output='screen'
    )

    return LaunchDescription([
        set_ld, set_gazebo_plugin, set_sw, set_gallium, set_qt,
        gazebo, rsp, spawn, cm, spawn_joint_state, spawn_vel,
        camera_bridge, joint_bridge
    ])
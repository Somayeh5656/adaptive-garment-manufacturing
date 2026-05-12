import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('fabric_cell_description')
    
    # Path to URDF file
    urdf_file = os.path.join(pkg_dir, 'urdf', 'cutting_cell.urdf')
    
    # Read the URDF file
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}],
        arguments=[urdf_file]
    )
    
    # Joint state publisher GUI (for manual joint control)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )
    
    # RViz2 for visualization
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_dir, 'config', 'view_robot.rviz')]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
 # Bridge for camera
camera_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=['/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image'],
    output='screen'
)

# Bridge for joint states (topic name may differ; check with `gz topic -l`)
joint_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=['/world/cutting_cell/model/cutting_robot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model'],
    output='screen'
)
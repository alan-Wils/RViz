#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    rviz_config = os.path.join(os.path.expanduser("~"), "slam_rviz_config.rviz")

    # Auto-generate a simple RViz layout if one does not exist.
    if not os.path.exists(rviz_config):
        with open(rviz_config, "w") as f:
            f.write(
                """
Panels:
  - Class: rviz/Displays
    Name: Displays

Visualization Manager:
  Class: ""
  Displays:
    - Name: Grid
      Class: rviz/Grid
      Enabled: true

    - Name: Map
      Class: rviz/Map
      Topic: /map
      Enabled: true

    - Name: Laser
      Class: rviz/LaserScan
      Topic: /scan
      Enabled: true

    - Name: RobotPose
      Class: rviz/Pose
      Topic: /slam_toolbox/pose
      Enabled: true
      Shape: Arrow
      Color: 255; 25; 25
      Alpha: 1.0
      Shaft Length: 1.0
      Shaft Radius: 0.05
      Head Length: 0.3
      Head Radius: 0.1

    - Name: Path
      Class: rviz/Path
      Topic: /plan
      Enabled: true

  Global Options:
    Fixed Frame: map
    Frame Rate: 30

  Tools:
    - Class: rviz/Interact
    - Class: rviz/Select
    - Class: rviz/Measure
    - Class: rviz/SetInitialPose
    - Class: rviz/SetGoal
"""
            )

    nav2_launch = PathJoinSubstitution(
        [get_package_share_directory("nav2_bringup"), "launch", "bringup_launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time", default_value="false", description="Use simulation (Gazebo) clock"
            ),
            Node(
                package="rplidar_ros",
                executable="rplidar_node",
                name="rplidar_node",
                parameters=[
                    {
                        "serial_port": "/dev/ttyUSB0",
                        "serial_baudrate": 115200,
                        "frame_id": "laser",
                        "angle_compensate": True,
                    }
                ],
                output="screen",
            ),
            Node(
                package="slam_toolbox",
                executable="sync_slam_toolbox_node",
                name="slam_toolbox",
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            # Static TF: base_link -> laser (replace with your actual transform)
            Node(
                package="rviz_click_nav",
                executable="lidar_tf_broadcaster",
                name="lidar_tf_broadcaster",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            # Bringup Nav2 with minimal overrides; tune params for your robot
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_launch),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),
            ExecuteProcess(cmd=["rviz2", "-d", rviz_config], output="screen"),
        ]
    )


if __name__ == "__main__":
    generate_launch_description()

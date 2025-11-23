# ROS 2 Click-to-Navigate Starter

This repository contains a minimal example for running SLAM + navigation with RViz on a Raspberry Pi 5 using ROS 2. It demonstrates how to launch RPLIDAR, SLAM Toolbox, Nav2, and RViz with a default configuration that enables "2D Nav Goal" click-to-navigate.

The setup targets Ubuntu on a Raspberry Pi 5 but can be adapted for other SBCs. Adjust device names, frames, and parameters for your specific robot (drive kinematics, odometry source, sensor frames, etc.).

## Launch file
The example launch file lives at [`click_to_navigate.launch.py`](click_to_navigate.launch.py). It will:

1. Start the RPLIDAR node on `/dev/ttyUSB0`.
2. Run SLAM Toolbox in synchronous mode.
3. Launch Nav2 using the default bringup launch file and a small parameter override.
4. Start a static TF broadcaster from `base_link` → `laser` (replace the transform with your actual sensor mounting transform).
5. Auto-generate a simple RViz layout if one is not present at `~/slam_rviz_config.rviz` and start RViz using it.

Use the RViz tool **2D Nav Goal** to click a destination. Nav2 will plan and drive the robot there once localization and TF are healthy.

## Setup steps
1. Install ROS 2 Humble or newer and workspace dependencies:
   ```bash
   sudo apt update
   sudo apt install ros-${ROS_DISTRO}-slam-toolbox ros-${ROS_DISTRO}-nav2-bringup ros-${ROS_DISTRO}-rplidar-ros
   ```

2. Place the files from this repo into your ROS 2 workspace (e.g., `~/ros2_ws/src/rviz_click_nav`). A typical layout looks like:

   ```text
   ros2_ws/
     src/
       rviz_click_nav/
         package.xml            # create a minimal package manifest
         setup.py / setup.cfg    # standard Python package metadata
         resource/               # optional, can hold package marker file
         launch/
           click_to_navigate.launch.py
         scripts/
           lidar_tf_broadcaster.py
         README.md
   ```

   You can also drop the two Python files directly under an existing package and add execution permissions to `lidar_tf_broadcaster.py` (`chmod +x`). The launch file will still generate `~/slam_rviz_config.rviz` at runtime.

3. Build the workspace and source the overlay:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

4. Run the launch file:
   ```bash
   ros2 launch rviz_click_nav click_to_navigate.launch.py
   ```

   > If you placed the files directly in this repository without making a package, use:
   > ```bash
   > ros2 launch /path/to/repo/click_to_navigate.launch.py
   > ```

## Notes
- Replace the static transform in `lidar_tf_broadcaster.py` with the correct translation/rotation between your base and lidar frames.
- For a wheeled base, ensure you have an odometry source that publishes `odom` → `base_link` (e.g., `diff_drive_controller`, `robot_localization`).
- The example Nav2 parameters are minimal; tune costmaps, planners, and controllers for your platform.
- Use `SetInitialPose` and `2D Nav Goal` tools in RViz when running mapping + navigation.

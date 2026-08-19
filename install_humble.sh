#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

sudo apt update
sudo apt install -y   ros-humble-desktop   ros-humble-moveit   ros-humble-moveit-resources-panda-moveit-config   ros-humble-moveit-resources-panda-description   ros-humble-pilz-industrial-motion-planner   ros-humble-moveit-simple-controller-manager   python3-colcon-common-extensions   python3-rosdep   python3-yaml

echo
echo "ROS/MoveIt packages installed."
echo "Now install MuJoCo for the Python interpreter that can import rclpy."
echo 'Test with: python3 -c "import rclpy; import mujoco; print(mujoco.__version__)"'

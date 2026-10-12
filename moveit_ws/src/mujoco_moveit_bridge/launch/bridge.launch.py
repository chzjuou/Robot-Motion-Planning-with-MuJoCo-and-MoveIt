from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scene_xml", default_value="wrs.xml"),
        DeclareLaunchArgument("panda_xml", default_value="panda.xml"),
        DeclareLaunchArgument("viewer", default_value="true"),
        Node(
            package="mujoco_moveit_bridge",
            executable="bridge",
            output="screen",
            parameters=[{
                "scene_xml": LaunchConfiguration("scene_xml"),
                "panda_xml": LaunchConfiguration("panda_xml"),
                "viewer": LaunchConfiguration("viewer"),
            }],
        ),
    ])

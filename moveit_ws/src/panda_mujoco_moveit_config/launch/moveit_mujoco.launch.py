import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    this_pkg = get_package_share_directory("panda_mujoco_moveit_config")
    panda_pkg = get_package_share_directory("moveit_resources_panda_moveit_config")

    moveit_config = (
        MoveItConfigsBuilder(
            "moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(
            file_path=os.path.join(this_pkg, "config", "moveit_controllers.yaml")
        )
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )

    pilz_limits = load_yaml(
        os.path.join(this_pkg, "config", "pilz_cartesian_limits.yaml")
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            pilz_limits,
            {
                "moveit_manage_controllers": False,
                "use_sim_time": False,
                "trajectory_execution.allowed_execution_duration_scaling": 4.0,
                "trajectory_execution.allowed_goal_duration_margin": 8.0,
                "trajectory_execution.allowed_start_tolerance": 0.05,
            },
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )
    
    world_to_panda = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_link0",
        output="screen",
        arguments=[
            "--x", "0.25",
            "--y", "0.45",
            "--z", "0.8",
            "--qx", "0.0",
            "--qy", "0.0",
            "--qz", "-0.70710678",
            "--qw", "0.70710678",
            "--frame-id", "world",
            "--child-frame-id", "panda_link0",
        ],
    )

    rviz_candidates = [
        os.path.join(panda_pkg, "launch", "moveit.rviz"),
        os.path.join(panda_pkg, "config", "moveit.rviz"),
    ]
    rviz_config = next((p for p in rviz_candidates if os.path.exists(p)), "")

    rviz_args = ["-d", rviz_config] if rviz_config else []

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            pilz_limits,
        ],
    )
    
    planner_server = Node(
        package="panda_moveit_planner",
        executable="planner_server",
        name="panda_moveit_planner",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {
                "use_sim_time": False,
            },
        ],
    )

    return LaunchDescription([
        world_to_panda,
        robot_state_publisher,
        move_group,
        planner_server,
        rviz,
    ])

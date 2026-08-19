# Robot Motion Planning with MuJoCo and MoveIt

Requirement: Ubuntu 22.04 + ROS 2 Humble + MoveIt + Panda + MuJoCo

Architecture:

```text
MoveIt 2 Humble
  ├─ OMPL / RRTConnect        free-space motion
  └─ Pilz LIN                 exact Cartesian straight-line motion
              |
              | control_msgs/action/FollowJointTrajectory
              v
mujoco_moveit_bridge
              |
              | panda_joint1..7 -> panda2_joint1..7
              v
MuJoCo position actuators
```

MuJoCo publishes:

```text
/joint_states
/mujoco/fingertip_center
```

MuJoCo provides:

```text
/panda_arm_controller/follow_joint_trajectory
/mujoco/gripper
/mujoco/wait_until_stopped
```

`/mujoco/wait_until_stopped` is important for ROS 2 Humble's Pilz `LIN`: Pilz requires a zero-velocity start state.

---

## 1. Platform

Use:

```text
Ubuntu 22.04 LTS
ROS 2 Humble
Python 3.10
```

Do not install ROS 2 Jazzy on this machine.

---

## 2. Install

From this workspace root:

```bash
chmod +x install_humble.sh
./install_humble.sh
```

Then install MuJoCo into a Python interpreter that can also import ROS 2 `rclpy`.

First check:

```bash
source /opt/ros/humble/setup.bash
python3 -c "import rclpy; print('rclpy OK')"
```

Then install/test MuJoCo:

```bash
python3 -m pip install --user mujoco
python3 -c "import rclpy, mujoco; print('rclpy OK, MuJoCo', mujoco.__version__)"
```

If the second command fails, do not create a separate conda environment yet. The bridge process must be able to import both `rclpy` and `mujoco`.

---

## 3. Workspace

Copy /moveit_ws and /mujoco_pj to the home directory.

There are four packages in the ROS workspace

```text
~/moveit_ws/
└── src/
    ├── mujoco_moveit_bridge
    ├── panda_mujoco_moveit_config
    ├── panda_moveit_planner
    └── panda_moveit_task
```

Build:

```bash
cd ~/moveit_ws
source /opt/ros/humble/setup.bash

sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash
```

---

## Launch Terminal 1: MuJoCo bridge

```bash
source /opt/ros/humble/setup.bash
source ~/moveit_ws/install/setup.bash

ros2 launch mujoco_moveit_bridge bridge.launch.py \
  scene_xml:=/home/$USER/mujoco_pj/my_scene.xml \
  panda_xml:=/home/$USER/mujoco_pj/panda.xml
```

Open another terminal and check:

```bash
ros2 topic echo /joint_states --once
```

Expected initial positions approximately:

```text
0
-0.785
0
-2.356
0
1.571
0.785
```

Check the MoveIt execution action:

```bash
ros2 action list | grep follow_joint_trajectory
```

Expected:

```text
/panda_arm_controller/follow_joint_trajectory
```

Check fingertip center:

```bash
ros2 topic echo /mujoco/fingertip_center --once
```

Check gripper:

```bash
ros2 service call /mujoco/gripper std_srvs/srv/SetBool "{data: true}"
ros2 service call /mujoco/gripper std_srvs/srv/SetBool "{data: false}"
```

Check zero-velocity service:

```bash
ros2 service call /mujoco/wait_until_stopped std_srvs/srv/Trigger "{}"
```

---

## Launch Terminal 2: MoveIt 2 + RViz

```bash
source /opt/ros/humble/setup.bash
source ~/moveit_ws/install/setup.bash

ros2 launch panda_mujoco_moveit_config moveit_mujoco.launch.py
```

This launches:

```text
robot_state_publisher
move_group
RViz2
```

It intentionally does NOT start Panda FakeSystem or ros2_control.

MoveIt reads the real joint state from the MuJoCo bridge and sends execution trajectories directly to:

```text
/panda_arm_controller/follow_joint_trajectory
```

Open another terminal and check:

```bash
ros2 param get /move_group planning_pipelines
```

Expected to include:

```text
ompl
pilz_industrial_motion_planner
```

Check Pilz plugin:

```bash
ros2 param get /move_group pilz_industrial_motion_planner.planning_plugin
```

Expected:

```text
pilz_industrial_motion_planner/CommandPlanner
```

Check Cartesian limits:

```bash
ros2 param list /move_group | grep cartesian_limits
```

Expected:

```text
robot_description_planning.cartesian_limits.max_trans_vel
robot_description_planning.cartesian_limits.max_trans_acc
robot_description_planning.cartesian_limits.max_trans_dec
robot_description_planning.cartesian_limits.max_rot_vel
```

If these parameters do not appear, do NOT run the pick-place program.

---

## Launch Terminal 3: Pick and place

```bash
source /opt/ros/humble/setup.bash
source ~/moveit_ws/install/setup.bash

ros2 run panda_moveit_task pick_place --ros-args --params-file ~/moveit_ws/src/panda_moveit_task/config/task.yaml
```

Sequence:

```text
1  OMPL      -> above grasp
2            -> open
3  Pilz LIN  -> straight down to grasp
4-6          -> close/grasp
7  Pilz LIN  -> straight +5 cm Z lift
9  Pilz LIN  -> horizontal translation
10 Pilz LIN  -> strict -4.4 cm world-Z placement
12           -> release
13 OMPL      -> ready
```

Target values are from the original MuJoCo script:

```text
GRASP_POS = [0.25, 0.12, 0.85]
ABOVE_GRASP = [0.25, 0.12, 0.95]
LIFT_HEIGHT = 0.05
horizontal = [-0.19, +0.03, 0.0]
PLACE_DOWN_DISTANCE = 0.044
```

The settings can be modified in ~/moveit_ws/src/panda_moveit_task/config/task.yaml

---

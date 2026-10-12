# Task 1 — WRS Parts Assembly with MuJoCo and MoveIt

Task 1 demonstrates robot motion planning and execution for WRS parts assembly using **ROS 2 Humble**, **MoveIt 2**, and **MuJoCo**. The task is divided into three subtasks: **L-shaped part assembly**, **motor assembly**, and **pulley assembly**.

## 1. Environment and project structure

Tested/targeted software environment described by the original project:

- Ubuntu 22.04 LTS
- ROS 2 Humble
- MoveIt 2 Humble
- Python with both `rclpy` and `mujoco` available

Keep the repository's relative directory structure:

```text
<repository-root>/
├── README.md
├── install_humble.sh
├── mujoco_pj/
│   ├── wrs.xml
│   ├── panda.xml
│   └── assets/ ...
└── moveit_ws/
    └── src/
        ├── mujoco_moveit_bridge/
        ├── panda_mujoco_moveit_config/
        ├── panda_moveit_planner/
        └── panda_moveit_task/
            ├── run_tasks_all.sh
            ├── run_subtask_1.sh
            ├── run_subtask_2.sh
            ├── run_subtask_3.sh
            └── config/
```

The Bridge and Screw modules include XML path-resolution logic for the repository layout, so the commands below do not require hard-coded paths to `wrs.xml` or `panda.xml`.

## 2. Rebuild the ROS 2 workspace

Open a terminal at the **repository root**, then run:

```bash
cd /path/to/Robot-Motion-Planning-with-MuJoCo-and-MoveIt
source /opt/ros/humble/setup.bash
cd moveit_ws

# First build, or after changing package dependencies:
rosdep install --from-paths src --ignore-src -r -y

# Rebuild after cloning or changing Python/launch/config files:
colcon build --symlink-install
source install/setup.bash
```

Replace `/path/to/Robot-Motion-Planning-with-MuJoCo-and-MoveIt` with your local repository directory.

**After rebuilding, open three separate terminals** and start the components **in the order below**.

## 3. Run Task 1 in three terminals

For convenience, in **each** terminal set `WS` to the absolute path of your workspace:

```bash
export WS="/path/to/Robot-Motion-Planning-with-MuJoCo-and-MoveIt/moveit_ws"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
```

### Terminal 1 — MuJoCo bridge

Start the MuJoCo simulation and its ROS 2 trajectory-execution bridge:

```bash
ros2 launch mujoco_moveit_bridge bridge.launch.py \
  scene_xml:=/home/$USER/mujoco_pj/wrs.xml \
  panda_xml:=/home/$USER/mujoco_pj/panda.xml
```

The bridge loads the WRS scene and Panda robot XML files. **Wait until the bridge is running before starting the next component.** It provides `/joint_states`, the trajectory action `/panda_arm_controller/follow_joint_trajectory`, and MuJoCo gripper and motion-state services.

### Terminal 2 — MoveIt 2 and RViz

In a **new terminal**, repeat the environment setup:

```bash
export WS="/path/to/Robot-Motion-Planning-with-MuJoCo-and-MoveIt/moveit_ws"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

ros2 launch panda_mujoco_moveit_config moveit_mujoco.launch.py
```

This starts MoveIt 2 (`move_group`), the robot state publisher, and RViz. Wait for MoveIt to initialize before executing the assembly sequence. Motion planning uses OMPL for free-space motion and Pilz `LIN` for linear Cartesian motion.

### Terminal 3 — Execute the full Task 1 sequence

In a **third terminal**, repeat the environment setup, then invoke the full task runner:

```bash
export WS="/path/to/Robot-Motion-Planning-with-MuJoCo-and-MoveIt/moveit_ws"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

bash "$WS/src/panda_moveit_task/run_tasks_all.sh"
```

Keep Terminals 1 and 2 running while Terminal 3 executes the task. The runner resolves its workspace and configuration directory from its own location, so you do not need to change the terminal's current directory to the repository root.

## 4. Run an individual subtask

Task 1 contains three independently callable subtask runners. **Use one of these commands in Terminal 3 instead of `run_tasks_all.sh`** if you want to execute a specific subtask:

| Subtask | Assembly stage | Script |
|---|---|---|
| 1 | L-shaped part assembly | `run_subtask_1.sh` |
| 2 | Motor assembly | `run_subtask_2.sh` |
| 3 | Pulley assembly | `run_subtask_3.sh` |

```bash
# Subtask 1 — L-shaped part assembly
bash "$WS/src/panda_moveit_task/run_subtask_1.sh"

# Subtask 2 — Motor assembly
bash "$WS/src/panda_moveit_task/run_subtask_2.sh"

# Subtask 3 — Pulley assembly
bash "$WS/src/panda_moveit_task/run_subtask_3.sh"
```

Run **one command at a time**. The individual subtasks may assume an appropriate initial simulation state or previously assembled parts; starting a later subtask from a fresh scene may not reproduce the state reached by the full sequence. Restart the simulation when a clean initial state is required.

## 5. Robot motion primitives

Task 1 is organized around five robot motion primitive types:

| Primitive | Purpose |
|---|---|
| `pick_place` | Pick up a part, transport it, and place it at its target pose |
| `regrasp` | Adjust the grasp or reorient a held part for subsequent assembly |
| `screw` | Perform screw-related manipulation or fastening actions |
| `move` | Move the robot between target poses or intermediate configurations |
| `return` | Return the robot to a designated pose or ready state |

The implementations are provided in the `panda_moveit_task` ROS 2 package. The numbered YAML files in its `config/` directory provide parameters for individual task steps.

## 6. Current status and known limitations

- The workflow is divided into **L-shaped part**, **motor**, and **pulley** assembly stages, with both full-sequence and per-subtask launch scripts.
- **Incomplete:** The pin insertion and fixation operations underneath the pulley still need further development and validation.
- Successful workspace compilation does not guarantee successful physical/kinematic execution. Check the MuJoCo simulation, MoveIt planning output, and task logs when testing each stage.

---

**Recommended startup order:** **Build workspace → Terminal 1 (Bridge) → Terminal 2 (MoveIt/RViz) → Terminal 3 (full Task 1 or one subtask).**

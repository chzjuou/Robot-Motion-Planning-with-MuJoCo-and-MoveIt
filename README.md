# Task 2 — WRS Parts Assembly with MuJoCo and MoveIt

Task 2 extends the robotic assembly framework developed in Task 1 using ROS 2 Humble, MoveIt 2, and MuJoCo.

## 1. Environment Setup and Workspace Build

The software environment, dependency installation, workspace build, and simulation startup procedures are the same as those described in [Task 1](https://github.com/chzjuou/Robot-Motion-Planning-with-MuJoCo-and-MoveIt/tree/Task1).

Please refer to the Task 1 README for detailed instructions.

After switching to the Task2 branch, rebuild the ROS 2 workspace before running the task.

## 2. Task 2 Execution

Follow the same three-terminal startup procedure as Task 1:

- **Terminal 1:** MuJoCo Bridge (`mujoco_moveit_bridge`)
- **Terminal 2:** MoveIt 2 and RViz (`panda_mujoco_moveit_config`)
- **Terminal 3:** Task 2 execution

While keeping Terminals 1 and 2 running, execute the Task 2 sequence in Terminal 3:

```bash
export WS="/path/to/repository/moveit_ws"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

bash "$WS/src/panda_moveit_task/run_tasks_all.sh"
```

Although the execution script has the same filename as in Task 1, its contents have been updated for Task 2.

The Task 2 sequence consists of 28 motion steps, configured through YAML files numbered from 38 to 65.

## 3. Robot Motion Primitives

The framework provides seven robot motion primitives:

| Primitive | Description |
|---|---|
| `pick_place` | Pick up, transport, and place a component. |
| `regrasp` | Adjust the grasp or orientation of a component. |
| `screw` | Perform screw manipulation and fastening operations. |
| `move` | Move the robot to a specified target pose. |
| `return` | Return the robot to a designated reference pose. |
| `rotate` | Rotate around the attached component's center axis, detach, and retreat. |
| `push` | Perform contact-aware linear pushing and retreat. |

## 4. Task Configuration

The execution sequence is defined in:

```text
moveit_ws/src/panda_moveit_task/run_tasks_all.sh
```

Individual motion parameters are stored in:

```text
moveit_ws/src/panda_moveit_task/config/
```

The assembly sequence and motion parameters can be modified through the execution script and the corresponding YAML configuration files.

---

For installation, workspace compilation, and detailed simulation startup instructions, please refer to the [Task 1 README](https://github.com/chzjuou/Robot-Motion-Planning-with-MuJoCo-and-MoveIt/tree/Task1).

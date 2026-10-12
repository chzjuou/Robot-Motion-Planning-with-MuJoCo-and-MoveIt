#!/usr/bin/env bash

# Resolve ROS 2 workspace from this script's location instead of a user-specific path.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${MOVEIT_WS:-}"
if [ -z "$WORKSPACE" ]; then
    SEARCH_DIR="$SCRIPT_DIR"
    while [ "$SEARCH_DIR" != / ]; do
        if [ -f "$SEARCH_DIR/src/panda_moveit_task/package.xml" ] || [ -f "$SEARCH_DIR/src/panda_moveit_task/setup.py" ]; then
            WORKSPACE="$SEARCH_DIR"
            break
        fi
        SEARCH_DIR="$(dirname "$SEARCH_DIR")"
    done
fi
if [ -z "$WORKSPACE" ]; then
    echo "ERROR: Cannot locate moveit_ws. Set MOVEIT_WS to your ROS 2 workspace directory."
    exit 1
fi
CONFIG_DIR="$WORKSPACE/src/panda_moveit_task/config"
PACKAGE="panda_moveit_task"

# Reuse current ROS environment; otherwise auto-detect /opt/ros/<distro>.
if ! command -v ros2 >/dev/null 2>&1; then
    ROS_SETUP=""
    for candidate in /opt/ros/*/setup.bash; do
        if [ -f "$candidate" ]; then
            ROS_SETUP="$candidate"
            break
        fi
    done
    if [ -z "$ROS_SETUP" ]; then
        echo "ERROR: ROS2 environment not found."
        exit 1
    fi
    source "$ROS_SETUP"
fi

if [ ! -f "$WORKSPACE/install/setup.bash" ]; then
    echo "ERROR: $WORKSPACE/install/setup.bash not found. Build workspace first."
    exit 1
fi
source "$WORKSPACE/install/setup.bash"

# One fixed orientation reference for the complete task sequence.
# IMPORTANT: the bridge creates/overwrites this file at BRIDGE STARTUP.
# run_tasks_all.sh must NEVER delete it and must NEVER recapture Q0.
export PANDA_TASK_ORIENTATION_REFERENCE="/tmp/panda_task_initial_orientation.json"

echo "Using BRIDGE-START GLOBAL Q0: $PANDA_TASK_ORIENTATION_REFERENCE"

if [ ! -f "$PANDA_TASK_ORIENTATION_REFERENCE" ]; then
    echo "ERROR: Global Q0 reference does not exist."
    echo "Start/restart the MuJoCo bridge first, then run this script."
    exit 1
fi

# Validate that the bridge-created reference has the expected orientation-only schema.
python3 - "$PANDA_TASK_ORIENTATION_REFERENCE" <<'PY_Q0_CHECK'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

q = [float(data[k]) for k in ("qx", "qy", "qz", "qw")]
n = math.sqrt(sum(v*v for v in q))
if n <= 1e-12:
    raise SystemExit("ERROR: bridge Q0 quaternion is zero/invalid")

q = [v/n for v in q]
print("BRIDGE-START GLOBAL Q0 =", q)
print("Reference file =", path)
PY_Q0_CHECK

if [ $? -ne 0 ]; then
    echo "ERROR: Invalid bridge-start GLOBAL Q0 reference."
    echo "Restart the MuJoCo bridge to regenerate it."
    exit 1
fi

TASKS=(
    "pick_place|29_pick_place.yaml"
    "return|30_return.yaml"
)

echo "============================================================"
echo " AUTOMATIC TASK SEQUENCE"
echo "============================================================"

for ROW in "${TASKS[@]}"; do
    EXECUTABLE="${ROW%%|*}"
    YAML_NAME="${ROW#*|}"
    YAML="$CONFIG_DIR/$YAML_NAME"
    INSTANCE="${YAML_NAME%.*}"

    echo
    echo "============================================================"
    echo " START $INSTANCE"
    echo " executable: $EXECUTABLE"
    echo " yaml:       $YAML_NAME"
    echo "============================================================"

    if [ ! -f "$YAML" ]; then
        echo "ERROR: YAML not found: $YAML"
        exit 1
    fi

    if ! ros2 pkg executables "$PACKAGE" | grep -q "^$PACKAGE $EXECUTABLE$"; then
        echo "ERROR: executable not found: $PACKAGE $EXECUTABLE"
        echo "Rebuild the package and source install/setup.bash."
        exit 1
    fi

    ros2 run "$PACKAGE" "$EXECUTABLE" --ros-args --params-file "$YAML"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "ERROR: $INSTANCE failed with exit code $STATUS"
        echo "Stopping remaining tasks."
        exit $STATUS
    fi

    echo "FINISHED $INSTANCE"
    sleep 1
done

echo
echo "============================================================"
echo " ALL TASKS FINISHED"
echo "============================================================"

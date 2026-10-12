#!/usr/bin/env python3
from pathlib import Path
import re

PACKAGE = "panda_moveit_task"
PKG_ROOT = Path(__file__).resolve().parent
PY_DIR = PKG_ROOT / PACKAGE
CONFIG_DIR = PKG_ROOT / "config"
SETUP_FILE = PKG_ROOT / "setup.py"
RUN_FILE = PKG_ROOT / "run_tasks_all.sh"

BEGIN_MARKER = "# BEGIN AUTO TASKS"
END_MARKER = "# END AUTO TASKS"

def natural_key(v):
    """Natural sort: xxx_2 before xxx_10."""
    text = v.name if isinstance(v, Path) else str(v)
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", text)
    ]


def has_main(path):
    """Check whether a Python module defines main()."""
    text = path.read_text(encoding="utf-8")
    return bool(
        re.search(
            r"^\s*def\s+main\s*\(",
            text,
            re.MULTILINE,
        )
    )

# Reusable Python task modules: pick_place.py, handover.py, ...
modules = []
for p in sorted(PY_DIR.glob("*.py"), key=natural_key):
    if p.name == "__init__.py" or p.name.startswith("_"):
        continue
    if has_main(p):
        modules.append(p)
    else:
        print(f"SKIP: {p.name} has no main()")

if not modules:
    raise RuntimeError(f"No task modules with main() found in {PY_DIR}")

# One executable per Python module; many YAML instances can use it.
#
# Preferred naming:
#   0_pick_place.yaml
#   4_handover.yaml
#   11_return.yaml
#
# Legacy naming is also accepted:
#   pick_place_0.yaml
#   handover_4.yaml
#   return_11.yaml
tasks = []
for p in modules:
    exe = p.stem

    candidates = (
        list(CONFIG_DIR.glob(f"*_{exe}.yaml"))
        + list(CONFIG_DIR.glob(f"*_{exe}.yml"))
        + list(CONFIG_DIR.glob(f"{exe}_*.yaml"))
        + list(CONFIG_DIR.glob(f"{exe}_*.yml"))
    )

    yamls = []
    for y in sorted(set(candidates), key=natural_key):
        preferred = re.fullmatch(
            rf"(\d+)_{re.escape(exe)}\.(?:yaml|yml)",
            y.name,
            re.IGNORECASE,
        )
        legacy = re.fullmatch(
            rf"{re.escape(exe)}_(\d+)\.(?:yaml|yml)",
            y.name,
            re.IGNORECASE,
        )
        if preferred or legacy:
            yamls.append(y)

    if not yamls:
        print(
            f"WARNING: {p.name}: no <number>_{exe}.yaml "
            f"(or legacy {exe}_<number>.yaml)"
        )

    for y in yamls:
        tasks.append((exe, y))

if not tasks:
    raise RuntimeError("No matching task YAML files found")


def get_task_sequence(yaml_path, exe):
    """Extract task number from either supported filename convention."""
    preferred = re.fullmatch(
        rf"(\d+)_{re.escape(exe)}\.(?:yaml|yml)",
        yaml_path.name,
        re.IGNORECASE,
    )
    if preferred:
        return int(preferred.group(1))

    legacy = re.fullmatch(
        rf"{re.escape(exe)}_(\d+)\.(?:yaml|yml)",
        yaml_path.name,
        re.IGNORECASE,
    )
    if legacy:
        return int(legacy.group(1))

    raise RuntimeError(
        f"Invalid task YAML name for {exe}: {yaml_path.name}. "
        f"Preferred form: <number>_{exe}.yaml"
    )


def task_sequence_key(task):
    """Global execution order is determined by the task number."""
    exe, yaml_path = task
    return (
        get_task_sequence(yaml_path, exe),
        natural_key(yaml_path),
        exe.lower(),
    )


# IMPORTANT: sort ALL task instances together by their numeric task sequence.
# Do not group execution by Python module.
tasks.sort(key=task_sequence_key)

# Reject duplicate sequence numbers because execution order would otherwise
# be ambiguous (for example pick_place_3.yaml and pick_insert_3.yaml).
seen_sequences = {}
for task in tasks:
    seq = task_sequence_key(task)[0]
    if seq in seen_sequences:
        raise RuntimeError(
            f"Duplicate task sequence number {seq}: "
            f"{seen_sequences[seq].name} and {task[1].name}"
        )
    seen_sequences[seq] = task[1]

print("Modules:")
for p in modules:
    print(f"  {p.name}")
print("Task instances:")
for exe, y in tasks:
    print(f"  {exe} <- {y.name}")

# setup.py: register only module names, not YAML instance names.
entries = "\n".join(
    f"            '{p.stem} = {PACKAGE}.{p.stem}:main'," for p in modules
)
block = f"{BEGIN_MARKER}\n{entries}\n            {END_MARKER}"
setup = SETUP_FILE.read_text(encoding="utf-8")

if BEGIN_MARKER in setup and END_MARKER in setup:
    pattern = re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER)
elif "# BEGIN AUTO PICK_PLACE" in setup and "# END AUTO PICK_PLACE" in setup:
    pattern = re.escape("# BEGIN AUTO PICK_PLACE") + r".*?" + re.escape("# END AUTO PICK_PLACE")
    print("Migrating old AUTO PICK_PLACE markers to AUTO TASKS.")
else:
    raise RuntimeError(
        "Add '# BEGIN AUTO TASKS' and '# END AUTO TASKS' inside setup.py console_scripts."
    )

setup = re.sub(pattern, block, setup, count=1, flags=re.DOTALL)
SETUP_FILE.write_text(setup, encoding="utf-8")

# Preserve the global execution order already sorted by each YAML's task number.
rows = "\n".join(f'    "{exe}|{y.name}"' for exe, y in tasks)

runner = f"""#!/usr/bin/env bash

# Resolve ROS 2 workspace from this script's location instead of a user-specific path.
SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
WORKSPACE="${{MOVEIT_WS:-}}"
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
CONFIG_DIR="$WORKSPACE/src/{PACKAGE}/config"
PACKAGE="{PACKAGE}"

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
{rows}
)

echo "============================================================"
echo " AUTOMATIC TASK SEQUENCE"
echo "============================================================"

for ROW in "${{TASKS[@]}}"; do
    EXECUTABLE="${{ROW%%|*}}"
    YAML_NAME="${{ROW#*|}}"
    YAML="$CONFIG_DIR/$YAML_NAME"
    INSTANCE="${{YAML_NAME%.*}}"

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
"""

RUN_FILE.write_text(runner, encoding="utf-8")
RUN_FILE.chmod(0o755)

print()
print("setup.py executables:")
for p in modules:
    print(f"  {p.stem}")
print()
print("Generated runner:", RUN_FILE)
print("Run with: ./run_tasks_all.sh")

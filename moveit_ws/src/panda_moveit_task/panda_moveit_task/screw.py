#!/usr/bin/env python3
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PointStamped, PoseStamped
from control_msgs.action import GripperCommand
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from panda_moveit_planner.srv import PlanToPose


def resolve_scene_xml(scene_xml: str) -> Path:
    """Find a MuJoCo scene independently of the shell working directory.

    Supports absolute paths, PANDA_PROJECT_ROOT, and the repository layout
    <project>/moveit_ws/src/panda_moveit_task and <project>/mujoco_pj.
    """
    requested = Path(scene_xml).expanduser()
    if requested.is_absolute():
        if requested.is_file():
            return requested.resolve()
        raise FileNotFoundError(f"Scene XML does not exist: {requested}")

    candidates = []
    project_root = os.environ.get("PANDA_PROJECT_ROOT")
    if project_root:
        candidates.append(Path(project_root).expanduser() / requested)

    # Also works with typical colcon install layouts, including symlink-install.
    for anchor in (Path(__file__).resolve(), Path(__file__).absolute()):
        for parent in anchor.parents:
            candidates.append(parent / requested)
            # A workspace might be invoked directly as the root.
            candidates.append(parent / "mujoco_pj" / requested.name)

    # Last fallback for users invoking the command from the project directory.
    candidates.append(Path.cwd() / requested)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot locate scene_xml={scene_xml!r}. "
        "Keep mujoco_pj beside moveit_ws, or set PANDA_PROJECT_ROOT "
        "to the directory containing mujoco_pj."
    )


class ScrewTask(Node):
    """
    Pick the screwdriver from its pose in scene_xml, keep it grasped, execute
    N screw operations, return the screwdriver to its original pose, release it,
    then return the robot to the pose at the beginning of this task.

    Per-screw YAML parameters:
      screw_i_tip_x/y/z
      screw_i_rx_deg
      screw_i_approach_distance
      screw_i_approach_direction
      screw_i_shaft_speed
      screw_i_down_distance

    approach_direction is the insertion/screwing direction.
    approach pose = tip - direction * approach_distance
    screw end     = tip + direction * down_distance
    """

    def __init__(self):
        super().__init__("screw_task")

        self.declare_parameter("scene_xml", "wrs.xml")
        # Select MoveIt service namespace: panda1 or panda2.
        self.declare_parameter("robot", "panda1")
        self.declare_parameter("screw_count", 1)

        # Screwdriver pickup. XYZ comes from scene_xml, not YAML.
        self.declare_parameter("pickup_approach_distance", 0.06)
        self.declare_parameter("pickup_grasp_width", 0.025)
        self.declare_parameter("pickup_lift_distance", 0.06)
        # Separate vertical approach/retreat distance when placing screwdriver back.
        self.declare_parameter("return_place_distance", 0.06)

        # Numbered screw parameters. ROS2 parameters must be declared before use.
        # 32 is only the declaration ceiling; screw_count decides how many run.
        for i in range(1, 33):
            p = f"screw_{i}"
            self.declare_parameter(f"{p}_tip_x", 0.0)
            self.declare_parameter(f"{p}_tip_y", 0.0)
            self.declare_parameter(f"{p}_tip_z", 0.85)
            self.declare_parameter(f"{p}_rx_deg", 90.0)
            self.declare_parameter(f"{p}_approach_distance", 0.05)
            self.declare_parameter(f"{p}_approach_direction", "-z")
            self.declare_parameter(f"{p}_shaft_speed", 4.0)
            self.declare_parameter(f"{p}_down_distance", 0.005)

        self.declare_parameter("velocity_scale", 0.25)
        self.declare_parameter("acceleration_scale", 0.20)
        self.declare_parameter("initial_settle_time", 0.5)

        self.center = None
        self.tcp_pose = None
        self.initial_tcp_pose = None

        self.orientation_reference_file = os.environ.get(
            "PANDA_TASK_ORIENTATION_REFERENCE",
            "/tmp/panda_task_initial_orientation.json",
        )
        self.sequence_base_orientation = None

        self.screwdriver_initial_pos = None
        self.screwdriver_initial_q = None
        self.screwdriver_tip_local = None

        # ============================================================
        # MODIFY SCREWDRIVER GRASP XYZ HERE
        # WORLD coordinates of the gripper TCP at the actual grasp point.
        # These values do NOT come from YAML.
        # ============================================================
        self.screwdriver_grasp_pos = [
            0.000,   # X [m]
            0.250,   # Y [m]
            0.802,   # Z [m]
        ]

        robot = str(self.param("robot")).strip().lower().strip("/")
        if robot not in ("panda1", "panda2"):
            raise RuntimeError(
                f"Invalid robot={robot!r}; expected 'panda1' or 'panda2'"
            )
        self.robot = robot
        self.robot_ns = f"/{robot}"

        # Robot-specific state/control endpoints.
        self.center_topic = (
            "/mujoco/panda1_fingertip_center"
            if robot == "panda1"
            else "/mujoco/fingertip_center"
        )
        self.gripper_action = (
            "/mujoco/panda1_gripper_command"
            if robot == "panda1"
            else "/mujoco/gripper_command"
        )
        self.tcp_pose_topic = (
            "/mujoco/panda1_tcp_pose"
            if robot == "panda1"
            else "/mujoco/panda2_tcp_pose"
        )

        self.create_subscription(
            PointStamped, self.center_topic, self.center_cb, 20
        )
        self.create_subscription(
            PoseStamped, self.tcp_pose_topic, self.tcp_pose_cb, 20
        )

        self.plan_service = f"{self.robot_ns}/plan_to_pose"
        self.execute_service = f"{self.robot_ns}/execute_next_plan"
        self.clear_queue_service = f"{self.robot_ns}/clear_plan_queue"

        self.get_logger().info(
            f"Selected robot: {self.robot}; MoveIt namespace: {self.robot_ns}"
        )

        self.plan_client = self.create_client(PlanToPose, self.plan_service)
        self.execute_next_client = self.create_client(
            Trigger, self.execute_service
        )
        self.clear_queue_client = self.create_client(
            Trigger, self.clear_queue_service
        )
        self.stop_client = self.create_client(
            Trigger, "/mujoco/wait_until_stopped"
        )
        self.screw_start_client = self.create_client(
            Trigger, "/mujoco/screw_start"
        )
        self.screw_stop_client = self.create_client(
            Trigger, "/mujoco/screw_stop"
        )
        self.screw_config_pub = self.create_publisher(
            Float64MultiArray, "/mujoco/screw_config", 10
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, self.gripper_action
        )


    def param(self, name):
        return self.get_parameter(name).value

    @staticmethod
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ]

    @staticmethod
    def quat_conjugate(q):
        return [-q[0], -q[1], -q[2], q[3]]

    @classmethod
    def rotate_vector(cls, q, v):
        vq = [float(v[0]), float(v[1]), float(v[2]), 0.0]
        return cls.quat_multiply(
            cls.quat_multiply(q, vq), cls.quat_conjugate(q)
        )[:3]

    @staticmethod
    def normalize_quat(q):
        n = math.sqrt(sum(float(v) * float(v) for v in q))
        if n <= 1e-12:
            raise RuntimeError("zero quaternion")
        return [float(v) / n for v in q]

    @classmethod
    def euler_xyz_deg_to_quat(cls, rx_deg, ry_deg, rz_deg):
        rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
        qx = [math.sin(rx/2), 0.0, 0.0, math.cos(rx/2)]
        qy = [0.0, math.sin(ry/2), 0.0, math.cos(ry/2)]
        qz = [0.0, 0.0, math.sin(rz/2), math.cos(rz/2)]
        return cls.normalize_quat(
            cls.quat_multiply(qz, cls.quat_multiply(qy, qx))
        )

    def rotated_x_orientation(self, base_q, rx_deg):
        rx = math.radians(float(rx_deg))
        qx = [math.sin(rx/2.0), 0.0, 0.0, math.cos(rx/2.0)]
        return self.normalize_quat(self.quat_multiply(qx, base_q))

    @staticmethod
    def direction_vector(raw):
        key = str(raw).strip().lower().replace(" ", "")
        table = {
            "+x": [1.0, 0.0, 0.0], "x+": [1.0, 0.0, 0.0],
            "-x": [-1.0, 0.0, 0.0], "x-": [-1.0, 0.0, 0.0],
            "+y": [0.0, 1.0, 0.0], "y+": [0.0, 1.0, 0.0],
            "-y": [0.0, -1.0, 0.0], "y-": [0.0, -1.0, 0.0],
            "+z": [0.0, 0.0, 1.0], "z+": [0.0, 0.0, 1.0],
            "-z": [0.0, 0.0, -1.0], "z-": [0.0, 0.0, -1.0],
        }
        if key not in table:
            raise RuntimeError(
                f"Invalid approach_direction={raw!r}; use +/-x, +/-y, +/-z"
            )
        return table[key]

    def center_cb(self, msg):
        self.center = [msg.point.x, msg.point.y, msg.point.z]

    def tcp_pose_cb(self, msg):
        q = self.normalize_quat([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        self.tcp_pose = (
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            q,
        )

    def wait_services(self):
        self.get_logger().info(
            f"Waiting for services for {self.robot}: "
            f"{self.plan_service}, {self.execute_service}, "
            f"{self.clear_queue_service}"
        )

        required_services = (
            (self.plan_client, self.plan_service),
            (self.execute_next_client, self.execute_service),
            (self.clear_queue_client, self.clear_queue_service),
            (self.stop_client, "/mujoco/wait_until_stopped"),
            (self.screw_start_client, "/mujoco/screw_start"),
            (self.screw_stop_client, "/mujoco/screw_stop"),
        )

        for client, name in required_services:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    f"Required service unavailable: {name}. "
                    f"Check bridge namespace for selected robot {self.robot!r}."
                )

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f"Required action unavailable: {self.gripper_action}"
            )

        deadline = time.monotonic() + 5.0
        while rclpy.ok() and (self.center is None or self.tcp_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > deadline:
                missing = []
                if self.center is None:
                    missing.append(self.center_topic)
                if self.tcp_pose is None:
                    missing.append(self.tcp_pose_topic)
                raise RuntimeError(
                    "No robot state received from: " + ", ".join(missing)
                )

    def wait_stopped(self):
        future = self.stop_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(
                "Robot stop service failed: "
                + ("no response" if result is None else result.message)
            )

    def clear_plan_queue(self):
        future = self.clear_queue_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(
                "Clear plan queue failed: "
                + ("no response" if result is None else result.message)
            )

    def gripper_width(self, width, require_grasp=False):
        goal = GripperCommand.Goal()
        goal.command.position = float(width)
        # Custom bridge convention already used by pick_place.py:
        # 0 = open/release only; 1 = grasp/attach allowed.
        goal.command.max_effort = 1.0 if require_grasp else 0.0

        f = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f)
        gh = f.result()
        if gh is None or not gh.accepted:
            raise RuntimeError("Gripper goal rejected")

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        wrapped = rf.result()
        if wrapped is None:
            raise RuntimeError("Gripper returned no result")
        if require_grasp and not wrapped.result.reached_goal:
            raise RuntimeError("Failed to grasp screwdriver")

    def load_or_create_sequence_reference(self, current_q):
        """
        Load Q0 captured by run_tasks_all.sh BEFORE any task moves the robot.
        Never create Q0 from this task's current pose.
        """
        path = self.orientation_reference_file
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Global orientation reference not found: {path}. "
                "Run through run_tasks_all.sh so Q0 is captured before task 0."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        q = self.normalize_quat([
            data["qx"], data["qy"], data["qz"], data["qw"]
        ])
        self.get_logger().info(
            f"Loaded whole-sequence GLOBAL Q0 from {path}: {q}"
        )
        return q

    def save_initial_pose(self):
        if self.tcp_pose is None:
            raise RuntimeError(
                f"No TCP pose received from {self.tcp_pose_topic}"
            )
        pos, q = self.tcp_pose
        self.initial_center = list(pos)
        self.initial_tcp_pose = (list(pos), list(q))
        self.sequence_base_orientation = (
            self.load_or_create_sequence_reference(q)
        )
        self.get_logger().info(
            f"Initial {self.robot} TCP: pos={pos}, q={q}"
        )

    def queue_fingertip_pose(self, fingertip_pos, orientation, linear=False):
        """Queue an absolute WORLD TCP pose; bridge solves the selected robot IK."""
        req = PlanToPose.Request()
        req.target.header.frame_id = "world"
        req.target.header.stamp = self.get_clock().now().to_msg()
        req.target.pose.position.x = float(fingertip_pos[0])
        req.target.pose.position.y = float(fingertip_pos[1])
        req.target.pose.position.z = float(fingertip_pos[2])
        req.target.pose.orientation.x = float(orientation[0])
        req.target.pose.orientation.y = float(orientation[1])
        req.target.pose.orientation.z = float(orientation[2])
        req.target.pose.orientation.w = float(orientation[3])
        req.linear = bool(linear)
        req.execute = False
        req.velocity_scale = float(self.param("velocity_scale"))
        req.acceleration_scale = float(self.param("acceleration_scale"))

        f = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        result = f.result()
        if result is None or not result.success:
            raise RuntimeError(
                "MoveIt planning failed: "
                + ("no response" if result is None else result.message)
            )

    def execute_next_plan(self, name):
        self.get_logger().info(f"EXECUTE: {name}")
        f = self.execute_next_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, f)
        result = f.result()
        if result is None or not result.success:
            raise RuntimeError(
                f"{name} failed: "
                + ("no response" if result is None else result.message)
            )
        self.wait_stopped()

    def plan_and_execute(self, fingertip_pos, orientation, name, linear=False):
        self.queue_fingertip_pose(fingertip_pos, orientation, linear=linear)
        self.execute_next_plan(name)

    def parse_screwdriver_from_xml(self):
        path = resolve_scene_xml(str(self.param("scene_xml")))

        root = ET.parse(path).getroot()
        body = root.find(".//body[@name='screwdriver']")
        if body is None:
            raise RuntimeError("scene_xml has no body named 'screwdriver'")

        pos = [float(v) for v in body.get("pos", "0 0 0").split()]
        if len(pos) != 3:
            raise RuntimeError("screwdriver pos must have 3 values")

        if body.get("quat") is not None:
            # MuJoCo XML quat is w x y z; internal ROS quaternion is x y z w.
            w, x, y, z = [float(v) for v in body.get("quat").split()]
            q = self.normalize_quat([x, y, z, w])
        else:
            e = [float(v) for v in body.get("euler", "0 0 0").split()]
            q = self.euler_xyz_deg_to_quat(e[0], e[1], e[2])

        tip = body.find(".//site[@name='screwdriver_tip']")
        if tip is None:
            raise RuntimeError("scene_xml has no site named 'screwdriver_tip'")

        # Accumulate body/site translations from screwdriver body to tip.
        # Current WRS screwdriver hierarchy has no intermediate rotations.
        tip_local = [0.0, 0.0, 0.0]
        shaft = body.find(".//body[@name='screwdriver_shaft']")
        if shaft is not None:
            sp = [float(v) for v in shaft.get("pos", "0 0 0").split()]
            tip_local = [tip_local[j] + sp[j] for j in range(3)]
        tp = [float(v) for v in tip.get("pos", "0 0 0").split()]
        tip_local = [tip_local[j] + tp[j] for j in range(3)]

        self.screwdriver_initial_pos = pos
        self.screwdriver_initial_q = q
        self.screwdriver_tip_local = tip_local

        self.get_logger().info(
            f"Screwdriver from XML: origin={pos}, q={q}, "
            f"tip_local={tip_local}"
        )

    def screw_params(self, i):
        p = f"screw_{i}"
        return {
            "tip": [
                float(self.param(f"{p}_tip_x")),
                float(self.param(f"{p}_tip_y")),
                float(self.param(f"{p}_tip_z")),
            ],
            "rx": float(self.param(f"{p}_rx_deg")),
            "approach_distance": abs(
                float(self.param(f"{p}_approach_distance"))
            ),
            "direction_name": str(
                self.param(f"{p}_approach_direction")
            ),
            "direction": self.direction_vector(
                self.param(f"{p}_approach_direction")
            ),
            "shaft_speed": float(self.param(f"{p}_shaft_speed")),
            "down_distance": abs(
                float(self.param(f"{p}_down_distance"))
            ),
        }

    def screwdriver_origin_for_tip(self, tip_world, rx_deg):
        """
        At pickup the gripper TCP is placed at the screwdriver body origin and
        hand orientation is Q0. Therefore the grasped screwdriver target
        orientation after a WORLD-X rotation is simply Rx * Q_screwdriver_initial.
        """
        rx = math.radians(float(rx_deg))
        qx = [math.sin(rx/2), 0.0, 0.0, math.cos(rx/2)]
        screwdriver_q = self.normalize_quat(
            self.quat_multiply(qx, self.screwdriver_initial_q)
        )
        tip_offset_world = self.rotate_vector(
            screwdriver_q, self.screwdriver_tip_local
        )
        return [
            tip_world[0] - tip_offset_world[0],
            tip_world[1] - tip_offset_world[1],
            tip_world[2] - tip_offset_world[2],
        ]

    def send_screw_config(
        self, direction, shaft_speed, down_distance, target_tip
    ):
        """
        Configure one screw operation.

        target_tip is the YAML screw_i_tip_x/y/z WORLD position.  The bridge
        uses THIS point only to choose which freejoint screw object belongs to
        the operation.  It no longer uses the actual screwdriver_tip position
        for object selection.
        """
        msg = Float64MultiArray()
        msg.data = [
            float(direction[0]), float(direction[1]), float(direction[2]),
            float(shaft_speed), float(down_distance),
            float(target_tip[0]), float(target_tip[1]), float(target_tip[2]),
        ]
        # Publish a few times so configuration is present before Trigger.
        for _ in range(3):
            self.screw_config_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.03)

    def trigger(self, client, name):
        f = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, f)
        result = f.result()
        if result is None or not result.success:
            raise RuntimeError(
                f"{name} failed: "
                + ("no response" if result is None else result.message)
            )
        self.get_logger().info(result.message)

    def run(self):
        self.wait_services()
        self.wait_stopped()
        time.sleep(max(0.0, float(self.param("initial_settle_time"))))
        for _ in range(8):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.save_initial_pose()
        self.parse_screwdriver_from_xml()

        count = int(self.param("screw_count"))
        if count < 1 or count > 32:
            raise RuntimeError("screw_count must be in [1, 32]")

        q0 = self.sequence_base_orientation
        self.get_logger().info(f"Pickup orientation = GLOBAL Q0: {q0}")

        # Actual gripper TCP grasp point is manually specified in this script.
        pickup = [float(v) for v in self.screwdriver_grasp_pos]
        pickup_d = abs(float(self.param("pickup_approach_distance")))
        pickup_approach = [pickup[0], pickup[1], pickup[2] + pickup_d]

        # Preserve the relationship between the manually selected gripper TCP
        # grasp point and the screwdriver XML body origin.  This lets us return
        # the screwdriver to its original XML pose using the same grasp point.
        screwdriver_home = self.screwdriver_initial_pos
        grasp_to_body_offset = [
            screwdriver_home[j] - pickup[j] for j in range(3)
        ]

        self.get_logger().info(
            f"Manual screwdriver grasp TCP XYZ={pickup}; "
            f"XML screwdriver origin={screwdriver_home}"
        )

        self.get_logger().info("========== PICK SCREWDRIVER ==========")
        self.gripper_width(0.08, require_grasp=False)
        self.plan_and_execute(
            pickup_approach, q0, "Move above screwdriver", linear=False
        )
        self.plan_and_execute(
            pickup, q0, "Approach screwdriver", linear=True
        )
        self.gripper_width(
            float(self.param("pickup_grasp_width")), require_grasp=True
        )

        lift = [
            pickup[0], pickup[1],
            pickup[2] + abs(float(self.param("pickup_lift_distance")))
        ]
        self.plan_and_execute(lift, q0, "Lift screwdriver", linear=True)

        for i in range(1, count + 1):
            cfg = self.screw_params(i)
            d = cfg["direction"]
            tip = cfg["tip"]
            approach_tip = [
                tip[j] - d[j] * cfg["approach_distance"]
                for j in range(3)
            ]
            down_tip = [
                tip[j] + d[j] * cfg["down_distance"]
                for j in range(3)
            ]

            hand_q = self.rotated_x_orientation(q0, cfg["rx"])
            approach_origin = self.screwdriver_origin_for_tip(
                approach_tip, cfg["rx"]
            )
            contact_origin = self.screwdriver_origin_for_tip(
                tip, cfg["rx"]
            )
            down_origin = self.screwdriver_origin_for_tip(
                down_tip, cfg["rx"]
            )

            self.get_logger().info(
                f"========== SCREW {i}/{count} =========="
            )
            self.get_logger().info(
                f"tip={tip}, rx={cfg['rx']:.3f} deg, "
                f"direction={cfg['direction_name']}, "
                f"approach={cfg['approach_distance']:.4f}, "
                f"shaft_speed={cfg['shaft_speed']:.3f}, "
                f"down={cfg['down_distance']:.4f}"
            )

            # Between screws we remain grasping the screwdriver.
            self.plan_and_execute(
                approach_origin, hand_q,
                f"Screw {i}: move to approach", linear=False
            )
            self.plan_and_execute(
                contact_origin, hand_q,
                f"Screw {i}: approach tip", linear=True
            )

            # Bridge selects the freejoint object nearest the YAML target
            # tip WORLD position. Actual screwdriver_tip is NOT used for selection.
            # Send the YAML contact-tip WORLD XYZ as the authoritative point
            # for selecting the screw object at /mujoco/screw_start.
            self.send_screw_config(
                d,
                cfg["shaft_speed"],
                cfg["down_distance"],
                tip,
            )
            self.trigger(self.screw_start_client, f"Screw {i} start")

            # The selected screw follows actual screwdriver-tip progress.
            self.plan_and_execute(
                down_origin, hand_q,
                f"Screw {i}: screw down", linear=True
            )
            self.trigger(self.screw_stop_client, f"Screw {i} stop")

            # Every screw retreats to its own approach pose and keeps grasping.
            self.plan_and_execute(
                approach_origin, hand_q,
                f"Screw {i}: retreat to approach", linear=True
            )

        # Last screw only: return using the SAME manually selected grasp TCP
        # point, release, then restore the robot pose captured at task start.
        # The rigid grasp transform was captured at pickup, so returning the TCP
        # here with q0 restores the screwdriver to the pose it had when grasped.
        self.get_logger().info("========== RETURN SCREWDRIVER ==========")
        return_d = float(self.param("return_place_distance"))
        if not math.isfinite(return_d) or return_d < 0.0:
            raise RuntimeError("return_place_distance must be finite and >= 0")
        return_approach = [
            pickup[0], pickup[1], pickup[2] + return_d
        ]
        self.get_logger().info(
            f"Screwdriver return placement distance = {return_d:.4f} m"
        )
        self.plan_and_execute(
            return_approach, q0, "Return above screwdriver home", linear=False
        )
        if return_d > 1e-9:
            self.plan_and_execute(
                pickup, q0, "Return screwdriver to manual grasp home", linear=True
            )
        self.gripper_width(0.08, require_grasp=False)

        # After releasing the screwdriver, return to the SAME pickup approach
        # pose used at the very beginning of this task.
        if return_d > 1e-9:
            self.plan_and_execute(
                return_approach,
                q0,
                "Robot retreat after placing screwdriver",
                linear=True,
            )

        self.get_logger().info("========== SCREW TASK FINISHED ==========")


def main(args=None):
    rclpy.init(args=args)
    node = ScrewTask()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(str(e))
        try:
            node.clear_plan_queue()
        except Exception:
            pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

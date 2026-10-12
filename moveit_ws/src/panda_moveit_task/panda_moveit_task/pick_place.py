import time
import math
import json
import os
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PointStamped, PoseStamped
from control_msgs.action import GripperCommand
from std_srvs.srv import Trigger
from panda_moveit_planner.srv import PlanToPose


class PickActionTask(Node):
    def __init__(self):
        super().__init__("pick_action_task")

        self.declare_parameter("arm", "panda2")  # panda1 or panda2

        # Source: part_0_3_1 body origin = (0.22, 0.15, 0.797)
        self.declare_parameter("grasp_x", 0.22)
        self.declare_parameter("grasp_y", 0.15)
        self.declare_parameter("grasp_z", 0.847)
        self.declare_parameter("grasp_rx_deg", 0.0)
        self.declare_parameter("grasp_ry_deg", 0.0)

        # Pre-grasp approach: start above grasp and move straight down.
        self.declare_parameter("approach_distance", 0.05)

        self.declare_parameter("lift_height", 0.05)

        # Destination: wrs_geom_0_0_1
        self.declare_parameter("place_x", 0.0)
        self.declare_parameter("place_y", 0.0)
        self.declare_parameter("place_z", 0.85)
        self.declare_parameter("place_rx_deg", 0.0)
        self.declare_parameter("place_ry_deg", 0.0)

        # Final place/insert direction: +y, -y, +z, -z.
        self.declare_parameter("place_direction", "-z")
        self.declare_parameter("place_distance", 0.05)
        self.declare_parameter("retreat_distance", 0.05)

        # true: release after placement and retreat; false: stop at final place pose.
        self.declare_parameter("open_and_lift_after_place", True)

        # gripper_open_width: width AFTER placing/releasing the object.
        # Before every grasp, the gripper is always reset to 0.08 m.
        self.declare_parameter("gripper_open_width", 0.08)
        self.declare_parameter("grasp_width", 0.06)

        self.declare_parameter("velocity_scale", 0.35)
        self.declare_parameter("acceleration_scale", 0.25)
        self.declare_parameter("initial_settle_time", 0.5)
        self.declare_parameter("debug_mode", False)

        self.center = None
        self.tcp_pose = None
        self.initial_center = None
        self.initial_tcp_pose = None
        # Fixed orientation reference captured by the bridge at bridge startup.
        # Tasks and run_tasks_all.sh must only read this file, never recreate Q0.
        self.orientation_reference_file = os.environ.get(
            "PANDA_TASK_ORIENTATION_REFERENCE",
            "/tmp/panda_task_initial_orientation.json",
        )
        self.sequence_base_orientation = None

        arm = str(self.get_parameter("arm").value).strip().lower()
        if arm not in ("panda1", "panda2"):
            raise ValueError(f"Invalid arm={arm!r}; expected panda1 or panda2")
        self.arm = arm
        center_topic = ("/mujoco/panda1_fingertip_center" if arm == "panda1"
                        else "/mujoco/fingertip_center")
        tcp_topic = f"/mujoco/{arm}_tcp_pose"
        self.create_subscription(
            PointStamped, center_topic, self.center_cb, 20)
        self.create_subscription(
            PoseStamped, tcp_topic, self.tcp_pose_cb, 20)

        # IMPORTANT: use the bridge Panda2 TRUE-TCP planner.
        # Its target orientation is the absolute WORLD panda2_hand/TCP rotation,
        # exactly the same frame used when bridge-start Q0 is captured.
        self.plan_client = self.create_client(PlanToPose, f"/{arm}/plan_to_pose")
        self.execute_next_client = self.create_client(
            Trigger, f"/{arm}/execute_next_plan")
        self.clear_queue_client = self.create_client(
            Trigger, f"/{arm}/clear_plan_queue")
        self.gripper_client = ActionClient(
            self, GripperCommand, ("/mujoco/panda1_gripper_command"
                                   if arm == "panda1" else "/mujoco/gripper_command"))
        self.stop_client = self.create_client(
            Trigger, "/mujoco/wait_until_stopped")

    def param(self, name):
        return self.get_parameter(name).value

    def get_grasp_pos(self):
        return [float(self.param("grasp_x")), float(self.param("grasp_y")), float(self.param("grasp_z"))]

    def get_horizontal_delta(self):
        return [float(self.param("horizontal_dx")), float(self.param("horizontal_dy")), float(self.param("horizontal_dz"))]

    def get_velocity_scale(self):
        return float(self.param("velocity_scale"))

    def get_acceleration_scale(self):
        return float(self.param("acceleration_scale"))

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

    def rotated_orientation(self, base_q, rx_deg, ry_deg):
        # Apply user rotations about WORLD X, then WORLD Y.
        rx = math.radians(float(rx_deg))
        ry = math.radians(float(ry_deg))
        qx = [math.sin(rx/2.0), 0.0, 0.0, math.cos(rx/2.0)]
        qy = [0.0, math.sin(ry/2.0), 0.0, math.cos(ry/2.0)]
        q_delta = self.quat_multiply(qy, qx)
        q = self.quat_multiply(q_delta, base_q)
        n = math.sqrt(sum(v*v for v in q))
        return [v/n for v in q]

    def get_place_direction(self):
        raw = str(self.param("place_direction")).strip().lower().replace(" ", "")
        directions = {
            "+y": (1, 1.0), "y+": (1, 1.0),
            "-y": (1, -1.0), "y-": (1, -1.0),
            "+z": (2, 1.0), "z+": (2, 1.0),
            "-z": (2, -1.0), "z-": (2, -1.0),
        }
        if raw not in directions:
            raise RuntimeError(
                f"Invalid place_direction={raw!r}; use '+y', '-y', '+z', or '-z'"
            )
        axis, sign = directions[raw]
        return raw, axis, sign

    def center_cb(self, msg):
        self.center = [msg.point.x, msg.point.y, msg.point.z]

    def tcp_pose_cb(self, msg):
        q = [
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ]
        n = math.sqrt(sum(v*v for v in q))
        if n <= 1e-12:
            return
        q = [v/n for v in q]
        self.tcp_pose = (
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            q,
        )

    def wait_services(self):
        self.get_logger().info(f"Waiting for {self.arm} TRUE-TCP/MuJoCo services...")
        while rclpy.ok():
            ready = (
                self.plan_client.wait_for_service(timeout_sec=0.2)
                and self.execute_next_client.wait_for_service(timeout_sec=0.2)
                and self.clear_queue_client.wait_for_service(timeout_sec=0.2)
                and self.gripper_client.wait_for_server(timeout_sec=0.2)
                and self.stop_client.wait_for_service(timeout_sec=0.2)
            )
            if ready:
                break

        while rclpy.ok() and (self.center is None or self.tcp_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            "All selected-arm TRUE-TCP services/state ready"
        )

    def wait_stopped(self):
        req = Trigger.Request()
        future = self.stop_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("Robot stop service returned no response")
        if not result.success:
            raise RuntimeError(f"Robot did not stop: {result.message}")
        self.get_logger().info(result.message)

    def gripper_width(self, width, require_grasp=False):
        """
        Command TOTAL gripper opening width in meters via
        /mujoco/gripper_command.

        For opening, reaching the requested physical width is sufficient.
        For grasping, reached_goal=True is required because the bridge uses
        that flag to confirm that a nearby freejoint object was attached.
        """
        width = float(width)

        goal = GripperCommand.Goal()
        goal.command.position = width

        # Use max_effort as an explicit command-intent flag for this custom
        # MuJoCo bridge:
        #   0.0 -> OPEN/RELEASE command: never attach an object
        #   1.0 -> GRASP command: attachment is allowed after closing
        #
        # This avoids inferring grasp/release intent from width direction.
        goal.command.max_effort = 1.0 if require_grasp else 0.0

        mode = "GRASP" if require_grasp else "OPEN"
        self.get_logger().info(
            f"Gripper {mode}: target total width = {width:.4f} m"
        )

        send_future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(
                f"Gripper {mode} goal was rejected at width {width:.4f} m"
            )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()

        if wrapped is None:
            raise RuntimeError("Gripper action returned no result")

        result = wrapped.result
        reported_width = float(result.position)
        width_error = abs(reported_width - width)

        if require_grasp:
            # Keep grasp validation strict: reached_goal=True means the bridge
            # successfully attached the nearest freejoint object.
            if not result.reached_goal:
                raise RuntimeError(
                    f"Grasp failed at target width {width:.4f} m; "
                    f"reported width={reported_width:.4f} m, "
                    f"width_error={width_error:.6f} m, "
                    f"stalled={result.stalled}"
                )
        else:
            # Opening near the Panda mechanical maximum may settle a few
            # millimeters below the requested 0.080 m in MuJoCo. Accept the
            # opening once it reaches at least 90% of the requested width,
            # while still accepting bridge reached_goal=True immediately.
            min_acceptable_open_width = max(0.0, width * 0.90)
            open_width_reached = reported_width >= min_acceptable_open_width

            if not result.reached_goal and not open_width_reached:
                raise RuntimeError(
                    f"Gripper open failed at target width {width:.4f} m; "
                    f"reported width={reported_width:.4f} m, "
                    f"minimum acceptable={min_acceptable_open_width:.4f} m, "
                    f"width_error={width_error:.6f} m"
                )

            if not result.reached_goal and open_width_reached:
                self.get_logger().warn(
                    f"Gripper OPEN accepted below exact target: "
                    f"target={width:.4f} m, reported={reported_width:.4f} m, "
                    f"minimum acceptable={min_acceptable_open_width:.4f} m"
                )

        self.get_logger().info(
            f"Gripper {mode} OK: target={width:.4f} m, "
            f"reported={reported_width:.4f} m"
        )

    def prepare_gripper_for_grasp(self):
        """Always restore the gripper to the full 0.08 m opening before grasping."""
        self.gripper_width(0.08, require_grasp=False)

    def release_gripper_after_place(self):
        """Open to the YAML gripper_open_width after placing the object."""
        width = float(self.param("gripper_open_width"))
        self.gripper_width(width, require_grasp=False)

    def close_gripper_for_grasp(self):
        width = float(self.param("grasp_width"))
        self.gripper_width(width, require_grasp=True)

    def load_sequence_reference(self):
        """
        Load bridge-start GLOBAL Q0.

        Q0 is panda2_hand / true-TCP WORLD orientation captured by the bridge
        before pick_place/screw task execution.  This task never creates or
        recalibrates Q0.
        """
        path = self.orientation_reference_file
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Bridge-start GLOBAL Q0 not found: {path}. "
                "Start/restart the modified bridge before running tasks."
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            q = [
                float(data["qx"]),
                float(data["qy"]),
                float(data["qz"]),
                float(data["qw"]),
            ]
            n = math.sqrt(sum(v*v for v in q))
            if n <= 1e-12:
                raise ValueError("zero quaternion")
            q = [v/n for v in q]
        except Exception as e:
            raise RuntimeError(f"Cannot load bridge-start GLOBAL Q0: {e}")

        self.get_logger().info(f"Loaded BRIDGE-START TCP Q0 = {q}")
        return q

    def save_initial_pose(self):
        if self.center is None or self.tcp_pose is None:
            raise RuntimeError("Cannot save initial state: no selected-arm TCP state")

        self.initial_center = self.center.copy()
        self.initial_tcp_pose = (
            self.tcp_pose[0].copy(),
            self.tcp_pose[1].copy(),
        )
        self.sequence_base_orientation = self.load_sequence_reference()

        self.get_logger().info(
            f"Current true TCP position    = {self.initial_tcp_pose[0]}"
        )
        self.get_logger().info(
            f"Current true TCP orientation = {self.initial_tcp_pose[1]}"
        )
        self.get_logger().info(
            f"Bridge-start GLOBAL Q0       = {self.sequence_base_orientation}"
        )


    def clear_plan_queue(self):
        req = Trigger.Request()
        future = self.clear_queue_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("Clear plan queue service returned no response")
        if not result.success:
            raise RuntimeError(f"Cannot clear plan queue: {result.message}")
        self.get_logger().info(result.message)

    def queue_tcp_pose(self, position, orientation, linear=False):
        req = PlanToPose.Request()
        req.target.header.frame_id = "world"
        req.target.header.stamp = self.get_clock().now().to_msg()
        req.target.pose.position.x = float(position[0])
        req.target.pose.position.y = float(position[1])
        req.target.pose.position.z = float(position[2])
        req.target.pose.orientation.x = float(orientation[0])
        req.target.pose.orientation.y = float(orientation[1])
        req.target.pose.orientation.z = float(orientation[2])
        req.target.pose.orientation.w = float(orientation[3])

        # linear=True asks the planner to prefer a Cartesian straight-line
        # segment with the end-effector orientation locked to the segment start.
        req.linear = bool(linear)
        req.execute = False
        req.velocity_scale = self.get_velocity_scale()
        req.acceleration_scale = self.get_acceleration_scale()

        self.get_logger().info(
            f"PREPLAN {'LINEAR' if linear else 'JOINT'} "
            f"TRUE_TCP=[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] "
            f"q={orientation}"
        )

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("Selected-arm TCP planner returned no response")
        if not result.success:
            raise RuntimeError(f"Selected-arm TCP preplanning failed: {result.message}")
        self.get_logger().info(result.message)

    def execute_next_plan(self, name):
        self.get_logger().info(f"EXECUTE: {name}")
        req = Trigger.Request()
        future = self.execute_next_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("Execute-next service returned no response")
        if not result.success:
            raise RuntimeError(f"Cached plan execution failed: {result.message}")
        self.get_logger().info(result.message)
        self.wait_stopped()

        # Refresh fingertip messages after execution.
        for _ in range(6):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.center is not None:
            self.get_logger().info(f"Actual fingertip = {self.center}")


    def build_all_targets(self):
        grasp = self.get_grasp_pos()

        approach_distance = abs(float(self.param("approach_distance")))
        approach = [
            grasp[0],
            grasp[1],
            grasp[2] + approach_distance,
        ]

        lift = [
            grasp[0],
            grasp[1],
            grasp[2] + float(self.param("lift_height")),
        ]

        # YAML place_x/y/z is the FINAL place/insert pose.
        place = [
            float(self.param("place_x")),
            float(self.param("place_y")),
            float(self.param("place_z")),
        ]

        direction, axis, sign = self.get_place_direction()
        distance = abs(float(self.param("place_distance")))
        if distance <= 0.0:
            raise RuntimeError("place_distance must be > 0")

        # No YAML "above" pose. Derive the entrance automatically.
        place_start = place.copy()
        place_start[axis] -= sign * distance

        retreat_distance = max(0.0, float(self.param("retreat_distance")))
        retreat = place.copy()
        retreat[axis] -= sign * retreat_distance

        self.get_logger().info(
            f"Place/insert: direction={direction.upper()}, "
            f"distance={distance:.4f} m, start={place_start}, end={place}"
        )

        return {
            "approach": approach,
            "grasp": grasp,
            "lift": lift,
            "place_start": place_start,
            "place": place,
            "retreat": retreat,
        }

    def preplan_entire_task(self):
        self.clear_plan_queue()

        targets = self.build_all_targets()

        # IMPORTANT: angle zero is the panda2_hand / TRUE-TCP WORLD
        # orientation captured by the bridge at bridge startup.
        # The planner below controls that exact same orientation frame.
        if self.sequence_base_orientation is None:
            raise RuntimeError("Sequence orientation reference is not initialized")

        base_orientation = self.sequence_base_orientation
        grasp_orientation = self.rotated_orientation(
            base_orientation, self.param("grasp_rx_deg"), self.param("grasp_ry_deg")
        )
        place_orientation = self.rotated_orientation(
            base_orientation, self.param("place_rx_deg"), self.param("place_ry_deg")
        )

        self.get_logger().info(
            f"Sequence base orientation = {base_orientation}"
        )

        # Unified pick -> place/insert.
        # approach is derived from grasp + approach_distance; YAML has no above pose.
        # Move to approach with OMPL, then approach grasp with a straight Cartesian segment.
        # Final place/insert motion is Cartesian along +/-Y or +/-Z.
        segments = [
            ("approach", False),
            ("grasp", True),
            ("lift", True),
            ("place_start", False),
            ("place", True),
        ]

        if bool(self.param("open_and_lift_after_place")):
            segments.append(("retreat", True))

        try:
            for name, linear in segments:
                fingertip_target = targets[name]
                orientation = (
                    grasp_orientation
                    if name in ("approach", "grasp", "lift")
                    else place_orientation
                )
                self.get_logger().info(
                    f"========== PREPLAN {name.upper()} "
                    f"({'LINEAR' if linear else 'JOINT'}) "
                    f"true_tcp={fingertip_target}, q={orientation} =========="
                )
                # DIRECT TRUE-TCP target. No panda_link8 TF conversion.
                self.queue_tcp_pose(
                    fingertip_target,
                    orientation,
                    linear=linear,
                )

        except Exception:
            # Server also clears on preplanning failure, but clear explicitly here too.
            try:
                self.clear_plan_queue()
            except Exception:
                pass
            raise

        self.get_logger().info("========================================")
        self.get_logger().info("ALL ARM PATHS PLANNED SUCCESSFULLY")
        self.get_logger().info("ROBOT HAS NOT MOVED YET")
        self.get_logger().info("========================================")

    def run(self):
        self.wait_services()
        self.get_logger().info("Waiting for initial robot state to settle...")
        self.wait_stopped()

        settle_time = float(self.param("initial_settle_time"))
        if settle_time > 0.0:
            time.sleep(settle_time)

        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.save_initial_pose()

        # ============================================================
        # PHASE A: PLAN EVERYTHING FIRST. NO ROBOT MOTION IS EXECUTED.
        # ============================================================
        self.preplan_entire_task()

        # ============================================================
        # PHASE B: ONLY AFTER EVERY ARM SEGMENT SUCCEEDED, EXECUTE.
        # Gripper actions are interleaved between cached arm trajectories.
        # ============================================================
        self.get_logger().info("========== START PICK ACTION EXECUTION ==========")

        self.get_logger().info(
            "1. Prepare for grasp: restore gripper to 0.0800 m"
        )
        self.prepare_gripper_for_grasp()
        time.sleep(0.3)

        self.execute_next_plan("2. Move to pre-grasp approach")
        self.execute_next_plan(
            f"3. Approach grasp straight down "
            f"{abs(float(self.param('approach_distance'))):.4f} m"
        )
        time.sleep(0.3)

        self.get_logger().info(
            f"4. Close and grasp at {float(self.param('grasp_width')):.4f} m"
        )
        self.close_gripper_for_grasp()
        time.sleep(0.5)

        self.execute_next_plan("5. Lift")
        time.sleep(0.4)

        self.execute_next_plan("6. Transfer to derived place/insert start")

        direction, _, _ = self.get_place_direction()
        distance = abs(float(self.param("place_distance")))
        self.execute_next_plan(
            f"7. Place/insert along {direction.upper()} {distance:.4f} m"
        )

        if bool(self.param("open_and_lift_after_place")):
            self.wait_stopped()
            time.sleep(0.4)
            release_width = float(self.param("gripper_open_width"))
            self.get_logger().info(
                f"8. Release after place to {release_width:.4f} m"
            )
            self.release_gripper_after_place()
            time.sleep(0.5)
            self.execute_next_plan(
                f"9. Retreat opposite {direction.upper()}"
            )
        else:
            self.get_logger().info(
                "open_and_lift_after_place=false: finish at final pose; "
                "gripper remains closed"
            )

        self.get_logger().info("========================================")
        self.get_logger().info("UNIFIED PICK ACTION FINISHED")
        self.get_logger().info("========================================")



def main(args=None):
    rclpy.init(args=args)
    node = PickActionTask()

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

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener
from panda_moveit_planner.srv import PlanToPose


class PickPlaceTask(Node):
    def __init__(self):
        super().__init__("panda_pick_place_task")

        # ============================================================
        # Parameters from task.yaml
        # ============================================================
        self.declare_parameter("grasp_x", 0.25)
        self.declare_parameter("grasp_y", 0.12)
        self.declare_parameter("grasp_z", 0.85)

        self.declare_parameter("above_x", 0.25)
        self.declare_parameter("above_y", 0.12)
        self.declare_parameter("above_z", 0.95)

        self.declare_parameter("lift_height", 0.05)

        self.declare_parameter("horizontal_dx", -0.188)
        self.declare_parameter("horizontal_dy", 0.028)
        self.declare_parameter("horizontal_dz", 0.0)

        self.declare_parameter("place_down_distance", 0.043)
        self.declare_parameter("retreat_height", 0.05)

        # Unified MoveIt speed
        self.declare_parameter("velocity_scale", 0.15)
        self.declare_parameter("acceleration_scale", 0.10)

        # Initial stabilization
        self.declare_parameter("initial_settle_time", 0.5)

        # Debug
        self.declare_parameter("debug_mode", False)

        # ============================================================
        # State
        # ============================================================
        self.center = None
        self.initial_center = None
        self.initial_link8_pose = None

        # ============================================================
        # ROS interfaces
        # ============================================================
        self.create_subscription(
            PointStamped,
            "/mujoco/fingertip_center",
            self.center_cb,
            20
        )

        self.plan_client = self.create_client(
            PlanToPose,
            "/plan_to_pose"
        )

        self.gripper_client = self.create_client(
            SetBool,
            "/mujoco/gripper"
        )

        self.stop_client = self.create_client(
            Trigger,
            "/mujoco/wait_until_stopped"
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

    # ================================================================
    # Parameters
    # ================================================================
    def param(self, name):
        return self.get_parameter(name).value

    def get_grasp_pos(self):
        return [
            float(self.param("grasp_x")),
            float(self.param("grasp_y")),
            float(self.param("grasp_z"))
        ]

    def get_above_grasp(self):
        return [
            float(self.param("above_x")),
            float(self.param("above_y")),
            float(self.param("above_z"))
        ]

    def get_horizontal_delta(self):
        return [
            float(self.param("horizontal_dx")),
            float(self.param("horizontal_dy")),
            float(self.param("horizontal_dz"))
        ]

    def get_velocity_scale(self):
        return float(self.param("velocity_scale"))

    def get_acceleration_scale(self):
        return float(self.param("acceleration_scale"))

    def print_parameters(self):
        self.get_logger().info("========================================")
        self.get_logger().info("TASK PARAMETERS")
        self.get_logger().info("========================================")
        self.get_logger().info(f"GRASP_POS           = {self.get_grasp_pos()}")
        self.get_logger().info(f"ABOVE_GRASP         = {self.get_above_grasp()}")
        self.get_logger().info(f"LIFT_HEIGHT         = {self.param('lift_height')}")
        self.get_logger().info(f"HORIZONTAL_DELTA    = {self.get_horizontal_delta()}")
        self.get_logger().info(f"PLACE_DOWN_DISTANCE = {self.param('place_down_distance')}")
        self.get_logger().info(f"RETREAT_HEIGHT      = {self.param('retreat_height')}")
        self.get_logger().info(f"VELOCITY_SCALE      = {self.get_velocity_scale()}")
        self.get_logger().info(f"ACCELERATION_SCALE  = {self.get_acceleration_scale()}")
        self.get_logger().info(f"INITIAL_SETTLE_TIME = {self.param('initial_settle_time')}")
        self.get_logger().info(f"DEBUG_MODE          = {self.param('debug_mode')}")
        self.get_logger().info("========================================")

    def debug_pause(self, step):
        if bool(self.param("debug_mode")):
            input(f"\nPress ENTER to execute step {step}...")

    # ================================================================
    # State callback
    # ================================================================
    def center_cb(self, msg):
        self.center = [
            msg.point.x,
            msg.point.y,
            msg.point.z
        ]

    # ================================================================
    # Wait ROS interfaces
    # ================================================================
    def wait_services(self):
        self.get_logger().info("Waiting for MoveIt/MuJoCo services...")

        while rclpy.ok():
            plan_ready = self.plan_client.wait_for_service(timeout_sec=0.2)
            gripper_ready = self.gripper_client.wait_for_service(timeout_sec=0.2)
            stop_ready = self.stop_client.wait_for_service(timeout_sec=0.2)

            if plan_ready and gripper_ready and stop_ready:
                break

        while rclpy.ok() and self.center is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("All services ready")

    # ================================================================
    # Wait robot fully stopped
    # ================================================================
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

    # ================================================================
    # Gripper
    # ================================================================
    def gripper(self, close):
        req = SetBool.Request()
        req.data = bool(close)

        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        if result is None:
            raise RuntimeError("Gripper service returned no response")

        if not result.success:
            raise RuntimeError(f"Gripper command failed: {result.message}")

        self.get_logger().info(result.message)

    # ================================================================
    # TF
    # ================================================================
    def get_link8_transform(self):
        for _ in range(100):
            try:
                return self.tf_buffer.lookup_transform(
                    "world",
                    "panda_link8",
                    rclpy.time.Time()
                )
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.05)

        raise RuntimeError("Cannot get TF: world -> panda_link8")

    # ================================================================
    # MoveIt request
    # ================================================================
    def plan_link8_pose(
        self,
        position,
        orientation,
        linear=False,
        execute=True
    ):
        # Pilz LIN requires a stationary start state
        if linear:
            self.get_logger().info(
                "Waiting for robot to settle before LIN..."
            )
            self.wait_stopped()

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

        req.linear = bool(linear)
        req.execute = bool(execute)

        # Speed from YAML
        req.velocity_scale = self.get_velocity_scale()
        req.acceleration_scale = self.get_acceleration_scale()

        self.get_logger().info(
            f"MoveIt target link8 = "
            f"[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}], "
            f"planner={'LIN' if linear else 'OMPL'}, "
            f"velocity={req.velocity_scale:.3f}, "
            f"acceleration={req.acceleration_scale:.3f}, "
            f"execute={execute}"
        )

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        if result is None:
            raise RuntimeError("MoveIt service returned no response")

        if not result.success:
            raise RuntimeError(f"MoveIt failed: {result.message}")

        self.get_logger().info(result.message)

    # ================================================================
    # Fingertip absolute target
    # ================================================================
    def move_fingertip_to(
        self,
        target_center,
        linear=False
    ):
        if self.center is None:
            raise RuntimeError("No fingertip center received")

        current_center = self.center.copy()

        tf = self.get_link8_transform()
        p = tf.transform.translation
        q = tf.transform.rotation

        dx = float(target_center[0]) - current_center[0]
        dy = float(target_center[1]) - current_center[1]
        dz = float(target_center[2]) - current_center[2]

        target_link8 = [
            p.x + dx,
            p.y + dy,
            p.z + dz
        ]

        orientation = [
            q.x,
            q.y,
            q.z,
            q.w
        ]

        self.get_logger().info(
            f"Fingertip current = "
            f"[{current_center[0]:.4f}, "
            f"{current_center[1]:.4f}, "
            f"{current_center[2]:.4f}]"
        )

        self.get_logger().info(
            f"Fingertip target  = "
            f"[{target_center[0]:.4f}, "
            f"{target_center[1]:.4f}, "
            f"{target_center[2]:.4f}]"
        )

        self.get_logger().info(
            f"World delta       = "
            f"[{dx:.4f}, {dy:.4f}, {dz:.4f}]"
        )

        self.plan_link8_pose(
            target_link8,
            orientation,
            linear=linear,
            execute=True
        )

        # Refresh latest MuJoCo state
        for _ in range(8):
            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

        if self.center is not None:
            self.get_logger().info(
                f"Actual fingertip  = "
                f"[{self.center[0]:.4f}, "
                f"{self.center[1]:.4f}, "
                f"{self.center[2]:.4f}]"
            )

    # ================================================================
    # Fingertip relative motion
    # ================================================================
    def move_fingertip_delta(
        self,
        delta,
        linear=True
    ):
        if self.center is None:
            raise RuntimeError("No fingertip center received")

        target = [
            self.center[0] + float(delta[0]),
            self.center[1] + float(delta[1]),
            self.center[2] + float(delta[2])
        ]

        self.move_fingertip_to(
            target,
            linear=linear
        )

    # ================================================================
    # Save initial pose
    # ================================================================
    def save_initial_pose(self):
        if self.center is None:
            raise RuntimeError(
                "Cannot save initial state: no fingertip center"
            )

        self.initial_center = self.center.copy()

        tf = self.get_link8_transform()
        p = tf.transform.translation
        q = tf.transform.rotation

        self.initial_link8_pose = (
            [p.x, p.y, p.z],
            [q.x, q.y, q.z, q.w]
        )

        self.get_logger().info(
            f"Initial fingertip = {self.initial_center}"
        )

    # ================================================================
    # Main task
    # ================================================================
    def run(self):
        # ------------------------------------------------------------
        # 0. Initialize and stabilize
        # ------------------------------------------------------------
        self.wait_services()

        self.get_logger().info(
            "Waiting for initial ROS/MuJoCo state..."
        )

        time.sleep(1.0)

        self.get_logger().info(
            "Waiting for initial robot state to settle..."
        )

        self.wait_stopped()

        settle_time = float(
            self.param("initial_settle_time")
        )

        if settle_time > 0.0:
            time.sleep(settle_time)

        # Refresh several joint / TF / fingertip messages after stopping
        for _ in range(10):
            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

        self.print_parameters()
        self.save_initial_pose()

        # ============================================================
        # 1. Move above L_part
        # ============================================================
        self.get_logger().info(
            "========== 1. Move above L_part =========="
        )

        self.debug_pause(1)

        self.move_fingertip_to(
            self.get_above_grasp(),
            linear=False
        )

        # ============================================================
        # 2. Open gripper
        # ============================================================
        self.get_logger().info(
            "========== 2. Open gripper =========="
        )

        self.debug_pause(2)
        self.gripper(False)
        time.sleep(0.3)

        # ============================================================
        # 3. Linear descend to grasp
        # ============================================================
        self.get_logger().info(
            "========== 3. Linear move to grasp =========="
        )

        self.debug_pause(3)

        self.move_fingertip_to(
            self.get_grasp_pos(),
            linear=True
        )

        # ============================================================
        # 4. Stabilize
        # ============================================================
        self.get_logger().info(
            "========== 4. Stabilize =========="
        )

        self.debug_pause(4)
        self.wait_stopped()
        time.sleep(0.3)

        # ============================================================
        # 5/6. Close and grasp
        # ============================================================
        self.get_logger().info(
            "========== 5/6. Close and grasp =========="
        )

        self.debug_pause(5)
        self.gripper(True)
        time.sleep(0.5)

        # ============================================================
        # 7. Vertical lift
        # ============================================================
        self.get_logger().info(
            "========== 7. Linear lift =========="
        )

        self.debug_pause(7)

        self.move_fingertip_delta(
            [
                0.0,
                0.0,
                float(self.param("lift_height"))
            ],
            linear=True
        )

        # ============================================================
        # 8. Hold
        # ============================================================
        self.get_logger().info(
            "========== 8. Hold =========="
        )

        self.debug_pause(8)
        self.wait_stopped()
        time.sleep(0.4)

        # ============================================================
        # 9. Horizontal translation
        # ============================================================
        self.get_logger().info(
            "========== 9. Linear horizontal translation =========="
        )

        self.debug_pause(9)

        self.move_fingertip_delta(
            self.get_horizontal_delta(),
            linear=True
        )

        # ============================================================
        # 10. Vertical placement
        # ============================================================
        self.get_logger().info(
            "========== 10. Linear vertical placement =========="
        )

        self.debug_pause(10)

        before_down = self.center.copy()

        self.move_fingertip_delta(
            [
                0.0,
                0.0,
                -float(
                    self.param(
                        "place_down_distance"
                    )
                )
            ],
            linear=True
        )

        after_down = self.center.copy()

        self.get_logger().info(
            f"Placement XY displacement = "
            f"[{after_down[0] - before_down[0]:.6f}, "
            f"{after_down[1] - before_down[1]:.6f}]"
        )

        self.get_logger().info(
            f"Placement Z displacement = "
            f"{after_down[2] - before_down[2]:.6f}"
        )

        # ============================================================
        # 11. Hold placement
        # ============================================================
        self.get_logger().info(
            "========== 11. Hold placement =========="
        )

        self.debug_pause(11)
        self.wait_stopped()
        time.sleep(0.4)

        # ============================================================
        # 12. Release
        # ============================================================
        self.get_logger().info(
            "========== 12. Release =========="
        )

        self.debug_pause(12)
        self.gripper(False)
        time.sleep(0.5)

        # ============================================================
        # 12.5 Retreat
        # ============================================================
        self.get_logger().info(
            "========== 12.5 Retreat upward =========="
        )

        self.debug_pause(12.5)

        self.move_fingertip_delta(
            [
                0.0,
                0.0,
                float(
                    self.param(
                        "retreat_height"
                    )
                )
            ],
            linear=True
        )

        # ============================================================
        # 13. Return
        # ============================================================
        self.get_logger().info(
            "========== 13. Return =========="
        )

        self.debug_pause(13)

        position, orientation = self.initial_link8_pose

        self.plan_link8_pose(
            position,
            orientation,
            linear=False,
            execute=True
        )

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "PICK AND PLACE FINISHED"
        )
        self.get_logger().info(
            "========================================"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceTask()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

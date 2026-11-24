import threading
import time
import json
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from panda_moveit_planner.srv import PlanToPose


class _ContinuousOrbitPlan(list):
    """Tagged trajectory: interpolate without stopping at internal waypoints."""
    pass


class MujocoMoveItBridge(Node):
    ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
    STATE_JOINTS = ARM_JOINTS + ["panda_finger_joint1"]
    MJ_ARM_JOINTS = [f"panda2_joint{i}" for i in range(1, 8)]
    MJ_STATE_JOINTS = MJ_ARM_JOINTS + ["panda2_finger_joint1"]

    P1_ARM_JOINTS = [f"panda1_joint{i}" for i in range(1, 8)]
    P1_STATE_JOINTS = P1_ARM_JOINTS + ["panda1_finger_joint1"]

    def __init__(self):
        super().__init__("mujoco_moveit_bridge")

        self.declare_parameter("scene_xml", "my_scene.xml")
        self.declare_parameter("panda_xml", "panda.xml")
        self.declare_parameter("viewer", True)
        self.declare_parameter("realtime_factor", 1.0)
        self.declare_parameter("gripper_close_ratio", 0.8)
        self.declare_parameter("trajectory_time_scale", 1.0)
        # Maximum body-origin distance from fingertip center for automatic grasp.
        # Increase this slightly if a mesh origin is far from its visible geometry.
        self.declare_parameter("grasp_max_distance", 0.12)

        scene_xml = self.get_parameter("scene_xml").value
        panda_xml = self.get_parameter("panda_xml").value
        self.use_viewer = bool(self.get_parameter("viewer").value)
        self.realtime_factor = float(self.get_parameter("realtime_factor").value)
        self.close_ratio = float(self.get_parameter("gripper_close_ratio").value)
        self.trajectory_time_scale = float(self.get_parameter("trajectory_time_scale").value)
        self.grasp_max_distance = float(self.get_parameter("grasp_max_distance").value)

        self.model, self.data = self._build_model(scene_xml, panda_xml)
        self.lock = threading.RLock()

        self.running = True
        self.grasped = False  # Legacy status; authoritative state is arm_grasps.
        self.arm_hold = True
        self.trajectory_active = False

        # Independent rigid grasp per arm. One object can have only one owner.
        self.arm_grasps = {"panda1": None, "panda2": None}
        self.grasped_object = None  # Legacy mirror for older integrations.
        self.grasp_robot = None
        self.grasp_offset = np.zeros(3, dtype=float)
        self.grasp_rel_pos_hand = np.zeros(3, dtype=float)
        self.grasp_rel_R_hand = np.eye(3, dtype=float)
        self.pick_objects = []
        # Rigid tip-to-object attachments, keyed by target body name.
        self.hex_tip_attachments = {}
        self.hex_tip_ever_attached = False  # latch: never attach a second time after detach
        self.hex_tip_contact_radius = 0.012  # m: contact must be close to the tip site
        self.hex_tip_auto_contact_steps = 3
        self.hex_tip_contact_streak = 0
        # Enabled by pick_place only after its complete execution sequence.
        self.part_32_33_auto_attach_enabled = False
        self.part_32_33_attachment = None
        self.part_32_33_contact_streak = 0
        self.part_32_33_contact_steps = 3
        self.part_32_33_attach_gap = 0.0015  # 1.5 mm mesh-surface tolerance
        self.part_32_33_check_count = 0
        self.part_32_33_last_report = 0.0

        # Screw action state. The task configures direction/speed/distance plus
        # YAML target tip XYZ. /mujoco/screw_start selects the freejoint object
        # nearest that YAML target. Progress still follows actual tip displacement.
        self.screw_direction = np.array([0.0, 0.0, -1.0], dtype=float)
        self.screw_shaft_speed = 4.0
        self.screw_down_distance = 0.005
        # Authoritative WORLD target point from YAML screw_i_tip_x/y/z.
        # Used only to select which screw object will move.
        self.screw_target_tip = None
        self.screw_active = False
        self.screw_selected_object = None
        self.screw_start_tip = None
        self.screw_start_qpos = None

        # During screw rotation, freeze the gripper of the robot that is NOT
        # holding the screwdriver.  Store both joint state and actuator target
        # so MuJoCo does not fight a teleported qpos with a different actuator
        # command (the source of the visible close-then-open transient).
        self.screw_hold_robot = None
        self.screw_hold_finger_qpos = None
        self.screw_hold_finger_ctrl = None
        # Also stabilize the screwdriver-owning gripper during motor rotation.
        self.screw_owner_hold_robot = None
        self.screw_owner_hold_qpos = None
        self.screw_owner_hold_ctrl = None

        # Do not release the inactive gripper hold in the exact simulation
        # frame in which screwdriver rotation stops.  Keep the hold for a few
        # complete MuJoCo steps so actuator/equality dynamics settle first.
        self.screw_stop_hold_steps = 5
        self.screw_hold_release_steps = 0

        # 7-D arm
        self.arm_joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.MJ_ARM_JOINTS]
        self.arm_qadr = np.array([self.model.jnt_qposadr[j] for j in self.arm_joints], dtype=int)
        self.arm_vadr = np.array([self.model.jnt_dofadr[j] for j in self.arm_joints], dtype=int)

        # 8-D /joint_states
        self.state_joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.MJ_STATE_JOINTS]
        self.state_qadr = np.array([self.model.jnt_qposadr[j] for j in self.state_joints], dtype=int)
        self.state_vadr = np.array([self.model.jnt_dofadr[j] for j in self.state_joints], dtype=int)

        # Hand / fingers
        self.hand = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda2_hand")
        self.finger_act = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "panda2_actuator8")
        self.finger_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, "panda2_finger_joint1"
        )
        self.finger_joints = [
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "panda2_finger_joint1"),
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "panda2_finger_joint2"),
        ]
        self.finger_qadr = np.array(
            [self.model.jnt_qposadr[j] for j in self.finger_joints], dtype=int
        )
        self.finger_vadr = np.array(
            [self.model.jnt_dofadr[j] for j in self.finger_joints], dtype=int
        )
        self.left_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda2_left_finger")
        self.right_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda2_right_finger")

        self.left_geoms = [
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == self.left_body and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        self.right_geoms = [
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == self.right_body and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        if not self.left_geoms or not self.right_geoms:
            raise RuntimeError("Cannot find panda2 fingertip box geoms")

        # Panda1 direct-control state.  Panda1 is intentionally kept separate
        # from the existing MoveIt-facing Panda2 interface.
        self.p1_arm_joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.P1_ARM_JOINTS]
        self.p1_arm_qadr = np.array([self.model.jnt_qposadr[j] for j in self.p1_arm_joints], dtype=int)
        self.p1_arm_vadr = np.array([self.model.jnt_dofadr[j] for j in self.p1_arm_joints], dtype=int)
        self.p1_state_joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.P1_STATE_JOINTS]
        self.p1_state_qadr = np.array([self.model.jnt_qposadr[j] for j in self.p1_state_joints], dtype=int)
        self.p1_state_vadr = np.array([self.model.jnt_dofadr[j] for j in self.p1_state_joints], dtype=int)
        self.p1_hand = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda1_hand")
        self.p1_finger_act = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "panda1_actuator8")
        self.p1_finger_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "panda1_finger_joint1")
        # Panda1 two-finger addresses are also used by the generic screw-time
        # inactive-gripper hold below.
        self.p1_finger_joints = [
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "panda1_finger_joint1"),
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "panda1_finger_joint2"),
        ]
        self.p1_finger_qadr = np.array(
            [self.model.jnt_qposadr[j] for j in self.p1_finger_joints], dtype=int
        )
        self.p1_finger_vadr = np.array(
            [self.model.jnt_dofadr[j] for j in self.p1_finger_joints], dtype=int
        )
        self.p1_left_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda1_left_finger")
        self.p1_right_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda1_right_finger")
        self.p1_left_geoms = [i for i in range(self.model.ngeom)
                              if self.model.geom_bodyid[i] == self.p1_left_body
                              and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
        self.p1_right_geoms = [i for i in range(self.model.ngeom)
                               if self.model.geom_bodyid[i] == self.p1_right_body
                               and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
        if not self.p1_left_geoms or not self.p1_right_geoms:
            raise RuntimeError("Cannot find panda1 fingertip box geoms")

        # Automatically discover every scene object whose body owns a free joint.
        # Panda robot bodies are excluded by name prefix.
        self._discover_pick_objects()
        self.part_32 = next((o for o in self.pick_objects if o["name"] == "part_3_2"), None)
        self.part_33 = next((o for o in self.pick_objects if o["name"] == "part_3_3"), None)
        if self.part_32 is None or self.part_33 is None:
            raise RuntimeError("Scene needs freejoint part_3_2 and part_3_3")
        self.part_32_geom_ids = frozenset(self._object_geom_ids(self.part_32))
        self.part_33_geom_ids = frozenset(self._object_geom_ids(self.part_33))
        self.hex_key_tip_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "hex_key_tip")
        self.hex_key_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "hex_key")
        self.hex_key_object = next(
            (obj for obj in self.pick_objects if obj["name"] == "hex_key"), None
        )
        if self.hex_key_object is None:
            raise RuntimeError("Scene needs a freejoint hex_key body")
        # Cache geom ownership once, not every simulation step.
        self.hex_geom_ids = frozenset(self._object_geom_ids(self.hex_key_object))
        self.hex_candidate_by_geom = {}
        for obj in self.pick_objects:
            if obj["name"] in ("hex_key", "screwdriver"):
                continue
            for gid in self._object_geom_ids(obj):
                self.hex_candidate_by_geom[gid] = obj
        self.hex_contact_check_counter = 0
        self.hex_tip_candidate_name = None
        self.hex_contact_check_interval = 10  # reduce Python work in simulation loop


        # Motorized screwdriver elements from WRS scene.
        self.screwdriver_tip_site = self._id(
            mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip"
        )
        self.screwdriver_motor = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "screwdriver_motor"
        )

        # Initial state
        self.q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)

        self._initialize_state()

        # Commands
        lo, hi = self.model.actuator_ctrlrange[self.finger_act]
        self.open_ctrl = hi
        self.close_ctrl = lo
        self.half_close_ctrl = lo + self.close_ratio * (hi - lo)

        self.q_command = self.q0.copy()
        self.v_command = np.zeros(7, dtype=float)
        self.gripper_command = self.open_ctrl

        self.p1_q_command = self.q0.copy()
        self.p1_v_command = np.zeros(7, dtype=float)
        p1_lo, p1_hi = self.model.actuator_ctrlrange[self.p1_finger_act]
        self.p1_open_ctrl = float(max(p1_lo, p1_hi))
        self.p1_gripper_command = self.p1_open_ctrl
        self.p1_trajectory_active = False

        # UI-like direct single-joint command (one arm, one joint, relative angle).
        self.direct_joint_motion = None
        self.direct_joint_last_seq = None

        with self.lock:
            self.data.qpos[self.arm_qadr] = self.q_command
            self.data.qvel[self.arm_vadr] = 0.0
            self.arm_hold = True
            self.trajectory_active = False
            mujoco.mj_forward(self.model, self.data)

        # ============================================================
        # GLOBAL task orientation reference
        # ============================================================
        # Q0 means the Panda2 hand/TCP orientation at BRIDGE STARTUP,
        # after the bridge initial state has been applied and BEFORE any
        # pick_place/screw task can command robot motion.
        #
        # The task runner must NOT delete or recreate this file.
        self.orientation_reference_file = os.environ.get(
            "PANDA_TASK_ORIENTATION_REFERENCE",
            "/tmp/panda_task_initial_orientation.json",
        )
        self._save_bridge_start_orientation_reference()

        # ROS
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 20)
        self.center_pub = self.create_publisher(PointStamped, "/mujoco/fingertip_center", 20)
        self.p2_tcp_pose_pub = self.create_publisher(
            PoseStamped, "/mujoco/panda2_tcp_pose", 20
        )
        self.p1_joint_pub = self.create_publisher(JointState, "/mujoco/panda1_joint_states", 20)
        self.p1_center_pub = self.create_publisher(PointStamped, "/mujoco/panda1_fingertip_center", 20)
        self.p1_tcp_pose_pub = self.create_publisher(
            PoseStamped, "/mujoco/panda1_tcp_pose", 20
        )

        self.gripper_srv = self.create_service(
            SetBool, "/mujoco/gripper", self._gripper_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.gripper_action_server = ActionServer(
            self,
            GripperCommand,
            "/mujoco/gripper_command",
            execute_callback=self._execute_gripper_width,
            goal_callback=self._gripper_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.settle_srv = self.create_service(
            Trigger, "/mujoco/wait_until_stopped", self._wait_until_stopped_cb,
            callback_group=ReentrantCallbackGroup()
        )

        self.screw_config_sub = self.create_subscription(
            Float64MultiArray,
            "/mujoco/screw_config",
            self._screw_config_cb,
            10,
            callback_group=ReentrantCallbackGroup(),
        )
        # Push-only contact latch. Enabled exclusively for the straight push segment.
        self.push_active = {"panda1": False, "panda2": False}
        self.push_object = {"panda1": None, "panda2": None}
        # Only the FIRST search trajectory may be interrupted by contact.
        self.push_search_interrupted = {"panda1": False, "panda2": False}
        self.push_contact_gap = 0.000  # signed collision-geom surface distance <= 0 mm
        self.push_contact_check_interval = 5  # simulation cycles between queries
        self.push_contact_cycle = 0
        self.push_previous_sample = {"panda1": None, "panda2": None}
        # The task supplies the WORLD push axis before enabling the latch.
        self.push_first_contact_tcp = {"panda1": None, "panda2": None}
        self.push_status_services = [
            self.create_service(Trigger, f"/mujoco/{arm}/push_status",
                lambda request, response, a=arm: self._push_status_cb(a, request, response),
                callback_group=ReentrantCallbackGroup())
            for arm in ("panda1", "panda2")
        ]
        self.push_direction = {"panda1": None, "panda2": None}
        self.push_direction_sub = self.create_subscription(
            Float64MultiArray, "/mujoco/push_direction",
            self._push_direction_cb, 10,
            callback_group=ReentrantCallbackGroup())
        self.push_front_tolerance = 0.002  # m, tolerance for front-most fingertip face
        self.push_services = [
            self.create_service(
                SetBool, f"/mujoco/{arm}/push_contact",
                lambda request, response, a=arm: self._push_contact_cb(a, request, response),
                callback_group=ReentrantCallbackGroup())
            for arm in ("panda1", "panda2")
        ]
        self.part_32_33_enable_srv = self.create_service(
            Trigger, "/mujoco/enable_part_3_2_3_3_attach",
            self._enable_part_32_33_attach_cb,
            callback_group=ReentrantCallbackGroup(),
        )
        self.hex_tip_attach_srv = self.create_service(
            Trigger, "/mujoco/hex_key_tip_attach", self._hex_tip_attach_cb,
            callback_group=ReentrantCallbackGroup(),
        )
        self.hex_tip_axis_srv = self.create_service(
            Trigger, "/mujoco/hex_key_attached_axis", self._hex_key_attached_axis_cb,
            callback_group=ReentrantCallbackGroup(),
        )
        self.hex_tip_detach_srv = self.create_service(
            Trigger, "/mujoco/hex_key_tip_detach", self._hex_tip_detach_cb,
            callback_group=ReentrantCallbackGroup(),
        )
        self.screw_start_srv = self.create_service(
            Trigger,
            "/mujoco/screw_start",
            self._screw_start_cb,
            callback_group=ReentrantCallbackGroup(),
        )
        self.screw_stop_srv = self.create_service(
            Trigger,
            "/mujoco/screw_stop",
            self._screw_stop_cb,
            callback_group=ReentrantCallbackGroup(),
        )

        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.p1_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/panda1_arm_controller/follow_joint_trajectory",
            execute_callback=self._execute_p1_trajectory,
            goal_callback=self._p1_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        self.p1_gripper_action_server = ActionServer(
            self,
            GripperCommand,
            "/mujoco/panda1_gripper_command",
            execute_callback=self._execute_p1_gripper_width,
            goal_callback=self._p1_gripper_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.p1_plan_queue = []
        self.p1_queue_target_R = None

        # Panda2 general pose-plan queue.  This mirrors the Panda1 service API so
        # task YAML can select robot: panda1 or robot: panda2 without changing
        # the task script.
        self.p2_plan_queue = []
        self.p2_queue_target_R = None
        self.p1_plan_srv = self.create_service(
            PlanToPose,
            "/panda1/plan_to_pose",
            self._p1_plan_to_pose_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.p1_exec_srv = self.create_service(
            Trigger,
            "/panda1/execute_next_plan",
            self._p1_execute_next_plan_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.p1_clear_srv = self.create_service(
            Trigger,
            "/panda1/clear_plan_queue",
            self._p1_clear_plan_queue_cb,
            callback_group=ReentrantCallbackGroup()
        )

        # Matching Panda2 API used by screw.py when YAML has robot: "panda2".
        self.p2_plan_srv = self.create_service(
            PlanToPose,
            "/panda2/plan_to_pose",
            self._p2_plan_to_pose_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.p2_exec_srv = self.create_service(
            Trigger,
            "/panda2/execute_next_plan",
            self._p2_execute_next_plan_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.p1_orbit_srv = self.create_service(
            PlanToPose, "/mujoco/panda1_plan_continuous_orbit",
            lambda req, resp: self._plan_continuous_orbit("panda1", req, resp),
            callback_group=ReentrantCallbackGroup(),
        )
        self.p2_orbit_srv = self.create_service(
            PlanToPose, "/mujoco/panda2_plan_continuous_orbit",
            lambda req, resp: self._plan_continuous_orbit("panda2", req, resp),
            callback_group=ReentrantCallbackGroup(),
        )
        self.p2_clear_srv = self.create_service(
            Trigger,
            "/panda2/clear_plan_queue",
            self._p2_clear_plan_queue_cb,
            callback_group=ReentrantCallbackGroup()
        )

        # Handover-only Panda2 relative-Z cache.  This avoids calling the
        # external MoveIt planner during handover while still precomputing the
        # entire +Z path before any robot motion.
        self.p2_relative_z_plan = None
        self.p2_relz_plan_srv = self.create_service(
            PlanToPose,
            "/mujoco/panda2_plan_relative_z",
            self._p2_plan_relative_z_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.p2_relz_exec_srv = self.create_service(
            Trigger,
            "/mujoco/panda2_execute_relative_z",
            self._p2_execute_relative_z_cb,
            callback_group=ReentrantCallbackGroup()
        )

        # Command: [sequence_id, arm(1/2), joint_index(1..7), delta_deg, duration_s]
        # Status: [sequence_id, 0=accepted/1=done/-1=rejected]
        self.direct_joint_status_pub = self.create_publisher(
            Float64MultiArray, "/mujoco/direct_joint_status", 10
        )
        self.direct_joint_sub = self.create_subscription(
            Float64MultiArray, "/mujoco/direct_joint_relative",
            self._direct_joint_relative_cb, 10,
            callback_group=ReentrantCallbackGroup()
        )

        self.viewer = (
            mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            if self.use_viewer else None
        )
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = [0.0, 0.15, 0.8]
            self.viewer.cam.distance = 1.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30

        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("MuJoCo bridge ready")
        self.get_logger().info(
            "Task pose services: "
            "/panda1/{plan_to_pose,execute_next_plan,clear_plan_queue}, "
            "/panda2/{plan_to_pose,execute_next_plan,clear_plan_queue}"
        )
        self.get_logger().info("Action: /panda_arm_controller/follow_joint_trajectory")
        self.get_logger().info("Services: /mujoco/gripper, /mujoco/wait_until_stopped")
        self.get_logger().info("Action: /mujoco/gripper_command (total opening width in meters)")
        self.get_logger().info(
            "Joint state mapping: panda_joint1..7 + panda_finger_joint1 "
            "<-> panda2_joint1..7 + panda2_finger_joint1"
        )
        self.get_logger().info(
            f"Kinematic MoveIt trajectory following enabled, time scale={self.trajectory_time_scale:.2f}"
        )
        self.get_logger().info(
            f"Automatic nearest-object grasp enabled, max distance={self.grasp_max_distance:.3f} m"
        )

    def _save_bridge_start_orientation_reference(self):
        """
        Save the sequence-global Q0 exactly once per bridge process startup.

        Q0 is the Panda2 hand orientation after _initialize_state() and the
        bridge's initial q_command have been applied.  It is therefore the
        orientation BEFORE any pick_place/screw task executes.

        File schema intentionally contains orientation only:
            qx, qy, qz, qw
        TCP offsets are not stored here.
        """
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            R0 = self.data.xmat[self.hand].reshape(3, 3).copy()

        q0 = self._R_to_ros_quat(R0)
        path = self.orientation_reference_file
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "qx": float(q0[0]),
                    "qy": float(q0[1]),
                    "qz": float(q0[2]),
                    "qw": float(q0[3]),
                },
                f,
                indent=2,
            )
        os.replace(tmp, path)

        self.get_logger().info(
            "Captured BRIDGE-START GLOBAL Q0 "
            f"(panda2 hand/TCP): {q0.tolist()}"
        )
        self.get_logger().info(
            f"GLOBAL Q0 reference saved to: {path}"
        )

    def _build_model(self, scene_xml, panda_xml):
        if not Path(scene_xml).exists():
            raise FileNotFoundError(f"scene_xml not found: {scene_xml}")
        if not Path(panda_xml).exists():
            raise FileNotFoundError(f"panda_xml not found: {panda_xml}")

        scene = mujoco.MjSpec.from_file(scene_xml)
        panda1 = mujoco.MjSpec.from_file(panda_xml)
        panda2 = mujoco.MjSpec.from_file(panda_xml)

        frame1 = scene.worldbody.add_frame(pos=[-0.25, 0.45, 0.8], quat=[0.7071, 0, 0, -0.7071])
        frame2 = scene.worldbody.add_frame(pos=[0.25, 0.45, 0.8], quat=[0.7071, 0, 0, -0.7071])
        scene.attach(panda1, prefix="panda1_", frame=frame1)
        scene.attach(panda2, prefix="panda2_", frame=frame2)

        model = scene.compile()
        model.opt.gravity[:] = 0.0
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        return model, mujoco.MjData(model)

    def _id(self, obj, name):
        i = mujoco.mj_name2id(self.model, obj, name)
        if i < 0:
            raise ValueError(f"Cannot find {name}")
        return i

    def _discover_pick_objects(self):
        """Find all non-robot bodies that are directly attached to a free joint."""
        objects = []

        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue

            body_id = int(self.model.jnt_bodyid[jid])
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jid
            )

            if body_name is None:
                continue

            # The two Panda models do not normally contain free joints, but
            # exclude them explicitly so future robot XML changes stay safe.
            if body_name.startswith("panda1_") or body_name.startswith("panda2_"):
                continue

            qpos_adr = int(self.model.jnt_qposadr[jid])
            qvel_adr = int(self.model.jnt_dofadr[jid])

            objects.append({
                "name": body_name,
                "body": body_id,
                "joint": jid,
                "joint_name": joint_name or f"<unnamed freejoint {jid}>",
                "qpos": qpos_adr,
                "qvel": qvel_adr,
                "last_pos": None,
                "quat": None,
            })

        self.pick_objects = objects

        if not self.pick_objects:
            self.get_logger().warn(
                "No pickable freejoint objects found in scene XML"
            )
        else:
            names = ", ".join(obj["name"] for obj in self.pick_objects)
            self.get_logger().info(
                f"Auto-discovered {len(self.pick_objects)} pickable objects: {names}"
            )

    def _geom_world_obb(self, gid):
        """Return (center, rotation, half_extents) for a MuJoCo geom."""
        center = self.data.geom_xpos[gid].copy()
        R = self.data.geom_xmat[gid].reshape(3, 3).copy()
        gtype = self.model.geom_type[gid]
        size = self.model.geom_size[gid].copy()

        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.array([size[0], size[0], size[0]], dtype=float)
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            half = np.array([size[0], size[0], size[1]], dtype=float)
        elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half = size[:3].astype(float)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = size[:3].astype(float)
        else:
            # Conservative bound for mesh/hfield/etc.
            r = float(self.model.geom_rbound[gid])
            half = np.array([r, r, r], dtype=float)
            R = np.eye(3)

        return center, R, half

    @staticmethod
    def _obb_overlap(c1, R1, e1, c2, R2, e2, eps=1e-7):
        """15-axis SAT overlap test for two oriented bounding boxes."""
        A = np.asarray(R1, dtype=float)
        B = np.asarray(R2, dtype=float)
        a = np.asarray(e1, dtype=float)
        b = np.asarray(e2, dtype=float)

        C = A.T @ B
        absC = np.abs(C) + eps
        t = A.T @ (np.asarray(c2) - np.asarray(c1))

        for i in range(3):
            if abs(t[i]) > a[i] + np.dot(b, absC[i, :]):
                return False

        for j in range(3):
            if abs(np.dot(t, C[:, j])) > b[j] + np.dot(a, absC[:, j]):
                return False

        for i in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            for j in range(3):
                j1, j2 = (j + 1) % 3, (j + 2) % 3
                lhs = abs(t[i2] * C[i1, j] - t[i1] * C[i2, j])
                rhs = (
                    a[i1] * absC[i2, j] + a[i2] * absC[i1, j]
                    + b[j1] * absC[i, j2] + b[j2] * absC[i, j1]
                )
                if lhs > rhs:
                    return False

        return True

    def _object_geom_ids(self, obj):
        """Return geoms belonging to the object's body subtree."""
        root = int(obj["body"])
        ids = []
        for gid in range(self.model.ngeom):
            cur = int(self.model.geom_bodyid[gid])
            while cur > 0:
                if cur == root:
                    ids.append(gid)
                    break
                cur = int(self.model.body_parentid[cur])
        return ids

    def _gripper_grasp_obb(self):
        """3-D oriented grasp region physically between the fingertips."""
        mujoco.mj_forward(self.model, self.data)

        lp = np.mean(self.data.geom_xpos[self.left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[self.right_geoms], axis=0)
        center = 0.5 * (lp + rp)

        x = rp - lp
        span = float(np.linalg.norm(x))
        if span < 1e-9:
            raise RuntimeError("Left/right fingertip centers are coincident")
        x /= span

        hand_R = self.data.xmat[self.hand].reshape(3, 3).copy()
        dots = [abs(float(np.dot(x, hand_R[:, i]))) for i in range(3)]
        helper = hand_R[:, int(np.argmin(dots))]
        z = np.cross(x, helper)
        z /= max(float(np.linalg.norm(z)), 1e-12)
        y = np.cross(z, x)
        y /= max(float(np.linalg.norm(y)), 1e-12)

        R = np.column_stack([x, y, z])

        # 3 mm tolerance beyond fingertip centers; 35 mm grasp depth/height.
        half = np.array([0.5 * span + 0.003, 0.035, 0.035], dtype=float)
        return center, R, half

    @staticmethod
    def _point_to_obb_distance(point, center, rotation, half_extents):
        """
        Euclidean distance from a WORLD point to an oriented bounding box.

        point/center are WORLD coordinates.
        rotation maps OBB-local axes into WORLD.
        half_extents are OBB half sizes in local coordinates.

        Returns 0 when the point lies inside the OBB.
        """
        p = np.asarray(point, dtype=float)
        c = np.asarray(center, dtype=float)
        R = np.asarray(rotation, dtype=float)
        e = np.asarray(half_extents, dtype=float)

        # WORLD -> OBB local coordinates.
        local = R.T @ (p - c)

        # Distance outside each slab. Components inside the box contribute 0.
        outside = np.maximum(np.abs(local) - e, 0.0)
        return float(np.linalg.norm(outside))

    def _nearest_pick_object(self):
        """
        Select an object only when one of its actual MuJoCo geoms overlaps the
        3-D region between the fingertips. Body/freejoint origin is NOT used
        for deciding whether the object is graspable.

        If several objects overlap, select the object whose overlapping geom
        center is nearest the gripper midpoint.
        """
        if not self.pick_objects:
            return None, float("inf")

        mujoco.mj_forward(self.model, self.data)
        gc, gR, ge = self._gripper_grasp_obb()

        candidates = []

        for obj in self.pick_objects:
            overlaps = []

            for gid in self._object_geom_ids(obj):
                oc, oR, oe = self._geom_world_obb(gid)
                if self._obb_overlap(gc, gR, ge, oc, oR, oe):
                    d = float(np.linalg.norm(oc - gc))
                    overlaps.append((d, gid))

            if not overlaps:
                self.get_logger().info(
                    f"Grasp candidate '{obj['name']}': mesh_overlap=False"
                )
                continue

            overlaps.sort(key=lambda item: item[0])
            d, gid = overlaps[0]
            gname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, gid
            )

            self.get_logger().info(
                f"Grasp candidate '{obj['name']}': mesh_overlap=True, "
                f"geom='{gname or gid}', geom_center_distance={d:.4f} m"
            )
            candidates.append((d, obj, gid))

        if not candidates:
            self.get_logger().error(
                "No pickable object geom overlaps the region between the fingertips"
            )
            return None, float("inf")

        candidates.sort(key=lambda item: (item[0], item[1]["name"]))
        distance, best, gid = candidates[0]
        gname = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, gid
        )

        self.get_logger().info(
            f"Mesh-overlap grasp selected '{best['name']}', "
            f"geom='{gname or gid}', distance={distance:.4f} m"
        )
        return best, distance

    def _initialize_state(self):
        for prefix in ["panda1", "panda2"]:
            for i in range(1, 8):
                jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint{i}")
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.q0[i - 1]

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Store the compiled XML pose of every freejoint object.
        for obj in self.pick_objects:
            obj["last_pos"] = self.data.qpos[obj["qpos"]:obj["qpos"] + 3].copy()
            obj["quat"] = self.data.qpos[obj["qpos"] + 3:obj["qpos"] + 7].copy()
            self.data.qvel[obj["qvel"]:obj["qvel"] + 6] = 0.0

    def _finger_center(self):
        """Center of the fingertip collision-pad geoms; used for grasp detection."""
        lp = np.mean(self.data.geom_xpos[self.left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[self.right_geoms], axis=0)
        return (lp + rp) / 2.0

    def _gripper_tcp_center(self, hand_id, left_geoms, right_geoms):
        """
        True gripper END center (TCP) between the two finger tips.

        panda.xml:
          finger body origin: hand-local z = 0.0584 m
          fingertip_pad_collision_1: center z = 0.0445 m, half-size z = 0.0085 m

        Therefore the physical tip plane is:
          z_TCP = 0.0584 + 0.0445 + 0.0085 = 0.1114 m from panda hand origin.

        The old published center was the arithmetic mean of the five fingertip
        box centers:
          z_old = 0.0584 + (0.0445+0.050+0.050+0.0395+0.0395)/5
                = 0.1031 m

        So the true end-center is +0.0083 m along the CURRENT hand-local +Z.
        """
        lp = np.mean(self.data.geom_xpos[left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[right_geoms], axis=0)
        geom_center = 0.5 * (lp + rp)

        hand_R = self.data.xmat[hand_id].reshape(3, 3)
        tcp_offset_from_geom_center_local = np.array([0.0, 0.0, 0.0083], dtype=float)
        return geom_center + hand_R @ tcp_offset_from_geom_center_local

    @staticmethod
    def _mat_to_mj_quat(R):
        """Convert a 3x3 rotation matrix to MuJoCo quaternion [w, x, y, z]."""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        q = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(q, R.reshape(9))
        n = float(np.linalg.norm(q))
        if n <= 1e-12:
            raise RuntimeError("Invalid zero quaternion from rotation matrix")
        return q / n

    def _grasp_hand_spec(self, robot):
        if robot == "panda1":
            return (
                self.p1_hand, self.p1_left_geoms, self.p1_right_geoms
            )
        if robot == "panda2":
            return self.hand, self.left_geoms, self.right_geoms
        raise RuntimeError(f"Unknown grasp robot: {robot!r}")

    def _finger_center_for_robot(self, robot):
        hand_id, left_geoms, right_geoms = self._grasp_hand_spec(robot)
        lp = np.mean(self.data.geom_xpos[left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[right_geoms], axis=0)
        return 0.5 * (lp + rp)

    def _nearest_pick_object_for_robot(self, robot):
        """Select nearest overlapping pickable object for the requested gripper."""
        if not self.pick_objects:
            return None, float("inf")

        mujoco.mj_forward(self.model, self.data)
        hand_id, left_geoms, right_geoms = self._grasp_hand_spec(robot)

        lp = np.mean(self.data.geom_xpos[left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[right_geoms], axis=0)
        gc = 0.5 * (lp + rp)

        x = rp - lp
        span = float(np.linalg.norm(x))
        if span < 1e-9:
            return None, float("inf")
        x /= span
        hand_R = self.data.xmat[hand_id].reshape(3, 3).copy()
        dots = [abs(float(np.dot(x, hand_R[:, i]))) for i in range(3)]
        helper = hand_R[:, int(np.argmin(dots))]
        z = np.cross(x, helper)
        z /= max(float(np.linalg.norm(z)), 1e-12)
        yy = np.cross(z, x)
        yy /= max(float(np.linalg.norm(yy)), 1e-12)
        gR = np.column_stack([x, yy, z])
        ge = np.array([0.5 * span + 0.003, 0.035, 0.035], dtype=float)

        candidates = []
        for obj in self.pick_objects:
            for gid in self._object_geom_ids(obj):
                oc, oR, oe = self._geom_world_obb(gid)
                if self._obb_overlap(gc, gR, ge, oc, oR, oe):
                    candidates.append(
                        (float(np.linalg.norm(oc - gc)), obj)
                    )
                    break
        if not candidates:
            return None, float("inf")
        candidates.sort(key=lambda item: (item[0], item[1]["name"]))
        return candidates[0][1], candidates[0][0]

    def _sync_legacy_grasp_state(self):
        """Keep legacy single-grasp fields coherent for older integrations."""
        selected = next(((arm, grasp) for arm, grasp in self.arm_grasps.items()
                         if grasp is not None and grasp["object"]["name"] == "screwdriver"), None)
        if selected is None:
            selected = next(((arm, grasp) for arm, grasp in self.arm_grasps.items()
                             if grasp is not None), None)
        self.grasped = selected is not None
        self.grasp_robot = selected[0] if selected else None
        self.grasped_object = selected[1]["object"] if selected else None
        if selected:
            grasp = selected[1]
            self.grasp_rel_pos_hand = grasp["rel_pos"].copy()
            self.grasp_rel_R_hand = grasp["rel_R"].copy()
            self.grasp_offset = grasp["offset"].copy()
        else:
            self.grasp_rel_pos_hand = np.zeros(3, dtype=float)
            self.grasp_rel_R_hand = np.eye(3, dtype=float)
            self.grasp_offset = np.zeros(3, dtype=float)

    def _object_owner(self, obj):
        for arm, grasp in self.arm_grasps.items():
            if grasp is not None and grasp["object"] is obj:
                return arm
        return None

    def _enable_part_32_33_attach_cb(self, request, response):
        with self.lock:
            self.part_32_33_auto_attach_enabled = True
            self.part_32_33_contact_streak = 0
            self.part_32_33_last_report = 0.0
            self.get_logger().info("Part 3-2/3-3 auto attach ENABLED; geometry-distance monitoring started")
            response.success = True
            response.message = ("part_3_2/part_3_3 already attached" if self.part_32_33_attachment
                                else "part_3_2/part_3_3 contact attachment enabled")
        return response

    def _part_32_33_surface_distance(self):
        """Minimum signed surface distance, even with contact masks disabled.

        mj_geomDistance performs a direct collision-distance query independent of
        model.geom_contype / geom_conaffinity.  It does not change the dynamics.
        """
        nearest = float("inf")
        pair = None
        for g1 in self.part_32_geom_ids:
            for g2 in self.part_33_geom_ids:
                # Skip distant pairs cheaply; a bounding sphere is only a
                # broad-phase rejection, NEVER sufficient to claim contact.
                center_distance = float(np.linalg.norm(
                    self.data.geom_xpos[g1] - self.data.geom_xpos[g2]))
                bound = (float(self.model.geom_rbound[g1])
                         + float(self.model.geom_rbound[g2]))
                if center_distance - bound > self.part_32_33_attach_gap:
                    continue
                fromto = np.zeros(6, dtype=float)
                distance = float(mujoco.mj_geomDistance(
                    self.model, self.data, int(g1), int(g2),
                    float(self.part_32_33_attach_gap + 0.01), fromto))
                if distance < nearest:
                    nearest, pair = distance, (int(g1), int(g2))
        return nearest, pair

    def _auto_part_32_33_attach(self):
        if not self.part_32_33_auto_attach_enabled or self.part_32_33_attachment:
            return
        # Allow attachment when either part is held. If BOTH are held by
        # independent arms, defer until one releases to avoid conflicting
        # kinematic drivers (two arms cannot both dictate a rigid assembly).
        owner32 = self._object_owner(self.part_32)
        owner33 = self._object_owner(self.part_33)
        if owner32 and owner33:
            self.part_32_33_contact_streak = 0
            now = time.monotonic()
            if now - self.part_32_33_last_report >= 2.0:
                self.part_32_33_last_report = now
                self.get_logger().info(
                    f"Part 3-2/3-3 attach waiting: both held "
                    f"(part_3_2={owner32}, part_3_3={owner33}); "
                    f"release either gripper first")
            return

        self.part_32_33_check_count += 1
        # Avoid doing mesh distance queries at every simulation step.
        if self.part_32_33_check_count % 5:
            return
        try:
            gap, pair = self._part_32_33_surface_distance()
        except Exception as exc:
            now = time.monotonic()
            if now - self.part_32_33_last_report >= 2.0:
                self.part_32_33_last_report = now
                self.get_logger().error(f"Part 3-2/3-3 geometry distance query failed: {exc}")
            return

        touching = pair is not None and gap <= self.part_32_33_attach_gap
        self.part_32_33_contact_streak = (
            self.part_32_33_contact_streak + 1 if touching else 0)
        now = time.monotonic()
        if now - self.part_32_33_last_report >= 2.0:
            self.part_32_33_last_report = now
            gap_label = f"{gap:.5f} m" if np.isfinite(gap) else "no close geometry pairs"
            self.get_logger().info(
                f"Part 3-2/3-3 attach watch: min_surface_gap={gap_label}, "
                f"threshold={self.part_32_33_attach_gap:.4f} m, "
                f"consecutive={self.part_32_33_contact_streak}/"
                f"{self.part_32_33_contact_steps}")
        if self.part_32_33_contact_streak < self.part_32_33_contact_steps:
            return

        # The currently held part must be the parent. This ensures that
        # subsequent arm movement drives the whole assembly, not vice versa.
        parent = self.part_32 if owner32 else self.part_33
        child = self.part_33 if parent is self.part_32 else self.part_32
        parent_pos = self.data.xpos[parent["body"]].copy()
        parent_R = self.data.xmat[parent["body"]].reshape(3, 3).copy()
        child_pos = self.data.xpos[child["body"]].copy()
        child_R = self.data.xmat[child["body"]].reshape(3, 3).copy()
        self.part_32_33_attachment = {
            "parent": parent["name"], "child": child["name"],
            "rel_pos": parent_R.T @ (child_pos - parent_pos),
            "rel_R": parent_R.T @ child_R,
        }
        self.get_logger().info(
            f"AUTO ATTACH: {child['name']} rigidly attached to {parent['name']} "
            f"(surface_gap={gap:.6f} m, geom_pair={pair}, "
            f"held_by={owner32 or owner33 or 'none'})")

    def _push_direction_cb(self, msg):
        """Data: [arm_number(1/2), world_dx, world_dy, world_dz]."""
        vals = list(msg.data)
        if len(vals) != 4 or not all(np.isfinite(v) for v in vals):
            self.get_logger().warn("Invalid /mujoco/push_direction payload")
            return
        arm = {1: "panda1", 2: "panda2"}.get(int(vals[0]))
        vec = np.asarray(vals[1:], dtype=float)
        norm = float(np.linalg.norm(vec))
        if arm is None or norm < 1e-8:
            self.get_logger().warn("Invalid push arm or zero direction")
            return
        with self.lock:
            self.push_direction[arm] = vec / norm

    def _push_pad_front_projection(self, pad, direction):
        """World-axis support coordinate of an actual fingertip BOX geom."""
        center = self.data.geom_xpos[pad]
        R = self.data.geom_xmat[pad].reshape(3, 3)
        half = self.model.geom_size[pad, :3]
        return float(np.dot(center, direction) + np.sum(np.abs(R.T @ direction) * half))

    def _push_status_cb(self, arm, request, response):
        """Return first-contact TCP position in world frame, if latched."""
        with self.lock:
            contact = self.push_first_contact_tcp[arm]
            response.success = True
            if contact is None or self.push_object[arm] is None:
                response.message = "waiting"
            else:
                response.message = json.dumps({
                    "object": self.push_object[arm]["name"],
                    "contact_tcp": [float(x) for x in contact],
                })
        return response

    def _push_contact_cb(self, arm, request, response):
        with self.lock:
            if request.data:
                if self.push_direction[arm] is None:
                    response.success = False
                    response.message = f"{arm}: no push direction received; update push.py"
                    return response
                if self.arm_grasps[arm] is not None:
                    response.success = False
                    response.message = f"{arm} already owns an object; release it before push"
                    return response
                self.push_active[arm] = True
                self.push_object[arm] = None
                self.push_search_interrupted[arm] = False
                self.push_first_contact_tcp[arm] = None
                self.push_contact_cycle = 0
                self.push_previous_sample[arm] = self._push_sample(arm)
                response.success = True
                response.message = f"{arm} push contact monitoring enabled"
            else:
                self.push_active[arm] = False
                obj = self.push_object[arm]
                if obj is not None and self._object_owner(obj) == arm:
                    self._release_current_object(arm)
                self.push_object[arm] = None
                self.push_search_interrupted[arm] = False
                self.push_first_contact_tcp[arm] = None
                self.push_previous_sample[arm] = None
                response.success = True
                response.message = f"{arm} push monitoring disabled; pushed object released"
            self.get_logger().info(response.message)
        return response

    def _push_sample(self, arm):
        """Record fingertip geometry and hand pose at a precise-query boundary."""
        hand, left, right = self._grasp_hand_spec(arm)
        pads = [int(g) for g in list(left) + list(right)]
        return {
            "hand_pos": self.data.xpos[hand].copy(),
            "pads": {g: self.data.geom_xpos[g].copy() for g in pads},
        }

    def _push_pair_gap_at_fraction(self, pad, obj_geom, previous_center,
                                   current_center, fraction, direction, front):
        """Query signed geometry gap at an interpolated fingertip location.

        The object is stationary before the first latch. We temporarily move
        only the fingertip geom center, never the actual robot joints or body.
        The geom rotation is unchanged during a straight push.
        """
        saved = self.data.geom_xpos[pad].copy()
        try:
            self.data.geom_xpos[pad] = (previous_center +
                                        fraction * (current_center - previous_center))
            fromto = np.zeros(6, dtype=float)
            distance = float(mujoco.mj_geomDistance(
                self.model, self.data, int(pad), int(obj_geom), 0.01, fromto))
            # Keep the original front-face requirement at the interpolated pose.
            shift = self.data.geom_xpos[pad] - current_center
            interpolated_front = front + float(np.dot(shift, direction))
            valid_front = (float(np.dot(fromto[:3], direction)) >=
                           interpolated_front - self.push_front_tolerance and
                           float(np.dot(fromto[3:], direction)) >=
                           interpolated_front - self.push_front_tolerance)
            return distance, valid_front
        finally:
            self.data.geom_xpos[pad] = saved

    def _auto_push_contact(self):
        """Every 5 cycles, find first contact by swept-pose binary search."""
        if not any(self.push_active[a] and self.push_object[a] is None
                   for a in ("panda1", "panda2")):
            return
        self.push_contact_cycle += 1
        if self.push_contact_cycle % self.push_contact_check_interval:
            return
        for arm in ("panda1", "panda2"):
            if not self.push_active[arm] or self.push_object[arm] is not None:
                continue
            if self.arm_grasps[arm] is not None:
                continue
            hand_id, left, right = self._grasp_hand_spec(arm)
            pads = [int(g) for g in list(left) + list(right)]
            direction = self.push_direction[arm]
            current = self._push_sample(arm)
            previous = self.push_previous_sample[arm]
            self.push_previous_sample[arm] = current
            if previous is None:
                continue
            displacement = current["hand_pos"] - previous["hand_pos"]
            advance = float(np.dot(displacement, direction))
            if advance < -1e-6:
                continue  # never attach during a reverse movement
            front = max(self._push_pad_front_projection(p, direction) for p in pads)
            best = None
            for obj in self.pick_objects:
                if self._object_owner(obj) is not None:
                    continue
                if obj["name"] in self.hex_tip_attachments:
                    continue
                if (self.part_32_33_attachment is not None and
                        obj["name"] == self.part_32_33_attachment["child"]):
                    continue
                for g in self._object_geom_ids(obj):
                    g = int(g)
                    for pad in pads:
                        now_center = current["pads"][pad]
                        old_center = previous["pads"][pad]
                        sphere_gap = (float(np.linalg.norm(now_center - self.data.geom_xpos[g]))
                                      - float(self.model.geom_rbound[pad])
                                      - float(self.model.geom_rbound[g]))
                        if sphere_gap > self.push_contact_gap:
                            continue
                        try:
                            end_gap, end_front = self._push_pair_gap_at_fraction(
                                pad, g, old_center, now_center, 1.0, direction, front)
                            if end_gap > self.push_contact_gap or not end_front:
                                continue
                            start_gap, start_front = self._push_pair_gap_at_fraction(
                                pad, g, old_center, now_center, 0.0, direction, front)
                            if start_gap <= self.push_contact_gap and start_front:
                                first_fraction = 0.0
                            else:
                                lo, hi = 0.0, 1.0
                                # At most 14 extra distance queries for the
                                # actual contact candidate, only once per push.
                                for _ in range(14):
                                    mid = (lo + hi) * 0.5
                                    gap, is_front = self._push_pair_gap_at_fraction(
                                        pad, g, old_center, now_center, mid, direction, front)
                                    if gap <= self.push_contact_gap and is_front:
                                        hi = mid
                                    else:
                                        lo = mid
                                first_fraction = hi
                        except Exception as exc:
                            self.get_logger().error(f"Push swept contact query failed: {exc}")
                            continue
                        # Earliest contact along the motion, not deepest penetration.
                        key = (first_fraction, obj["name"], pad, g)
                        if best is None or key < best[0]:
                            best = (key, obj, end_gap)
            if best is None:
                continue
            (fraction, _, pad, g), obj, end_gap = best
            # Compensate for the distance traveled AFTER first contact but
            # BEFORE the current fifth-cycle detection. Without this correction
            # the object would move too little and finish at a variable pose.
            contact_hand_pos = (previous["hand_pos"] + fraction * displacement)
            overshoot = max(0.0, float(np.dot(
                current["hand_pos"] - contact_hand_pos, direction)))
            correction = overshoot * direction
            qpos = int(obj["qpos"])
            self.data.qpos[qpos:qpos + 3] += correction
            obj["last_pos"] = self.data.qpos[qpos:qpos + 3].copy()
            mujoco.mj_forward(self.model, self.data)
            hand_pos = self.data.xpos[hand_id].copy()
            hand_R = self.data.xmat[hand_id].reshape(3, 3).copy()
            obj_pos = self.data.xpos[obj["body"]].copy()
            obj_R = self.data.xmat[obj["body"]].reshape(3, 3).copy()
            self.arm_grasps[arm] = {
                "object": obj,
                "rel_pos": hand_R.T @ (obj_pos - hand_pos),
                "rel_R": hand_R.T @ obj_R,
                "offset": obj_pos - self._finger_center_for_robot(arm),
            }
            self._sync_legacy_grasp_state()
            self.push_object[arm] = obj
            # Recover the true TCP position at first contact, not at the
            # delayed fifth-cycle detection. The TCP offset rotates with hand.
            tcp_now = self._gripper_tcp_center(hand_id, left, right)
            self.push_first_contact_tcp[arm] = (
                tcp_now - (current["hand_pos"] - contact_hand_pos)
            ).copy()
            self.get_logger().info(
                f"PUSH SWEPT CONTACT: {arm} -> {obj['name']}, "
                f"first_fraction={fraction:.5f}, compensation={overshoot*1000:.3f} mm, "
                f"end_surface_gap={end_gap:.6f} m, geom_pair=({pad}, {g})")

    def _update_pick_objects(self):
        """Update each independently held object from its own hand pose."""
        mujoco.mj_forward(self.model, self.data)
        held = {}
        for arm, grasp in self.arm_grasps.items():
            if grasp is None:
                continue
            hand_id, _, _ = self._grasp_hand_spec(arm)
            hand_pos = self.data.xpos[hand_id].copy()
            hand_R = self.data.xmat[hand_id].reshape(3, 3).copy()
            held[id(grasp["object"])] = (hand_pos + hand_R @ grasp["rel_pos"],
                                         hand_R @ grasp["rel_R"])
        for obj in self.pick_objects:
            if (self.part_32_33_attachment is not None and
                    obj["name"] == self.part_32_33_attachment["child"]):
                continue  # child driven by its rigidly attached parent below
            if obj["name"] in self.hex_tip_attachments:
                continue  # driven below by the hex_key pose
            qpos, qvel = obj["qpos"], obj["qvel"]
            if id(obj) in held:
                pos, obj_R = held[id(obj)]
                quat = self._mat_to_mj_quat(obj_R)
                self.data.qpos[qpos:qpos + 3] = pos
                self.data.qpos[qpos + 3:qpos + 7] = quat
                obj["last_pos"] = pos.copy()
                obj["quat"] = quat.copy()
            else:
                if obj["last_pos"] is not None:
                    self.data.qpos[qpos:qpos + 3] = obj["last_pos"]
                if obj["quat"] is not None:
                    self.data.qpos[qpos + 3:qpos + 7] = obj["quat"]
            self.data.qvel[qvel:qvel + 6] = 0.0

        if self.part_32_33_attachment is not None:
            mujoco.mj_forward(self.model, self.data)
            rel = self.part_32_33_attachment
            parent = self.part_32 if rel["parent"] == "part_3_2" else self.part_33
            child = self.part_33 if rel["child"] == "part_3_3" else self.part_32
            parent_pos = self.data.xpos[parent["body"]].copy()
            parent_R = self.data.xmat[parent["body"]].reshape(3, 3).copy()
            qpos, qvel = child["qpos"], child["qvel"]
            pos = parent_pos + parent_R @ rel["rel_pos"]
            quat = self._mat_to_mj_quat(parent_R @ rel["rel_R"])
            self.data.qpos[qpos:qpos + 3] = pos
            self.data.qpos[qpos + 3:qpos + 7] = quat
            self.data.qvel[qvel:qvel + 6] = 0.0
            child["last_pos"] = pos.copy()
            child["quat"] = quat.copy()

        # The hex key may itself be held by either Panda arm. Refresh its
        # world transform after updating the freejoint, then drive attached
        # objects with their original rigid offsets (also after release).
        if self.hex_tip_attachments:
            mujoco.mj_forward(self.model, self.data)
            parent_pos = self.data.xpos[self.hex_key_body].copy()
            parent_R = self.data.xmat[self.hex_key_body].reshape(3, 3).copy()
            for attachment in self.hex_tip_attachments.values():
                obj = attachment["object"]
                qpos, qvel = obj["qpos"], obj["qvel"]
                pos = parent_pos + parent_R @ attachment["rel_pos"]
                quat = self._mat_to_mj_quat(parent_R @ attachment["rel_R"])
                self.data.qpos[qpos:qpos + 3] = pos
                self.data.qpos[qpos + 3:qpos + 7] = quat
                self.data.qvel[qvel:qvel + 6] = 0.0
                obj["last_pos"] = pos.copy()
                obj["quat"] = quat.copy()

    def _screw_config_cb(self, msg):
        """
        Message format:
          [dx, dy, dz, shaft_speed, down_distance, tip_x, tip_y, tip_z]

        tip_x/y/z is the YAML contact-tip WORLD position and is authoritative
        for choosing the synchronized screw object.
        """
        if len(msg.data) < 8:
            self.get_logger().error(
                "/mujoco/screw_config requires "
                "[dx,dy,dz,shaft_speed,down_distance,tip_x,tip_y,tip_z]"
            )
            return

        d = np.asarray(msg.data[:3], dtype=float)
        n = float(np.linalg.norm(d))
        if n <= 1e-12:
            self.get_logger().error("Screw direction cannot be zero")
            return

        target_tip = np.asarray(msg.data[5:8], dtype=float)
        if not np.all(np.isfinite(target_tip)):
            self.get_logger().error("Screw target tip contains non-finite values")
            return

        with self.lock:
            self.screw_direction = d / n
            self.screw_shaft_speed = float(msg.data[3])
            self.screw_down_distance = abs(float(msg.data[4]))
            self.screw_target_tip = target_tip.copy()

        self.get_logger().info(
            "Screw config: "
            f"direction={self.screw_direction.tolist()}, "
            f"shaft_speed={self.screw_shaft_speed:.3f}, "
            f"down_distance={self.screw_down_distance:.4f}, "
            f"YAML target_tip={target_tip.tolist()}"
        )

    def _hex_tip_nearest_candidate(self):
        """Select nearest movable object by current geom position, screw-style.

        Use each geom's conservative bounding radius as a broad proximity
        estimate; unlike data.contact this also works without collision pairs.
        All candidates are cached at startup, so no scene-tree scan per step.
        """
        tip = np.asarray(self.data.site_xpos[self.hex_key_tip_site], dtype=float)
        best = None
        for gid, obj in self.hex_candidate_by_geom.items():
            if self._object_owner(obj) is not None:
                continue
            center = self.data.geom_xpos[gid]
            center_distance = float(np.linalg.norm(center - tip))
            # geom_rbound is MuJoCo's bounding-sphere radius.  It is a
            # conservative surface estimate for non-spherical geometry.
            bound = float(self.model.geom_rbound[gid])
            surface_gap = max(0.0, center_distance - bound)
            if surface_gap > self.hex_tip_contact_radius:
                continue
            key = (surface_gap, center_distance, obj["name"])
            if best is None or key < best[0]:
                best = (key, obj, gid)
        return best

    def _attach_hex_target(self, target, distance):
        """Store the first attachment only; called under the simulation lock."""
        if self.hex_tip_attachments or self.hex_tip_ever_attached:
            return False
        parent_R = self.data.xmat[self.hex_key_body].reshape(3, 3).copy()
        parent_pos = self.data.xpos[self.hex_key_body].copy()
        child_R = self.data.xmat[int(target["body"])].reshape(3, 3).copy()
        child_pos = self.data.xpos[int(target["body"])].copy()
        self.hex_tip_ever_attached = True
        self.hex_tip_attachments[target["name"]] = {
            "object": target,
            "rel_pos": parent_R.T @ (child_pos - parent_pos),
            "rel_R": parent_R.T @ child_R,
        }
        self.get_logger().info(
            f"HEX TIP attached '{target['name']}' to hex_key "
            f"(estimated geom surface gap={distance:.5f} m); single target locked"
        )
        return True

    def _auto_hex_tip_attach(self):
        """Low-cost periodic geometry proximity check, independent of contacts."""
        if self.hex_tip_attachments or self.hex_tip_ever_attached:
            return
        holding_hex = any(
            grasp is not None and grasp["object"] is self.hex_key_object
            for grasp in self.arm_grasps.values()
        )
        if not holding_hex:
            self.hex_tip_contact_streak = 0
            self.hex_contact_check_counter = 0
            self.hex_tip_candidate_name = None
            return
        self.hex_contact_check_counter += 1
        if self.hex_contact_check_counter < self.hex_contact_check_interval:
            return
        self.hex_contact_check_counter = 0
        match = self._hex_tip_nearest_candidate()
        name = match[1]["name"] if match is not None else None
        if name is not None and name == self.hex_tip_candidate_name:
            self.hex_tip_contact_streak += 1
        else:
            self.hex_tip_candidate_name = name
            self.hex_tip_contact_streak = 1 if name is not None else 0
        if match is not None and self.hex_tip_contact_streak >= self.hex_tip_auto_contact_steps:
            self._attach_hex_target(match[1], match[0][0])
            self.hex_tip_contact_streak = 0

    def _hex_tip_attach_cb(self, request, response):
        """Optional legacy service: same geometry-based, single-target rule."""
        with self.lock:
            if self.hex_tip_attachments or self.hex_tip_ever_attached:
                response.success = False
                response.message = "hex_key attachment already used; second attachment prohibited"
                return response
            mujoco.mj_forward(self.model, self.data)
            match = self._hex_tip_nearest_candidate()
            if match is None:
                response.success = False
                response.message = (
                    "No unheld freejoint object near hex_key_tip "
                    f"(estimated surface gap <= {self.hex_tip_contact_radius:.3f} m)"
                )
                return response
            response.success = self._attach_hex_target(match[1], match[0][0])
            response.message = (f"Attached '{match[1]['name']}'" if response.success
                                else "Already attached")
            return response

    def _hex_key_attached_axis_cb(self, request, response):
        """Return the attached object's current WORLD center-of-mass XY.

        A WORLD-Z-parallel rotation axis passes through this point.
        This read-only service does not alter the attachment state.
        """
        with self.lock:
            if not self.hex_tip_attachments:
                response.success = False
                response.message = "No object attached to hex_key_tip"
                return response
            attachment = next(iter(self.hex_tip_attachments.values()))
            obj = attachment["object"]
            body = int(obj["body"])
            # The simulation loop maintains xipos; no expensive forward here.
            center = self.data.xipos[body]
            response.success = True
            response.message = json.dumps({
                "body": obj["name"],
                "axis_x": float(center[0]),
                "axis_y": float(center[1]),
                "axis_z": float(center[2]),
            })
            return response

    def _hex_tip_detach_cb(self, request, response):
        """Release attached part at its current WORLD pose; keep one-shot latch."""
        with self.lock:
            if not self.hex_tip_attachments:
                response.success = False
                response.message = "No object attached to hex_key_tip"
                return response
            # _update_pick_objects stores last_pos/quat every simulation step;
            # capture the exact current world pose before releasing the rigid link.
            mujoco.mj_forward(self.model, self.data)
            names = []
            for name, attachment in self.hex_tip_attachments.items():
                obj = attachment["object"]
                body = int(obj["body"])
                obj["last_pos"] = self.data.xpos[body].copy()
                obj["quat"] = self._mat_to_mj_quat(
                    self.data.xmat[body].reshape(3, 3).copy())
                names.append(name)
            self.hex_tip_attachments.clear()
            self.hex_tip_ever_attached = True
            self.hex_tip_contact_streak = 0
            self.hex_tip_candidate_name = None
            response.success = True
            response.message = "Detached " + ", ".join(names) + "; auto reattach disabled"
            self.get_logger().info(response.message)
            return response

    def _screw_start_cb(self, request, response):
        with self.lock:
            if (
                not self.grasped
                or self.grasped_object is None
                or self.grasped_object["name"] != "screwdriver"
                or self.grasp_robot not in ("panda1", "panda2")
            ):
                response.success = False
                response.message = "Screwdriver is not currently grasped"
                return response

            if self.screw_target_tip is None:
                response.success = False
                response.message = (
                    "No YAML screw target tip configured. "
                    "Publish /mujoco/screw_config before /mujoco/screw_start."
                )
                return response

            # IMPORTANT: establish the inactive gripper hold from its COMMAND,
            # not from the instantaneous MuJoCo qpos.  Sampling qpos here can
            # capture a one-step actuator/equality transient and make that single
            # bad frame visible exactly when screw rotation begins.
            if self.grasp_robot == "panda2":
                self.screw_hold_robot = "panda1"
                hold_width = self._p1_ctrl_to_total_width(
                    self.p1_gripper_command
                )
                hold_each = 0.5 * hold_width
                self.screw_hold_finger_qpos = np.array(
                    [hold_each, hold_each], dtype=float
                )
                self.screw_hold_finger_ctrl = float(self.p1_gripper_command)
                self.data.qpos[self.p1_finger_qadr] = self.screw_hold_finger_qpos
                self.data.qvel[self.p1_finger_vadr] = 0.0
                self.data.ctrl[self.p1_finger_act] = self.screw_hold_finger_ctrl
            else:
                self.screw_hold_robot = "panda2"
                hold_width = self._actuator_ctrl_to_total_width(
                    self.gripper_command
                )
                hold_each = 0.5 * hold_width
                self.screw_hold_finger_qpos = np.array(
                    [hold_each, hold_each], dtype=float
                )
                self.screw_hold_finger_ctrl = float(self.gripper_command)
                self.data.qpos[self.finger_qadr] = self.screw_hold_finger_qpos
                self.data.qvel[self.finger_vadr] = 0.0
                self.data.ctrl[self.finger_act] = self.screw_hold_finger_ctrl

            # Freeze the ACTIVE gripper at its real grasped width too.  Otherwise
            # screwdriver motor dynamics can close and reopen Panda2 fingers.
            self.screw_owner_hold_robot = self.grasp_robot
            if self.grasp_robot == "panda2":
                self.screw_owner_hold_qpos = self.data.qpos[self.finger_qadr].copy()
                self.screw_owner_hold_ctrl = float(self.gripper_command)
            else:
                self.screw_owner_hold_qpos = self.data.qpos[self.p1_finger_qadr].copy()
                self.screw_owner_hold_ctrl = float(self.p1_gripper_command)

            # Forward kinematics is evaluated only after the inactive gripper has
            # been placed at the deterministic command-consistent hold pose.
            mujoco.mj_forward(self.model, self.data)

            candidates = [
                obj for obj in self.pick_objects
                if obj["name"] != "screwdriver"
            ]
            if not candidates:
                response.success = False
                response.message = "No freejoint screw candidates exist"
                return response

            target_tip = self.screw_target_tip.copy()

            # IMPORTANT:
            # Object selection is based ONLY on YAML screw_i_tip_x/y/z versus
            # each object's CURRENT MuJoCo body position. The actual screwdriver
            # tip does not participate in this selection.
            candidate_distances = []
            for obj in candidates:
                # Select by CURRENT geom CENTER in WORLD coordinates.
                #
                # YAML target_tip is only a spatial reference for identifying
                # the intended object. It does NOT need to equal the actual
                # screwdriver_tip position at screw-start time.
                #
                # If an object has multiple geoms, use the geom center closest
                # to the YAML target as that object's selection distance.
                geom_results = []

                for gid in self._object_geom_ids(obj):
                    geom_center_world = self.data.geom_xpos[int(gid)].copy()
                    geom_distance = float(
                        np.linalg.norm(geom_center_world - target_tip)
                    )
                    geom_results.append(
                        (geom_distance, int(gid), geom_center_world)
                    )

                if not geom_results:
                    self.get_logger().warn(
                        f"Screw target candidate '{obj['name']}': "
                        "no geometry found in object subtree; skipped"
                    )
                    continue

                geom_results.sort(key=lambda item: item[0])
                distance, nearest_gid, nearest_geom_center = geom_results[0]

                geom_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, nearest_gid
                )

                candidate_distances.append(
                    (
                        distance,
                        obj,
                        nearest_gid,
                        nearest_geom_center,
                    )
                )

                self.get_logger().info(
                    f"Screw target candidate '{obj['name']}': "
                    f"nearest_geom='{geom_name or nearest_gid}', "
                    f"geom_center_world={nearest_geom_center.tolist()}, "
                    f"YAML_target_tip={target_tip.tolist()}, "
                    f"geom_center_distance={distance:.6f} m"
                )

            if not candidate_distances:
                response.success = False
                response.message = (
                    "No screw candidate has geometry for YAML-target selection"
                )
                return response

            candidate_distances.sort(
                key=lambda item: (item[0], item[1]["name"])
            )
            (
                distance,
                selected,
                selected_gid,
                selected_geom_center,
            ) = candidate_distances[0]

            selected_geom_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, selected_gid
            )

            self.screw_selected_object = selected

            # Selection uses YAML target_tip, but synchronized DOWN motion still
            # follows the ACTUAL screwdriver tip displacement after screw_start.
            actual_tip = self.data.site_xpos[
                self.screwdriver_tip_site
            ].copy()
            self.screw_start_tip = actual_tip.copy()

            qpos = int(selected["qpos"])
            self.screw_start_qpos = self.data.qpos[qpos:qpos + 7].copy()

            # Inactive-gripper hold was established before mj_forward above.
            # A new screw operation owns the hold immediately; cancel any
            # delayed release left from the previous screw.
            self.screw_hold_release_steps = 0
            self.screw_active = True
            self.data.ctrl[self.screwdriver_motor] = self.screw_shaft_speed

            self.get_logger().info(
                f"Screw SELECTED from YAML target using CURRENT GEOM CENTER: "
                f"'{selected['name']}', "
                f"nearest_geom='{selected_geom_name or selected_gid}', "
                f"geom_center_world={selected_geom_center.tolist()}, "
                f"geom_center_distance={distance:.6f} m, "
                f"actual_screwdriver_tip={actual_tip.tolist()}"
            )

            response.success = True
            response.message = (
                f"Screw started: YAML-nearest object='{selected['name']}', "
                f"geom-center distance={distance:.4f} m, "
                f"shaft_speed={self.screw_shaft_speed:.3f}, "
                f"down_distance={self.screw_down_distance:.4f}"
            )
            return response

    def _screw_stop_cb(self, request, response):
        with self.lock:
            name = (
                self.screw_selected_object["name"]
                if self.screw_selected_object is not None else None
            )
            if self.screw_selected_object is not None:
                obj = self.screw_selected_object
                qpos = int(obj["qpos"])
                obj["last_pos"] = self.data.qpos[qpos:qpos + 3].copy()
                obj["quat"] = self.data.qpos[qpos + 3:qpos + 7].copy()

            self.screw_active = False
            self.data.ctrl[self.screwdriver_motor] = 0.0
            self.screw_selected_object = None
            self.screw_start_tip = None
            self.screw_start_qpos = None

            # IMPORTANT: do not clear screw_hold_* here.  Releasing the
            # inactive gripper in the same frame as the motor stop creates the
            # stop-side counterpart of the one-frame close/open transient.
            # The simulation loop will keep qpos/qvel/ctrl frozen for several
            # full mj_step() calls and then release it cleanly.
            if (
                self.screw_hold_robot is not None
                and self.screw_hold_finger_qpos is not None
                and self.screw_hold_finger_ctrl is not None
            ):
                self.screw_hold_release_steps = self.screw_stop_hold_steps
            else:
                self.screw_hold_release_steps = 0

        response.success = True
        response.message = (
            "Screw stopped" if name is None else f"Screw stopped: '{name}'"
        )
        return response

    def _update_screw_motion(self):
        """
        Synchronize the selected screw with actual screwdriver-tip progress.

        The selected screw is chosen from the object nearest the configured YAML
        target tip when /mujoco/screw_start is called. Its translation follows the projection
        of the ACTUAL screwdriver tip displacement onto screw_direction, capped
        at DOWN_DISTANCE.  Thus it remains synchronized with MoveIt motion
        rather than with an independent timer.
        """
        if (
            not self.screw_active
            or self.screw_selected_object is None
            or self.screw_start_tip is None
            or self.screw_start_qpos is None
        ):
            return

        mujoco.mj_forward(self.model, self.data)
        tip = self.data.site_xpos[self.screwdriver_tip_site].copy()
        progress = float(
            np.dot(tip - self.screw_start_tip, self.screw_direction)
        )
        progress = float(np.clip(
            progress, 0.0, self.screw_down_distance
        ))

        obj = self.screw_selected_object
        qpos = int(obj["qpos"])
        qvel = int(obj["qvel"])

        pose = self.screw_start_qpos.copy()
        pose[:3] += self.screw_direction * progress
        self.data.qpos[qpos:qpos + 7] = pose
        self.data.qvel[qvel:qvel + 6] = 0.0
        obj["last_pos"] = pose[:3].copy()
        obj["quat"] = pose[3:7].copy()

        self.data.ctrl[self.screwdriver_motor] = self.screw_shaft_speed

    def _direct_joint_status(self, seq, state):
        msg = Float64MultiArray()
        msg.data = [float(seq), float(state)]
        self.direct_joint_status_pub.publish(msg)

    def _direct_joint_relative_cb(self, msg):
        values = list(msg.data)
        if len(values) != 5 or not all(np.isfinite(v) for v in values):
            self.get_logger().error("Invalid direct joint command: expected 5 finite numbers")
            return
        seq, arm_raw, index_raw, delta_deg, duration = values
        if seq != int(seq) or arm_raw not in (1.0, 2.0) or index_raw not in range(1, 8) or duration <= 0.0:
            self._direct_joint_status(seq, -1)
            return
        arm = "panda1" if arm_raw == 1.0 else "panda2"
        with self.lock:
            if seq == self.direct_joint_last_seq:
                # Ignore retransmissions; do not apply the relative delta twice.
                return
            if (self.direct_joint_motion is not None or self.trajectory_active
                    or self.p1_trajectory_active):
                self._direct_joint_status(seq, -1)
                return
            qadr = self.p1_arm_qadr if arm == "panda1" else self.arm_qadr
            current = self.data.qpos[qadr].copy()
            target = current.copy()
            joint = int(index_raw) - 1
            target[joint] += np.deg2rad(delta_deg)
            jid = (self.p1_arm_joints if arm == "panda1" else self.arm_joints)[joint]
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                if target[joint] < lo or target[joint] > hi:
                    self.get_logger().error(
                        f"Direct joint target outside limits: {arm} joint {joint+1}: "
                        f"{target[joint]:.4f} rad not in [{lo:.4f}, {hi:.4f}]")
                    self._direct_joint_status(seq, -1)
                    return
            # Snapshot all seven positions at execution start.
            # Other six entries stay byte-for-byte unchanged throughout motion.
            command = self.p1_q_command if arm == "panda1" else self.q_command
            velocity = self.p1_v_command if arm == "panda1" else self.v_command
            command[:] = current
            velocity[:] = 0.0
            self.direct_joint_last_seq = int(seq)
            self.direct_joint_motion = dict(
                seq=int(seq), arm=arm, joint=joint, start=current,
                target=target, duration=float(duration), started=time.monotonic()
            )
        self._direct_joint_status(seq, 0)
        self.get_logger().info(
            f"DIRECT JOINT: {arm} joint {joint+1}, delta={delta_deg:+.3f} deg, "
            f"duration={duration:.3f}s; other six joints held fixed")

    def _step_direct_joint_motion(self):
        motion = self.direct_joint_motion
        if motion is None:
            return
        t = min(max((time.monotonic() - motion["started"]) / motion["duration"], 0.0), 1.0)
        smooth = t*t*(3.0 - 2.0*t)
        cmd = motion["start"].copy()
        j = motion["joint"]
        cmd[j] = motion["start"][j] + smooth * (motion["target"][j] - motion["start"][j])
        if motion["arm"] == "panda1":
            self.p1_q_command[:] = cmd
            self.p1_v_command[:] = 0.0
        else:
            self.q_command[:] = cmd
            self.v_command[:] = 0.0
        if t >= 1.0:
            self.direct_joint_motion = None
            self._direct_joint_status(motion["seq"], 1)

    def _simulation_loop(self):
        dt = float(self.model.opt.timestep)
        publish_period = 1.0 / 100.0
        viewer_period = 1.0 / 60.0
        last_pub = 0.0
        last_viewer = 0.0

        while self.running and rclpy.ok():
            tic = time.monotonic()

            with self.lock:
                self._step_direct_joint_motion()
                # MoveIt trajectory directly defines Panda2 state.
                self.data.qpos[self.arm_qadr] = self.q_command
                self.data.qvel[self.arm_vadr] = self.v_command if self.trajectory_active else 0.0

                # Normal gripper commands.
                p2_ctrl = self.gripper_command
                p1_ctrl = self.p1_gripper_command

                # While screwing, make the inactive robot's actuator target
                # exactly match the frozen physical finger pose.
                hold_inactive_gripper = (
                    self.screw_active
                    or self.screw_hold_release_steps > 0
                )

                if (
                    hold_inactive_gripper
                    and self.screw_hold_finger_qpos is not None
                    and self.screw_hold_finger_ctrl is not None
                ):
                    if self.screw_hold_robot == "panda1":
                        p1_ctrl = self.screw_hold_finger_ctrl
                        self.data.qpos[self.p1_finger_qadr] = (
                            self.screw_hold_finger_qpos
                        )
                        self.data.qvel[self.p1_finger_vadr] = 0.0
                    elif self.screw_hold_robot == "panda2":
                        p2_ctrl = self.screw_hold_finger_ctrl
                        self.data.qpos[self.finger_qadr] = (
                            self.screw_hold_finger_qpos
                        )
                        self.data.qvel[self.finger_vadr] = 0.0

                if hold_inactive_gripper and self.screw_owner_hold_qpos is not None:
                    if self.screw_owner_hold_robot == "panda2":
                        p2_ctrl = self.screw_owner_hold_ctrl
                        self.data.qpos[self.finger_qadr] = self.screw_owner_hold_qpos
                        self.data.qvel[self.finger_vadr] = 0.0
                    elif self.screw_owner_hold_robot == "panda1":
                        p1_ctrl = self.screw_owner_hold_ctrl
                        self.data.qpos[self.p1_finger_qadr] = self.screw_owner_hold_qpos
                        self.data.qvel[self.p1_finger_vadr] = 0.0

                self.data.ctrl[self.finger_act] = p2_ctrl

                self.data.qpos[self.p1_arm_qadr] = self.p1_q_command
                self.data.qvel[self.p1_arm_vadr] = self.p1_v_command if self.p1_trajectory_active else 0.0
                self.data.ctrl[self.p1_finger_act] = p1_ctrl

                if not self.screw_active:
                    self.data.ctrl[self.screwdriver_motor] = 0.0

                # Keep MuJoCo physics for gripper / scene, but arm pose is prescribed.
                mujoco.mj_step(self.model, self.data)

                # mj_step may modify arm qpos/qvel, overwrite again to exactly follow MoveIt.
                self.data.qpos[self.arm_qadr] = self.q_command
                self.data.qvel[self.arm_vadr] = self.v_command if self.trajectory_active else 0.0
                self.data.qpos[self.p1_arm_qadr] = self.p1_q_command
                self.data.qvel[self.p1_arm_vadr] = self.p1_v_command if self.p1_trajectory_active else 0.0

                # mj_step may still perturb constrained finger joints slightly;
                # restore the inactive gripper exactly before publishing/viewing.
                if (
                    hold_inactive_gripper
                    and self.screw_hold_finger_qpos is not None
                ):
                    if self.screw_hold_robot == "panda1":
                        self.data.qpos[self.p1_finger_qadr] = (
                            self.screw_hold_finger_qpos
                        )
                        self.data.qvel[self.p1_finger_vadr] = 0.0
                    elif self.screw_hold_robot == "panda2":
                        self.data.qpos[self.finger_qadr] = (
                            self.screw_hold_finger_qpos
                        )
                        self.data.qvel[self.finger_vadr] = 0.0

                if hold_inactive_gripper and self.screw_owner_hold_qpos is not None:
                    if self.screw_owner_hold_robot == "panda2":
                        self.data.qpos[self.finger_qadr] = self.screw_owner_hold_qpos
                        self.data.qvel[self.finger_vadr] = 0.0
                    elif self.screw_owner_hold_robot == "panda1":
                        self.data.qpos[self.p1_finger_qadr] = self.screw_owner_hold_qpos
                        self.data.qvel[self.p1_finger_vadr] = 0.0

                # Count only completed post-stop simulation steps.  The hold is
                # cleared after the last protected step, never before mj_step().
                if (
                    not self.screw_active
                    and self.screw_hold_release_steps > 0
                ):
                    self.screw_hold_release_steps -= 1
                    if self.screw_hold_release_steps == 0:
                        self.screw_hold_robot = None
                        self.screw_hold_finger_qpos = None
                        self.screw_hold_finger_ctrl = None
                        self.screw_owner_hold_robot = None
                        self.screw_owner_hold_qpos = None
                        self.screw_owner_hold_ctrl = None

                self._update_pick_objects()
                self._update_screw_motion()
                mujoco.mj_forward(self.model, self.data)
                self._auto_hex_tip_attach()
                self._auto_part_32_33_attach()
                self._auto_push_contact()

                now = time.monotonic()

                if now - last_pub >= publish_period:
                    self._publish_state()
                    last_pub = now

                if self.viewer is not None and self.viewer.is_running() and now - last_viewer >= viewer_period:
                    self.viewer.sync()
                    last_viewer = now

            elapsed = time.monotonic() - tic
            desired = dt / max(self.realtime_factor, 1e-6)
            if elapsed < desired:
                time.sleep(desired - elapsed)

    @staticmethod
    def _R_to_ros_quat(R):
        """Convert 3x3 rotation matrix to normalized ROS quaternion [x,y,z,w]."""
        mat = np.ascontiguousarray(R, dtype=np.float64).reshape(9)
        q_wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q_wxyz, mat)
        q_wxyz /= max(float(np.linalg.norm(q_wxyz)), 1e-12)
        return np.array(
            [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
            dtype=float
        )

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.STATE_JOINTS)
        msg.position = self.data.qpos[self.state_qadr].astype(float).tolist()
        msg.velocity = self.data.qvel[self.state_vadr].astype(float).tolist()
        self.joint_pub.publish(msg)

        # Publish the TRUE gripper end/TCP center.  Keep _finger_center()
        # unchanged for physical grasp-overlap detection.
        center = self._gripper_tcp_center(
            self.hand, self.left_geoms, self.right_geoms
        )
        p = PointStamped()
        p.header.stamp = msg.header.stamp
        p.header.frame_id = "mujoco_world"
        p.point.x, p.point.y, p.point.z = map(float, center)
        self.center_pub.publish(p)

        pose2 = PoseStamped()
        pose2.header.stamp = msg.header.stamp
        pose2.header.frame_id = "mujoco_world"
        pose2.pose.position.x, pose2.pose.position.y, pose2.pose.position.z = map(float, center)
        q2 = self._R_to_ros_quat(
            self.data.xmat[self.hand].reshape(3, 3)
        )
        pose2.pose.orientation.x = float(q2[0])
        pose2.pose.orientation.y = float(q2[1])
        pose2.pose.orientation.z = float(q2[2])
        pose2.pose.orientation.w = float(q2[3])
        self.p2_tcp_pose_pub.publish(pose2)

        p1 = JointState()
        p1.header.stamp = msg.header.stamp
        p1.name = list(self.P1_STATE_JOINTS)
        p1.position = self.data.qpos[self.p1_state_qadr].astype(float).tolist()
        p1.velocity = self.data.qvel[self.p1_state_vadr].astype(float).tolist()
        self.p1_joint_pub.publish(p1)

        c1 = self._gripper_tcp_center(
            self.p1_hand, self.p1_left_geoms, self.p1_right_geoms
        )
        pc1 = PointStamped()
        pc1.header.stamp = msg.header.stamp
        pc1.header.frame_id = "mujoco_world"
        pc1.point.x, pc1.point.y, pc1.point.z = map(float, c1)
        self.p1_center_pub.publish(pc1)

        pose1 = PoseStamped()
        pose1.header.stamp = msg.header.stamp
        pose1.header.frame_id = "mujoco_world"
        pose1.pose.position.x, pose1.pose.position.y, pose1.pose.position.z = map(float, c1)
        q1 = self._R_to_ros_quat(
            self.data.xmat[self.p1_hand].reshape(3, 3)
        )
        pose1.pose.orientation.x = float(q1[0])
        pose1.pose.orientation.y = float(q1[1])
        pose1.pose.orientation.z = float(q1[2])
        pose1.pose.orientation.w = float(q1[3])
        self.p1_tcp_pose_pub.publish(pose1)

    def _goal_callback(self, goal_request):
        names = list(goal_request.trajectory.joint_names)

        if set(names) != set(self.ARM_JOINTS):
            self.get_logger().error(f"Unexpected trajectory joints: {names}")
            return GoalResponse.REJECT

        if not goal_request.trajectory.points:
            return GoalResponse.REJECT

        with self.lock:
            if self.trajectory_active or self.direct_joint_motion is not None:
                self.get_logger().warn("Rejecting trajectory because another trajectory is active")
                return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    @staticmethod
    def _duration_sec(d):
        return float(d.sec) + float(d.nanosec) * 1e-9

    def _reorder(self, names, values):
        table = dict(zip(names, values))
        return np.array([table[n] for n in self.ARM_JOINTS], dtype=float)

    def _hold_current(self):
        with self.lock:
            q = self.data.qpos[self.arm_qadr].copy()
            self.q_command[:] = q
            self.v_command[:] = 0.0
            self.data.qpos[self.arm_qadr] = q
            self.data.qvel[self.arm_vadr] = 0.0
            self.trajectory_active = False
            self.arm_hold = True
            mujoco.mj_forward(self.model, self.data)

    def _execute_trajectory(self, goal_handle):
        traj = goal_handle.request.trajectory
        names = list(traj.joint_names)
        points = traj.points
        result = FollowJointTrajectory.Result()

        if not points:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result

        try:
            qs = [self._reorder(names, p.positions) for p in points]
            vs = [
                self._reorder(names, p.velocities)
                if len(p.velocities) == len(names) else np.zeros(7, dtype=float)
                for p in points
            ]

            ts = np.array(
                [self._duration_sec(p.time_from_start) for p in points],
                dtype=float
            ) * self.trajectory_time_scale

            with self.lock:
                q_start = self.data.qpos[self.arm_qadr].copy()
                self.arm_hold = False
                self.trajectory_active = True
                self.v_command[:] = 0.0

            if ts[0] > 1e-6:
                qs.insert(0, q_start)
                vs.insert(0, np.zeros(7, dtype=float))
                ts = np.insert(ts, 0, 0.0)

            start_wall = time.monotonic()
            control_period = 1.0 / 250.0
            segment_idx = 0
            last_feedback = 0.0

            nominal_duration = ts[-1] / max(self.trajectory_time_scale, 1e-9)
            self.get_logger().info(
                f"Executing trajectory: {len(points)} points, "
                f"MoveIt duration={nominal_duration:.3f}s, actual duration={ts[-1]:.3f}s"
            )

            while True:
                if goal_handle.is_cancel_requested:
                    self._hold_current()
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = "Trajectory canceled"
                    return result

                elapsed = time.monotonic() - start_wall
                if elapsed >= ts[-1]:
                    break

                while segment_idx + 1 < len(ts) - 1 and elapsed > ts[segment_idx + 1]:
                    segment_idx += 1

                t0, t1 = ts[segment_idx], ts[segment_idx + 1]
                segment_dt = max(t1 - t0, 1e-6)
                u = np.clip((elapsed - t0) / segment_dt, 0.0, 1.0)

                q0, q1 = qs[segment_idx], qs[segment_idx + 1]

                # Original MoveIt velocities must be scaled with the stretched time axis.
                v0 = vs[segment_idx] / self.trajectory_time_scale
                v1 = vs[segment_idx + 1] / self.trajectory_time_scale

                # Cubic Hermite position
                h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
                h10 = u**3 - 2.0 * u**2 + u
                h01 = -2.0 * u**3 + 3.0 * u**2
                h11 = u**3 - u**2
                cmd = h00 * q0 + h10 * segment_dt * v0 + h01 * q1 + h11 * segment_dt * v1

                # Exact derivative of Hermite interpolation
                dh00 = (6.0 * u**2 - 6.0 * u) / segment_dt
                dh10 = 3.0 * u**2 - 4.0 * u + 1.0
                dh01 = (-6.0 * u**2 + 6.0 * u) / segment_dt
                dh11 = 3.0 * u**2 - 2.0 * u
                vcmd = dh00 * q0 + dh10 * v0 + dh01 * q1 + dh11 * v1

                with self.lock:
                    self.q_command[:] = cmd
                    self.v_command[:] = vcmd
                    actual = self.data.qpos[self.arm_qadr].copy()

                now = time.monotonic()

                if now - last_feedback >= 0.02:
                    fb = FollowJointTrajectory.Feedback()
                    fb.header.stamp = self.get_clock().now().to_msg()
                    fb.joint_names = list(self.ARM_JOINTS)
                    fb.desired = JointTrajectoryPoint()
                    fb.actual = JointTrajectoryPoint()
                    fb.error = JointTrajectoryPoint()
                    fb.desired.positions = cmd.tolist()
                    fb.desired.velocities = vcmd.tolist()
                    fb.actual.positions = actual.tolist()
                    fb.error.positions = (cmd - actual).tolist()
                    goal_handle.publish_feedback(fb)
                    last_feedback = now

                time.sleep(control_period)

            # Final state — this is not a teleport:
            # q_command has continuously reached final_q along the planned trajectory.
            final_q = qs[-1].copy()

            with self.lock:
                self.q_command[:] = final_q
                self.v_command[:] = 0.0
                self.data.qpos[self.arm_qadr] = final_q
                self.data.qvel[self.arm_vadr] = 0.0
                self.trajectory_active = False
                self.arm_hold = True
                mujoco.mj_forward(self.model, self.data)

            self.get_logger().info("Trajectory execution completed exactly at final MoveIt state")

            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "MuJoCo trajectory execution completed"
            return result

        except Exception as e:
            self._hold_current()
            self.get_logger().error(f"Trajectory execution exception: {e}")

            try:
                goal_handle.abort()
            except Exception:
                pass

            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(e)
            return result

    def _p2_hand_rotation(self):
        return self.data.xmat[self.hand].reshape(3, 3).copy()

    def _p2_solve_ik(self, target_pos, target_R, seed):
        """Solve Panda2 fingertip-center IK without changing the live robot state."""
        with self.lock:
            live_q = self.data.qpos[self.arm_qadr].copy()
            live_v = self.data.qvel[self.arm_vadr].copy()
            q = np.asarray(seed, dtype=float).copy()
            jp1 = np.zeros((3, self.model.nv))
            jr1 = np.zeros((3, self.model.nv))
            jp2 = np.zeros((3, self.model.nv))
            jr2 = np.zeros((3, self.model.nv))
            try:
                for _ in range(8000):
                    self.data.qpos[self.arm_qadr] = q
                    self.data.qvel[self.arm_vadr] = 0.0
                    mujoco.mj_forward(self.model, self.data)

                    tcp = self._gripper_tcp_center(
                        self.hand, self.left_geoms, self.right_geoms
                    )
                    pe = np.asarray(target_pos, dtype=float) - tcp
                    R = self._p2_hand_rotation()
                    er = self._rotation_error(R, target_R)

                    mujoco.mj_jacBody(self.model, self.data, jp1, jr1, self.left_body)
                    mujoco.mj_jacBody(self.model, self.data, jp2, jr2, self.right_body)
                    Jp = 0.5 * (jp1[:, self.arm_vadr] + jp2[:, self.arm_vadr])
                    Jr = 0.5 * (jr1[:, self.arm_vadr] + jr2[:, self.arm_vadr])
                    # TCP is 8.3 mm beyond the old fingertip-geom center along
                    # hand-local +Z. Add omega x r to the translational Jacobian.
                    r_world = R @ np.array([0.0, 0.0, 0.0083], dtype=float)
                    Jp = Jp - np.array([
                        [0.0, -r_world[2], r_world[1]],
                        [r_world[2], 0.0, -r_world[0]],
                        [-r_world[1], r_world[0], 0.0],
                    ]) @ Jr
                    J = np.vstack([Jp, 0.45 * Jr])
                    e = np.r_[pe, 0.45 * er]
                    dq = J.T @ np.linalg.solve(J @ J.T + 2e-4*np.eye(6), e)
                    q += np.clip(0.12*dq, -0.025, 0.025)

                    for i, jid in enumerate(self.arm_joints):
                        if self.model.jnt_limited[jid]:
                            q[i] = np.clip(q[i], *self.model.jnt_range[jid])

                    if np.linalg.norm(pe) < 0.0001 and np.linalg.norm(er) < 0.012:
                        return q.copy()
                raise RuntimeError(
                    f"Panda2 IK failed: pos_err={np.linalg.norm(pe):.5f}, "
                    f"rot_err={np.linalg.norm(er):.5f}"
                )
            finally:
                self.data.qpos[self.arm_qadr] = live_q
                self.data.qvel[self.arm_vadr] = live_v
                mujoco.mj_forward(self.model, self.data)

    def _p2_plan_to_pose_cb(self, request, response):
        """Plan Panda2 TCP to an absolute WORLD pose."""
        try:
            target = np.array([
                request.target.pose.position.x,
                request.target.pose.position.y,
                request.target.pose.position.z
            ], dtype=float)
            Rt = self._quat_to_R(
                request.target.pose.orientation.w,
                request.target.pose.orientation.x,
                request.target.pose.orientation.y,
                request.target.pose.orientation.z
            )

            with self.lock:
                if self.p2_plan_queue:
                    seed = self.p2_plan_queue[-1][-1][0].copy()
                    live_q = self.data.qpos[self.arm_qadr].copy()
                    live_v = self.data.qvel[self.arm_vadr].copy()
                    self.data.qpos[self.arm_qadr] = seed
                    self.data.qvel[self.arm_vadr] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                    start_pos = self._gripper_tcp_center(
                        self.hand, self.left_geoms, self.right_geoms
                    ).copy()
                    self.data.qpos[self.arm_qadr] = live_q
                    self.data.qvel[self.arm_vadr] = live_v
                    mujoco.mj_forward(self.model, self.data)
                else:
                    seed = self.data.qpos[self.arm_qadr].copy()
                    start_pos = self._gripper_tcp_center(
                        self.hand, self.left_geoms, self.right_geoms
                    ).copy()

            if bool(request.linear):
                steps = 25
                qs = [seed.copy()]
                q = seed.copy()
                for u in np.linspace(1.0 / steps, 1.0, steps):
                    p = (1.0 - u) * start_pos + u * target
                    q = self._p2_solve_ik(p, Rt, q)
                    qs.append(q.copy())
            else:
                qgoal = self._p2_solve_ik(target, Rt, seed)
                steps = 50
                qs = [
                    (1.0 - u) * seed + u * qgoal
                    for u in np.linspace(0.0, 1.0, steps + 1)
                ]

            self.p2_plan_queue.append(self._p1_make_traj(qs, 1.5))
            response.success = True
            response.message = f"Panda2 plan queued ({len(qs)} points)"
            self.get_logger().info(response.message)
        except Exception as e:
            self.p2_plan_queue.clear()
            self.p2_queue_target_R = None
            response.success = False
            response.message = str(e)
            self.get_logger().error(
                f"Panda2 planning failed; queue cleared: {e}"
            )
        return response

    def _p2_clear_plan_queue_cb(self, request, response):
        del request
        self.p2_plan_queue.clear()
        self.p2_queue_target_R = None
        response.success = True
        response.message = "Panda2 plan queue cleared"
        return response

    def _p2_execute_next_plan_cb(self, request, response):
        del request
        if not self.p2_plan_queue:
            response.success = False
            response.message = "Panda2 plan queue is empty"
            return response
        if self.trajectory_active:
            response.success = False
            response.message = "Panda2 trajectory already active"
            return response

        plan = self.p2_plan_queue.pop(0)
        try:
            with self.lock:
                self.trajectory_active = True

            start = time.monotonic()
            seg = 0
            ts = np.array([p[1] for p in plan], dtype=float)
            qs = [p[0] for p in plan]

            push_interrupted = False
            while True:
                with self.lock:
                    if (self.push_active["panda2"] and self.push_object["panda2"] is not None
                            and not self.push_search_interrupted["panda2"]):
                        self.push_search_interrupted["panda2"] = True
                        push_interrupted = True
                        break
                t = time.monotonic() - start
                if t >= ts[-1]:
                    break
                while seg + 1 < len(ts) - 1 and t > ts[seg + 1]:
                    seg += 1

                dt = max(ts[seg + 1] - ts[seg], 1e-6)
                u = float(np.clip((t - ts[seg]) / dt, 0.0, 1.0))
                s = 3.0 * u * u - 2.0 * u * u * u
                ds = (6.0 * u - 6.0 * u * u) / dt
                q = (1.0 - s) * qs[seg] + s * qs[seg + 1]
                v = ds * (qs[seg + 1] - qs[seg])

                with self.lock:
                    self.q_command[:] = q
                    self.v_command[:] = v
                time.sleep(1.0 / 250.0)

            if push_interrupted:
                with self.lock:
                    self.q_command[:] = self.data.qpos[self.arm_qadr]
                    self.v_command[:] = 0.0
                    self.trajectory_active = False
                response.success = True
                response.message = "Panda2 continuous search interrupted at detected push contact"
                return response
            with self.lock:
                self.q_command[:] = qs[-1]
                self.v_command[:] = 0.0
                self.data.qpos[self.arm_qadr] = qs[-1]
                self.data.qvel[self.arm_vadr] = 0.0
                self.trajectory_active = False
                mujoco.mj_forward(self.model, self.data)

            response.success = True
            response.message = "Panda2 queued plan executed"
        except Exception as e:
            with self.lock:
                self.trajectory_active = False
                self.v_command[:] = 0.0
            response.success = False
            response.message = str(e)
        return response

    def _p2_plan_relative_z_cb(self, request, response):
        """request.target.pose.position.z is interpreted as a +Z displacement."""
        try:
            dz = float(request.target.pose.position.z)
            if dz <= 0.0:
                raise RuntimeError(f"relative Z must be > 0, got {dz}")

            with self.lock:
                seed = self.data.qpos[self.arm_qadr].copy()
                start = self._finger_center().copy()
                Rt = self._p2_hand_rotation()

            steps = 25
            qs = [seed.copy()]
            q = seed.copy()
            for u in np.linspace(1.0/steps, 1.0, steps):
                p = start + np.array([0.0, 0.0, dz*u])
                q = self._p2_solve_ik(p, Rt, q)
                qs.append(q.copy())

            duration = 2.5
            self.p2_relative_z_plan = self._p1_make_traj(qs, duration)
            response.success = True
            response.message = (
                f"Panda2 +Z {dz:.4f} m preplanned ({len(qs)} points), "
                "orientation locked"
            )
            self.get_logger().info(response.message)
        except Exception as e:
            self.p2_relative_z_plan = None
            response.success = False
            response.message = str(e)
            self.get_logger().error(f"Panda2 relative-Z planning failed: {e}")
        return response

    def _p2_execute_relative_z_cb(self, request, response):
        del request
        if self.p2_relative_z_plan is None:
            response.success = False
            response.message = "No preplanned Panda2 relative-Z path"
            return response

        plan = self.p2_relative_z_plan
        self.p2_relative_z_plan = None
        try:
            with self.lock:
                self.trajectory_active = True
            start_wall = time.monotonic()
            seg = 0
            ts = np.array([p[1] for p in plan], dtype=float)
            qs = [p[0] for p in plan]

            while True:
                t = time.monotonic() - start_wall
                if t >= ts[-1]:
                    break
                while seg + 1 < len(ts)-1 and t > ts[seg+1]:
                    seg += 1
                dt = max(ts[seg+1]-ts[seg], 1e-6)
                u = float(np.clip((t-ts[seg])/dt, 0.0, 1.0))
                s = u if isinstance(plan, _ContinuousOrbitPlan) else 3*u*u - 2*u*u*u
                ds = (1.0/dt if isinstance(plan, _ContinuousOrbitPlan)
                      else (6*u - 6*u*u)/dt)
                q = (1-s)*qs[seg] + s*qs[seg+1]
                v = ds*(qs[seg+1]-qs[seg])
                with self.lock:
                    self.q_command[:] = q
                    self.v_command[:] = v
                time.sleep(1.0/250.0)

            with self.lock:
                self.q_command[:] = qs[-1]
                self.v_command[:] = 0.0
                self.data.qpos[self.arm_qadr] = qs[-1]
                self.data.qvel[self.arm_vadr] = 0.0
                self.trajectory_active = False
                mujoco.mj_forward(self.model, self.data)
            response.success = True
            response.message = "Panda2 preplanned relative-Z path executed"
        except Exception as e:
            with self.lock:
                self.trajectory_active = False
                self.v_command[:] = 0.0
            response.success = False
            response.message = str(e)
        return response

    def _p1_finger_center(self):
        lp = np.mean(self.data.geom_xpos[self.p1_left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[self.p1_right_geoms], axis=0)
        return 0.5 * (lp + rp)

    @staticmethod
    def _rotation_error(R, Rt):
        return 0.5 * (
            np.cross(R[:, 0], Rt[:, 0]) +
            np.cross(R[:, 1], Rt[:, 1]) +
            np.cross(R[:, 2], Rt[:, 2])
        )

    def _p1_solve_ik(self, target_pos, target_R, seed):
        # Work on the bridge's already-compiled model, but restore the live
        # Panda1 state after each solve so planning itself never moves the robot.
        with self.lock:
            live_q = self.data.qpos[self.p1_arm_qadr].copy()
            live_v = self.data.qvel[self.p1_arm_vadr].copy()
            q = np.asarray(seed, dtype=float).copy()
            jp1 = np.zeros((3, self.model.nv))
            jr1 = np.zeros((3, self.model.nv))
            jp2 = np.zeros((3, self.model.nv))
            jr2 = np.zeros((3, self.model.nv))
            try:
                for _ in range(8000):
                    self.data.qpos[self.p1_arm_qadr] = q
                    self.data.qvel[self.p1_arm_vadr] = 0.0
                    mujoco.mj_forward(self.model, self.data)

                    tcp = self._gripper_tcp_center(
                        self.p1_hand, self.p1_left_geoms, self.p1_right_geoms
                    )
                    pe = np.asarray(target_pos, dtype=float) - tcp
                    R = self.data.xmat[self.p1_hand].reshape(3, 3).copy()
                    er = self._rotation_error(R, target_R)

                    mujoco.mj_jacBody(self.model, self.data, jp1, jr1, self.p1_left_body)
                    mujoco.mj_jacBody(self.model, self.data, jp2, jr2, self.p1_right_body)
                    Jp = 0.5 * (jp1[:, self.p1_arm_vadr] + jp2[:, self.p1_arm_vadr])
                    Jr = 0.5 * (jr1[:, self.p1_arm_vadr] + jr2[:, self.p1_arm_vadr])
                    r_world = R @ np.array([0.0, 0.0, 0.0083], dtype=float)
                    Jp = Jp - np.array([
                        [0.0, -r_world[2], r_world[1]],
                        [r_world[2], 0.0, -r_world[0]],
                        [-r_world[1], r_world[0], 0.0],
                    ]) @ Jr
                    J = np.vstack([Jp, 0.45 * Jr])
                    e = np.r_[pe, 0.45 * er]
                    dq = J.T @ np.linalg.solve(J @ J.T + 2e-4*np.eye(6), e)
                    q += np.clip(0.12 * dq, -0.025, 0.025)

                    for i, jid in enumerate(self.p1_arm_joints):
                        if self.model.jnt_limited[jid]:
                            q[i] = np.clip(q[i], *self.model.jnt_range[jid])

                    if np.linalg.norm(pe) < 0.0020 and np.linalg.norm(er) < 0.012:
                        return q.copy()
                raise RuntimeError(
                    f"Panda1 IK failed: pos_err={np.linalg.norm(pe):.5f}, "
                    f"rot_err={np.linalg.norm(er):.5f}"
                )
            finally:
                self.data.qpos[self.p1_arm_qadr] = live_q
                self.data.qvel[self.p1_arm_vadr] = live_v
                mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _quat_to_R(qw, qx, qy, qz):
        q = np.array([qw, qx, qy, qz], dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise RuntimeError("Panda1 target quaternion has zero norm")
        q /= n
        Rflat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(Rflat, q)
        return Rflat.reshape(3, 3)

    def _plan_continuous_orbit(self, arm, request, response):
        """One dense Cartesian circular orbit -> one continuous joint trajectory.

        Service encoding (PlanToPose, avoiding a new .srv build):
          target.pose.position.x/y = WORLD orbit axis XY
          target.pose.position.z   = signed rotation angle in RADIANS
          velocity_scale/acceleration_scale = speed fractions
        Start TCP pose and orientation are read from the actual MuJoCo robot.
        """
        try:
            cx = float(request.target.pose.position.x)
            cy = float(request.target.pose.position.y)
            theta = float(request.target.pose.position.z)
            speed = float(request.velocity_scale)
            accel = float(request.acceleration_scale)
            if not np.all(np.isfinite([cx, cy, theta, speed, accel])):
                raise ValueError("Nonfinite orbit parameters")
            if not (0.0 < speed <= 1.0 and 0.0 < accel <= 1.0):
                raise ValueError("velocity/acceleration scales must be in (0,1]")
            if abs(theta) > 2 * np.pi + 1e-8:
                raise ValueError("Orbit angle exceeds 360 degrees")
            if arm == "panda1":
                adr, vadr = self.p1_arm_qadr, self.p1_arm_vadr
                hand, lg, rg = self.p1_hand, self.p1_left_geoms, self.p1_right_geoms
                solver, queue = self._p1_solve_ik, self.p1_plan_queue
            else:
                adr, vadr = self.arm_qadr, self.arm_vadr
                hand, lg, rg = self.hand, self.left_geoms, self.right_geoms
                solver, queue = self._p2_solve_ik, self.p2_plan_queue
            with self.lock:
                if queue:
                    raise RuntimeError("Clear existing plan queue before orbit")
                mujoco.mj_forward(self.model, self.data)
                q0 = self.data.qpos[adr].copy()
                p0 = self._gripper_tcp_center(hand, lg, rg).copy()
                R0 = self.data.xmat[hand].reshape(3, 3).copy()
            # Internal sampling is independent of YAML: 1 degree max per point.
            steps = max(1, int(np.ceil(abs(theta) / np.deg2rad(1.0))))
            dx, dy = p0[0] - cx, p0[1] - cy
            qs = [q0.copy()]
            q = q0.copy()
            for i in range(1, steps + 1):
                a = theta * i / steps
                c, s = np.cos(a), np.sin(a)
                target = np.array([cx + c * dx - s * dy,
                                   cy + s * dx + c * dy, p0[2]])
                Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                q = solver(target, Rz @ R0, q)
                qs.append(q.copy())
            # Conservative duration based on joint-space path length. Existing
            # executor follows this entire plan in ONE call, no waypoint pauses.
            path = np.asarray(qs)
            joint_travel = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0)))
            duration = max(0.4, joint_travel / max(0.5 * speed, 0.01),
                           0.4 / np.sqrt(accel))
            plan = _ContinuousOrbitPlan(self._p1_make_traj(qs, duration))
            with self.lock:
                queue.append(plan)
            response.success = True
            response.message = (f"{arm} continuous orbit queued: {steps} IK samples, "
                                f"duration={duration:.2f}s, TCP Z={p0[2]:.6f}")
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f"{arm} orbit planning failed: {e}")
        return response

    def _p1_make_traj(self, qs, duration):
        traj = []
        n = max(len(qs) - 1, 1)
        for i, q in enumerate(qs):
            traj.append((np.asarray(q, dtype=float).copy(), float(duration) * i / n))
        return traj

    def _p1_plan_to_pose_cb(self, request, response):
        try:
            target = np.array([
                request.target.pose.position.x,
                request.target.pose.position.y,
                request.target.pose.position.z
            ], dtype=float)
            Rt = self._quat_to_R(
                request.target.pose.orientation.w,
                request.target.pose.orientation.x,
                request.target.pose.orientation.y,
                request.target.pose.orientation.z
            )

            with self.lock:

                if self.p1_plan_queue:
                    seed = self.p1_plan_queue[-1][-1][0].copy()
                    # Obtain the virtual start fingertip center from the queued endpoint.
                    live_q = self.data.qpos[self.p1_arm_qadr].copy()
                    live_v = self.data.qvel[self.p1_arm_vadr].copy()
                    self.data.qpos[self.p1_arm_qadr] = seed
                    self.data.qvel[self.p1_arm_vadr] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                    start_pos = self._gripper_tcp_center(
                        self.p1_hand, self.p1_left_geoms, self.p1_right_geoms
                    ).copy()
                    self.data.qpos[self.p1_arm_qadr] = live_q
                    self.data.qvel[self.p1_arm_vadr] = live_v
                    mujoco.mj_forward(self.model, self.data)
                else:
                    seed = self.data.qpos[self.p1_arm_qadr].copy()
                    start_pos = self._gripper_tcp_center(
                        self.p1_hand, self.p1_left_geoms, self.p1_right_geoms
                    ).copy()

            if bool(request.linear):
                steps = 25
                qs = [seed.copy()]
                q = seed.copy()
                for u in np.linspace(1.0/steps, 1.0, steps):
                    p = (1.0-u)*start_pos + u*target
                    q = self._p1_solve_ik(p, Rt, q)
                    qs.append(q.copy())
            else:
                qgoal = self._p1_solve_ik(target, Rt, seed)
                # Smooth joint-space interpolation for the pre-handover leg.
                steps = 50
                qs = [(1.0-u)*seed + u*qgoal for u in np.linspace(0.0, 1.0, steps+1)]

            duration = 2.5
            self.p1_plan_queue.append(self._p1_make_traj(qs, duration))
            response.success = True
            response.message = f"Panda1 plan queued ({len(qs)} points)"
            self.get_logger().info(response.message)
        except Exception as e:
            self.p1_plan_queue.clear()
            response.success = False
            response.message = str(e)
            self.get_logger().error(f"Panda1 planning failed; queue cleared: {e}")
        return response

    def _p1_clear_plan_queue_cb(self, request, response):
        del request
        self.p1_plan_queue.clear()
        self.p1_queue_target_R = None
        response.success = True
        response.message = "Panda1 plan queue cleared"
        return response

    def _p1_execute_next_plan_cb(self, request, response):
        del request
        if not self.p1_plan_queue:
            response.success = False
            response.message = "Panda1 plan queue is empty"
            return response
        if self.p1_trajectory_active:
            response.success = False
            response.message = "Panda1 trajectory already active"
            return response

        plan = self.p1_plan_queue.pop(0)
        try:
            with self.lock:
                self.p1_trajectory_active = True
            start = time.monotonic()
            seg = 0
            ts = np.array([p[1] for p in plan], dtype=float)
            qs = [p[0] for p in plan]
            push_interrupted = False
            while True:
                with self.lock:
                    if (self.push_active["panda1"] and self.push_object["panda1"] is not None
                            and not self.push_search_interrupted["panda1"]):
                        self.push_search_interrupted["panda1"] = True
                        push_interrupted = True
                        break
                t = time.monotonic() - start
                if t >= ts[-1]:
                    break
                while seg + 1 < len(ts)-1 and t > ts[seg+1]:
                    seg += 1
                dt = max(ts[seg+1]-ts[seg], 1e-6)
                u = float(np.clip((t-ts[seg])/dt, 0.0, 1.0))
                s = u if isinstance(plan, _ContinuousOrbitPlan) else 3*u*u - 2*u*u*u
                ds = (1.0/dt if isinstance(plan, _ContinuousOrbitPlan)
                      else (6*u - 6*u*u)/dt)
                q = (1-s)*qs[seg] + s*qs[seg+1]
                v = ds*(qs[seg+1]-qs[seg])
                with self.lock:
                    self.p1_q_command[:] = q
                    self.p1_v_command[:] = v
                time.sleep(1.0/250.0)

            if push_interrupted:
                with self.lock:
                    self.p1_q_command[:] = self.data.qpos[self.p1_arm_qadr]
                    self.p1_v_command[:] = 0.0
                    self.p1_trajectory_active = False
                response.success = True
                response.message = "Panda1 continuous search interrupted at detected push contact"
                return response
            with self.lock:
                self.p1_q_command[:] = qs[-1]
                self.p1_v_command[:] = 0.0
                self.data.qpos[self.p1_arm_qadr] = qs[-1]
                self.data.qvel[self.p1_arm_vadr] = 0.0
                self.p1_trajectory_active = False
                mujoco.mj_forward(self.model, self.data)
            response.success = True
            response.message = "Panda1 queued plan executed"
        except Exception as e:
            with self.lock:
                self.p1_trajectory_active = False
                self.p1_v_command[:] = 0.0
            response.success = False
            response.message = str(e)
        return response

    def _p1_goal_callback(self, goal_request):
        names = list(goal_request.trajectory.joint_names)
        if set(names) != set(self.P1_ARM_JOINTS) or not goal_request.trajectory.points:
            self.get_logger().error(f"Unexpected Panda1 trajectory joints: {names}")
            return GoalResponse.REJECT
        with self.lock:
            if self.p1_trajectory_active or self.direct_joint_motion is not None:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _p1_reorder(self, names, values):
        table = dict(zip(names, values))
        return np.array([table[n] for n in self.P1_ARM_JOINTS], dtype=float)

    def _execute_p1_trajectory(self, goal_handle):
        result = FollowJointTrajectory.Result()
        try:
            traj = goal_handle.request.trajectory
            names = list(traj.joint_names)
            points = traj.points
            qs = [self._p1_reorder(names, p.positions) for p in points]
            ts = np.array([self._duration_sec(p.time_from_start) for p in points], dtype=float)
            if len(ts) == 1:
                ts[0] = max(ts[0], 0.01)
            with self.lock:
                q_start = self.data.qpos[self.p1_arm_qadr].copy()
                self.p1_trajectory_active = True
            if ts[0] > 1e-6:
                qs.insert(0, q_start)
                ts = np.insert(ts, 0, 0.0)
            start = time.monotonic()
            seg = 0
            while True:
                if goal_handle.is_cancel_requested:
                    with self.lock:
                        self.p1_q_command[:] = self.data.qpos[self.p1_arm_qadr]
                        self.p1_v_command[:] = 0.0
                        self.p1_trajectory_active = False
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    return result
                t = time.monotonic() - start
                if t >= ts[-1]:
                    break
                while seg + 1 < len(ts) - 1 and t > ts[seg + 1]:
                    seg += 1
                dt = max(ts[seg + 1] - ts[seg], 1e-6)
                u = float(np.clip((t - ts[seg]) / dt, 0.0, 1.0))
                s = 3*u*u - 2*u*u*u
                cmd = (1-s)*qs[seg] + s*qs[seg+1]
                ds = (6*u - 6*u*u) / dt
                vel = ds*(qs[seg+1] - qs[seg])
                with self.lock:
                    self.p1_q_command[:] = cmd
                    self.p1_v_command[:] = vel
                time.sleep(1.0/250.0)
            with self.lock:
                self.p1_q_command[:] = qs[-1]
                self.p1_v_command[:] = 0.0
                self.data.qpos[self.p1_arm_qadr] = qs[-1]
                self.data.qvel[self.p1_arm_vadr] = 0.0
                self.p1_trajectory_active = False
                mujoco.mj_forward(self.model, self.data)
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "Panda1 trajectory completed"
            return result
        except Exception as e:
            with self.lock:
                self.p1_trajectory_active = False
                self.p1_v_command[:] = 0.0
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(e)
            return result

    def _p1_total_width_range(self):
        lo, hi = self.model.jnt_range[self.p1_finger_joint]
        return 2.0*float(min(lo,hi)), 2.0*float(max(lo,hi))

    def _p1_width_to_ctrl(self, width):
        lo, hi = self._p1_total_width_range()
        a, b = map(float, self.model.actuator_ctrlrange[self.p1_finger_act])
        clo, chi = min(a,b), max(a,b)
        w = float(np.clip(width, lo, hi))
        return clo + (w-lo)/max(hi-lo,1e-12)*(chi-clo)

    def _p1_ctrl_to_total_width(self, ctrl):
        """Inverse of _p1_width_to_ctrl(), used for deterministic screw hold."""
        lo, hi = self._p1_total_width_range()
        a, b = map(float, self.model.actuator_ctrlrange[self.p1_finger_act])
        clo, chi = min(a, b), max(a, b)
        if chi - clo <= 1e-12:
            return lo
        u = (float(ctrl) - clo) / (chi - clo)
        u = float(np.clip(u, 0.0, 1.0))
        return lo + u * (hi - lo)

    def _p1_gripper_goal_callback(self, goal_request):
        width = float(goal_request.command.position)
        lo, hi = self._p1_total_width_range()
        return GoalResponse.ACCEPT if lo-1e-6 <= width <= hi+1e-6 else GoalResponse.REJECT

    def _wait_gripper_width(self, robot, target, timeout=4.0, tolerance=0.001):
        """Wait for actual two-finger joint width; log both joints for diagnosis."""
        qadr = self.p1_finger_qadr if robot == "panda1" else self.finger_qadr
        deadline = time.monotonic() + timeout
        last = None
        stable = 0
        while True:
            with self.lock:
                fingers = self.data.qpos[qadr].copy()
                velocities = self.data.qvel[self.p1_finger_vadr if robot == "panda1" else self.finger_vadr].copy()
            actual = float(fingers[0] + fingers[1])
            if abs(actual - target) <= tolerance:
                break
            if last is not None and abs(actual - last) < 0.00002 and float(np.max(np.abs(velocities))) < 0.001:
                stable += 1
            else:
                stable = 0
            if stable >= 15 or time.monotonic() >= deadline:
                break
            last = actual
            time.sleep(0.02)
        self.get_logger().info(
            f"{robot} gripper measured: target={target:.5f}, "
            f"finger1={fingers[0]:.6f}, finger2={fingers[1]:.6f}, "
            f"joint_sum={actual:.6f}, old_2xfinger1={2*fingers[0]:.6f}, "
            f"error={actual-target:+.6f}, stable_samples={stable}"
        )
        return actual

    def _execute_p1_gripper_width(self, goal_handle):
        """Panda1 total-width command with explicit OPEN/RELEASE vs GRASP intent."""
        result = GripperCommand.Result()
        width = float(goal_handle.request.command.position)
        effort = float(goal_handle.request.command.max_effort)
        grasp_command = effort > 0.0
        lo, hi = self._p1_total_width_range()
        width = float(np.clip(width, lo, hi))
        ctrl = self._p1_width_to_ctrl(width)

        with self.lock:
            released = None
            if not grasp_command:
                released = self._release_current_object("panda1")
            self.p1_gripper_command = ctrl

        actual = self._wait_gripper_width("panda1", width)

        if not grasp_command:
            goal_handle.succeed()
            result.position = actual
            result.effort = 0.0
            result.stalled = False
            result.reached_goal = abs(actual - width) <= 0.002
            self.get_logger().info(
                f"Panda1 OPEN/RELEASE target={width:.4f}, actual={actual:.4f}"
                + (f", detached '{released}'" if released else "")
            )
            return result

        with self.lock:
            obj, distance, error = self._establish_nearest_grasp("panda1")
        if error is not None:
            goal_handle.abort()
            result.position = actual
            result.effort = effort
            result.stalled = True
            result.reached_goal = False
            self.get_logger().error(f"Panda1 grasp failed: {error}")
            return result

        goal_handle.succeed()
        result.position = actual
        result.effort = effort
        result.stalled = False
        result.reached_goal = True
        self.get_logger().info(
            f"Panda1 grasped '{obj['name']}', distance={distance:.4f} m"
        )
        return result

    def _wait_until_stopped_cb(self, request, response):
        del request
        with self.lock:
            q2 = self.data.qpos[self.arm_qadr].copy()
            v2 = self.data.qvel[self.arm_vadr].copy()
            e2 = float(np.max(np.abs(q2 - self.q_command)))
            vmax2 = float(np.max(np.abs(v2)))
            q1 = self.data.qpos[self.p1_arm_qadr].copy()
            v1 = self.data.qvel[self.p1_arm_vadr].copy()
            e1 = float(np.max(np.abs(q1 - self.p1_q_command)))
            vmax1 = float(np.max(np.abs(v1)))
            active = self.trajectory_active or self.p1_trajectory_active

        response.success = not active
        response.message = (
            f"{'Robot still executing' if active else 'Robots stopped'}, "
            f"panda1 vmax={vmax1:.6f} qerr={e1:.6f}, "
            f"panda2 vmax={vmax2:.6f} qerr={e2:.6f}"
        )
        return response

    def _gripper_total_width_range(self):
        """
        Return the physical TOTAL gripper opening range in meters.

        Panda finger_joint1 is one-finger displacement. Because finger_joint2
        mirrors it, total opening width = 2 * one-finger displacement.
        """
        joint_lo, joint_hi = self.model.jnt_range[self.finger_joint]
        total_lo = 2.0 * float(min(joint_lo, joint_hi))
        total_hi = 2.0 * float(max(joint_lo, joint_hi))
        return total_lo, total_hi

    def _width_to_actuator_ctrl(self, total_width):
        """
        Convert physical total gripper width [m] to actuator8 control units.

        MuJoCo Menagerie Panda uses:
          finger joint range : 0 .. 0.04 m
          actuator8 ctrlrange: 0 .. 255

        Do NOT send meters directly to actuator8.
        """
        total_lo, total_hi = self._gripper_total_width_range()
        ctrl_a, ctrl_b = map(float, self.model.actuator_ctrlrange[self.finger_act])
        ctrl_lo = min(ctrl_a, ctrl_b)
        ctrl_hi = max(ctrl_a, ctrl_b)

        width = float(np.clip(total_width, total_lo, total_hi))

        if total_hi - total_lo <= 1e-12:
            return ctrl_lo

        u = (width - total_lo) / (total_hi - total_lo)
        ctrl = ctrl_lo + u * (ctrl_hi - ctrl_lo)
        return float(ctrl)

    def _actuator_ctrl_to_total_width(self, ctrl):
        """Inverse conversion, useful for diagnostics."""
        total_lo, total_hi = self._gripper_total_width_range()
        ctrl_a, ctrl_b = map(float, self.model.actuator_ctrlrange[self.finger_act])
        ctrl_lo = min(ctrl_a, ctrl_b)
        ctrl_hi = max(ctrl_a, ctrl_b)

        if ctrl_hi - ctrl_lo <= 1e-12:
            return total_lo

        u = (float(ctrl) - ctrl_lo) / (ctrl_hi - ctrl_lo)
        u = float(np.clip(u, 0.0, 1.0))
        return total_lo + u * (total_hi - total_lo)

    def _gripper_goal_callback(self, goal_request):
        width = float(goal_request.command.position)
        lo, hi = self._gripper_total_width_range()

        if width < lo - 1e-6 or width > hi + 1e-6:
            self.get_logger().error(
                f"Rejecting gripper width {width:.4f} m; "
                f"physical valid range [{lo:.4f}, {hi:.4f}] m"
            )
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _release_current_object(self, robot=None):
        """Release only the specified arm's object, preserving the other arm."""
        if robot is None:
            robot = self.grasp_robot
        grasp = self.arm_grasps.get(robot)
        if grasp is None:
            return None
        obj = grasp["object"]
        qpos = obj["qpos"]
        obj["last_pos"] = self.data.qpos[qpos:qpos + 3].copy()
        obj["quat"] = self.data.qpos[qpos + 3:qpos + 7].copy()
        self.arm_grasps[robot] = None
        self._sync_legacy_grasp_state()
        return obj["name"]

    def _establish_nearest_grasp(self, robot="panda2"):
        """Grasp an overlapping free object, or transfer an overlapping held one."""
        existing = self.arm_grasps[robot]
        if existing is not None:
            return existing["object"], 0.0, None

        # Candidate selection considers ALL objects; a held object only triggers
        # handover if it actually overlaps this receiver's fingertips.
        obj, distance = self._nearest_pick_object_for_robot(robot)
        if obj is None:
            return None, float("inf"), f"no pickable object overlaps {robot} fingertips"
        if distance > self.grasp_max_distance:
            return obj, distance, (
                f"nearest object '{obj['name']}' is {distance:.4f} m away "
                f"(limit {self.grasp_max_distance:.4f} m)")

        if obj["name"] in self.hex_tip_attachments:
            return obj, distance, (
                f"'{obj['name']}' is already attached to hex_key"
            )
        previous_owner = self._object_owner(obj)
        mujoco.mj_forward(self.model, self.data)
        hand_id, _, _ = self._grasp_hand_spec(robot)
        qpos = int(obj["qpos"])
        object_pos = self.data.qpos[qpos:qpos + 3].copy()
        object_R = self.data.xmat[int(obj["body"])].reshape(3, 3).copy()
        hand_pos = self.data.xpos[hand_id].copy()
        hand_R = self.data.xmat[hand_id].reshape(3, 3).copy()
        center = self._finger_center_for_robot(robot)
        grasp = {
            "object": obj,
            "rel_pos": hand_R.T @ (object_pos - hand_pos),
            "rel_R": hand_R.T @ object_R,
            "offset": object_pos - center,
        }
        # Atomic ownership transfer: never let both arms own the same object.
        if previous_owner is not None and previous_owner != robot:
            self.arm_grasps[previous_owner] = None
        self.arm_grasps[robot] = grasp
        self._sync_legacy_grasp_state()
        obj["last_pos"] = object_pos.copy()
        obj["quat"] = self.data.qpos[qpos + 3:qpos + 7].copy()
        if previous_owner is not None and previous_owner != robot:
            self.get_logger().info(
                f"ATOMIC REGRASP TRANSFER: {previous_owner} -> {robot}, "
                f"object='{obj['name']}'")
        self.get_logger().info(
            f"Rigid grasp attached '{obj['name']}' to {robot}: "
            f"hand-local offset={grasp['rel_pos'].tolist()}")
        return obj, distance, None

    def _execute_gripper_width(self, goal_handle):
        """
        Command physical TOTAL opening width in meters.

        Command intent is explicit:
          max_effort <= 0.0 : OPEN/RELEASE -- detach immediately and NEVER attach.
          max_effort >  0.0 : GRASP        -- attachment is allowed after motion.

        Do not infer grasp/release intent from whether target width is larger
        than the current width.  In particular, an OPEN command whose target
        equals the current width must not accidentally re-attach the previous
        object.
        """
        result = GripperCommand.Result()

        requested_width = float(goal_handle.request.command.position)
        requested_effort = float(goal_handle.request.command.max_effort)
        grasp_command = requested_effort > 0.0

        lo, hi = self._gripper_total_width_range()
        width = float(np.clip(requested_width, lo, hi))
        actuator_ctrl = self._width_to_actuator_ctrl(width)

        with self.lock:
            qadr = int(self.model.jnt_qposadr[self.finger_joint])
            current_width = 2.0 * float(self.data.qpos[qadr])

            # Every OPEN/RELEASE command clears attachment state immediately,
            # regardless of target width or current physical width.
            released_name = None
            if not grasp_command:
                released_name = self._release_current_object("panda2")

            self.gripper_command = actuator_ctrl

        mode = "GRASP" if grasp_command else "OPEN/RELEASE"
        self.get_logger().info(
            f"Gripper {mode}: current={current_width:.4f} m, "
            f"target={width:.4f} m, max_effort={requested_effort:.3f}, "
            f"ctrl={actuator_ctrl:.3f}" +
            (f", detached '{released_name}'" if released_name else "")
        )

        # Wait for actual motion rather than sampling after a fixed 0.4 s.
        actual_width = self._wait_gripper_width("panda2", width)

        if not grasp_command:
            # IMPORTANT: an OPEN/RELEASE command returns here.  It can never
            # fall through to _establish_nearest_grasp().
            goal_handle.succeed()
            result.position = actual_width
            result.effort = 0.0
            result.stalled = False
            result.reached_goal = abs(actual_width - width) <= 0.002

            self.get_logger().info(
                f"Gripper OPEN/RELEASE finished: target={width:.4f} m, "
                f"actual={actual_width:.4f} m, attachment disabled"
            )
            return result

        # Only an explicit GRASP command is allowed to attach an object.
        with self.lock:
            obj, distance, error = self._establish_nearest_grasp("panda2")

        if error is not None:
            goal_handle.abort()
            result.position = actual_width
            result.effort = requested_effort
            result.stalled = True
            result.reached_goal = False
            self.get_logger().error(
                f"Grasp failed: target={width:.4f} m, "
                f"actual={actual_width:.4f} m, ctrl={actuator_ctrl:.3f}: {error}"
            )
            return result

        goal_handle.succeed()
        result.position = actual_width
        result.effort = requested_effort
        result.stalled = False
        result.reached_goal = True

        self.get_logger().info(
            f"Gripper grasped '{obj['name']}': target={width:.4f} m, "
            f"actual={actual_width:.4f} m, ctrl={actuator_ctrl:.3f}, "
            f"distance={distance:.4f} m"
        )
        return result

    def _gripper_cb(self, request, response):
        """Legacy Panda2 SetBool gripper endpoint."""
        if request.data:
            with self.lock:
                self.gripper_command = self.half_close_ctrl
            time.sleep(0.6)
            with self.lock:
                obj, distance, error = self._establish_nearest_grasp("panda2")
            response.success = error is None
            response.message = (f"Gripper closed and grasp established on '{obj['name']}' "
                                f"(distance {distance:.4f} m)" if error is None else error)
        else:
            with self.lock:
                released_name = self._release_current_object("panda2")
                self.gripper_command = self.open_ctrl
            time.sleep(0.4)
            response.success = True
            response.message = (f"Gripper opened and '{released_name}' released"
                                if released_name else "Gripper opened")
        return response

    def destroy_node(self):
        self.running = False

        if hasattr(self, "sim_thread"):
            self.sim_thread.join(timeout=1.0)

        if self.viewer is not None:
            self.viewer.close()

        self.action_server.destroy()
        self.gripper_action_server.destroy()
        self.p1_action_server.destroy()
        self.p1_gripper_action_server.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoMoveItBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

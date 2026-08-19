import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


class MujocoMoveItBridge(Node):
    ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
    STATE_JOINTS = ARM_JOINTS + ["panda_finger_joint1"]
    MJ_ARM_JOINTS = [f"panda2_joint{i}" for i in range(1, 8)]
    MJ_STATE_JOINTS = MJ_ARM_JOINTS + ["panda2_finger_joint1"]

    def __init__(self):
        super().__init__("mujoco_moveit_bridge")

        self.declare_parameter("scene_xml", "my_scene.xml")
        self.declare_parameter("panda_xml", "panda.xml")
        self.declare_parameter("viewer", True)
        self.declare_parameter("realtime_factor", 1.0)
        self.declare_parameter("gripper_close_ratio", 0.8)
        self.declare_parameter("trajectory_time_scale", 1.0)

        scene_xml = self.get_parameter("scene_xml").value
        panda_xml = self.get_parameter("panda_xml").value
        self.use_viewer = bool(self.get_parameter("viewer").value)
        self.realtime_factor = float(self.get_parameter("realtime_factor").value)
        self.close_ratio = float(self.get_parameter("gripper_close_ratio").value)
        self.trajectory_time_scale = float(self.get_parameter("trajectory_time_scale").value)

        self.model, self.data = self._build_model(scene_xml, panda_xml)
        self.lock = threading.RLock()

        self.running = True
        self.grasped = False
        self.arm_hold = True
        self.trajectory_active = False
        self.lpart_offset = np.zeros(3)
        self.last_lpart_pos = None

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

        # L_part
        self.lpart_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "l_part")
        self.lpart_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "l_part_free")
        self.lpart_qpos = self.model.jnt_qposadr[self.lpart_joint]
        self.lpart_qvel = self.model.jnt_dofadr[self.lpart_joint]

        # Initial state
        self.q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
        self.L_PART_INITIAL_POS = np.array([0.25, 0.1, 0.80], dtype=float)
        self.L_PART_INITIAL_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self._initialize_state()

        # Commands
        lo, hi = self.model.actuator_ctrlrange[self.finger_act]
        self.open_ctrl = hi
        self.close_ctrl = lo
        self.half_close_ctrl = lo + self.close_ratio * (hi - lo)

        self.q_command = self.q0.copy()
        self.v_command = np.zeros(7, dtype=float)
        self.gripper_command = self.open_ctrl

        with self.lock:
            self.data.qpos[self.arm_qadr] = self.q_command
            self.data.qvel[self.arm_vadr] = 0.0
            self.arm_hold = True
            self.trajectory_active = False
            mujoco.mj_forward(self.model, self.data)

        # ROS
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 20)
        self.center_pub = self.create_publisher(PointStamped, "/mujoco/fingertip_center", 20)

        self.gripper_srv = self.create_service(
            SetBool, "/mujoco/gripper", self._gripper_cb,
            callback_group=ReentrantCallbackGroup()
        )
        self.settle_srv = self.create_service(
            Trigger, "/mujoco/wait_until_stopped", self._wait_until_stopped_cb,
            callback_group=ReentrantCallbackGroup()
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

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if self.use_viewer else None
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = [0.0, 0.15, 0.8]
            self.viewer.cam.distance = 1.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30

        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("MuJoCo bridge ready")
        self.get_logger().info("Action: /panda_arm_controller/follow_joint_trajectory")
        self.get_logger().info("Services: /mujoco/gripper, /mujoco/wait_until_stopped")
        self.get_logger().info(
            "Joint state mapping: panda_joint1..7 + panda_finger_joint1 "
            "<-> panda2_joint1..7 + panda2_finger_joint1"
        )
        self.get_logger().info(
            f"Kinematic MoveIt trajectory following enabled, time scale={self.trajectory_time_scale:.2f}"
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

    def _initialize_state(self):
        for prefix in ["panda1", "panda2"]:
            for i in range(1, 8):
                jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint{i}")
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.q0[i - 1]

        self.data.qvel[:] = 0.0
        self.data.qpos[self.lpart_qpos:self.lpart_qpos + 3] = self.L_PART_INITIAL_POS
        self.data.qpos[self.lpart_qpos + 3:self.lpart_qpos + 7] = self.L_PART_INITIAL_QUAT
        self.data.qvel[self.lpart_qvel:self.lpart_qvel + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.last_lpart_pos = self.data.xpos[self.lpart_body].copy()

    def _finger_center(self):
        lp = np.mean(self.data.geom_xpos[self.left_geoms], axis=0)
        rp = np.mean(self.data.geom_xpos[self.right_geoms], axis=0)
        return (lp + rp) / 2.0

    def _update_lpart(self):
        if self.grasped:
            center = self._finger_center()
            pos = center + self.lpart_offset
            self.data.qpos[self.lpart_qpos:self.lpart_qpos + 3] = pos
            self.last_lpart_pos = pos.copy()
        else:
            if self.last_lpart_pos is None:
                self.last_lpart_pos = self.L_PART_INITIAL_POS.copy()
            self.data.qpos[self.lpart_qpos:self.lpart_qpos + 3] = self.last_lpart_pos

        self.data.qpos[self.lpart_qpos + 3:self.lpart_qpos + 7] = self.L_PART_INITIAL_QUAT
        self.data.qvel[self.lpart_qvel:self.lpart_qvel + 6] = 0.0

    def _simulation_loop(self):
        dt = float(self.model.opt.timestep)
        publish_period = 1.0 / 100.0
        viewer_period = 1.0 / 60.0
        last_pub = 0.0
        last_viewer = 0.0

        while self.running and rclpy.ok():
            tic = time.monotonic()

            with self.lock:
                # MoveIt trajectory directly defines Panda2 state.
                self.data.qpos[self.arm_qadr] = self.q_command
                self.data.qvel[self.arm_vadr] = self.v_command if self.trajectory_active else 0.0

                self.data.ctrl[self.finger_act] = self.gripper_command

                # Keep MuJoCo physics for gripper / scene, but arm pose is prescribed.
                mujoco.mj_step(self.model, self.data)

                # mj_step may modify arm qpos/qvel, overwrite again to exactly follow MoveIt.
                self.data.qpos[self.arm_qadr] = self.q_command
                self.data.qvel[self.arm_vadr] = self.v_command if self.trajectory_active else 0.0

                self._update_lpart()
                mujoco.mj_forward(self.model, self.data)

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

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.STATE_JOINTS)
        msg.position = self.data.qpos[self.state_qadr].astype(float).tolist()
        msg.velocity = self.data.qvel[self.state_vadr].astype(float).tolist()
        self.joint_pub.publish(msg)

        center = self._finger_center()
        p = PointStamped()
        p.header.stamp = msg.header.stamp
        p.header.frame_id = "mujoco_world"
        p.point.x, p.point.y, p.point.z = map(float, center)
        self.center_pub.publish(p)

    def _goal_callback(self, goal_request):
        names = list(goal_request.trajectory.joint_names)

        if set(names) != set(self.ARM_JOINTS):
            self.get_logger().error(f"Unexpected trajectory joints: {names}")
            return GoalResponse.REJECT

        if not goal_request.trajectory.points:
            return GoalResponse.REJECT

        with self.lock:
            if self.trajectory_active:
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

    def _wait_until_stopped_cb(self, request, response):
        del request

        with self.lock:
            q = self.data.qpos[self.arm_qadr].copy()
            v = self.data.qvel[self.arm_vadr].copy()
            qerr = float(np.max(np.abs(q - self.q_command)))
            vmax = float(np.max(np.abs(v)))
            active = self.trajectory_active

        if active:
            response.success = False
            response.message = (
                f"Robot still executing, vmax={vmax:.6f} rad/s, qerr={qerr:.6f} rad"
            )
        else:
            response.success = True
            response.message = (
                f"Robot stopped, vmax={vmax:.6f} rad/s, qerr={qerr:.6f} rad"
            )

        return response

    def _gripper_cb(self, request, response):
        if request.data:
            with self.lock:
                self.gripper_command = self.half_close_ctrl

            time.sleep(0.6)

            with self.lock:
                center = self._finger_center()
                lpart_pos = self.data.xpos[self.lpart_body].copy()
                self.lpart_offset = lpart_pos - center
                self.grasped = True
                self.last_lpart_pos = lpart_pos.copy()

            response.success = True
            response.message = "Gripper closed and grasp established"

        else:
            with self.lock:
                if self.grasped:
                    self.last_lpart_pos = self.data.xpos[self.lpart_body].copy()

                self.grasped = False
                self.gripper_command = self.open_ctrl

            time.sleep(0.4)

            response.success = True
            response.message = "Gripper opened and object released"

        return response

    def destroy_node(self):
        self.running = False

        if hasattr(self, "sim_thread"):
            self.sim_thread.join(timeout=1.0)

        if self.viewer is not None:
            self.viewer.close()

        self.action_server.destroy()
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
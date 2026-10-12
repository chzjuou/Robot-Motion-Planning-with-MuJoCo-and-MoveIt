import math
import time
import json
import os

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from panda_moveit_planner.srv import PlanToPose


class HandoverTask(Node):
    def __init__(self):
        super().__init__("handover_task")
        for n, v in {
            "arm": "panda1",  # receiving/approaching arm: panda1 or panda2
            "handover_x": 0.0,
            "handover_y": 0.10,
            "handover_z": 0.90,
            "approach_x": 0.0,
            "approach_y": 0.0,
            "approach_z": 1.0,
            "approach_distance": 0.05,
            "x_rotation_deg": 0.0,
            "panda1_x_rotation_deg": 0.0,  # legacy fallback
            "receiver_grasp_width": 0.037,
            "giver_open_width": 0.08,
            "giver_retreat_height": 0.05,
            "velocity_scale": 0.35,
            "acceleration_scale": 0.25,
        }.items():
            self.declare_parameter(n, v)

        self.orientation_reference_file = os.environ.get(
            "PANDA_TASK_ORIENTATION_REFERENCE",
            "/tmp/panda_task_initial_orientation.json",
        )
        self.sequence_base_orientation = None
        self.p1_tcp_pose = None
        self.create_subscription(PoseStamped, "/mujoco/panda1_tcp_pose",
                                 self._p1_tcp_cb, 20)

        self.p2_plan = self.create_client(PlanToPose, "/panda2/plan_to_pose")
        self.p2_exec = self.create_client(Trigger, "/panda2/execute_next_plan")
        self.p2_clear = self.create_client(Trigger, "/panda2/clear_plan_queue")

        self.p1_plan = self.create_client(PlanToPose, "/panda1/plan_to_pose")
        self.p1_exec = self.create_client(Trigger, "/panda1/execute_next_plan")
        self.p1_clear = self.create_client(Trigger, "/panda1/clear_plan_queue")

        self.p2_relz_plan = self.create_client(
            PlanToPose, "/mujoco/panda2_plan_relative_z")
        self.p2_relz_exec = self.create_client(
            Trigger, "/mujoco/panda2_execute_relative_z")

        self.p1_grip = ActionClient(
            self, GripperCommand, "/mujoco/panda1_gripper_command")
        self.p2_grip = ActionClient(
            self, GripperCommand, "/mujoco/gripper_command")


    def _p1_tcp_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        self.p1_tcp_pose = ([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])

    def p(self, n):
        return self.get_parameter(n).value


    def wait_all(self):
        arm = str(self.p("arm")).lower()
        for name, c in ([
            ("Panda1 plan", self.p1_plan),
            ("Panda1 execute", self.p1_exec),
            ("Panda1 clear", self.p1_clear),
            ("Panda2 relative-Z plan", self.p2_relz_plan),
            ("Panda2 relative-Z execute", self.p2_relz_exec),
        ] if arm == "panda1" else [
            ("Panda2 plan", self.p2_plan),
            ("Panda2 execute", self.p2_exec),
            ("Panda2 clear", self.p2_clear),
            ("Panda1 plan", self.p1_plan),
            ("Panda1 execute", self.p1_exec),
            ("Panda1 clear", self.p1_clear),
        ]):
            if not c.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Missing service: {name}")

        for name, a in [
            ("Panda1 gripper", self.p1_grip),
            ("Panda2 gripper", self.p2_grip),
        ]:
            if not a.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"Missing action: {name}")


    def trigger(self, client, label):
        f = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, f)
        r = f.result()
        if r is None or not r.success:
            raise RuntimeError(
                f"{label} failed: {getattr(r, 'message', '')}")

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

    @classmethod
    def rotate_world_x(cls, base_q, angle_deg):
        a = math.radians(float(angle_deg))
        qx = [math.sin(a/2.0), 0.0, 0.0, math.cos(a/2.0)]
        q = cls.quat_multiply(qx, base_q)
        n = math.sqrt(sum(v*v for v in q))
        if n <= 1e-12:
            raise RuntimeError("Invalid handover target quaternion")
        return [v/n for v in q]

    def load_sequence_reference(self):
        path = self.orientation_reference_file
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Bridge-start GLOBAL Q0 not found: {path}. "
                "Start/restart the modified bridge before running handover."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        q = [float(data[k]) for k in ("qx", "qy", "qz", "qw")]
        n = math.sqrt(sum(v*v for v in q))
        if n <= 1e-12:
            raise RuntimeError("Bridge-start GLOBAL Q0 is invalid")
        q = [v/n for v in q]
        self.get_logger().info(f"Loaded BRIDGE-START GLOBAL Q0 = {q}")
        return q

    def queue_p1(self, pos, linear, orientation=None):
        req = PlanToPose.Request()
        req.target.header.frame_id = "world"
        req.target.pose.position.x = float(pos[0])
        req.target.pose.position.y = float(pos[1])
        req.target.pose.position.z = float(pos[2])

        # Bridge planner now expects an ABSOLUTE WORLD hand/TCP orientation.
        # Zero degrees means exactly the bridge-start GLOBAL Q0.
        q = orientation if orientation is not None else self.rotate_world_x(
            self.sequence_base_orientation,
            self.p("x_rotation_deg"),
        )
        req.target.pose.orientation.x = q[0]
        req.target.pose.orientation.y = q[1]
        req.target.pose.orientation.z = q[2]
        req.target.pose.orientation.w = q[3]

        req.linear = bool(linear)
        req.execute = False
        req.velocity_scale = float(self.p("velocity_scale"))
        req.acceleration_scale = float(self.p("acceleration_scale"))

        f = self.p1_plan.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        r = f.result()
        if r is None or not r.success:
            raise RuntimeError(
                f"Panda1 planning failed: {getattr(r, 'message', '')}")

    def queue_p2_lift(self):
        # Bridge interprets target.position.z as RELATIVE world +Z displacement.
        req = PlanToPose.Request()
        req.target.header.frame_id = "world"
        req.target.pose.position.z = abs(float(self.p("giver_retreat_height")))
        req.target.pose.orientation.w = 1.0
        req.linear = True
        req.execute = False
        req.velocity_scale = float(self.p("velocity_scale"))
        req.acceleration_scale = float(self.p("acceleration_scale"))

        f = self.p2_relz_plan.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        r = f.result()
        if r is None or not r.success:
            raise RuntimeError(
                f"Panda2 +Z preplanning failed: {getattr(r, 'message', '')}")
        self.get_logger().info(getattr(r, "message", "Panda2 +Z ready"))

    def grip(self, client, width, label, grasp=False):
        goal = GripperCommand.Goal()
        goal.command.position = float(width)
        goal.command.max_effort = 1.0 if grasp else 0.0

        f = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f)
        handle = f.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"{label} rejected")

        f = handle.get_result_async()
        rclpy.spin_until_future_complete(self, f)
        wrapped = f.result()
        if wrapped is None or wrapped.status != 4:
            raise RuntimeError(f"{label} failed (status={getattr(wrapped, 'status', None)})")
        if grasp and not wrapped.result.reached_goal:
            raise RuntimeError(f"{label}: receiver did not attach object (reached_goal=False)")

    def queue_p2(self, pos, linear):
        req = PlanToPose.Request()
        req.target.header.frame_id = "world"
        req.target.pose.position.x = float(pos[0])
        req.target.pose.position.y = float(pos[1])
        req.target.pose.position.z = float(pos[2])
        q = self.rotate_world_x(self.sequence_base_orientation,
                                self.p("x_rotation_deg"))
        req.target.pose.orientation.x = q[0]
        req.target.pose.orientation.y = q[1]
        req.target.pose.orientation.z = q[2]
        req.target.pose.orientation.w = q[3]
        req.linear = bool(linear)
        req.execute = False
        req.velocity_scale = float(self.p("velocity_scale"))
        req.acceleration_scale = float(self.p("acceleration_scale"))
        f = self.p2_plan.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        r = f.result()
        if r is None or not r.success:
            raise RuntimeError(f"Panda2 planning failed: {getattr(r, 'message', '')}")

    def run(self):
        arm = str(self.p("arm")).lower().strip()
        if arm not in ("panda1", "panda2"):
            raise RuntimeError("arm must be panda1 or panda2")
        self.get_logger().info(f"Regrasp receiving arm: {arm}")
        self.wait_all()
        distance = abs(float(self.p("approach_distance")))
        if distance <= 1e-9:
            self.get_logger().info(
                "approach_distance=0: skip RECEIVER planning/motion; "
                "retain GIVER retreat after transfer")
            # Only preplan the giver's departure. The receiver stays at its
            # current pose and closes its gripper without any approach plan.
            if arm == "panda1":
                self.queue_p2_lift()
            else:
                self.trigger(self.p1_clear, "clear Panda1 retreat queue")
                deadline = time.monotonic() + 5.0
                while self.p1_tcp_pose is None and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
                if self.p1_tcp_pose is None:
                    raise RuntimeError("No /mujoco/panda1_tcp_pose for retreat")
                p1_position, p1_orientation = self.p1_tcp_pose
                p1_retreat = [p1_position[0], p1_position[1],
                              p1_position[2] + abs(float(self.p("giver_retreat_height")))]
                self.queue_p1(p1_retreat, True, orientation=p1_orientation)
            self.get_logger().info("GIVER RETREAT PLANNING SUCCEEDED")
            self.grip(self.p1_grip if arm == "panda1" else self.p2_grip,
                      self.p("receiver_grasp_width"),
                      f"{arm} receive/attach", grasp=True)
            time.sleep(0.3)
            self.grip(self.p2_grip if arm == "panda1" else self.p1_grip,
                      self.p("giver_open_width"), "Giver open/detach")
            time.sleep(0.3)
            if arm == "panda1":
                self.trigger(self.p2_relz_exec, "Panda2 +Z retreat")
            else:
                self.trigger(self.p1_exec, "Panda1 +Z retreat")
            self.get_logger().info("REGRASP SEQUENCE FINISHED (zero approach)")
            return
        self.sequence_base_orientation = self.load_sequence_reference()

        # The approach axis is defined in the END-EFFECTOR local frame,
        # then transformed into world coordinates by the exact target quaternion.
        # This keeps translation aligned with the gripper orientation.
        q = self.rotate_world_x(self.sequence_base_orientation,
                                self.p("x_rotation_deg"))
        ax, ay, az = (float(self.p(k)) for k in
                      ("approach_x", "approach_y", "approach_z"))
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-9:
            raise RuntimeError("local approach axis cannot be zero")
        vx, vy, vz = ax/norm, ay/norm, az/norm
        qx, qy, qz, qw = q
        # v_world = v_local + 2*qw*(q_xyz cross v_local)
        #                      + 2*(q_xyz cross (q_xyz cross v_local))
        tx = 2.0*(qy*vz-qz*vy)
        ty = 2.0*(qz*vx-qx*vz)
        tz = 2.0*(qx*vy-qy*vx)
        direction = [vx+qw*tx+(qy*tz-qz*ty),
                     vy+qw*ty+(qz*tx-qx*tz),
                     vz+qw*tz+(qx*ty-qy*tx)]
        final = [float(self.p(k)) for k in
                 ("handover_x", "handover_y", "handover_z")]
        distance = abs(float(self.p("approach_distance")))
        pre = [final[i] - distance*direction[i] for i in range(3)]
        self.get_logger().info(
            f"Approach world direction={direction}, pre={pre}, final={final}")

        # Preplan all required arm movements before moving either robot.
        if arm == "panda1":
            self.trigger(self.p1_clear, "clear Panda1 queue")
            self.queue_p1(pre, False)
            if distance > 1e-9:
                self.queue_p1(final, True)
            self.queue_p2_lift()
            self.get_logger().info("ALL PLANNING SUCCEEDED")
            self.trigger(self.p1_exec, "Panda1 pre-approach")
            if distance > 1e-9:
                self.trigger(self.p1_exec, "Panda1 linear approach")
            self.grip(self.p1_grip, self.p("receiver_grasp_width"),
                      "Panda1 receive/attach", grasp=True)
            time.sleep(0.3)
            self.grip(self.p2_grip, self.p("giver_open_width"),
                      "Panda2 open/detach")
            time.sleep(0.3)
            self.trigger(self.p2_relz_exec, "Panda2 +Z lift")
        else:
            # Reverse direction: Panda2 receives an object from Panda1.
            self.trigger(self.p2_clear, "clear Panda2 queue")
            self.queue_p2(pre, False)
            if distance > 1e-9:
                self.queue_p2(final, True)
            self.trigger(self.p1_clear, "clear Panda1 queue")
            # The releasing Panda1 must retreat from its OWN current TCP pose,
            # not from Panda2's handover target. Preserve Panda1 orientation.
            # Otherwise the old code teleports toward Panda2's approach pose.
            deadline = time.monotonic() + 5.0
            while self.p1_tcp_pose is None and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.p1_tcp_pose is None:
                raise RuntimeError("No /mujoco/panda1_tcp_pose for retreat")
            p1_position, p1_orientation = self.p1_tcp_pose
            p1_retreat = [p1_position[0], p1_position[1],
                          p1_position[2] + abs(float(self.p("giver_retreat_height")))]
            self.get_logger().info(
                f"Panda1 WORLD +Z retreat: {p1_position} -> {p1_retreat}; "
                f"keep orientation={p1_orientation}")
            self.queue_p1(p1_retreat, True, orientation=p1_orientation)
            self.get_logger().info("ALL PLANNING SUCCEEDED")
            self.trigger(self.p2_exec, "Panda2 pre-approach")
            if distance > 1e-9:
                self.trigger(self.p2_exec, "Panda2 linear approach")
            self.grip(self.p2_grip, self.p("receiver_grasp_width"),
                      "Panda2 receive/attach", grasp=True)
            time.sleep(0.3)
            self.grip(self.p1_grip, self.p("giver_open_width"),
                      "Panda1 open/old-owner release")
            time.sleep(0.3)
            self.trigger(self.p1_exec, "Panda1 +Z retreat")
        self.get_logger().info("REGRASP SEQUENCE FINISHED")


def main(args=None):
    rclpy.init(args=args)
    node = HandoverTask()
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(str(e))
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PointStamped, PoseStamped
from panda_moveit_planner.srv import PlanToPose
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

class ReturnTask(Node):
    def __init__(self):
        super().__init__("return_task")
        defaults = {
            "arm": "both", "retreat_distance": 0.05,
            "gripper_open_width": 0.08, "return_duration": 3.0,
            "velocity_scale": 0.35, "acceleration_scale": 0.25,
            "initial_j1": 0.0, "initial_j2": -0.785, "initial_j3": 0.0,
            "initial_j4": -2.356, "initial_j5": 0.0,
            "initial_j6": 1.571, "initial_j7": 0.785,
        }
        for k, v in defaults.items(): self.declare_parameter(k, v)

        self.p1_plan = self.create_client(PlanToPose, "/panda1/plan_to_pose")
        self.p1_exec = self.create_client(Trigger, "/panda1/execute_next_plan")
        self.p1_clear = self.create_client(Trigger, "/panda1/clear_plan_queue")
        self.p2_plan = self.create_client(PlanToPose, "/panda2/plan_to_pose")
        self.p2_exec = self.create_client(Trigger, "/panda2/execute_next_plan")
        self.p2_clear = self.create_client(Trigger, "/panda2/clear_plan_queue")
        self.p1_grip = ActionClient(self, GripperCommand, "/mujoco/panda1_gripper_command")
        self.p2_grip = ActionClient(self, GripperCommand, "/mujoco/gripper_command")
        self.p1_traj = ActionClient(self, FollowJointTrajectory, "/panda1_arm_controller/follow_joint_trajectory")
        self.p2_traj = ActionClient(self, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory")
        self.p1_center = None
        self.p1_tcp_pose = None
        self.p2_tcp_pose = None
        self.create_subscription(
            PointStamped, "/mujoco/panda1_fingertip_center", self._p1_center_cb, 20)
        self.create_subscription(
            PoseStamped, "/mujoco/panda1_tcp_pose", self._p1_tcp_cb, 20)
        self.create_subscription(
            PoseStamped, "/mujoco/panda2_tcp_pose", self._p2_tcp_cb, 20)

    def p(self, n): return self.get_parameter(n).value
    def _p1_center_cb(self, m): self.p1_center = [m.point.x, m.point.y, m.point.z]

    @staticmethod
    def _pose_from_msg(m):
        return (
            [m.pose.position.x, m.pose.position.y, m.pose.position.z],
            [m.pose.orientation.x, m.pose.orientation.y,
             m.pose.orientation.z, m.pose.orientation.w],
        )

    def _p1_tcp_cb(self, m): self.p1_tcp_pose = self._pose_from_msg(m)
    def _p2_tcp_cb(self, m): self.p2_tcp_pose = self._pose_from_msg(m)

    def arms(self):
        a = str(self.p("arm")).strip().lower()
        a = {"1":"panda1","p1":"panda1","2":"panda2","p2":"panda2","all":"both"}.get(a, a)
        if a not in ("panda1","panda2","both"):
            raise RuntimeError("arm must be panda1, panda2, or both")
        return a

    def wait_all(self, a):
        svcs, acts = [], []
        if a in ("panda1","both"):
            svcs += [self.p1_plan,self.p1_exec,self.p1_clear]
            acts += [self.p1_grip,self.p1_traj]
        if a in ("panda2","both"):
            svcs += [self.p2_plan,self.p2_exec,self.p2_clear]
            acts += [self.p2_grip,self.p2_traj]
        for c in svcs:
            while not c.wait_for_service(timeout_sec=.5): pass
        for c in acts:
            while not c.wait_for_server(timeout_sec=.5): pass
        end=time.monotonic()+5
        while time.monotonic()<end:
            need_p1 = a in ("panda1","both") and self.p1_tcp_pose is None
            need_p2 = a in ("panda2","both") and self.p2_tcp_pose is None
            if not need_p1 and not need_p2:
                break
            rclpy.spin_once(self, timeout_sec=.05)
        if a in ("panda1","both") and self.p1_tcp_pose is None:
            raise RuntimeError("No Panda1 TCP pose")
        if a in ("panda2","both") and self.p2_tcp_pose is None:
            raise RuntimeError("No Panda2 TCP pose")

    def trigger(self, c, label):
        f=c.call_async(Trigger.Request()); rclpy.spin_until_future_complete(self,f); r=f.result()
        if r is None or not r.success: raise RuntimeError(f"{label} failed: {getattr(r,'message','')}")
        self.get_logger().info(f"{label}: {r.message}")

    def open(self, c, label):
        g=GripperCommand.Goal(); g.command.position=float(self.p("gripper_open_width")); g.command.max_effort=0.0
        f=c.send_goal_async(g); rclpy.spin_until_future_complete(self,f); gh=f.result()
        if gh is None or not gh.accepted: raise RuntimeError(f"{label} open rejected")
        f=gh.get_result_async(); rclpy.spin_until_future_complete(self,f)
        self.get_logger().info(f"{label} opened")

    @staticmethod
    def retreat_along_tool_axis(position, quaternion, distance):
        """Retreat along the current TCP local -Z axis (away from the tool tip)."""
        x, y, z, w = (float(v) for v in quaternion)
        norm = (x*x + y*y + z*z + w*w) ** 0.5
        if norm < 1e-12:
            raise RuntimeError("Invalid TCP quaternion for retreat")
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        # Third column of the local-to-world rotation matrix.
        axis = [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
        return [float(position[i]) - distance*axis[i] for i in range(3)]

    def plan_p1_retreat(self):
        self.trigger(self.p1_clear,"Clear Panda1 queue")
        d=abs(float(self.p("retreat_distance")))
        p,q=self.p1_tcp_pose
        target=self.retreat_along_tool_axis(p,q,d)
        self.get_logger().info(f"P1 retreat along TCP local -Z: {p} -> {target}")
        req=PlanToPose.Request(); req.target.header.frame_id="world"
        req.target.pose.position.x=target[0]; req.target.pose.position.y=target[1]
        req.target.pose.position.z=target[2]
        # Absolute current WORLD TCP orientation: preserve handover orientation.
        req.target.pose.orientation.x=q[0]; req.target.pose.orientation.y=q[1]
        req.target.pose.orientation.z=q[2]; req.target.pose.orientation.w=q[3]
        req.linear=True; req.execute=False
        req.velocity_scale=float(self.p("velocity_scale")); req.acceleration_scale=float(self.p("acceleration_scale"))
        f=self.p1_plan.call_async(req); rclpy.spin_until_future_complete(self,f); x=f.result()
        if x is None or not x.success: raise RuntimeError(f"Panda1 retreat plan failed: {getattr(x,'message','')}")
        self.get_logger().info(x.message)

    def plan_p2_retreat(self):
        self.trigger(self.p2_clear,"Clear Panda2 queue")
        d=abs(float(self.p("retreat_distance")))
        p,q=self.p2_tcp_pose
        target=self.retreat_along_tool_axis(p,q,d)
        self.get_logger().info(f"P2 retreat along TCP local -Z: {p} -> {target}")
        req=PlanToPose.Request(); req.target.header.frame_id="world"
        req.target.pose.position.x=target[0]; req.target.pose.position.y=target[1]
        req.target.pose.position.z=target[2]
        req.target.pose.orientation.x=q[0]; req.target.pose.orientation.y=q[1]
        req.target.pose.orientation.z=q[2]; req.target.pose.orientation.w=q[3]
        req.linear=True; req.execute=False
        req.velocity_scale=float(self.p("velocity_scale")); req.acceleration_scale=float(self.p("acceleration_scale"))
        f=self.p2_plan.call_async(req); rclpy.spin_until_future_complete(self,f); x=f.result()
        if x is None or not x.success: raise RuntimeError(f"Panda2 retreat plan failed: {getattr(x,'message','')}")
        self.get_logger().info(x.message)

    def q0(self): return [float(self.p(f"initial_j{i}")) for i in range(1,8)]

    def go_q0(self, c, panda1):
        g=FollowJointTrajectory.Goal()
        prefix="panda1_joint" if panda1 else "panda_joint"
        g.trajectory.joint_names=[f"{prefix}{i}" for i in range(1,8)]
        pt=JointTrajectoryPoint(); pt.positions=self.q0()
        t=max(float(self.p("return_duration")),0.1); pt.time_from_start.sec=int(t)
        pt.time_from_start.nanosec=int(round((t-int(t))*1e9)); g.trajectory.points=[pt]
        f=c.send_goal_async(g); rclpy.spin_until_future_complete(self,f); gh=f.result()
        if gh is None or not gh.accepted: raise RuntimeError("Return-to-initial rejected")
        f=gh.get_result_async(); rclpy.spin_until_future_complete(self,f)

    def run(self):
        a=self.arms(); self.get_logger().info(f"Return arm={a}"); self.wait_all(a)
        # Retreats are preplanned before any arm motion.
        if a in ("panda1","both"): self.plan_p1_retreat()
        if a in ("panda2","both"): self.plan_p2_retreat()
        self.get_logger().info("RETREAT PLANNING SUCCEEDED")
        # Open first. Panda2 open also detaches any currently grasped object.
        if a in ("panda1","both"): self.open(self.p1_grip,"Panda1")
        if a in ("panda2","both"): self.open(self.p2_grip,"Panda2")
        # Retreat along each robot TCP local -Z axis.
        if a in ("panda1","both"): self.trigger(self.p1_exec,"Panda1 tool-axis retreat")
        if a in ("panda2","both"): self.trigger(self.p2_exec,"Panda2 tool-axis retreat")
        # Return to the configured initial joint state.
        if a in ("panda1","both"): self.go_q0(self.p1_traj,True)
        if a in ("panda2","both"): self.go_q0(self.p2_traj,False)
        self.get_logger().info("RETURN TASK COMPLETE")

def main(args=None):
    rclpy.init(args=args); n=ReturnTask()
    try: n.run()
    except Exception as e:
        n.get_logger().error(str(e)); n.destroy_node(); rclpy.shutdown(); raise SystemExit(1)
    n.destroy_node(); rclpy.shutdown()

if __name__=="__main__": main()

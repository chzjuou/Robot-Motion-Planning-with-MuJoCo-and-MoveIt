#!/usr/bin/env python3
"""Move one selected Panda arm to a world-frame TCP pose, optionally command gripper."""
import math
import json
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from panda_moveit_planner.srv import PlanToPose


class MoveTask(Node):
    def __init__(self):
        super().__init__('move_task')
        defaults = {
            'arm': 'panda2',
            'move_x': 0.25,
            'move_y': 0.0,
            'move_z': 0.95,
            'move_rx_deg': 0.0,
            'move_ry_deg': 0.0,
            'planning_mode': 'ompl',  # ompl / linear / joint_relative
            'joint_index': 1,
            'joint_delta_deg': 0.0,
            'joint_duration': 3.0,
            'velocity_scale': 0.35,
            'acceleration_scale': 0.25,
            'gripper_action': 'none',  # none / open / close
            'gripper_width': 0.08,    # total finger opening in meters
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.arm = str(self.get_parameter('arm').value).strip().lower()
        if self.arm not in ('panda1', 'panda2'):
            raise ValueError('arm must be panda1 or panda2')
        self.orientation_reference_file = os.environ.get(
            "PANDA_TASK_ORIENTATION_REFERENCE",
            "/tmp/panda_task_initial_orientation.json",
        )
        self.tcp_pose = None
        self.create_subscription(PoseStamped, f'/mujoco/{self.arm}_tcp_pose', self.tcp_cb, 20)
        self.plan = self.create_client(PlanToPose, f'/{self.arm}/plan_to_pose')
        self.execute = self.create_client(Trigger, f'/{self.arm}/execute_next_plan')
        self.clear = self.create_client(Trigger, f'/{self.arm}/clear_plan_queue')
        gripper_topic = ('/mujoco/panda1_gripper_command' if self.arm == 'panda1'
                         else '/mujoco/gripper_command')
        self.gripper = ActionClient(self, GripperCommand, gripper_topic)
        self.direct_joint_pub = self.create_publisher(
            Float64MultiArray, '/mujoco/direct_joint_relative', 10)
        self.direct_joint_status = {}
        self.create_subscription(
            Float64MultiArray, '/mujoco/direct_joint_status',
            self.direct_joint_status_cb, 10)


    def p(self, name):
        return self.get_parameter(name).value

    def tcp_cb(self, msg):
        self.tcp_pose = msg.pose

    def call_trigger(self, client, label):
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f'{label} failed: {getattr(response, "message", "no response")}')

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

    def target_orientation(self):
        # Same reference and WORLD X -> WORLD Y rotation convention as pick_place.py.
        # Do not overwrite or recapture bridge-start Q0.
        with open(self.orientation_reference_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        base = [float(data[k]) for k in ('qx', 'qy', 'qz', 'qw')]
        n = math.sqrt(sum(v*v for v in base))
        if n <= 1e-12:
            raise RuntimeError('Invalid bridge-start Q0 quaternion')
        base = [v/n for v in base]
        rx = math.radians(float(self.p('move_rx_deg')))
        ry = math.radians(float(self.p('move_ry_deg')))
        qx = [math.sin(rx/2), 0.0, 0.0, math.cos(rx/2)]
        qy = [0.0, math.sin(ry/2), 0.0, math.cos(ry/2)]
        q = self.quat_multiply(self.quat_multiply(qy, qx), base)
        n = math.sqrt(sum(v*v for v in q))
        return [v/n for v in q]

    def command_gripper(self, mode):
        width = float(self.p('gripper_width'))
        if not math.isfinite(width) or not 0.0 <= width <= 0.08:
            raise ValueError('gripper_width must be between 0.0 and 0.08 m')
        while rclpy.ok() and not self.gripper.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {self.arm} gripper action...')
        if not rclpy.ok():
            raise RuntimeError('ROS shutdown while waiting for gripper')
        goal = GripperCommand.Goal()
        goal.command.position = width
        # Bridge convention: 0 = OPEN/release; 1 = CLOSE/grasp.
        goal.command.max_effort = 0.0 if mode == 'open' else 1.0
        future = self.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f'Gripper {mode} goal rejected')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None or wrapped.status != 4:
            raise RuntimeError(f'Gripper {mode} did not succeed')
        if mode == 'close' and not wrapped.result.reached_goal:
            raise RuntimeError('Gripper close/grasp failed')
        if mode == 'open' and not (wrapped.result.reached_goal or
                                  wrapped.result.position >= width * 0.90):
            raise RuntimeError('Gripper open failed to reach requested width')
        self.get_logger().info(f'{self.arm} gripper {mode} completed, width={width:.4f} m')

    def direct_joint_status_cb(self, msg):
        if len(msg.data) >= 2:
            self.direct_joint_status[int(msg.data[0])] = int(msg.data[1])

    def move_joint_relative(self):
        index = int(self.p('joint_index'))
        delta = float(self.p('joint_delta_deg'))
        duration = float(self.p('joint_duration'))
        if index not in range(1, 8) or not math.isfinite(delta) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('joint_index must be 1..7, delta finite, duration > 0')
        seq = int(time.time_ns() % 1000000000000)
        cmd = Float64MultiArray()
        cmd.data = [float(seq), float(1 if self.arm == 'panda1' else 2),
                    float(index), delta, duration]
        # Publish until the bridge acknowledges receipt, then wait for completion.
        deadline = time.monotonic() + 10.0
        next_send = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            if time.monotonic() >= next_send:
                self.direct_joint_pub.publish(cmd)
                next_send = time.monotonic() + 0.2
            rclpy.spin_once(self, timeout_sec=0.05)
            if seq in self.direct_joint_status:
                break
        if seq not in self.direct_joint_status:
            raise RuntimeError('Direct joint command not acknowledged; restart updated bridge')
        if self.direct_joint_status[seq] < 0:
            raise RuntimeError('Direct joint command rejected by bridge')
        deadline = time.monotonic() + duration + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            status = self.direct_joint_status.get(seq, 0)
            if status == 1:
                self.get_logger().info(
                    f'{self.arm} joint {index} moved {delta:+.3f} deg; other joints unchanged')
                return
            if status < 0:
                raise RuntimeError('Direct joint motion failed')
        raise RuntimeError('Direct joint motion completion timeout')

    def run(self):
        planning_mode = str(self.p('planning_mode')).strip().lower()
        if planning_mode not in ('ompl', 'linear', 'joint_relative'):
            raise ValueError('planning_mode must be ompl, linear or joint_relative')
        mode = str(self.p('gripper_action')).strip().lower()
        if mode not in ('none', 'open', 'close'):
            raise ValueError('gripper_action must be none, open, or close')
        if mode != 'none':
            width = float(self.p('gripper_width'))
            if not math.isfinite(width) or not 0.0 <= width <= 0.08:
                raise ValueError('gripper_width must be between 0.0 and 0.08 m')
        if planning_mode == 'joint_relative':
            self.move_joint_relative()
            if mode != 'none':
                self.command_gripper(mode)
            return

        for name, client in [('plan', self.plan), ('execute', self.execute), ('clear', self.clear)]:
            while rclpy.ok() and not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {self.arm} {name} service...')
        while rclpy.ok() and self.tcp_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not rclpy.ok():
            raise RuntimeError('ROS shutdown before robot became ready')

        # Absolute WORLD TCP orientation = Ry(move_ry_deg) * Rx(move_rx_deg) * Q0.
        # Angles are relative to the bridge-start orientation, NOT the current pose.
        q = self.target_orientation()
        def plan_pose(position, orientation, linear, label):
            request = PlanToPose.Request()
            request.target.header.frame_id = 'world'
            request.target.header.stamp = self.get_clock().now().to_msg()
            request.target.pose.position.x = float(position[0])
            request.target.pose.position.y = float(position[1])
            request.target.pose.position.z = float(position[2])
            request.target.pose.orientation.x = float(orientation[0])
            request.target.pose.orientation.y = float(orientation[1])
            request.target.pose.orientation.z = float(orientation[2])
            request.target.pose.orientation.w = float(orientation[3])
            request.linear = bool(linear)
            request.execute = False
            request.velocity_scale = float(self.p('velocity_scale'))
            request.acceleration_scale = float(self.p('acceleration_scale'))
            future = self.plan.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result is None or not result.success:
                raise RuntimeError(f'{label} planning failed: {getattr(result, "message", "no response")}')
            self.call_trigger(self.execute, f'Execute {label}')

        target_pos = [float(self.p(k)) for k in ('move_x', 'move_y', 'move_z')]
        self.call_trigger(self.clear, 'Clear plan queue')
        if planning_mode == 'linear':
            # Like pick_place approach: the Cartesian translation must start
            # with its target orientation already established.  The bridge's
            # linear IK uses the requested orientation at EVERY XYZ waypoint.
            current_q = [float(getattr(self.tcp_pose.orientation, k))
                         for k in ('x', 'y', 'z', 'w')]
            current_n = math.sqrt(sum(v*v for v in current_q))
            if current_n <= 1e-12:
                raise RuntimeError('Invalid current TCP quaternion')
            current_q = [v/current_n for v in current_q]
            alignment = abs(sum(a*b for a, b in zip(current_q, q)))
            if alignment < math.cos(math.radians(1.0)/2.0):
                current_pos = [float(getattr(self.tcp_pose.position, k))
                               for k in ('x', 'y', 'z')]
                self.get_logger().info(
                    'Linear: first align TCP orientation at current XYZ '
                    '(joint-space motion), then translate in a straight line')
                plan_pose(current_pos, q, False, 'orientation alignment')
            # Reuse the same target orientation for the entire Cartesian leg.
            self.get_logger().info(f'{self.arm} LINEAR TCP translation to {target_pos}')
            plan_pose(target_pos, q, True, 'linear translation')
        else:
            self.get_logger().info(f'{self.arm} OMPL/joint-space move to {target_pos}')
            plan_pose(target_pos, q, False, 'joint-space move')
        self.get_logger().info(f'{self.arm} moved to requested world TCP pose with Q0-relative X/Y rotations')
        if mode != 'none':
            self.command_gripper(mode)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MoveTask()
        node.run()
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

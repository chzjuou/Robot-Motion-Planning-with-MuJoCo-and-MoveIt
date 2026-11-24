#!/usr/bin/env python3
"""Advance, orbit TCP about attached part center axis with Z yaw, detach, retreat."""
import math
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from panda_moveit_planner.srv import PlanToPose


class RotateTask(Node):
    def __init__(self):
        super().__init__('rotate_task')
        defaults = {
            'arm': 'panda2',
            'advance_direction': '-z',
            'advance_distance': 0.01,
            'rotate_angle_deg': 90.0,
            'velocity_scale': 0.20,
            'acceleration_scale': 0.15,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.arm = str(self.p('arm')).strip().lower()
        if self.arm not in ('panda1', 'panda2'):
            raise ValueError('arm must be panda1 or panda2')
        self.tcp_pose = None
        self.create_subscription(PoseStamped, f'/mujoco/{self.arm}_tcp_pose', self._tcp_cb, 20)
        self.plan = self.create_client(PlanToPose, f'/{self.arm}/plan_to_pose')
        self.execute = self.create_client(Trigger, f'/{self.arm}/execute_next_plan')
        self.clear = self.create_client(Trigger, f'/{self.arm}/clear_plan_queue')
        self.axis = self.create_client(Trigger, '/mujoco/hex_key_attached_axis')
        self.orbit_plan = self.create_client(PlanToPose, f'/mujoco/{self.arm}_plan_continuous_orbit')
        self.detach = self.create_client(Trigger, '/mujoco/hex_key_tip_detach')
        self.stopped = self.create_client(Trigger, '/mujoco/wait_until_stopped')

    def p(self, key):
        return self.get_parameter(key).value

    def _tcp_cb(self, msg):
        self.tcp_pose = msg.pose

    @staticmethod
    def quat_multiply(a, b):
        x, y, z, w = a
        X, Y, Z, W = b
        return [w*X+x*W+y*Z-z*Y,
                w*Y-x*Z+y*W+z*X,
                w*Z+x*Y-y*X+z*W,
                w*W-x*X-y*Y-z*Z]

    @staticmethod
    def normalized(q):
        n = math.sqrt(sum(x*x for x in q))
        if n < 1e-12:
            raise RuntimeError('Invalid TCP quaternion')
        return [x/n for x in q]

    @staticmethod
    def pose_components(pose):
        pos = [float(getattr(pose.position, a)) for a in 'xyz']
        q = [float(getattr(pose.orientation, a)) for a in 'xyzw']
        return pos, RotateTask.normalized(q)

    def trigger(self, client, label):
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f'{label}: {getattr(result, "message", "no response")}')
        return result.message

    def plan_and_execute(self, pos, q, linear, label):
        req = PlanToPose.Request()
        req.target.header.frame_id = 'world'
        req.target.header.stamp = self.get_clock().now().to_msg()
        req.target.pose.position.x, req.target.pose.position.y, req.target.pose.position.z = map(float, pos)
        req.target.pose.orientation.x, req.target.pose.orientation.y, req.target.pose.orientation.z, req.target.pose.orientation.w = map(float, q)
        req.linear = bool(linear)
        req.execute = False
        req.velocity_scale = float(self.p('velocity_scale'))
        req.acceleration_scale = float(self.p('acceleration_scale'))
        f = self.plan.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        result = f.result()
        if result is None or not result.success:
            raise RuntimeError(f'{label} planning failed: {getattr(result, "message", "no response")}')
        self.trigger(self.execute, f'{label} execute')
        self.trigger(self.stopped, f'{label} settle')
        self.get_logger().info(f'{label}: completed')

    def run(self):
        direction = str(self.p('advance_direction')).strip().lower().replace(' ', '')
        axes = {'+x': (1,0,0), '-x': (-1,0,0),
                '+y': (0,1,0), '-y': (0,-1,0),
                '+z': (0,0,1), '-z': (0,0,-1)}
        if direction not in axes:
            raise ValueError('advance_direction must be +x, -x, +y, -y, +z, or -z')
        distance = float(self.p('advance_distance'))
        angle = float(self.p('rotate_angle_deg'))
        if not all(map(math.isfinite, [distance, angle])):
            raise ValueError('All motion parameters must be finite')
        velocity = float(self.p('velocity_scale'))
        acceleration = float(self.p('acceleration_scale'))
        if not (0.0 < velocity <= 1.0 and 0.0 < acceleration <= 1.0):
            raise ValueError('velocity_scale and acceleration_scale must be in (0, 1]')
        if distance < 0:
            raise ValueError('advance_distance must be >= 0')
        for client, label in [(self.plan, 'plan'), (self.execute, 'execute'),
                              (self.clear, 'clear'), (self.axis, 'attached axis'), (self.orbit_plan, 'continuous orbit'), (self.detach, 'detach'),
                              (self.stopped, 'wait_until_stopped')]:
            while rclpy.ok() and not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {label} service...')
        while rclpy.ok() and self.tcp_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not rclpy.ok():
            raise RuntimeError('ROS shutdown before task started')
        start, q0 = self.pose_components(self.tcp_pose)
        displacement = [distance*v for v in axes[direction]]
        advanced = [start[i]+displacement[i] for i in range(3)]
        self.trigger(self.clear, 'Clear plan queue')
        if distance > 0:
            self.plan_and_execute(advanced, q0, True, 'Advance')
        # Read the real attached object's WORLD center from Bridge, not YAML.
        axis_message = self.trigger(self.axis, 'Get attached part axis')
        axis = json.loads(axis_message)
        axis_x, axis_y = float(axis['axis_x']), float(axis['axis_y'])
        if not all(map(math.isfinite, [axis_x, axis_y])):
            raise RuntimeError('Invalid attached part axis')
        self.get_logger().info(
            f"Attached part '{axis['body']}' axis: WORLD x={axis_x:.6f}, y={axis_y:.6f}; parallel to Z")
        # One Bridge-side Cartesian orbit plan, one uninterrupted execution.
        # Bridge measures the actual TCP start pose after the advance stage.
        rclpy.spin_once(self, timeout_sec=0.05)
        req = PlanToPose.Request()
        req.target.header.frame_id = 'world'
        req.target.pose.position.x = axis_x
        req.target.pose.position.y = axis_y
        req.target.pose.position.z = math.radians(angle)  # service-specific encoding
        req.velocity_scale = velocity
        req.acceleration_scale = acceleration
        future = self.orbit_plan.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f'Continuous orbit planning failed: {getattr(result, "message", "no response")}')
        self.get_logger().info(result.message)
        self.trigger(self.execute, 'Continuous orbit execute')
        self.trigger(self.stopped, 'Continuous orbit settle')
        rclpy.spin_once(self, timeout_sec=0.05)
        end_pos, end_q = self.pose_components(self.tcp_pose)
        # A successful detach is required BEFORE retreat; never silently continue.
        msg = self.trigger(self.detach, 'Hex key tip detach')
        self.get_logger().info(f'Detached: {msg}')
        retreat = [end_pos[i]-displacement[i] for i in range(3)]
        if distance > 0:
            self.plan_and_execute(retreat, end_q, True, 'Retreat')
        self.get_logger().info('Rotate task completed')


def main(args=None):
    rclpy.init(args=args)
    node = RotateTask()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

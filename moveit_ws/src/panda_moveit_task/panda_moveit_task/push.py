"""ROS2/MuJoCo straight push: approach world TCP pose, then translate along one world axis.

Requires bridge_push_continuous_search.py and panda_moveit_planner.srv.PlanToPose.
"""
import math
import os
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, SetBool
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from panda_moveit_planner.srv import PlanToPose


class PushTask(Node):
    def __init__(self):
        super().__init__('push_task')
        for name, default in {
            'arm': 'panda1',
            'target_x': 0.0, 'target_y': 0.0, 'target_z': 0.9,
            'target_rx_deg': 0.0, 'target_ry_deg': 0.0, 'target_rz_deg': 0.0,
            'push_direction': '+x', 'push_distance': 0.05,
            'contact_search_max_distance': 0.15,
            'gripper_width': 0.04, 'retreat_distance': 0.03,
            'velocity_scale': 0.20, 'acceleration_scale': 0.20,
            'orientation_reference_file': os.environ.get(
                'PANDA_TASK_ORIENTATION_REFERENCE', '/tmp/panda_task_initial_orientation.json'),
        }.items():
            self.declare_parameter(name, default)
        self.arm = str(self.p('arm')).lower().strip()
        if self.arm not in ('panda1', 'panda2'):
            raise ValueError('arm must be panda1 or panda2')
        self.plan = self.create_client(PlanToPose, f'/{self.arm}/plan_to_pose')
        self.execute = self.create_client(Trigger, f'/{self.arm}/execute_next_plan')
        self.clear = self.create_client(Trigger, f'/{self.arm}/clear_plan_queue')
        self.stop = self.create_client(Trigger, '/mujoco/wait_until_stopped')
        self.push = self.create_client(SetBool, f'/mujoco/{self.arm}/push_contact')
        self.push_status = self.create_client(Trigger, f'/mujoco/{self.arm}/push_status')
        self.push_direction_pub = self.create_publisher(Float64MultiArray, '/mujoco/push_direction', 10)
        gripper_action = ('/mujoco/panda1_gripper_command' if self.arm == 'panda1'
                          else '/mujoco/gripper_command')
        self.gripper = ActionClient(self, GripperCommand, gripper_action)
        self.tcp = None
        self.create_subscription(PoseStamped, f'/mujoco/{self.arm}_tcp_pose',
                                 self.on_tcp, 20)

    def p(self, name):
        return self.get_parameter(name).value

    def on_tcp(self, msg):
        self.tcp = msg

    def call(self, client, request, label):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f'{label}: {getattr(result, "message", "no response")}')
        self.get_logger().info(f'{label}: {result.message}')
        return result

    def set_gripper_width(self, width):
        if not math.isfinite(width) or not 0.0 <= width <= 0.08:
            raise ValueError('gripper_width must be finite and within [0, 0.08] metres')
        self.get_logger().info(f'Setting {self.arm} gripper total width to {width:.4f} m')
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = 0.0  # OPEN/RELEASE intent: never grasp a part
        future = self.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError('gripper width command rejected')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None or result.status != 4:  # action_msgs/GoalStatus.STATUS_SUCCEEDED
            raise RuntimeError(f'gripper width command failed: status={getattr(result, "status", None)}')
        self.get_logger().info(f'Gripper width set: actual={result.result.position:.4f} m')

    @staticmethod
    def qmul(a, b):
        x, y, z, w = a
        X, Y, Z, W = b
        return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
                w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]

    def orientation(self):
        path = str(self.p('orientation_reference_file'))
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        q = [float(d[k]) for k in ('qx', 'qy', 'qz', 'qw')]
        rx, ry, rz = [math.radians(float(self.p(k))) for k in
                      ('target_rx_deg', 'target_ry_deg', 'target_rz_deg')]
        qx = [math.sin(rx/2), 0, 0, math.cos(rx/2)]
        qy = [0, math.sin(ry/2), 0, math.cos(ry/2)]
        qz = [0, 0, math.sin(rz/2), math.cos(rz/2)]
        q = self.qmul(self.qmul(self.qmul(qz, qy), qx), q)
        norm = math.sqrt(sum(x*x for x in q))
        return [x/norm for x in q]

    def queue(self, pos, q, linear):
        req = PlanToPose.Request()
        req.target.header.frame_id = 'world'
        req.target.header.stamp = self.get_clock().now().to_msg()
        req.target.pose.position.x, req.target.pose.position.y, req.target.pose.position.z = pos
        (req.target.pose.orientation.x, req.target.pose.orientation.y,
         req.target.pose.orientation.z, req.target.pose.orientation.w) = q
        req.linear = linear
        req.execute = False
        req.velocity_scale = float(self.p('velocity_scale'))
        req.acceleration_scale = float(self.p('acceleration_scale'))
        self.call(self.plan, req, 'preplan linear' if linear else 'preplan approach')

    def run(self):
        clients = (self.plan, self.execute, self.clear, self.stop, self.push, self.push_status)
        self.get_logger().info('Waiting for bridge planning and push services...')
        for client in clients:
            while rclpy.ok() and not client.wait_for_service(timeout_sec=0.5):
                pass
        if not rclpy.ok():
            return
        while rclpy.ok() and not self.gripper.wait_for_server(timeout_sec=0.5):
            pass
        if not rclpy.ok():
            return
        self.call(self.stop, Trigger.Request(), 'initial stop')
        self.call(self.clear, Trigger.Request(), 'clear plan queue')
        direction = str(self.p('push_direction')).strip().lower().replace(' ', '')
        vecs = {'+x': (1, 0, 0), '-x': (-1, 0, 0),
                '+y': (0, 1, 0), '-y': (0, -1, 0),
                '+z': (0, 0, 1), '-z': (0, 0, -1)}
        if direction not in vecs:
            raise ValueError('push_direction must be +x/-x/+y/-y/+z/-z')
        distance = float(self.p('push_distance'))
        if not math.isfinite(distance) or distance < 0:
            raise ValueError('push_distance must be finite and >= 0')
        retreat = float(self.p('retreat_distance'))
        if not math.isfinite(retreat) or retreat < 0:
            raise ValueError('retreat_distance must be finite and >= 0')
        width = float(self.p('gripper_width'))
        start = [float(self.p(k)) for k in ('target_x', 'target_y', 'target_z')]
        if not all(map(math.isfinite, start)):
            raise ValueError('target position must be finite')
        axis = vecs[direction]
        q = self.orientation()
        search_max = float(self.p('contact_search_max_distance'))
        if not math.isfinite(search_max) or search_max < 0:
            raise ValueError('contact_search_max_distance must be >= 0')
        self.get_logger().info(
            f'Approach={start}, search up to {search_max:.4f} m, '
            f'then push {distance:.4f} m AFTER first contact')
        self.queue(start, q, linear=False)
        active = False
        try:
            self.call(self.execute, Trigger.Request(), 'execute approach')
            self.call(self.stop, Trigger.Request(), 'approach stopped')
            self.set_gripper_width(width)
            final_pos = list(start)
            if distance > 1e-9:
                msg = Float64MultiArray()
                msg.data = [1.0 if self.arm == 'panda1' else 2.0, *map(float, axis)]
                self.push_direction_pub.publish(msg)
                import time
                time.sleep(0.15)
                self.call(self.push, SetBool.Request(data=True), 'start push monitoring')
                active = True
                # One uninterrupted linear motion to the search limit. Bridge
                # interrupts the queued trajectory immediately when it latches
                # the first contact, instead of finishing a search segment.
                contact = None
                search_end = [start[i] + axis[i] * search_max for i in range(3)]
                self.queue(search_end, q, linear=True)
                self.call(self.execute, Trigger.Request(), 'continuous contact search')
                self.call(self.stop, Trigger.Request(), 'contact search stopped')
                result = self.call(self.push_status, Trigger.Request(), 'read push contact status')
                if result.message != 'waiting':
                    contact = json.loads(result.message)
                if contact is None:
                    raise RuntimeError(
                        f'No push contact within {search_max:.4f} m; '
                        'object was not pushed the requested distance')
                contact_tcp = contact['contact_tcp']
                # Endpoint is measured from the TCP at actual contact DETECTION,
                # with no backtracking or first-contact interpolation.
                final_pos = [contact_tcp[i] + axis[i] * distance for i in range(3)]
                self.get_logger().info(
                    f'Detected contact with {contact["object"]} at TCP={contact_tcp}; '
                    f'push endpoint={final_pos}')
                self.queue(final_pos, q, linear=True)
                self.call(self.execute, Trigger.Request(), 'execute post-contact push')
                self.call(self.stop, Trigger.Request(), 'push stopped')
            if active:
                self.call(self.push, SetBool.Request(data=False), 'release push latch')
                active = False
            if retreat > 1e-9:
                back = [final_pos[i] - axis[i] * retreat for i in range(3)]
                self.queue(back, q, linear=True)
                self.call(self.execute, Trigger.Request(), 'execute retreat')
                self.call(self.stop, Trigger.Request(), 'retreat stopped')
            self.get_logger().info('PUSH TASK FINISHED')
        finally:
            if active:
                try:
                    self.call(self.push, SetBool.Request(data=False), 'release push latch')
                except Exception as e:
                    self.get_logger().error(f'Unable to release push latch: {e}')
            try:
                self.call(self.clear, Trigger.Request(), 'clear unused plans')
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = PushTask()
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(f'Push failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

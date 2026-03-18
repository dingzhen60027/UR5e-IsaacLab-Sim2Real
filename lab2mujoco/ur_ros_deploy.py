import time
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState

from mdp.obs import ObsCfg, build_policy_obs
from mdp.policy import TorchScriptPolicy, PolicyCfg

class CommandGenerator:
    def __init__(self):
        # 随机模式参数
        self.pos_centre = np.array([0.5, 0.0, 0.45], dtype=np.float64)
        self.pos_range = np.array([0.25, 0.25, 0.15], dtype=np.float64)
        self.rot_centre = np.array([np.pi, 0.0, -np.pi / 2], dtype=np.float64)
        self.rot_range = np.array([np.pi / 4, np.pi / 4, np.pi / 2], dtype=np.float64)
        self.last_resample_time = -100.0
        
        # 默认指令
        self.current_command = np.array([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, sim_time: float, mode=0, fixed_goal=None):
        # 模式 1: 固定点模式
        if mode == 1 and fixed_goal is not None:
            return fixed_goal.copy()
            
        # 模式 0: 随机采样模式
        if sim_time - self.last_resample_time >= 8.0:
            pos = self.pos_centre + np.random.uniform(-self.pos_range, self.pos_range)
            rot_euler = self.rot_centre + np.random.uniform(-self.rot_range, self.rot_range)
            r = R.from_euler('xyz', rot_euler)
            quat_xyzw = r.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
            self.current_command = np.concatenate([pos, quat_wxyz])
            self.last_resample_time = sim_time
        return self.current_command.copy()

class UR7eRealDeploy(Node):
    def __init__(self):
        super().__init__('ur7e_real_deploy')

        self.standard_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.default_qpos = np.array([3.1415, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)

        # --- 模式切换与固定点设置 ---
        self.control_mode = 0  # 0: 随机模式, 1: 固定点模式
        # 设置你的实验固定点 [x, y, z, qw, qx, qy, qz]
        self.fixed_goal = np.array([0.3, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # --------------------------

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.marker_pub = self.create_publisher(Marker, '/policy_target_marker', 10)

        self.policy = TorchScriptPolicy(PolicyCfg(model_path="./assets/exported/policy.pt", device="cpu"))
        self.obs_cfg = ObsCfg()
        
        self.control_hz = 5.0 
        self.base_action_scale = 0.07  
        self.k_gain = 3.0             
        self.stop_threshold = 0.015    

        self.cmd_gen = CommandGenerator()
        self.start_wall_time = time.time()
        self.goal_in_flight = False
        self.is_initialized = False

        self.move_group_client.wait_for_server()
        self.run_initial_pose_setup()

    def get_ee_position(self):
        try:
            t = self.tf_buffer.lookup_transform('base', 'tool0', rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        except Exception: return None

    def get_exponential_scale(self, target_pos):
        curr_pos = self.get_ee_position()
        if curr_pos is None: return 0.0
        dist = np.linalg.norm(curr_pos - target_pos)
        if dist < self.stop_threshold: return 0.0
        return self.base_action_scale * (1.0 - math.exp(-self.k_gain * dist))

    def joint_state_cb(self, msg):
        self.current_joint_state = msg

    def get_aligned_state(self):
        if self.current_joint_state is None: return None, None
        try:
            name_map = dict(zip(self.current_joint_state.name, self.current_joint_state.position))
            qpos = np.array([name_map[n] for n in self.standard_joint_names], dtype=np.float64)
            return qpos, np.zeros(6)
        except: return None, None

    def run_initial_pose_setup(self):
        while rclpy.ok() and self.current_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        qpos, _ = self.get_aligned_state()
        if np.linalg.norm(qpos - self.default_qpos) < 0.05:
            self.start_loop()
        else:
            self.send_moveit_goal(self.default_qpos, is_init=True)

    def start_loop(self):
        self.is_initialized = True
        self.start_wall_time = time.time()
        self.create_timer(1.0/self.control_hz, self.control_step)
        mode_str = "固定点" if self.control_mode == 1 else "随机采样"
        self.get_logger().info(f"控制循环启动 | 模式: {mode_str}")

    def control_step(self):
        if self.goal_in_flight or not self.is_initialized: return
        
        qpos, qvel = self.get_aligned_state()
        if qpos is None: return

        sim_time = time.time() - self.start_wall_time
        
        # 根据当前模式获取目标
        command = self.cmd_gen.update(sim_time, mode=self.control_mode, fixed_goal=self.fixed_goal)
        self.publish_marker(command)

        current_scale = self.get_exponential_scale(command[:3])
        if current_scale == 0.0: return

        mock_robot = type('Mock', (), {'get_joint_pos': lambda: qpos, 'get_joint_vel': lambda: qvel})
        obs = build_policy_obs(mock_robot, command, cfg=self.obs_cfg)
        raw_action = self.policy.act(obs)

        target_qpos = qpos + (raw_action * current_scale)
        self.send_moveit_goal(target_qpos)

    def send_moveit_goal(self, target_qpos, is_init=False):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.allowed_planning_time = 0.05 
        goal_msg.request.max_velocity_scaling_factor = 0.15
        
        constraints = Constraints()
        for name, val in zip(self.standard_joint_names, target_qpos):
            jc = JointConstraint(joint_name=name, position=float(val), tolerance_above=0.012, tolerance_below=0.012, weight=1.0)
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints = [constraints]
        
        self.goal_in_flight = True
        future = self.move_group_client.send_goal_async(goal_msg)
        
        def cb(f):
            gh = f.result()
            if gh and gh.accepted:
                gh.get_result_async().add_done_callback(self.goal_done)
                if is_init: self.start_loop()
            else: self.goal_in_flight = False
        future.add_done_callback(cb)

    def goal_done(self, _):
        self.goal_in_flight = False

    def publish_marker(self, cmd):
        m = Marker()
        m.header.frame_id, m.header.stamp = "base", self.get_clock().now().to_msg()
        m.type, m.pose.position.x, m.pose.position.y, m.pose.position.z = Marker.SPHERE, *map(float, cmd[:3])
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = map(float, cmd[3:])
        m.scale.x = m.scale.y = m.scale.z = 0.05
        # 固定点模式显示绿色，随机模式显示蓝色
        m.color.r = 0.0 if self.control_mode == 1 else 0.2
        m.color.g = 1.0 if self.control_mode == 1 else 0.6
        m.color.b = 0.0 if self.control_mode == 1 else 1.0
        m.color.a = 0.8
        self.marker_pub.publish(m)

def main():
    rclpy.init()
    node = UR7eRealDeploy()
    try: rclpy.spin(node)
    except: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()

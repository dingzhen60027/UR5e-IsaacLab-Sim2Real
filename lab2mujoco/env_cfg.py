import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# 导入自定义模块
from robot.ur5e import RobotCfg, MujocoRobot
from mdp.obs import ObsCfg, build_policy_obs
from mdp.policy import TorchScriptPolicy, PolicyCfg, OnnxPolicy

# ROS 2 相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

## --- ROS 2 发布者节点 ---
class JointStatePublisher(Node):
    def __init__(self, joint_names):
        super().__init__('mujoco_joint_publisher')
        # 创建发布者，话题名为 'joint_states'，这是 ROS 2 机器人状态的标准话题
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.joint_names = joint_names

    def publish_joints(self, qpos, qvel):
        """将 MuJoCo 数据打包并发布到 ROS 2"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = qpos.tolist()
        msg.velocity = qvel.tolist()
        self.publisher_.publish(msg)

## --- 1. 配置初始化 ---

robot_cfg = RobotCfg(
    mjcf_path="./assets/robots/universal_robots_ur5e/ur5e.xml",
    joint_names=[
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ],
    actuator_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
    default_qpos={
        "shoulder_pan_joint": 3.1415, "shoulder_lift_joint": -1.5708,
        "elbow_joint": 1.5708, "wrist_1_joint": -1.5708,
        "wrist_2_joint": -1.5708, "wrist_3_joint": 0.0,
    }
)

policy_cfg = PolicyCfg(
    model_path="./assets/exported/policy.pt",
    device="cpu",
)

## --- 2. 目标生成器 ---

class CommandGenerator:
    def __init__(self):
        self.pos_centre = np.array([0.5, 0.0, 0.3]) 
        self.pos_range = np.array([0.25, 0.25, 0.15]) 
        self.rot_centre = np.array([np.pi, 0.0, -np.pi / 2])
        self.rot_range = np.array([np.pi / 4, np.pi / 4, np.pi / 2]) 
        self.last_resample_time = -100.0
        self.current_command = np.zeros(7)

    def update(self, sim_time):
        if sim_time - self.last_resample_time >= 2.0:
            pos = self.pos_centre + np.random.uniform(-self.pos_range, self.pos_range)
            rot_euler = self.rot_centre + np.random.uniform(-self.rot_range, self.rot_range)
            r = R.from_euler('xyz', rot_euler)
            quat_xyzw = r.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.current_command = np.concatenate([pos, quat_wxyz])
            self.last_resample_time = sim_time
            print(f"[{sim_time:.1f}s] New Goal -> Pos: {pos.round(2)}")
        return self.current_command

## --- 3. 运行环境 ---

def run_simulation():
    # ROS 2 系统初始化
    rclpy.init()
    
    # 实例化机器人、策略、命令生成器
    robot = MujocoRobot(robot_cfg)
    obs_cfg = ObsCfg()
    policy = TorchScriptPolicy(policy_cfg)
    cmd_gen = CommandGenerator()

    # 实例化 ROS 2 节点
    ros_node = JointStatePublisher(robot_cfg.joint_names)
    
    action_scale = 0.0625
    robot.model.opt.timestep = 1.0 / 120.0

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        robot.reset()
        print("仿真开始！ROS 2 正在发布 /joint_states 话题。按 ESC 退出。")
        
        while viewer.is_running():
            start_time = time.time()
            
            # 1. 获取指令并可视化目标红球
            command = cmd_gen.update(robot.data.time)
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0, 0], 
                pos=command[:3], 
                mat=np.eye(3).flatten(), 
                rgba=[1, 0, 0, 0.5]
            )

            # 2. 发布当前关节状态到 ROS 2
            current_qpos = robot.get_joint_pos()
            current_qvel = robot.get_joint_vel()
            ros_node.publish_joints(current_qpos, current_qvel)

            # 3. 构建观测并进行策略推理
            current_obs = build_policy_obs(robot, command, cfg=obs_cfg)
            raw_action = policy.act(current_obs)
            
            # 4. 执行控制 (增量式转绝对位置)
            target_qpos = current_qpos + (raw_action * action_scale)
            for _ in range(2):
                robot.set_ctrl(target_qpos)
                robot.step()

            viewer.sync()

            # 处理 ROS 2 回调（虽然目前主要是发布，但也需要 spin 以维持心跳）
            rclpy.spin_once(ros_node, timeout_sec=0)
            
            # 维持控制频率 60Hz
            expected_dt = 1.0 / 60.0
            actual_dt = time.time() - start_time
            if actual_dt < expected_dt:
                time.sleep(expected_dt - actual_dt)

    # 退出前清理 ROS 2
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    run_simulation()
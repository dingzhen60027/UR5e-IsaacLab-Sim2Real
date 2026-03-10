# sim2sim_mujoco/mdp/obs.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ObsCfg:
    """观测配置，需与训练时的 ObservationsCfg 保持一致"""
    concatenate_terms: bool = True
    # 这里的 clip 应参考训练时的配置，通常 Isaac Lab 默认为 100.0 或不裁剪
    clip: float = 100.0 
    
    # 增加：训练时的默认关节位置（用于计算相对位置）
    # 根据你之前的配置，UR5e 的初始姿态：
    default_joint_pos: np.ndarray = field(default_factory=lambda: np.array([
        3.14159, -1.5708, 1.5708, -1.5708, -1.5708, 0.0
    ], dtype=np.float32))

def joint_pos(robot, default_pos: np.ndarray | None = None) -> np.ndarray:
    """获取关节位置。
    注意：很多训练配置会输入 (current_pos - default_pos)，
    但根据你的 ObservationsCfg 源码，它使用的是原始 mdp.joint_pos。
    """
    q = robot.get_joint_pos().astype(np.float32).reshape(-1)
    # 如果训练时使用了减去默认位置的操作，取消下面一行的注释：
    # if default_pos is not None: q -= default_pos
    return q

def joint_vel(robot) -> np.ndarray:
    """获取关节速度。"""
    return robot.get_joint_vel().astype(np.float32).reshape(-1)

def generated_commands(ee_pose_command: np.ndarray) -> np.ndarray:
    
    """
    ee_pose_command 应该是 7 维向量: [pos_x, pos_y, pos_z, qw, qx, qy, qz]
    """
    return np.asarray(ee_pose_command, dtype=np.float32).reshape(-1)

def build_policy_obs(
    robot,
    ee_pose_command: np.ndarray,
    cfg: ObsCfg | None = None,
) -> np.ndarray:
    """
    严格按照训练时的顺序构建观测向量：
    1. joint_pos (6维)
    2. joint_vel (6维)
    3. pose_command (7维)
    总计：19维
    """
    if cfg is None:
        cfg = ObsCfg()

    # 1. 获取各项数据
    q = joint_pos(robot, cfg.default_joint_pos)
    dq = joint_vel(robot)
    cmd = generated_commands(ee_pose_command)

    # 2. 拼接
    # 顺序必须是：joint_pos -> joint_vel -> pose_command
    if cfg.concatenate_terms:
        obs = np.concatenate([q, dq, cmd], axis=0).astype(np.float32)
    else:
        raise ValueError("Sim2Sim requires concatenate_terms=True")

    # 3. 裁剪 (防止数值爆炸)
    obs = np.clip(obs, -cfg.clip, cfg.clip)
    
    return obs
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e 配置文件。"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# 配置
##

UR5e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 3.141592653589793,
            "shoulder_lift_joint": -1.5707963267948966,
            "elbow_joint": 1.5707963267948966,
            "wrist_1_joint": -1.5707963267948966,
            "wrist_2_joint": -1.5707963267948966,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # 机械臂本体执行器
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
    },
)

"""UR5e 本体配置。"""


UR5e_ROBOTIQ_GRIPPER_CFG = UR5e_CFG.copy()
"""UR5e + Robotiq_2f_140 夹爪配置。"""

UR5e_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2f_140"}
UR5e_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True

# 夹爪初始状态
UR5e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR5e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR5e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_pad_joint"] = 0.0
UR5e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0

# 夹爪主驱动
UR5e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)

# 夹爪辅助驱动
UR5e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)

# 夹爪被动关节
UR5e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)


UR5e_ROBOTIQ_2F_85_CFG = UR5e_CFG.copy()
"""UR5e + Robotiq 2F-85 夹爪配置。"""

UR5e_ROBOTIQ_2F_85_CFG.spawn.variants = {"Gripper": "Robotiq_2f_85"}
UR5e_ROBOTIQ_2F_85_CFG.spawn.rigid_props.disable_gravity = True

# 夹爪初始状态
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_knuckle_joint"] = 0.0
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0

# 夹爪主驱动
UR5e_ROBOTIQ_2F_85_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)

# 夹爪辅助驱动
UR5e_ROBOTIQ_2F_85_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)

# 夹爪被动关节
UR5e_ROBOTIQ_2F_85_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)

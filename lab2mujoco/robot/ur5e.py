# sim2sim_mujoco/robot/ur5e.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


@dataclass
class RobotCfg:
    mjcf_path: str
    joint_names: Sequence[str]
    actuator_names: Sequence[str]
    default_qpos: dict[str, float] = field(default_factory=dict)
    base_joint_name: str | None = None


class MujocoRobot:
    """Lightweight robot wrapper for MuJoCo inference."""

    def __init__(self, cfg: RobotCfg):
        self.cfg = cfg
        self.mjcf_path = Path(cfg.mjcf_path).expanduser().resolve()
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF not found: {self.mjcf_path}")

        # Load compiled model from MJCF XML path.
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        self.joint_ids = self._resolve_joint_ids(cfg.joint_names)
        self.actuator_ids = self._resolve_actuator_ids(cfg.actuator_names)

        # qpos / qvel address for controlled joints
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids],
            dtype=np.int32,
        )
        self.qvel_adr = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids],
            dtype=np.int32,
        )

        self.default_qpos_vec = self._build_default_qpos()

    def _resolve_joint_ids(self, names: Sequence[str]) -> np.ndarray:
        ids = []
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint not found in MJCF: {name}")
            ids.append(jid)
        return np.asarray(ids, dtype=np.int32)

    def _resolve_actuator_ids(self, names: Sequence[str]) -> np.ndarray:
        ids = []
        for name in names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"Actuator not found in MJCF: {name}")
            ids.append(aid)
        return np.asarray(ids, dtype=np.int32)

    def _build_default_qpos(self) -> np.ndarray:
        qpos = np.array(self.data.qpos, copy=True)
        for joint_name, value in self.cfg.default_qpos.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid == -1:
                raise ValueError(f"default_qpos joint not found: {joint_name}")
            adr = self.model.jnt_qposadr[jid]
            qpos[adr] = value
        return qpos

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos_vec
        mujoco.mj_forward(self.model, self.data)

    def get_joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def get_joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape[0] != len(self.actuator_ids):
            raise ValueError(
                f"ctrl size mismatch: got {ctrl.shape[0]}, expected {len(self.actuator_ids)}"
            )
        self.data.ctrl[self.actuator_ids] = ctrl

    def step(self, nstep: int = 1) -> None:
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)

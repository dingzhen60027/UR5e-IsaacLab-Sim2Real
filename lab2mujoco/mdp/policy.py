# sim2sim_mujoco/mdp/policy.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class PolicyCfg:
    """Policy config for sim2sim inference."""
    model_path: str | None = None
    action_dim: int = 6
    provider: str = "CPUExecutionProvider"
    device: str = "cpu"


class BasePolicy:
    """Base policy interface."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DummyPolicy(BasePolicy):
    """A dummy policy that always outputs zeros."""

    def __init__(self, cfg: PolicyCfg):
        self.cfg = cfg

    def act(self, obs: np.ndarray) -> np.ndarray:
        _ = obs
        return np.zeros(self.cfg.action_dim, dtype=np.float32)


class OnnxPolicy(BasePolicy):
    """ONNX policy wrapper for sim2sim inference."""

    def __init__(self, cfg: PolicyCfg):
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. Please install it before using OnnxPolicy."
            )
        if cfg.model_path is None:
            raise ValueError("PolicyCfg.model_path must be provided for OnnxPolicy.")

        self.cfg = cfg
        model_path = Path(cfg.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.session = ort.InferenceSession(
            str(model_path),
            providers=[cfg.provider],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = self.session.run(
            [self.output_name],
            {self.input_name: obs},
        )[0]
        return np.asarray(action[0], dtype=np.float32).reshape(-1)


class TorchScriptPolicy(BasePolicy):
    """TorchScript .pt policy wrapper for sim2sim inference."""

    def __init__(self, cfg: PolicyCfg):
        if torch is None:
            raise ImportError(
                "torch is not installed. Please install it before using TorchScriptPolicy."
            )
        if cfg.model_path is None:
            raise ValueError("PolicyCfg.model_path must be provided for TorchScriptPolicy.")

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        model_path = Path(cfg.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")

        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    @torch.inference_mode()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )
        action = self.model(obs_tensor)

        if isinstance(action, (tuple, list)):
            action = action[0]

        action = action.detach().to("cpu").numpy()
        return np.asarray(action[0], dtype=np.float32).reshape(-1)


def build_policy(cfg: PolicyCfg) -> BasePolicy:
    """Factory function for building a policy."""
    if cfg.model_path is None:
        return DummyPolicy(cfg)

    model_path = Path(cfg.model_path).expanduser().resolve()
    suffix = model_path.suffix.lower()

    if suffix == ".onnx":
        return OnnxPolicy(cfg)

    if suffix == ".pt":
        return TorchScriptPolicy(cfg)

    raise ValueError(
        f"Unsupported policy format: {suffix}. "
        "Supported formats are: .onnx, .pt (TorchScript)"
    )

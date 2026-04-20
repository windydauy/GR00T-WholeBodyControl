from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return _normalize(_quat_conjugate(q))


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    q = _normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dq = _quat_mul(_quat_inv(q1), q2)
    dq = _normalize(dq)
    return 2.0 * torch.atan2(dq[..., 1:].norm(dim=-1), dq[..., 0].abs().clamp(min=1e-9))


def _subtract_frame_transforms(
    root_pos: torch.Tensor, root_quat: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor
):
    rel_pos = _quat_apply(_quat_inv(root_quat), target_pos - root_pos)
    rel_quat = _quat_mul(_quat_inv(root_quat), target_quat)
    return rel_pos, rel_quat


def _register_isaaclab_stubs():
    isaaclab_mod = types.ModuleType("isaaclab")
    isaaclab_utils_mod = types.ModuleType("isaaclab.utils")
    isaaclab_utils_mod.configclass = lambda cls: cls
    isaaclab_math_mod = types.ModuleType("isaaclab.utils.math")
    isaaclab_math_mod.matrix_from_quat = _matrix_from_quat
    isaaclab_math_mod.quat_apply = _quat_apply
    isaaclab_math_mod.quat_apply_inverse = lambda q, v: _quat_apply(_quat_inv(q), v)
    isaaclab_math_mod.quat_apply_yaw = _quat_apply
    isaaclab_math_mod.quat_conjugate = _quat_conjugate
    isaaclab_math_mod.quat_inv = _quat_inv
    isaaclab_math_mod.quat_mul = _quat_mul
    isaaclab_math_mod.quat_error_magnitude = _quat_error_magnitude
    isaaclab_math_mod.subtract_frame_transforms = _subtract_frame_transforms

    isaaclab_managers_mod = types.ModuleType("isaaclab.managers")
    isaaclab_managers_mod.ObservationGroupCfg = object
    isaaclab_managers_mod.SceneEntityCfg = object

    sys.modules["isaaclab"] = isaaclab_mod
    sys.modules["isaaclab.utils"] = isaaclab_utils_mod
    sys.modules["isaaclab.utils.math"] = isaaclab_math_mod
    sys.modules["isaaclab.managers"] = isaaclab_managers_mod


def _register_mdp_stubs():
    commands_mod = types.ModuleType("gear_sonic.envs.manager_env.mdp.commands")
    commands_mod.TrackingCommand = object
    commands_mod.ForceTrackingCommand = object
    commands_mod._get_body_indexes = (
        lambda command, body_names=None: list(range(len(command.cfg.endpoint_body)))
        if body_names is None
        else [command.cfg.endpoint_body.index(name) for name in body_names]
    )

    utils_mod = types.ModuleType("gear_sonic.envs.manager_env.mdp.utils")
    mdp_pkg = types.ModuleType("gear_sonic.envs.manager_env.mdp")
    mdp_pkg.commands = commands_mod
    mdp_pkg.utils = utils_mod

    sys.modules["gear_sonic.envs.manager_env.mdp"] = mdp_pkg
    sys.modules["gear_sonic.envs.manager_env.mdp.commands"] = commands_mod
    sys.modules["gear_sonic.envs.manager_env.mdp.utils"] = utils_mod


def _load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lite_modules():
    _register_isaaclab_stubs()
    _register_mdp_stubs()
    observations = _load_module(
        "test_observations_module", "gear_sonic/envs/manager_env/mdp/observations.py"
    )
    rewards = _load_module("test_rewards_module", "gear_sonic/envs/manager_env/mdp/rewards.py")
    return observations, rewards


class _FakeCommandManager:
    def __init__(self, command):
        self._command = command

    def get_term(self, _name: str):
        return self._command


class _FakeEnv:
    def __init__(self, command):
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.command_manager = _FakeCommandManager(command)


def _identity_quat(*shape):
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).repeat(*shape, 1)


def _make_command():
    cfg = types.SimpleNamespace(endpoint_body=["lw", "rw", "lf", "rf"])
    command = types.SimpleNamespace()
    command.cfg = cfg
    command.num_future_frames = 2
    command.anchor_pos_w_multi_future = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
    command.anchor_quat_w_multi_future = _identity_quat(1, 2)
    command.root_rot_w_multi_future = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0] * 2])
    command.endpoint_body_pos_w_multi_future = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[2.0, 2.0, 3.0], [3.0, 2.0, 3.0], [4.0, 2.0, 3.0], [5.0, 2.0, 3.0]],
            ]
        ]
    )
    command.endpoint_body_quat_w_multi_future = _identity_quat(1, 2, 4)
    command.reward_point_body_pos_w_multi_future = torch.full((1, 2, 4, 3), 99.0)
    command.anchor_pos_w = torch.tensor([[0.0, 0.0, 0.0]])
    command.anchor_quat_w = _identity_quat(1)
    command.endpoint_body_pos_w = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]
    )
    command.endpoint_body_quat_w = _identity_quat(1, 4)
    command.robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 0.0]])
    command.robot_anchor_quat_w = _identity_quat(1)
    command.robot_endpoint_body_pos_w = command.endpoint_body_pos_w.clone()
    command.robot_endpoint_body_quat_w = command.endpoint_body_quat_w.clone()
    return command


def test_motion_anchor_pose_w_mf_combines_future_root_position_and_rotation(lite_modules):
    observations, _ = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    pose = observations.motion_anchor_pose_w_mf(env, "motion")

    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    )
    assert pose.shape == (1, 18)
    assert torch.allclose(pose, expected)


def test_endpoint_pose_root_local_mf_uses_endpoint_bodies_not_reward_points(lite_modules):
    observations, _ = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    pose = observations.endpoint_pose_root_local_mf(env, "motion")

    expected_frame0 = torch.tensor(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ]
    )
    assert pose.shape == (1, 72)
    assert torch.allclose(pose[0, :18], expected_frame0)
    assert not torch.any(pose == 99.0)


def test_tracking_endpoint_local_pos_is_one_for_perfect_match_and_drops_with_offset(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    perfect = rewards.tracking_endpoint_local_pos_error(env, "motion", std=0.1)
    command.robot_endpoint_body_pos_w = command.robot_endpoint_body_pos_w + torch.tensor(
        [[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    shifted = rewards.tracking_endpoint_local_pos_error(env, "motion", std=0.1)

    assert torch.allclose(perfect, torch.ones_like(perfect))
    assert shifted.item() < perfect.item()


def test_tracking_endpoint_local_ori_is_one_for_perfect_match_and_drops_with_rotation(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    perfect = rewards.tracking_endpoint_local_ori_error(env, "motion", std=0.4)
    command.robot_endpoint_body_quat_w[:, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    rotated = rewards.tracking_endpoint_local_ori_error(env, "motion", std=0.4)

    assert torch.allclose(perfect, torch.ones_like(perfect))
    assert rotated.item() < perfect.item()

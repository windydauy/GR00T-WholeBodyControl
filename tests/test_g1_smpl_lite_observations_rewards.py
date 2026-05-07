from __future__ import annotations

import importlib.util
import math
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
        self.wrapper = types.SimpleNamespace(current_global_step=0)


def _identity_quat(*shape):
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).repeat(*shape, 1)


def _make_command():
    cfg = types.SimpleNamespace(endpoint_body=["lw", "rw", "lf", "rf"], dt_future_ref_frames=0.1)
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
    command.anchor_lin_vel_w = torch.zeros((1, 3), dtype=torch.float32)
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


def test_adaptive_sigma_clamps_and_interpolates(lite_modules):
    _, rewards = lite_modules

    speed = torch.tensor([0.0, 0.075, 0.2], dtype=torch.float32)
    sigma = rewards._adaptive_sigma(speed, 0.05, 0.10, 0.01, 0.10)

    expected = torch.tensor([0.01, 0.055, 0.10], dtype=torch.float32)
    assert torch.allclose(sigma, expected, atol=1e-6)


def test_adaptive_endpoint_pos_reward_falls_back_to_legacy_params(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    reward = rewards.adp_tracking_endpoint_local_pos_error(
        env,
        "motion",
        std=0.1,
        lin_vel_threshold=0.5,
    )

    assert torch.allclose(reward, torch.ones_like(reward))


def test_adaptive_endpoint_gate_zeroes_reward_when_base_reference_moves_fast(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    command.anchor_lin_vel_w[:] = torch.tensor([[0.03, 0.0, 0.0]])
    env = _FakeEnv(command)

    reward = rewards.adp_tracking_endpoint_local_pos_error(
        env,
        "motion",
        std=0.1,
        base_vel_gate_threshold=0.02,
        ee_speed_min=0.05,
        ee_speed_max=0.10,
        sigma_pos_min_start=0.10,
        sigma_pos_min_final=0.01,
        sigma_pos_max=0.10,
        ee_weight_start=0.1,
        ee_weight_final=0.5,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
    )

    assert torch.allclose(reward, torch.zeros_like(reward))


def test_adaptive_endpoint_reward_uses_weight_schedule_when_active(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    env.wrapper.current_global_step = 0
    reward_start = rewards.adp_tracking_endpoint_local_pos_error(
        env,
        "motion",
        std=0.1,
        base_vel_gate_threshold=0.02,
        ee_speed_min=0.05,
        ee_speed_max=0.10,
        sigma_pos_min_start=0.10,
        sigma_pos_min_final=0.01,
        sigma_pos_max=0.10,
        ee_weight_start=0.1,
        ee_weight_final=0.5,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
    )
    env.wrapper.current_global_step = 12500
    reward_mid = rewards.adp_tracking_endpoint_local_pos_error(
        env,
        "motion",
        std=0.1,
        base_vel_gate_threshold=0.02,
        ee_speed_min=0.05,
        ee_speed_max=0.10,
        sigma_pos_min_start=0.10,
        sigma_pos_min_final=0.01,
        sigma_pos_max=0.10,
        ee_weight_start=0.1,
        ee_weight_final=0.5,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
    )
    env.wrapper.current_global_step = 20000
    reward_end = rewards.adp_tracking_endpoint_local_pos_error(
        env,
        "motion",
        std=0.1,
        base_vel_gate_threshold=0.02,
        ee_speed_min=0.05,
        ee_speed_max=0.10,
        sigma_pos_min_start=0.10,
        sigma_pos_min_final=0.01,
        sigma_pos_max=0.10,
        ee_weight_start=0.1,
        ee_weight_final=0.5,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
    )

    assert reward_start.item() == pytest.approx(0.1, abs=1e-6)
    assert reward_mid.item() == pytest.approx(0.3, abs=1e-6)
    assert reward_end.item() == pytest.approx(0.5, abs=1e-6)


def test_adaptive_endpoint_orientation_reward_uses_same_weight_schedule(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    env = _FakeEnv(command)
    env.wrapper.current_global_step = 12500

    reward = rewards.adp_tracking_endpoint_local_ori_error(
        env,
        "motion",
        std=0.4,
        base_vel_gate_threshold=0.02,
        ee_speed_min=0.05,
        ee_speed_max=0.10,
        sigma_rot_min=5.0,
        sigma_rot_max=20.0,
        ee_weight_start=0.1,
        ee_weight_final=0.5,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
    )

    assert reward.item() == pytest.approx(0.3, abs=1e-6)


def test_curriculum_helpers_follow_requested_sigma_schedule(lite_modules):
    _, rewards = lite_modules

    assert rewards._linear_ramp(0, 10000, 15000, 0.10, 0.01) == pytest.approx(0.10)
    assert rewards._linear_ramp(12500, 10000, 15000, 0.10, 0.01) == pytest.approx(0.055)
    assert rewards._linear_ramp(20000, 10000, 15000, 0.10, 0.01) == pytest.approx(0.01)


def test_tracking_anchor_position_curriculum_applies_weight_and_sigma(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    command.robot_anchor_pos_w = torch.tensor([[0.30, 0.0, 0.0]])
    env = _FakeEnv(command)
    env.wrapper.current_global_step = 12500

    reward = rewards.tracking_anchor_pos_error(
        env,
        "motion",
        std=0.3,
        use_curriculum=True,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
        weight_scale_start=0.0,
        weight_scale_final=1.0,
        sigma_start=0.60,
        sigma_final=0.30,
    )

    expected_sigma = 0.45
    expected_weight = 0.5
    expected = expected_weight * math.exp(-(0.30**2) / (expected_sigma**2))
    assert reward.item() == pytest.approx(expected, abs=1e-6)


def test_tracking_anchor_orientation_curriculum_applies_weight_and_sigma(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    angle = 0.2
    command.robot_anchor_quat_w = torch.tensor(
        [[math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]],
        dtype=torch.float32,
    )
    env = _FakeEnv(command)
    env.wrapper.current_global_step = 12500

    reward = rewards.tracking_anchor_ori_error(
        env,
        "motion",
        std=0.4,
        use_curriculum=True,
        curriculum_start_step=10000,
        curriculum_end_step=15000,
        weight_scale_start=0.0,
        weight_scale_final=1.0,
        sigma_start=0.80,
        sigma_final=0.40,
    )

    expected_sigma = 0.60
    expected_weight = 0.5
    expected = expected_weight * math.exp(-(angle**2) / (expected_sigma**2))
    assert reward.item() == pytest.approx(expected, abs=1e-6)


def test_invalid_adaptive_sigma_config_raises(lite_modules):
    _, rewards = lite_modules

    with pytest.raises(ValueError, match="ee_speed_max must be greater"):
        rewards._adaptive_sigma(torch.zeros(1), 0.10, 0.10, 0.01, 0.10)


def test_invalid_curriculum_window_raises(lite_modules):
    _, rewards = lite_modules
    command = _make_command()
    env = _FakeEnv(command)

    with pytest.raises(ValueError, match="curriculum_end_step must be greater"):
        rewards.tracking_anchor_pos_error(
            env,
            "motion",
            std=0.3,
            use_curriculum=True,
            curriculum_start_step=10000,
            curriculum_end_step=10000,
        )

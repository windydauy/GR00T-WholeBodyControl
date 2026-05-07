from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str):
    return OmegaConf.load(REPO_ROOT / relative_path)


def test_g1_only_experiment_uses_parallel_full_size_components():
    cfg = load_yaml(
        "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_g1_only_5point_ori.yaml"
    )

    defaults = cfg.defaults
    assert any(
        "override /actor_critic" in str(item)
        and "g1_only_mlp_v1_5point_ori" in str(item)
        for item in defaults
    )
    assert any(
        "override /manager_env/observations/tokenizer" in str(item)
        and "unitoken_g1_only_5point_ori" in str(item)
        for item in defaults
    )
    assert any(
        "override /manager_env/observations/policy" in str(item)
        and "local_dir_hist_anchor" in str(item)
        for item in defaults
    )
    assert any(
        "override /manager_env/rewards" in str(item)
        and "tracking/base_5point_local_endpoint_ori_feet_acc" in str(item)
        for item in defaults
    )
    assert any(
        "/aux_losses" in str(item) and "universal_token/g1_recon_only" in str(item)
        for item in defaults
    )

    motion = cfg.manager_env.commands.motion
    assert motion.encoder_sample_probs == {"g1": 1.0}
    assert motion.reward_point_body == [
        "pelvis",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    assert motion.reward_point_body_offset == [
        [0.0, 0.0, 0.0],
        [0.18, -0.025, 0.0],
        [0.18, 0.025, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert "teleop_sample_prob_when_smpl" not in motion
    motion_lib_cfg = OmegaConf.to_container(motion.motion_lib_cfg, resolve=False)
    assert "smpl_motion_file" not in motion_lib_cfg
    assert cfg.algo.config.actor.backbone.reencode_smpl_g1_recon is False


def test_g1_only_actor_keeps_release_widths_without_smpl_or_teleop():
    cfg = load_yaml("gear_sonic/config/actor_critic/universal_token/g1_only_mlp_v1_5point_ori.yaml")
    g1_dyn_cfg = load_yaml("gear_sonic/config/actor_critic/decoders/g1_dyn_mlp_5point_ori.yaml")

    backbone = cfg.algo.config.actor.backbone
    assert cfg.manager_env.commands.motion.encoder_sample_probs == {"g1": 1.0}
    assert set(backbone.encoders.keys()) == {"g1"}
    assert "smpl" not in backbone.encoders
    assert "teleop" not in backbone.encoders
    assert backbone.decoders.g1_dyn.inputs == [
        "token_flattened",
        "proprioception",
        "motion_anchor_pose_w_mf",
        "endpoint_pose_root_local_mf",
    ]
    assert g1_dyn_cfg.g1_dyn.params.module_config_dict.layer_config.hidden_dims == [
        2048,
        2048,
        1024,
        1024,
        512,
        512,
    ]


def test_g1_only_tokenizer_policy_reward_and_aux_configs():
    tokenizer_cfg = load_yaml(
        "gear_sonic/config/manager_env/observations/tokenizer/unitoken_g1_only_5point_ori.yaml"
    )
    policy_cfg = load_yaml("gear_sonic/config/manager_env/observations/policy/local_dir_hist_anchor.yaml")
    reward_cfg = load_yaml(
        "gear_sonic/config/manager_env/rewards/tracking/base_5point_local_endpoint_ori_feet_acc.yaml"
    )
    aux_cfg = load_yaml("gear_sonic/config/aux_losses/universal_token/g1_recon_only.yaml")

    tokenizer_defaults = tokenizer_cfg.defaults
    assert any("../terms/encoder_index@_here_" in str(item) for item in tokenizer_defaults)
    assert any("../terms/command_multi_future_nonflat@_here_" in str(item) for item in tokenizer_defaults)
    assert any("../terms/motion_anchor_ori_b_mf_nonflat@_here_" in str(item) for item in tokenizer_defaults)
    assert any("../terms/motion_anchor_pose_w_mf@_here_" in str(item) for item in tokenizer_defaults)
    assert any("../terms/endpoint_pose_root_local_mf@_here_" in str(item) for item in tokenizer_defaults)
    assert "smpl" not in str(tokenizer_defaults)
    assert "teleop" not in str(tokenizer_defaults)

    assert any("../terms/robot_anchor_pose_w@_here_" in str(item) for item in policy_cfg.defaults)
    assert OmegaConf.to_container(policy_cfg.robot_anchor_pose_w, resolve=False)["history_length"] == (
        "${actor_prop_history_length}"
    )

    reward_defaults = reward_cfg.defaults
    assert any("tracking_vr_5point_local" in str(item) for item in reward_defaults)
    assert any("tracking_endpoint_local_ori" in str(item) for item in reward_defaults)
    assert not any("tracking_endpoint_local_pos" in str(item) for item in reward_defaults)

    assert set(aux_cfg.aux_loss_coef.keys()) == {"g1_recon"}
    assert aux_cfg.aux_loss_coef.g1_recon == 0.01


def test_global_body_curriculum_preset_exposes_new_anchor_and_ee_schedule_params():
    cfg = load_yaml(
        "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_g1_only_5point_ori_lite_adp_global_body_curriculum.yaml"
    )

    defaults = cfg.defaults
    assert any(
        "/exp/manager/universal_token/all_modes/sonic_g1_only_5point_ori_lite_adp_global_body"
        in str(item)
        for item in defaults
    )

    tracking_anchor_pos = cfg.manager_env.rewards.tracking_anchor_pos.params
    assert tracking_anchor_pos.use_curriculum is True
    assert tracking_anchor_pos.curriculum_start_step == 10000
    assert tracking_anchor_pos.curriculum_end_step == 15000
    assert tracking_anchor_pos.weight_scale_start == 0.0
    assert tracking_anchor_pos.weight_scale_final == 1.0
    assert tracking_anchor_pos.sigma_start == 0.60
    assert tracking_anchor_pos.sigma_final == 0.30

    tracking_anchor_ori = cfg.manager_env.rewards.tracking_anchor_ori.params
    assert tracking_anchor_ori.use_curriculum is True
    assert tracking_anchor_ori.sigma_start == 0.80
    assert tracking_anchor_ori.sigma_final == 0.40

    ee_pos = cfg.manager_env.rewards.adp_tracking_endpoint_local_pos.params
    assert ee_pos.base_vel_gate_threshold == 0.02
    assert ee_pos.ee_speed_min == 0.05
    assert ee_pos.ee_speed_max == 0.10
    assert ee_pos.sigma_pos_min_start == 0.10
    assert ee_pos.sigma_pos_min_final == 0.01
    assert ee_pos.sigma_pos_max == 0.10
    assert ee_pos.ee_weight_start == 0.1
    assert ee_pos.ee_weight_final == 0.5

    ee_ori = cfg.manager_env.rewards.adp_tracking_endpoint_local_ori.params
    assert ee_ori.sigma_rot_min == 5.0
    assert ee_ori.sigma_rot_max == 20.0
    assert ee_ori.curriculum_start_step == 10000
    assert ee_ori.curriculum_end_step == 15000

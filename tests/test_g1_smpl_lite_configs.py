from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str):
    return OmegaConf.load(REPO_ROOT / relative_path)


def test_lite_experiment_config_exists_and_uses_lite_components():
    cfg = load_yaml("gear_sonic/config/exp/manager/universal_token/all_modes/sonic_g1_smpl_lite.yaml")

    defaults = cfg.defaults
    assert any("override /actor_critic" in str(item) and "all_mlp_v1_lite" in str(item) for item in defaults)
    assert any(
        "override /manager_env/observations/tokenizer" in str(item)
        and "unitoken_g1_smpl_lite" in str(item)
        for item in defaults
    )
    assert any(
        "override /manager_env/observations/policy" in str(item) and "local_dir_hist_lite" in str(item)
        for item in defaults
    )
    assert any(
        "override /manager_env/rewards" in str(item)
        and "tracking/base_endpoint_local_feet_acc" in str(item)
        for item in defaults
    )
    assert cfg.manager_env.commands.motion.encoder_sample_probs == {"g1": 1.0, "smpl": 1.0}


def test_lite_actor_critic_config_removes_teleop_and_adds_geometry_inputs():
    cfg = load_yaml("gear_sonic/config/actor_critic/universal_token/all_mlp_v1_lite.yaml")

    assert cfg.manager_env.commands.motion.encoder_sample_probs == {"g1": 1.0, "smpl": 1.0}
    assert "teleop" not in cfg.algo.config.actor.backbone.encoders
    assert cfg.algo.config.actor.backbone.decoders.g1_dyn.inputs == [
        "token_flattened",
        "proprioception",
        "motion_anchor_pose_w_mf",
        "endpoint_pose_root_local_mf",
    ]


def test_lite_network_widths_are_shrunk():
    actor_cfg = load_yaml("gear_sonic/config/actor_critic/universal_token/all_mlp_v1_lite.yaml")
    g1_cfg = load_yaml("gear_sonic/config/actor_critic/encoders/g1_mf_mlp_lite.yaml")
    smpl_cfg = load_yaml("gear_sonic/config/actor_critic/encoders/smpl_mlp_lite.yaml")
    g1_kin_cfg = load_yaml("gear_sonic/config/actor_critic/decoders/g1_kin_mf_mlp_lite.yaml")
    g1_dyn_cfg = load_yaml("gear_sonic/config/actor_critic/decoders/g1_dyn_mlp_lite.yaml")

    assert g1_cfg.g1.params.module_config_dict.layer_config.hidden_dims == [1024, 512, 256, 256]
    assert smpl_cfg.smpl.params.module_config_dict.layer_config.hidden_dims == [1024, 512, 256, 256]
    assert g1_kin_cfg.g1_kin.params.module_config_dict.layer_config.hidden_dims == [
        1024,
        512,
        256,
        256,
    ]
    assert g1_dyn_cfg.g1_dyn.params.module_config_dict.layer_config.hidden_dims == [
        1024,
        1024,
        512,
        512,
        256,
        256,
    ]
    assert actor_cfg.algo.config.actor.backbone.num_fsq_levels == 32
    assert actor_cfg.algo.config.actor.backbone.max_num_tokens == 2


def test_lite_policy_and_tokenizer_configs_add_geometry_observations():
    policy_cfg = load_yaml("gear_sonic/config/manager_env/observations/policy/local_dir_hist_lite.yaml")
    tokenizer_cfg = load_yaml(
        "gear_sonic/config/manager_env/observations/tokenizer/unitoken_g1_smpl_lite.yaml"
    )

    policy_defaults = policy_cfg.defaults
    tokenizer_defaults = tokenizer_cfg.defaults

    assert any("../terms/robot_anchor_pose_w@_here_" in str(item) for item in policy_defaults)
    assert OmegaConf.to_container(policy_cfg.robot_anchor_pose_w, resolve=False)["history_length"] == "${actor_prop_history_length}"
    assert "noise" not in policy_cfg.robot_anchor_pose_w

    assert any("../terms/motion_anchor_pose_w_mf@_here_" in str(item) for item in tokenizer_defaults)
    assert any("../terms/endpoint_pose_root_local_mf@_here_" in str(item) for item in tokenizer_defaults)
    assert "teleop" not in str(tokenizer_defaults)


def test_lite_reward_and_aux_configs_use_endpoint_terms_only():
    reward_cfg = load_yaml(
        "gear_sonic/config/manager_env/rewards/tracking/base_endpoint_local_feet_acc.yaml"
    )
    aux_cfg = load_yaml("gear_sonic/config/aux_losses/universal_token/g1_recon_and_smpl_latent_lite.yaml")
    endpoint_pos_cfg = load_yaml(
        "gear_sonic/config/manager_env/rewards/terms/tracking_endpoint_local_pos.yaml"
    )
    endpoint_ori_cfg = load_yaml(
        "gear_sonic/config/manager_env/rewards/terms/tracking_endpoint_local_ori.yaml"
    )

    reward_defaults = reward_cfg.defaults
    assert not any("tracking_vr_5point_local" in str(item) for item in reward_defaults)
    assert any("tracking_endpoint_local_pos" in str(item) for item in reward_defaults)
    assert any("tracking_endpoint_local_ori" in str(item) for item in reward_defaults)

    assert set(aux_cfg.aux_loss_coef.keys()) == {"g1_recon", "g1_smpl_latent", "reencoded_smpl_g1_latent"}
    assert endpoint_pos_cfg.tracking_endpoint_local_pos.weight == 1.0
    assert endpoint_pos_cfg.tracking_endpoint_local_pos.params.std == 0.1
    assert endpoint_ori_cfg.tracking_endpoint_local_ori.weight == 1.0
    assert endpoint_ori_cfg.tracking_endpoint_local_ori.params.std == 0.4

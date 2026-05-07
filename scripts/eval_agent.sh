python gear_sonic/eval_agent_trl.py \
    +checkpoint=/home/mole/yzh/GR00T-WholeBodyControl/sonic_release/our_ckpt/last.pt \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=8 \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=/home/mole/yzh/GR00T-WholeBodyControl/sonic_release/render_ours_agent" \
    ++manager_env.config.env_spacing=10.0 \
    ++manager_env.recorders.dataset_export_mode=0 \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/mole/yzh/GR00T-WholeBodyControl/data/robot_filtered_clean" \
    "~manager_env/recorders=empty" "+manager_env/recorders=render" \
    +device=cuda:2

    
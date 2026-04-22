export WANDB_API_KEY=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
export WANDB_MODE=offline
export WANDB_ENTITY=yzh_academic-shanghai-jiao-tong-university  # 可留着，但当前这份代码主要读上面那个
# export CUDA_VISIBLE_DEVICES=1

export WANDB_PROJECT=TRL_G1_Track
export RUN_NAME=sonic_release_our_dataset_full_training

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 gear_sonic/train_agent_trl.py \
+exp=manager/universal_token/all_modes/sonic_release \
use_wandb=True \
project_name=${WANDB_PROJECT} \
experiment_name=${RUN_NAME} \
wandb.wandb_entity=${WANDB_ENTITY} \
num_envs=4 headless=True \
++manager_env.commands.motion.motion_lib_cfg.motion_file=data/optitrack_ours/robot_filted

# python gear_sonic/train_agent_trl.py \
# +exp=manager/universal_token/all_modes/sonic_release \
# use_wandb=True \
# project_name=${WANDB_PROJECT} \
# experiment_name=${RUN_NAME} \
# wandb.wandb_entity=${WANDB_ENTITY} \
# num_envs=16384 headless=True \
# ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/optitrack_ours/robot_filted

# python gear_sonic/train_agent_trl.py \
# +exp=manager/universal_token/all_modes/sonic_release \
# +checkpoint=sonic_release/last.pt \
# use_wandb=True \
# project_name=${WANDB_PROJECT} \
# experiment_name=${RUN_NAME} \
# wandb.wandb_entity=${WANDB_ENTITY} \
# num_envs=16384 headless=True \
# ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/optitrack_ours/robot_filted
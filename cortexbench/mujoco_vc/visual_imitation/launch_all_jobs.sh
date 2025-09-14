#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#

# Set W&B to public instance to log to shared team
export WANDB_BASE_URL="https://api.wandb.ai"

# DMC
CUDA_VISIBLE_DEVICES=0 python hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=local \
        wandb.project=dmc_test wandb.entity=dongwang \
        env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
           seed=100,200,300 embedding=$(python -m core_model_set) data_dir=$(pwd)/data/datasets/dmc-expert-v1.0 &

# Adroit
CUDA_VISIBLE_DEVICES=1 python hydra_launcher.py --config-name Adroit_BC_config.yaml --multirun hydra/launcher=local \
    wandb.project=adroit_test wandb.entity=dongwang \
    env=pen-v0,relocate-v0 seed=100,200,300 embedding=$(python -m core_model_set) data_dir=$(pwd)/data/datasets/adroit-expert-v1.0 &

# Metaworld
CUDA_VISIBLE_DEVICES=2 python hydra_launcher.py --config-name Metaworld_BC_config.yaml --multirun hydra/launcher=local \
        wandb.project=metaworld_test wandb.entity=dongwang \
        env=assembly-v2-goal-observable,bin-picking-v2-goal-observable,button-press-topdown-v2-goal-observable,drawer-open-v2-goal-observable,hammer-v2-goal-observable \
        seed=100,200,300 embedding=$(python -m core_model_set) data_dir=$(pwd)/data/datasets/metaworld-expert-v1.0 &

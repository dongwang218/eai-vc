cd /checkpoint/data/dongwang/workspace/github/clip/eai-vc/cortexbench/mujoco_vc/visual_imitation

# test
python hydra_launcher.py --config-name Adroit_BC_config.yaml   wandb.project=adroit_test wandb.entity=dongwang   env=pen-v0 seed=100 embedding=pixo_vith16_reg8_256_cat_cls_reg data_dir=$(pwd)/data/datasets/adroit-expert-v1.0/

python hydra_launcher.py --config-name DMC_BC_config.yaml  --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=h100_data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=3600       wandb.project=dmc_test wandb.entity=dongwang         env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1            seed=100,200,300 embedding=pixo_vith16_reg8_256_cat_cls_reg data_dir=$(pwd)/data/datasets/dmc-expert-v1.0

python hydra_launcher.py --config-name Adroit_BC_config.yaml  --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=h100_data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=3600     wandb.project=adroit_test wandb.entity=dongwang     env=pen-v0,relocate-v0 seed=100,200,300 embedding=pixo_vith16_reg8_256_cat_cls_reg data_dir=$(pwd)/data/datasets/adroit-expert-v1.0


python hydra_launcher.py --config-name Metaworld_BC_config.yaml --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=h100_data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=3600         wandb.project=metaworld_test wandb.entity=dongwang         env=assembly-v2-goal-observable,bin-picking-v2-goal-observable,button-press-topdown-v2-goal-observable,drawer-open-v2-goal-observable,hammer-v2-goal-observable         seed=100,200,300 embedding=pixo_vith16_reg8_256_cat_cls_reg data_dir=$(pwd)/data/datasets/metaworld-expert-v1.0

cd /checkpoint/data/dongwang/workspace/github/clip/eai-vc/cortexbench/trifinger_vc
python bc_experiments/train_bc.py --config-name bc_default.yaml  --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=h100_data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=3600 algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube task.n_outer_iter=100 no_wandb=True run_name=reach_cube_test task.n_epoch_every_log=5      seed=100,200,300 algo.pretrained_rep=pixo_vith16_reg8_256_cat_cls_reg

python bc_experiments/train_bc.py --config-name bc_default.yaml  --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=h100_data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=3600 algo.freeze_pretrained_rep=true rep_to_policy=none task=move_cube task.n_outer_iter=1000 no_wandb=True run_name=move_cube_test task.n_epoch_every_log=50      seed=100,200,300 algo.pretrained_rep=pixo_vith16_reg8_256_cat_cls_reg


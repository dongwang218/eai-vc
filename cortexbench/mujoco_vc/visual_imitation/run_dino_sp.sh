# more memory
# more round

python hydra_launcher.py --config-name DMC_BC_config.yaml  --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=300 hydra.launcher.timeout_min=3600       wandb.project=dmc_test wandb.entity=dongwang         env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1            seed=100,200,300 embedding=dinov2_vitl_sp,dinov3_vitl_sp data_dir=$(pwd)/data/datasets/dmc-expert-v1.0 eval_num_traj=25

python hydra_launcher.py --config-name Adroit_BC_config.yaml  --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=300 hydra.launcher.timeout_min=3600     wandb.project=adroit_test wandb.entity=dongwang     env=pen-v0,relocate-v0 seed=100,200,300 embedding=dinov2_vitl_sp,dinov3_vitl_sp data_dir=$(pwd)/data/datasets/adroit-expert-v1.0 eval_num_traj=25


python hydra_launcher.py --config-name Metaworld_BC_config.yaml --multirun hydra/launcher=submitit_slurm     hydra.launcher.account=data hydra.launcher.qos=data_high hydra.launcher.cpus_per_task=10 hydra.launcher.gpus_per_task=1 hydra.launcher.mem_gb=300 hydra.launcher.timeout_min=3600         wandb.project=metaworld_test wandb.entity=dongwang         env=assembly-v2-goal-observable,bin-picking-v2-goal-observable,button-press-topdown-v2-goal-observable,drawer-open-v2-goal-observable,hammer-v2-goal-observable         seed=100,200,300 embedding=dinov2_vitl_sp,dinov3_vitl_sp data_dir=$(pwd)/data/datasets/metaworld-expert-v1.0 eval_num_traj=25

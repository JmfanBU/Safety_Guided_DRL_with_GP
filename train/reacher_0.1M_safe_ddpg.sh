xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=safe_ddpg --env=ReacherGP-v2 --network=mlp --num_timesteps=1e5 --nb_rollout_steps=50\
    --num_hidden=400 --num_layers=2 --value_activation=relu --critic_lr=1e-3 --actor_lr=1e-5 --guard_lr=1e-2 --gamma=0.9 --batch_size=400 --dataset_size=100 --log_path=data_ddpg/reacher_0.1M_safe_ddpg

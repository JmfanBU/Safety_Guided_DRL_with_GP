xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=safe_ddpg --env=PendulumGP-v0 --network=mlp --num_timesteps=1e5 --nb_rollout_steps=200\
    --num_hidden=64 --num_layers=2 --value_activation=relu --critic_lr=1e-3 --guard_lr=1e-3 --actor_lr=1e-4 --gamma=0.99 --dataset_size=400 --log_path=data_ddpg/pendumum_0.1M_safe_ddpg

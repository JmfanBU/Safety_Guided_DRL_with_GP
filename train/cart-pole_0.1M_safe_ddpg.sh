xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=safe_ddpg --env=InvertedPendulumGP-v2 --network=mlp --num_timesteps=2e5 --nb_rollout_steps=1000\
    --num_hidden=400 --num_layers=2 --value_activation=relu --actor_activation=relu --critic_lr=1e-3 --guard_lr=1e-1 --actor_lr=1e-2 --gamma=0.99 --batch_size=100 --log_path=data_ddpg/cart-pole_0.1M_safe_ddpg

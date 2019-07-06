xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=ddpg_baseline --env=InvertedPendulumBL-v2 --network=mlp --num_timesteps=2e4 --nb_rollout_steps=1000\
    --num_hidden=400 --num_layers=2 --actor_activation=relu --save_video_interval=1 --log_path=data_ddpg/cart-pole_test_ddpg --load_actor=data_ddpg/cart-pole_0.1M_safe_ddpg/params/actor_params.pkl

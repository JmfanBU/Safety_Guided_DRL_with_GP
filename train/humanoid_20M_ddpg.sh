xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=safe_ddpg --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --nb_rollout_steps=1000\
    --num_hidden=128 --num_layers=3 --save_path=data_ddpg/humanoid_20M_ddpg --save_video_interval=100 --log_path=data_ddpg/humanoid_20M_ddpg

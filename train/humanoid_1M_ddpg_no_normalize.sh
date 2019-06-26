xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=safe_ddpg --env=Humanoid-v2 --network=mlp --num_timesteps=1e6 --nb_rollout_steps=1000\
    --num_hidden=128 --num_layers=3 --save_path=data_ddpg/humanoid_1M_ddpg_no_normalize --save_video_interval=20 --log_path=data_ddpg/humanoid_1M_ddpg_no_normalize

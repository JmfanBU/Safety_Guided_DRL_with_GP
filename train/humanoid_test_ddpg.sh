xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=ddpg_baseline --env=HumanoidBL-v2 --network=mlp --num_timesteps=2e4 --nb_rollout_steps=1000 \
    --num_hidden=400 --num_layers=2 --save_video_interval=1 --log_path=data_ddpg/humanoid_test_ddpg --load_actor=data_ddpg/humanoid_20M_ddpg/params/actor_params.pkl

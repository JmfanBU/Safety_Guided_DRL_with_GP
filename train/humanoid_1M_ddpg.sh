xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=ddpg_baseline --env=HumanoidBL-v2 --network=mlp --num_timesteps=1e6 --nb_rollout_steps=1000\
    --num_hidden=128 --num_layers=3 --log_path=data_ddpg/humanoid_1M_ddpg

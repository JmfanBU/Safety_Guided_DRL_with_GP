xvfb-run -a -s "-screen 0 1920x1080x24" python -m SafetyGuided_DRL.run --alg=ddpg_baseline --env=ReacherBL-v2 --network=mlp --num_timesteps=1e6 --nb_rollout_steps=50\
    --num_hidden=400 --num_layers=2 --value_activation=relu --critic_lr=1e-3 --actor_lr=1e-4 --gamma=0.9 --batch_size=256 --log_path=data_ddpg/reacher_1M_ddpg

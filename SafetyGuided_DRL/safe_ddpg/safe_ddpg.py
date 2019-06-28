import os
import time
from collections import deque
import pickle

from SafetyGuided_DRL.safe_ddpg.ddpg_learner import DDPG
from SafetyGuided_DRL.safe_ddpg.models import Actor, Critic, Guard
from SafetyGuided_DRL.safe_ddpg.memory import Memory
from SafetyGuided_DRL.safe_ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from baselines.common import set_global_seeds
import baselines.common.tf_util as U

from baselines import logger
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

DELTA = 0.1
DIFF_TAU = 1e-5

def learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type=None,
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          guard_lr=1e-2,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          callback=None,
          load_actor_params=None,
          **network_kwargs):

    set_global_seeds(seed)

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    nb_actions = env.action_space.shape[-1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(network=network, **network_kwargs)
    guard = Guard(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    action_noise = None
    param_noise = None

    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(actor, critic, guard, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, guard_lr=guard_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, noise_delta=DELTA, max_action=max_action)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=1000)
    episode_rewards_history = deque(maxlen=1000)
    episode_costs_history = deque(maxlen=1000)
    sess = U.get_session()
    # Prepare everything.
    agent.initialize(sess)

    if load_actor_params is not None:
        logger.info('Load pretrained actor for testing.')
        action_noise=None
        param_noise=None
        nb_train_steps = 0
        cur_scope = actor.vars[0].name[0:actor.vars[0].name.find('/')]
        orig_scope = list(load_actor_params.keys())[0][0:list(load_actor_params.keys())[0].find('/')]
        print("current scope: {}, original scope: {}".format(cur_scope,orig_scope))
        for i in range(len(actor.vars)):
            if actor.vars[i].name.replace(cur_scope, orig_scope, 1) in load_actor_params:
                assign_op = actor.vars[i].assign(load_actor_params[actor.vars[i].name.replace(cur_scope, orig_scope, 1)])
                sess.run(assign_op)
    else:
        logger.info("Starting from scratch")

    sess.graph.finalize()

    agent.reset()

    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]

    episode_reward = np.zeros(nenvs, dtype=np.float32) #vector
    episode_cost = np.zeros(nenvs, dtype=np.float32)
    episode_step = np.zeros(nenvs, dtype=int) # vector
    episodes = 0 #scalar
    t = 0 # scalar

    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_costs = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_gs = []
    epoch_episodes = 0

    # pre-defined feature and lable variables
    X_feature = np.empty((0, env.observation_space.shape[0] + env.action_space.shape[0]), dtype=np.float32)
    Y_label = np.empty((0, 1), dtype=np.float32)

    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                # of the environments, so resetting here instead
                agent.reset()
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q, g, _ = agent.step(obs, apply_noise=False, compute_Q=True)

                # Execute next action.
                if rank == 0 and render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                new_obs, r, cost, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_cost += cost
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                epoch_gs.append(g)
                _, new_q, new_g, _ = agent.step(new_obs, apply_noise=False, compute_Q=True)
                agent.store_transition(obs, action, r, cost, new_obs, done) #the batched data will be unrolled in memory.py's append.

                if load_actor_params is not None:
                    # update GP every iterations for safety
                    feature = np.concatenate((obs, max_action*action), axis=-1)
                    g_hat = new_g - g
                    skip_flag = True
                    if np.absolute(-cost - (g_hat)) <= DELTA:
                        skip_flag = False
                    elif np.absolute(cost - (g_hat)) <= DELTA:
                        skip_flag = False
                    if not skip_flag and (np.absolute(g_hat) >= DELTA):
                        # delete features if they have been seen before
                        num_eliminate = 0
                        for idx, feature_i in enumerate(X_feature):
                            if cauchy_schwartz_check(feature, feature_i):
                                X_feature = np.delete(X_feature, idx-num_eliminate, axis=0)
                                Y_label = np.delete(Y_label, idx-num_eliminate, axis=0)
                                num_eliminate = 0
                        # delete older features that have been seen before
                        num_eliminate = 0
                        for idx, feature_i in enumerate(agent.X_feature):
                            if cauchy_schwartz_check(feature, feature_i):
                                agent.eliminate_data_point(np.array(idx - num_eliminate))
                                num_eliminate += 1

                        X_feature = np.vstack((X_feature, np.atleast_2d(feature.astype(np.float32))))
                        Y_label = np.vstack((Y_label, np.atleast_2d(g_hat)))

                if g_hat >= 0:
                    print('Safe action with g_hat: {}'.format(g_hat))
                else:
                    print('Unsafe action with g_hat: {}'.format(g_hat))

                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward[d])
                        epoch_episode_costs.append(episode_cost[d])
                        episode_rewards_history.append(episode_reward[d])
                        episode_costs_history.append(episode_cost[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0
                        episode_cost[d] = 0
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        if nenvs == 1:
                            agent.reset()


            # store new data
            agent.add_new_data(X_feature, Y_label, dataset_size=2000)
            # clear local feature and lable
            X_feature = np.empty((0, env.observation_space.shape[0] + env.action_space.shape[0]), dtype=np.float32)
            Y_label = np.empty((0, 1), dtype=np.float32)
            # dataset size
            logger.info("Dataset size: {}".format(agent.X_feature.shape))

            # Update GP hyperparameters
            old_log_likelihood = -np.inf
            diff_likelihood = 1
            while diff_likelihood > DIFF_TAU:
                log_likelihood = agent.gp_optimization()
                diff_likelihood = np.exp(log_likelihood) - np.exp(old_log_likelihood)
                old_log_likelihood = log_likelihood

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_guard_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al, gl = agent.train()
                epoch_critic_losses.append(cl)
                epoch_guard_losses.append(gl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
        combined_stats['rollout/safety_return'] = np.mean(epoch_episode_costs)
        combined_stats['rollout/safety_return_std'] = np.std(epoch_episode_costs)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['rollout/G_mean'] = np.mean(epoch_gs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/loss_guard'] = np.mean(epoch_guard_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)
        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)

        if callback: callback(locals(), globals())

    return agent


def cauchy_schwartz_check(x, y):
    #Compute the inner product and the norms
    inner_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    #Check whether the two input are the same
    if abs(inner_product - norm_x * norm_y) > 1e-4:
        return False
    else:
        return True

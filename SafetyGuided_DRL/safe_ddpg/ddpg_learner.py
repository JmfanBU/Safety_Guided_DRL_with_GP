from copy import copy
from functools import reduce

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd

from SafetyGuided_DRL.gp_models import GPR
from SafetyGuided_DRL.gp_models import RBF_regularized
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, actor, critic, guard, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf), safety_return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, guard_lr=1e-3, clip_norm=None, reward_scale=1., noise_delta=0.1, max_action=None):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.costs = tf.placeholder(tf.float32, shape=(None, 1), name='costs')  # safety costs
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.guard_target = tf.placeholder(tf.float32, shape=(None, 1), name='guard_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        self.mu = tf.placeholder(tf.float32, name='barrier_coefficient')
        self.X = tf.placeholder(tf.float32, shape=(None, observation_shape[0]+action_shape[0]), name='gp_feature')
        self.Y = tf.placeholder(tf.float32, shape=(None, 1), name='gp_label')
        self.obs_shape = observation_shape

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.safety_return_range = safety_return_range
        self.observation_range = observation_range
        self.critic = critic
        self.guard = guard
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.guard_lr = guard_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.guard_l2_reg = critic_l2_reg
        self.mu_value = 1e-3
        self.min_sigular_value = np.inf

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
                self.safety_ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None
            self.safety_ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic
        target_guard = copy(guard)
        target_guard.name = 'target_guard'
        self.target_guard = target_guard

        # Create networks and core TF parts that are shared across setup parts.
        # actor network
        self.actor_tf = actor(normalized_obs0)

        # gaussian process
        with tf.device('/device:GPU:1'):
            self.kernel = RBF_regularized(
                input_dim=observation_shape[0]+action_shape[0],
                ARD=True, noise=noise_delta) # gp kernel
            self.gp = GPR(self.X, self.Y, self.kernel)
            self.gp.likelihood.variance = noise_delta
            self.mean, self.cinterval = self.gp.build_evaluation(
                tf.concat([self.obs0, self.actions], axis=-1))
            self.mean_with_actor_tf, self.cinterval_with_actor_tf = self.gp.build_evaluation(
                tf.concat([self.obs0, self.actor_tf], axis=-1))
            self.test_var, self.test_beta = self.gp.test_function(
                tf.concat([self.obs0, self.actor_tf], axis=-1))
            self.log_likelihood = self.gp.likelihood_tensor
            self.cov_K = self.kernel.K(self.X)
            self.inv_K = tf.matrix_inverse(self.cov_K)
        self.X_feature = np.empty((0, observation_shape[0]+action_shape[0]), dtype=np.float32)
        self.Y_label = np.empty((0, 1), dtype=np.float32)

        # critic network
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf_gp = critic(normalized_obs0, self.actor_tf, self.mean_with_actor_tf, self.cinterval_with_actor_tf, self.mu, reuse=True)
        self.critic_with_actor_tf_gp = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf_gp, self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # guard network
        self.normalized_guard_tf = guard(normalized_obs0, self.actions)
        self.guard_tf = denormalize(tf.clip_by_value(self.normalized_guard_tf, self.safety_return_range[0], self.return_range[1]), self.safety_ret_rms)
        self.normalized_guard_with_actor_tf = guard(normalized_obs0, self.actor_tf, reuse=True)
        self.guard_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_guard_with_actor_tf, self.safety_return_range[0], self.return_range[1]), self.safety_ret_rms)
        G_obs1 = denormalize(target_guard(normalized_obs1, target_actor(normalized_obs1, reuse=True)), self.safety_ret_rms)
        self.target_G = self.costs + (1. - self.terminals1) * G_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        self.setup_guard_optimizer()
        self.setup_gp_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None  # recurrent architectures not supported yet

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        guard_init_updates, guard_soft_updates = get_target_updates(self.guard.vars, self.target_guard.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates, guard_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates, guard_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf_gp)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if var.name.endswith('/w:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_guard_optimizer(self):
        logger.info('setting up guard optimizer')
        normalized_guard_target_tf = tf.clip_by_value(normalize(self.guard_target, self.safety_ret_rms), self.safety_return_range[0], self.safety_return_range[1])
        self.guard_loss = tf.reduce_mean(tf.square(self.normalized_guard_tf - normalized_guard_target_tf))
        if self.guard_l2_reg > 0.:
            guard_reg_vars = [var for var in self.guard.trainable_vars if var.name.endswith('/w:0') and 'output' not in var.name]
            for var in guard_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.guard_l2_reg))
            guard_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.guard_l2_reg),
                weights_list=guard_reg_vars
            )
            self.guard_loss += guard_reg
        guard_shapes = [var.get_shape().as_list() for var in self.guard.trainable_vars]
        guard_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in guard_shapes])
        logger.info('  guard shapes: {}'.format(guard_shapes))
        logger.info('  guard params: {}'.format(guard_nb_params))
        self.guard_grads = U.flatgrad(self.guard_loss, self.guard.trainable_vars, clip_norm=self.clip_norm)
        self.guard_optimizer = MpiAdam(var_list=self.guard.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_gp_optimizer(self):
        with tf.device('/device:GPU:2'):
            self.gp_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-self.log_likelihood)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean
        self.old_cost_std = tf.placeholder(tf.float32, shape=[1], name='old_cost_std')
        new_cost_std = self.safety_ret_rms.std
        self.old_cost_mean = tf.placeholder(tf.float32, shape=[1], name='old_cost_mean')
        new_cost_mean = self.safety_ret_rms.mean

        self.renormalize_Q_outputs_op = []
        self.renormalize_G_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_varsi, self.guard.output_vars, self.target_guard.output_vars]:
            assert len(vs) == 4
            M, b, M_cost, b_cost = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]
            # Guard
            assert 'kernel' in M_cost.name
            assert 'bias' in b_cost.name
            assert M_cost.get_shape()[-1] == 1
            assert b_cost.get_shape()[-1] == 1
            self.renormalize_G_outputs_op += [M_cost.assign(M_cost * self.old_cost_std / new_cost_std)]
            self.renormalize_G_outputs_op += [b_cost.assign((b_cost * self.old_cost_std + self.old_cost_mean - new_cost_mean) / new_cost_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']
            ops += [self.safety_ret_rms.mean, self.safety_ret_rms.std]
            names += ['safety_ret_rms_mean', 'safety_ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        # Guard
        ops += [tf.reduce_mean(self.guard_tf)]
        names += ['reference_G_mean']
        ops += [reduce_std(self.guard_tf)]
        names += ['reference_G_std']

        ops += [tf.reduce_mean(self.guard_with_actor_tf)]
        names += ['reference_actor_G_mean']
        ops += [reduce_std(self.guard_with_actor_tf)]
        names += ['reference_actor_G_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def step(self, obs, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        if compute_Q:
            action, q, g = self.sess.run([actor_tf, self.critic_with_actor_tf, self.guard_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
            g = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action[0].shape
            action += noise

        if np.isnan(action).any():
            raise ValueError("action is None and obs: {}".format(obs))
        action = np.clip(action, self.action_range[0], self.action_range[1])

        return action, q, g, None

    def store_transition(self, obs0, action, reward, cost, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], cost[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })

            old_cost_mean, old_cost_std, target_G = self.sess.run([self.safety_ret_rms.mean, self.safety_ret_rms.std, self.target_G], feed_dict={
                self.obs1: batch['obs1'],
                self.costs: batch['costs'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.safety_ret_rms.update(target_G.flatten())
            self.sess.run(self.renormalize_G_outputs_op, feed_dict={
                self.old_cost_std : np.array([old_cost_std]),
                self.old_cost_mean : np.array([old_cost_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q, new_mean, new_std)
            # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            target_G = self.sess.run(self.target_G, feed_dict={
                self.obs1: batch['obs1'],
                self.costs: batch['costs'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss, self.guard_grads, self.guard_loss, self.test_var, self.test_beta]
        feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
            self.guard_target: target_G,
            self.X: self.X_feature,
            self.Y: self.Y_label,
            self.mu: self.mu_value
        }
        actor_grads, actor_loss, critic_grads, critic_loss, guard_grads, guard_loss, mean, cinterval = self.sess.run(ops, feed_dict=feed_dict)
        if np.isnan(actor_grads).any():
            print("B: {} \n; gamma_n: {}".format(mean, cinterval))
            raise ValueError("stop for debugging.")
            print("lower bound: {}".format(mean-cinterval))
            logger.info("Approach the safety boundary with gradient {}.".format(actor_grads))
            actor_grads[np.isnan(actor_grads)] = 0.0
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)
        self.guard_optimizer.update(guard_grads, stepsize=self.guard_lr)
        # print("actor grads: {}, mu value: {}".format(actor_grads.max(), self.mu_value))

        return critic_loss, actor_loss, guard_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer(), feed_dict=self.gp.update_feed_dict())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def gp_optimization(self):
        if self.min_sigular_value > 2e-5:
            feed_dict = {self.X: self.X_feature,
                         self.Y: self.Y_label}
            self.sess.run(self.gp_optimizer, feed_dict=feed_dict)
            return self.sess.run(self.log_likelihood, feed_dict=feed_dict)
        else:
            return None

    def add_new_data(self, x, y, dataset_size=100000):
        self.X_feature = np.vstack((self.X_feature, np.atleast_2d(x)))
        self.Y_label = np.vstack((self.Y_label, np.atleast_2d(y)))

        # Check sigularity of the convariance matrix and eliminate
        # dependent features
        feed_dict={self.X: self.X_feature}
        cov_K = self.sess.run(self.cov_K, feed_dict=feed_dict)
        idx = self.independency_check(cov_K)
        if idx is not None:
            self.eliminate_data_point(idx)

        length = self.X_feature.shape[0]
        if length > dataset_size:
            # eliminate data with lower independent score
            self.independent_score(dataset_size)

    def independency_check(self, cov_K):
        K = np.flip(np.flip(cov_K, 0), 1)
        q, r = np.linalg.qr(K)
        diag = np.absolute(np.diag(r))
        tol = 1e-5
        self.min_sigular_value = np.amin(diag)
        logger.info("Minimum singular value: {}".format(np.amin(diag)))
        if diag[diag > tol].shape[0] != cov_K.shape[0]:
            idx = np.where(diag <= tol)[0]
            logger.info("Eliminate {} instances due to sigularity.".format(idx.shape[0]))
            return cov_K.shape[0] - 1 - idx

    def eliminate_data_point(self, idx):
        self.X_feature = np.delete(self.X_feature, idx, axis=0)
        self.Y_label = np.delete(self.Y_label, idx, axis=0)

    def independent_score(self, dataset_size):
        new_data = self.X_feature[dataset_size:]
        new_label = self.Y_label[dataset_size:]
        logger.info("Detect {} instances with lowest independent score.".format(new_data.shape[0]))
        for data, label in zip(new_data, new_label):
            X = np.vstack((self.X_feature[:dataset_size], np.atleast_2d(data)))
            Y = np.vstack((self.Y_label[:dataset_size], np.atleast_2d(label)))
            scores = np.zeros(dataset_size + 1)
            feed_dict = {self.X: X}
            K_inv = self.sess.run(self.inv_K, feed_dict=feed_dict)
            scores = 1/np.diag(K_inv)

            min_id = np.where(scores == scores.min())[0]
            self.eliminate_data_point(min_id[-1])

    def gp_validation(self):
        feed_dict = {self.obs0: self.X_feature[:,:self.obs_shape[0]],
                     self.actions: self.X_feature[:,self.obs_shape[0]:],
                     self.X: self.X_feature,
                     self.Y: self.Y_label}
        #feed_dict.update(self.gp.update_feed_dict())
        fmean = self.sess.run(self.mean, feed_dict=feed_dict)
        error = (fmean - self.Y_label) ** 2
        accuracy = np.sum(error < 1e-5)/ float(fmean.shape[0])
        mse = np.mean((fmean - self.Y_label)**2)

        return accuracy, mse

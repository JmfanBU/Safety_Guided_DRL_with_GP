import tensorflow as tf
from baselines.common.models import get_network_builder


class Model(object):
    def __init__(self, name, network='mlp', activation_type='relu', **network_kwargs):
        self.name = name
        if activation_type == 'relu':
            self.activation = tf.nn.relu
        elif activation_type == 'tanh':
            self.activation = tf.nn.tanh
        elif activation_type == 'sigmoid':
            self.activation = tf.nn.sigmoid
        self.network_builder = get_network_builder(network)(activation=self.activation, **network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', activation_type='relu',**network_kwargs):
        super().__init__(name=name, network=network, activation_type=activation_type, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network_builder(obs)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', network='mlp', activation_type='relu', **network_kwargs):
        super().__init__(name=name, network=network, activation_type=activation_type, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, mean=None, cinterval=None, mu=None, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1) # this assumes observation and action can be concatenated
            x = self.network_builder(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
            if mean is not None and cinterval is not None and mu is not None:
                # x = x + mu * tf.log(tf.clip_by_value(mean - cinterval, 1e-20, 1.0)) + 0.1 * tf.exp(-tf.square(mean - cinterval))
                # x = x + mu * tf.log(tf.nn.relu(mean - cinterval)) + 0.1 * tf.exp(-tf.square(mean - cinterval))
                x = x - 110 * tf.nn.relu(-mean + cinterval) + 10 * tf.exp(-tf.square(mean - cinterval))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class Guard(Model):
    def __init__(self, name='guard', network='mlp', activation_type='relu', **network_kwargs):
        super().__init__(name=name, network=network, activation_type=activation_type, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1) # this assumes observation and action can be concatenated
            x = self.network_builder(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

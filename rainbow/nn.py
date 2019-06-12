"""
Modification of NN architecture for mineRL
"""

import math
from functools import partial

import tensorflow as tf

from anyrl.models.dqn_scalar import noisy_net_dense
from anyrl.models.dqn_dist import DistQNetwork
from anyrl.models.util import product

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def mine_cnn(obs_batch):
    """
    Apply the CNN architecture from the Nature DQN paper.

    The result is a batch of feature vectors.
    """
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.orthogonal_initializer(gain=math.sqrt(2))
    }
    with tf.variable_scope('layer_1'):
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 3, 2, **conv_kwargs)
    with tf.variable_scope('layer_2'):
        cnn_2 = tf.layers.conv2d(cnn_1, 64, 3, 1, **conv_kwargs)
    with tf.variable_scope('layer_3'):
        cnn_3 = tf.layers.conv2d(cnn_2, 64, 3, 1, **conv_kwargs)
    flat_size = product([x.value for x in cnn_3.get_shape()[1:]])
    flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))

    # The orthogonal initializer appears to be unstable
    # for large matrices. With ortho init, I see huge
    # max outputs for some environments.
    del conv_kwargs['kernel_initializer']

    return flat_in

def mine_nn(obs_batch):

    s = shape_list(obs_batch)

    obs_batch = tf.reshape(obs_batch, (-1, 64*64*3+1, 4))
    pov, compass = obs_batch[:, :-1, :], obs_batch[:, -1:, :]

    pov = tf.reshape(pov, [-1, 64, 64, 3, 4])
    pov = tf.transpose(pov, perm=[0,3,4,1,2])
    pov = tf.reshape(pov, [-1, 12, 64, 64])

    compass = tf.layers.flatten(compass)

    pov_h_flat = mine_cnn(pov)
    h_flat = tf.concat([pov_h_flat, compass], axis=-1)
    return tf.layers.dense(h_flat, 512, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(gain=math.sqrt(2)))


class MineDistQNetwork(DistQNetwork):
    """
    A distributional Q-network model for mineRL

    """

    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 num_atoms,
                 min_val,
                 max_val,
                 dueling=False,
                 dense=tf.layers.dense,
                 input_dtype=tf.uint8,
                 input_scale=1 / 0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        super().__init__(session, num_actions, obs_vectorizer, name,
                         num_atoms, min_val, max_val,
                         dueling=dueling, dense=dense)

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        return mine_nn(obs_batch)


def mine_rainbow_online_target(session,
                               num_actions,
                               obs_vectorizer,
                               num_atoms=51,
                               min_val=-10,
                               max_val=10,
                               sigma0=0.5):
    """
    Create the models used for Rainbow
    (https://arxiv.org/abs/1710.02298).

    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      num_atoms: number of distribution atoms.
      min_val: minimum atom value.
      max_val: maximum atom value.
      sigma0: initial Noisy Net noise.

    Returns:
      A tuple (online, target).
    """
    def maker(name):
        return MineDistQNetwork(session, num_actions, obs_vectorizer, name,
                                  num_atoms, min_val, max_val, dueling=True,
                                  dense=partial(noisy_net_dense, sigma0=sigma0))
    return maker('online'), maker('target')
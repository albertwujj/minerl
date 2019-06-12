#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import minerl

from env_util import NavigateWrapper, DiscreteActionWrapper, get_env

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
from nn import mine_rainbow_online_target

def main():
    """Run DQN until the environment throws an exception."""
    env = NavigateWrapper(DiscreteActionWrapper(get_env()))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        online, target = mine_rainbow_online_target(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200)
        dqn = DQN(online, target)
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)


if __name__=='__main__':
    main()

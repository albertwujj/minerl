#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import minerl
import tensorflow as tf
from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer


from nn import mine_rainbow_online_target, mine_cnn
from eval_util import evaluate
from env_util import SimpleNavigateEnvWrapper, get_env
from experts_util import ImitationPlayer


def main():

    env_name = 'MineRLNavigateExtremeDense-v0'
    """Run DQN until the environment throws an exception."""
    base_env = [SimpleNavigateEnvWrapper(get_env(env_name)) for _ in range(1)]
    env = BatchedFrameStack(BatchedGymEnv([base_env]), num_images=4, concat=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        online, target = mine_rainbow_online_target(mine_cnn, sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200)
        dqn = DQN(online, target)
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())

        buffer_capacity = 500000

        replay_buffer = PrioritizedReplayBuffer(buffer_capacity, 0.5, 0.4, epsilon=0.1)
        data_iter = minerl.data.make(env_name).seq_iter(num_epochs=1, max_sequence_len=32)
        expert_player = NStepPlayer(ImitationPlayer(data_iter, 500000), 3)

        for traj in expert_player.play():
            replay_buffer.add_sample(traj, init_weight=1)

        print('starting training')
        dqn.train(num_steps=5000,
                  player=player,
                  replay_buffer=replay_buffer,
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

        print('starting eval')
        player._cur_states = None
        score = evaluate(player)
        print(score)

if __name__=='__main__':
    main()

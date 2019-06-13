from collections import deque

import minerl
import numpy as np


def single_seq_iter(seq_iter):
    for obs_batch, rew_batch, done_batch, act_batch in seq_iter:

        batch_size = obs_batch[list(obs_batch.keys())[0]].shape[0]

        # TODO: Make recursive for nested dicts.
        obs_batch = [{key: values[i] for key, values in obs_batch.items() if key != 'inventory'} for i in range(batch_size)]
        act_batch = [{key: values[i] for key, values in act_batch.items()} for i in range(batch_size)]
        
        for i in range(batch_size):
            yield obs_batch[i], rew_batch[i], done_batch[i], act_batch[i]

def simple_navigate_experts(env_name, max_trajs=500000):
    data = minerl.data.make(env_name)
    obses = []
    model_outs = []
    rewards = []
    episode_step = []
    total_reward = []

    t = 0
    ret = 0
    framestack_obs = deque(maxlen=4)

    trajs = 0
    for obs, rew, done, act in single_seq_iter(data.seq_iter(num_epochs=1, max_sequence_len=32)):
        if trajs >= max_trajs:
            break
        if trajs and trajs % 1000 == 0:
            print(f'Trajs # {trajs}')
        # obs, model_outs, rewards, new_obs, episode_step, is_last, total_reward
        while len(framestack_obs) < 4:
            framestack_obs.append(obs['pov'])

        obses.append(np.concatenate(framestack_obs, axis=-1))
        rewards.append(rew)
        ret += rew
        total_reward.append(ret)
        model_outs.append({'action': act})
        episode_step.append(t)

        trajs += 1
        t += 1
        if done:
            t = 0

    transes = []
    for obs, model_out, reward, new_ob in zip(obses, model_outs, rewards, obses[1:]):
        transes.append({'obs': obs, 'model_outs': model_out, 'rewards': reward, 'new_obs': new_ob, 'weight':1})
    return transes


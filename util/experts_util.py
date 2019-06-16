import time
from collections import deque
from itertools import islice

import minerl
import numpy as np
from anyrl.rollouts import Player


def single_seq_iter(seq_iter):
    for obs_batch, rew_batch, done_batch, act_batch in seq_iter:

        ok = True
        for row in obs_batch['pov']:
            if row.shape != (64,64,3):
                ok = False

        if ok:
            batch_size = obs_batch[list(obs_batch.keys())[0]].shape[0]

            # TODO: Make recursive for nested dicts.
            obs_batch = [{key: values[i] for key, values in obs_batch.items() if key != 'inventory'} for i in range(batch_size)]
            act_batch = [{key: values[i] for key, values in act_batch.items()} for i in range(batch_size)]

            for i in range(batch_size):
                yield obs_batch[i], rew_batch[i], done_batch[i], act_batch[i]

def non_bugged_data_arr(env_name, num_trajs):
    ret = []
    bugged_iter = minerl.data.make(env_name).seq_iter(num_epochs=1, max_sequence_len=32)
    bugged_iter = single_seq_iter(bugged_iter)
    for obs, rew, done, act in islice(bugged_iter, num_trajs):
        if obs['pov'].shape == (64, 64, 3):
            ret.append((obs, rew, done, act))
    return ret

class SingularIterator():
    def __init__(self, env_name, num_trajs=100000):
        self.arr = non_bugged_data_arr(env_name, num_trajs=num_trajs)
        self.i = 0
        self.in_the_middle = False

    def get_one_traj(self):
        ret = []
        for obs, rew, done, act in self.arr:
            ret.append((obs, rew, done, act))
            if done:
                break

            self.i += 1
        return ret

    def get_all(self):
        return self.arr


class ImitationPlayer(Player):
    """
    A Player that uses an iterator to gather
    sequential batches of 1-step transitions, including 'model outs'
    """

    def __init__(self, batch_seq, batch_size):
        self.single_seq = single_seq_iter(batch_seq)
        self.batch_size = batch_size
        self._needs_reset = True
        self._last_obs = None
        self._episode_id = -1
        self._episode_step = 0
        self._total_reward = 0.0


    def play(self):
        print("imitation play")
        ret = []
        for start_obs, _, _, act in self.single_seq:
            if self._needs_reset:
                self._needs_reset = False
                self._framestack_obs = deque([start_obs['pov']] * 4, maxlen=4)
                self._last_obs = np.concatenate(self._framestack_obs, -1)
                self._last_acts = act
                self._episode_id += 1
                self._episode_step = 0
                self._total_reward = 0.0

            new_obs_single, rew, self._needs_reset, new_act = next(self.single_seq)
            self._total_reward += rew

            self._framestack_obs.append(new_obs_single['pov'])
            new_obs = np.concatenate(self._framestack_obs, -1)
            res = {
                'obs': self._last_obs,
                'model_outs': {'actions': self._last_acts},
                'rewards': [rew],
                'new_obs': (new_obs if not self._needs_reset else None),
                'info': {},
                'episode_id': self._episode_id,
                'episode_step': self._episode_step,
                'end_time': time.time(),
                'is_last': self._needs_reset,
                'total_reward': self._total_reward
            }
            self._last_acts = new_act
            self._episode_step += 1
            ret.append(res)
        return ret


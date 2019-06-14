import time
from collections import deque

import minerl
import numpy as np
from anyrl.rollouts import Player

from debug_util import check

def single_seq_iter(seq_iter):
    for obs_batch, rew_batch, done_batch, act_batch in seq_iter:

        batch_size = obs_batch[list(obs_batch.keys())[0]].shape[0]

        for row in obs_batch['pov']:
            if row.shape != (64,64,3):
                print('bug in mineRL')
                print(row.shape)


        # TODO: Make recursive for nested dicts.
        obs_batch = [{key: values[i] for key, values in obs_batch.items() if key != 'inventory'} for i in range(batch_size)]
        act_batch = [{key: values[i] for key, values in act_batch.items()} for i in range(batch_size)]

        for i in range(batch_size):


            yield obs_batch[i], rew_batch[i], done_batch[i], act_batch[i]

class ImitationPlayer(Player):
    """
    A Player that uses an iterator to gather
    sequential batches of 1-step transitions, including 'model outs'
    """

    def __init__(self, batch_data_iter, batch_size, obs_func=lambda obs: obs['pov']):
        self.seq_iter = single_seq_iter(batch_data_iter)
        self.batch_size = batch_size
        self._needs_reset = True
        self._last_obs = None
        self._episode_id = -1
        self._episode_step = 0
        self._total_reward = 0.0
        self._obs_func = obs_func


    def play(self):
        return [self._gather_transition() for _ in range(self.batch_size)]

    def _gather_transition(self):
        if self._needs_reset:
            self._needs_reset = False
            start_obs, _, _, act = next(self.seq_iter)
            self._framestack_obs = deque([start_obs['pov']] * 4, maxlen=4)
            self._last_obs = np.concatenate(self._framestack_obs, -1)
            self._last_acts = act
            self._episode_id += 1
            self._episode_step = 0
            self._total_reward = 0.0


        new_obs_single, rew, self._needs_reset, new_act = next(self.seq_iter)
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
        return res

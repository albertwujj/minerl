from functools import reduce
from operator import mul

import gym
import numpy as np
from gym.core import Wrapper, ObservationWrapper, ActionWrapper

import minerl

def get_env(env_name='MineRLNavigateDense-v0'):
    env = gym.make(env_name)
    return env

class ComboDiscrete():
    def __init__(self, sizes):
        self.sizes = sizes
        self.n = reduce(mul, sizes)

    def num_to_options(self, num):
        option_selections = []
        for size in self.sizes:
            option_selections.append(num // size)
            num = num % size
        return option_selections

class SimpleNavigateEnvWrapper(Wrapper):
    def __init__(self, env):
        super(SimpleNavigateEnvWrapper, self).__init__(env)
        self.last_compass = 0
        self.observation_space=gym.spaces.Box(0, 255, shape=(64,64,3))
        self.action_space = gym.spaces.Discrete(4)

    def step(self, num):
        action = self.env.action_space.noop()
        action['camera'] = [0, (num * .005 + .02) * self.last_compass]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1
        obs, reward, done, info = self.env.step(action)
        self.last_compass = obs['compassAngle']
        info['compassAngle'] = obs['compassAngle']
        return obs['pov'], reward, done, info

    def reset(self):
        obs, info = self.env.reset()
        self.last_compass = obs['compassAngle']
        info['compassAngle'] = obs['compassAngle']
        return obs['pov']


camera_actions = 9
movement_actions = 4
class NavigateActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(NavigateActionWrapper, self).__init__(env)
        self.discretizer = ComboDiscrete([4])
        self.action_space = gym.spaces.Discrete(self.discretizer.n)

    def action(self, num):
        option_selections = self.discretizer.num_to_options(num)
        coef = option_selections[0] * .01 + .015
        action = self.env.action_space.noop()
        action['camera'] = coef
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1
        return action


class NavigateWrapper(ObservationWrapper):
    def __init__(self, env):
        super(NavigateWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(0, 255,shape=(64*64*3+1,))

    def observation(self, base_obs):
        base_obs, info = base_obs
        obs = np.concatenate([base_obs['pov'].flatten(), [base_obs['compassAngle']]])
        return obs


import torch
class PytorchWrapper(Wrapper):
    def __init__(self, env, device=torch.device('cuda:0')):
        super(PytorchWrapper, self).__init__(env)
        self.device = device

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs


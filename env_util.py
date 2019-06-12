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

camera_actions = 9
movement_actions = 4
class DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.discretizer = ComboDiscrete([6,3,3,2])
        self.action_space = gym.spaces.Discrete(self.discretizer.n)


    def action(self, num):

        option_selections = self.discretizer.num_to_options(num)

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
        obs, reward, done, info = self.env.step(action)=

        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs


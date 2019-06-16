"""
[pitch, yaw] = [vertical, horizontal]
"""

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




class FullNavigateEnvWrapper(Wrapper):
    # TODO: IN PROGRESS
    def __init__(self, env):
        super(FullNavigateEnvWrapper, self).__init__(env)
        self.last_compass = 0
        self.observation_space=gym.spaces.Box(0, 255, shape=(64,64,3))
        self.action_space = gym.spaces.Discrete(4)
        self.discretizer = ComboDiscrete([3])

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

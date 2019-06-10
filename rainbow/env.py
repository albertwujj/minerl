import minerl
import gym
import numpy as np

def get_env(env_name='MineRLNavigateDense-v0'):
    env = gym.make(env_name)
    return env

class DiscreteNavigateWrapper():
    def __init__(self, env):
        self.env = env

    def __wrap_obs(self, base_obs):
        obs = []
        obs.append()
        obs.append(base_obs['compassAngle'])
        obs = np.concatenate([base_obs['pov'].flatten(), base_obs['compassAngle'].flatten()])
        return obs

    def step(self, action):
        base_obs, reward, done, info = self.env.step(action)
        return self.__wrap_obs(base_obs), reward, done, info

    def reset(self):
        base_obs, info = self.env.reset()
        return self.__wrap_obs(base_obs), info
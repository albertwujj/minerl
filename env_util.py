import minerl
import gym
import numpy as np
from gym.core import Wrapper, ObservationWrapper, ActionWrapper
from gym.spaces import Space

def get_env(env_name='MineRLNavigateDense-v0'):
    env = gym.make(env_name)
    return env

class DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.action_space = Discrete(200)

    def action(self, discrete_action):

        camera = discrete_action // 9
        camera = [int(character) for character in bin(camera)]
        pitch = camera[0]
        yaw = camera[1]
        discrete_action = discrete_action % 9

        wasd = discrete_action // 4
        wasd = [True if i == wasd else False for i in range(4)]
        left, right, forward, back = wasd
        rest_of_options = discrete_action % 4

        rest_of_options = [int(character) for character in bin(rest_of_options)]
        attack = rest_of_options[4]
        jump = rest_of_options[5]
        place = rest_of_options[6]

        action = self.action_space.noop()
        action['camera'] = [pitch, yaw]
        action['left'] = left
        action['right'] = right
        action['forward'] = forward
        action['back'] = back
        action['attack'] = attack
        action['jump'] = jump
        action['place'] = place

        return action

class NavigateWrapper(ObservationWrapper):
    def __init__(self, env):
        super(NavigateWrapper, self).__init__(env)

    def observation(self, base_obs):
        obs = []
        obs.append()
        obs.append(base_obs['compassAngle'])
        obs = np.concatenate([base_obs['pov'].flatten(), base_obs['compassAngle'].flatten()])
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
        obs, info = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info


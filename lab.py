import minerl
from minerl.env.core import MineRLEnv
import gym
env = gym.make('MineRLNavigateDense-v0')
print('made env')

obs, _ = env.reset()
done = False
net_reward = 0

print('starting run')

action = env.action_space.noop()
print(vars(action))
obs, reward, done, info = env.step(
    action)
print(vars(obs))


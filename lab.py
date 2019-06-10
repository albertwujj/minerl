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

    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1
    print()

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)

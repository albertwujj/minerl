import minerl
import gym
env = gym.make('MineRLNavigateExtremeDense-v0')


obs, _ = env.reset()
done = False
net_reward = 0

while not done:
    action = env.action_space.noop()

    print(obs['compassAngle'])
    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 0

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)

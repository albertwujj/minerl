import minerl
import gym

def evaluate(player):
    transes = player.play()
    rets = [transes[i]['total_reward'] for i in range(len(transes))]
    avg_ret = sum(rets) / len(rets)
    return avg_ret
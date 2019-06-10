import minerl
import os

data_dir = '/home/paperspace/pycharm/albert/minerl/data/'
print(os.path.exists(data_dir))
data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='/home/paperspace/pycharm/albert/minerl/data/')



# Iterate through a single epoch gathering sequences of at most 32 steps
for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=32):
    print("Number of diffrent actions:", len(act))
    for action in act:
        print(act)
    print("Number of diffrent observations:", len(obs), obs)
    #for observation in obs:
        #print(obs)
    print("Rewards:", rew)
    print("Dones:", done)

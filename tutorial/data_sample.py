import minerl

data = minerl.data.make('MineRLNavigateExtremeDense-v0')

pov_shape = (64,64,3)
# Iterate through a single epoch gathering sequences of at most 32 steps
for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=32):
    for x in obs['pov']:
        if x.shape != pov_shape:
            print(f'mineRL bug, pov shape: {x.shape}')

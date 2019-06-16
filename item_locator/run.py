import minerl

from backtrack import get_visibles
from model import LocationModel
from util.experts_util import SingularIterator


env_name = 'MineRLTreechop-v0'
model = LocationModel()

back_traj = None
num_trajs = 10000
iter = SingularIterator(env_name, num_trajs=num_trajs)

test_obs, test_locations, test_angles = None, None, None

for t in range(num_trajs):
    if t and t % 100 == 0:
        print(f'timestep {t}')

    traj = iter.get_one_traj()

    for i, (obs, rew, done, act) in enumerate(traj):
        if rew > 0: # acquired a wood
            back_traj = traj[:i+1]

    if back_traj is None:
        continue

    obses, locations, angles = get_visibles(back_traj)
    if not obses:
        continue


    test_obs, test_locations, test_angles = obses, locations, angles

    print('training')
    model.train(obses, locations, angles)
    back_traj = None


pred_locations, pred_angles = model.predict(test_obs)
print(pred_locations.mean(), ((pred_locations - test_locations) **2).mean())


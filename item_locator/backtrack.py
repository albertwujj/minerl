import numpy as np


def get_visibles(back_traj):

    # returned visuals and corresponding item locations
    obses = []
    angles = []
    locations = []


    angle = 0
    location = np.zeros((2,)) # vert, horiz
    for i, (obs, rew, done, act) in enumerate(back_traj):

        # 'starting' camera angle
        angle_delta = np.asarray(act['camera'][1])
        prev_angle = angle - angle_delta

        # WASD movements
        left = act['left']
        right = act['right']
        forward = act['forward']
        back = act['back']
        vert = forward - back
        horiz = right - left

        # get movement angle
        unanchored_theta = np.arcsin(vert/horiz)
        anchored_theta = prev_angle + unanchored_theta

        # get movement vector
        dist = 1
        if (left + right + forward + back == 2) and not ((left and right) or (forward and back)):
            dist = np.sqrt(2)
        location -= np.asarray([np.sin(anchored_theta) * dist, np.cos(anchored_theta) * dist])
        angle = prev_angle


        theta_item = np.arctan(location[0]/location[1])
        ideal = 180 + theta_item # yes

        range = 45
        if angle > ideal - range and angle < ideal + range: # item is visible
            obses.append(obs)
            angles.append(np.asarray([-angle]))
            locations.append(-locations)


    return obses, locations, angles


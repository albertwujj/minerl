def check(obs):
    if obs.shape != (64, 64, 3):
        print(obs.shape)
        return False
    return True
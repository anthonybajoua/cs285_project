import numpy as np

class SampleEnv(object):
    def __init__(self):
        self.observation_space = np.zeros((1000, 3))
        self.action_space = np.zeros((1000))

    def get_action(self):
        return

    def seed(self, s):
        self.seed_ = s
        return

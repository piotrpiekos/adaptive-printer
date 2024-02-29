from scipy.io import loadmat

import numpy as np


class RealSimulator:
    def __init__(self, box_max=1099, box_min=0):
        matlab_corruptions = loadmat('data/simulation/add_dudw_data.mat')
        self.corruptions = matlab_corruptions['all_data'][0:5000]

        self.box_max = box_max
        self.box_min = box_min

    def get_next_step(self, x, direction, corr_id):
        """
        direction - can be 1 or -1
        """
        return np.clip(x + direction * self.corruptions[int(corr_id) - 1, int(round(x)) - 1],
                       self.box_min, self.box_max)

    def get_next1d(self, x: float, ctrl_x: int, corr_id: int):
        direction = np.sign(ctrl_x)
        cur_x = x
        while ctrl_x != 0:
            cur_x = self.get_next_step(cur_x, direction, corr_id)
            ctrl_x -= direction
        return cur_x

    def get_next(self, state, controls, corr_ids):
        return self.get_next1d(state[0], controls[0], corr_ids[0]), self.get_next1d(state[1], controls[1], corr_ids[1])

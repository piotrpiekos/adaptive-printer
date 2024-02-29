import numpy as np

from src.Simulator import RealSimulator

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import lightgbm as lgb


class DirectClassicML:
    def adapt_to_traj(self, T):
        X, y = T[:, [4, 9, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]] # [ideal_xi, ideal_yi, cx, cy] -> [diff_xi, diff_yi]
        X[:, [0, 1]] = X[:, [0, 1]] % 200
        self.model.fit(X, y)

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2, 3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2, 3]])
        inputs[:, [0, 1]] = inputs[:, [0, 1]] % 200

        diff = self.model.predict(inputs)
        return abs_vals * diff

class DirectKNN(DirectClassicML):
    def __init__(self, k):
        super(DirectKNN, self).__init__()
        self.model = KNeighborsRegressor(k, weights='distance')


class DirectRandomForest(DirectClassicML):
    def __init__(self, max_leaf_nodes=None):
        super(DirectRandomForest, self).__init__()
        self.model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes)


class SmallRandomForest:
    def __init__(self):
        self.model1, self.model2 = None, None

    def adapt_to_traj(self, T):
        self.model1, self.model2 = RandomForestRegressor(), RandomForestRegressor()
        X, y = T[:, [4, 9, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]
        #X[:, [0, 1]] =X[:, [0,1]] % 200

        x_sel, y_sel = X[:, 2]!=0, X[:, 3]!=0
        self.model1.fit(X[x_sel][:, [0]], y[x_sel, 0]*X[x_sel, 2])
        self.model2.fit(X[y_sel][:, [1]], y[y_sel, 1]*X[y_sel, 3])

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return np.abs(control) * diff

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2, 3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2, 3]])
        #inputs[:, [0, 1]] = inputs[:, [0, 1]] % 200

        diff1, diff2 = self.model1.predict(inputs[:, [0]])*inputs[:, 2], self.model2.predict(inputs[:, [1]])*inputs[:, 3]
        diff = np.stack((diff1, diff2)).T
        return abs_vals * diff


class DirectLightGBM:
    def __init__(self):
        self.model1, self.model2 = None, None

    def adapt_to_traj(self, T):
        self.model1, self.model2 = lgb.LGBMRegressor(), lgb.LGBMRegressor()
        X, y = T[:, [4, 9, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]

        self.model1.fit(X, y[:, 0])
        self.model2.fit(X, y[:, 1])

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2, 3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2, 3]])

        diff1, diff2 = self.model1.predict(inputs), self.model2.predict(inputs)
        diff = np.stack((diff1, diff2)).T
        return abs_vals * diff



class SmallLightGBM:
    def __init__(self):
        self.model1, self.model2 = None, None

    def adapt_to_traj(self, T):
        self.model1, self.model2 = lgb.LGBMRegressor(), lgb.LGBMRegressor()
        X, y = T[:, [4, 9, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]

        x_sel, y_sel = X[:, 2]!=0, X[:, 3]!=0
        self.model1.fit(X[x_sel][:, [0]], y[x_sel, 0]*X[x_sel, 2])
        self.model2.fit(X[y_sel][:, [1]], y[y_sel, 1]*X[y_sel, 3])

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2, 3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2, 3]])

        diff1, diff2 = self.model1.predict(inputs[:, [0]])*inputs[:, 2], self.model2.predict(inputs[:, [1]])*inputs[:, 3]
        diff = np.stack((diff1, diff2)).T
        return abs_vals * diff


class DirectLinearRegression(DirectClassicML):
    def __init__(self):
        self.model = LinearRegression()


class Oracle:
    def __init__(self):
        self.sim = RealSimulator()
        self.corr_ids = None

    def adapt_to_traj(self, T):
        self.corr_ids = T[0, [1, 2]]

    def __call__(self, opt_state, control):
        return self.sim.get_next(opt_state, control, self.corr_ids) - opt_state

    def batch_call(self, inputs):
        diffs = []
        for row in inputs:
            opt_state = row[:2]
            control = row[2:]
            diff = self(opt_state, control)
            diffs.append(diff)
        return np.stack(diffs)



class Naive:
    def __init__(self):
        pass

    def adapt_to_traj(self, T):
        pass

    def __call__(self, state, control):
        return control[0],  control[1]

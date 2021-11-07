import numpy as np

eps = 1e-8

class PortfolioSimulator(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names, config):
        self.asset_names = asset_names
        self.config = config
        self.cost = self.config['trading_cost']
        self.time_cost = self.config['time_cost']
        self.steps = self.config['max_ep_length']
        self.infos = []
        # if have cash asset, initial weight should be [1, 0, 0, ....]
        self.w0 = self._init_weights()
        self.dw = self._init_weights()  # w_prime record the weight change in the period
        self.p0 = 1.0
        self.l2 = 0.3  # lambda of risk regularization penalty
        self.p_rtn_history = list()  # store log return history for calculating sharpe ratio
        self.eps = eps

    def _init_weights(self):
        if self.config['add_cash_asset']:
            return np.array([1.0] + [0.0] * (len(self.asset_names) - 1))
        else:
            return np.array([0.0] * len(self.asset_names))

    def step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        # print(w1, y1)
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        if self.config['add_cash_asset']:
            assert y1[0] == 1.0, 'y1[0] must be 1 if added cash'

        w0 = self.w0
        p0 = self.p0

        # dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into
        dw0 = self.dw / np.sum(np.abs(self.dw))  # modified from original for weight from -1 to 1 instead 0 to 1

        mu1 = self.cost * (np.abs(dw0[1:] - w1[1:])).sum()  # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        # p1 = p0 * (1 - mu1) * np.dot(y1, w0)  # (eq11) final portfolio value
        p1 = p0 * (1 - mu1) * (1 + np.sum((y1 - 1) * w1))  # modified from original for weight from -1 to 1 instead 0 to 1

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns

        # 1) reward option 1
        r1 = np.log((p1 + self.eps) / (p0 + self.eps))  # log rate of return
        reward = r1 - self.l2 * np.square(r1)  # add a l2 regularization penalty for risk variance element
        # reward = r1 / self.steps * 1000  # (22) average logarithmic accumulated return

        # 2) reward option 2: average sharpe ratio
        # self.p_rtn_history.append(r1)
        # if len(self.p_rtn_history) <= 2:
        #     reward = 0
        # else:
        #     reward = (np.mean(self.p_rtn_history) / np.std(self.p_rtn_history)) - (np.mean(self.p_rtn_history[:-1]) / np.std(self.p_rtn_history[:-1]))
        # reward = reward * 100
        # print(reward)

        # remember for next step
        self.w0 = w1
        self.dw = (y1 - 1) * w1 * np.sign(w1) + w1
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 <= 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)

        return reward, info, done

    def reset(self):
        self.infos.clear()
        self.w0 = self._init_weights()
        self.dw = self._init_weights()
        self.p_rtn_history.clear()
        self.p0 = 1.0
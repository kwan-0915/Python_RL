import gym
import numpy as np
import pandas as pd
from pprint import pprint

import utilities.visualization
from utilities import utils
from signals.signals_controller import SignalsController
from portfolio.portfolio_simulator import PortfolioSimulator

class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, config):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
        """
        self.config = config
        self.window_length = self.config['window_length']
        self.infos = []

        # setup data source by data controller
        self.data_controller = SignalsController(self.config)
        self.num_asset = self.data_controller.num_asset
        self.ticker_names = self.data_controller.ticker_names

        # load data feed from data controller when reset
        self.data_feed = None

        self.sim = PortfolioSimulator(asset_names=self.ticker_names, config=self.config)

        # openai gym attributes
        # action will be the portfolio weights from -1 to 1 for each asset
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.num_asset,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.set_observation_space(shape=(self.num_asset, self.window_length, self.data_controller.data.shape[-1]))

    def set_observation_space(self, shape: tuple):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from -1 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim.step for description
        """
        # print('portfolio core action: ', action.shape)
        np.testing.assert_almost_equal(action.shape, self.num_asset)

        # normalise just in case
        action = np.clip(action, -1, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]

        assert ((action >= -1) * (action <= 1)).all(), 'all action values should be between -1 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(np.abs(weights)), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        next_state, data_done = self.data_controller.step()

        # relative price vector of last observation day (close/open)
        close_price_vector = next_state[:, -1, 3]
        prev_close_price_vector = next_state[:, -2, 3]
        y1 = close_price_vector / prev_close_price_vector
        reward, info, sim_done = self.sim.step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.data_controller.get_current_date()
        info['steps'] = self.data_controller.current_step
        self.infos.append(info)

        done = data_done or sim_done
        return next_state, reward, done, info

    def reset(self):
        self.infos.clear()
        self.sim.reset()
        state = self.data_controller.reset()
        # load data feed from data controller after data controller reset
        self.data_feed = self.data_controller.data_feed
        return state

    def render(self, mode='human'):
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = utils.max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = utils.sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        utilities.visualization.make_figure(df_info.index, df_info["portfolio_value"], df_info["market_value"], title=title, xtitle='Timesteps', ytitle='Value')
        # make_figure(df_info.index.strftime("%Y/%m/%d"), df_info["portfolio_value"], df_info["market_value"], title=title, xtitle='Timesteps', ytitle='Value')

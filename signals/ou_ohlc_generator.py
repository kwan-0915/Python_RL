import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utils import make_figure

class OuOHLCGenerator:
    """
    Ornstein-Uhlenbeck motion (brownian random walk) data generator for sanity check.
    noise and damping can be added.
    """

    def __init__(self, seed, start_date, num_asset, num_data, sigma=1.0, tau=0.05):
        if seed:
            self.seed = seed
            np.random.seed(self.seed)

        self.start_date = start_date
        self.ticker_names = ['ou_{}'.format(i) for i in range(num_asset)]

        self.num_asset = num_asset
        self.num_data = num_data

        self.sigma = sigma
        self.tau = tau

    @staticmethod
    def ou_gen(dt=0.001,
               sigma=1.0,
               mu=100.0,
               tau=0.05,
               verbose=0):
        """Generate simulated stock data via Ornstein-Uhlenbeck process"""

        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)

        x = mu
        t = 0
        yield t, x, mu

        while True:
            x = x + dt * (-(x - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()
            t += dt
            yield t, x, mu

    @staticmethod
    def plot_df(df):
        make_figure(df.index, df['close'],
                    title="Simulated Stock Price Data As Ornstein-Uhlenbeck process",
                    xtitle='Timesteps',
                    ytitle='Value'
                    )

    def get_ou_ohlc_df(self, total_time, sigma, mu, tau):
        gen = self.ou_gen(dt=total_time / self.num_data,
                          sigma=sigma,
                          mu=mu,
                          tau=tau)

        trend_series = []
        stock_series = []
        time_series = []

        for i in range(self.num_data):
            t, stock_price, trend_index = next(gen)
            stock_series.append(stock_price)
            trend_series.append(trend_index)
            time_series.append(t)

        df = pd.DataFrame({'open': stock_series,
                           'high': stock_series,
                           'low': stock_series,
                           'close': stock_series,
                           'volume': trend_series},
                          index=pd.bdate_range(start=self.start_date, freq='1min', periods=self.num_data))
        # print(df)
        return df

    # common main method called by data controller
    def get_multi_data_df(self):
        ou_dfs = []
        for i in range(self.num_asset):
            # total_time = 1, sigma = 1.0, mu = 100, tau = 0.05
            total_time = np.random.randint(1, 3)
            mu = np.random.randint(50, 150)
            ou_dfs.append(self.get_ou_ohlc_df(total_time, self.sigma, mu, self.tau))
        return pd.concat(ou_dfs, keys=self.ticker_names)

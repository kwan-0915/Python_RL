import time
import numpy as np
import pandas as pd
from utilities.utils import make_figure

class ShmOHLCGenerator:
    """
    simple harmonic motion (sinewave) data generator for sanity check.
    noise and damping can be added.
    """

    def __init__(self, seed, start_date, num_asset, num_data, noise=0.0, damping=0.0):
        if seed:
            self.seed = seed
            np.random.seed(self.seed)

        self.start_date = start_date
        self.ticker_names = ['shm_{}'.format(i) for i in range(num_asset)]

        self.num_asset = num_asset
        self.num_data = num_data

        self.trend_per_tick = 0.0
        self.noise = noise
        self.damping = damping

    @staticmethod
    def shm_gen(dt=0.001,
                coef=100,  # coef = k/m
                amplitude=2,
                start_trend=100,
                trend_per_tick=0.0,
                noise=0.0,
                damping=0.0,
                verbose=0):
        """Generate simple harmonic motion around trend, with noise and damping"""

        period = 2 * np.pi * np.sqrt(1 / coef)

        if verbose:
            print("%s Amplitude: %.3f" % (time.strftime("%H:%M:%S"), amplitude))
            print("%s Period: %.3f" % (time.strftime("%H:%M:%S"), period))

        # initial stock price
        stock_price = start_trend + np.random.choice([-1, 1]) * amplitude
        stock_velocity = 0.0

        trend_index = start_trend
        t = 0.0

        while True:
            # acceleration based on distance from trend
            acc = - coef * (stock_price - trend_index)
            stock_velocity += acc * dt
            # add noise to velocity
            stock_velocity += np.random.normal(loc=0, scale=noise)
            # damp velocity by a % (could also make this a constant)
            stock_velocity *= (1 - damping)
            # increment stock price
            stock_price += stock_velocity * dt
            # add noise; doesn't impact velocity which makes velocity a partly hidden state variable
            stock_price += np.random.normal(loc=0, scale=noise / 2)

            yield (t, stock_price, trend_index)
            t += dt

    @staticmethod
    def plot_df(df):
        make_figure(df.index, df['close'],
                    title="Simulated Stock Price Data As Simple Harmonic Motion (Sine Wave)",
                    xtitle='Timesteps',
                    ytitle='Value'
                    )

    def get_shm_ohlc_df(self, total_time=1, coef=100, amplitude=2, start_trend=100):
        gen = self.shm_gen(dt=total_time / self.num_data,
                           coef=coef,
                           amplitude=amplitude,
                           start_trend=start_trend,
                           trend_per_tick=self.trend_per_tick,
                           noise=self.noise,
                           damping=self.damping,
                           verbose=0)

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
        shm_dfs = []
        for i in range(self.num_asset):
            # total_time = 1, coef = 100, amplitude = 2, start_trend = 100
            total_time = np.random.randint(1, 5)
            coef = np.random.randint(90, 110)
            amplitude = np.random.randint(2, 8)
            start_trend = np.random.randint(50, 150)
            shm_dfs.append(self.get_shm_ohlc_df(total_time, coef, amplitude, start_trend))
        return pd.concat(shm_dfs, keys=self.ticker_names)

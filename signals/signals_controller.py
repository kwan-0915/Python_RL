import numpy as np
import pandas as pd
from shm_ohlc_generator import ShmOHLCGenerator
from ou_ohlc_generator import OuOHLCGenerator
from fx_ohlc_generator import FXOHLCGenerator

class SignalsController:
    """
    Acts as data provider for each new episode.
    """

    def __init__(self, config):
        self.config = config

        # set seed for data feed randint start idx
        if 'random_seed' in config:
            self.seed = config['random_seed']
            np.random.seed(self.seed)
        else:
            self.seed = None

        self.data_source = config['data_source']
        # whether include timestamp in the data feed, default is False
        if 'include_ts' in config: self.include_ts = config['include_ts']
        else: self.include_ts = False

        self.multi_df = self.get_multi_df()
        self.ticker_names = tuple(self.multi_df.index.levels[0])
        self.num_asset = len(self.ticker_names)
        self.data = self.get_data()
        self.window_length = config['window_length']
        self.steps = config['max_ep_length']
        self.data_feed = None  # load data feed at reset
        self.current_step = 0
        self.start_index = self.window_length

    def get_multi_df(self):
        if self.data_source == 'sinewave':
            start_date = self.config['start_date']
            num_asset = self.config['num_asset']
            num_data = self.config['num_data']
            noise = self.config['noise']
            damping = self.config['damping']
            data_gen = ShmOHLCGenerator(seed=self.seed, start_date=start_date, num_asset=num_asset, num_data=num_data, noise=noise, damping=damping)
            source_multi_data_df = data_gen.get_multi_data_df()
        elif self.data_source == 'ou':
            start_date = self.config['start_date']
            num_asset = self.config['num_asset']
            num_data = self.config['num_data']
            sigma = self.config['sigma']
            tau = self.config['ou_tau']
            data_gen = OuOHLCGenerator(seed=self.seed, start_date=start_date, num_asset=num_asset, num_data=num_data, sigma=sigma, tau=tau)
            source_multi_data_df = data_gen.get_multi_data_df()
        elif self.data_source == 'fx':
            start_date = self.config['start_date']
            num_asset = self.config['num_asset']
            num_data = self.config['num_data']
            ohlc_interval = self.config['ohlc_interval'] if 'ohlc_interval' in self.config else None
            path = str(self.config['path'])
            file_extension = str(self.config['file_extension'])
            data_gen = FXOHLCGenerator(start_date=start_date, num_asset=num_asset, num_data=num_data, ohlc_interval=ohlc_interval, path=path, file_extension=file_extension)
            source_multi_data_df = data_gen.get_multi_data_df()
        else:
            raise ValueError('data source: {} is not available. please check config yml file.')

        # add cash asset
        if self.config['add_cash_asset']: return self.add_cash(source_multi_data_df)
        else: return source_multi_data_df

    def get_data(self):
        # print(self.multi_df)
        values = self.multi_df.values

        # ) include time stamp if include_ts = True
        if self.include_ts:
            time_index = self.multi_df.index.get_level_values(1)
            ts = np.expand_dims(time_index.values.astype(np.int64) // 10 ** 9, axis=1)
            values_with_ts = np.concatenate((values, ts), axis=1)
            # reshape to (num_asset, num_data, ohlcv+ts)
            data_out = values_with_ts.reshape((self.num_asset, -1, 6))
        else:
            # reshape to (num_asset, num_data, ohlcv)
            data_out = values.reshape((self.num_asset, -1, 5))

        return data_out

    def get_current_date(self):
        current_idx = self.start_index + self.current_step - 1  # minus 1 because idx start from 0
        datetime_index = self.multi_df.index.levels[1]
        return datetime_index[current_idx]

    @staticmethod
    def add_cash(multi_df):
        multi_index = tuple(multi_df.index.levels[0])
        cash_df = multi_df.loc[multi_index[0]].copy()
        cash_df['ticker'] = 'cash'
        cash_df.set_index('ticker', append=True, inplace=True)
        cash_df = cash_df.swaplevel()

        for col in cash_df.columns:
            cash_df[col].values[:] = 1

        return pd.concat([cash_df, multi_df])

    def step(self):
        self.current_step += 1
        next_state = self.data_feed[:, self.current_step:self.current_step + self.window_length, :].copy()
        done = self.current_step >= self.steps

        return next_state, done

    def reset(self):
        self.current_step = 0
        self.start_index = np.random.randint(low=self.window_length, high=self.data.shape[1] - self.steps)
        self.data_feed = self.data[:, self.start_index - self.window_length:self.start_index + self.steps + 1, :]
        state = self.data_feed[:, self.current_step:self.current_step + self.window_length, :].copy()

        return state

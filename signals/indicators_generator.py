import talib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from talib import abstract

class IndicatorsGenerator:
    def __init__(self, ohlcv):
        self.ohlcv = ohlcv
        self.asset_dfs = [pd.DataFrame(ohlcv[i][:, :5], columns=['open', 'high', 'low', 'close', 'volume'], index=None).astype('float') for i in range(ohlcv.shape[0])]
        self.all_indicators = {
            'god_price': self.get_god_price,
            'god_return': self.get_god_return,
            'god_future_dir': self.get_god_future_dir,
            'log_return': self.get_log_return,
            'above_mean': self.get_above_mean,
            'norm_price': self.get_norm_price,
            'mov_avg_5': self.get_mov_avg_5,
            'rsi': self.get_rsi,
            'macd_hist': self.get_macd_hist,
            'stochastic': self.get_stochastic,
            'stochRSI': self.get_stochRSI,
            'spinning_top': self.get_spinning_top,
            'hikkake': self.get_hikkake,
        }
        self.norm = True

    def get_indicator(self, indicator_name):
        func = self.all_indicators.get(indicator_name, lambda: 'Invalid indicator:{}'.format(indicator_name))
        return func()

    @staticmethod
    def normalize(output):
        return np.nan_to_num(stats.zscore(output, axis=1, nan_policy='omit'))

    # god price is current price - TRUE price, only available for SHM and OU
    def get_god_price(self):
        output = ((self.ohlcv[:, :, 3] - self.ohlcv[:, :, 4]) / self.ohlcv[:, :, 4])
        if self.norm:
            output = self.normalize(output)
        return output

    # god return is log(NEXT FUTURE price) - log(last close)
    def get_god_return(self):
        future_price = self.ohlcv[:, :, 4]
        close_price = self.ohlcv[:, :, 3]
        # print('future_price: ', future_price)
        # print('close_price: ', close_price)
        output = np.log(future_price) - np.log(close_price)
        output = np.nan_to_num(output) * 100
        return output

    # god return is log(NEXT FUTURE price) - log(last close)
    def get_god_future_dir(self):
        future_price = self.ohlcv[:, :, 4]
        close_price = self.ohlcv[:, :, 3]
        output = np.sign(future_price - close_price)
        output[:, :-1] = 0
        return output

    # get log return on close
    def get_log_return(self):
        close_price = self.ohlcv[:, :, 3]
        output = np.diff(np.log(close_price), axis=1)
        if self.norm:
            output = self.normalize(output)
        return output

    # get how far above mean, i.e. close - mean
    def get_above_mean(self):
        close_price = self.ohlcv[:, :, 3]
        mean_price = np.mean(close_price, axis=1, keepdims=True)
        output = ((close_price - mean_price) / close_price)
        if self.norm:
            output = self.normalize(output)
        return output

    # get normalized price
    def get_norm_price(self):
        close_price = self.ohlcv[:, :, 3]
        min_max_range = np.expand_dims((np.max(close_price, axis=1) - np.min(close_price, axis=1)), axis=1)
        norm_price = np.nan_to_num((close_price - np.expand_dims(np.min(close_price, axis=1), axis=1)) / min_max_range)
        return norm_price

    # get moving average period=5
    def get_mov_avg_5(self):
        close_price = self.ohlcv[:, :, 3]
        sma = talib.abstract.Function('sma')
        # ) ---------------------------use Dataframe----------------------------------
        # output = np.vstack([sma(df, timeperiod=5, price='close').values for df in self.asset_dfs])

        # ) ---------------------------use numpy close--------------------------------
        output = np.vstack([sma(close_price[i], timeperiod=5) for i in range(self.ohlcv.shape[0])])
        # if norm = True, then normalize the output
        if self.norm:
            output = self.normalize(output)
        return output

    # get RSI
    def get_rsi(self):
        close_price = self.ohlcv[:, :, 3]
        rsi = talib.abstract.Function('RSI')
        # output = np.vstack([rsi(df, timeperiod=5, price='close').values for df in self.asset_dfs])
        # ) ---------------------------use numpy close--------------------------------
        output = np.vstack([rsi(close_price[i], timeperiod=14) for i in range(self.ohlcv.shape[0])])
        output = np.nan_to_num(output) / 100
        return output

    # get Momentum
    def get_macd_hist(self):
        close_price = self.ohlcv[:, :, 3]
        macd = talib.abstract.MACD
        # macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        output = np.vstack([macd(close_price[i], fastperiod=12, slowperiod=26, signalperiod=9)[2] for i in range(self.ohlcv.shape[0])])  # get MACD histogram as output
        if self.norm:
            output = self.normalize(output)
        return output

    # get stochastic
    def get_stochastic(self):
        all_slow_d = []

        highs = self.ohlcv[:, :, 1]
        lows = self.ohlcv[:, :, 2]
        closes = self.ohlcv[:, :, 3]
        stoch = talib.STOCH
        fastk_period = 5
        slowk_period = 3
        slowk_matype = 0
        slowd_period = 3
        slowd_matype = 0

        for i in range(closes.shape[0]):
            slowk, slowd = stoch(high=highs[i],
                                 low=lows[i],
                                 close=closes[i],
                                 fastk_period=fastk_period,
                                 slowk_period=slowk_period,
                                 slowk_matype=slowk_matype,
                                 slowd_period=slowd_period,
                                 slowd_matype=slowd_matype)
            all_slow_d.append(slowd)

        output = np.vstack(all_slow_d)
        output = np.nan_to_num(output) / 100
        return output

    def get_stochRSI(self):
        close_price = self.ohlcv[:, :, 3]
        # fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        stochRSI = talib.STOCHRSI
        # get fastk  as output
        output = np.vstack([stochRSI(close_price[i], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0] for i in range(self.ohlcv.shape[0])])
        output = np.nan_to_num(output) / 100
        return output

    def get_spinning_top(self):
        pattern = talib.CDLSPINNINGTOP
        output = np.vstack([pattern(self.asset_dfs[i].open, self.asset_dfs[i].high, self.asset_dfs[i].low, self.asset_dfs[i].close) for i in range(len(self.asset_dfs))])
        output = np.nan_to_num(output) / 100
        return output

    def get_hikkake(self):
        pattern = talib.CDLHIKKAKE
        output = np.vstack([pattern(self.asset_dfs[i].open, self.asset_dfs[i].high, self.asset_dfs[i].low, self.asset_dfs[i].close) for i in range(len(self.asset_dfs))])
        output = np.nan_to_num(output) / 100
        return output
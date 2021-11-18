import datetime
import pandas as pd
from os import listdir

class FXOHLCGenerator:

    def __init__(self, start_date, num_asset, num_data, ohlc_interval=None, path='', file_extension=''):
        self.start_date = start_date
        self.num_asset = num_asset
        self.num_data = num_data
        self.ohlc_interval = ohlc_interval
        self.path = path
        self.file_extension = file_extension

        if len(self.path) == 0: raise Exception('File path cannot be empty')
        if len(self.file_extension) == 0 or not self.file_extension.startswith('.'): raise Exception('File extension cannot be empty OR not start with "."')

        self.fx_files = [filename for filename in listdir(path) if filename.endswith(file_extension)]

    def dateparse(self, time_in_secs):
        return datetime.datetime.utcfromtimestamp(float(time_in_secs))

    def get_multi_data_df(self):
        fx_dfs = []
        ticker_name = [fn.split(self.file_extension)[0] for fn in self.fx_files]
        header_cols = ['ots', 'otms', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']

        # print(self.fx_files)
        for i in range(len(self.fx_files)):
            df = pd.read_csv('/'.join((self.path, self.fx_files[i])), parse_dates=True, date_parser=self.dateparse, index_col='ots', names=header_cols)
            df = df[self.start_date:]
            df = df.rename_axis(index=None)

            # resample ohlc interval
            if self.ohlc_interval:
                ohlc = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                }
                df = df.resample(self.ohlc_interval).agg(ohlc).dropna(how='any')

            # add future price (god price) as volume
            df['volume'] = df['close'].shift(-1).fillna(method='ffill')
            # get num_data out of df
            df = df.iloc[:self.num_data]
            fx_dfs.append(df)

        return pd.concat(fx_dfs, keys=ticker_name)

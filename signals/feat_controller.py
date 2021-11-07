from indicators_generator import IndicatorsGenerator

class FeatController:
    def __init__(self, ohlcv_data, indicators):
        self.ohlcv_data = ohlcv_data
        self.indicators = indicators
        self.indicators_generator = IndicatorsGenerator(ohlcv_data)

    def get_feats(self):
        if self.indicators:
            return [self.indicators_generator.get_indicator(indicator) for indicator in self.indicators]
        else:
            return list()
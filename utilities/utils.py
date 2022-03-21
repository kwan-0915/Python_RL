import numpy as np

from portfolio.fx_portfolio_wrapper import FXPortfolioWrapper


def sharpe(returns, freq=30, rfr=0, eps=1e-8):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns, eps=1e-8):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)


def create_env_wrapper(config):
    env_name = config['env']
    if env_name == 'FX': return FXPortfolioWrapper(config)
    else: raise KeyError('No such environment')
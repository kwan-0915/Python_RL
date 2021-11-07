import os
import imageio
import numpy as np
import plotly.graph_objects as go
from glob import glob

def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
    Returns: None
    """
    batch_sort = lambda s: int(s[s.rfind('/')+1:s.rfind('.')])
    image_paths = sorted(glob(os.path.join(source_dir, "*.png")), key=batch_sort)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))

    imageio.mimsave(output, images)

def make_figure(*series, title="", xtitle="", ytitle=""):
    fig = go.Figure()
    series = list(series)
    x = series.pop(0)

    for s in series:
        fig.add_trace(go.Scatter(y=s, x=x))

    fig.update_layout(
        title=dict(text=title,
                   x=0.5,
                   xanchor='center'),
        xaxis=dict(
            title=xtitle,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            title=ytitle,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        showlegend=False
    )
    return fig.show()

def sharpe(returns, freq=30, rfr=0, eps=1e-8):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def max_drawdown(returns, eps=1e-8):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)
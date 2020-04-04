import numpy as np
import pandas as pd


def preEmphasis(signal: list, alpha: float) -> np.array:
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])

    return emphasized_signal


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
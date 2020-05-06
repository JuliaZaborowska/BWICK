from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def preEmphasis(signal: list, alpha: float) -> np.array:
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal


def envelope(y, rate, threshold):
    mask: List[bool] = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=(1 / rate))
    Y = abs(np.fft.rfft(y / n))
    return Y, freq


def spectrogram(x, rate, windowsize=512, off=0.01, draw=True, title="Spectrogram"):
    """
    example
    x, rate = librosa.load(file, sr=16000)
    spectrogram(x, rate)
    """
    window_size = windowsize
    offset = off  # 10 ms
    step = int(rate * offset)
    nfft = window_size
    hamming = np.hamming(window_size)

    step_range = range(step, len(x) - window_size, step)
    F = np.zeros((len(step_range), nfft // 2))

    for i, n in enumerate(step_range):
        window = x[n: n + window_size] * hamming
        z = np.fft.fft(window)
        F[i, :] = np.abs(z[:nfft // 2])

    freq = np.arange(0, window_size / 2) * rate / window_size
    time = np.arange(0, len(step_range)) * offset

    if draw:
        plt.pcolormesh(time, freq, F.T)
        plt.title(title)
        plt.show()

    return time, freq, F.T

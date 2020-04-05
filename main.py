import librosa
import pandas as pd
import python_speech_features
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

from statistics import *
from processing import *

df = pd.read_csv('RAVDESS_db.csv')  # Wczytywanie danych o plikach audio z bazy
classes = list(np.unique(df["emotion"]))

# Normal emotions
for c in classes:
    if len(df[(df["emotion"] == c) & (df["intensivity"] == "strong")]) > 0:
        wav_file = df[(df["emotion"] == c) & (df["intensivity"] == "strong")].iloc[0, 1]
        signal, rate = librosa.load('RAVDESS/' + wav_file + ".wav", sr=16000)  # File assumed to be in the same directory
        signal = signal[0: int(2.5 * rate)]  # Keep the first 2.5 seconds

        signal = preEmphasis(signal, alpha=0.97)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]  # Pozbywanie się szumów

        Y, freq = calc_fft(signal, rate)

        axes = plt.gca()
        axes.set_ylim([-0.1, 0.1])
        plt.plot(signal)
        plt.title(str(c))
        plt.show()

        mfcc_speech = python_speech_features.mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                                                  nfilt=26, nfft=512, lowfreq=0, preemph=0.0,
                                                  ceplifter=22, appendEnergy=True)
        plt.title(str(c))
        plt.imshow(mfcc_speech, cmap='hot', interpolation='nearest')
        plt.show()

        # plt.plot(freq, Y)
        # plt.title(str(c))
        # plt.show()

# Strong emotions
"""
for c in classes:

    if len(df[(df["emotion"] == c) & (df["intensivity"] == "strong")]) > 0:
        wav_file = df[(df["emotion"] == c) & (df["intensivity"] == "strong")].iloc[0, 1]
        signal, rate = librosa.load('RAVDESS/' + wav_file + ".wav", sr=16000)  # File assumed to be in the same directory
        signal = signal[0: int(2.5 * rate)]  # Keep the first 2.5 seconds

        signal = preEmphasis(signal, alpha=0.97)
        mask = envelope(signal, rate, 0.00005)
        signal = signal[mask]  # Pozbywanie się szumów

        axes = plt.gca()
        axes.set_ylim([-0.1, 0.1])
        plt.plot(signal)
        plt.title(str(c))
        plt.show()
        
"""

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import python_speech_features


from processing import preEmphasis, envelope, spectrogram

df = pd.read_csv('RAVDESS_db.csv')  # Wczytywanie danych o plikach audio z bazy
classes = list(np.unique(df["emotion"]))

# Normal emotions
"""
for c in classes:
    if len(df[(df["emotion"] == c) & (df["intensivity"] == "normal")]) > 0:
        wav_file = df[(df["emotion"] == c) & (df["intensivity"] == "normal")].iloc[0, 1]
        signal, rate = librosa.load('RAVDESS/' + wav_file + ".wav",
                                    sr=16000)  # File assumed to be in the same directory

        signal = preEmphasis(signal, alpha=0.97)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]  # Pozbywanie się szumów

        spectrogram(signal, rate, title=str(c) + ' normal')

        mfcc_speech = python_speech_features.mfcc(signal, samplerate=16000, winlen=0.032, winstep=0.01, numcep=13,
                                                  nfilt=40, nfft=512, lowfreq=0, preemph=0.0,
                                                  ceplifter=0, appendEnergy=True)

        plt.title(str(c) + ' normal')
        plt.imshow(mfcc_speech, cmap='hot', interpolation='nearest')
        plt.show()
"""

for c in classes:
    if len(df[(df["emotion"] == c) & (df["intensivity"] == "normal")]) > 0:
        wav_file = df[(df["emotion"] == c) & (df["intensivity"] == "normal")].iloc[0, 1]
        signal, rate = librosa.load('RAVDESS/' + wav_file + ".wav",
                                    sr=16000)  # File assumed to be in the same directory

        signal = preEmphasis(signal, alpha=0.97)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]  # Pozbywanie się szumów

        spectrogram(signal, rate, title=str(c) + ' normal')

        mfcc_speech = python_speech_features.mfcc(signal, samplerate=16000, winlen=0.032, winstep=0.01, numcep=13,
                                                  nfilt=40, nfft=512, lowfreq=0, preemph=0.0,
                                                  ceplifter=0, appendEnergy=True)

        plt.title(str(c) + ' normal')
        plt.imshow(mfcc_speech, cmap='hot', interpolation='nearest')
        plt.show()

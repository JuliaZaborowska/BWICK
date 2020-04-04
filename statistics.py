import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def classDistibutionTime():
    df = pd.read_csv('RAVDESS_db.csv')   # Wczytywanie danych o plikach audio z bazy
    df.set_index('fname', inplace=True)  # Zmiana klucza/indeksu na 'fname'

    for audiofile in df.index:
        signal, rate = librosa.load('RAVDESS/' + audiofile + ".wav", sr=16000)  # Wczytanie baudrate i wartości synału do pamięci
        df.at[audiofile, 'length'] = signal.shape[0] / rate  # Dodanie kolumny z długością utworu

    class_dist = df.groupby(['emotion'])['length'].mean()

    fig, ax = plt.subplots()
    ax.set_title('Rozłożenie pod względem\nczasu trwania nagrania', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()

    df.reset_index(inplace=True)


def classStd():
    df = pd.read_csv('RAVDESS_db.csv')   # Wczytywanie danych o plikach audio z bazy
    df.set_index('fname', inplace=True)  # Zmiana klucza/indeksu na 'fname'

    for audiofile in df.index:
        signal, rate = librosa.load('RAVDESS/' + audiofile + ".wav", sr=16000)  # Wczytanie baudrate i wartości synału do pamięci
        df.at[audiofile, 'mean'] = np.mean(signal)                # Dodanie kolumny z długością utworu

    class_dist = df.groupby(['emotion'])['mean'].std()

    plt.bar(class_dist.index, class_dist)
    plt.show()

import os
from tqdm import tqdm
from librosa.output import write_wav
from librosa import get_duration
import pandas as pd
import librosa
import numpy as np

from processing import envelope

'''
To jest skrypt który służy do czyszczenia ciszy z nagrań i wydłużenia ich do 3s. Puszcza się go tylko raz,
na samym początku. 
'''


def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data


def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high=5) * 500)
    return np.roll(data, s_range)


def stretch(data, rate=0.8):
    """
    Streching the Sound.
    """
    data = librosa.effects.time_stretch(data, rate)
    return data


def pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                       sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data


try:
    os.mkdir('./clear')
except OSError as error:
    print(error)

csvName: str = 'RAVDESS_db.csv'
input: str = 'RAVDESS'
output: str = 'clear'

df: pd.DataFrame = pd.read_csv(csvName)  # Wczytywanie danych o plikach audio z bazy

sumTime: float = 0
sumTimeTrimmed: float = 0
bitrate = 44100

if len(os.listdir(output)) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load(os.path.join(input, f + '.wav'), sr=bitrate)
        mask = envelope(signal, rate, 0.0005)
        write_wav(path=os.path.join(output, f + '.wav'), sr=rate, y=signal[mask], norm=False)
        #signal = librosa.effects.trim()
        sumTime += get_duration(signal)
        sumTimeTrimmed += get_duration(signal)

    print("Czas nagrań przed obróbką = " + str(sumTime) + "s , Czas nagrań po obróbce = " + str(sumTimeTrimmed) + "s.")

for f in tqdm(df.fname):
    signal, rate = librosa.load(os.path.join(output, f + ".wav"), sr=16000)
    length = signal.shape[0] / rate
    signal = np.array(signal)
    singleSignal = np.copy(signal)
    while length < 3.0:
        signal = np.hstack((signal, singleSignal))
        length = signal.shape[0] / rate

    write_wav(path=os.path.join(output, f + '.wav'), sr=rate, y=signal, norm=False)

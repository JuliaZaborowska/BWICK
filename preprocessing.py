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

try:
    os.mkdir('./clear')
except OSError as error:
    print(error)

csvName: str = 'myAudio_db.csv'
input: str = 'MY_AUDIO'
output: str = 'clear'

df: pd.DataFrame = pd.read_csv(csvName)  # Wczytywanie danych o plikach audio z bazy

sumTime: float = 0
sumTimeTrimmed: float = 0
bitrate = 44100

"""
if len(os.listdir(output)) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load(os.path.join(input, f + '.wav'), sr=bitrate)
        mask = envelope(signal, rate, 0.0005)
        write_wav(path=os.path.join(output, f + '.wav'), sr=rate, y=signal[mask], norm=False)
        sumTime += get_duration(signal)
        sumTimeTrimmed += get_duration(signal)

    print("Czas nagrań przed obróbką = " + str(sumTime) + "s , Czas nagrań po obróbce = " + str(sumTimeTrimmed) + "s.")

for f in tqdm(df.fname):
    signal, rate = librosa.load(os.path.join(output, f + ".wav"), sr=bitrate)
    length = signal.shape[0] / rate
    signal = np.array(signal)
    singleSignal = np.copy(signal)
    while length < 3.0:
        signal = np.hstack((signal, singleSignal))
        length = signal.shape[0] / rate

    write_wav(path=os.path.join(output, f + '.wav'), sr=rate, y=signal, norm=False)
"""

for f in tqdm(df.fname):
    signal, rate = librosa.load(os.path.join(input, f + ".wav"), sr=bitrate)
    length = signal.shape[0] / rate
    signal = np.array(signal)
    minTime = 3.0
    if length < minTime:
        timeToAdd = minTime - length
        samplesToAdd = timeToAdd * rate
        signal = np.hstack((np.zeros(int(samplesToAdd // 2)), signal, (np.zeros(int(samplesToAdd // 2) + 1))))

    write_wav(path=os.path.join(output, f + '.wav'), sr=rate, y=signal, norm=False)
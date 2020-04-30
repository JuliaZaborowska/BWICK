from scipy.io import wavfile
import os
from tqdm import tqdm

from statistics import *
from processing import *

'''
!!!!!!!!!!!!!!!!!!!!!UWAGA!!!!!!!!!!!!!!!!!!!!
To jest skrypt który służy do czyszczenia ciszy z nagrań. Puszcza się go tylko raz,
na samym początku. 
'''

df = pd.read_csv('myAudiofile_db.csv')  # Wczytywanie danych o plikach audio z bazy

if len(os.listdir('myAudioClear')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('myAudioFiles/' + f + '.wav', sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='myAudioClear/' + f + '.wav', rate=rate, data=signal[mask])

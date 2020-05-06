import pickle
import os

import librosa
import numpy as np
from librosa.feature import melspectrogram
from scipy.signal.windows import windows
from tqdm import tqdm
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
from processing import spectrogram
from python_speech_features import mfcc

def build_predicitions():
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for f in tqdm(df.index):
        signal, rate = librosa.load(os.path.join(inputDir, f + '.wav'), res_type='kaiser_fast', sr=bitrate, duration=2.8, offset=0.2)
        label = df.at[f, 'emotion']
        c = classes.index(label)

        mel = mfcc(signal, rate, numcep=13, nfilt=26, nfft=2048, winfunc=np.hamming)


        #mel = melspectrogram(signal, rate, n_mels=128, n_fft=512, hop_length=128, window=windows.hamming(512))
        mel = mel.reshape(1, mel.shape[0], mel.shape[1], 1)
        y_res = model.predict(mel)

        y_true.append(c)
        y_pred.append(np.argmax(y_res))
        fn_prob[f] = y_res

    return y_true, y_pred, fn_prob

bitrate = 44100
database = 'RAVDESS_db_TEST.csv'
inputDir: str = 'RAVDESS'
p_path = os.path.join('models', 'conv.model')


df = pd.read_csv(database)  # Wczytywanie danych o plikach audio z bazy
df.set_index('fname', inplace=True)

classes = list(np.unique(df["emotion"]))
model = load_model(p_path)
y_true, y_pred, fn_prob = build_predicitions()

acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
print(acc_score)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[i]
    y_probs.append(y_prob[0])
    for c, p in zip(classes, y_prob[0]):
        df.at[i, c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions/test.csv', index=False)

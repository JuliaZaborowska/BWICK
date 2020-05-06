# region Imports

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import accuracy_score
from python_speech_features import mfcc
from sklearn.metrics import confusion_matrix

# endregion


def build_predicitions():
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for f in tqdm(df.index):
        signal, rate = librosa.load(os.path.join(inputDir, f + '.wav'), res_type='kaiser_fast', sr=bitrate,
                                    duration=2.8, offset=0.2)
        label = df.at[f, 'emotion']
        c = classes.index(label)

        mel = mfcc(signal, rate, numcep=13, nfilt=26, nfft=2048, winfunc=np.hamming)
        mel = mel.reshape(1, mel.shape[0], mel.shape[1], 1)
        y_res = model.predict(mel)

        y_true.append(c)
        y_pred.append(np.argmax(y_res))
        fn_prob[f] = y_res

    return y_true, y_pred, fn_prob


# region Initialization

bitrate: int = 44100
database = 'RAVDESS_db_TEST.csv'
inputDir: str = 'RAVDESS'
p_path = os.path.join('models', 'conv.model')

df = pd.read_csv(database)  # Wczytywanie danych o plikach audio z bazy
df.set_index('fname', inplace=True)  # Ustawienie indeksu na nazwę pliku

# endregion

# region Przepuszczenie danych przez model

classes = list(np.unique(df["emotion"]))
model = load_model(p_path)
y_true, y_pred, fn_prob = build_predicitions()

acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
print(acc_score)

# endregion

# region Zapisywanie w pliku

y_predicted = np.copy(y_pred)
y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[i]
    y_probs.append(y_prob[0])
    for c, p in zip(classes, y_prob[0]):
        df.at[i, c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions/test2.csv', index=False)

# endregion

# region confusion matrix

data = {'y_Actual':    y_true,
        'y_Predicted': y_predicted
        }

df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'])

sn.heatmap(conf_matrix, annot=True)
plt.show()

# endregion
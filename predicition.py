# region Imports

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from python_speech_features import mfcc

# endregion


def build_predicitions():
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for f in tqdm(df.index):
        signal, rate = librosa.load(os.path.join(inputDir, f + '.wav'), res_type='kaiser_fast', sr=bitrate,
                                    duration=2.8, offset=0.1)
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
#database = 'RAVDESS_db_TEST.csv'
database = 'myAudio_db.csv'
inputDir: str = 'clear'
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

df.to_csv('predictions/predicition.csv', index=False)

# endregion

# region confusion matrix

data = confusion_matrix(y_true, y_predicted)
df_cm = pd.DataFrame(data, columns=classes, index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)
figure = sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
fig = figure.get_figure()
fig.savefig('confusion_matrix.png')
plt.show()


# endregion
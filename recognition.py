# region Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
from tqdm import tqdm

from config import Config
from CNN import CNN

# endregion

# region Inicjalizacja
bitrate: int = 44100
config = Config('conv')

# Wczytywanie danych o plikach audio z bazy do pamięci
inputDir: str = 'RAVDESS'
df = pd.read_csv('RAVDESS_db.csv')
df.set_index('fname', inplace=True)

# endregion

# region Wybór zbioru danych

"""
Wybrany zbiór danych trzeba odkomentować. Jeśli wszystkie są zakomentowane, do działa rozpoznawanie 
wszystkich 8 emocji.
"""

# Ograniczenie danych do 2 emocji: happy, sad
df = df[((df["emotion"] == "happy") | (df["emotion"] == "sad"))]

# Ograniczenie danych do 5 emocji: angry, suprised, calm, sad, happy, fearful
# df = df[((df["emotion"] != "fearful") & (df["emotion"] != "disgust") & (df["emotion"] != "neutral"))]

# Ograniczenie danych do 2 emocji: angry, calm
# df = df[((df["emotion"] == "angry") | (df["emotion"] == "calm"))]

# Ograniczenie danych do 3 emocji: fearful, disgust, angry
# df = df[((df["emotion"] == "angry") | (df["emotion"] == "fearful") | (df["emotion"] == "disgust"))]

# Ograniczenie danych do 2 emocji: neutral, calm
# df = df[((df["emotion"] == "angry") | (df["emotion"] == "fearful") | (df["emotion"] == "disgust"))]

# endregion

# region Podział na zbiór testowy i treningowy

df, df_test = train_test_split(df, test_size=0.1)

df_test.to_csv("RAVDESS_db_TEST.csv", index=True)
df.to_csv("RAVDESS_db_TRAIN.csv", index=True)

# endregion

# region Wyznaczanie prawdopodobieństwa wylosowania próbki danej emocji z bazy danych

print("Obliczanie długości nagrań:")
# Obliczenie czasu każdego nagrania i dopisanie go do tabeli df
for f in tqdm(df.index):
    signal, rate = librosa.load(os.path.join(inputDir, f + ".wav"), sr=bitrate, res_type='kaiser_fast')
    df.at[f, 'length'] = signal.shape[0] / rate

print("Najkrótsze nagranie ma: " + str(df['length'].min()) + "s")
print("Najdłuższe nagranie ma: " + str(df['length'].max()) + "s")

classes = list(np.unique(df['emotion']))  # zwraca nazwy emocji jakie są w plikach
class_dist = df.groupby(['emotion'])['length'].mean()  # oblicza średnią długość nagrania dla każdej emocji
prob_dist = class_dist / class_dist.sum()

# Rysowanie wykresu
fig, ax = plt.subplots()
ax.set_title('Prawdopodobieństwo wylosowania próbki \ndanej emocji z biblioteki nagrań')
ax.pie(class_dist, labels=class_dist.index, shadow=False, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.show()

# endregion

# region Obliczanie cech dla każdego nagrania w zbiorze treningowym

X = []
Y = []
for f in tqdm(df.index):
    signal, rate = librosa.load(os.path.join(inputDir, f + '.wav'),
                                res_type='kaiser_fast',
                                sr=bitrate,
                                duration=2.8,
                                offset=0.2)
    label = df.at[f, 'emotion']
    mel = mfcc(signal, rate, numcep=13, nfilt=26, nfft=2048, winfunc=np.hamming)
    X.append(mel)
    Y.append(classes.index(label))

last_layer_output = len(classes)
Y = to_categorical(Y, num_classes=last_layer_output)
X, Y = np.array(X), np.array(Y)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
input_shape = (X.shape[1], X.shape[2], 1)

# endregion

# region Nauka siecią splotową i zapisywanie modelu

try:
    os.mkdir('./models')
except OSError as error:
    print(error)

# zapisywanie modelu sieci tylko wtedy kiedy kolejna epoka okaże się lepsza niż poprzednia
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model = CNN(input_shape=input_shape, class_number=last_layer_output)
history = model.fit(X, Y, epochs=50, batch_size=128, shuffle=True, validation_split=0.2, callbacks=[checkpoint])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# endregion
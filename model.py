import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from scipy.io import wavfile
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os

from config import Config


def check_data():
    if os.path.isfile(config.p_path):
        print('Loading model {}'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None


def buildRandFeat():
    tmp = check_data()
    if tmp:
        return tmp[0], tmp[1]  # Jeśli dane są już policzone to nie liczymy od nowa
    X = []
    Y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index,
                                      p=prob_dist)  # wyznacza losową klasę/emocję uwzględniając rozkład prob_dist
        file = np.random.choice(df[df['emotion'] == rand_class].index)  # wyznacza losowy plik z wybranej emocji
        rate, signal = wavfile.read('clear/' + file + ".wav")  # wczytuje wybrany plik
        label = df.at[file, 'emotion']  # zapamiętuje emocję nagrania
        rand_index = np.random.randint(0, signal.shape[0] - config.step)  # losuje próbkę w pliku
        sample = signal[rand_index:rand_index + config.step]  # pobiera wylosowaną próbkę z pliku

        # wyznaczanie współczynników cepstralnych dla wylosowanej próbki (transponowane, bo ta biblioteka dziwnie liczy)
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft, winfunc=np.hamming)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        # X.append(X_sample if config.mode == 'conv' else X_sample.T)
        Y.append(classes.index(label))

    config.min = _min
    config.max = _max

    X, Y = np.array(X), np.array(Y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    Y = to_categorical(Y, num_classes=10)

    config.data = (X, Y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)

    return X, Y


def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    # 16 filters, 3x3 sploty (convolusion), padding = same pilnuje zeby wymiary były takie same
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def get_recurrent_model():
    # rozmiar RNN = (n x time x feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))  # Long Short-Term-Memory_unit
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(8, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


df = pd.read_csv('RAVDESS_db.csv')  # Wczytywanie danych o plikach audio z bazy
df.set_index('fname', inplace=True)
#df = df[df['statement'] == 'dogs']
#df = df[df['sex'] == 'female']
#df = df[df['intensivity'] == 'strong']

for f in tqdm(df.index):
    signal, rate = librosa.load('clear/' + f + ".wav", sr=16000)
    df.at[f, 'length'] = signal.shape[0] / rate  # czas w sekundach

classes = list(np.unique(df['emotion']))  # zwraca nazwy emocji jakie są w plikach
class_dist = df.groupby(['emotion'])['length'].mean()  # oblicza średnią długość nagrania dla każdej emocji

step = 0.05
n_samples = 2 * int(df['length'].sum() / step)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index,
                           p=prob_dist)  # wyznacza losową klasę/emocję uwzględniając rozkład prob_dist
file = np.random.choice(df[df['emotion'] == choices].index)
label = df.at[file, 'emotion']

fig, ax = plt.subplots()
ax.set_title('Prawdopodobieństwo wylosowania próbki \ndanej emocji z biblioteki nagrań')
ax.pie(class_dist, labels=class_dist.index, shadow=False, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.show()

config = Config(mode='conv')

if config.mode == 'conv':
    X, Y = buildRandFeat()
    Y_flat = np.argmax(Y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

elif config.mode == 'time':
    X, Y = buildRandFeat()
    Y_flat = np.argmax(Y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

class_weight = compute_class_weight('balanced', np.unique(Y_flat), Y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model.fit(X, Y, epochs=20, batch_size=32, shuffle=True, class_weight=class_weight,
          validation_split=0.1, callbacks=[checkpoint])

#model.save(config.model_path)

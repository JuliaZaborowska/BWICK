from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.feature import melspectrogram
from librosa import display
import os
from scipy.signal import windows

from config import Config
from CNN import CNN

# Wczytywanie danych o plikach audio z bazy do pamięci
inputDir: str = 'clear'
df = pd.read_csv('RAVDESS_db.csv')
df.set_index('fname', inplace=True)
df = df[((df["emotion"] == "happy") | (df["emotion"] == "sad"))]

print("Obliczanie długości nagrań:")
# Obliczenie czasu każdego nagrania i dopisanie go do tabeli df
for f in tqdm(df.index):
    signal, rate = librosa.load(os.path.join(inputDir, f + ".wav"), sr=16000, res_type='kaiser_fast')
    df.at[f, 'length'] = signal.shape[0] / rate

print("Najkrótsze nagranie ma: " + str(df['length'].min()) + "s")
print("Najdłuższe nagranie ma: " + str(df['length'].max()) + "s")

classes = list(np.unique(df['emotion']))  # zwraca nazwy emocji jakie są w plikach
class_dist = df.groupby(['emotion'])['length'].mean()  # oblicza średnią długość nagrania dla każdej emocji
prob_dist = class_dist / class_dist.sum()

fig, ax = plt.subplots()
ax.set_title('Prawdopodobieństwo wylosowania próbki \ndanej emocji z biblioteki nagrań')
ax.pie(class_dist, labels=class_dist.index, shadow=False, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.show()

config = Config('conv')

X = []
Y = []
for f in tqdm(df.index):
    # chcemy rozmiar 128
    # duration = (128 - 1) * hop_length / rate
    signal, rate = librosa.load(os.path.join(inputDir, f + '.wav'), res_type='kaiser_fast', sr=16000, duration=1.016)
    label = df.at[f, 'emotion']

    mel = melspectrogram(signal, rate, n_mels=128, n_fft=512, hop_length=128, window=windows.hamming(512))
    X.append(mel)
    Y.append(classes.index(label))

Y = to_categorical(Y, num_classes=len(classes))
X, Y = np.array(X), np.array(Y)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
input_shape = (X.shape[1], X.shape[2], 1)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model = CNN(input_shape=input_shape, class_number=len(classes))
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
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('clear/03-01-05-02-01-02-14.wav')
hop_length = int(len(y) / 256)
print(hop_length)
D = np.abs(librosa.stft(y=y, n_fft=512, window="hann", center=True, hop_length=hop_length))

print(D.shape)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
import librosa.display
import os

from statistics import *
from processing import *

inputdir: str = 'clear'

df = pd.read_csv('RAVDESS_db.csv')  # Wczytywanie danych o plikach audio z bazy
classes = list(np.unique(df["emotion"]))

# Normal emotions
for c in classes:
    wav_file = df[(df["emotion"] == c)].iloc[0, 0]
    signal, rate = librosa.load(os.path.join(inputdir, wav_file + ".wav"), sr=16000)

    signal = preEmphasis(signal, alpha=0.97)

    nwin = int(rate * 0.025)
    step = int(rate * 0.01)
    nfft = nwin

    window = np.hamming(nwin)

    nn = range(nwin, len(signal), step)
    X = np.zeros((len(nn), nfft // 2))

    for i, n in enumerate(nn):
        xseg = signal[n - nwin: n]
        z = np.fft.fft(window * xseg)
        X[i, :] = np.log(np.abs(z[:nfft // 2]))

    librosa.display.waveplot(signal, rate)
    plt.title(str(c))
    plt.show()

    plt.imshow(X.T, interpolation='nearest', origin='lower', aspect='auto')
    plt.title(str(c))
    plt.show()

    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=26)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC: ' + str(c))
    plt.tight_layout()
    plt.show()

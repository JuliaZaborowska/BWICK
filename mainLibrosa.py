import librosa
import librosa.display

from statistics import *
from processing import *

df = pd.read_csv('RAVDESS_db.csv')  # Wczytywanie danych o plikach audio z bazy
classes = list(np.unique(df["emotion"]))

# Normal emotions
for c in classes:
    if len(df[(df["emotion"] == c) & (df["intensivity"] == "strong")]) > 0:
        wav_file = df[(df["emotion"] == c) & (df["intensivity"] == "strong")].iloc[0, 1]
        signal, rate = librosa.load('RAVDESS/' + wav_file + ".wav",
                                    sr=16000)  # File assumed to be in the same directory

        signal = preEmphasis(signal, alpha=0.97)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]  # Pozbywanie się szumów

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

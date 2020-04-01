from scipy.io import wavfile


def load_voice(filename):
    fs, data = wavfile.read(filename)

    print('Data:', data)
    print('Sampling rate:', fs)
    print('Audio length:', data.size / fs, 'seconds')
    print('Lowest amplitude:', min(data))
    print('Highest amplitude:', max(data))

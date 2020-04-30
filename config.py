import os


class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000, step=0.1):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate * step)

        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        self.data = None

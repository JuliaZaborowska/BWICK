import os


class Config:
    def __init__(self, name='conv', nfilt=26, nfeat=13, nfft=512, rate=16000, duration_frame=128, hop_length=128):
        self.name = name
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.time = (duration_frame - 1) * hop_length / rate
        self.test = None
        self.train = None

        self.model_path = os.path.join('models', name + '.model')
        self.p_path = os.path.join('pickles', name + '.p')

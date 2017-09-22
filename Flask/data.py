import numpy as np

def generate_tone(frequency, sample_rate, length, phase=0, func=np.sin):
    """
    frequency - number of cycles per second
    sample_rate - number of samples per second
    length - number of samples
    """
    period = 1./frequency
    return func([2*np.pi/period/sample_rate*i for i in range(length)]+phase)

class Generator(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __iter__(self):
        return self

    def next(self):
        """
        Return a pair (features, label), where `features` is a list of
        numerical values, and `label` a list containing the expected output.
        """
        return (None, None)

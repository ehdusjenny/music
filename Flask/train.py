import numpy as np
import librosa
from tqdm import tqdm
import model
import data

def train(model):
    dg = data.NoisyGenerator(44100)

    data_x_wav = []
    data_y = []
    # Generate data set
    for _ in range(100):
        [x,y] = dg.next()
        data_x_wav.append(x)
        data_y.append(y)
    # Generate chromagrams
    data_x_chrom = np.stack([librosa.feature.chroma_cqt(y=x,sr=44100).mean(axis=1) for x in data_x_wav])
    data_y = np.stack(data_y)
    m.fit(data_x_chrom,data_y)

    return m

def test(model):
    dg = data.NoisyGenerator(44100)

    data_x_wav = []
    data_y = []
    for _ in range(10):
        [x,y] = dg.next()
        data_x_wav.append(x)
        data_y.append(y)
    # Generate chromagrams
    data_x_chrom = np.stack([librosa.feature.chroma_cqt(y=x,sr=44100).mean(axis=1) for x in data_x_wav])
    data_y = np.stack(data_y)

    print(m.evaluate(data_x_chrom,data_y))

    return m

def main():
    pass

if __name__ == "__main__":
    m = model.get_model('model.h5')
    m.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    while True:
        train(m)
        test(m)
        m.save('model.h5')

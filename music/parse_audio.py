import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
import librosa.display

import numpy as np

def compute_chroma(audio_path):
    print(audio_path)
    y, sr = librosa.load(audio_path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    return C,sr

def list_notes(chroma):
    chroma = np.copy(chroma)
    print(chroma)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    order = []
    for _ in range(12):
        m = 0
        mi = 0
        for i,x in enumerate(chroma):
            if i in order:
                continue
            if x > m:
                m = x
                mi = i
        chroma[(mi+1)%12] /= 2
        chroma[mi-1] /= 2
        order.append(mi)
    n = [notes[i] for i in order]
    print(n)

def find_key(chroma):
    length = chroma.shape[1]
    note_frequency = [0]*12
    for i in range(length):
        x = chroma[:,i]
        amax = np.argmax(x)
        note_frequency[amax] += 1
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print(notes)
    print(note_frequency)
    tmp = list(zip(notes, note_frequency))
    tmp.sort(key=lambda pair: pair[1])
    tmp = tmp[::-1]
    print(tmp)
    return note_frequency

def to_midi(audio_path):
    y, sr = librosa.load(audio_path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    d = librosa.core.stft(y_harmonic,n_fft=2000-2)

if __name__ == "__main__":
    argparser.add_argument("--path", help="relative path to the audio file")
    args = argparser.parse_args()
    audio_path = args.path

    chroma, sample_rate = compute_chroma(audio_path)

    # Plot graph
    librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.figure(figsize=(12,8))
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('chromagram.png')

    chords = {'major': [0,4,7],
              'minor': [0,3,7],
              'augmented': [0,4,8],
              'diminished': [0,3,6],
              'major7': [0,4,7,11]}

    find_key(C)

import numpy as np
import pretty_midi
import pickle
import os
import itertools
import librosa
from random import randint
from tqdm import tqdm

DEFAULT_SAMPLING_RATE = 44100
DEFAULT_SAMPLE_LENGTH = 44100
chords_to_num = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7,
        "G#":8, "A":9, "A#":10, "B":11}

def generate_tone(note_number, program_number, sample_length, sampling_rate):
    label = np.zeros(12)
    note_name_octave = pretty_midi.note_number_to_name(note_number)
    note_name = note_name_octave[:-1]
    note_name_clean = note_name.replace("-", "")
    chord_index = chords_to_num[note_name_clean]
    label[chord_index] = 1

    velocity = 100

    chord = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program_number)
    note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=0, end=1)
    instrument.notes.append(note)
    chord.instruments.append(instrument)
    audio_data = chord.synthesize(fs=sampling_rate)

    return (audio_data[:sample_length], label)

def generate_chord(note_numbers, program_number, sample_length, sampling_rate):
    label = np.zeros(12)
    label = [n in note_numbers for n in range(128)]

    velocity = 100

    chord = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program_number)
    for note_number in note_numbers:
        note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=0, end=1)
        instrument.notes.append(note)
    chord.instruments.append(instrument)
    audio_data = chord.synthesize(fs=sampling_rate)

    return (audio_data[:sample_length], label)

def random_clips(n, num_notes=3, sample_length=DEFAULT_SAMPLE_LENGTH,
        sampling_rate=DEFAULT_SAMPLING_RATE):
    for i in itertools.count():
        if i >= n:
            break
        note_number = np.random.choice(range(24,119), [num_notes]).tolist()
        # Programs that don't work: 0,9,12,34
        program_number = np.random.choice(range(1,112))
        yield generate_chord(note_number, program_number, sample_length, sampling_rate)

def white_noise(n, sample_length=DEFAULT_SAMPLE_LENGTH):
    for i in itertools.count():
        if i >= n:
            break
        yield np.random.rand(sample_length)

if __name__=="__main__":
    import scipy.io.wavfile
    noise = list(white_noise(1))[0]
    clip = list(random_clips(1))[0]
    scipy.io.wavfile.write("f.wav",44100, clip[0])
    print("Computing stft")
    d = librosa.core.stft(clip[0],n_fft=2000-2)
    mag = np.abs(d)
    ang = np.angle(d)
    x = np.vstack([mag,ang])

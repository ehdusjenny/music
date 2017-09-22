import numpy as np
import pretty_midi
import pickle
from random import randint

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
        features = pickle.load(open("features.p", "rb"))
        labels = pickle.load(open("labels.p", "rb"))
        index = randint(0, len(features))
        return (features[index], labels[index])

    def create_data(self):
        """
        MIDI note numbers range from 0 - 127.
        MIDI program number (non-percussion instruments) range from 0 - 111
        
        """
        chords_to_num = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
        features = np.array([])
        labels = np.array([])
        for note_number in range(128):
            for program_number in range(112):
                    #setting the label list
                    label = np.zeros(12)
                    note_name_octave = pretty_midi.note_number_to_name(note_number)
                    note_name = note_name_octave[:-1]
                    note_name_clean = note_name.replace("-", "")
                    chord_index = chords_to_num[note_name_clean]
                    label[chord_index] = 1
                    labels = np.append(labels, label)

                    velocity = 100

                    chord = pretty_midi.PrettyMIDI()
                    instrument = pretty_midi.Instrument(program=program_number)
                    note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=0, end=0.5)
                    instrument.notes.append(note)
                    chord.instruments.append(instrument)
                    audio_data = chord.synthesize()
                    features = np.append(features, audio_data)
        print(features)
        print(labels)
        
        pickle.dump(features, open("features.p", "wb"))
        pickle.dump(labels, ope("labels.p", "wb"))

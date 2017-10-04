import numpy as np
import pretty_midi
import pickle
import os
import abc
from random import randint
from tqdm import tqdm

class Generator(metaclass=abc.ABCMeta):
    def __init__(self, sample_rate, sample_length):
        self.sample_rate = sample_rate
        self.sample_length= sample_length

    def __iter__(self):
        return self

    @abc.abstractmethod
    def next(self):
        pass

class Generator1(Generator): # TODO: Give me a better name
    def __init__(self, sample_rate, sample_length):
        super().__init__(sample_rate, sample_length)
        self.create_data()

    def __iter__(self):
        return self

    def next(self):
        """
        Return a pair (features, label), where `features` is a list of
        numerical values, and `label` a list containing the expected output.
        """
        index = randint(0, len(self.features)-1)
        return (self.features[index], self.labels[index])

    def create_data(self):
        """
        MIDI note numbers range from 0 - 127.
        MIDI program number (non-percussion instruments) range from 0 - 111
        """

        if os.path.isfile("features.p") and os.path.isfile("labels.p"):
            self.features = pickle.load(open("features.p", "rb"))
            self.labels = pickle.load(open("labels.p", "rb"))
            return

        chords_to_num = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
        features = []
        labels = []
        for note_number in tqdm(range(128)):
            for program_number in range(112):
                #setting the label list
                label = np.zeros(12)
                note_name_octave = pretty_midi.note_number_to_name(note_number)
                note_name = note_name_octave[:-1]
                note_name_clean = note_name.replace("-", "")
                chord_index = chords_to_num[note_name_clean]
                label[chord_index] = 1
                labels.append(label)

                velocity = 100

                chord = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=program_number)
                note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=0, end=0.5)
                instrument.notes.append(note)
                chord.instruments.append(instrument)
                audio_data = chord.synthesize(fs=self.sample_rate)
                features.append(audio_data)
        
        pickle.dump(features, open("features.p", "wb"))
        pickle.dump(labels, open("labels.p", "wb"))

        self.features = features
        self.labels = labels

class Generator2(Generator): # TODO: Give me a better name
    def __init__(self, sample_rate, sample_length):
        super().__init__(sample_rate, sample_length)

    def __iter__(self):
        return self

    def next(self):
        """
        Return a pair (features, label), where `features` is a list of
        numerical values, and `label` a list containing the expected output.
        """
        chords_to_num = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
        #note_number = np.random.choice(range(128))
        note_number = np.random.choice(range(24,119))
        # Programs that don't work: 0,9,12,34
        program_number = np.random.choice(range(1,112))
        print("Note number: %d" % note_number)
        print("Program number: %d" % program_number)

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
        audio_data = chord.synthesize(fs=self.sample_rate)

        return (audio_data[:self.sample_length], label)

class WhiteNoiseGenerator(Generator):
    def __init__(self, sample_rate, sample_length):
        super().__init__(sample_rate, sample_length)

    def next(self) -> np.ndarray:
        return np.random.rand(self.sample_length)

class NoisyGenerator(Generator):
    def __init__(self, sample_rate, sample_length):
        super().__init__(sample_rate, sample_length)
        self.dg = Generator2(self.sample_rate, self.sample_length)
        self.ng = WhiteNoiseGenerator(self.sample_rate, self.sample_length)

    def next(self):
        point = self.dg.next()
        noise = self.ng.next()
        return (point[0]+noise,point[1])

if __name__=="__main__":
    import scipy.io.wavfile
    dg = NoisyGenerator(44100, 44100)
    scipy.io.wavfile.write("f.wav",44100, dg.next()[0])

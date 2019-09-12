import numpy as np
import pretty_midi
import pickle
import os
import itertools
import librosa
from random import randint
from tqdm import tqdm

import torch
import torchaudio

DEFAULT_SAMPLING_RATE = 44100
DEFAULT_SAMPLE_LENGTH = 44100
chords_to_num = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7,
        "G#":8, "A":9, "A#":10, "B":11}

class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self,max_tones=5,min_tones=0,transforms=None):
        self.transforms = transforms

        all_notes = itertools.product(*[[True,False]]*12)
        def filter_func(x):
            s = sum(x)
            return s <= max_tones and s >= min_tones
        filtered_notes = filter(filter_func,all_notes)
        self.notes = list(filtered_notes)

    def __getitem__(self,index):
        output = {'notes': self.notes[index]}
        if self.transforms is not None:
            return self.transforms(output)
        return output

    def __len__(self):
        return len(self.notes)

class ToNoteNumber(object):
    def __init__(self, starting_octave=3, num_octaves=1):
        assert starting_octave >= -1 and starting_octave <= 9
        assert num_octaves >= 1 and num_octaves <= 11

        lowest_note = (starting_octave+1)*12
        highest_note = min((starting_octave+1+num_octaves)*12,128)

        # Generate notes between [lowest_note,highest_note)
        self.notes = [[] for _ in range(12)]
        for n in range(lowest_note,highest_note):
            self.notes[n%12].append(n)

    def __call__(self, sample):
        notes = sample['notes']
        note_numbers = []
        for i,n in enumerate(notes):
            if n:
                note_numbers.append(np.random.choice(self.notes[i]))
        return {'notes': notes, 'note_numbers': note_numbers}

class SynthesizeSounds(object):
    def __init__(self, program_numbers=list(range(1,112)), velocity=100,
            sample_length=DEFAULT_SAMPLE_LENGTH,
            sampling_rate=DEFAULT_SAMPLING_RATE):
        assert len(program_numbers) > 0
        self.program_numbers = program_numbers
        self.velocity = velocity
        self.sampling_rate = sampling_rate
        self.sample_length = sample_length

    def __call__(self, sample):
        note_numbers = sample['note_numbers']

        program_number = np.random.choice(self.program_numbers)

        chord = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program_number)
        for note_number in note_numbers:
            note = pretty_midi.Note(velocity=self.velocity, pitch=note_number, start=0, end=1)
            instrument.notes.append(note)
        chord.instruments.append(instrument)
        audio_data = chord.synthesize(fs=self.sampling_rate)

        output = sample.copy()
        output['audio'] = audio_data[:-self.sampling_rate] # Last second is empty
        return output

class WhiteNoise(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        audio_data = sample['input']

        # TODO

        output = sample.copy()
        output['input'] = audio_data
        return output

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        notes = sample['notes']
        note_numbers = sample['note_numbers']
        audio_data = sample['audio']

        output = sample.copy()
        output['audio'] = torch.tensor(audio_data)
        return output

class Spectrogram(object):
    def __init__(self, *args, **kwargs):
        self.transform = torchaudio.transforms.Spectrogram(*args,**kwargs)

    def __call__(self, sample):
        audio_data = sample['audio']

        output = sample.copy()
        output['spectrogram'] = self.transform(audio_data.unsqueeze(0).float())
        return output

class Compose(object):
    """Copied from torchvision.transforms.Compose"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

if __name__=="__main__":
    transforms = Compose([
        ToNoteNumber(),
        SynthesizeSounds(),
        ToTensor(),
        Spectrogram()
    ])
    dataset = GeneratedDataset(transforms=transforms)
    print(dataset[-2]['audio'].shape)

    import scipy.io.wavfile
    scipy.io.wavfile.write("f.wav",44100, dataset[0]['audio'].numpy())


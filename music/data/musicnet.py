#import config
import numpy as np
from tqdm import tqdm
import torch
import os
import csv
import pickle
from intervaltree import IntervalTree
import scipy
from scipy.io import wavfile
import pretty_midi

class Memoize(object):
    def __init__(self, file_name, func):
        self.file_name = file_name
        self.func = func
    def get(self):
        if os.path.isfile(self.file_name):
            with open(self.file_name,'rb') as f:
                return pickle.load(f)
        else:
            val = self.func()
            with open(self.file_name,'wb') as f:
                pickle.dump(val,f)
            return val

##################################################
# Dataset
##################################################

class MusicNetDataset(torch.utils.data.Dataset):
    def __init__(self, musicnet_dir, train=True, transforms=None,
            window_size=400):
        self.transforms = transforms
        url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
        if train:
            self.labels_dir = os.path.join(musicnet_dir,'train_labels')
            self.data_dir = os.path.join(musicnet_dir,'train_data')
            labels_file_name = 'train_labels.pkl'
        else:
            self.labels_dir = os.path.join(musicnet_dir,'test_labels')
            self.data_dir = os.path.join(musicnet_dir,'test_data')
            labels_file_name = 'test_labels.pkl'
        assert os.path.isdir(self.labels_dir)
        assert os.path.isdir(self.data_dir)

        labels = Memoize(
                labels_file_name,
                lambda: self.process_labels(self.labels_dir))

        self.labels = labels.get()
        self.keys = list(self.labels.keys())
        self.data = list(self.labels.keys())

    def __getitem__(self,index):
        #k,t = self.data[index]
        index = self.data[index]
        wav_file_name = os.path.join(self.data_dir,'%d.wav'%index)
        rate,data = wavfile.read(wav_file_name)
        #start = t-int(self.window_size/2)
        #end = start+self.window_size
        #data = data[start:end]
        #output = {'intervals': self.labels[k][t], 'audio': data}
        output = {'interval_tree': self.labels[index], 'audio': data}
        if self.transforms is not None:
            return self.transforms(output)
        return output

    def __len__(self):
        return len(self.data)

    def process_labels(self, path):
        trees = dict()
        for item in tqdm(os.listdir(path)):
            if not item.endswith('.csv'): continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(path,item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)
            trees[uid] = tree
        return trees

class DiscretizedMusicNetDataset(MusicNetDataset):
    def __init__(self, musicnet_dir, train=True, transforms=None,
            window_size=400, points_per_song=10):
        super().__init__(musicnet_dir=musicnet_dir, train=train, transforms=transforms)
        self.window_size = window_size

        data = []
        for k,l in self.labels.items():
            length = l.end()
            # TODO: Compute indices for the n equidistant points
            # Divide song into n+1 equal parts
            # Each part is length/(n+1) in size
            d = int(length/(points_per_song+1))
            for i in range(d,length,d):
                data.append((k,i))
        self.data = data

    def __getitem__(self,index):
        k,t = self.data[index]
        wav_file_name = os.path.join(self.data_dir,'%d.wav'%k)
        rate,data = wavfile.read(wav_file_name)
        start = t-int(self.window_size/2)
        end = start+self.window_size
        data = data[start:end]
        output = {'intervals': self.labels[k][t], 'audio': data}
        if self.transforms is not None:
            return self.transforms(output)
        return output

##################################################
# Transforms
##################################################

class RandomCrop(object):
    def __init__(self, window_size=10000):
        self.window_size = window_size
    def __call__(self,sample):
        interval_tree = sample['interval_tree']
        audio = sample['audio']

        length = interval_tree.end()
        start = np.random.randint(0,length-self.window_size)
        end = start+self.window_size

        intervals = interval_tree[(start+end)//2]
        audio = audio[start:end]

        return {'intervals': intervals, 'audio': audio}

class IntervalsToNoteNumbers(object):
    def __call__(self,sample):
        intervals = sample['intervals']

        note_numbers = []
        for (start,end,(instrument,note,measure,beat,note_value)) in intervals:
            note_numbers.append(note)

        output = sample.copy()
        output['note_numbers'] = note_numbers
        return output

def interval_tree_to_midi(interval_tree,rate=44100):
    midi = pretty_midi.PrettyMIDI()
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    instrument = pretty_midi.Instrument(program=cello_program)
    for interval in interval_tree.all_intervals:
        start,end,(_,note,_,_,_) = interval
        note = pretty_midi.Note(velocity=100, pitch=note,
                start=start/rate, end=end/rate)
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi

if __name__=="__main__":
    from generator import Compose, NoteNumbersToVector, Spectrogram, ToTensor
    transforms = Compose([
        RandomCrop(),
        IntervalsToNoteNumbers(),
        NoteNumbersToVector(),
        ToTensor(),
        Spectrogram()
    ])
    dataset = MusicNetDataset('/home/howard/Datasets/musicnet',
            train=False,transforms=transforms)

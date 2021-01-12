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
        index = self.data[index]
        wav_file_name = os.path.join(self.data_dir,'%d.wav'%index)
        rate,data = wavfile.read(wav_file_name)
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
            min_window_size=2048, overlap=2048, points_per_song=10):
        super().__init__(musicnet_dir=musicnet_dir, train=train, transforms=transforms)

        """
         window 1   window 2     window 3
        |--------------|        |--------------|
                    |--------------|
                    |..|        |..|
                     overlap 1   overlap 2
        - There are `points_per_song-1` overlapping regions
            - i.e. Overlaps account for `overlap*(points_per_song-1)` points
        - The remaining regions are split evenly between `points_per_song` windows
            - i.e. `(L-overlap*(points_per_song-1))/points_per_song`
        - window size = (L-overlap*(points_per_song-1))/points_per_song
        - stride = window size - overlap
        """

        data = []
        for k,l in self.labels.items():
            length = l.end()
            window_size = (length-overlap*(points_per_song-1))//points_per_song
            stride = window_size-overlap
            if window_size < min_window_size:
                window_size = min_window_size
                stride = (window_size*points_per_song-length)//points_per_song
            for i in range(points_per_song):
                data.append((k,i*stride,i*stride+window_size))
        self.data = data

    def __getitem__(self,index):
        song_id,start,end = self.data[index]
        wav_file_name = os.path.join(self.data_dir,'%d.wav'%song_id)
        rate,data = wavfile.read(wav_file_name)
        output = {
                'interval_tree': self.labels[song_id],
                'audio': data,
                'start': start,
                'end': end
        }
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
        start = sample.pop('start',0)
        end = sample.pop('end',len(audio))

        length = end-start
        start = start+np.random.randint(0,length-self.window_size+1)
        end = start+self.window_size

        intervals = interval_tree[(start+end)//2]
        audio = audio[start:end]

        return {
                **sample,
                'intervals': intervals, 'audio': audio, 'start': start, 'end': end
        }

class CentreCrop(object):
    def __init__(self, window_size=10000):
        self.window_size = window_size
    def __call__(self,sample):
        interval_tree = sample['interval_tree']
        audio = sample['audio']
        start = sample.pop('start',0)
        end = sample.pop('end',len(audio))

        length = end-start
        start = start+(length+self.window_size)//2
        end = start+self.window_size

        intervals = interval_tree[(start+end)//2]
        audio = audio[start:end]

        return {
                **sample,
                'intervals': intervals, 'audio': audio, 'start': start, 'end': end
        }

class CropIntervals(object):
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride
    def __call__(self,sample):
        interval_tree = sample['interval_tree']
        start = sample.pop('start')
        end = sample.pop('end')

        intervals = []
        window_start = start
        while window_start+self.window_size <= end:
            intervals.append(interval_tree[window_start+self.window_size//2])
            window_start += self.stride

        return {
            **sample,
            'intervals': intervals
        }

class IntervalsToNoteNumbers(object):
    def __call__(self,sample):
        intervals = sample['intervals']

        if type(intervals) is list:
            note_numbers = []
            for i in intervals:
                note_numbers.append([])
                for (start,end,(instrument,note,measure,beat,note_value)) in i:
                    note_numbers[-1].append(note)
        else: # if type(intervals) is set
            note_numbers = []
            for (start,end,(instrument,note,measure,beat,note_value)) in intervals:
                note_numbers.append(note)

        output = sample.copy()
        output['note_numbers'] = note_numbers
        return output

class NoteNumbersToVector(object):
    def __call__(self, sample):
        note_numbers = sample['note_numbers']
        assert type(note_numbers) is list

        def convert(note_numbers):
            vector = torch.zeros([128])
            for n in note_numbers:
                vector[n] = 1
            return vector

        output = sample.copy()
        if len(note_numbers) > 0 and type(note_numbers[0]) is list:
            output['note_numbers'] = torch.stack([convert(n) for n in note_numbers])
        else:
            output['note_numbers'] = convert(note_numbers)
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

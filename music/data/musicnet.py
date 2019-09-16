#import config
import numpy as np
from tqdm import tqdm
import torch
import os
import csv
import pickle
from intervaltree import IntervalTree
import scipy
import pretty_midi

def load_data(mem=False):
    file_name = config.MUSICNET_FILE
    if mem:
        return dict(np.load(open(file_name,'rb'),encoding='latin1'))
    else:
        return dict(np.load(open(file_name,'rb'),encoding='latin1',mmap_mode="r"))

def split_input_output(data, input_dims=2048, sampling_rate=44100, stride=512, random=False):
    # TODO: Does not currently work for different stride lengths
    # TODO: Compute the number of data points using data['2382'][0].shape[0]
    features = 0    # first element of (X,Y) data tuple
    labels = 1      # second element of (X,Y) data tuple

    ids = list(data.keys())
    if random:
        x = np.empty([len(data),input_dims])
        y = np.zeros([len(data),128])
        for i in range(len(ids)):
            # Pick a random spot in the audio track
            s = np.random.randint(
                    input_dims/2,
                    len(data[ids[i]][features])-input_dims/2)
            x[i] = data[ids[i]][features][int(s-input_dims/2):int(s+input_dims/2)]
            for label in data[ids[i]][labels][s]:
                y[i,label.data[1]] = 1
    else:
        x = np.empty([len(data)*7500,input_dims])
        y = np.zeros([len(data)*7500,128])
        for i in range(len(ids)):
            for j in range(7500):
                index = sampling_rate+j*stride # start from one second to give us some wiggle room for larger segments
                x[7500*i + j] = data[ids[i]][features][index:index+input_dims]
                
                # label stuff that's on in the center of the window
                for label in data[ids[i]][labels][index+input_dims/2]:
                    y[7500*i + j,label.data[1]] = 1

    x = Variable(torch.from_numpy(x).float(), requires_grad=False)
    y = Variable(torch.from_numpy(y).float(), requires_grad=False)

    return x,y

def get_random_batch(data, ids, input_dims=2048, sampling_rate=44100):
    x = np.empty([len(data),input_dims])
    y = np.zeros([len(data),128])
    for i in range(len(ids)):
        # Pick a random spot in the audio track
        s = np.random.randint(
                input_dims/2,
                len(data.get(ids[i]).get("data"))-input_dims/2)
        x[i] = data.get(ids[i]).get("data")[int(s-input_dims/2):int(s+input_dims/2)]
        y[i] = np.unpackbits(data.get(ids[i]).get("labels")[s])
    x = Variable(torch.from_numpy(x).float(), requires_grad=False).cuda()
    y = Variable(torch.from_numpy(y).float(), requires_grad=False).cuda()
    return x,y

def get_test_set(data, ids, input_dims=2048, sampling_rate=44100):
    stride = 512*8
    x = np.empty([len(ids)*750,input_dims])
    y = np.zeros([len(ids)*750,128])
    for i in range(len(ids)):
        for j in tqdm(range(750)):
            index = sampling_rate+j*stride # start from one second to give us some wiggle room for larger segments
            x[750*i + j] = data.get(ids[i]).get("data")[index:index+input_dims]
            y[750*i + j] = np.unpackbits(data.get(ids[i]).get("labels")[index+input_dims/2])
    x = Variable(torch.from_numpy(x).float(), requires_grad=False,
            volatile=True).cuda()
    y = Variable(torch.from_numpy(y).float(), requires_grad=False,
            volatile=True).cuda()
    return x,y

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

class MusicNetDataset(torch.utils.data.Dataset):
    def __init__(self, musicnet_dir, train=True, transforms=None,
            points_per_song=1, window_size=10000):
        self.transforms = transforms
        self.window_size = window_size
        self.points_per_song = points_per_song
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
        self.data = []
        for k,l in self.labels.items():
            length = l.end()
            # TODO: Compute indices for the n equidistant points
            # Divide song into n+1 equal parts
            # Each part is length/(n+1) in size
            d = int(length/(points_per_song+1))
            for i in range(d,length,d):
                self.data.append((k,i))

    def __getitem__(self,index):
        k,t = self.data[index]
        wav_file_name = os.path.join(self.data_dir,'%d.wav'%k)
        rate,data = scipy.io.wavfile.read(wav_file_name)
        start = t-int(self.window_size/2)
        end = start+self.window_size
        data = data[start:end]
        output = {'intervals': self.labels[k][t], 'audio': data}
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

class RandomWindow(object):
    def __init__(self, window_size=10000):
        self.window_size = window_size
    def __call__(self,sample):
        interval_tree = sample['interval_tree']
        audio = sample['audio']

        length = interval_tree.end()
        start = np.random.randint(0,length-self.window_size)
        end = start+self.window_size

        intervals = interval_tree[int((start+end)/2)]
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
        #RandomWindow(),
        IntervalsToNoteNumbers(),
        NoteNumbersToVector(),
        ToTensor(),
        Spectrogram()
    ])
    dataset = MusicNetDataset('/mnt/ppl-3/musicnet/musicnet',
            train=False,transforms=transforms)

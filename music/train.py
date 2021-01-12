import os
import copy
import dill
from tqdm import tqdm
import scipy.io.wavfile
import torch
import torchaudio
import pretty_midi
import itertools
import typer
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from experiment import Experiment

import data
import data.generator
import data.musicnet

class Net(torch.nn.Module):
    def __init__(self, structure=[1025,512,128]):
        super().__init__()
        seq = []
        for in_size,out_size in zip(structure,structure[1:]):
            seq.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            seq.append(torch.nn.ReLU())
        seq = seq[:-1] # Remove last ReLU
        self.seq = torch.nn.Sequential(*seq)
    def forward(self,x):
        sample_length = x.shape[1]
        x = torchaudio.functional.spectrogram(
                waveform=x,
                n_fft=2048, # librosa recommends 2048 for music.
                win_length=sample_length,
                hop_length=sample_length+1,
                pad=0, 
                normalized=False,
                window=torch.hann_window(sample_length), # See https://en.wikipedia.org/wiki/Window_function
                power=2
        )
        x = x.squeeze()
        return self.seq(x).squeeze()

class Net2(torch.nn.Module):
    def __init__(self, structure=[2048,512,128]):
        super().__init__()
        seq = []
        for in_size,out_size in zip(structure,structure[1:]):
            seq.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            seq.append(torch.nn.ReLU())
        seq = seq[:-1] # Remove last ReLU
        self.seq = torch.nn.Sequential(*seq)
    def forward(self,x):
        return self.seq(x).squeeze()

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1,out_channels=128,kernel_size=2048)
        )
    def forward(self,x):
        x = x.unsqueeze(1)
        return self.seq(x).squeeze()

class RecurrentUnit(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        #self.gru = torch.nn.GRUCell(input_size=input_size,hidden_size=hidden_size)
        self.fc_output = torch.nn.Linear(in_features=input_size+hidden_size,out_features=output_size),
        self.fc_hidden = torch.nn.Linear(in_features=input_size+hidden_size,out_features=hidden_size),
    def forward(self, x, h):
        cat_input = torch.cat([x,h],1)
        #x = self.gru(x,h)
        return x,x

class GRUNet(torch.nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        self.input_size = input_size
        self.seq = torch.nn.Sequential(
                torch.nn.Linear(in_features=input_size,out_features=1024),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=1024,out_features=512),
                torch.nn.ReLU(),
        )
        #self.gru = torch.nn.GRUCell(input_size=512,hidden_size=512)
        self.gru = RecurrentUnit(input_size=512,hidden_size=512,output_size=512)
        self.seq2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=512,out_features=128),
        )
    def forward(self, x, stride):
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        length = x.shape[1]
        outputs = []
        for i in range((length-self.input_size)//stride+1):
            left = i*stride
            right = left+self.input_size

            # Pass through neural network
            o = self.seq(x[:,left:right])
            o,hidden = self.gru(o,hidden)
            o = self.seq2(o)
            outputs.append(o)
        return torch.stack(outputs,1) # batch size, seq len, features
    def init_hidden(self, batch_size):
        return -10*torch.ones([batch_size, 512])

def get_datasets(dataset_dir='/mnt/ppl-3/musicnet/musicnet', window_size=2048):
    #generated_transforms = data.generator.Compose([
    #    data.generator.ToNoteNumber(),
    #    data.generator.SynthesizeSounds(),
    #    data.generator.NoteNumbersToVector(),
    #    data.generator.ToTensor(),
    #    data.generator.Spectrogram(),
    #    data.generator.Filter(['note_numbers','spectrogram'])
    #])
    #generated_dataset = data.generator.GeneratedDataset(transforms=generated_transforms)

    # Training set
    musicnet_transforms = data.generator.Compose([
        data.musicnet.RandomCrop(window_size),
        data.musicnet.IntervalsToNoteNumbers(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Filter(['note_numbers','audio'])
    ])
    musicnet_train_dataset = data.musicnet.DiscretizedMusicNetDataset(
            dataset_dir,transforms=musicnet_transforms,
            train=True, min_window_size=window_size, points_per_song=1000, overlap=2048)

    # Validation set
    musicnet_transforms_val = data.generator.Compose([
        data.musicnet.CentreCrop(window_size),
        data.musicnet.IntervalsToNoteNumbers(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Filter(['note_numbers','audio'])
    ])
    musicnet_val_dataset = data.musicnet.DiscretizedMusicNetDataset(
            dataset_dir,transforms=musicnet_transforms_val,
            train=False, min_window_size=window_size, points_per_song=10)

    return musicnet_train_dataset, musicnet_val_dataset
    #return generated_dataset, musicnet_test_dataset

def get_datasets_recurrent(dataset_dir='/mnt/ppl-3/musicnet/musicnet', window_size=2048, seq_len=5, stride=1024):
    # Training set
    musicnet_transforms = data.generator.Compose([
        data.musicnet.RandomCrop(window_size+stride*(seq_len-1)),
        data.musicnet.CropIntervals(window_size,stride),
        data.musicnet.IntervalsToNoteNumbers(),
        data.musicnet.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Filter(['note_numbers','audio'])
    ])
    musicnet_train_dataset = data.musicnet.DiscretizedMusicNetDataset(
            dataset_dir,transforms=musicnet_transforms,
            train=True, min_window_size=window_size+stride*(seq_len-1), points_per_song=256, overlap=2048)

    # Validation set
    musicnet_transforms_val = data.generator.Compose([
        #data.musicnet.CentreCrop(window_size+stride*(seq_len-1)),
        data.musicnet.CropAudio(),
        data.musicnet.CropIntervals(window_size,stride),
        data.musicnet.IntervalsToNoteNumbers(),
        data.musicnet.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Filter(['note_numbers','audio'])
    ])
    musicnet_val_dataset = data.musicnet.DiscretizedMusicNetDataset(
            dataset_dir,transforms=musicnet_transforms_val,
            train=False, min_window_size=window_size, points_per_song=1)

    return musicnet_train_dataset, musicnet_val_dataset
    #return generated_dataset, musicnet_test_dataset

##################################################
# Checkpointing
##################################################

def save_checkpoint(file_name, model, optim, loss_history):
    checkpoint = {
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'loss_history': loss_history
    }
    torch.save(checkpoint,file_name)

def load_checkpoint(file_name, model, optim, loss_history):
    try:
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        for k,v in checkpoint['loss_history'].items():
            loss_history[k] = v
    except FileNotFoundError:
        print('Checkpoint not found. Skipping.')

##################################################
# Conversion to MIDI/wav
##################################################

def to_midi(net,wav_file_name,n_fft=400,win_length=None,hop_length=None,
        threshold=0.7):
    win_length = win_length if win_length is not None else n_fft
    hop_length = hop_length if hop_length is not None else win_length // 2

    # Compute Note Prediction
    rate,data = scipy.io.wavfile.read(wav_file_name)
    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
            win_length=win_length, hop_length = hop_length)
    spectrogram = transform(
            torch.tensor(data).view(1,-1)) # (channel, freq, time)
    spectrogram = spectrogram.permute(2,0,1).squeeze() # (time, freq)
    prediction = net(spectrogram) > threshold

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    instrument = pretty_midi.Instrument(program=cello_program)

    # Convert prediction to MIDI
    notes = [None]*128
    for t in tqdm(range(prediction.shape[0]), desc='Converting to MIDI'):
        for n in range(128):
            if prediction[t,n]:
                if notes[n] is None:
                    notes[n] = (t,t+1)
                else:
                    start,end = notes[n]
                    notes[n] = (start,end+1)
            else:
                if notes[n] is None:
                    continue
                else:
                    start,end = notes[n]
                    start = (win_length/2+start*hop_length)/rate
                    end = (win_length/2+end*hop_length)/rate
                    notes[n] = None
                    #tqdm.write('Note %d played from %f to %f.' % (n,start,end))
                    note = pretty_midi.Note(velocity=100, pitch=n,
                            start=start, end=end)
                    instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi

def to_midi2(convert,wav_file_name,win_length=400,hop_length=None):
    hop_length = hop_length if hop_length is not None else win_length // 2

    # Compute Note Prediction
    rate,data = scipy.io.wavfile.read(wav_file_name)
    #data = data[:44100*10]
    slices = [slice(i,i+win_length) for i in range(0,len(data)-win_length,hop_length)]
    predictions = (convert(data[s]) for s in slices)

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    instrument = pretty_midi.Instrument(program=cello_program)

    # Convert prediction to MIDI
    return notes_to_midi(predictions)
    #notes = [None]*128
    #for t,p in tqdm(enumerate(predictions), desc='Converting to MIDI', total=len(slices)):
    #    for n in range(128):
    #        if p[n]:
    #            if notes[n] is None:
    #                notes[n] = (t,t+1)
    #            else:
    #                start,end = notes[n]
    #                notes[n] = (start,end+1)
    #        else:
    #            if notes[n] is None:
    #                continue
    #            else:
    #                start,end = notes[n]
    #                start = (win_length/2+start*hop_length)/rate
    #                end = (win_length/2+end*hop_length)/rate
    #                notes[n] = None
    #                #tqdm.write('Note %d played from %f to %f.' % (n,start,end))
    #                note = pretty_midi.Note(velocity=100, pitch=n,
    #                        start=start, end=end)
    #                instrument.notes.append(note)
    #midi.instruments.append(instrument)
    #return midi

def get_convert_fn(net,n_fft=400,threshold=0.7):
    def convert(data):
        #transform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
        #        win_length=len(data), hop_length=len(data))
        #d = torch.tensor(data).view(1,-1)
        #spectrogram = transform(d) # (channel, freq, time)
        #spectrogram = spectrogram.permute(2,0,1).squeeze() # (time, freq)
        data = torch.tensor(data).unsqueeze(0)
        prediction = torch.sigmoid(net(data)) > threshold
        return prediction
    return convert

def get_convert_fn_ground_truth(full_data,interval_tree):
    start_index = 0
    def convert(data):
        nonlocal start_index
        output = torch.zeros([1,128])
        for i in range(0,len(full_data)-start_index,200): 
            for j in range(len(data)): # Check if index i is a match
                if abs(full_data[start_index+i+j] - data[j]) > 1e-10:
                    break
            else:
                tqdm.write('%d %d' % (start_index+i,j))
                intervals = interval_tree[start_index+i+j+len(data)//2]
                start_index = start_index+i
                for _,_,(_,note,_,_,_) in intervals:
                    output[0,note] = 1
                return output
        return output
    return convert

def notes_to_midi(notes, win_length, hop_length, rate):
    """
    Convert a tensor of notes into a MIDI object.

    Args:
        notes: A boolean tensor of size (song length, 128).
    """

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    instrument = pretty_midi.Instrument(program=cello_program)

    # Convert prediction to MIDI
    last_notes = [None]*128
    for t,p in tqdm(enumerate(notes), desc='Converting to MIDI', total=notes.shape[0]):
        for n in range(128):
            if p[n]:
                if last_notes[n] is None:
                    last_notes[n] = (t,t+1)
                else:
                    start,end = last_notes[n]
                    last_notes[n] = (start,end+1)
            else:
                if last_notes[n] is None:
                    continue
                else:
                    start,end = last_notes[n]
                    start = (win_length/2+start*hop_length)/rate
                    end = (win_length/2+end*hop_length)/rate
                    last_notes[n] = None
                    # Create and add MIDI note
                    note = pretty_midi.Note(velocity=100, pitch=n,
                            start=start, end=end)
                    instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi

##################################################
# Experiment
##################################################

class MusicNetExperiment(Experiment):
    def setup(self, config):
        output_dir = config.pop('output_dir','./output')
        self.output_dir = output_dir
        musicnet_dir = config.pop('musicnet_dir','/home/howard/Datasets/musicnet')
        self.best_model_file_name = os.path.join(output_dir,'best_model.pkl')
        self.checkpoint_file_name = os.path.join(output_dir,'checkpoint.pkl')
        network_structure = config.pop('network_structure',[])
        learning_rate = config.pop('learning_rate',0.01)

        if not os.path.isdir(output_dir):
            print('%s does not exist. Creating directory.' % output_dir)
            os.mkdir(output_dir)

        #self.net = Net(network_structure)
        #self.net = ConvNet()
        self.net = Net2(network_structure)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.state['loss_history'] = {
                'validation': [],
                'train': [],
                'iteration': [],
        }

        batch_size = config.pop('batch_size',64)
        train_dataset, val_dataset = get_datasets(musicnet_dir)
        self.train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=batch_size, shuffle=False)

        self.state['best_model_train'] = copy.deepcopy(self.net)
        self.state['best_loss_train'] = float('inf')
        self.state['best_model_val'] = copy.deepcopy(self.net)
        self.state['best_loss_val'] = float('inf')

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def run_step(self,iteration):
        criterion = self.criterion
        net = self.net
        optimizer = self.optimizer

        total_val_loss = 0
        for d in tqdm(self.val_dataloader,desc='Validation'):
            x = d['audio']
            y = d['note_numbers']
            est = net(x)
            loss = criterion(est,y)
            total_val_loss += loss.item()
        total_val_loss /= len(self.val_dataloader)

        total_train_loss = 0
        for d in tqdm(self.train_dataloader,desc='Training'):
            x = d['audio']
            y = d['note_numbers']
            est = net(x)
            loss = criterion(est,y)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_loss /= len(self.train_dataloader)

        if total_train_loss < self.state['best_loss_train']:
            print('New best model saved.')
            self.state['best_model_train'] = copy.deepcopy(self.net)
            self.state['best_loss_train'] = total_train_loss
            best_model_path = os.path.join(self.directory, 'best_model_train.pt')
            torch.save(self.net.state_dict(), best_model_path)

        if total_val_loss < self.state['best_loss_val']:
            print('New best model saved.')
            self.state['best_model_val'] = copy.deepcopy(self.net)
            self.state['best_loss_val'] = total_val_loss
            best_model_path = os.path.join(self.directory, 'best_model_val.pt')
            torch.save(self.net.state_dict(), best_model_path)

        self.state['loss_history']['train'].append(total_train_loss)
        self.state['loss_history']['validation'].append(total_val_loss)
        self.state['loss_history']['iteration'].append(iteration)
        print('%d\t Train Loss: %f \t Val Loss: %f' % 
                (iteration,total_train_loss,total_val_loss))

    def after_epoch(self, iteration):
        loss_history = self.state['loss_history']
        output_dir = self.output_dir

        plt.figure()
        plt.plot(loss_history['iteration'],loss_history['train'],label='train')
        plt.plot(loss_history['iteration'],loss_history['validation'],label='validation')
        plt.xlabel('Training Iterations')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.grid()
        plot_filename = os.path.join(output_dir, 'plot.png')
        plt.savefig(plot_filename)
        print('Saved plot at %s' % plot_filename)

        plt.figure()
        plt.plot(loss_history['iteration'],[np.log(y) for y in loss_history['train']],label='train')
        plt.plot(loss_history['iteration'],[np.log(y) for y in loss_history['validation']],label='validation')
        plt.xlabel('Training Iterations')
        plt.ylabel('Log Loss')
        plt.legend(loc='best')
        plt.grid()
        plot_filename = os.path.join(output_dir, 'log-plot.png')
        plt.savefig(plot_filename)
        print('Saved plot at %s' % plot_filename)

class MusicNetExperimentRNN(Experiment):
    def setup(self, config):
        output_dir = config.pop('output_dir','./output')
        self.output_dir = output_dir
        musicnet_dir = config.pop('musicnet_dir','/home/howard/Datasets/musicnet')
        self.best_model_file_name = os.path.join(output_dir,'best_model.pkl')
        self.checkpoint_file_name = os.path.join(output_dir,'checkpoint.pkl')
        learning_rate = config.pop('learning_rate',0.01)

        window_size = config.pop('window_size',2048)
        seq_len = config.pop('seq_len',10)
        self.stride = config.pop('stride',1024)

        if not os.path.isdir(output_dir):
            print('%s does not exist. Creating directory.' % output_dir)
            os.mkdir(output_dir)

        self.net = GRUNet(window_size)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.state['loss_history'] = {
                'validation': [],
                'train': [],
                'iteration': [],
        }

        batch_size = config.pop('batch_size',64)
        train_dataset, val_dataset = get_datasets_recurrent(musicnet_dir, window_size=window_size, seq_len=seq_len,stride=self.stride)
        self.train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=batch_size, shuffle=False)

        self.state['best_model_train'] = copy.deepcopy(self.net)
        self.state['best_loss_train'] = float('inf')
        self.state['best_model_val'] = copy.deepcopy(self.net)
        self.state['best_loss_val'] = float('inf')

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def run_step(self,iteration):
        criterion = self.criterion
        net = self.net
        optimizer = self.optimizer

        total_val_loss = []
        for d in tqdm(self.val_dataloader,desc='Validation'):
            x = d['audio']
            y = d['note_numbers']
            est = net(x,self.stride)

            loss = criterion(est,y)
            total_val_loss.append(loss.item())
        total_val_loss = np.mean(total_val_loss)
        val_pred_mean = ((torch.sigmoid(est)>0.5)+0.).mean()

        total_train_loss = []
        for d in tqdm(self.train_dataloader,desc='Training'):
            x = d['audio']
            y = d['note_numbers']
            est = net(x, self.stride)

            loss = criterion(est,y)
            total_train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_loss = np.mean(total_train_loss)

        if total_train_loss < self.state['best_loss_train']:
            print('New best model saved.')
            self.state['best_model_train'] = copy.deepcopy(self.net)
            self.state['best_loss_train'] = total_train_loss
            best_model_path = os.path.join(self.directory, 'best_model_train.pt')
            torch.save(self.net.state_dict(), best_model_path)

        if total_val_loss < self.state['best_loss_val']:
            print('New best model saved.')
            self.state['best_model_val'] = copy.deepcopy(self.net)
            self.state['best_loss_val'] = total_val_loss
            best_model_path = os.path.join(self.directory, 'best_model_val.pt')
            torch.save(self.net.state_dict(), best_model_path)

        self.state['loss_history']['train'].append(total_train_loss)
        self.state['loss_history']['validation'].append(total_val_loss)
        self.state['loss_history']['iteration'].append(iteration)
        print('%d\t Train Loss: %f \t Val Loss: %f\t%f' % 
                (iteration,total_train_loss,total_val_loss,val_pred_mean))

    def after_epoch(self, iteration):
        loss_history = self.state['loss_history']
        output_dir = self.output_dir

        plt.figure()
        plt.plot(loss_history['iteration'],loss_history['train'],label='train')
        plt.plot(loss_history['iteration'],loss_history['validation'],label='validation')
        plt.xlabel('Training Iterations')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.grid()
        plot_filename = os.path.join(output_dir, 'plot.png')
        plt.savefig(plot_filename)
        print('Saved plot at %s' % plot_filename)

        plt.figure()
        plt.plot(loss_history['iteration'],[np.log(y) for y in loss_history['train']],label='train')
        plt.plot(loss_history['iteration'],[np.log(y) for y in loss_history['validation']],label='validation')
        plt.xlabel('Training Iterations')
        plt.ylabel('Log Loss')
        plt.legend(loc='best')
        plt.grid()
        plot_filename = os.path.join(output_dir, 'log-plot.png')
        plt.savefig(plot_filename)
        print('Saved plot at %s' % plot_filename)

##################################################
# Evaluation
##################################################

app = typer.Typer()

@app.command()
def train(output_dir='./output', best_model_file_name=None,
        checkpoint_file_name=None, musicnet_dir=None):
    config = {
        #'network_structure': [1025,1024,128],
        'network_structure': [2048,1024,128],
        'batch_size': 64
    }
    exp = MusicNetExperiment(
            epoch=5,
            checkpoint_frequency=10,
            directory=output_dir,
            config=config
    )
    exp.run()

@app.command()
def train_rnn(output_dir='./output', best_model_file_name=None,
        checkpoint_file_name=None, musicnet_dir=None):
    config = {
        'batch_size': 64
    }
    exp = MusicNetExperimentRNN(
            epoch=5,
            checkpoint_frequency=10,
            directory=output_dir,
            config=config
    )
    exp.run()

@app.command()
def foo(checkpoint_path, name=1727):
    with open(checkpoint_path,'rb') as f:
        checkpoint = dill.load(f)
    net = checkpoint['best_model']

@app.command()
def convert(checkpoint_path, audio_file_path, output_dir='./output', threshold=0.7):
    with open(checkpoint_path,'rb') as f:
        checkpoint = dill.load(f)
    net = checkpoint['best_model_train']
    #midi = to_midi(net,os.path.join(musicnet_dir,'test_data','1759.wav'))
    #midi = to_midi(net,os.path.join(musicnet_dir,'train_data','1727.wav'))
    #midi = data.musicnet.interval_tree_to_midi(val_dataset.labels[1759])
    convert = get_convert_fn(net, n_fft=2048, threshold=threshold)

    #input_wav_file_name = os.path.join(musicnet_dir,'test_data','1759.wav')
    input_wav_file_name = audio_file_path
    rate,data = scipy.io.wavfile.read(input_wav_file_name)
    #convert = get_convert_fn_ground_truth(data,val_dataset.labels[1759])

    # Output MIDI
    midi = to_midi2(convert, input_wav_file_name, win_length=2048)
    midi_file_name = os.path.join(output_dir, 'output.mid')
    midi.write(midi_file_name)

    # Output synthesized MIDI as wav file
    wav = midi.fluidsynth()
    wav_file_name = os.path.join(output_dir, 'output.wav')
    scipy.io.wavfile.write(wav_file_name,rate=44100,data=wav)

@app.command()
def convert_rnn(checkpoint_path, audio_file_path, output_dir='./output', threshold=0.5):
    with open(checkpoint_path,'rb') as f:
        checkpoint = dill.load(f)
    net = checkpoint['best_model_train']
    rate,data = scipy.io.wavfile.read(audio_file_path)
    data = torch.tensor(data).unsqueeze(0)
    notes = torch.sigmoid(net(data,1024)) > threshold
    notes = notes.squeeze()

    # Output MIDI
    midi = notes_to_midi(notes, win_length=2048, hop_length=1024, rate=rate)
    midi_file_name = os.path.join(output_dir, 'output.mid')
    midi.write(midi_file_name)

    # Output synthesized MIDI as wav file
    wav = midi.fluidsynth()
    wav_file_name = os.path.join(output_dir, 'output.wav')
    scipy.io.wavfile.write(wav_file_name,rate=44100,data=wav)

@app.command()
def convert_ground_truth(output_dir='./output'):
    musicnet_dir = '/home/howard/Datasets/musicnet'
    window_size = 2048
    seq_len = 1000
    stride = 1024
    rate = 44100
    train_dataset, val_dataset = get_datasets_recurrent(musicnet_dir, window_size=window_size, seq_len=seq_len,stride=stride)

    x = val_dataset[0]
    notes = x['note_numbers']

    # Output MIDI
    midi = notes_to_midi(notes, win_length=2048, hop_length=1024, rate=rate)
    midi_file_name = os.path.join(output_dir, 'output.mid')
    midi.write(midi_file_name)

    # Output synthesized MIDI as wav file
    wav = midi.fluidsynth()
    wav_file_name = os.path.join(output_dir, 'output.wav')
    scipy.io.wavfile.write(wav_file_name,rate=44100,data=wav)

    wav_file_name = os.path.join(output_dir, 'output2.wav')
    scipy.io.wavfile.write(wav_file_name,rate=44100,data=x['audio'].numpy())

if __name__=="__main__":
    app()

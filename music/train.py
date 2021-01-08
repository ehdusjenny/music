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
    def __init__(self, structure=[201,200,128], input_size=400):
        super().__init__()
        seq = []
        for in_size,out_size in zip(structure,structure[1:]):
            seq.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            seq.append(torch.nn.ReLU())
        seq = seq[:-1] # Remove last ReLU
        self.seq = torch.nn.Sequential(*seq)
        self.n_fft = (structure[0]-1)*2
        #self.fft = torchaudio.transforms.Spectrogram(n_fft=n_fft,
        #        win_length=input_size, hop_length=input_size)
    def forward(self,x):
        sample_length = x.shape[1]
        x = torchaudio.functional.spectrogram(waveform=x, n_fft=self.n_fft, win_length=sample_length, hop_length=sample_length+1, pad=0, normalized=False, window=torch.hann_window(sample_length), power=2)
        x = x.squeeze()
        return self.seq(x).squeeze()

class ConvNet(torch.nn.Module):
    def __init__(self, structure=[201,200,128], input_size=400):
        super().__init__()
        self.seq = torch.nn.Sequential(
                torch.nn.Conv1D(in_channels=1,out_channels=16,kernel_size=512)
        )
        self.n_fft = (structure[0]-1)*2
        #self.fft = torchaudio.transforms.Spectrogram(n_fft=n_fft,
        #        win_length=input_size, hop_length=input_size)
    def forward(self,x):
        sample_length = x.shape[1]
        x = torchaudio.functional.spectrogram(waveform=x, n_fft=self.n_fft, win_length=sample_length, hop_length=sample_length+1, pad=0, normalized=False, window=torch.hann_window(sample_length), power=1)
        x = self.fft(x)
        return self.seq(x).squeeze()

def get_datasets(dataset_dir='/mnt/ppl-3/musicnet/musicnet'):
    #generated_transforms = data.generator.Compose([
    #    data.generator.ToNoteNumber(),
    #    data.generator.SynthesizeSounds(),
    #    data.generator.NoteNumbersToVector(),
    #    data.generator.ToTensor(),
    #    data.generator.Spectrogram(),
    #    data.generator.Filter(['note_numbers','spectrogram'])
    #])
    #generated_dataset = data.generator.GeneratedDataset(transforms=generated_transforms)
    musicnet_transforms = data.generator.Compose([
        data.musicnet.RandomCrop(400),
        data.musicnet.IntervalsToNoteNumbers(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        #data.generator.Spectrogram(),
        data.generator.Filter(['note_numbers','audio'])
    ])
    musicnet_train_dataset = data.musicnet.MusicNetDataset(
            dataset_dir,transforms=musicnet_transforms,
            train=True)
    musicnet_transforms_val = data.generator.Compose([
        data.musicnet.IntervalsToNoteNumbers(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Filter(['note_numbers','audio'])
    ])
    musicnet_test_dataset = data.musicnet.DiscretizedMusicNetDataset(
            dataset_dir,transforms=musicnet_transforms_val,
            train=False, window_size=400, points_per_song=10)
    #return musicnet_train_dataset, musicnet_test_dataset
    return musicnet_test_dataset, musicnet_test_dataset
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
    notes = [None]*128
    for t,p in tqdm(enumerate(predictions), desc='Converting to MIDI', total=len(slices)):
        for n in range(128):
            if p[n]:
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

        if not os.path.isdir(output_dir):
            print('%s does not exist. Creating directory.' % output_dir)
            os.mkdir(output_dir)

        self.net = Net([201,200,200,128])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.state['loss_history'] = {
                'validation': [],
                'train': [],
                'iteration': [],
        }

        batch_size = config.pop('batch_size',32)
        train_dataset, val_dataset = get_datasets(musicnet_dir)
        train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,num_samples=batch_size)
        val_sampler = torch.utils.data.RandomSampler(val_dataset,replacement=True,num_samples=batch_size)
        self.train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset, sampler=train_sampler, batch_size=batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False)

        self.state['best_model'] = copy.deepcopy(self.net)
        self.state['best_loss'] = float('inf')

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
        if total_train_loss < 0.05:
            breakpoint()

        if total_val_loss < self.state['best_loss']:
            print('New best model saved.')
            self.state['best_model'] = copy.deepcopy(self.net)
            self.state['best_loss'] = total_val_loss
            best_model_path = os.path.join(self.directory, 'best_model.pt')
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

##################################################
# Evaluation
##################################################

app = typer.Typer()

@app.command()
def train(output_dir='./output', best_model_file_name=None,
        checkpoint_file_name=None, musicnet_dir=None):
    """ Train a model on the MusicNet dataset. """

    if not os.path.isdir(output_dir):
        print('%s does not exist. Creating directory.' % output_dir)
        os.mkdir(output_dir)

    best_model_file_name = os.path.join(output_dir,'best_model.pkl')
    checkpoint_file_name = os.path.join(output_dir,'checkpoint.pkl')
    musicnet_dir = '/home/howard/Datasets/musicnet'

    train_dataset, val_dataset = get_datasets(musicnet_dir)

    net = Net([201,200,200,128])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_history = {
            'validation': [],
            'train': [],
            'iteration': [],
    }
    load_checkpoint(checkpoint_file_name,net,optimizer,loss_history)

    batch_size = 32
    train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,num_samples=batch_size)
    val_sampler = torch.utils.data.RandomSampler(val_dataset,replacement=True,num_samples=batch_size)
    train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, sampler=train_sampler, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False)

    best_model = copy.deepcopy(net)
    best_loss = float('inf')

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    try:
        #for iteration in range(2):
        for iteration in itertools.count():
            total_val_loss = 0
            for d in tqdm(val_dataloader,desc='Validation'):
                x = d['audio']
                y = d['note_numbers']
                est = net(x)
                loss = criterion(est,y)
                total_val_loss += loss.item()
            total_val_loss /= len(val_dataloader)

            total_train_loss = 0
            for d in tqdm(train_dataloader,desc='Training'):
                x = d['audio']
                y = d['note_numbers']
                est = net(x)
                loss = criterion(est,y)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_train_loss /= len(train_dataloader)

            if total_val_loss < best_loss:
                print('New best model saved.')
                best_model = copy.deepcopy(net)
                best_loss = total_val_loss

            loss_history['train'].append(total_train_loss)
            loss_history['validation'].append(total_val_loss)
            loss_history['iteration'].append(iteration)
            print('%d\t Train Loss: %f \t Val Loss: %f' % 
                    (iteration,total_train_loss,total_val_loss))
            #print('Train Loss: %f' % total_train_loss)
            save_checkpoint(checkpoint_file_name,net,optimizer,loss_history)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')

    plt.figure()
    plt.plot(loss_history['iteration'],loss_history['train'],label='train')
    plt.plot(loss_history['iteration'],loss_history['validation'],label='validation')
    plt.legend(loc='best')
    plot_filename = os.path.join(output_dir, 'plot.png')
    plt.savefig(plot_filename)
    print('Saved plot at %s' % plot_filename)

    plt.figure()
    plt.plot(loss_history['iteration'],[np.log(y) for y in loss_history['train']],label='train')
    plt.plot(loss_history['iteration'],[np.log(y) for y in loss_history['validation']],label='validation')
    plt.legend(loc='best')
    plot_filename = os.path.join(output_dir, 'log-plot.png')
    plt.savefig(plot_filename)
    print('Saved plot at %s' % plot_filename)

@app.command()
def train2(output_dir='./output', best_model_file_name=None,
        checkpoint_file_name=None, musicnet_dir=None):
    config = {}
    exp = MusicNetExperiment(
            epoch=5,
            checkpoint_frequency=10,
            directory=output_dir,
            config=config
    )
    exp.run()

@app.command()
def evaluate(checkpoint_path, audio_file_path, output_dir='./output'):
    with open(checkpoint_path,'rb') as f:
        checkpoint = dill.load(f)
    net = checkpoint['best_model']
    #midi = to_midi(net,os.path.join(musicnet_dir,'test_data','1759.wav'))
    #midi = to_midi(net,os.path.join(musicnet_dir,'train_data','1727.wav'))
    #midi = data.musicnet.interval_tree_to_midi(val_dataset.labels[1759])
    convert = get_convert_fn(net)

    #input_wav_file_name = os.path.join(musicnet_dir,'test_data','1759.wav')
    input_wav_file_name = audio_file_path
    rate,data = scipy.io.wavfile.read(input_wav_file_name)
    #convert = get_convert_fn_ground_truth(data,val_dataset.labels[1759])

    # Output MIDI
    midi = to_midi2(convert, input_wav_file_name)
    midi_file_name = os.path.join(output_dir, 'output.mid')
    midi.write(midi_file_name)

    # Output synthesized MIDI as wav file
    wav = midi.fluidsynth()
    wav_file_name = os.path.join(output_dir, 'output.wav')
    scipy.io.wavfile.write(wav_file_name,rate=44100,data=wav)

if __name__=="__main__":
    app()

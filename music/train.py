import os
import copy
from tqdm import tqdm
import scipy.io.wavfile
import torch
import torchaudio
import pretty_midi

import data
import data.generator
import data.musicnet

class Net(torch.nn.Module):
    def __init__(self, structure=[201,200,128]):
        super().__init__()
        seq = []
        for in_size,out_size in zip(structure,structure[1:]):
            seq.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            seq.append(torch.nn.ReLU())
        seq = seq[:-1] + [torch.nn.Sigmoid()] # Remove last ReLU and replace with sigmoid
        self.seq = torch.nn.Sequential(*seq)
    def forward(self,x):
        return self.seq(x).squeeze()

def get_datasets():
    generated_transforms = data.generator.Compose([
        data.generator.ToNoteNumber(),
        data.generator.SynthesizeSounds(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Spectrogram(),
        data.generator.Filter(['note_numbers','spectrogram'])
    ])
    generated_dataset = data.generator.GeneratedDataset(transforms=generated_transforms)
    musicnet_transforms = data.generator.Compose([
        data.musicnet.IntervalsToNoteNumbers(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Spectrogram(),
        data.generator.Filter(['note_numbers','spectrogram'])
    ])
    musicnet_train_dataset = data.musicnet.MusicNetDataset(
            '/mnt/ppl-3/musicnet/musicnet',transforms=musicnet_transforms,
            train=True,points_per_song=1)
    musicnet_test_dataset = data.musicnet.MusicNetDataset(
            '/mnt/ppl-3/musicnet/musicnet',transforms=musicnet_transforms,
            train=False,points_per_song=100)
    return musicnet_train_dataset, musicnet_test_dataset

def save_checkpoint(file_name, model, optim):
    checkpoint = {
            'model': model.state_dict(),
            'optim': optim.state_dict()
    }
    torch.save(checkpoint,file_name)

def load_checkpoint(file_name, model, optim):
    try:
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
    except FileNotFoundError:
        print('Checkpoint not found. Skipping.')

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

if __name__=="__main__":
    output_dir = ''
    best_model_file_name = os.path.join(output_dir,'best_model.pkl')
    checkpoint_file_name = os.path.join(output_dir,'checkpoint.pkl')

    train_dataset, val_dataset = get_datasets()

    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    load_checkpoint(checkpoint_file_name,net,optimizer)

    train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=32, shuffle=True)

    best_model = copy.deepcopy(net)
    best_loss = float('inf')

    criterion = torch.nn.BCELoss(reduction='sum')
    for _ in range(10):
        total_train_loss = 0
        for d in tqdm(train_dataloader,desc='Training'):
            x = d['spectrogram']
            y = d['note_numbers']
            est = net(d['spectrogram'])
            loss = criterion(est,y)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_loss /= len(train_dataloader)

        total_val_loss = 0
        for d in tqdm(val_dataloader,desc='Validation'):
            x = d['spectrogram']
            y = d['note_numbers']
            est = net(d['spectrogram'])
            loss = criterion(est,y)
            total_val_loss += loss.item()
        total_val_loss /= len(val_dataloader)

        if total_val_loss < best_loss:
            print('New best model saved.')
            best_model = copy.deepcopy(net)
            best_loss = total_val_loss

        print('Train Loss: %f \t Val Loss: %f' % 
                (total_train_loss,total_val_loss))
        save_checkpoint(checkpoint_file_name,net,optimizer)

    output = to_midi(net,'/mnt/ppl-3/musicnet/musicnet/test_data/1759.wav')

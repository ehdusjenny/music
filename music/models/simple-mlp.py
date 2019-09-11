from models.base import BaseModel
import data.generator

import torch
import torch.nn.functional as F

def get_data_from_file(file_name):
    y, sr = librosa.load(file_name)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    d = librosa.core.stft(y_harmonic,n_fft=2000-2)

    x = np.vstack([np.abs(d),np.angle(d)])
    return x.transpose()

def get_data(n, c=None):
    data_x_wav = []
    data_y = []
    for num_notes in range(1,10):
        # Generate data set
        for x,y in data.generator.random_clips(n, num_notes=num_notes):
            data_x_wav.append(x)
            data_y.append(y)
        # Generate chromagrams
        inputs = []
        outputs = []
        for wav,y in zip(data_x_wav,data_y):
            d = librosa.core.stft(wav,n_fft=2000-2)
            xs = np.vstack([np.abs(d),np.angle(d)])
            if c is None:
                for i in range(xs.shape[1]):
                    inputs.append(xs[:,i])
                    outputs.append(y)
            else:
                for i in np.random.choice(range(xs.shape[1]), c):
                    inputs.append(xs[:,i])
                    outputs.append(y)

    inputs = torch.from_numpy(np.array(inputs)).float()
    outputs = torch.from_numpy(np.array(outputs).astype(float)).float()

    return inputs, outputs

class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 128)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class Model(BaseModel):
    def __init__(self):
        self.net = SimpleMLP()

    def to_midi(self, file_name):
        import pretty_midi
        import scipy.io.wavfile

        print("Loading data")
        x_np = get_data_from_file(file_name)

        print("Creating Variable")
        x_var = Variable(torch.from_numpy(x_np), requires_grad=False)

        print("Converting to piano roll")
        output = np.array([model(x).data.numpy() for x in tqdm(x_var)])
        output = output > 0.7

        print("Nearest neighbour")
        k = 1
        output_conv = np.zeros(output.shape)
        for i in range(-int(np.floor(k/2)),int(np.ceil(k/2))):
            output_conv += np.roll(output,i,0)
        output_conv *= 1/k
        output_conv = (output_conv >= 0.5) + 0
        output = output_conv

        print("Generating midi")
        def get_length(o,t,n):
            i = 0
            while t+i < o.shape[0]:
                if o[t+i][n]:
                    o[t+i][n] = False
                    i+=1
                    continue
                else:
                    return i
        output2 = np.zeros(output.shape)
        for t in tqdm(range(output.shape[0])):
            for note_number in range(128):
                if output[t][note_number]:
                    output2[t][note_number] = get_length(output,t,note_number)
        music = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=41)
        for t in tqdm(range(output.shape[0])):
            for note_number in range(128):
                if output2[t][note_number] > 0:
                    note = pretty_midi.Note(velocity=100, pitch=note_number,
                            start=t*1998/44100/2,
                            end=(t+output2[t][note_number])*1998/44100/2)
                    instrument.notes.append(note)
        music.instruments.append(instrument)

        print("Synthesizing")
        #audio_data = music.synthesize(fs=44100)

        print("Saving audio")
        #scipy.io.wavfile.write("f2.wav",44100, audio_data)
        music.write('f.mid')

        return output,output2

    def train(self):
        inputs, outputs = get_data(10)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        criterion = torch.nn.MSELoss()
        for i in tqdm(range(100)):
            for x,y in tqdm(zip(inputs,outputs), leave=False):
                x_var = Variable(x, requires_grad=False)
                y_var = Variable(y, requires_grad=False)
                pred_y = self.net(x_var)
                loss = criterion(pred_y,y_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self, inputs, outputs):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        criterion = torch.nn.MSELoss()
        score = []
        for i in tqdm(range(1)):
            for x,y in tqdm(zip(inputs,outputs), leave=False):
                x_var = Variable(x, requires_grad=False)
                y_var = Variable(y, requires_grad=False)
                pred_y = net(x_var)
                loss = criterion(pred_y,y_var)
                score.append(loss.data[0])
        return np.mean(score)

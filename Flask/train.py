import logging
import os
import numpy as np
import librosa
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.optim
import torch.nn

import model
import data

def get_data(n, c=None):
    data_x_wav = []
    data_y = []
    for num_notes in range(1,10):
        # Generate data set
        for x,y in data.random_clips(n, num_notes=num_notes):
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

def get_data_from_file(file_name):
    y, sr = librosa.load(file_name)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    d = librosa.core.stft(y_harmonic,n_fft=2000-2)

    x = np.vstack([np.abs(d),np.angle(d)])
    return x.transpose()

def to_midi(file_name, model):
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
    k = 1 # This doesn't seem to work for any other value of k
    output_conv = np.zeros(output.shape)
    for i in range(-int(np.floor(k/2)),int(np.ceil(k/2))):
        output_conv += np.roll(output,i)
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

def train(model, optimizer):
    inputs, outputs = get_data(100)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    criterion = torch.nn.MSELoss()
    for i in tqdm(range(100)):
        for x,y in tqdm(zip(inputs,outputs), leave=False):
            x_var = Variable(x, requires_grad=False)
            y_var = Variable(y, requires_grad=False)
            pred_y = net(x_var)
            loss = criterion(pred_y,y_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test(model):
    inputs, outputs = get_data(1,1)

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

def test_one(model):
    inputs, outputs = get_data(1,1)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    score = []
    for i in tqdm(range(1)):
        for x,y in tqdm(zip(inputs,outputs), leave=False):
            x_var = Variable(x, requires_grad=False)
            y_var = Variable(y, requires_grad=False)
            pred_y = net(x_var)
            loss = criterion(pred_y,y_var)
            score.append(loss.data[0])
            np.set_printoptions(suppress=True)
            print(np.stack([y.numpy(),pred_y.data.numpy()]).transpose())
            print(loss.data[0])
            print("")
            np.set_printoptions() # Reset to default

def test_all_randomly(model):
    inputs, outputs = get_data(5000,1)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    score = Variable(torch.zeros(128), requires_grad=False)
    notes = Variable(torch.zeros(128), requires_grad=False)
    for x,y in tqdm(zip(inputs,outputs)):
        x_var = Variable(x, requires_grad=False)
        y_var = Variable(y, requires_grad=False)
        pred_y = net(x_var)
        np.set_printoptions(suppress=True)
        score += (pred_y-y_var).pow(2)
        notes += y_var

    fig = plt.figure()
    plt.bar(range(128), score.data.numpy(), color="blue")
    plt.xlabel("Notes")
    plt.ylabel("Mean Squared Error")
    plt.savefig("fig.png")

    return score, notes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Music thing. Hello world!")
    parser.add_argument("--train",
            action="store_true")
    parser.add_argument("--test",
            action="store_true")
    parser.add_argument("--test-all",
            action="store_true")
    parser.add_argument("--model-file",
            type=str, default="torchmodel.pkl",
            help="Name of the file in which the neural network is stored.")
    parser.add_argument("--save-every-epoch",
            action="store_true")
    parser.add_argument("--epochs",
            type=int, default=1000,
            help="Number of training epochs to perform")
    args = parser.parse_args()

    model_file_name = args.model_file

    logging.info("Initializing neural network")
    net = model.Net()
    if os.path.isfile(model_file_name):
        net.load_state_dict(torch.load(model_file_name, map_location=lambda
            storage, loc: storage))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    logging.info("Cudafying")
    if torch.cuda.is_available():
        net.cuda()
    if args.train:
        for i in range(args.epochs):
            logging.info("Training")
            train(net, optimizer)
            logging.info("Testing")
            score = test(net)
            print("Score: %f" % score)
            logging.info("Saving model")
            if args.save_every_epoch:
                torch.save(net.state_dict(), model_file_name+"."+i)
            else:
                torch.save(net.state_dict(), model_file_name)
    elif args.test:
        test_one(net)
    elif args.test_all:
        score, notes = test_all_randomly(net)
    else:
        print("I do nothing")

    """
    http://subsynth.sourceforge.net/midinote2freq.html
    C0 = 261/2^5 = 8Hz
    C1 = 261/2^4 = 16.3Hz
    C2 = 261/2^3 = 32.6Hz
    """

from tqdm import tqdm
import numpy as np                                       # fast vectors and matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # plotting
from scipy.fftpack import fft

import librosa

from intervaltree import Interval,IntervalTree

import os
from time import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#%matplotlib inline

class ConvNet(torch.nn.Module):
    def __init__(self, window_size):
        super(ConvNet, self).__init__()
        self.conv = torch.nn.Conv1d(1,500,2048,stride=8)
        self.pool = torch.nn.AvgPool1d(16,stride=8)
        self.fc = torch.nn.Linear(223*500,128)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.view(-1,223*500))
        return x

def traincn(net, train_data, test_data, window_size=2048):
    global Xtest, Ytest
    Xtest, Ytest = split_input_output(test_data, input_dims=window_size, random=False)
    Xtest.cuda()
    Ytest.cuda()
    Xtest = Xtest.view(-1,1,window_size) #

    criterion = torch.nn.MSELoss()

    square_error = []
    average_precision = []

    lr = .0001
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    np.random.seed(999)
    start = time()
    print('iter\tsquare_loss\tavg_precision\ttime')
    for i in tqdm(range(250000)):
        if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
            Yhattestbase = net(Xtest)
            loss = criterion(Yhattestbase, Ytest)
            square_error.append(loss.data[0])
            
            yflat = Ytest.view(-1)
            yhatflat = Yhattestbase.view(-1)
            average_precision.append(average_precision_score(yflat.data.numpy(), yhatflat.data.numpy()))
            
            if i % 10000 == 0:
                end = time()
                print(i,'\t', round(square_error[-1],8),\
                        '\t', round(average_precision[-1],8),\
                        '\t', round(end-start,8))
                start = time()
        
        Xmb, Ymb = split_input_output(train_data, input_dims=window_size, random=True)
        Xmb.cuda()
        Ymb.cuda()
        Xmb = Xmb.view(-1,1,window_size) #
        loss = criterion(net(Xmb), Ymb)
        loss.backward()
        opt.step()

def plot_weights(net):
    window = 2048
    f, ax = plt.subplots(20,2, sharey=False)
    f.set_figheight(20)
    f.set_figwidth(20)
    for i in range(20):
        #ax[i,0].plot(w.eval(session=sess)[:,i], color=(41/255.,104/255.,168/255.))
        ax[i,0].plot(net.fc1.weight[i].data.numpy(), color=(41/255.,104/255.,168/255.))
        ax[i,0].set_xlim([0,d])
        ax[i,0].set_xticklabels([])
        ax[i,0].set_yticklabels([])
        #ax[i,1].plot(np.abs(fft(w.eval(session=sess)[:,0+i]))[0:200], color=(41/255.,104/255.,168/255.))
        ax[i,1].plot(np.abs(fft(net.fc1.weight[i].data.numpy()))[0:200], color=(41/255.,104/255.,168/255.))
        ax[i,1].set_xticklabels([])
        ax[i,1].set_yticklabels([])
        
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    plt.savefig("weights.png")

def get_data_from_file(file_name):
    y, sr = librosa.load(file_name)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_harmonic

def predict_labels(net, file_name, window_size=2048, hop_length=512):
    data = get_data_from_file(file_name)
    n = int((data.shape[0]-window_size)/hop_length)
    global x
    x = np.array([data[(hop_length*i):(hop_length*i+window_size)] for i in range(n)])
    return net(Variable(torch.from_numpy(x), requires_grad=False))

def boolify(data):
    def f(x):
        return 1 if x > 0.3 else 0
    f = np.vectorize(f)
    return f(data)

def make_midi(labels, sample_rate, hop_length):
    """
    labels - A 2D np.array of booleans with indices [time][note_number]
    """
    import pretty_midi

    def get_length(l,t,n):
        i = 0
        while t+i < l.shape[0]:
            if l[t+i][n]:
                l[t+i][n] = False
                i+=1
                continue
            else:
                return i

    labels2 = np.zeros(labels.shape)
    for t in tqdm(range(labels.shape[0])):
        for note_number in range(128):
            if labels[t][note_number]:
                labels2[t][note_number] = get_length(labels,t,note_number)

    music = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=41)
    for t in tqdm(range(labels2.shape[0])):
        for note_number in range(128):
            if labels2[t][note_number] > 0:
                note = pretty_midi.Note(velocity=100, pitch=note_number,
                        start=t*1998/44100/2,
                        end=(t+labels2[t][note_number])*1998/44100/2)
                instrument.notes.append(note)
    music.instruments.append(instrument)

    return music

#d = 2048        # input dimensions
#m = 128         # number of notes
#k = 500         # number of hidden units

#if os.path.isfile("mlp2.pkl"):
#    net = torch.load("mlp2.pkl")
#else:
#    train_data, test_data = load_data('/NOBACKUP/hhuang63/musicnet/musicnet.npz')
#    net = Net2(d,k,m)
#    train(net, train_data, test_data)
#    torch.save(net, "mlp2.pkl")

if os.path.isfile("mlp.conv.pkl"):
    net = torch.load("mlp.conv.pkl")
else:
    train_data, test_data = load_data('/NOBACKUP/hhuang63/musicnet/musicnet.npz')
    net = ConvNet(16384).cuda()
    traincn(net, train_data, test_data, window_size=16384)
    torch.save(net, "mlp.conv.pkl")

#plot_weights(net)

print("Labelling")
labels = predict_labels(net, "bach.mp3")
labels = labels.data.numpy()
labels = boolify(labels)
print("Making MIDI")
midi = make_midi(labels, 44100, 512)
print("Saving file")
midi.write('f.mid')

#import dill
##train_data, test_data = load_data('/NOBACKUP/hhuang63/musicnet/musicnet.npz')
#x,y = dill.load(open("test-16384-small-np.pkl", "rb"))
#x = Variable(torch.from_numpy(x))
#y = Variable(torch.from_numpy(y))
#print("Done Loading")
#cn = ConvNet(16384)
#print(cn(x[0].view(1,1,-1)))
#print(cn(x[:2].view(-1,1,16384)))

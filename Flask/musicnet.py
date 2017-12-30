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
class Net2(torch.nn.Module):
    def __init__(self, d,k,m):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(d,k)
        self.fc2 = torch.nn.Linear(k,m)
    def forward(self, x):
        x = torch.log(1+F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, d,k,m):
        super(Net2, self).__init__()
        self.fc1 = torch.nn.Linear(d,k)
        self.fc2 = torch.nn.Linear(k,m)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = torch.nn.Conv1d(1,500,2048,stride=8)
        self.pool = torch.nn.AvgPool1d(16,stride=8)
        self.fc = torch.nn.Linear(223*500,128)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.view(-1))
        return x

def load_data(file_name):
    train_data = dict(np.load(open(file_name,'rb'),encoding='latin1'))

    # split our the test set
    test_data = dict()
    for id in (2303,2382,1819): # test set
        test_data[str(id)] = train_data.pop(str(id))

    return train_data, test_data

def split_input_output(data, input_dims=2048, sampling_rate=44100, stride=512, random=False):
    # TODO: Does not currently work for different stride lengths
    # TODO: Compute the number of data points using data['2382'][0].shape[0]
    features = 0    # first element of (X,Y) data tuple
    labels = 1      # second element of (X,Y) data tuple

    ids = list(data.keys())
    if random:
        x = np.empty([len(data),input_dims])
        y = np.zeros([len(data),m])
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
        y = np.zeros([len(data)*7500,m])
        for i in range(len(ids)):
            for j in range(7500):
                index = sampling_rate+j*stride # start from one second to give us some wiggle room for larger segments
                x[7500*i + j] = data[ids[i]][features][index:index+input_dims]
                
                # label stuff that's on in the center of the window
                for label in data[ids[i]][labels][index+input_dims/2]:
                    y[7500*i + j,label.data[1]] = 1

    x = Variable(torch.from_numpy(x).float(), requires_grad=False)
    y = Variable(torch.from_numpy(y).long(), requires_grad=False)

    return x,y

def train(net, train_data, test_data):
    Xtest, Ytest = split_input_output(test_data, input_dims=d, random=False)

    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.MultiLabelMarginLoss()

    square_error = []
    average_precision = []

    lr = .0001
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    np.random.seed(999)
    start = time()
    print('iter\tsquare_loss\tavg_precision\ttime')
    for i in tqdm(range(250000)):
        if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
            loss = criterion(net(Xtest), Ytest)
            square_error.append(loss.data[0])
            
            Yhattestbase = net(Xtest)
            yflat = Ytest.view(-1)
            yhatflat = Yhattestbase.view(-1)
            average_precision.append(average_precision_score(yflat.data.numpy(), yhatflat.data.numpy()))
            
            if i % 10000 == 0:
                end = time()
                print(i,'\t', round(square_error[-1],8),\
                        '\t', round(average_precision[-1],8),\
                        '\t', round(end-start,8))
                start = time()
        
        Xmb, Ymb = split_input_output(train_data, input_dims=d, random=True)
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

d = 2048        # input dimensions
m = 128         # number of notes
k = 500         # number of hidden units

if os.path.isfile("mlp.pkl"):
    net = torch.load("mlp.pkl")
else:
    train_data, test_data = load_data('/NOBACKUP/hhuang63/musicnet/musicnet.npz')
    net = Net(d,k,m)
    train(net, train_data, test_data)
    torch.save(net, "mlp.pkl")

plot_weights(net)


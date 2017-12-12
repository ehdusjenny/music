import numpy as np                                       # fast vectors and matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # plotting
from scipy.fftpack import fft

from intervaltree import Interval,IntervalTree

import os
from time import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#%matplotlib inline
class Net(torch.nn.Module):
    def __init__(self, d,k,m):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(d,k)
        self.fc2 = torch.nn.Linear(k,m)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

d = 2048        # input dimensions
m = 128         # number of notes
fs = 44100      # samples/second
features = 0    # first element of (X,Y) data tuple
labels = 1      # second element of (X,Y) data tuple

if os.path.isfile("mlp.pkl"):
    net = torch.load("mlp.pkl")
else:
    train_data = dict(np.load(open('/NOBACKUP/hhuang63/musicnet/musicnet.npz','rb'),encoding='latin1'))

    # split our the test set
    test_data = dict()
    for id in (2303,2382,1819): # test set
        test_data[str(id)] = train_data.pop(str(id))
        
    train_ids = list(train_data.keys())
    test_ids = list(test_data.keys())
        
    print(len(train_data))
    print(len(test_data))

    # create the test set
    Xtest = np.empty([3*7500,d])
    Ytest = np.zeros([3*7500,m])
    for i in range(len(test_ids)):
        for j in range(7500):
            index = fs+j*512 # start from one second to give us some wiggle room for larger segments
            Xtest[7500*i + j] = test_data[test_ids[i]][features][index:index+d]
            
            # label stuff that's on in the center of the window
            for label in test_data[test_ids[i]][labels][index+d/2]:
                Ytest[7500*i + j,label.data[1]] = 1

    Xtest = Variable(torch.from_numpy(Xtest).float(), requires_grad=False)
    Ytest = Variable(torch.from_numpy(Ytest).float(), requires_grad=False)

    # MLP
    k = 500

    wscale = .001


    net = Net(d,k,m)
    criterion = torch.nn.MSELoss()

    square_error = []
    average_precision = []

    lr = .0001
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    np.random.seed(999)
    start = time()
    print('iter\tsquare_loss\tavg_precision\ttime')
    for i in range(250000):
        if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
            #square_error.append(sess.run(L, feed_dict={x: Xtest, y_: Ytest})/Xtest.shape[0])
            loss = criterion(net(Xtest), Ytest)
            square_error.append(loss.data[0])
            
            #Yhattestbase = sess.run(y,feed_dict={x: Xtest})
            Yhattestbase = net(Xtest)
            #yflat = Ytest.reshape(Ytest.shape[0]*Ytest.shape[1])
            yflat = Ytest.view(-1)
            #yhatflat = Yhattestbase.reshape(Yhattestbase.shape[0]*Yhattestbase.shape[1])
            yhatflat = Yhattestbase.view(-1)
            average_precision.append(average_precision_score(yflat.data.numpy(), yhatflat.data.numpy()))
            
            if i % 10000 == 0:
                end = time()
                print(i,'\t', round(square_error[-1],8),\
                        '\t', round(average_precision[-1],8),\
                        '\t', round(end-start,8))
                start = time()
        
        Ymb = np.zeros([len(train_data),m])
        Xmb = np.empty([len(train_data),d])
        for j in range(len(train_ids)):
            s = np.random.randint(d/2,len(train_data[train_ids[j]][features])-d/2)
            Xmb[j] = train_data[train_ids[j]][features][int(s-d/2):int(s+d/2)]
            for label in train_data[train_ids[j]][labels][s]:
                Ymb[j,label.data[1]] = 1
        Ymb = Variable(torch.from_numpy(Ymb).float(), requires_grad=False)
        Xmb = Variable(torch.from_numpy(Xmb).float(), requires_grad=False)
        
        #sess.run(train_step, feed_dict={x: Xmb, y_: Ymb})
        loss = criterion(net(Xmb), Ymb)
        loss.backward()
        opt.step()

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

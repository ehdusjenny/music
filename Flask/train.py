import logging
import os
import numpy as np
import librosa
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim
import torch.nn

import model
import data

def get_data(n, c=None):
    dg = data.random_clips(n, num_notes=10)

    data_x_wav = []
    data_y = []
    # Generate data set
    for x,y in dg:
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Music thing. Hello world!")
    parser.add_argument("--train",
            action="store_true")
    parser.add_argument("--test",
            action="store_true")
    parser.add_argument("--model-file",
            type=str, default="torchmodel.pkl",
            help="Name of the file in which the neural network is stored.")
    parser.add_argument("--epochs",
            type=int, default=100,
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
        for _ in range(args.epochs):
            logging.info("Training")
            train(net, optimizer)
            logging.info("Testing")
            score = test(net)
            print("Score: %f" % score)
            logging.info("Saving model")
            torch.save(net.state_dict(), model_file_name)
    elif args.test:
        test_one(net)

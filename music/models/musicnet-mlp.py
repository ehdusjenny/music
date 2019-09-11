import config
import data.musicnet

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Net2(torch.nn.Module):
    def __init__(self, d,k,m):
        super(Net2, self).__init__()
        self.fc1 = torch.nn.Linear(d,k)
        self.fc2 = torch.nn.Linear(k,m)
    def forward(self, x):
        x = torch.log(1+F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, d,k,m):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(d,k)
        self.fc2 = torch.nn.Linear(k,m)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(BaseModel):
    def __init__(self):
        self.net = SimpleMLP()
        self.data = None

    def to_midi(self, file_name):
        pass
    
    def train(self):
        if self.data is None:
            self.data = data.musicnet.load_data()
        train_ids = list(self.data.keys())
        test_ids = [str(i) for i in config.MUSICNET_TEST]
        for i in test_ids:
            train_ids.remove(i)
        Xtest, Ytest = data.musicnet.get_test_set(self.data, test_ids,
                input_dims=2048)

        criterion = torch.nn.MSELoss()

        square_error = []
        average_precision = []

        lr = .0001
        opt = torch.optim.SGD(net.parameters(), lr=lr)
        np.random.seed(999)
        start = time()
        print('iter\tsquare_loss\ttime')
        for i in tqdm(range(250000)):
            if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
                loss = self.test(Xtest, Ytest)
                square_error.append(loss)
                if i % 10000 == 0:
                    end = time()
                    print(i,'\t', round(square_error[-1],8),\
                            '\t', round(end-start,8))
                    start = time()
            
            Xmb, Ymb = data.musicnet.get_random_batch(self.data, train_ids,
                    input_dims=2048)
            loss = criterion(net(Xmb), Ymb)
            loss.backward()
            opt.step()

    def test(self, inputs, outputs):
        # TODO: check if inputs are pytorch Variables or something else
        # For now, I assume they're pytorch Variables

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        criterion = torch.nn.MSELoss()
        pred_y = net(inputs)
        loss = criterion(pred_y,outputs)
        return loss

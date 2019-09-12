import torch
from tqdm import tqdm

import data
import data.generator

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

if __name__=="__main__":
    transforms = data.generator.Compose([
        data.generator.ToNoteNumber(),
        data.generator.SynthesizeSounds(),
        data.generator.NoteNumbersToVector(),
        data.generator.ToTensor(),
        data.generator.Spectrogram(),
        data.generator.Filter(['note_numbers','spectrogram'])
    ])
    dataset = data.generator.GeneratedDataset(transforms=transforms)
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32,
            shuffle=True)

    criterion = torch.nn.BCELoss(reduction='sum')
    for _ in range(10):
        total_loss = 0
        for d in tqdm(dataloader):
            x = d['spectrogram']
            y = d['note_numbers']
            est = net(d['spectrogram'])
            loss = criterion(est,y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= len(dataloader)
        print('Loss: %f' % (total_loss))

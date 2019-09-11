import config
import numpy as np

def load_data(mem=False):
    file_name = config.MUSICNET_FILE
    if mem:
        return dict(np.load(open(file_name,'rb'),encoding='latin1'))
    else:
        return dict(np.load(open(file_name,'rb'),encoding='latin1',mmap_mode="r"))

def split_input_output(data, input_dims=2048, sampling_rate=44100, stride=512, random=False):
    # TODO: Does not currently work for different stride lengths
    # TODO: Compute the number of data points using data['2382'][0].shape[0]
    features = 0    # first element of (X,Y) data tuple
    labels = 1      # second element of (X,Y) data tuple

    ids = list(data.keys())
    if random:
        x = np.empty([len(data),input_dims])
        y = np.zeros([len(data),128])
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
        y = np.zeros([len(data)*7500,128])
        for i in range(len(ids)):
            for j in range(7500):
                index = sampling_rate+j*stride # start from one second to give us some wiggle room for larger segments
                x[7500*i + j] = data[ids[i]][features][index:index+input_dims]
                
                # label stuff that's on in the center of the window
                for label in data[ids[i]][labels][index+input_dims/2]:
                    y[7500*i + j,label.data[1]] = 1

    x = Variable(torch.from_numpy(x).float(), requires_grad=False)
    y = Variable(torch.from_numpy(y).float(), requires_grad=False)

    return x,y

def get_random_batch(data, ids, input_dims=2048, sampling_rate=44100):
    x = np.empty([len(data),input_dims])
    y = np.zeros([len(data),128])
    for i in range(len(ids)):
        # Pick a random spot in the audio track
        s = np.random.randint(
                input_dims/2,
                len(data.get(ids[i]).get("data"))-input_dims/2)
        x[i] = data.get(ids[i]).get("data")[int(s-input_dims/2):int(s+input_dims/2)]
        y[i] = np.unpackbits(data.get(ids[i]).get("labels")[s])
    x = Variable(torch.from_numpy(x).float(), requires_grad=False).cuda()
    y = Variable(torch.from_numpy(y).float(), requires_grad=False).cuda()
    return x,y

def get_test_set(data, ids, input_dims=2048, sampling_rate=44100):
    stride = 512*8
    x = np.empty([len(ids)*750,input_dims])
    y = np.zeros([len(ids)*750,128])
    for i in range(len(ids)):
        for j in tqdm(range(750)):
            index = sampling_rate+j*stride # start from one second to give us some wiggle room for larger segments
            x[750*i + j] = data.get(ids[i]).get("data")[index:index+input_dims]
            y[750*i + j] = np.unpackbits(data.get(ids[i]).get("labels")[index+input_dims/2])
    x = Variable(torch.from_numpy(x).float(), requires_grad=False,
            volatile=True).cuda()
    y = Variable(torch.from_numpy(y).float(), requires_grad=False,
            volatile=True).cuda()
    return x,y

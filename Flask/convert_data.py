import numpy as np
import h5py
from tqdm import tqdm
import os

input_file = "/NOBACKUP/hhuang63/musicnet/musicnet.h5"
output_file = "/NOBACKUP/hhuang63/musicnet/musicnet2.h5"

fin = h5py.File(input_file,"r")
fout = h5py.File(output_file,"w")

def parse_intervals(intervals, length):
    output = np.zeros([length,128],dtype=np.int)
    for interval in intervals:
        start_time = interval[6]
        end_time = interval[1]
        note = interval[4]
        output[start_time:end_time,note]=1
    output = np.packbits(output, axis=1)
    return output

if os.path.isfile(output_file):
    print("File already exists. Nothing to do.")
else:
    for k in tqdm(list(fin.keys())):
        group = fout.create_group(str(k))
        group.create_dataset("data", data=fin.get(k).get("data"))
        labels = parse_intervals(
                fin.get(k).get("labels"),
                fin.get(k).get("data").shape[0])
        group.create_dataset("labels", data=labels)


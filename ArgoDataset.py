from glob import glob
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, random_split

ROOT_PATH = "./argo2/"

cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc", "palo-alto"]
splits = ["train", "validation", "test"]

def get_city_trajectories(city="palo-alto", split="train", normalized=False, vels=False, accs=False):
    if split=="train" or split=="validation":
        if city == "all":
            inputs = None
            outputs = None
            for c in cities:
                f_in = ROOT_PATH + "train" + "/" + c + "_inputs"
                i = pickle.load(open(f_in, "rb"))
                i = np.asarray(i)
                inputs = i if inputs is None else np.concatenate((inputs,i))
                f_out = ROOT_PATH + "train" + "/" + c + "_outputs"
                o = pickle.load(open(f_out, "rb"))
                o = np.asarray(o)
                outputs = o if outputs is None else np.concatenate((outputs,o))

        else:
            f_in = ROOT_PATH + "train" + "/" + city + "_inputs"
            inputs = pickle.load(open(f_in, "rb"))
            inputs = np.asarray(inputs)
            f_out = ROOT_PATH + "train" + "/" + city + "_outputs"
            outputs = pickle.load(open(f_out, "rb"))
            outputs = np.asarray(outputs)
        
#         if split == "train":
#             inputs = inputs[:inputs.shape[0]*4//5]
#             outputs = outputs[:outputs.shape[0]*4//5]
#         else:
#             inputs = inputs[inputs.shape[0]*4//5:]
#             outputs = outputs[outputs.shape[0]*4//5:]
        
        if vels:
            inp_vels, out_vels = get_deltas(inputs, outputs)
            inputs = np.concatenate((inputs, inp_vels), axis=2)
            outputs = np.concatenate((outputs, out_vels), axis=2)
            if accs:
                inp_accs, out_accs = get_deltas(inp_vels, out_vels)
                inputs = np.concatenate((inputs, inp_accs), axis=2)
                outputs = np.concatenate((outputs, out_accs), axis=2)         
            
    else:
        f_in = ROOT_PATH + "test" + "/" + city + "_inputs"
        inputs = pickle.load(open(f_in, "rb"))
        inputs = np.asarray(inputs)
        outputs = None
        
        if vels:
            inp_vels = np.zeros(shape=(inputs.shape))
            for idx in range(len(inputs)):
                for i in range(inputs.shape[1]-1):
                    vel = inputs[idx,i+1, :2]-inputs[idx,i, :2]
                    inp_vels[idx,i] = vel
                inp_vels[idx,-1] = inp_vels[idx,-2]
            inputs = np.concatenate((inputs, inp_vels), axis=2) 
            if accs:
                inp_accs = np.zeros(shape=(inp_vels.shape))
                for idx in range(len(inp_vels)):
                    for i in range(inp_vels.shape[1]-1):
                        acc = inp_vels[idx,i+1, :2]-inp_vels[idx,i, :2]
                        inp_accs[idx,i] = acc
                    inp_accs[idx,-1] = inp_accs[idx,-2]
                inputs = np.concatenate((inputs, inp_accs), axis=2) 
    return inputs, outputs

class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""
    def __init__(self, city: str, split:str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.transform = transform 
        self.inputs, self.outputs = get_city_trajectories(city=city, split=split, normalized=False, vels=True, accs=True)
        print(self.inputs.shape)
        for idx in range(len(self.inputs)):
            start = np.zeros(shape=(2))
            start[:2] = self.inputs[idx, 0, :2]
            self.inputs[idx, :, :2] = self.inputs[idx, :, :2]-start
            if split != "test":
                self.outputs[idx, :, :2] = self.outputs[idx, :, :2]-start

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):           
        #normalize start pos
        data = (self.inputs[idx], self.outputs[idx])
        if self.transform:
            data = self.transform(data)
        

        return data
    
    def split(self, split_ratio=0.2):
        valid_size = round(split_ratio * len(self.inputs))
        train_size = len(self.inputs) - valid_size

        return random_split(self, [train_size, valid_size])

def get_deltas(inputs, outputs):
    inp_delta = np.zeros(shape=(inputs.shape))
    out_delta = np.zeros(shape=(outputs.shape))

    for idx in range(len(inputs)):
        for i in range(inputs.shape[1]-1):
            vel = inputs[idx,i+1, :2]-inputs[idx,i, :2]
            inp_delta[idx,i] = vel
        inp_delta[idx,-1] = outputs[idx,0, :2]-inputs[idx,-1, :2]
        for o in range(outputs.shape[1]-1):
            vel = outputs[idx,o+1, :2]-outputs[idx,o, :2]
            out_delta[idx,o] = vel
        out_delta[idx,-1] = out_delta[idx,-2]
    return inp_delta, out_delta


def show_sample_batch(sample_batch):
    """visualize the trajectory for a batch of samples"""
    inp, out = sample_batch
    batch_sz = inp.size(0)
    agent_sz = inp.size(1)
    
    fig, axs = plt.subplots(1,batch_sz, figsize=(15, 3), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()   
    for i in range(batch_sz):
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])
        
        # first two feature dimensions are (x,y) positions
        axs[i].scatter(inp[i,:,0], inp[i,:,1])
        axs[i].scatter(out[i,:,0], out[i,:,1])
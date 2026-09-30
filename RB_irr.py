import os
import sys
import torch
import numpy as np
import scipy.io
import h5py
import argparse
from random import SystemRandom

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.utils2D import *
from lib.functions2D import DeepONet1, GRU_unit, Decoder
from lib.ProposeModel2D import *
from lib.BaselineModel2D import *

##### Generative model for noisy data based on ODE
def args(): 
    parser = argparse.ArgumentParser('Latent Neural Operator')
    parser.add_argument('--niters', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
    parser.add_argument('--dataset', type=str, default='NS')
    
    parser.add_argument('--input_dimx', type=int, default=128, help="Dimensionality of the input system.")
    parser.add_argument('--input_dimy', type=int, default=128, help="Dimensionality of the input system.")
    #parser.add_argument('--rec_dims', type=int, default=100, help="Dimensionality of the recognition model (NO or RNN).")
    #parser.add_argument('--latents', type=int, default=169, help="Size of the latent state")
    #parser.add_argument('--gru_units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
    parser.add_argument('--rec_dims', type=int, default=100, help="Dimensionality of the recognition model (NO or RNN).")
    parser.add_argument('--latents', type=int, default=169, help="Size of the latent state")
    parser.add_argument('--gru_units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
    
    parser.add_argument('--rec_len', type=int, default=8, help="The length of observation data")
    parser.add_argument('--n_traj_samples', type=int, default=3, help="The number of trajectory samples")
    parser.add_argument('--noise_weight', type=float, default=0.0, help="Noise amplitude for generated traejctories")
    parser.add_argument('--obs_ratio', type=float, default=1.0, help="Observation ratio for generated traejctories")
    
    args = parser.parse_args(args=[])
    return(args)
    
args = args()
    
#experimentID = int(SystemRandom().random()*100000)
experimentID = 900
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = 'LNO'
ckpt_path = os.path.join("experiments/experiment_" + str(experimentID) + '.ckpt')


class MyDataset(Dataset):
    def __init__(self, s_data, t_data, u_data):
        self.s_data = s_data
        self.t_data = t_data
        self.u_data = u_data
    
    def __len__(self):
        return len(self.s_data)
    
    def __getitem__(self, idx):
        sample_s = self.s_data[idx]
        sample_t = self.t_data[idx]
        sample_u = self.u_data[idx]
        return sample_s, sample_t, sample_u
    
# read dataset
def read_data():
    Lx, Lz, Nt, t_red = 4, 1, 100, 2
    Nx, Nz, n_red = 128, 128, 1 
    nnt = 20
    stop_sim_time = 50
    xs = torch.linspace(0, Lx, Nx+1)[:-1][::n_red]
    ys = torch.linspace(0, Lz, Nz+1)[:-1][::n_red]

    '''
    rs = np.arange(40)
    for i in tqdm(range(len(rs))):
        #reader = np.load("data/dataset/rayleigh_benard/rayleigh_benard_" + str(rs[i]) + ".npz")
        reader = np.load("/home/featurize/work/latent-deeponet-main/code/data/rayleigh_benard/rayleigh_benard_" + str(rs[i]) + ".npz")
        outp = torch.tensor(reader['outputs'])[:, ::t_red, ::n_red, ::n_red][:,:nnt]
        if i == 0:
            Xs = outp
            ts = torch.linspace(0, stop_sim_time, Nt+1)[:-1][None,:].repeat(Xs.shape[0],1)[:,::t_red][:,:nnt]
        else:
            Xs = torch.cat((Xs, outp), axis=0)
    ts = ts.repeat(len(rs),1)
    '''
    rs = np.arange(40)
    for i in tqdm(range(len(rs))):
        reader = np.load("/home/featurize/work/latent-deeponet-main/code/data2/rayleigh_benard/rayleigh_benard_" + str(rs[i]) + ".npz")
        data = torch.tensor(reader['data'])[:, ::t_red, ::n_red, ::n_red][:,:nnt]
        if i == 0:
            Xs = data
            ts = torch.linspace(0, stop_sim_time, Nt+1)[:-1][None,:][:,::t_red][:,:nnt]
        else:
            Xs = torch.cat((Xs, data), axis=0)
    ts = ts.repeat(Xs.shape[0],1)
            
    noise = (np.random.sample(Xs.shape) - 0.5) * 2.0
    #noise = torch.tensor(np.random.randn(Xs.shape)).float()
    Xs_noise = Xs + args.noise_weight * noise
    
    n = Xs.shape[0]
    T = ts.shape[-1]
    s_GT = Xs[:n].float()
    s = Xs_noise[:n]
    ts = ts[:n]

    us = torch.zeros((Xs.shape[0],Xs.shape[2],Xs.shape[3])) 
    us = us[:n]
    
    observed_points = int(T*args.obs_ratio)
    s_mask = torch.zeros(n, observed_points, len(xs), len(ys))
    t_mask = torch.zeros(n, observed_points)
    s_GT_mask = torch.zeros(n, observed_points, len(xs), len(ys))
    mask = np.zeros((n, T), dtype=int)
    for i in range(n):
        observed_indices = np.random.choice(np.arange(1,T), observed_points-1, replace=False)
        sorted_indices = np.sort(observed_indices)
        mask[i, 0] = 1
        mask[i, observed_indices] = 1
        s_mask[i,0] = s[i,0]
        s_mask[i,1:] = s[i,sorted_indices]
        s_GT_mask[i,0] = s_GT[i,0]
        s_GT_mask[i,1:] = s_GT[i,sorted_indices]
        t_mask[i,0] = ts[i,0]
        t_mask[i,1:] = ts[i,sorted_indices]
    mask = torch.tensor(mask)    
    
    Ntr = int(n * 0.9)
    
    s_data_tr = s_mask[:Ntr]; t_data_tr = t_mask[:Ntr]; u_data_tr = us[:Ntr] 
    s_data_te = s_mask[Ntr:]; t_data_te = t_mask[Ntr:]; u_data_te = us[Ntr:] 
    test_GT_mask = s_GT_mask[Ntr:]; test_GT = s_GT[Ntr:]
    
    print("s.shape:", s.shape)
    print("s_data_tr.shape:", s_data_tr.shape)
    print("t_data_tr.shape:", t_data_tr.shape)
    print("u_data_tr.shape:", u_data_tr.shape)
    print("Ntr:", Ntr)
    
    return(s_data_tr, t_data_tr, u_data_tr, s_data_te, t_data_te, u_data_te, xs, ys, ts, mask, test_GT_mask, test_GT) 


s_data_tr, t_data_tr, u_data_tr, s_data_te, t_data_te, u_data_te, xs, ys, ts, mask, test_GT_mask, test_GT = read_data()
args.input_dimx = len(xs)
args.input_dimy = len(ys)

Ntr = s_data_tr.shape[0]
Nte = s_data_te.shape[0]
batch_size = min(args.batch_size, Ntr)

my_dataset = MyDataset(s_data_tr.to(device), t_data_tr.to(device), u_data_tr.to(device))
train_dataloader = DataLoader(my_dataset, batch_size = batch_size, shuffle=True)
train_dataloader_iter = iter(train_dataloader)
test_dataset_mask = [s_data_te.to(device), t_data_te.to(device), u_data_te.to(device)]
test_dataset = [test_GT.to(device), ts[Ntr:].to(device), u_data_te.to(device)]

## Training
if experimentID == 900:
    model = RLNO(args, device)
elif experimentID == 901:
    model = DeepONet(args, device) 
elif experimentID == 902:
    model = GRUVAE(args, device)
elif experimentID == 903:
    model = GRUDecay(args, device)
elif experimentID == 904:
    model = MLAE_LD(args, device, s_data_tr, if_train_vae = True)
elif experimentID == 905:
    model = LNODE(args, device)
elif experimentID == 906:
    model = Base_PDENET(args, device, xs)
elif experimentID == 907:
    model = RLNO_MI(args, device, u_data_tr.shape[-1])
elif experimentID == 908:
    model = MIONet(args, device, u_data_tr.shape[-1])
elif experimentID == 909:
    model = RLNO_cn(args, device)

log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
if not os.path.exists("logs/"):
    utils.makedirs("logs/")
logger = get_logger(logpath=log_path, filepath='work/LNO_/LNO_1.0.ipynb')

optimizer = optim.Adamax(model.parameters(), lr = args.lr)

num_batches = len(train_dataloader)

for itr in range(1, num_batches * (args.niters + 1)):
    optimizer.zero_grad()
    update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)
    wait_until_kl_inc = 10
    if itr // num_batches < wait_until_kl_inc:
        kl_coef = 0.
    else:
        kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
    
    try:
        batch = next(train_dataloader_iter)
    except StopIteration:
        train_dataloader_iter = iter(train_dataloader)
        batch = next(train_dataloader_iter)
        
    train_res, pred_y = model.compute_all_losses(batch, kl_coef = kl_coef)
    train_res["loss"].backward()
    optimizer.step()
    
    n_iters_to_viz = 1
    if itr % (n_iters_to_viz * num_batches) == 0:
        with torch.no_grad():
            model.TestInfo(experimentID, test_dataset_mask, train_res, itr, num_batches, kl_coef, logger)

        torch.save({'args': args,'state_dict': model.state_dict(),}, ckpt_path)

torch.save({'args': args,'state_dict': model.state_dict(),}, ckpt_path)

## testing
batch_mask = test_dataset_mask
batch = test_dataset
pred_y = model.test(batch).cpu()
pred_y_mask = model.test(batch_mask).cpu()
print("Pred:", pred_y.shape, pred_y_mask.shape)

index = 0
s = batch[0][index].cpu()
s_GT = test_GT[index].cpu()
#s_mask = s * mask[index].unsqueeze(-1).repeat(1,len(xs)).t()
if len(pred_y.shape) == 4:
    print("error:", torch.mean((torch.mean(pred_y,axis=0).reshape(test_GT.shape) - test_GT)**2))
    print("error_before:", torch.mean((torch.mean(pred_y,axis=0).reshape(test_GT.shape) - test_GT)[:args.rec_len]**2))
    print("error_later:", torch.mean((torch.mean(pred_y,axis=0).reshape(test_GT.shape) - test_GT)[args.rec_len:]**2))
    print("error_mask:", torch.mean((torch.mean(pred_y_mask,axis=0).reshape(test_GT_mask.shape) - test_GT_mask)**2))
    sp = torch.mean(pred_y,axis=0).reshape(test_GT.shape)[index].cpu().detach().numpy()
else:
    print("error:", torch.mean((pred_y.reshape(test_GT.shape) - test_GT)**2))
    print("error_before:", torch.mean((pred_y.reshape(test_GT.shape) - test_GT)[:args.rec_len]**2))
    print("error_later:", torch.mean((pred_y.reshape(test_GT.shape) - test_GT)[args.rec_len:]**2))
    print("error_mask:", torch.mean((pred_y_mask.reshape(test_GT_mask.shape) - test_GT_mask)**2))
    sp = pred_y.reshape(test_GT.shape)[index].cpu().detach().numpy()

fig = plt.figure(figsize=(15,5))
num = 10; l = 2
for i in range(num):   
    ax = fig.add_subplot(4,num,i+1)
    data = pd.DataFrame(s[i*l,:,:])
    data.replace(0, np.nan, inplace=True)
    cmap = sns.heatmap(data,center=s.mean())  
    
    ax = fig.add_subplot(4,num,num+i+1)
    data = pd.DataFrame(s_GT[i*l,:,:])
    data.replace(0, np.nan, inplace=True)
    cmap = sns.heatmap(data,center=s.mean())  
    
    ax = fig.add_subplot(4,num,2*num+i+1)
    data = pd.DataFrame(sp[i*l,:,:])
    data.replace(0, np.nan, inplace=True)
    cmap = sns.heatmap(data,center=s.mean()) 
    
    ax = fig.add_subplot(4,num,3*num+i+1)
    data = pd.DataFrame((s_GT - sp)[i*l,:,:])
    data.replace(0, np.nan, inplace=True)
    cmap = sns.heatmap(data,center=s.mean()) 
    
plt.savefig("results/{}.png".format(experimentID))

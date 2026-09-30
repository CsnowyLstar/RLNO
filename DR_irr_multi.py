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

from lib.utils import *
from lib.functions import DeepONet1, GRU_unit, Decoder
from lib.ProposeModel import *
from lib.BaselineModel import *


##### Generative model for noisy data based on ODE
def args(): 
    parser = argparse.ArgumentParser('Latent Neural Operator')
    parser.add_argument('--niters', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
    parser.add_argument('--dataset', type=str, default='DR')
    parser.add_argument('--data_version', type=str, default='dr', help="Select from [dr, drV2]")

    parser.add_argument('--input_dim', type=int, default=64, help="Dimensionality of the input system.")
    parser.add_argument('--rec_dims', type=int, default=64, help="Dimensionality of the recognition model (NO or RNN).")
    parser.add_argument('--latents', type=int, default=64, help="Size of the latent state")
    parser.add_argument('--gru_units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
    parser.add_argument('--rec_len', type=int, default=10, help="The length of observation data")
    parser.add_argument('--n_traj_samples', type=int, default=3, help="The number of trajectory samples")
    parser.add_argument('--noise_weight', type=float, default=0.6, help="Noise amplitude for generated traejctories")
    parser.add_argument('--obs_ratio', type=float, default=0.2, help="Noise amplitude for generated traejctories")
    
    args = parser.parse_args(args=[])
    return(args)

args = args()

#experimentID = int(SystemRandom().random()*100000)
experimentID = 611
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
    reader = MatReader("data/dataset/data_" + args.data_version + ".mat")
    Xs = reader.read_field('Xs').permute(0,2,1)
    ts = reader.read_field('ts').repeat(Xs.shape[0],1)
    xs = reader.read_field('xs')[0]
    try:
        us = reader.read_field('us')
    except:
        us = torch.zeros((Xs.shape[0],Xs.shape[2])) 
    noise = (torch.tensor(np.random.sample(Xs.shape)).float() - 0.5) * 2.0
    #noise = torch.tensor(np.random.randn(*Xs.shape)).float()
    Xs_noise = Xs + args.noise_weight * noise
    
    n = 5000
    T = 100
    s_GT = Xs[:n,:T,:]
    s = Xs_noise[:n,:T,:]
    us = us[:n]
    ts = ts[:n,:T]
    
    observed_points = int(T*args.obs_ratio)
    s_mask = torch.zeros(n, observed_points, len(xs))
    t_mask = torch.zeros(n, observed_points)
    s_GT_mask = torch.zeros(n, observed_points, len(xs))
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
    
    return(s_data_tr, t_data_tr, u_data_tr, s_data_te, t_data_te, u_data_te, xs, ts, mask, test_GT_mask, test_GT)

s_data_tr, t_data_tr, u_data_tr, s_data_te, t_data_te, u_data_te, xs, ts, mask, test_GT_mask, test_GT = read_data()
args.input_dim = len(xs)

Ntr = s_data_tr.shape[0]
Nte = s_data_te.shape[0]
batch_size = min(args.batch_size, Ntr)

my_dataset = MyDataset(s_data_tr.to(device), t_data_tr.to(device), u_data_tr.to(device))
train_dataloader = DataLoader(my_dataset, batch_size = batch_size, shuffle=True)
train_dataloader_iter = iter(train_dataloader)
test_dataset_mask = [s_data_te.to(device), t_data_te.to(device), u_data_te.to(device)]
test_dataset = [test_GT.to(device), ts[Ntr:].to(device), u_data_te.to(device)]

## Training
if experimentID == 600:
    model = RLNO(args, device)
elif experimentID == 601:
    model = DeepONet(args, device) 
elif experimentID == 602:
    model = GRUVAE(args, device)
elif experimentID == 603:
    model = GRUDecay(args, device)
elif experimentID == 604:
    model = MLAE_LD(args, device, s_data_tr, if_train_vae = True)
elif experimentID == 605:
    model = LNODE(args, device)
elif experimentID == 606:
    model = Base_PDENET(args, device, xs)
elif experimentID == 607:
    model = RLNO_MI(args, device, u_data_tr.shape[-1])
elif experimentID == 608:
    model = MIONet(args, device, u_data_tr.shape[-1])
elif experimentID == 609:
    model = LNO_Ab1(args, device)
elif experimentID == 610:
    model = LNO_Ab2(args, device)
elif experimentID == 611:
    model = LNO_Ab3(args, device)
elif experimentID == 612:
    model = LNO_Ab4(args, device)

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

index = 0
s = batch[0][index].t().cpu()
s_GT = test_GT[index].t().cpu()
s_mask = s * mask[index].unsqueeze(-1).repeat(1,len(xs)).t()
if len(pred_y.shape) == 4:
    print("error:", torch.mean((torch.mean(pred_y,axis=0)-test_GT)**2))
    #print("error_before:", torch.mean((torch.mean(pred_y,axis=0)-test_GT)[:args.rec_len]**2))
    #print("error_later:", torch.mean((torch.mean(pred_y,axis=0)-test_GT)[args.rec_len:]**2))
    print("error_mask:", torch.mean((torch.mean(pred_y_mask,axis=0)-test_GT_mask)**2))
    sp = pred_y[0,index].t().cpu().detach().numpy()
else:
    print("error:", torch.mean((pred_y-test_GT)**2))
    #print("error_before:", torch.mean((pred_y-test_GT)[:args.rec_len]**2))
    #print("error_later:", torch.mean((pred_y-test_GT)[args.rec_len:]**2))
    print("error_mask:", torch.mean((pred_y_mask-test_GT_mask)**2))
    sp = pred_y[index].t().cpu().detach().numpy()

fig = plt.figure(figsize=(30,5))
ax0 = fig.add_subplot(1,4,1)
data = pd.DataFrame(s_mask)
data.replace(0, np.nan, inplace=True)
cmap = sns.heatmap(data,center=s.mean())

ax1 = fig.add_subplot(1,4,2)
data = pd.DataFrame(s_GT)
cmap = sns.heatmap(data,center=s.mean())

ax2 = fig.add_subplot(1,4,3)
data = pd.DataFrame(sp)
cmap = sns.heatmap(data,center=s.mean())

ax3 = fig.add_subplot(1,4,4)
data = pd.DataFrame(sp-s_GT.numpy())
cmap = sns.heatmap(data)

plt.savefig("results/{}.png".format(experimentID))





import os
import sys
import numpy as np
import scipy.io
import argparse
from random import SystemRandom
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.distributions import kl_divergence
import torch.nn.init as init

import torchdiffeq as ode

from lib.functions2D import *
from lib.utils2D import *

################################################################
###   Key baselines                                          ###
################################################################

################################################################
###   Classic DeepONet                                       ###
################################################################
class DeepONet(nn.Module):
    def __init__(self, args, device):
        super(DeepONet, self).__init__()
        
        self.device = device
        self.args = args
        self.input_dimx = args.input_dimx
        self.input_dimy = args.input_dimy
        self.operator_solver = DeepONet1(input_size1=args.input_dimx*args.input_dimy, input_size2=1, hidden_size=100, p=8, device=device).to(device)
        #self.operator_solver = DeepONet_2D(input_size1=args.input_dimx*args.input_dimy, input_size2=1, hidden_size=64, p=4, device=device).to(device)
        
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args

        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        ## Operator for reconstruction and prediction
        y1 = s_batch[:,0,:,:].reshape(n_traj,-1).unsqueeze(-2).unsqueeze(0).repeat(1,1,n_tp,1)
        y2 = t_batch.unsqueeze(0).unsqueeze(-1)
        
        pred_y = self.operator_solver(y1,y2)
        mse = torch.mean((s_batch.reshape(n_traj,n_tp,-1) - pred_y)**2)
        
        results = {}
        results["loss"] = torch.mean(mse)
        
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)

        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test loss: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)


class MIONet(nn.Module):
    def __init__(self, args, device, input_size2):
        super(MIONet, self).__init__()
        
        self.device = device
        self.args = args
        self.input_dimx = args.input_dimx
        self.input_dimy = args.input_dimy
        self.operator_solver = DeepONet_MI(input_size1=args.input_dimx*args.input_dimy, input_size2=input_size2**2, input_size3=1, hidden_size=64, p=16, device=device).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args

        s_batch = batch[0]
        t_batch = batch[1]
        u_batch = batch[2]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        ## Operator for reconstruction and prediction
        y1 = s_batch[:,0,:,:].reshape(n_traj,-1).unsqueeze(-2).repeat(1,n_tp,1)
        y2 = u_batch.reshape(n_traj,-1).unsqueeze(-2).repeat(1,n_tp,1)
        y3 = t_batch.unsqueeze(-1)
        
        pred_y = self.operator_solver(y1,y2,y3)
        mse = (s_batch.reshape(n_traj,n_tp,-1) - pred_y)**2
        
        results = {}
        results["loss"] = torch.mean(mse)
        
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)

        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test loss: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)


################################################################
###   GRUVAE                                                 ###
################################################################        
class GRUVAE(nn.Module):
    def __init__(self, args, device):
        super(GRUVAE, self).__init__()
        
        self.device = device
        self.args = args
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents; self.gru_units = gru_units
        
        
        self.w_input_decay =  Parameter(torch.Tensor(1, int(input_dimx*input_dimy))).to(device)
        self.b_input_decay =  Parameter(torch.Tensor(1, int(input_dimx*input_dimy))).to(device)
        self.rnn_cell_enc = GRUCell(2*input_dimx*input_dimy + 1, rec_dims).to(device)
        self.rnn_cell_dec = GRUCell(2*input_dimx*input_dimy + 1, latents).to(device)
        
        self.z0_net = nn.Sequential(nn.Linear(rec_dims, 100), nn.Tanh(), nn.Linear(100, latents*2),).to(device)
        init_network_weights(self.z0_net)
        self.decoder = nn.Sequential(nn.Linear(latents, 100), nn.Tanh(), nn.Linear(100, input_dimx*input_dimy),).to(device)
        init_network_weights(self.decoder)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        
        ## RNN-decay for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp, :, :]
        time_steps = batch[1][:, :encoder_ntp]
        
        idx = [i for i in range(data.size(1)-1, -1, -1)]
        data = data[:, idx]
        delta_ts = time_steps[:,1:] - time_steps[:,:-1]
        idx = [i for i in range(delta_ts.size(1)-1, -1, -1)]
        delta_ts = delta_ts[:, idx]
        zero_delta_t = torch.zeros(delta_ts.shape[0],1).to(device)
        delta_ts = torch.cat((delta_ts, zero_delta_t), axis=1).unsqueeze(-1)
        
        input_decay_params = (self.w_input_decay, self.b_input_decay)
        hidden_state, _ = run_rnn(data.reshape(n_traj,encoder_ntp,-1), delta_ts, cell = self.rnn_cell_enc, input_decay_params = input_decay_params)
        assert(not torch.isnan(hidden_state).any())
        
        z0_mean, z0_std = split_last_dim(self.z0_net(hidden_state))
        z0_std = z0_std.abs()
        z0_sample = sample_standard_gaussian(z0_mean, z0_std)
        
        # Decoder # # # # # # # # # # # # # # # # # # # #
        delta_ts = torch.cat((zero_delta_t, t_batch[:, 1:] - t_batch[:, :-1]), axis=1).unsqueeze(-1)
        
        _, all_hiddens = run_rnn(s_batch.reshape(n_traj,n_tp,-1), delta_ts, cell = self.rnn_cell_dec,
                                first_hidden = z0_sample, feed_previous = True, 
                                n_steps = t_batch.size(1),
                                decoder = self.decoder,
                                input_decay_params = input_decay_params)
        outputs = self.decoder(all_hiddens)
        # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
        first_point = data[:,0,:]
        outputs = shift_outputs(outputs, first_point.reshape(n_traj,-1))

        extra_info = {"first_point": (z0_mean.unsqueeze(0), z0_std.unsqueeze(0), z0_sample.unsqueeze(0))}
        
        pred_y, info = outputs, extra_info
        
        #get_reconstruction done -- computing likelihood
        obsrv_std = torch.Tensor([0.01]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_std[fp_std==0] = 1e-6
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, z0_prior)
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))
        
        # Compute likelihood of all the points
        rec_likelihood = get_gaussian_likelihood(s_batch.reshape(n_traj,n_tp,-1), pred_y, obsrv_std)
        mse = get_mse(s_batch.reshape(n_traj,n_tp,-1), pred_y)
        pois_log_likelihood = torch.Tensor([0.]).to(device) ###

        ################################
        # Compute CE loss for binary classification on Physionet
        ce_loss = torch.Tensor([0.]).to(device) ###

        # IWAE loss
        kl_coef = 0.1
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            itr//num_batches, 
            test_res["loss"].detach(), test_res["likelihood"].detach(), 
            test_res["kl_first_p"], test_res["std_first_p"])
        
        logger.info("Experiment " + str(experimentID))
        logger.info(message)
        logger.info("KL coef: {}".format(kl_coef))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

        if "mse" in test_res:
            logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

        if "pois_likelihood" in test_res:
            logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

        if "ce_loss" in test_res:
            logger.info("CE loss: {}".format(test_res["ce_loss"]))
    
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)


################################################################
###   GRUDecay                                               ###
################################################################     
class GRUDecay(nn.Module):
    def __init__(self, args, device):
        super(GRUDecay, self).__init__()
        
        self.device = device
        self.args = args
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents; self.gru_units = gru_units
        
        
        self.w_input_decay =  Parameter(torch.Tensor(1, int(input_dimx*input_dimy))).to(device)
        self.b_input_decay =  Parameter(torch.Tensor(1, int(input_dimx*input_dimy))).to(device)
        self.rnn_cell_enc = GRUCellExpDecay(
                            input_size = 2*input_dimx*input_dimy,
                            input_size_for_decay = input_dimx*input_dimy,
                            hidden_size = rec_dims, 
                            device = device).to(device)
        self.rnn_cell_dec = GRUCellExpDecay(
                            input_size = 2*input_dimx*input_dimy,
                            input_size_for_decay = input_dimx*input_dimy,
                            hidden_size = latents, 
                            device = device).to(device)
        
        self.z0_net = nn.Sequential(nn.Linear(rec_dims, 100), nn.Tanh(), nn.Linear(100, latents*2),).to(device)
        init_network_weights(self.z0_net)
        self.decoder = nn.Sequential(nn.Linear(latents, 100), nn.Tanh(), nn.Linear(100, input_dimx*input_dimy),).to(device)
        init_network_weights(self.decoder)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        
        ## RNN-decay for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp]
        time_steps = batch[1][:, :encoder_ntp]
        
        idx = [i for i in range(data.size(1)-1, -1, -1)]
        data = data[:, idx]
        delta_ts = time_steps[:,1:] - time_steps[:,:-1]
        idx = [i for i in range(delta_ts.size(1)-1, -1, -1)]
        delta_ts = delta_ts[:, idx]
        zero_delta_t = torch.zeros(delta_ts.shape[0],1).to(device)
        delta_ts = torch.cat((delta_ts, zero_delta_t), axis=1).unsqueeze(-1)
        
        input_decay_params = (self.w_input_decay, self.b_input_decay)
        hidden_state, _ = run_rnn(data.reshape(n_traj,encoder_ntp,-1), delta_ts, cell = self.rnn_cell_enc, input_decay_params = input_decay_params)
        assert(not torch.isnan(hidden_state).any())
        
        z0_mean, z0_std = split_last_dim(self.z0_net(hidden_state))
        z0_std = z0_std.abs()
        z0_sample = sample_standard_gaussian(z0_mean, z0_std)
        
        # Decoder # # # # # # # # # # # # # # # # # # # #
        delta_ts = torch.cat((zero_delta_t, t_batch[:, 1:] - t_batch[:, :-1]), axis=1).unsqueeze(-1)
        
        _, all_hiddens = run_rnn(s_batch.reshape(n_traj,n_tp,-1), delta_ts, cell = self.rnn_cell_dec,
                                first_hidden = z0_sample, feed_previous = True, 
                                n_steps = t_batch.size(1),
                                decoder = self.decoder,
                                input_decay_params = input_decay_params)
        outputs = self.decoder(all_hiddens)
        # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
        first_point = data[:,0,:]
        outputs = shift_outputs(outputs, first_point.reshape(n_traj,-1))

        extra_info = {"first_point": (z0_mean.unsqueeze(0), z0_std.unsqueeze(0), z0_sample.unsqueeze(0))}
        
        pred_y, info = outputs, extra_info
        
        #get_reconstruction done -- computing likelihood
        obsrv_std = torch.Tensor([0.01]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_std[fp_std==0] = 1e-6
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, z0_prior)
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))
        
        # Compute likelihood of all the points
        rec_likelihood = get_gaussian_likelihood(s_batch.reshape(n_traj,n_tp,-1), pred_y, obsrv_std)
        mse = get_mse(s_batch.reshape(n_traj,n_tp,-1), pred_y)
        pois_log_likelihood = torch.Tensor([0.]).to(device) ###
        
        ################################
        # Compute CE loss for binary classification on Physionet
        ce_loss = torch.Tensor([0.]).to(device) ###

        # IWAE loss
        kl_coef = 0.1
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            itr//num_batches, 
            test_res["loss"].detach(), test_res["likelihood"].detach(), 
            test_res["kl_first_p"], test_res["std_first_p"])
        
        logger.info("Experiment " + str(experimentID))
        logger.info(message)
        logger.info("KL coef: {}".format(kl_coef))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

        if "mse" in test_res:
            logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

        if "pois_likelihood" in test_res:
            logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

        if "ce_loss" in test_res:
            logger.info("CE loss: {}".format(test_res["ce_loss"]))
    
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)

################################################################
###   MLAE_LD                                                ###
################################################################   
class MLP_VAE(nn.Module):
    def __init__(self, args, device):
        super(MLP_VAE, self).__init__()
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        latents = args.latents
        self.encoder = MLAE_enc(input_dimx*input_dimy, latents).to(device)
        self.decoder = MLAE_dec(latents, input_dimx*input_dimy).to(device)
        
    def forward(self, batch):
        data = batch
        latent_state = self.encoder(data)
        pred_state = self.decoder(latent_state)
        mse = torch.mean((pred_state - data)**2)      
        return(mse)
        

class MLAE_LD(nn.Module):
    def __init__(self, args, device, train_dataset = None, if_train_vae = False):
        super(MLAE_LD, self).__init__()
        
        self.device = device
        self.args = args        
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        latents = args.latents
        self.latents = latents
        
        self.encoder = MLAE_enc(input_dimx*input_dimy, latents).to(device)
        self.decoder = MLAE_dec(latents, input_dimx*input_dimy).to(device)
        self.if_train_vae = if_train_vae
        
        if self.if_train_vae:
            self.train_dataset = train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[1],-1)
            self.pre_train()
            self.if_train_vae = False
        else:
            self.encoder.load_state_dict(torch.load("experiments/"+args.dataset+"_LD_encoder.pth"))
            self.decoder.load_state_dict(torch.load("experiments/"+args.dataset+"_LD_decoder.pth"))
            self.encoder.requires_grad_(False); self.decoder.requires_grad_(False)
        
        #self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.operator_solver = DeepONet_2D(input_size1=latents, input_size2=1, hidden_size=64, p=4, device=device).to(device)
    
    def pre_train(self):
        device = self.device
        args = self.args
        
        train_dataset = self.train_dataset
        train_dataloader = DataLoader(train_dataset.to(device), batch_size = args.batch_size, shuffle=True)
        mlpvae = MLP_VAE(args,device).to(device)
        optimizer = optim.Adamax(mlpvae.parameters(), lr = args.lr)
        
        for epoch in range(args.niters):
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                #update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 100)
                loss = mlpvae(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, MLP_VAE Loss: {total_loss / len(train_dataloader)}")
        
        for param in mlpvae.parameters():
            param.requires_grad = False
            
        self.encoder = mlpvae.encoder
        self.decoder = mlpvae.decoder
        torch.save(mlpvae.encoder.state_dict(), "experiments/"+args.dataset+"_LD_encoder.pth")
        torch.save(mlpvae.decoder.state_dict(), "experiments/"+args.dataset+"_LD_decoder.pth")
        self.encoder.requires_grad_(False); self.decoder.requires_grad_(False)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        latents = self.latents
        
        ## MLE for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        data = batch[0][:, 0]
        latent_data = self.encoder(data.reshape(n_traj,-1))
        
        ## Operator for reconstruction and prediction
        y1 = latent_data.unsqueeze(-2).unsqueeze(0).repeat(1,1,n_tp,1) 
        y2 = t_batch.unsqueeze(0).unsqueeze(-1)
        sol_y = self.operator_solver(y1,y2)
        
        pred_y = self.decoder(sol_y)
        mse = (s_batch.reshape(n_traj,n_tp,-1) - pred_y)**2
        
        results = {}
        results["loss"] = torch.mean(mse)
            
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test MSE: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)       


################################################################
###   LNODE                                                  ###
################################################################          
class LNODE(nn.Module):
    
    def __init__(self, args, device):
        super(LNODE, self).__init__()
        self.device = device
        self.args = args
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents; self.gru_units = gru_units
            
        ode_func_net = create_net(rec_dims, rec_dims, n_layers = 1, n_units = 100, nonlinear = nn.Tanh)
        rec_ode_func = ODEFunc(input_dim = input_dimx+input_dimy, latent_dim = rec_dims, ode_func_net = ode_func_net, device = device).to(device)
        self.z0_diffeq_solver = DiffeqSolver(input_dimx+input_dimy, rec_ode_func, "euler", odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
        ode_func_net = create_net(latents, latents, n_layers = 1, n_units = 100, nonlinear = nn.Tanh)
        gen_ode_func = ODEFunc(input_dim = input_dimx, latent_dim = args.latents, ode_func_net = ode_func_net, device = device).to(device)
        #self.diffeq_solver = DiffeqSolver(input_dim, gen_ode_func, 'dopri5', odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        self.diffeq_solver = DiffeqSolver(input_dimx, gen_ode_func, 'euler', odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims*2, 100), nn.Tanh(), nn.Linear(100, latents*2),).to(device)
        init_network_weights(self.transform_z0)
        
        self.GRU_update = GRU_unit(rec_dims, input_dimx, n_units=gru_units, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
        
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        latents = self.latents
        gru_units = self.gru_units
        n_traj_samples = args.n_traj_samples

        ## ODE-based RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0].reshape(n_traj,-1).unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]
            minimum_step = min(time_steps[:,-1] - time_steps[:,0]) / 50
            
            latent_ys = []
            time_points_iter = reversed(range(0, encoder_ntp))
            
            for i in time_points_iter:
                yi_ode = torch.zeros(1, n_traj, args.rec_dims).to(device)
                for j in range(len(prev_t)):
                    if (prev_t[j] - t_i[j]) < minimum_step:
                        time_points = torch.stack((prev_t[j], t_i[j]))
                        inc = self.z0_diffeq_solver.ode_func(prev_t[j], prev_y[:,j:j+1]) * (t_i[j] - prev_t[j])
                        assert(not torch.isnan(inc).any())
                        ode_sol = prev_y[:,j:j+1] + inc
                        ode_sol = torch.stack((prev_y[:,j:j+1], ode_sol), 2).to(device)
                        assert(not torch.isnan(ode_sol).any())
                    else:
                        n_intermediate_tp = max(2, ((prev_t[j] - t_i[j]) / minimum_step).int())
                        time_points = linspace_vector(prev_t[j], t_i[j], n_intermediate_tp).to(device)
                        ode_sol = self.z0_diffeq_solver(prev_y[:,j:j+1], time_points)
                        assert(not torch.isnan(ode_sol).any())
                    yi_ode[:,j:j+1] = ode_sol[:, :, -1, :]
                
                xi = data[:,i].reshape(n_traj,-1).unsqueeze(0)
                yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
                prev_y, prev_std = yi, yi_std
                prev_t, t_i = time_steps[:,i],  time_steps[:,i-1]
                latent_ys.append(yi)
            
            latent_ys = torch.stack(latent_ys, 1)
            last_yi, last_yi_std = yi, yi_std            
        
        means_z0 = last_yi.reshape(1, n_traj, rec_dims)
        std_z0 = last_yi_std.reshape(1, n_traj, rec_dims)
        mean_z0, std_z0 = split_last_dim(self.transform_z0(torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        first_point_mu, first_point_std = mean_z0, std_z0
        
        ## ODE for reconstruction
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        first_point_std = first_point_std.abs()
        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0
        
        pred_y = torch.zeros(n_traj_samples, n_traj, n_tp, n_dimsx*n_dimsy).to(device)
        for j in range(n_traj):
            sol_y = self.diffeq_solver(first_point_enc_aug[:,j:j+1], t_batch[j])
            predj = self.decoder(sol_y)
            pred_y[:,j:j+1] = predj
        '''
        ministep = 0.01
        max_t = torch.max(t_batch)
        time_range = (torch.arange(max_t//ministep + 1) * ministep).to(device)
        sol_y = self.diffeq_solver(first_point_enc_aug, time_range)
        pind = (t_batch//ministep).unsqueeze(0).unsqueeze(-1).expand(sol_y.size(0),-1,-1,sol_y.size(3))
        sol_yz = torch.gather(sol_y, 2, pind.to(torch.int64))
        pred_y = self.decoder(sol_yz)
        '''
        
        info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}
        
        
        #get_reconstruction done -- computing likelihood
        obsrv_std = torch.Tensor([0.01]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_std[fp_std==0] = 1e-6
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, z0_prior)
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))
        
        # Compute likelihood of all the points
        rec_likelihood = get_gaussian_likelihood(s_batch.reshape(n_traj, n_tp, -1), pred_y, obsrv_std)
        mse = get_mse(s_batch.reshape(n_traj, n_tp, -1), pred_y)
        pois_log_likelihood = torch.Tensor([0.]).to(device) ###

        ################################
        # Compute CE loss for binary classification on Physionet
        ce_loss = torch.Tensor([0.]).to(device) ###

        # IWAE loss
        kl_coef = 0.1
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            itr//num_batches, 
            test_res["loss"].detach(), test_res["likelihood"].detach(), 
            test_res["kl_first_p"], test_res["std_first_p"])
        
        logger.info("Experiment " + str(experimentID))
        logger.info(message)
        logger.info("KL coef: {}".format(kl_coef))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

        if "mse" in test_res:
            logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

        if "pois_likelihood" in test_res:
            logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

        if "ce_loss" in test_res:
            logger.info("CE loss: {}".format(test_res["ce_loss"]))
    
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)    
    




































################################################################
###   Classic DeepONet (version 2)                           ###
################################################################    
class Vanilla_DeepONet2(nn.Module):
    def __init__(self, args, device, xs):
        super(Vanilla_DeepONet2, self).__init__()
    
        self.device = device
        self.args = args
        self.xs = xs.to(device)
        self.operator_solver = DeepONet2(input_size1=args.input_dimx, input_size2=1+len(xs.shape), hidden_size=64, p=32, device=device).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        xs = self.xs
        
        data_to_predict = batch[..., :-1]
        time_steps_to_predict = batch[..., -1]
        n_traj, n_tp, n_dims = data_to_predict.size()
        
        ## Operator for reconstruction and prediction
        y1 = data_to_predict[:,0:1,:].unsqueeze(-2).repeat(1,n_tp,n_dims,1)
        y2x = xs.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(n_traj,n_tp,1,1)        
        y2t = time_steps_to_predict.unsqueeze(-1).unsqueeze(-1).repeat(1,1,n_dims,1)
        y2 = torch.cat((y2x,y2t),axis=-1)
        
        pred_y = self.operator_solver(y1,y2)
        mse = (data_to_predict - pred_y)**2
        
        results = {}
        results["loss"] = torch.mean(mse)
        
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)

        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test loss: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y)    

        
################################################################
###   MLAE                                                   ###
################################################################     
class Base_MLAE(nn.Module):
    def __init__(self, args, device):
        super(Base_MLAE, self).__init__()
        
        self.device = device
        self.args = args
        input_dim = args.input_dim
        latents = args.latents
        self.input_dim = input_dim; self.latents = latents

        #self.encoder = MLAE_enc(input_dim, latents).to(device)
        self.encoder = MLAE_enc(input_dim**2*args.rec_len, latents).to(device)
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = MLAE_dec(latents, input_dim**2).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        latents = self.latents
        
        ## MLE for encoder 
        data_to_predict = batch[..., :-1]
        time_steps_to_predict = batch[:, :, 0, -1]
        n_traj, n_tp, n_dimsx, n_dimsy = data_to_predict.size()
        
        #data = batch[:, 0, :-1]
        #time_steps = batch[:, 0, -1]
        data = batch[:, :args.rec_len, :, :-1]
        latent_data = self.encoder(data.reshape(n_traj,-1))
        
        ## Operator for reconstruction and prediction
        y1 = latent_data.unsqueeze(-2).repeat(1,n_tp,1)
        y2 = time_steps_to_predict.unsqueeze(-1)
        sol_y = self.operator_solver(y1,y2)
        
        pred_y = self.decoder(sol_y)
        mse = (data_to_predict.reshape(n_traj, n_tp, -1) - pred_y)**2
        
        results = {}
        results["loss"] = torch.mean(mse)
            
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test MSE: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        with torch.no_grad():
            test_res, pred_y = self.compute_all_losses(batch)
            
            return(pred_y) 
        

################################################################
###   PDE-NET                                                ###
################################################################     
class Sepi(nn.Module): 
    def __init__(self, device, lent, dx):
        super(Sepi, self).__init__()
        self.device = device
        self.lent = lent
        self.dx = dx
        self.conv1d = torch.nn.functional.conv1d
        
        kks = [torch.tensor([1]).float().to(device)]
        kks.append(torch.tensor([-1,1]).float().to(device))
        kks.append(torch.tensor([1,-2,1]).float().to(device))
        kks.append(torch.tensor([-1,3,-3,1]).float().to(device))
        kks.append(torch.tensor([1,-4,6,-4,1]).float().to(device))
        self.kks = kks

        
        '''
        q = 4     
        self.q = q
        bn = q+1

        self.line1 = nn.Linear(bn, 2, bias=True).to(device)
        self.line2 = nn.Linear(bn+1, 2, bias=True).to(device)
        self.line3 = nn.Linear(bn+2, 1, bias=True).to(device)
        init.zeros_(self.line1.weight)
        init.zeros_(self.line2.weight)
        init.zeros_(self.line3.weight)
        init.zeros_(self.line1.bias)
        init.zeros_(self.line2.bias)
        init.zeros_(self.line3.bias)
        '''
        
        q = len(kks)
        self.q = q
        layers = 1
        self.layers = layers
        self.lines = nn.ModuleList().to(device)
        for i in range(layers):
            linear = nn.Linear(q+i, 2, bias=False)
            torch.nn.init.uniform_(linear.weight, a=-0.001, b=0.001)
            self.lines.append(linear)
        
        lineout = nn.Linear(q+i+1, 1, bias=True).to(device)
        torch.nn.init.uniform_(lineout.weight, a=-0.001, b=0.001)
        self.lineout = lineout
        
        
    def partial(self, o, order):
        kks = self.kks; dx = self.dx; conv1d = self.conv1d; lent = self.lent
        oh = torch.hstack((o[:,lent-order:lent],o))
        m = conv1d(oh.unsqueeze(1), kks[order].view(1,1,-1))[:,0]
        return(m/(dx**order))
    
    def forward(self, t, x):
        partial = self.partial
        lent = self.lent
        q = self.q
        device = self.device
        
        
        xp = torch.zeros(x.shape[0],lent,q).to(device)
        for i in range(q):
            xp[...,i] = partial(x,i)    
            
        for i in range(self.layers):
            xi_eta = self.lines[i](xp)
            fi_new = (xi_eta[...,0] * xi_eta[...,1]).unsqueeze(-1)
            xp = torch.cat((xp,fi_new), axis=-1)
            
        FF = self.lineout(xp).squeeze(-1) 
        
        '''
        fi = torch.zeros(x.shape[0],q+1,lent).to(device)
        for i in range(q+1):
            fi[:,i] = partial(x,i) 
        
        xi_eta = self.line1(fi.permute(0,2,1))
        fi_new = (xi_eta[...,0] * xi_eta[...,1]).unsqueeze(1)
        fi2 = torch.cat((fi,fi_new), axis=1)
        
        xi_eta2 = self.line2(fi2.permute(0,2,1))
        fi_new2 = (xi_eta2[...,0] * xi_eta2[...,1]).unsqueeze(1)
        fi3 = torch.cat((fi2,fi_new2), axis=1)
        
        FF = self.line3(fi3.permute(0,2,1)).squeeze(-1)
        '''
        
        #FF = 0.01 * xp[...,2] + 0.01*xp[...,0]**2
        return(FF)
        
class Base_PDENET(nn.Module):
    def __init__(self, args, device, xs):
        super(Base_PDENET, self).__init__()
        
        self.device = device
        self.args = args        
        input_dim = args.input_dim
        latents = args.latents
        self.latents = latents
        
        dx = xs[-1]/(len(xs)-1)
            
        self.sepi = Sepi(device, input_dim, dx).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        latents = self.latents
        
        n_traj, n_tp, n_dims = batch[...,:-1].size()
        
        le = 2
        i0 = random.randint(0, n_tp-le)
        data = batch[:, i0, :-1]
        data_to_predict = batch[:, i0:i0+le, :-1]
        time_steps_to_predict = batch[:, i0:i0+le, -1]
        
        pred_y = torch.zeros(n_traj, le, n_dims).float().to(device)
        for i in range(n_traj):
            y0 = data[i:i+1]
            tt = time_steps_to_predict[i] - time_steps_to_predict[i,0]
            pred = ode.odeint(self.sepi, y0, tt, rtol=1e-4, atol=1e-6, method='euler').permute(1,0,2)
            #pred = ode.odeint(self.sepi, y0, tt, rtol=1e-4, atol=1e-6, method='euler', options={'step_size': 0.5}).permute(1,0,2)
            pred_y[i:i+1] = pred
        
        '''
        ministep = 0.01
        max_t = torch.max(time_steps_to_predict)
        time_range = (torch.arange(max_t//ministep + 1) * ministep).to(device)
        #sol_y = ode.odeint(self.sepi, data, time_range, rtol=1e-4, atol=1e-6, method="euler").permute(1,0,2)
        sol_y = torch.zeros(n_traj, len(time_range), n_dims).float().to(device)
        sol_y[:,0] = data
        for tri in range(len(time_range)-1):
            data = self.sepi.forward(time_range[tri], data) * ministep + data
            sol_y[:,tri+1] = data
        pind = (time_steps_to_predict//ministep).unsqueeze(-1).expand(-1,-1,sol_y.size(2))
        pred_y = torch.gather(sol_y, 1, pind.to(torch.int64))
        '''    
        
        mse = (data_to_predict - pred_y)**2
        
        loss = torch.mean(mse)
        for param in self.sepi.parameters():
            loss += 1e-6 * torch.norm(param, 1)
        
        results = {}
        results["loss"] = loss
            
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test MSE: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        device = self.device
        args = self.args
        latents = self.latents
        
        with torch.no_grad():
            #test_res, pred_y = self.compute_all_losses(batch)
            n_traj, n_tp, n_dims = batch[...,:-1].size()
            data = batch[:, 0, :-1]
            data_to_predict = batch[:, :, :-1]
            time_steps_to_predict = batch[:, :, -1]

            pred_y = torch.zeros(n_traj, n_tp, n_dims).float().to(device)
            for i in range(n_traj):
                y0 = data[i:i+1]
                tt = time_steps_to_predict[i]
                pred = ode.odeint(self.sepi, y0, tt, rtol=1e-4, atol=1e-6, method='euler').permute(1,0,2)
                pred_y[i:i+1] = pred            
            
        return(pred_y)        
        


################################################################
###   MP-PDE                                                 ###
################################################################     
class Base_MPPDE(nn.Module):
    def __init__(self, args, device, xs):
        super(Base_MPPDE, self).__init__()
        
        self.device = device
        self.args = args        
        input_dim = args.input_dim
        latents = args.latents
        self.latents = latents
        
        dx = xs[-1]/(len(xs)-1)
        layers = 2
        
        self.sepi = Sepi(device, input_dim, layers, dx).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        latents = self.latents
        
        n_traj, n_tp, n_dims = batch[...,:-1].size()
        
        i0 = random.randint(0, 50)
        le = 5
        data = batch[:, i0, :-1]
        data_to_predict = batch[:, i0:i0+le, :-1]
        time_steps_to_predict = batch[:, i0:i0+le, -1]
        
        pred_y = torch.zeros(n_traj, le, n_dims).float().to(device)
        for i in range(n_traj):
            y0 = data[i:i+1]
            tt = time_steps_to_predict[i]
            pred = ode.odeint(self.sepi, y0, tt, rtol=1e-4, atol=1e-6, method='euler').permute(1,0,2)
            pred_y[i:i+1] = pred
        
        mse = (data_to_predict - pred_y)**2
        
        results = {}
        results["loss"] = torch.mean(mse)
            
        return(results, pred_y)
    
    def TestInfo(self, experimentID, test_dataset, train_res, itr, num_batches, kl_coef, logger):
        test_res, pred_y = self.compute_all_losses(test_dataset)
        
        logger.info("Experiment " + str(experimentID))
        logger.info("Epoch {:04d}".format(itr//num_batches))
        logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
        logger.info("Test MSE: {:.4f}".format(test_res["loss"]))
    
    def test(self, batch):
        device = self.device
        args = self.args
        latents = self.latents
        
        with torch.no_grad():
            #test_res, pred_y = self.compute_all_losses(batch)
            n_traj, n_tp, n_dims = batch[...,:-1].size()
            data = batch[:, 0, :-1]
            data_to_predict = batch[:, :, :-1]
            time_steps_to_predict = batch[:, :, -1]

            pred_y = torch.zeros(n_traj, n_tp, n_dims).float().to(device)
            for i in range(n_traj):
                y0 = data[i:i+1]
                tt = time_steps_to_predict[i]
                pred = ode.odeint(self.sepi, y0, tt, rtol=1e-4, atol=1e-6, method='dopri5').permute(1,0,2)
                pred_y[i:i+1] = pred            
            
        return(pred_y)              
        
        
        
        
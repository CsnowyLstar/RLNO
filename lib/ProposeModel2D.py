import os
import sys
import numpy as np
import scipy.io
import argparse
from random import SystemRandom

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from lib.functions2D import *
from lib.utils2D import *

################################################################
###   Proposed methods                                       ###
################################################################
class RLNO_MI(nn.Module):
    def __init__(self, args, device, input_size2, ostd = 0.01):
        super(RLNO_MI, self).__init__()
        
        self.device = device
        self.args = args
        self.ostd = ostd
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents
        self.gru_units = gru_units
        
        self.z0_operator_solver = DeepONet1(input_size1=rec_dims, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims*2, 128), nn.Tanh(), nn.Linear(128, latents*2),).to(device)
        init_network_weights(self.transform_z0)

        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        #self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.operator_solver = DeepONet_MI(input_size1=latents, input_size2=input_size2**2, input_size3=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*imput_dimy).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        n_traj_samples = args.n_traj_samples
        
        ## Operator-based RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        u_batch = batch[2]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp, :, :]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:,:].unsqueeze(0).reshape(1,n_traj,-1)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]

            latent_ys = []
            # Run NO backwards and combine the y(t) estimates using gating
            time_points_iter = reversed(range(0, encoder_ntp))

            for i in time_points_iter:
                Delta_t = t_i - prev_t
                y1 = prev_y
                y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                inc = self.z0_operator_solver(y1,y2)
                assert(not torch.isnan(inc).any())

                yi_no = inc
                xi = data[:,i,:,:].unsqueeze(0).reshape(1,n_traj,-1)
                yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
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
        
        ## Operator for reconstruction and prediction
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        first_point_std = first_point_std.abs()
        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0
        
        y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,n_tp,1)
        y2 = u_batch.reshape(n_traj,-1).unsqueeze(0).unsqueeze(-2).repeat(n_traj_samples,1,n_tp,1)
        y3 = t_batch.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
        sol_y = self.operator_solver(y1,y2,y3)
        
        pred_x = self.decoder(sol_y)
        all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

        pred_y, info = pred_x, all_extra_info
        
        
        #print("get_reconstruction done -- computing likelihood")
        obsrv_std = torch.Tensor([self.ostd]).to(device)
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
    
    def test_full(self, batch, time_full):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        n_traj_samples = args.n_traj_samples
        
        with torch.no_grad():
            ## Operator-based RNN for encoder 
            s_batch = batch[0]
            t_batch = batch[1]
            u_batch = batch[2]
            n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()

            encoder_ntp = min(args.rec_len, n_tp)
            data = batch[0][:, :encoder_ntp]
            time_steps = batch[1][:, :encoder_ntp]

            if time_steps.shape[1] == 1:
                prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
                xi = data[:,0,:,:].unsqueeze(0)
                last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
                extra_info = None
            else:
                prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]

                latent_ys = []
                # Run NO backwards and combine the y(t) estimates using gating
                time_points_iter = reversed(range(0, encoder_ntp))

                for i in time_points_iter:
                    Delta_t = t_i - prev_t
                    y1 = prev_y
                    y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                    inc = self.z0_operator_solver(y1,y2)
                    assert(not torch.isnan(inc).any())

                    yi_no = inc
                    xi = data[:,i,:].unsqueeze(0)
                    yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
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

            ## Operator for reconstruction and prediction
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
            first_point_std = first_point_std.abs()
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

            y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,time_full.shape[-1],1)
            y2 = u_batch.reshape(n_traj,-1).unsqueeze(0).unsqueeze(-2).repeat(n_traj_samples,1,time_full.shape[-1],1)
            y3 = time_full.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
            sol_y = self.operator_solver(y1,y2,y3)
        
            pred_x = self.decoder(sol_y)
            all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

            pred_y, info = pred_x, all_extra_info
            
            return(pred_y)

class RLNO(nn.Module):
    def __init__(self, args, device, ostd = 0.01):
        super(RLNO, self).__init__()
        
        self.device = device
        self.args = args
        self.ostd = ostd
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents
        self.gru_units = gru_units
        
        self.z0_operator_solver = DeepONet1(input_size1=rec_dims, input_size2=1, hidden_size=100, p=8, device=device).to(device)
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims*2, 128), nn.Tanh(), nn.Linear(128, latents*2),).to(device)
        init_network_weights(self.transform_z0)

        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=100, p=8, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        n_traj_samples = args.n_traj_samples
        
        ## Operator-based RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp, :, :]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:,:].unsqueeze(0).reshape(1,n_traj,-1)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]

            latent_ys = []
            # Run NO backwards and combine the y(t) estimates using gating
            time_points_iter = reversed(range(0, encoder_ntp))

            for i in time_points_iter:
                Delta_t = t_i - prev_t
                y1 = prev_y
                y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                inc = self.z0_operator_solver(y1,y2)
                assert(not torch.isnan(inc).any())

                yi_no = inc
                xi = data[:,i,:,:].unsqueeze(0).reshape(1,n_traj,-1)
                yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
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
        
        ## Operator for reconstruction and prediction
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        first_point_std = first_point_std.abs()
        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0
        
        y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,n_tp,1)
        y2 = t_batch.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
        sol_y = self.operator_solver(y1,y2)

        pred_x = self.decoder(sol_y)
        all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

        pred_y, info = pred_x, all_extra_info
        
        #print("get_reconstruction done -- computing likelihood")
        obsrv_std = torch.Tensor([self.ostd]).to(device)
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
    
    def test_full(self, batch, time_full):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        n_traj_samples = args.n_traj_samples
        
        with torch.no_grad():
            ## Operator-based RNN for encoder 
            s_batch = batch[0]
            t_batch = batch[1]
            n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()

            encoder_ntp = min(args.rec_len, n_tp)
            data = batch[0][:, :encoder_ntp]
            time_steps = batch[1][:, :encoder_ntp]

            if time_steps.shape[1] == 1:
                prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
                xi = data[:,0,:].unsqueeze(0)
                last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
                extra_info = None
            else:
                prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]
                
                latent_ys = []
                # Run NO backwards and combine the y(t) estimates using gating
                time_points_iter = reversed(range(0, encoder_ntp))
                
                for i in time_points_iter:
                    Delta_t = t_i - prev_t
                    y1 = prev_y
                    y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                    inc = self.z0_operator_solver(y1,y2)
                    assert(not torch.isnan(inc).any())

                    yi_no = inc
                    xi = data[:,i,:].unsqueeze(0)
                    yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
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
            
            ## Operator for reconstruction and prediction
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
            first_point_std = first_point_std.abs()
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

            y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,time_full.shape[-1],1)
            y2 = time_full.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
            sol_y = self.operator_solver(y1,y2)

            pred_x = self.decoder(sol_y)
            all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

            pred_y, info = pred_x, all_extra_info
            
            return(pred_y)
        
class RLNO_cn(nn.Module):
    def __init__(self, args, device, ostd = 0.01):
        super(RLNO_cn, self).__init__()
        
        self.device = device
        self.args = args
        self.ostd = ostd
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents
        self.gru_units = gru_units
        
        self.z0_operator_solver = DeepONet1(input_size1=rec_dims, input_size2=1, hidden_size=64, p=16, device=device).to(device)
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims*2, 128), nn.Tanh(), nn.Linear(128, latents*2),).to(device)
        init_network_weights(self.transform_z0)
        
        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        self.operator_solver = DeepONet_2D(input_size1=latents, input_size2=1, hidden_size=64, p=4, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        n_traj_samples = args.n_traj_samples
        
        ## Operator-based RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp, :, :]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:,:].unsqueeze(0).reshape(1,n_traj,-1)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]

            latent_ys = []
            # Run NO backwards and combine the y(t) estimates using gating
            time_points_iter = reversed(range(0, encoder_ntp))

            for i in time_points_iter:
                Delta_t = t_i - prev_t
                y1 = prev_y
                y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                inc = self.z0_operator_solver(y1,y2)
                assert(not torch.isnan(inc).any())

                yi_no = inc
                xi = data[:,i,:,:].unsqueeze(0).reshape(1,n_traj,-1)
                yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
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
        
        ## Operator for reconstruction and prediction
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        first_point_std = first_point_std.abs()
        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0
        
        y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,n_tp,1)
        y2 = t_batch.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
        sol_y = self.operator_solver(y1,y2)
        
        pred_x = self.decoder(sol_y)
        all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

        pred_y, info = pred_x, all_extra_info
        
        #print("get_reconstruction done -- computing likelihood")
        obsrv_std = torch.Tensor([self.ostd]).to(device)
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
    
    def test_full(self, batch, time_full):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        n_traj_samples = args.n_traj_samples
        
        with torch.no_grad():
            ## Operator-based RNN for encoder 
            s_batch = batch[0]
            t_batch = batch[1]
            n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()

            encoder_ntp = min(args.rec_len, n_tp)
            data = batch[0][:, :encoder_ntp]
            time_steps = batch[1][:, :encoder_ntp]

            if time_steps.shape[1] == 1:
                prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
                xi = data[:,0,:].unsqueeze(0)
                last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
                extra_info = None
            else:
                prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
                prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]
                
                latent_ys = []
                # Run NO backwards and combine the y(t) estimates using gating
                time_points_iter = reversed(range(0, encoder_ntp))
                
                for i in time_points_iter:
                    Delta_t = t_i - prev_t
                    y1 = prev_y
                    y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                    inc = self.z0_operator_solver(y1,y2)
                    assert(not torch.isnan(inc).any())

                    yi_no = inc
                    xi = data[:,i,:].unsqueeze(0)
                    yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
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
            
            ## Operator for reconstruction and prediction
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
            first_point_std = first_point_std.abs()
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

            y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,time_full.shape[-1],1)
            y2 = time_full.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
            sol_y = self.operator_solver(y1,y2)

            pred_x = self.decoder(sol_y)
            all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

            pred_y, info = pred_x, all_extra_info
            
            return(pred_y)

################################################################
###   Ablation experiments                                   ###
################################################################            

################################################################
###   RNN encoder                                            ###
################################################################   
class LNO_Ab1(nn.Module):
    def __init__(self, args, device, ostd = 0.01):
        super(LNO_Ab1, self).__init__()
        
        self.device = device
        self.args = args
        self.ostd = ostd
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; 
        self.latents = latents; self.gru_units = gru_units
        
        self.z0_operator_solver = DeepONet1(input_size1=rec_dims, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims*2, 100), nn.Tanh(), nn.Linear(100, latents*2),).to(device)
        init_network_weights(self.transform_z0)
        
        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        latents = self.latents
        gru_units = self.gru_units
        n_traj_samples = args.n_traj_samples
        
        ## RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:,:].unsqueeze(0).reshape(1,n_traj,-1)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]

            latent_ys = []
            # Run NO backwards and combine the y(t) estimates using gating
            time_points_iter = reversed(range(0, encoder_ntp))

            for i in time_points_iter:
                xi = data[:,i,:,:].unsqueeze(0).reshape(1,n_traj,-1)
                yi, yi_std = self.GRU_update(prev_y, prev_std, xi)
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
        
        ## Operator for reconstruction and prediction
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        first_point_std = first_point_std.abs()
        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0
        
        y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,n_tp,1)
        y2 = t_batch.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
        sol_y = self.operator_solver(y1,y2)

        pred_x = self.decoder(sol_y)
        all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

        pred_y, info = pred_x, all_extra_info
        
        #print("get_reconstruction done -- computing likelihood")
        obsrv_std = torch.Tensor([self.ostd]).to(device)
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
###   GRUDecay encoder                                       ###
################################################################  
class LNO_Ab2(nn.Module):
    def __init__(self, args, device):
        super(LNO_Ab2, self).__init__()
        
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
        
        self.z0_net = nn.Sequential(nn.Linear(rec_dims, 100), nn.Tanh(), nn.Linear(100, latents*2),).to(device)
        init_network_weights(self.z0_net)
        
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        latents = self.latents
        gru_units = self.gru_units
        n_traj_samples = args.n_traj_samples
        
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
        
        z0_mean, z0_std = split_last_dim(self.z0_net(hidden_state.unsqueeze(0)))
        z0_std = z0_std.abs()
        means_z0 = z0_mean.repeat(n_traj_samples, 1, 1)
        sigma_z0 = z0_std.repeat(n_traj_samples, 1, 1)
        z0_sample = sample_standard_gaussian(means_z0, sigma_z0)
        
        first_point_enc_aug = z0_sample
        y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,n_tp,1)
        y2 = t_batch.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
        
        sol_y = self.operator_solver(y1,y2)

        pred_x = self.decoder(sol_y)
        all_extra_info = {"first_point": (z0_mean, z0_std, z0_sample), "latent_traj": sol_y.detach()}

        pred_y, info = pred_x, all_extra_info

        
        #print("get_reconstruction done -- computing likelihood")
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
###   ELBO --> MSE                                           ###
################################################################  
class LNO_Ab3(nn.Module):
    def __init__(self, args, device):
        super(LNO_Ab3, self).__init__()
        
        self.device = device
        self.args = args
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents; self.gru_units = gru_units

        self.z0_operator_solver = DeepONet1(input_size1=rec_dims, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims, 100), nn.Tanh(), nn.Linear(100, latents),).to(device)
        init_network_weights(self.transform_z0)
        
        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        latents = self.latents
        gru_units = self.gru_units
        
        ## Operator-based RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dimsx, n_dimsy = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:,:].unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]
            
            latent_ys = []
            # Run NO backwards and combine the y(t) estimates using gating
            time_points_iter = reversed(range(0, encoder_ntp))

            for i in time_points_iter:
                Delta_t = t_i - prev_t
                y1 = prev_y
                y2 = Delta_t.unsqueeze(0).unsqueeze(-1)
                inc = self.z0_operator_solver(y1,y2)
                assert(not torch.isnan(inc).any())

                yi_no = inc
                xi = data[:,i,:,:].unsqueeze(0).reshape(1,n_traj,-1)
                yi, yi_std = self.GRU_update(yi_no, prev_std, xi)
                prev_y = yi
                prev_t, t_i = time_steps[:,i],  time_steps[:,i-1]
                latent_ys.append(yi)
            
            latent_ys = torch.stack(latent_ys, 1)
            last_yi = yi
        
        means_z0 = last_yi.reshape(1, n_traj, rec_dims)
        mean_z0 = self.transform_z0(means_z0)
        
        ## Operator for reconstruction and prediction
        y1 = mean_z0.squeeze(0).unsqueeze(-2).repeat(1,n_tp,1)
        y2 = t_batch.unsqueeze(-1)
        sol_y = self.operator_solver(y1,y2)
        
        pred_y = self.decoder(sol_y)
        mse = (s_batch.reshape(n_traj, n_tp, -1) - pred_y)**2
        
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
###   Other ablation methods                                 ###
################################################################

################################################################
###   RNN encoder + mse                                      ###
################################################################
class LNO_Ab4(nn.Module):
    def __init__(self, args, device):
        super(LNO_Ab4, self).__init__()
        
        self.device = device
        self.args = args
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; 
        self.latents = latents; self.gru_units = gru_units
        
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims, 100), nn.Tanh(), nn.Linear(100, latents),).to(device)
        init_network_weights(self.transform_z0)
        
        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
        self.iftest = False
    
    def compute_all_losses(self, batch, kl_coef = 1.):
        device = self.device
        args = self.args
        rec_dims = self.rec_dims
        latents = self.latents
        gru_units = self.gru_units
        n_traj_samples = args.n_traj_samples
        
        ## RNN for encoder 
        s_batch = batch[0]
        t_batch = batch[1]
        n_traj, n_tp, n_dims = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp]
        time_steps = batch[1][:, :encoder_ntp]

        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:].unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]

            latent_ys = []
            # Run NO backwards and combine the y(t) estimates using gating
            time_points_iter = reversed(range(0, encoder_ntp))

            for i in time_points_iter:
                xi = data[:,i,:].unsqueeze(0)
                yi, yi_std = self.GRU_update(prev_y, prev_std, xi)
                prev_y, prev_std = yi, yi_std
                prev_t, t_i = time_steps[:,i],  time_steps[:,i-1]
                latent_ys.append(yi)
        
            latent_ys = torch.stack(latent_ys, 1)
            last_yi = yi

        means_z0 = last_yi.reshape(1, n_traj, rec_dims)
        mean_z0 = self.transform_z0(means_z0)
        
        ## Operator for reconstruction and prediction
        y1 = mean_z0.squeeze(0).unsqueeze(-2).repeat(1,n_tp,1)
        y2 = t_batch.unsqueeze(-1)
        sol_y = self.operator_solver(y1,y2)

        pred_y = self.decoder(sol_y)
        mse = (s_batch - pred_y)**2
        
        results = {}
        results["loss"] = torch.mean(mse)

        if self.iftest:
            self.iftest = False
            y1 = mean_z0.squeeze(0).unsqueeze(-2).repeat(1,self.time_full.shape[-1],1)
            y2 = self.time_full.unsqueeze(-1)
            sol_y = self.operator_solver(y1,y2)
            pred_y = self.decoder(sol_y)  

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

    def test_full(self, batch, time_full):
        with torch.no_grad():
            self.iftest = True
            self.time_full = time_full
            test_res, pred_y = self.compute_all_losses(batch)
        
        return(pred_y)

################################################################
###   ODE-RNN encoder                                        ###
################################################################  
class LNO_Ab5(nn.Module):
    def __init__(self, args, device):
        super(LNO_Ab5, self).__init__()
        
        self.device = device
        self.args = args
        input_dimx = args.input_dimx
        input_dimy = args.input_dimy
        rec_dims = args.rec_dims
        latents = args.latents
        gru_units = args.gru_units
        self.rec_dims = rec_dims; self.latents = latents; self.gru_units = gru_units
        
        ode_func_net = create_net(rec_dims, rec_dims, n_layers = 1, n_units = 100, nonlinear = nn.Tanh)
        rec_ode_func = ODEFunc(input_dim = input_dimx+input_dimy, latent_dim = rec_dims, 
                               ode_func_net = ode_func_net, device = device).to(device)
        self.z0_diffeq_solver = DiffeqSolver(input_dimx+input_dimy, rec_ode_func, "euler", 
                                        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        #self.z0_operator_solver = DeepONet1(input_size1=rec_dims, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.transform_z0 = nn.Sequential(nn.Linear(rec_dims*2, 100), nn.Tanh(), nn.Linear(100, latents*2),).to(device)
        init_network_weights(self.transform_z0)

        self.GRU_update = GRU_unit(rec_dims, input_dimx, input_dimy, n_units=gru_units, device=device).to(device)
        self.operator_solver = DeepONet1(input_size1=latents, input_size2=1, hidden_size=64, p=32, device=device).to(device)
        self.decoder = Decoder(latents, input_dimx*input_dimy).to(device)
        self.iftest = False
    
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
        n_traj, n_tp, n_dims = s_batch.size()
        
        encoder_ntp = min(args.rec_len, n_tp)
        data = batch[0][:, :encoder_ntp]
        time_steps = batch[1][:, :encoder_ntp]
        
        if time_steps.shape[1] == 1:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            xi = data[:,0,:].unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            prev_y = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_std = torch.zeros((1, n_traj, rec_dims)).to(device)
            prev_t, t_i = time_steps[:,-1] + 0.01,  time_steps[:,-1]
            minimum_step = min(time_steps[:,-1] - time_steps[:,0]) / 50
            
            latent_ys = []
            # Run ODE backwards and combine the y(t) estimates using gating
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
                
                xi = data[:,i,:].unsqueeze(0)
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
        
        ## Operator for reconstruction and prediction
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        first_point_std = first_point_std.abs()
        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0
        
        y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,n_tp,1)
        y2 = t_batch.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
        sol_y = self.operator_solver(y1,y2)

        pred_x = self.decoder(sol_y)
        all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach()}

        pred_y, info = pred_x, all_extra_info

        
        #print("get_reconstruction done -- computing likelihood")
        obsrv_std = torch.Tensor([0.01]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_std[fp_std==0] = 1e-6
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, z0_prior)
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))
        
        # Compute likelihood of all the points
        rec_likelihood = get_gaussian_likelihood(s_batch, pred_y, obsrv_std)
        mse = get_mse(s_batch, pred_y)
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

        if self.iftest:
            self.iftest = False
            y1 = first_point_enc_aug.unsqueeze(-2).repeat(1,1,self.time_full.shape[-1],1)
            y2 = self.time_full.unsqueeze(0).unsqueeze(-1).repeat(n_traj_samples,1,1,1)
            sol_y = self.operator_solver(y1,y2)
            pred_y = self.decoder(sol_y)   
        
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
        
    def test_full(self, batch, time_full):
        with torch.no_grad():
            self.iftest = True
            self.time_full = time_full
            test_res, pred_y = self.compute_all_losses(batch)
        
        return(pred_y)        
        
        
        
        
        
        
        
        
        
        
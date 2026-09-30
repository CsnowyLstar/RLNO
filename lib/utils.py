import torch
import numpy as np
import scipy.io
import h5py
import logging

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions import Independent

import operator


# read data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, mode='r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float
        

def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)
          
        
def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d) 
        log_prob = log_prob / n_data_points 
    else:
        log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
    return log_prob


def gaussian_log_density(mu, data, obsrv_std):
    if (len(mu.size()) == 3):
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)
        
    if (len(data.size()) == 2):
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    assert(data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
    data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)

    res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
    res = res.reshape(n_traj_samples, n_traj).transpose(0,1)

    return res


def get_gaussian_likelihood(truth, pred_y, obsrv_std):
    n_traj, n_tp, n_dim = truth.size()
    truth_repeated = truth.unsqueeze(0).repeat(pred_y.size(0), 1, 1, 1)

    log_density_data = gaussian_log_density(pred_y, truth_repeated, obsrv_std = obsrv_std)
    log_density_data = log_density_data.permute(1,0)
    log_density = torch.mean(log_density_data, 1)

    return log_density


def mse(mu, data, indices = None):
    n_data_points = mu.size()[-1]

    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).to(get_device(data)).squeeze()
    return mse


def compute_mse(mu, data, mask = None):
    # these cases are for plotting through plot_estim_density
    if (len(mu.size()) == 3):
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if (len(data.size()) == 2):
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert(data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
    data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
    res = mse(mu_flat, data_flat)

    return res


def get_mse(truth, pred_y):
    n_traj, n_tp, n_dim = truth.size()
    truth_repeated = truth.unsqueeze(0).repeat(pred_y.size(0), 1, 1, 1)
    log_density_data = compute_mse(pred_y, truth_repeated)
    return torch.mean(log_density_data)


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr
        
        
def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()

    return batch_dict

def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res, 
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def shift_outputs(outputs, first_datapoint = None):
    outputs = outputs[:,:,:-1,:]

    if first_datapoint is not None:
        n_traj, n_dims = first_datapoint.size()
        first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
        outputs = torch.cat((first_datapoint, outputs), 2)
    return outputs

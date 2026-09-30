import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase
from lib.utils import *
from torchdiffeq import odeint as odeint

class DeepONet1(nn.Module):  # myModel
    def __init__(self, input_size1, input_size2, hidden_size, p, device):
        super(DeepONet1, self).__init__()

        self.branch = nn.Sequential(nn.Linear(input_size1, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p*input_size1, bias=True)).to(device)
        
        self.trunk = nn.Sequential(nn.Linear(input_size2, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p*input_size1, bias=False)).to(device)
        
        self.param = torch.ones(1, requires_grad=True).to(device)
        
        self.p = p
        self.m = input_size1
    
    def forward(self, x1, x2):
        new_shape = x1.shape[:-1] + (self.m, self.p)
        y_branch = self.branch(x1).reshape(*new_shape)
        y_trunk = self.trunk(x2).reshape(*new_shape)
        guy = (torch.einsum("...i,...i->...", y_branch, y_trunk) + self.param)
        return(guy)


class DeepONet2(nn.Module):  # myModel
    def __init__(self, input_size1, input_size2, hidden_size, p, device):
        super(DeepONet2, self).__init__()
        
        self.branch = nn.Sequential(nn.Linear(input_size1, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p, bias=True)).to(device)
        
        self.trunk = nn.Sequential(nn.Linear(input_size2, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p, bias=False)).to(device)
        
        self.param = torch.ones(1, requires_grad=True).to(device)
    
    def forward(self, x1, x2):
        y_branch = self.branch(x1)
        y_trunk = self.trunk(x2)
        guy = (torch.einsum("...i,...i->...", y_branch, y_trunk) + self.param)
        return(guy)

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class DeepONet_2D(nn.Module):  # myModel
    def __init__(self, input_size1, input_size2, hidden_size, p, device):
        super(DeepONet_2D, self).__init__()
        self.p = p
        self.m = input_size1
        self.branch = nn.Sequential(*self.build_branch_net()).to(device)
        self.trunk = nn.Sequential(*self.build_trunk_net()).to(device)
        self.param = torch.ones(1, requires_grad=True).to(device)
        self.device = device

    def build_branch_net(self):
        ls=[]
        ls.append(nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0, bias=False),
            SinActivation(),  nn.BatchNorm2d(32)))
        if self.m > 16:  # reduce conv layers for small latent dimensions
            ls.append(nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=0, bias=False),
                SinActivation(),  nn.BatchNorm2d(16)))
        if self.m > 36:  # reduce conv layers for small latent dimensions
            ls.append(nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False),
                SinActivation(),  nn.BatchNorm2d(16)))
        if self.m > 64:  # reduce conv layers for small latent dimensions
            ls.append(nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False),
                SinActivation(),  nn.BatchNorm2d(16)))
        ls.append(nn.Flatten())
        ls.append(nn.LazyLinear(self.m * self.p))
        return ls
    
    def build_trunk_net(self):
        trunk_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=100),
            SinActivation(),
            nn.Linear(100, 100),
            SinActivation(),
            nn.Linear(100, self.m * self.p)
        )
        return trunk_net

    def forward(self,x1,x2):
        sqrt = int(np.sqrt(x1.shape[-1]))
        x1_func = x1.reshape(x1.shape[0]*x1.shape[1]*x1.shape[2], 1, sqrt, sqrt)
        x1_func = self.branch(x1_func)
        #y_loc = 10*(X_loc-0.5)
        x2_loc = self.trunk(x2)
        y_branch = x1_func.view(-1, self.m, self.p)
        y_trunk = x2_loc.view(-1, self.m, self.p)
        Y = (torch.einsum("...i,...i->...", y_branch, y_trunk) + self.param)
        return(Y.reshape(x1.shape[0], x1.shape[1], x1.shape[2], self.m))
    
    '''
    def forward(self, x1, x2):
        print("x1,x2,shape:",x1.shape,x2.shape)
        sqrt = int(np.sqrt(x1.shape[-1]))
        x1_func = x1.reshape(x1.shape[0]*x1.shape[1]*x1.shape[2], 1, sqrt, sqrt)
        x2_loc = x2[0,0]
        for i in range(len(self.branch)):
            x1_func = self.branch[i](x1_func)
        #y_loc = 10*(X_loc-0.5)
        for i in range(len(self.trunk)):
            x2_loc = self.trunk[i](x2_loc)
        y_func = x1_func.view(-1, self.m, self.p)
        y_loc = x2_loc.view(-1, self.m, self.p)
        Y = torch.einsum('ijk,pjk->ipj', y_func, y_loc)
        return(Y)
    '''

class DeepONet_MI(nn.Module):  # myModel
    def __init__(self, input_size1, input_size2, input_size3, hidden_size, p, device):
        super(DeepONet_MI, self).__init__()

        self.branch1 = nn.Sequential(nn.Linear(input_size1, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p*input_size1, bias=True)).to(device)

        self.branch2 = nn.Sequential(nn.Linear(input_size2, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p*input_size1, bias=True)).to(device)
        
        self.trunk = nn.Sequential(nn.Linear(input_size3, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                               nn.Linear(hidden_size, p*input_size1, bias=False)).to(device)
        
        self.param = torch.ones(1, requires_grad=True).to(device)
        
        self.p = p
        self.m = input_size1
    
    def forward(self, x1, x2, x3):
        new_shape = x1.shape[:-1] + (self.m, self.p)
        y_branch1 = self.branch1(x1).reshape(*new_shape)
        y_branch2 = self.branch2(x2).reshape(*new_shape)
        y_trunk = self.trunk(x3).reshape(*new_shape)
        guy = (torch.einsum("...i,...i->...", y_branch1*y_branch2, y_trunk) + self.param)
        return(guy)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dimx, input_dimy, update_gate = None, reset_gate = None, new_state_net = None,
        n_units = 100, device = torch.device("cpu")):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dimx*input_dimy, n_units),
               nn.Tanh(), nn.Linear(n_units, latent_dim), nn.Sigmoid())
            init_network_weights(self.update_gate)
        else: 
            self.update_gate  = update_gate
        
        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dimx*input_dimy, n_units),
               nn.Tanh(), nn.Linear(n_units, latent_dim), nn.Sigmoid())
            init_network_weights(self.reset_gate)
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dimx*input_dimy, n_units),
               nn.Tanh(), nn.Linear(n_units, latent_dim * 2))
            init_network_weights(self.new_state_net)
        else: 
            self.new_state_net  = new_state_net

    
    def forward(self, y_mean, y_std, x, masked_update = False):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        assert(not torch.isnan(new_y).any())

        new_y_std = new_y_std.abs()
        return new_y, new_y_std
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Exponential decay of the hidden states for RNN
# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
# Exp decay between hidden states
class GRUCellExpDecay(RNNCellBase):
    def __init__(self, input_size, input_size_for_decay, hidden_size, device, bias=True):
        super(GRUCellExpDecay, self).__init__(input_size, hidden_size, bias, num_chunks=3)

        self.device = device
        self.input_size_for_decay = input_size_for_decay
        self.decay = nn.Sequential(nn.Linear(input_size_for_decay, 1),)
        init_network_weights(self.decay)
    
    def gru_exp_decay_cell(self, input, hidden, w_ih, w_hh, b_ih, b_hh):
        # INPORTANT: assumes that cum delta t is the last dimension of the input
        batch_size, n_dims = input.size()

        # "input" contains the data, mask and also cumulative deltas for all inputs
        cum_delta_ts = input[:, -self.input_size_for_decay:]
        data = input[:, :-self.input_size_for_decay]
        
        decay = torch.exp( - torch.min(torch.max(
            torch.zeros([1]).to(self.device), self.decay(cum_delta_ts)), 
            torch.ones([1]).to(self.device) * 1000 ))
        
        hidden = hidden * decay
        
        gi = torch.mm(data, w_ih.t()) + b_ih
        gh = torch.mm(hidden, w_hh.t()) + b_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        #self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        #self.check_forward_hidden(input, hx, '')
        
        return self.gru_exp_decay_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh
        )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Imputation with a weighed average of previous value and empirical mean 
# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
def get_cum_delta_ts(data, delta_ts, mask):
    n_traj, n_tp, n_dims = data.size()

    cum_delta_ts = delta_ts.repeat(1, 1, n_dims)
    missing_index = np.where(mask.cpu().numpy() == 0)

    for idx in range(missing_index[0].shape[0]):
        i = missing_index[0][idx] 
        j = missing_index[1][idx]
        k = missing_index[2][idx]

        if j != 0 and j != (n_tp-1):
            cum_delta_ts[i,j+1,k] = cum_delta_ts[i,j+1,k] + cum_delta_ts[i,j,k]
    cum_delta_ts = cum_delta_ts / cum_delta_ts.max() # normalize

    return cum_delta_ts


# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
# very slow
def impute_using_input_decay(data, delta_ts, mask, w_input_decay, b_input_decay):
    n_traj, n_tp, n_dims = data.size()

    cum_delta_ts = delta_ts.repeat(1, 1, n_dims)
    missing_index = np.where(mask.cpu().numpy() == 0)

    data_last_obsv = np.copy(data.cpu().numpy())
    for idx in range(missing_index[0].shape[0]):
        i = missing_index[0][idx] 
        j = missing_index[1][idx]
        k = missing_index[2][idx]

        if j != 0 and j != (n_tp-1):
            cum_delta_ts[i,j+1,k] = cum_delta_ts[i,j+1,k] + cum_delta_ts[i,j,k]
        if j != 0:
            data_last_obsv[i,j,k] = data_last_obsv[i,j-1,k] # last observation
    cum_delta_ts = cum_delta_ts / cum_delta_ts.max() # normalize

    data_last_obsv = torch.Tensor(data_last_obsv).to(get_device(data))

    zeros = torch.zeros([n_traj, n_tp, n_dims]).to(get_device(data))
    decay = torch.exp( - torch.min( torch.max(zeros, 
        w_input_decay * cum_delta_ts + b_input_decay), zeros + 1000 ))

    data_means = torch.mean(data, 1).unsqueeze(1)

    data_imputed = data * mask + (1-mask) * (decay * data_last_obsv + (1-decay) * data_means)
    return data_imputed


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def run_rnn(inputs, delta_ts, cell, first_hidden=None, 
    mask = None, feed_previous=False, n_steps=0,
    decoder = None, input_decay_params = None,
    feed_previous_w_prob = 0.,
    masked_update = True):
    if (feed_previous or feed_previous_w_prob) and decoder is None:
        raise Exception("feed_previous is set to True -- please specify RNN decoder")
    
    if n_steps == 0:
        n_steps = inputs.size(1)
    
    #if (feed_previous or feed_previous_w_prob) and mask is None:
    if mask is None:
        mask = torch.ones((inputs.size(0), n_steps, inputs.size(-1))).to(get_device(inputs))
    
    if isinstance(cell, GRUCellExpDecay):
        cum_delta_ts = get_cum_delta_ts(inputs, delta_ts, mask)
    
    if input_decay_params is not None:
        w_input_decay, b_input_decay = input_decay_params
        inputs = impute_using_input_decay(inputs, delta_ts, mask, w_input_decay, b_input_decay)
    
    all_hiddens = []
    hidden = first_hidden

    if hidden is not None:
        all_hiddens.append(hidden)
        n_steps -= 1

    for i in range(n_steps):
        delta_t = delta_ts[:,i]
        if i == 0:
            rnn_input = inputs[:,i]
        elif feed_previous:
            rnn_input = decoder(hidden)
        elif feed_previous_w_prob > 0:
            feed_prev = np.random.uniform() > feed_previous_w_prob
            if feed_prev:
                rnn_input = decoder(hidden)
            else:
                rnn_input = inputs[:,i]
        else:
            rnn_input = inputs[:,i]

        if mask is not None:
            mask_i = mask[:,i,:]
            rnn_input = torch.cat((rnn_input, mask_i), -1)

        if isinstance(cell, GRUCellExpDecay):
            cum_delta_t = cum_delta_ts[:,i]
            input_w_t = torch.cat((rnn_input, cum_delta_t), -1).squeeze(1)
        else:
            input_w_t = torch.cat((rnn_input, delta_t), -1).squeeze(1)

        prev_hidden = hidden
        hidden = cell(input_w_t, hidden)
        
        if masked_update and (mask is not None) and (prev_hidden is not None):
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            summed_mask = (torch.sum(mask_i, -1, keepdim = True) > 0).float()
            assert(not torch.isnan(summed_mask).any())
            hidden = summed_mask * hidden + (1-summed_mask) * prev_hidden

        all_hiddens.append(hidden)

    all_hiddens = torch.stack(all_hiddens, 0)
    all_hiddens = all_hiddens.permute(1,0,2).unsqueeze(0)
    return hidden, all_hiddens



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an NO back to the data space

        #decoder = nn.Sequential(nn.Linear(latent_dim, input_dim),)
        decoder = nn.Sequential(nn.Linear(latent_dim, 169, bias=True), nn.ReLU(), 
                                nn.Linear(169, 256, bias=True), nn.ReLU(),
                                nn.Linear(256, 400, bias=True), nn.ReLU(),
                                nn.Linear(400, input_dim, bias=True),)

        init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)

    
class MLAE_enc(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MLAE_enc, self).__init__()

        encoder = nn.Sequential(nn.Linear(input_dim, 400, bias=True), nn.ReLU(), 
                                nn.Linear(400, 256, bias=True), nn.ReLU(),
                                nn.Linear(256, 169, bias=True), nn.ReLU(),
                                nn.Linear(169, latent_dim, bias=True), nn.ReLU(),)

        init_network_weights(encoder)
        self.encoder = encoder

    def forward(self, data):
        return self.encoder(data)    
    
class MLAE_dec(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(MLAE_dec, self).__init__()

        decoder = nn.Sequential(nn.Linear(latent_dim, 169, bias=True), nn.ReLU(), 
                                nn.Linear(169, 256, bias=True), nn.ReLU(),
                                nn.Linear(256, 400, bias=True), nn.ReLU(),
                                nn.Linear(400, input_dim, bias=True),)

        init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)    
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #        
class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)    
    
class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, 
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards = False):
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)

        assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
        n_traj_samples = 1):
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y    
    
def create_net(n_inputs, n_outputs, n_layers = 1, 
    n_units = 100, nonlinear = nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

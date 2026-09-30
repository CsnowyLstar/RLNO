import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq as ode
from cubic_spline import CSpline
from torch.fft import fft, ifft
from tqdm import tqdm
import scipy.io
from multiprocessing import Pool

def RBF(x1, x2, params):
    length_scale, output_scale = params 
    diffs = np.expand_dims(x1 / length_scale, 1) - \
            np.expand_dims(x2 / length_scale, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

def generate_gaussain_sample(gp_params,N,T):
    np.random.seed(None)
    gp_samples = np.zeros((N,T))
    length_scale, output_scale = gp_params 
    jitter = 1e-10
    X = np.linspace(0.0, 1.0, T)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(T))
    for i in range(N):
        gp_sample = np.dot(L, np.random.normal(loc=0,scale=1,size=T))
        gp_samples[i,:] = gp_sample
    return gp_samples

class genc2(nn.Module):
    def __init__(self, kk, u0):
        super(genc2, self).__init__()
        self.D = 0.01
        self.K = 0.01
        self.kk = kk
        self.u0 = u0
        self.fit_data=None 
    
    def forward(self, t, x):
        u0 = self.u0
        ut = self.fit_data.fit(t).expand(u0.shape)
        u_b = u0+ut
        s_tilde = fft(x,axis=1)
        
        gu = self.D*ifft(-1*self.kk**2*s_tilde,axis=1).real +\
            self.K*x**2 + u_b #dr方程 
        return(gu)
    
def gen_one2(xx, y0, u, u0):
    tt = torch.linspace(0, T, Nt)
    
    k = torch.tensor(kk).unsqueeze(0)
    gend = genc2(k,u0)
    
    funcu = CSpline(tt, u[0])
    gend.fit_data = funcu
    
    sol = ode.odeint(gend, y0, tt, rtol=1e-6, atol=1e-8, method='dopri5')
    return(xx, tt, sol[:,0].numpy())

def gen_OneData(ind, gp_params, Nx, Nt, xx):
    u0 = torch.tensor(generate_gaussain_sample(gp_params, 1, Nx))
    y0 = torch.tensor(generate_gaussain_sample(gp_params, 1, Nx))
    x,t,s = gen_one2(xx, y0, torch.zeros(1,Nt), u0)
    return(ind, s.transpose(), t, u0)

N = 10000
Nx = 64
Nt = 100
T = 1
L = 1
kk = np.concatenate((np.arange(0, Nx/2),np.array([0]),np.arange(-Nx/2+1, 0)))*2*np.pi/L
xx = torch.linspace(0, L, Nx+1)[:-1]

length_scale = 0.4
output_scale = 1.0
gp_params = (length_scale, output_scale)

s_train = np.zeros((N,Nx,Nt))
u_train = np.zeros((N,Nx))
Nprocesses = 20
for ii in tqdm(range(N//Nprocesses)):
    pool = Pool(processes=Nprocesses)
    result = []
    for i in range(Nprocesses):
        ind = ii*Nprocesses + i
        result.append(pool.apply_async(gen_OneData, args = (ind, gp_params, Nx, Nt, xx)))
    pool.close()
    pool.join()
    for i in range(Nprocesses):
        ind, s, t, u0 = result[i].get()
        s_train[ind] = s
        u_train[ind] = u0[0]

scipy.io.savemat('dataset/data_drV3.mat', mdict={'Xs': s_train, 'ts': t, 'xs': xx, 'us': u_train})
print("Data successfully generated!", s_train.shape)
            





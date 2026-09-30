import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq as ode
from cubic_spline import CSpline
from random_fields import GaussianRF
import math
import torch.nn.functional as F
import seaborn as sns
import time
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
    def __init__(self, k_x, k_y, f0):
        super(genc2, self).__init__()
        self.k_x = k_x
        self.k_y = k_y
        self.f0 = f0
        self.fit_data = None
        self.nu = 1e-3
        
        self.kx1 = (1j*k_x)**1
        self.kx2 = (1j*k_x)**2
        self.ky1 = (1j*k_y)**1
        self.ky2 = (1j*k_y)**2
        
        # Negative Laplace operator
        lap = k_x**2 + k_y**2
        lap[0,0] = 1.0
        
        self.lap = (1.0+0j)*lap
        self.dlap = (1.0+0j)/lap
        
    def forward(self, t, x):
        '''
        f0 = self.f0
        ut = self.fit_data.fit(t).expand(f0.shape)
        f_h = f0*ut
        '''
        kx1 = self.kx1; ky1 = self.ky1; nu = self.nu
        lap = self.lap; dlap = self.dlap
        s_tilde = torch.fft.fft2(x)
        psi_h = s_tilde*dlap
        q = ky1*psi_h
        v = -kx1*psi_h
        w_x = kx1*s_tilde
        w_y = ky1*s_tilde
        F_h = torch.fft.ifft2(q).real * torch.fft.ifft2(w_x).real + torch.fft.ifft2(v).real * torch.fft.ifft2(w_y).real 
        gu = -F_h - nu*torch.fft.ifft2(lap*s_tilde).real + self.f0
        return(gu)
        
def gen_one2(sensor, f0):
    tt = torch.linspace(0, T, Nt+1)[:-1]
    xx = torch.linspace(0, L, Nx+1)[:-1]
    yy = torch.linspace(0, L, Ny+1)[:-1]
    
    #GRF = GaussianRF(2, Nx, alpha=2.5, tau=7)
    GRF = GaussianRF(2, Nx, alpha=alpha, tau=tau)
    y0 = GRF.sample(1).squeeze(0)
    
    k_max = math.floor(Nx/2.0)
    
    #Wavenumbers in y-direction
    k_y = 2*math.pi/L*torch.cat((torch.arange(start=0, end=k_max, step=1, device=y0.device),\
                    torch.arange(start=-k_max, end=0, step=1, device=y0.device)), 0).repeat(Ny,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    
    gend = genc2(k_x, k_y, f0)
    
    funcu = CSpline(tt, sensor[0]*0)
    gend.fit_data = funcu
    
    sol = ode.odeint(gend, y0, tt, rtol=1e-6, atol=1e-8, method='dopri5')
    return(xx, yy, tt, sol)

def gen_OneData(ind, gp_params, Nx, Nt, f0):
    xx, yy, tt, s = gen_one2(torch.zeros(1,Nt), f0)
    return(ind, xx, yy, s, tt)

N = 5000
Nx = 32
Ny = Nx
Nt = 100
T = 3
L = 2
#L = 2*np.pi

length_scale = 0.1
output_scale = 1
gp_params = (length_scale, output_scale)

alpha = 4; tau = 8
GRF = GaussianRF(2, Nx, alpha=alpha, tau=tau)
f0 = GRF.sample(1).squeeze(0)

t = torch.linspace(0, L, Nx+1)
t = t[0:-1]
X,Y = torch.meshgrid(t, t, indexing='ij')

ext_threshold = 100
u_train = np.zeros((N,Nt))
s_train = np.zeros((N,Nt,Nx,Ny))

Nprocesses = 500
for ii in tqdm(range(N//Nprocesses)):
    pool = Pool(processes=Nprocesses)
    result = []
    for i in range(Nprocesses):
        ind = ii*Nprocesses + i
        result.append(pool.apply_async(gen_OneData, args = (ind, gp_params, Nx, Nt, f0)))
    pool.close()
    pool.join()
    for i in range(Nprocesses):
        ind, xx, yy, s, tt = result[i].get()
        s_train[ind] = s
            
scipy.io.savemat('dataset/data_ns.mat', mdict={'Xs': s_train, 'ts': tt, 'xs': xx, 'ys': yy})
print("Data successfully generated!", s_train.shape)

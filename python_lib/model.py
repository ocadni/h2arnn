import torch
import numpy as np
import math
from numba import jit, jitclass
from numba import int32, float32, float64    # import the types

#int32 = np.int32
#int64 = np.int64
#float64 = np.float64
type_default = torch.float64
device="cpu"

class model():
    def __init__(self, N, H, J, J_interaction):
        self.N = N
        self.J = J
        self.H = H
        self.H_torch = torch.from_numpy(self.H).cpu().to(type_default)
        self.J_torch = torch.from_numpy(self.J).cpu().to(type_default)
        self.J_interaction = J_interaction
        self.Corr = torch.zeros(J_interaction.shape, device=device, dtype=type_default)
        assert N == len(H)
        assert J.size == N*N
        
    def exact(self, beta):
        assert self.N < 28
        J = self.J_torch
        H = self.H_torch
        N = self.N
        E_min = 0
        n_total = int(math.pow(2, self.N))
        Z = 0
        #print('Enumerating...')
        E_mean = 0
        M_mean = 0
        S_mean = 0
        M_i_mean = np.zeros(N)
        Corr = np.zeros(J.shape)
        for d in range(n_total):
            s = np.binary_repr(d, width=N)
            b_numpy = np.array(list(s)).astype(np.float32)
            b_numpy[b_numpy < 0.5] = -1
            b_ = torch.from_numpy(b_numpy).to(type_default)
            b = b_.view(N, 1)
            E = (- 0.5 * b.t() @ J @ b - b.t() @ H).data[0][0]
            if E < E_min:
                E_min = E
            Z_temp = torch.exp(-beta * E)
            E_mean += (E * Z_temp)
            M_mean += b.sum() * Z_temp
            M_i_mean += b_numpy * Z_temp.numpy()
            Z += Z_temp
            S_mean += - Z_temp * torch.log(Z_temp)
            Corr += np.outer(b_numpy,b_numpy) * Z_temp.numpy()
            
            if d % 1000 == 0:
                print('\r{} / {}({:.2%}), E = {:.3}, Z = {:.3}, F = {:.3}'.format(
                        d, 
                        n_total,
                        d/n_total,
                        E.numpy(),
                        Z.numpy(),
                        -(1/beta)*math.log(Z.numpy())), 
                      end="")
        
        print("\r", end="")
        self.free_energy = -(1/beta) * math.log((Z).numpy()) *(1./ self.N)
        self.E_mean = (E_mean / (Z * self.N)).numpy()
        self.M_mean = (M_mean / (Z * self.N)).numpy()
        self.S_mean = (S_mean/ (Z * self.N) - beta * self.free_energy).numpy()
        self.M_i_mean = (M_i_mean / Z.numpy())
        self.Corr = Corr / Z.numpy() - np.outer(self.M_i_mean, self.M_i_mean)
        #Corr -= torch.ger(M_i_mean,M_i_mean)   
        print("beta: {2:.1f}, Fe: {0:.3f}".format(self.free_energy, 
                                                  self.E_mean - (1./beta)*self.S_mean, beta)
             ,end=" ")
        print("Energy: {0:.3} M: {1:.3} S: {2:.3}".format(
                        self.E_mean, 
                        self.M_mean,
                        self.S_mean,
                        ),)
        
        return -(1/beta)* math.log(Z.numpy())
    
    def energy(self, samples):
        """
        Compute energy of samples, samples should be of size [m, n] where n is the number of spins, m is the number of samples.
        """
        samples = samples.view(samples.shape[0], -1).to(type_default)
        assert samples.shape[1] == self.N
        m = samples.shape[0]
        inter = - 0.5 * ((samples @ self.J_torch).view(m, 1, self.N) @ samples.view(
            m, self.N, 1)).squeeze()
        field = -(samples.view(m, 1, self.N) @ self.H_torch.view(self.N, 1)).squeeze()
        return (inter + field)

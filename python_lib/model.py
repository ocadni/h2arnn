import torch
import numpy as np
import math
from numba import jit, jitclass
from numba import int32, float32, float64    # import the types

#int32 = np.int32
#int64 = np.int64
#float64 = np.float64

class model():
    def __init__(self, N, H, J, J_interaction, beta = 1):
        self.N = N
        self.J = J
        self.H = H
        self.H_torch = torch.from_numpy(self.H).cpu().double()
        self.J_torch = torch.from_numpy(self.J).cpu().double()
        self.J_interaction = J_interaction
        self.beta = beta
        assert N == len(H)
        assert J.size == N*N
        
    def exact(self):
        assert self.N < 28
        J = torch.from_numpy(self.J).cpu().to(torch.float64)
        H = torch.from_numpy(self.H).cpu().to(torch.float64)
        beta = self.beta
        N = self.N
        E_min = 0
        n_total = int(math.pow(2, self.N))
        Z = 0
        print('Enumerating...')
        E_mean = 0
        M_mean = 0
        S_mean = 0
        for d in range(n_total):
            s = np.binary_repr(d, width=N)
            b = np.array(list(s)).astype(np.float32)
            b[b < 0.5] = -1
            b = torch.from_numpy(b).view(N, 1).to(torch.float64)
            E = - 0.5 * b.t() @ J @ b - b.t() @ H
            if E < E_min:
                E_min = E
            Z_temp = torch.exp(-beta * E)
            E_mean += (E * Z_temp)
            M_mean += b.sum() * Z_temp
            Z += Z_temp
            S_mean += -Z_temp * torch.log(Z_temp)
            
            if d % 1000 == 0:
                print('\r{} / {}({:.2%}), E = {:.3}, Z = {:.3}, F = {:.3}'.format(
                        d, 
                        n_total,
                        d/n_total,
                        E.numpy()[0][0],
                        Z.numpy()[0][0],
                        -(1/beta)*math.log(Z.numpy()[0][0])), 
                      end="")
        print("\r")
        self.free_energy = -(1/beta) * math.log((Z).numpy()[0][0]) *(1./ self.N)
        self.E_mean = (E_mean / (Z * self.N)).numpy()[0][0]
        self.M_mean = (M_mean / (Z * self.N)).numpy()[0][0]
        self.S_mean = (S_mean/ (Z * self.N) - beta * self.free_energy).numpy()[0][0]
        print("Energy: {0:.3} \nM: {1:.3} \nS: {2:.3}".format(
                        self.E_mean, 
                        self.M_mean,
                        self.S_mean,
                        ))
        print("Free_energy: {0:.3f} ({1:.3f})".format(self.free_energy, 
                                                  self.E_mean - (1./beta)*self.S_mean))
        
        return -(1/beta)* math.log(Z.numpy()[0][0])
    
    def energy(self, samples):
        """
        Compute energy of samples, samples should be of size [m, n] where n is the number of spins, m is the number of samples.
        """
        samples = samples.view(samples.shape[0], -1)
        assert samples.shape[1] == self.N
        m = samples.shape[0]
        inter = -0.5 * ((samples @ self.J_torch).view(m, 1, self.N) @ samples.view(
            m, self.N, 1)).squeeze()
        field = -(samples.view(m, 1, self.N) @ self.H_torch.view(self.N, 1)).squeeze()
        return (inter + field)

    #@jit(nopython=True)
    def exact_numba(self):
        #assert self.N < 28
        J = self.J
        H = self.H
        beta = self.beta
        N = self.N
        E_min = 0
        n_total = int(math.pow(2, self.N))
        Z = 0
        print('Enumerating...')
        E_mean = 0
        M_mean = 0
        S_mean = 0
        for d in range(n_total):
            s = np.binary_repr(d, width=N)
            b = np.array(list(s)).astype(np.float32)
            b[b < 0.5] = -1
            b = b
            E = - 0.5 * b.transpose() @ J @ b - b.transpose() @ H
            if E < E_min:
                E_min = E
            Z_temp = np.exp(-beta * E)
            E_mean += (E * Z_temp)
            M_mean += b.sum() * Z_temp
            Z += Z_temp
            S_mean += -Z_temp * np.log(Z_temp)
            
            if d % 10000 == 0:
                print('\r{} / {}({:.2%}), E = {:.3}, Z = {:.3}, F = {:.3}'.format(
                        d, 
                        n_total,
                        d/n_total,
                        E,
                        Z,
                        -(1/beta) * math.log(Z)))
        print("\r")
        self.free_energy = -(1/beta) * math.log(Z) * (1. / N)
        self.E_mean = (E_mean / (Z * N))
        self.M_mean = (M_mean / (Z * N))
        self.S_mean = (S_mean/ (Z * N) - beta * self.free_energy)
        print("Energy: {0:.3} \nM: {1:.3} \nS: {2:.3}".format(
                        self.E_mean, 
                        self.M_mean,
                        self.S_mean,
                        ))
        print("Free_energy: {0:.3f} ({1:.3f})".format(self.free_energy, 
                                                  self.E_mean - (1./beta)*self.S_mean))
        
        return -(1/beta)* math.log(Z)

spec = [
    ('N', int32),               # a simple scalar field
    ('H', float64[:]),          # an array field
    ('J', float64[:,:]),          # an array field
    ('J_interaction', float64[:,:]),          # an array field
    ("beta", float64),
    ("free_energy", float64),
    ("M_mean", float64),
    ("E_mean", float64),
    ("S_mean", float64),
]

#@jit
def bin_numba(n):
    if n==0: return [0.]
    else:
        temp = [int(n%2)]
        temp.extend(bin_numba(n//2))
        return temp

#@jitclass(spec)
class model_numba(object):
    def __init__(self, N, H, J, J_interaction, beta):
        self.N = N
        self.J = J
        self.H = H
        self.J_interaction = J_interaction
        self.beta = beta
        self.free_energy = 0.
        self.M_mean = 0
        self.S_mean = 0
        assert N == len(H)
        assert J.size == N*N
        
        
    #@jit(nopython=True)
    def exact_numba(self):
        #assert self.N < 28
        J = self.J
        H = self.H
        beta = self.beta
        N = self.N
        E_min = 0
        n_total = int(math.pow(2, self.N))
        Z = 0
        print('Enumerating...')
        E_mean = 0
        M_mean = 0
        S_mean = 0
        
        for i in range(n_total):
            #print(i)
            s = bin_numba(i)
            #ss = s
            #ss[s < 0.5] = -1
            b = -1 * np.ones(N)
            #print(s, len(s))
            for ii in range(len(s)):
                if s[ii] < 0.5:
                    b[ii] = -1
                else:
                    b[ii] = 1
                    
            #ss = np.array(s).astype(np.float64)
            #print(i, s, b)
            #b = s#np.pad(s, (N-len(s),0), "constant", constant_values=(0.))
            
            #print(i, b)
            E = - 0.5 * b.dot(b.dot(J)) - b.dot(H)
            
            if E < E_min:
                E_min = E
            Z_temp = np.exp(-beta * E)
            E_mean += (E * Z_temp)
            M_mean += b.sum() * Z_temp
            Z += Z_temp
            S_mean += -Z_temp * np.log(Z_temp)
            #print(i, n_total, float(i) % 10)
            if i % int(n_total/100) == 0:
                print("\r\r", i, n_total, i/n_total)
                
                #string = str(int(i))
                #+ " / " +  str(n_total) + " " + str(i/n_total)
                #string += ", E = " + str(E)
                #string += ", F = " + str(-(1/beta) * math.log(Z))
                #print(string)

        self.free_energy = - (1/beta) * math.log(Z) * (1./ N)
        self.E_mean = (E_mean / (Z * N))
        self.M_mean = (M_mean / (Z * N))
        self.S_mean = (S_mean/ (Z * N) - beta * self.free_energy)
        print("Energy: ", self.E_mean, " \nM: ",  self.M_mean, " \nS: ",
                        self.S_mean)
        print("Free_energy: ", self.free_energy, " ", self.E_mean - (1./beta)*self.S_mean)
        
        return -(1/beta)* math.log(Z)



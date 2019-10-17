import numpy as np

class BP_solver():
    def __init__(self, model_):
        self.J = model_.J
        self.J_interaction = model_.J_interaction
        self.H = model_.H
        #self.beta = model_.beta
        self.N = model_.N
        
        # matrix of message (represent +1 probability)
        self.M = self.J_interaction.copy().astype(float)
        # random initialization
        self.M = np.random.rand(self.M.shape[0], self.M.shape[1]) * self.M

        #list of neghs
        m_neighs = self.J_interaction != 0
        max_n = 0
        for jj in range(len(m_neighs)):
            max_n = max(sum(m_neighs[jj]), max_n)
        max_n
        list_neighs = np.full((len(m_neighs), max_n), -2)
        num_neighs = np.zeros((len(m_neighs),))
    
        for ii in range(len(m_neighs)):
            index_jj = 0
            for jj in range(len(m_neighs[ii])):
                if m_neighs[ii][jj] != 0:
                    list_neighs[ii][index_jj] = jj
                    index_jj += 1
            num_neighs[ii] = index_jj 
            
        self.neighs = list_neighs
        self.num_neighs = num_neighs
        self.Z_ij_p = self.M.copy()
        self.Z_ij_m = self.M.copy()
        self.Z_i_p = self.H.copy().astype(float)
        self.Z_i_m = self.H.copy().astype(float)
        #self.Z_i = self.H.copy().astype(float)
        self.F_i = 0
        self.F_ij = 0

            
    def factor_ij(self, i, j, s_i, s_j, beta):
        n_neigh_i = self.num_neighs[i]
        n_neigh_j = self.num_neighs[j]
        hi = self.H[i] / n_neigh_i
        hj = self.H[j] / n_neigh_j
        j_v = self.J[i][j]
        val = s_i * hi + s_j * hj + j_v * s_j * s_i
        val *= beta
        factor_temp = np.exp(val)
        return  factor_temp
    
    def BP_update(self, beta):
        old_M = self.M.copy()
        neighs = self.neighs
        factor_ij = self.factor_ij
        for i in range(len(neighs)):
            z_i_p = 1
            z_i_m = 1
            for j in neighs[i][neighs[i] != -2]:
                #print(i, j, self.M[i][j])
                prod_p = 1.
                prod_m = 1.
                for k in neighs[i][neighs[i] != -2]:
                    if k != j:
                        prod_p *= self.M[k][i]
                        prod_m *= (1. - self.M[k][i])
                        
                M_p_p = factor_ij(i, j, 1, 1, beta) * prod_p
                M_m_p = factor_ij(i, j, -1, 1, beta) * prod_m
                M_p_m = factor_ij(i, j, 1, -1, beta) * prod_p
                M_m_m = factor_ij(i, j, -1, -1, beta) * prod_m
                
                self.Z_ij_p[i][j] = prod_p
                self.Z_ij_m[i][j] = prod_m
                
                M_p = M_p_p + M_m_p
                M_m = M_p_m + M_m_m
                
                self.M[i][j] = M_p / (M_p + M_m)
                
                z_i_p *= self.M[j][i]
                z_i_m *= (1. - self.M[j][i])
                #print(self.M[i][j], self.Z_i_p, self.Z_i_m)

            self.Z_i_p[i] = z_i_p
            self.Z_i_m[i] = z_i_m
            #print(self.M)
        return np.sum(np.abs(old_M - self.M))
    
    def free_energy(self, beta, print_=False):
        F_i = 0
        Z_ij_p = self.Z_ij_p
        Z_ij_m = self.Z_ij_m
        F_ij = 0
        neighs = self.neighs
        factor_ij = self.factor_ij

        for i in range(self.N):
            n_neigh = self.num_neighs[i]
            F_i += (1 - n_neigh) * np.log(self.Z_i_p[i] + self.Z_i_m[i])
        
        for i in range(len(neighs)):
            n_neigh_i = self.num_neighs[i]
            z_i_p = 1
            z_i_m = 1
            for j in neighs[i][neighs[i] != -2]:
                if j > i:
                    Z_ij_temp = 0
                    Z_ij_temp += factor_ij(i, j, 1, 1, beta) * Z_ij_p[i][j] * Z_ij_p[j][i]
                    Z_ij_temp += factor_ij(i, j, 1, -1, beta) * Z_ij_p[i][j] * Z_ij_m[j][i]
                    Z_ij_temp += factor_ij(i, j, -1, 1, beta) * Z_ij_m[i][j] * Z_ij_p[j][i]
                    Z_ij_temp += factor_ij(i, j, -1, -1, beta)  * Z_ij_m[i][j] * Z_ij_m[j][i]
                    F_ij += np.log(Z_ij_temp)
                    #print(Z_ij_temp,  factor_ij(i, j, 1, 1) * Z_ij_p[i][j] * Z_ij_p[j][i])
        #print(Z_ij)
        self.Z_ij = np.exp(F_ij)
        self.F =  -1. / beta * (F_ij + F_i) * (1./self.N)
        if print_:
            print("free energy: ", self.F)
        #print("...", self.energy())
        return self.F
    
    def magnetization(self, print_=False):
        M = 0
        H = self.H
        J = self.J
        M_i = np.zeros(self.N)
        for i in range(self.N):
            #num_neigh = len(self.neighs[i][self.neighs[i] != -2])
            M_i_temp = self.Z_i_p[i] - self.Z_i_m[i]
            #M += M_i_temp/(self.Z_i_p[i] + self.Z_i_m[i])
            M_i[i] =  M_i_temp/(self.Z_i_p[i] + self.Z_i_m[i])
        self.M_mean = abs(M_i).mean()
        self.M_i = M_i
        if print_:
            print("M: {0:.3}".format(self.M_mean))
        return self.M_mean
    
    def entropy(self, beta, print_=False):
        S_i = 0
        factor_ij = self.factor_ij

        for i in range(self.N):
            num_neigh = len(self.neighs[i][self.neighs[i] != -2])
            p_i_p = self.Z_i_p[i] / (self.Z_i_p[i] + self.Z_i_m[i])
            p_i_m = self.Z_i_m[i] / (self.Z_i_p[i] + self.Z_i_m[i])
            S_i_temp = p_i_p * np.log(p_i_p) + p_i_m * np.log(p_i_m)
            S_i += (1. - num_neigh) * S_i_temp
            #print(i, num_neigh)
        
        S_ij = 0
        Z_ij_p = self.Z_ij_p
        Z_ij_m = self.Z_ij_m
        neighs = self.neighs
        for i in range(len(neighs)):
            for j in neighs[i][neighs[i] != -2]:
                if j > i:
                    S_ij_temp = 0
                    p_p = factor_ij(i, j, 1, 1, beta) * Z_ij_p[i][j] * Z_ij_p[j][i]
                    p_m = factor_ij(i, j, 1, -1, beta)  * Z_ij_p[i][j] * Z_ij_m[j][i]
                    m_p = factor_ij(i, j, -1, 1, beta)  * Z_ij_m[i][j] * Z_ij_p[j][i]
                    m_m = factor_ij(i, j, -1, -1, beta)  * Z_ij_m[i][j] * Z_ij_m[j][i]
                    norm_ij = p_p + p_m + m_p + m_m
                    S_ij_temp += (p_p / norm_ij) * np.log(p_p / norm_ij)
                    S_ij_temp += (p_m / norm_ij) * np.log(p_m / norm_ij)
                    S_ij_temp += (m_p / norm_ij) * np.log(m_p / norm_ij)
                    S_ij_temp += (m_m / norm_ij) * np.log(m_m / norm_ij)
                    S_ij += S_ij_temp
        
        self.S = - (S_i + S_ij) / self.N
        if print_:
            print("S: {0:.3}".format( self.S))
        return self.S
    
    def energy(self, beta, print_=False):
        E_i = 0
        E_ij = 0
        H = self.H
        J = self.J
        Z_ij_p = self.Z_ij_p
        Z_ij_m = self.Z_ij_m
        neighs = self.neighs
        factor_ij = self.factor_ij

        for i in range(self.N):
            #num_neigh = len(self.neighs[i][self.neighs[i] != -2])
            E_i_temp = - H[i] * self.Z_i_p[i] + H[i] * self.Z_i_m[i]
            E_i += E_i_temp/(self.Z_i_p[i] + self.Z_i_m[i])
        
        for i in range(len(neighs)):
            z_i_p = 1
            z_i_m = 1
            for j in neighs[i][neighs[i] != -2]:
                if j > i:
                    E_ij_temp = 0
                    p_p = factor_ij(i, j, 1, 1, beta) * Z_ij_p[i][j] * Z_ij_p[j][i]
                    p_m = factor_ij(i, j, 1, -1, beta)  * Z_ij_p[i][j] * Z_ij_m[j][i]
                    m_p = factor_ij(i, j, -1, 1, beta)  * Z_ij_m[i][j] * Z_ij_p[j][i]
                    m_m = factor_ij(i, j, -1, -1, beta)  * Z_ij_m[i][j] * Z_ij_m[j][i]
                    E_ij_temp += - J[i][j] * p_p
                    E_ij_temp += J[i][j] * p_m
                    E_ij_temp += J[i][j] * m_p
                    E_ij_temp += - J[i][j] * m_m
                    E_ij_temp /= p_p + p_m + m_p + m_m
                    #print(i, j, Z_ij_p[i][j], Z_ij_p[j][i], Z_ij_m[i][j], Z_ij_m[j][i])
                    assert p_p + p_m + m_p + m_m > 0
                    E_ij += E_ij_temp
        self.E_mean = (E_ij + E_i) / self.N
        if print_:
            print("Energy: {:.3}".format(self.E_mean))
        return self.E_mean

    def corr(self, beta):
        E_ij = 0
        H = self.H
        J = self.J
        Corr = np.zeros(J.shape)
        Z_ij_p = self.Z_ij_p
        Z_ij_m = self.Z_ij_m
        neighs = self.neighs
        factor_ij = self.factor_ij
        self.magnetization(print_=False)
                
        for i in range(len(neighs)):
            z_i_p = 1
            z_i_m = 1
            for j in neighs[i][neighs[i] != -2]:
                #if j > i:
                C_ij_temp = 0
                p_p = factor_ij(i, j, 1, 1, beta) * Z_ij_p[i][j] * Z_ij_p[j][i]
                p_m = factor_ij(i, j, 1, -1, beta)  * Z_ij_p[i][j] * Z_ij_m[j][i]
                m_p = factor_ij(i, j, -1, 1, beta)  * Z_ij_m[i][j] * Z_ij_p[j][i]
                m_m = factor_ij(i, j, -1, -1, beta)  * Z_ij_m[i][j] * Z_ij_m[j][i]
                C_ij_temp = (p_p - p_m - m_p +  m_m) / (p_p + p_m + m_p +  m_m)
                Corr[i][j] = C_ij_temp
                #print(i,j)
                #print("{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} {4:.3f}".format(C_ij_temp, p_p, p_m, m_p,  p_p))
        self.Corr = Corr - np.outer(self.M_i, self.M_i)
        self.Corr_neigh = self.J_interaction * self.Corr
        #print(self.M_i, np.outer(self.M_i, self.M_i))
        return self.Corr

    
    def small_rand(self, val = 1):
        M_small = (np.random.random(self.M.shape) * 2 - 1) * val * self.M
        self.M += M_small
        self.M[self.M > 1] = 1.
        return self.M
    
    def converge(self, beta, error = 1e-10, 
                 max_iter = 1000, 
                 rand_init = True, val_rand=0.1):
        if rand_init:
            self.small_rand(val = val_rand)
        err_temp = error + 1
        iter_ = 0
        while err_temp > error and iter_ < max_iter:
            err_temp = self.BP_update(beta)
            iter_ += 1
            print("\r iter:{1},  err: {0:.3f} free_energy {2:.2f}".format(
                err_temp, iter_,
                self.free_energy(beta, print_=False)), end="")
        print("\r", end="")
        E = self.energy(beta, print_=False)
        M = self.magnetization(print_=False)
        S = self.entropy(beta, print_=False)
        Corr = self.corr(beta)
        fe = self.F
        print("fe: {0:.3f}, ener: {1:.3f}, M: {2:.3f}, iter {iter_}".format(fe, E, M, iter_ = iter_))
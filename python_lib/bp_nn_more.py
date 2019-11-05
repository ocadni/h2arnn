import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

type_default = torch.float64
device = "cpu"



class bp_nn_more(nn.Module):
    def __init__(self, model, bias, espilon = 1e-10):
        super(bp_nn_more, self).__init__()
        self.n = model.N
        self.epsilon = espilon
        self.model = model
        self.J_interaction = torch.from_numpy(model.J_interaction).to(type_default)
        self.bias = bias
        # Force the first x_hat to be 0.5

        self.nn_one = []
        self.num_n = []
        for n in range(self.n):
            num_n_less = sum(self.J_interaction[n][0:n])
            num_n_less = 1 if num_n_less == 0 else int(num_n_less.item())
            self.nn_one.append(nn.Linear(num_n_less,1, bias))
            self.num_n.append(num_n_less)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_hat = torch.zeros(x.shape)
        for n in range(self.n):
            list_n = []
            for n_n_i, n_n in enumerate(self.J_interaction[n][0:n]):
                if n_n == 1:
                    #print(list_n,list_n[cc],x,   cc, n_n_i)
                    list_n.append(n_n_i)
            if list_n != []:
                list_n = torch.tensor(list_n)
                input_x = torch.index_select(x,1,list_n)
            else:
                list_n = torch.tensor([0])
                #print(n, list_n)
                input_x = torch.index_select(x,1,list_n)
                input_x*=0.
                
            #print(input_x, self.nn_one[n])
            x_hat[:,n] = self.nn_one[n](input_x).t()
                                 
        return torch.sigmoid(x_hat)
    
    def prob_sample(self, sample, x_hat):
        with torch.no_grad():
            mask = (sample + 1) / 2
            p_sample = (x_hat ** mask) * ((1 - x_hat) ** (1 - mask))
            p_sample = p_sample.prod(dim=1)
            norm = p_sample.sum()
            p_sample /= norm
            p_sample.to(type_default)
            self.x_hat = x_hat
            self.sample_ = sample
            self.p_sample = p_sample
        return p_sample, norm
    
    
    def sample_unique(self, batch_size):
        sample = torch.zeros(
            [batch_size, self.n])
        for i in range(self.n):
            x_hat = self.forward(sample)
            #print(x_hat)
            sample[:, i] = torch.bernoulli(
                    x_hat[:, i]) * 2 - 1
        with torch.no_grad():
            sample, count = sample.unique(dim=0, return_counts=True)
            x_hat = self.forward(sample)
            p_sample, norm = self.prob_sample(sample, x_hat)
        
        '''if self.z2:
            # Binary random int 0/1
            flip = torch.randint(
                2, [len(sample), 1], dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip'''

        return sample.to(type_default), x_hat.to(type_default)
    

    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, self.n])
        for i in range(self.n):
            x_hat = self.forward(sample)
            #print(x_hat)
            sample[:, i] = torch.bernoulli(
                    x_hat[:, i]) * 2 - 1
            
        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat + self.epsilon) * mask +
                    torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        sample = sample.view(sample.shape[0], -1)
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)
                
        return log_prob
    
    def compute_stat(self, beta, batch_size = 10000, print_=True):
        free_energy_mean = 0
        with torch.no_grad():
            sample, x_hat = self.sample(batch_size)
            log_prob = self.log_prob(sample).double()
            energy = self.model.energy(sample.double())
            loss = log_prob + beta * energy
            
            free_energy_mean = loss.mean() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            entropy_mean = -log_prob.mean() / self.n
            energy_mean = energy.mean() / self.n
            mag = sample.mean(dim=0)
            mag_mean = abs(mag).mean()
            
            self.F = free_energy_mean
            self.M = mag_mean
            self.E = energy_mean
            self.S = entropy_mean
            self.M_i = mag.numpy()
            self.F_std = free_energy_std
            self.Corr = torch.zeros(self.J_interaction.shape)
            for s in sample:
                self.Corr += torch.ger(s,s)
            self.Corr /= batch_size
            self.Corr -= torch.ger(mag,mag)
            self.Corr_neigh = self.J_interaction * self.Corr
            
            if print_:
                print("\nfree_energy: {0:.3f},  std_fe: {1:.5f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f}".format(free_energy_mean.double(),
                                                    free_energy_std,
                                                    mag_mean,
                                                    entropy_mean,
                                                    energy_mean,
                                                        ), end="")

        return free_energy_mean

    def prob_sample_is(self, sample, x_hat, beta):
        eps = self.epsilon
        with torch.no_grad():
            p_sample = self._log_prob(sample, x_hat).to(type_default)

            p_sample_exact = - beta * self.model.energy(sample.double())
            p_sample_res = p_sample_exact - p_sample
            p_sample_res = torch.exp(p_sample_res)
            norm = p_sample_res.sum()
            p_sample_res /= norm
            p_sample_res.to(type_default)
            self.x_hat = x_hat
            self.sample_ = sample
            self.p_sample_is = p_sample_res
        return p_sample_res, norm

    def compute_stat_is(self, beta, batch_size = 10000, print_=True):
        free_energy_mean = 0
        with torch.no_grad():
            sample, x_hat = self.sample(batch_size)
            energy = self.model.energy(sample.double()).to(type_default)
            p_sample, norm_sample = self.prob_sample_is(sample, x_hat, beta)
            p_sample_nn = torch.exp(self._log_prob(sample, x_hat)).to(type_default)
            sampled_prob = torch.exp(self.log_prob(sample).double()) * len(sample)
            
            #print(norm_sample, norm_ext)
            #log_prob = self._log_prob(sample, x_hat).to(type_default)
            #norm_ext = torch.exp(- beta * self.model.energy(sample)).sum()
            log_prob = torch.log(p_sample * sampled_prob).to(type_default)
            #p_sample_loss = 1./len(sample)
            p_sample_loss = p_sample
            loss = p_sample_loss * (log_prob + beta * energy)
            w_sample = torch.diag(p_sample) @ sample.to(type_default)
            free_energy_mean = loss.sum() / self.n / beta
            free_energy_std = loss.std() / beta / self.n
            entropy_mean = - (p_sample * log_prob).sum() / self.n
            energy_mean = (p_sample * energy).sum() / self.n
            mag = w_sample.sum(dim=0) 
            mag_mean = abs(mag).sum() / self.n
            
            self.F = free_energy_mean
            self.M = mag_mean
            self.E = energy_mean
            self.S = entropy_mean
            self.M_i = mag.numpy()
            self.F_std = free_energy_std
            self.Corr = torch.zeros(self.J_interaction.shape, dtype=type_default )
            for s_i, s in enumerate(sample):
                s = s.to(type_default)
                self.Corr += p_sample[s_i] * torch.ger(s,s)
            #self.Corr /= batch_size
            self.Corr -= torch.ger(mag,mag)
            self.Corr_neigh = self.J_interaction.to(type_default) * self.Corr
            
            if print_:
                print("\nfree_energy: {0:.3f},  std_fe: {1:.5f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f}".format(free_energy_mean.double(),
                                                    free_energy_std,
                                                    mag_mean,
                                                    entropy_mean,
                                                    energy_mean,
                                                        ), end="")

        return free_energy_mean
    
    
    def compute_stat_unique(self, beta, batch_size = 10000, print_=True):
        free_energy_mean = 0
        with torch.no_grad():
            sample, p_sample = self.sample_unique(batch_size)
            log_prob = torch.log(p_sample).double()
            energy = self.model.energy(sample.to(torch.float64))
            loss = log_prob + beta * energy
            p_sample = p_sample.to(torch.float64)
            free_energy_mean = (p_sample * loss).sum() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            #free_energy_mean = loss.mean() / self.n
            #free_energy_std = loss.std() / self.n
            entropy_mean = - (p_sample * log_prob).sum() / self.n
            energy_mean = (p_sample * energy).sum() / self.n
            mag = 0
            for i_s, s in enumerate(sample):
                mag += s * p_sample[i_s]
            mag_mean = mag.mean()
            
            self.F = free_energy_mean
            self.M = mag_mean
            self.E = energy_mean
            self.S = entropy_mean
            self.M_i = mag.numpy()
            self.F_std = free_energy_std
            self.Corr = torch.zeros(self.J_interaction.shape, dtype=torch.float64)
            for s in sample:
                self.Corr += torch.ger(s,s)
            self.Corr /= batch_size
            self.Corr -= torch.ger(mag,mag)
            self.Corr_neigh = self.J_interaction.double() * self.Corr
            
            if print_:
                print("\nfree_energy: {0:.3f},  std_fe: {1:.5f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f}".format(free_energy_mean.double(),
                                                    free_energy_std,
                                                    mag_mean,
                                                    entropy_mean,
                                                    energy_mean,
                                                        ), end="")

        return free_energy_mean

    
    def rand_grad(self, prob_zero = 0.5):
        non_z = torch.nonzero(self.net[0].weight.grad)
        for non_z in torch.nonzero(self.net[0].weight.grad):
            if random.random() > prob_zero:
                self.net[0].weight.grad[non_z[0], non_z[1]] = 0
        return self.net[0].weight.grad
    
    
    
    def train(self, lr=1e-3, opt = "adam", beta = 1, max_step = 10000, 
              batch_size = 1000, std_fe_limit = 1e-4,
             batch_mean=10000,
             random_update = False,
             random_zero = 0.5):
        params = []
        for n in range(self.n):
            params.extend(list(self.nn_one[n].parameters()))
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))

        #named_params = list(self.net.named_parameters())
        #print("lr:{0}".format(lr))
        if opt == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr)
        elif opt == 'sgdm':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        elif opt == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.99)
        elif opt == 'adam':
            optimizer = torch.optim.Adam(params, 
                                         lr=lr, 
                                         betas=(0.99, 0.999))
        elif opt == 'adam0.5':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))

        else:
            print("optimizer not found, setted Adam")
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9999))
        self.optimizer = optimizer
        
        optimizer.zero_grad()
        for step in range(0, max_step + 1):
            optimizer.zero_grad()
            with torch.no_grad():
                sample, x_hat = self.sample(batch_size)
            assert not sample.requires_grad
            assert not x_hat.requires_grad
            
            log_prob = self.log_prob(sample).to(type_default)
            
            with torch.no_grad():
                energy = self.model.energy(sample)
                loss =  log_prob + beta * energy
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()
            if random_update:
                self.rand_grad(prob_zero = random_zero)
            optimizer.step()

            free_energy_mean = loss.mean() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            entropy_mean = -log_prob.mean() / self.n
            energy_mean = energy.mean() / self.n
            mag = sample.mean(dim=0)
            mag_mean = abs(mag).mean()
            B1 = (self.nn_one[0].bias.data[0] if self.bias else 0)
            B2 = (self.nn_one[1].bias.data[0] if self.bias else 0)
            print("\r {beta:.2f} {step} fe: {0:.3f} +- {1:.5f} E: {E:.3f}, S: {S:.3f}, M: {2:.3}, B1 = {B1:.3f}".format(
                free_energy_mean.double(),
                                                free_energy_std,
                                                mag_mean,
                                                step=step, beta=beta,
                                                W1 = self.nn_one[1].weight.data[0][0],
                                                W2 = self.nn_one[2].weight.data[0][0],
                                                B1 = B1,
                B2=B2,
                E = energy_mean,
                S = entropy_mean
            ), end="")
            if free_energy_std < std_fe_limit:
                break
                
        return {"free_energy_mean":free_energy_mean,
               "free_energy_std":free_energy_std,
               "entropy_mean":entropy_mean,
               "energy_mean":energy_mean,
               "mag":mag,
                "mag_mean":mag_mean
               }
    
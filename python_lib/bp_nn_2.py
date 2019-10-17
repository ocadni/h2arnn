import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bp_nn
type_default = torch.float64

default_dtype_torch = torch.float64

class bp_nn_2(bp_nn.bp_nn):
    def __init__(self, n, model, bias):
        
        super(bp_nn_2, self).__init__(n, model, bias)
        
        mask_layer1 = []

        for r_i, row in enumerate(self.J_interaction):
            mask_layer1.append([0] * n)
            for c_i, val in enumerate(row):
                if c_i < r_i and val != 0:
                    inter_e = [0] * n
                    inter_e[c_i] = 1
                    #print(r_i,c_i, val, inter_e)
                    mask_layer1.append(inter_e)
        mask_layer1 = np.array(mask_layer1)
        self.mask_layer1 = mask_layer1
        layer1 = bp_nn.myLinear(mask_layer1, bias)
        
        layer2 = nn.Sigmoid()

        mask_out_tensor = []
        count_node = 0
        for r_i, row in enumerate(self.J_interaction):
            inter_e = [0] * mask_layer1.shape[0]
            count_row = 0
            #print(inter_e, count_node)
            inter_e[count_node] = 1
            count_node += 1
            for c_i, val in enumerate(row):
                if c_i < r_i and val != 0:
                    inter_e[count_node] = 1
                    count_node += 1
                    count_row += 1
            inter_e[count_node - count_row - 1] = (1-count_row)
            mask_out_tensor.append(inter_e)
        mask_out_tensor = torch.Tensor(mask_out_tensor)
        mask_out = bp_nn.myLinear(mask_out_tensor, False)
        mask_out.weight = torch.nn.Parameter(mask_out_tensor, requires_grad=False) 
        self.mask_out = mask_out

        layers = [layer1, layer2]
        self.net = nn.Sequential(*layers)
    
    def prob_m(self, sample):
        return self.net.forward(sample)
    
    def forward_m(self, sample):
        prob_marginals = self.prob_m(sample)
        #print(prob_marginals)
        prob_plus = torch.log(prob_marginals)
        prob_minus = torch.log(1. - prob_marginals)
        #print(prob_plus)
        #print(prob_minus)
        prob_plus = self.mask_out.forward(prob_plus)
        prob_minus = self.mask_out.forward(prob_minus)
        prob_plus = torch.exp(prob_plus)
        prob_minus = torch.exp(prob_minus)
        #print(prob_plus)
        #print(prob_minus)
        self.prob_plus = prob_plus
        self.prob_minus = prob_minus
        return prob_plus, prob_minus
        
    def _log_prob(self, sample, x_hat_p, x_hat_m):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat_p + self.epsilon) * mask +
                    torch.log(x_hat_m + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        
        return log_prob

    def log_prob(self, sample):
        sample = sample.view(sample.shape[0], -1)
        x_hat_p, x_hat_m = self.forward_m(sample)
        log_prob = self._log_prob(sample, x_hat_p, x_hat_m)
        
        return log_prob
    

    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, self.n])
        for i in range(self.n):
            x_hat_p, x_hat_m = self.forward_m(sample)
            x_hat = (x_hat_p + 1. - x_hat_m)/2
            #x_hat = x_hat_m
            #print(x_hat)
            sample[:, i] = torch.bernoulli(
                    x_hat[:, i]) * 2 - 1

        return sample, x_hat_p, x_hat_m
    
    def sample_unique(self, batch_size):
        sample, x_hat_p, x_hat_m = self.sample(batch_size)
        sample = sample.unique(dim=0)
        x_hat_p, x_hat_m = self.forward_m(sample)
        return sample, x_hat_p, x_hat_m
    
    def prob_sample(self, sample, x_hat_p, x_hat_m):
        mask = (sample + 1) / 2
        p_sample = (x_hat_p ** mask) * (x_hat_m ** (1 - mask))
        p_sample = p_sample.prod(dim=1)
        norm = p_sample.sum()
        p_sample /= norm
        p_sample.to(type_default)
        self.sample_ = sample
        self.p_sample = p_sample
        return p_sample, norm

    def compute_stat_(self, sample, p_sample, loss, log_prob, beta, print_ = False):
        self.sample_=sample
        self.p_sample_=p_sample
        self.loss_=loss
        with torch.no_grad():
            free_energy_mean = (p_sample * loss).sum() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            entropy_mean = - (p_sample * log_prob).sum() / self.n
            energy_mean = (p_sample * (loss-log_prob)/beta).sum() / self.n
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            
            self.F = free_energy_mean
            self.M = mag_mean
            self.E = energy_mean
            self.S = entropy_mean
            self.M_i = mag.numpy()
            self.F_std = free_energy_std
            
            self.Corr = torch.zeros(self.J_interaction.shape)
            for s in sample:
                self.Corr += torch.ger(s,s)
            self.Corr /= sample.size()[0]
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
    
    def train(self, lr=1e-3, opt = "adam", beta = 1, max_step = 10000, 
              batch_size = 1000, std_fe_limit = 1e-4,
             batch_mean=10000):

        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print('Total number of trainable parameters: {}'.format(nparams))
        named_params = list(self.net.named_parameters())
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
                sample, x_hat_p, x_hat_m = self.sample_unique(batch_size)
                p_sample, n_sample = self.prob_sample(sample, x_hat_p, x_hat_m)
                p_sample = p_sample.to(default_dtype_torch)
            assert not sample.requires_grad

            log_prob = self.log_prob(sample).to(type_default) - torch.log(n_sample)
            
            with torch.no_grad():
                energy = self.model.energy(sample)
                #loss =  log_prob * (1. / beta) + energy
                loss =  log_prob + beta * energy 
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = (p_sample * ((loss - (loss * p_sample).sum()) * log_prob)).sum()
            loss_reinforce.backward()
            optimizer.step()
            with torch.no_grad():
                self.compute_stat_(sample, p_sample, loss, log_prob, beta)

            B1 = (self.net[0].bias.data[0] if self.bias else 0)
            B2 = (self.net[0].bias.data[1] if self.bias else 0)
            print("\r {beta:.2f} {step} fe: {0:.3f} +- {1:.5f}, E: {E:.3f}, S: {S:.3f}, M: {2:.3}, #s: {n_sample} B1: {B1:.3f}".format(
                self.F,
                self.F_std,
                self.M,
                n_sample = len(sample),
                step=step, beta=beta,
                W1 = self.net[0].weight.data[1][0],
                W2 = self.net[0].weight.data[2][0],
                B1 = B1,
                B2=B2,
                E = self.E,
                S = self.S,
            ), end="")
            if self.F_std < std_fe_limit:
                break
            
            
        return step

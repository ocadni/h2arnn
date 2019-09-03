import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class myLayer(nn.Linear):
    def __init__(self, n, J_interaction, bias, diagonal=-1, identity=False):
        super(myLayer, self).__init__( n, n, bias)
        self.n = n

        self.register_buffer('mask', torch.tril(J_interaction, diagonal=diagonal))
        if identity:
            self.mask += torch.eye(self.n)
        
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

class myLinear(nn.Linear):
    def __init__(self, J_interaction, bias):
        super(myLinear, self).__init__(J_interaction.shape[1],
                                       J_interaction.shape[0], 
                                       bias)

        self.register_buffer('mask', torch.Tensor(J_interaction))
        #self.register_buffer('mask_bias', torch.Tensor(bias))
        self.weight.data *= self.mask
        #self.bias.data *= self.mask_bias
        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
        
        
    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


    
default_dtype_torch = torch.float64

class bp_nn(nn.Module):
    def __init__(self, n, model, bias, diagonal=-1, identity=False):
        super(bp_nn, self).__init__()
        self.n = n
        self.epsilon = 1e-10
        self.model = model
        self.J_interaction = torch.from_numpy(model.J_interaction).float()
        layers = []
        layer1 = myLayer(n, self.J_interaction, bias, diagonal=diagonal, identity=identity)
        layers.append(layer1)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.net(x)


    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.n])
        for i in range(self.n):
            x_hat = self.forward(sample)
            #print(x_hat)
            sample[:, :, i] = torch.bernoulli(
                    x_hat[:, :, i]) * 2 - 1
            
        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat + self.epsilon) * mask +
                    torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
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
            mag_mean = mag.mean()
            self.F = free_energy_mean
            self.M = mag_mean
            self.E = energy_mean
            self.S = entropy_mean
            if print_:
                print("\nfree_energy: {0:.3f},  std_fe: {1:.3f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f}".format(free_energy_mean.double(),
                                                    free_energy_std,
                                                    mag_mean,
                                                    entropy_mean,
                                                    energy_mean,
                                                        ), end="")

        return free_energy_mean
    
    def train(self, lr=1e-3, opt = "Adam", beta = 1, max_step = 10000, 
              batch_size = 1000, std_fe_limit = 1e-4,
             batch_mean=10000):
        
        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print('Total number of trainable parameters: {}'.format(nparams))
        named_params = list(self.net.named_parameters())

        if opt == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr)
        elif opt == "Adam":   
            optimizer = torch.optim.Adam(params, lr=lr, 
                                         betas=(0.9, 0.999),
                                        amsgrad=True)
        else:
            print("optimizer not found, setted Adam")
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9999))
        
        optimizer.zero_grad()
        for step in range(0, max_step + 1):
            optimizer.zero_grad()
            with torch.no_grad():
                sample, x_hat = self.sample(batch_size)
            assert not sample.requires_grad
            assert not x_hat.requires_grad

            log_prob = self.log_prob(sample).double()
            with torch.no_grad():
                energy = self.model.energy(sample.double())
                loss = log_prob + beta * energy
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()
            optimizer.step()

            free_energy_mean = loss.mean() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            entropy_mean = -log_prob.mean() / self.n
            energy_mean = energy.mean() / self.n
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            print("\r {beta:.2f} {step} fe: {0:.3f} +- {1:.3f} M: {2:.3}".format(
                free_energy_mean.double(),
                                                free_energy_std,
                                                mag_mean,
                                                step=step, beta=beta), end="")
            if free_energy_std < std_fe_limit:
                break
        
        with torch.no_grad():
            sample, x_hat = self.sample(batch_mean)
        log_prob = self.log_prob(sample).double()
        with torch.no_grad():
            energy = self.model.energy(sample.double())
            loss = log_prob + beta * energy

        free_energy_mean = loss.mean() / beta / self.n
        free_energy_std = loss.std() / beta / self.n
        entropy_mean = -log_prob.mean() / self.n
        energy_mean = energy.mean() / self.n
        mag = sample.mean(dim=0)
        mag_mean = mag.mean()
        
        self.F_mean = free_energy_mean
        self.M_mean = mag_mean
        self.E_mean = energy_mean
        return {"free_energy_mean":free_energy_mean,
               "free_energy_std":free_energy_std,
               "entropy_mean":entropy_mean,
               "energy_mean":energy_mean,
               "mag":mag,
                "mag_mean":mag_mean
               }
    

class bp_nn2(bp_nn):
    def __init__(self, n, model, bias):
        super(bp_nn2, self).__init__(n, model, bias)
        num_edges = int(self.J_interaction.sum()/2)
        nodes_edges = []
        bias_layer1 = []
        for r_i, row in enumerate(self.J_interaction):
            for c_i, val in enumerate(row):
                if c_i < r_i and val != 0:
                    inter_e = [0] * n
                    inter_e[c_i] = 1
                    #print(r_i,c_i, val, inter_e)
                    nodes_edges.append(inter_e)
                    bias_layer1.append(1)
        nodes_edges = np.array(nodes_edges)
        layer1 = myLinear(nodes_edges, bias)
        layer2 = myLinear(np.transpose(nodes_edges), bias)
        layers = [layer1, layer2, nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

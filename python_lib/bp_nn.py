import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class myLayer(nn.Linear):
    def __init__(self, in_channels, out_channels, n, J_interaction, bias):
        super(myLayer, self).__init__(in_channels * n, out_channels * n,
                                            bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.register_buffer('mask', torch.tril(J_interaction, diagonal=-1))

        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

default_dtype_torch = torch.float64

class bp_nn(nn.Module):
    def __init__(self, in_channels, out_channels, n, model, bias):
        super(bp_nn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.epsilon = 1e-10
        self.model = model
        self.J_interaction = torch.from_numpy(model.J_interaction).float()
        layers = []
        layer1 = myLayer(in_channels, out_channels, n, self.J_interaction, bias)
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

    def train(self, lr=1e-3, opt = "Adam", beta = 1, max_step = 10000, 
              batch_size = 1000, std_fe_limit = 1e-4,
             batch_mean=10000):
        
        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        print('Total number of trainable parameters: {}'.format(nparams))
        named_params = list(self.net.named_parameters())

        if opt == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr)
        elif opt == "Adam":   
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        else:
            print("optimizer not found, setted Adam")
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        
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
            print("\r {0:.3f} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5}".format(free_energy_mean.double(),
                                                free_energy_std,
                                                mag_mean,
                                                entropy_mean,
                                                energy_mean,
                                                self.net[0].weight.data[0][1]
                                                    ), end="")
            if free_energy_std < std_fe_limit:
                break
        
        with torch.no_grad():
            sample, x_hat = self.sample(batch_mean)
        log_prob = self.log_prob(sample).double()
        with torch.no_grad():
            energy = self.model.energy(sample.double())
            loss = log_prob + beta * energy
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        free_energy_mean = loss.mean() / beta / self.n
        free_energy_std = loss.std() / beta / self.n
        entropy_mean = -log_prob.mean() / self.n
        energy_mean = energy.mean() / self.n
        mag = sample.mean(dim=0)
        mag_mean = mag.mean()

        print("\nfree_energy: {0:.3f},  std_fe: {1:.3f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f} weight: {5:.2f}".format(free_energy_mean.double(),
                                                free_energy_std,
                                                mag_mean,
                                                entropy_mean,
                                                energy_mean,
                                                self.net[0].weight.data[0][1]
                                                    ), end="")
        return {"free_energy_mean":free_energy_mean,
               "free_energy_std":free_energy_std,
               "entropy_mean":entropy_mean,
               "energy_mean":energy_mean,
               "mag":mag,
                "mag_mean":mag_mean
               }
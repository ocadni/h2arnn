import torch
import torch.nn as nn
import torch.nn.functional as F

class myLayer(nn.Linear):
    def __init__(self, in_channels, out_channels, n, J_interaction, bias):
        super(myLayer, self).__init__(in_channels * n, out_channels * n,
                                            bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.register_buffer('mask', J_interaction)

        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

default_dtype_torch = torch.float64

class bp_nn(nn.Module):
    def __init__(self, in_channels, out_channels, n, J_interaction, bias):
        super(bp_nn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.epsilon = 1e-10
        layers = []
        layer1 = myLayer(in_channels, out_channels, n, J_interaction, bias)
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

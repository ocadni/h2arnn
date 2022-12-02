import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import init


class zero_linear(nn.Module):
    def __init__(self, out_features, bias):
        super(zero_linear, self).__init__()
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.has_bias = True
        else:
            self.bias = torch.ones(out_features) / 2
            self.has_bias = False
        self.reset_parameters()

    def reset_parameters(self):
        if self.has_bias:
            init.uniform_(self.bias, -1, 1)

    def forward(self, input):
        return self.bias

    def extra_repr(self):
        return 'in_features=0, out_features={}, bias={}'.format(self.out_features, self.bias is not None
                                                                )


def my_linear(in_feat, out_feat, bias):
    if in_feat > 0:
        return nn.Linear(in_feat, out_feat, bias)
    else:
        return zero_linear(out_feat, bias)


class rsb_layer_shared(nn.Module):
    def __init__(self, Jqp, k=0, bias=True, J_learn=False, in_func=F.logsigmoid()):
        super(deep_linear, self).__init__()
        if k < 0:
            raise TypeError("k<0, k must be larger or equals to zero!")
        layers = []
        self.Js = Js
        self.in_func = in_func
        if J_learn:
            self.Js = my_linear(Jqp.shape[0], Jqp.shape[1], bias)
        layers.append(my_linear(1, 1, bias))
        for kk in range(k):
            layers.append(my_linear(1, 1, bias))

    def forward(self, x):
        num_samples = x.shape[0]
        x = self.beta * Js * x
        for ll in layers:
            x = ll(x.view(-1))
            x = self.in_func(x)
        x = 2*self.beta * +
        return x.view(num_samples, -1)


def compute_stats(samples, loss, log_prob, energy, beta, model, ifprint=True):
    loss = loss.cpu().detach().numpy()
    log_prob = log_prob.cpu().detach().numpy()
    energy = energy.cpu().detach().numpy()
    N = model.N
    free_energy_mean = loss.mean() / (beta * N)
    free_energy_std = loss.std() / (beta * N)
    entropy_mean = -log_prob.mean() / N
    energy_mean = energy.mean() / N
    mag = samples.mean(dim=0).cpu().detach().numpy()
    mag_mean = abs(mag).mean()
    if ifprint:
        print(
            f"\r {beta:.2f} fe: {free_energy_mean:.3f} +- {free_energy_std:.5f} E: {energy_mean:.3f}, S: {entropy_mean:.3f}, M: {mag_mean:.3}",
            end="",
        )

    return {
        "beta": beta,
        "free_energy_mean": free_energy_mean,
        "free_energy_std": free_energy_std,
        "entropy_mean": entropy_mean,
        "energy_mean": energy_mean,
        "mag": mag,
        "mag_mean": mag_mean,
    }


class myLayer(nn.Linear):
    def __init__(self, N, J_interaction, bias, device, dtype):
        super(myLayer, self).__init__(N, N, bias, device=device, dtype=dtype)
        self.N = N

        self.register_buffer("mask", torch.tril(J_interaction, diagonal=-1))
        self.weight.data *= self.mask
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
        if bias:
            self.bias.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class simple_layer(nn.Module):
    def __init__(
        self,
        N,
        model,
        bias,
        espilon=1e-10,
        dtype=torch.float32,
        device="cpu",
    ):
        super(simple_layer, self).__init__()
        self.N = N
        self.epsilon = espilon
        self.model = model
        self.J_interaction = model.J_interaction.to(dtype=dtype, device=device)
        self.bias = bias
        self.device = device
        self.dtype = dtype
        layers = []
        layer1 = myLayer(
            self.N,
            self.J_interaction,
            bias,
            device=device,
            dtype=dtype
        )
        layers.append(layer1)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        print(f'Total number of trainable parameters: {nparams}')
        self.params = params

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_hat = self.net(x)
        return x_hat

    def prob_sample_normed(self, sample, x_hat):
        with torch.no_grad():
            mask = (sample + 1) / 2
            p_sample = (x_hat**mask) * ((1 - x_hat) ** (1 - mask))
            p_sample = p_sample.prod(dim=1)
            norm = p_sample.sum()
            p_sample /= norm
            p_sample.to(type_default)
            self.x_hat = x_hat
            self.sample_ = sample
            self.p_sample = p_sample
        return p_sample, norm

    def sample(self, batch_size):
        sample = torch.zeros([batch_size, self.N],
                             device=self.device, dtype=self.dtype)
        for i in range(self.N):
            x_hat = self.forward(sample)
            # print(x_hat)
            sample[:, i] = torch.bernoulli(x_hat[:, i]) * 2 - 1
        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(
            1 - x_hat + self.epsilon
        ) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        sample = sample.view(sample.shape[0], -1)
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)
        return log_prob

    def compute_stats(self, beta, batch_size=10000, print_=True):
        with torch.no_grad():
            samples, x_hat = self.sample(batch_size)
            log_prob = self._log_prob(samples, x_hat)
            energy = self.model.energy(samples)
            loss = log_prob + beta * energy
            stats = compute_stats(samples, loss, log_prob,
                                  energy, beta, self.model, ifprint=print_)
        return stats

    def prob_sample_is(self, sample, x_hat, beta):
        eps = self.epsilon
        with torch.no_grad():
            p_sample = self._log_prob(sample, x_hat).to(type_default)

            p_sample_exact = -beta * self.model.energy(sample.double())
            p_sample_res = p_sample_exact - p_sample
            p_sample_res = torch.exp(p_sample_res)
            norm = p_sample_res.sum()
            p_sample_res /= norm
            p_sample_res.to(type_default)
            self.x_hat = x_hat
            self.sample_ = sample
            self.p_sample_is = p_sample_res
        return p_sample_res, norm

    def train(
        self,
        lr=1e-3,
        opt="adam",
        beta=1,
        max_step=10000,
        batch_size=1000,
        std_fe_limit=1e-4,
        batch_mean=10000,
        ifprint=True
    ):
        params = self.params
        if opt == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr)
        elif opt == "sgdm":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        elif opt == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.99)
        elif opt == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.99, 0.999))
        elif opt == "adam0.5":
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        else:
            print("optimizer not found, setted Adam")
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9999))
        self.optimizer = optimizer

        optimizer.zero_grad()

        for step in range(0, max_step + 1):
            optimizer.zero_grad()
            with torch.no_grad():
                samples, x_hat = self.sample(batch_size)

            log_prob = self.log_prob(samples)

            with torch.no_grad():
                energy = self.model.energy(samples)
                # loss =  log_prob * (1. / beta) + energy
                loss = log_prob + beta * energy

            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()
            optimizer.step()

            B1 = self.net[0].bias.data[0] if self.bias else 0
            B2 = self.net[0].bias.data[1] if self.bias else 0
            stats = compute_stats(samples, loss, log_prob, energy,
                                  beta, self.model, ifprint=ifprint)
            if stats["free_energy_std"] < std_fe_limit:
                break

        return stats

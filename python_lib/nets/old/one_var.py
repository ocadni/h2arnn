import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


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


class one_J_layer(nn.Module):
    def __init__(self, model, device, dtype):
        super(one_J_layer, self).__init__()
        self.weight.data = torch.zeros(model.N).to(device=device, dtype=dtype)
        self.weight.data.requires_grad_()
        self.mask = torch.tril(
            model.J_interaction, diagonal=-1).to(device=device, dtype=dtype)

    def forward(self, x):
        res = nn.functional.linear(x, self.weight.unsqueeze(-1)*self.mask)
        return F.sigmoid(res)


class one_var(nn.Module):
    def __init__(
        self,
        N,
        model,
        bias,
        espilon=1e-10,
        dtype=torch.float32,
        device="cpu",
    ):
        super(one_var, self).__init__()
        self.N = N
        self.epsilon = espilon
        self.model = model
        self.J_interaction = model.J_interaction.to(dtype=dtype, device=device)
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.net = one_J_layer(
            model,
            device=device,
            dtype=dtype
        )
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

            stats = compute_stats(samples, loss, log_prob, energy,
                                  beta, self.model, ifprint=ifprint)
            if stats["free_energy_std"] < std_fe_limit:
                break

        return stats

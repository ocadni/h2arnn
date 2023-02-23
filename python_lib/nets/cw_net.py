import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_lib.nets.ann import ANN
import random
from scipy.special import comb


class one_var_(nn.Module):
    def __init__(self, N, device, dtype):
        super(one_var_, self).__init__()
        self.N = N
        self.dtype = dtype
        self.device = device
        one_vars = torch.zeros((1, N), device=device, dtype=dtype)
        self.one_vars = nn.Parameter(one_vars)
        torch.nn.init.normal_(self.one_vars, mean=0.0, std=1/N)

    def forward(self, x):
        m = x.shape[0]
        res = torch.zeros((m, self.N),
                          device=self.device, dtype=self.dtype)
        torch.cumsum(x[:, :-1], dim=1, out=res[:, 1:])
        return torch.sigmoid(self.one_vars * res)


class one_var(ANN):
    """
    TODO ass bias for dealing with external field 
    """

    def __init__(
        self,
        model,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"bias": False},
    ):

        net = one_var_(model.N, device=device, dtype=dtype)
        super(one_var, self).__init__(
            model, net, dtype=dtype, device=device, eps=eps)

    def sample(self, batch_size):
        x = torch.zeros([batch_size, self.N],
                        device=self.device, dtype=self.dtype)
        x_hat = torch.zeros([batch_size, self.N],
                            device=self.device, dtype=self.dtype)
        res = torch.zeros((batch_size, self.N),
                          device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for n_i in range(self.N):
                if n_i > 0:
                    res[:, n_i] = res[:, n_i-1] + x[:, n_i - 1]
                x_hat[:, n_i] = torch.sigmoid(
                    self.net.one_vars[0, n_i] * res[:, n_i])
                x[:, n_i] = torch.bernoulli(x_hat[:, n_i]) * 2 - 1
        return x, x_hat


class one_var_sign_(nn.Module):
    def __init__(self, N, device, dtype):
        super(one_var_sign_, self).__init__()
        self.N = N
        self.dtype = dtype
        self.device = device
        one_vars = torch.zeros((1, N), device=device, dtype=dtype)
        shared_vars = torch.zeros((1), device=device, dtype=dtype)
        self.one_vars = nn.Parameter(one_vars)
        self.shared_var = nn.Parameter(shared_vars)
        torch.nn.init.normal_(self.one_vars, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.shared_var, mean=0.0, std=1/N)

    def forward(self, x):
        m = x.shape[0]
        res = torch.zeros((m, self.N),
                          device=self.device, dtype=self.dtype)
        torch.cumsum(x[:, :-1], dim=1, out=res[:, 1:])
        x1 = self.one_vars*torch.sign(res)
        x2 = self.shared_var * res
        return torch.sigmoid(x2 + x1)


class one_var_sign(ANN):
    """
    TODO ass bias for dealing with external field 
    """

    def __init__(
        self,
        model,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"bias": False},
    ):

        net = one_var_sign_(model.N, device=device, dtype=dtype)
        super(one_var_sign, self).__init__(
            model, net, dtype=dtype, device=device, eps=eps)

    def sample(self, batch_size):
        x = torch.zeros([batch_size, self.N],
                        device=self.device, dtype=self.dtype)
        x_hat = torch.zeros([batch_size, self.N],
                            device=self.device, dtype=self.dtype)
        res = torch.zeros((batch_size, self.N),
                          device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for n_i in range(self.N):
                if n_i > 0:
                    res[:, n_i] = res[:, n_i-1] + x[:, n_i - 1]
                x1 = self.net.one_vars[0, n_i]*torch.sign(res[:, n_i])
                x2 = self.net.shared_var * res[:, n_i]
                x_hat[:, n_i] = torch.sigmoid(x1 + x2)
                x[:, n_i] = torch.bernoulli(x_hat[:, n_i]) * 2 - 1
        return x, x_hat

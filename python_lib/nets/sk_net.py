import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_lib.nets.ann import ANN
import random
from scipy.special import comb


class SK_net_krsb_(nn.Module):
    """Test for SK net with krsb, i.e. the one variable net with krsb.

    """

    def __init__(self, J, device, dtype, k=0, learn_first_l=False):
        """Initialize the SK net with krsb. 
        J is the coupling matrix. 
        learn_first_l is whether to learn the first layer.
        dtype and device are used to initialize the parameters. 
        k is the depth of nets (KRSB)."""

        super(SK_net_krsb_, self).__init__()
        self.N = J.shape[0]
        self.dtype = dtype
        self.device = device
        self.J = J.clone()
        self.k = k
        N = self.N

        weight_p = torch.zeros((self.k+2, N, N), device=device, dtype=dtype)
        weight_m = torch.zeros((self.k+2,  N, N), device=device, dtype=dtype)
        bias_p = torch.zeros((self.k+1,  N, N), device=device, dtype=dtype)
        bias_m = torch.zeros((self.k+1,  N, N), device=device, dtype=dtype)
        weight_0 = torch.zeros((1, N), device=device, dtype=dtype)
        bias_0 = torch.zeros((1, N), device=device, dtype=dtype)

        self.weight_p = nn.Parameter(weight_p)
        self.bias_p = nn.Parameter(bias_p)
        self.weight_m = nn.Parameter(weight_m)
        self.bias_m = nn.Parameter(bias_m)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_p, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_p, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.weight_m, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_m, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

        self.register_buffer("mask", torch.tril(torch.ones(
            (N, N), device=device, dtype=dtype), diagonal=-1))

    def forward(self, x):
        m = x.shape[0]
        assert (x.shape[1] == self.N)
        x = torch.unsqueeze(x, dim=1)
        res = x * self.J
        res = torch.cumsum(res, dim=1)
        res_i = torch.diagonal(res, dim1=1, dim2=2)
        res_all = self.mask * res
        res_p = F.logsigmoid(self.bias_p[0] + self.weight_p[0] * res_all)
        res_m = F.logsigmoid(self.bias_m[0] + self.weight_m[0] * res_all)
        for kk in range(1, self.k+1):
            res_p = F.logsigmoid(self.bias_p[kk] + self.weight_p[kk] * res_p)
            res_m = F.logsigmoid(self.bias_m[kk] + self.weight_m[kk] * res_m)

        res_p = self.weight_p[-1] * res_p
        res_m = self.weight_m[-1] * res_m
        res_i = self.bias_0 + self.weight_0 * res_i
        return torch.sigmoid(res_i + res_p.sum(dim=-1) + res_m.sum(dim=-1))


class SK_net_krsb(ANN):
    """Test for SK net with krsb, i.e. the one variable net with krsb.

    """

    def __init__(
        self,
        model,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"k": 0},
    ):
        """Initialize the SK net with krsb. model is the model. device is pytorch device. dtype is pytorch dtype."""
        self.k = dict_nets["k"]
        net = SK_net_krsb_(model.J, device=device, dtype=dtype, k=self.k)
        super(SK_net_krsb, self).__init__(
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
                    new_res = torch.unsqueeze(
                        x, dim=1)[:, :, n_i-1] * self.net.J[:, n_i-1]
                    res += new_res
                res_i = res[:, n_i]
                res_all = self.net.mask[:, n_i] * res
                res_p = F.logsigmoid(
                    self.net.bias_p[0, :, n_i] + self.net.weight_p[0, :, n_i] * res_all)
                res_m = F.logsigmoid(
                    self.net.bias_m[0, :, n_i] + self.net.weight_m[0, :, n_i] * res_all)
                for kk in range(1, self.k+1):
                    res_p = F.logsigmoid(
                        self.net.bias_p[kk, :, n_i] + self.net.weight_p[kk, :, n_i] * res_p)
                    res_m = F.logsigmoid(
                        self.net.bias_m[kk, :, n_i] + self.net.weight_m[kk, :, n_i] * res_m)

                res_p = self.net.weight_p[-1, :, n_i] * res_p
                res_m = self.net.weight_m[-1, :, n_i] * res_m
                res_i = self.net.bias_0[0, n_i] + \
                    self.net.weight_0[0, n_i] * res_i

                x_hat[:, n_i] = torch.sigmoid(
                    res_i + res_p.sum(dim=-1) + res_m.sum(dim=-1))
                x[:, n_i] = torch.bernoulli(x_hat[:, n_i]) * 2 - 1
        return x, x_hat


class SK_net_krsb_not_sample(ANN):
    """Test for SK net with krsb, i.e. the one variable net with krsb.

    """

    def __init__(
        self,
        model,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"k": 0},
    ):
        """Initialize the SK net with krsb. model is the model. device is pytorch device. dtype is pytorch dtype."""
        self.k = dict_nets["k"]
        net = SK_net_krsb_(model.J, device=device, dtype=dtype, k=self.k)
        super(SK_net_krsb_not_sample, self).__init__(
            model, net, dtype=dtype, device=device, eps=eps)

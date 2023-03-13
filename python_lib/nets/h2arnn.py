import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_lib.nets.ann import ANN
from scipy.special import comb


class SK_krsb(nn.Module):

    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"k": 0},
                 dtype=torch.float32,
                 device="cpu",
                 ):
        super().__init__()
        N = model.N
        N_i = N - n_i - 1
        self.n_i = n_i
        self.N_i = N_i
        self.k = dict_nets["k"] if "k" in dict_nets else 0
        if N_i > 0:
            # K layers
            weight_p = torch.zeros((self.k+2, N_i), device=device, dtype=dtype)
            weight_m = torch.zeros((self.k+2, N_i), device=device, dtype=dtype)
            bias_p = torch.zeros((self.k+1, N_i), device=device, dtype=dtype)
            bias_m = torch.zeros((self.k+1, N_i), device=device, dtype=dtype)

            # nn.Parameter is a Tensor that's a module parameter.
            self.weight_p = nn.Parameter(weight_p)
            self.bias_p = nn.Parameter(bias_p)
            self.weight_m = nn.Parameter(weight_m)
            self.bias_m = nn.Parameter(bias_m)

            # initialize weights and biases
            torch.nn.init.normal_(self.weight_p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_m, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_m, mean=0.0, std=1/N)
        else:
            self.k = 0
            self.weight_p = torch.zeros((2, 1), device=device, dtype=dtype)
            self.weight_m = torch.zeros((2, 1), device=device, dtype=dtype)
            self.bias_p = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_m = torch.zeros((1, 1), device=device, dtype=dtype)

        weight_0 = torch.zeros((1), device=device, dtype=dtype)
        bias_0 = torch.zeros((1), device=device, dtype=dtype)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

    def forward(self, m):
        m_n_i = m[:, 1:]
        m_i = m[:, 0]
        res_p = F.logsigmoid(self.bias_p[0] + self.weight_p[0] * m_n_i)
        res_m = F.logsigmoid(self.bias_m[0] + self.weight_m[0] * m_n_i)
        for kk in range(1, self.k+1):
            res_p = F.logsigmoid(self.bias_p[kk] + self.weight_p[kk] * res_p)
            res_m = F.logsigmoid(self.bias_m[kk] + self.weight_m[kk] * res_m)

        res_p = self.weight_p[-1] * res_p
        res_m = self.weight_m[-1] * res_m
        res_0 = self.bias_0 + self.weight_0 * m_i

        return torch.sigmoid(res_0 + res_p.sum(dim=1) + res_m.sum(dim=1))

    def set_params_exact(self, model, beta):
        return 1


class h2arnn(ANN):
    def __init__(
        self,
        model,
        single_net,
        input_mask,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"set_exact": False},
        learn_first_l=False
    ):

        net = []

        for n_i in range(model.N):
            net.append(single_net(model, n_i, device=device,
                       dtype=dtype, dict_nets=dict_nets))

        super(h2arnn, self).__init__(
            model, net, dtype=dtype, device=device, eps=eps, print_num_params=False)

        self.input_mask = input_mask.to(dtype=torch.bool)

        if dict_nets["set_exact"]:
            self.set_params_exact(model, 0.1)

        self.J = model.J.clone()

        self.learn_first_l = learn_first_l
        if self.learn_first_l:
            self.J = nn.Parameter(self.J)

        self.print_num_params(train=True)
        self.print_num_params(train=False)

    def set_params_first_l(self, model):
        if self.learn_first_l:
            self.J = nn.Parameter(model.J.clone())
        else:
            self.J = model.J.clone()

    def first_l(self, x, n_i):
        '''consider populated the lower triangular values of the matrix J'''
        return F.linear(x, self.J[n_i:, 0:n_i])

    def parameters(self):
        params = []
        for n_i in range(self.N):
            params.append({"params": self.net[n_i].parameters()})
        if self.learn_first_l:
            params.append({"params": self.J})
        return params

    def num_params(self, train=True):
        num_params = 0
        for n_i in range(self.N):
            n_temp = 0
            params = self.net[n_i].parameters()
            if train:
                n_temp = sum(p.numel() for p in params if p.requires_grad)
            else:
                n_temp = sum(p.numel() for p in params)
            num_params += n_temp
        if self.learn_first_l:
            num_params += self.J.numel()

        return num_params

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_hat = torch.zeros(x.shape,
                            device=self.device, dtype=self.dtype)
        for n_i in range(0, self.N):
            mask_n_i = self.input_mask[n_i, :]
            input_x = x[:, mask_n_i]
            input_x = self.first_l(input_x, n_i)
            x_hat[:, n_i] = self.net[n_i](input_x).t()
        return x_hat

    def sample(self, batch_size):
        x = torch.zeros([batch_size, self.N],
                        device=self.device, dtype=self.dtype)
        x_hat = torch.zeros([batch_size, self.N],
                            device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for n_i in range(self.N):
                mask_n_i = self.input_mask[n_i, :]
                input_x = x[:, mask_n_i]
                input_x = self.first_l(input_x, n_i)
                x_hat[:, n_i] = self.net[n_i](input_x).t()
                x[:, n_i] = torch.bernoulli(x_hat[:, n_i]) * 2 - 1
        return x, x_hat

    def set_params_exact(self, model, beta):
        for nn in self.net:
            nn.set_params_exact(model, beta)

    def train(
        self,
        lr=1e-3,
        opt="adam",
        beta=1,
        max_step=10000,
        batch_size=1000,
        std_fe_limit=1e-4,
        batch_iter=1,
        ifprint=True,
        exact=False,
        set_optim=True
    ):
        if not exact:
            return super().train(lr=lr,
                                 opt=opt,
                                 beta=beta,
                                 max_step=max_step,
                                 batch_size=batch_size,
                                 std_fe_limit=std_fe_limit,
                                 batch_iter=batch_iter,
                                 ifprint=ifprint,
                                 set_optim=set_optim)

        else:
            self.set_params_exact(self.model, beta)
            # print("EXACT!!")
            stats = self.compute_stats(
                beta, batch_size, print_=ifprint, batch_iter=batch_iter)
        return stats

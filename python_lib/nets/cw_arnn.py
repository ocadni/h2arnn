import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_lib.nets.ann import ANN
import random
from scipy.special import comb


class oneP_(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, N, device, dtype):
        """_summary_"""
        super(oneP_, self).__init__()
        self.N = N
        self.dtype = dtype
        self.device = device
        one_vars = torch.zeros((1, N), device=device, dtype=dtype)
        self.one_vars = nn.Parameter(one_vars)
        torch.nn.init.normal_(self.one_vars, mean=0.0, std=1/N)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = x.shape[0]
        res = torch.zeros((m, self.N),
                          device=self.device, dtype=self.dtype)
        torch.cumsum(x[:, :-1], dim=1, out=res[:, 1:])
        return torch.sigmoid(self.one_vars * res)


class oneP(ANN):
    """_summary_"""
    def __init__(
        self,
        model,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"bias": False},
    ):
        """_summary_

        Args:
            model (_type_): _description_
            dtype (_type_, optional): _description_. Defaults to torch.float32.
            device (str, optional): _description_. Defaults to "cpu".
            eps (_type_, optional): _description_. Defaults to 1e-10.
            dict_nets (dict, optional): _description_. Defaults to {"bias": False}.
        """
        net = oneP_(model.N, device=device, dtype=dtype)
        super(oneP, self).__init__(
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


class CWARNN_inf_(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, N, device, dtype):
        """_summary_"""
        super(CWARNN_inf_, self).__init__()
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


class CWARNN_inf(ANN):
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

        net = CWARNN_inf_(model.N, device=device, dtype=dtype)
        super(CWARNN_inf, self).__init__(
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


class CWARNN_(nn.Module):
    def __init__(self,
                 model,
                 n_i,
                 dict_nets={},
                 dtype=torch.float32,
                 device="cpu",
                 ):
        super().__init__()
        N = model.N
        N_i = N - n_i - 1
        self.n_i = n_i
        self.N_i = N_i
        if N_i > 0:
            weight_p = torch.zeros((1, N_i+1), device=device, dtype=dtype)
            weight_m = torch.zeros((1, N_i+1), device=device, dtype=dtype)
            bias_p = torch.zeros((1, N_i+1), device=device, dtype=dtype)
            bias_m = torch.zeros((1, N_i+1), device=device, dtype=dtype)
            weight_0p = torch.zeros((1), device=device, dtype=dtype)
            weight_0m = torch.zeros((1), device=device, dtype=dtype)
            # nn.Parameter is a Tensor that's a module parameter.
            self.weight_p = nn.Parameter(weight_p)
            self.bias_p = nn.Parameter(bias_p)
            self.weight_m = nn.Parameter(weight_m)
            self.bias_m = nn.Parameter(bias_m)
            self.weight_0p = nn.Parameter(weight_0p)
            self.weight_0m = nn.Parameter(weight_0m)

            # initialize weights and biases
            torch.nn.init.normal_(self.weight_0p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_0m, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_m, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_m, mean=0.0, std=1/N)
        else:
            self.weight_p = torch.zeros((1, 1), device=device, dtype=dtype)
            self.weight_m = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_p = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_m = torch.zeros((1, 1), device=device, dtype=dtype)
            self.weight_0p = torch.zeros((1), device=device, dtype=dtype)
            self.weight_0m = torch.zeros((1), device=device, dtype=dtype)

        weight_0 = torch.zeros((1), device=device, dtype=dtype)
        bias_0 = torch.zeros((1), device=device, dtype=dtype)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

    def forward(self, x):
        m_i = x.sum(-1)
        res_p = self.bias_p + self.weight_p * torch.unsqueeze(m_i, dim=1)
        res_p = self.weight_0p * torch.logsumexp(res_p, 1)
        res_m = self.bias_m + self.weight_m * torch.unsqueeze(m_i, dim=1)
        res_m = self.weight_0m * torch.logsumexp(res_m, 1)
        res_0 = self.bias_0 + self.weight_0 * m_i
        return torch.sigmoid(res_0 + res_p + res_m)

    def set_params_exact(self, model, beta):
        for p in self.parameters():
            p.requires_grad_(False)
        J_N = model.J[0][1] * 2
        h = model.H[0]
        N_i = self.N_i
        n_i = self.n_i
        N = N_i + n_i + 1
        #print(f"n_i={n_i}, N_i={N_i}, N={N}, beta={beta:.2f}, J_2N:{J_2N:.3} ")
        # if n_i > 0:
        assert (N == model.N)
        JJ = beta * J_N

        for k in range(0, N_i+1):
            m = N_i - 2*k
            #print(m, comb(N_i, k))
            b_ = np.log(comb(N_i, k)) + (JJ/2)*m**2 + beta*h*m
            b_p = b_ + JJ*m
            b_m = b_ - JJ*m
            w_ = JJ*m
            self.weight_p[0][k] = w_
            self.weight_m[0][k] = w_
            self.bias_p[0][k] = b_p
            self.bias_m[0][k] = b_m

        self.weight_0[0] = 2 * JJ
        self.bias_0[0] = 2 * beta * h
        self.weight_0p[0] = + 1
        self.weight_0m[0] = - 1


class CWARNN(ANN):
    def __init__(
        self,
        model,
        input_mask,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"set_exact": False},
    ):
        net = []
        for n_i in range(model.N):
            net.append(CWARNN_(model, n_i, device=device,
                       dtype=dtype, dict_nets=dict_nets))
        super(CWARNN, self).__init__(
            model, net, dtype=dtype, device=device, eps=eps, print_num_params=False)
        self.input_mask = input_mask.to(dtype=torch.bool)
        if dict_nets["set_exact"]:
            self.set_params_exact(model, 0.1)
        self.print_num_params(train=True)
        self.print_num_params(train=False)

    def parameters(self):
        params = []
        for n_i in range(self.N):
            params.append({"params": self.net[n_i].parameters()})
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
        return num_params

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_hat = torch.zeros(x.shape,
                            device=self.device, dtype=self.dtype)
        for n_i in range(0, self.N):
            mask_n_i = self.input_mask[n_i, :]
            input_x = x[:, mask_n_i]
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

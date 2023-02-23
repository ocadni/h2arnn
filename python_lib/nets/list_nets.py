import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_lib.nets.ann import ANN
import random
from scipy.special import comb


def find_neigh(model_):
    """
    This function finds the neighbors of a given model. 

    Parameters: 
    model_: the model whose neighbors are to be found 

    Variables: 
    n: the number of nodes in the model 
    neighs: a list of lists containing the neighbors of each node in the model 
    num_neighs: a list containing the number of neighbors for each node in the model 
    n_i: an index used to iterate through all nodes in the model 
    neigh_i: an index used to iterate through all possible neighbors for each node in the model 
    val: a boolean value indicating whether or not two nodes are connected by an edge  

    The function loops through all nodes in the model and creates a list containing all of its neighbors. It also creates another list containing the number of neighbors for each node. Finally, it returns both lists."""
    n = model_.N
    neighs = []
    num_neighs = []
    for n_i in range(n):
        num_n_less = 0
        neighs.append([])
        for neigh_i, val in enumerate(model_.J_interaction[n_i][0:n_i]):
            if val:
                num_n_less += 1
                neighs[n_i].append(neigh_i)
        num_neighs.append(num_n_less)
    return neighs


def rand_grad(self, prob_zero=0.5):
    non_z = torch.nonzero(self.net[0].weight.grad)
    for non_z in torch.nonzero(self.net[0].weight.grad):
        if random.random() > prob_zero:
            self.net[0].weight.grad[non_z[0], non_z[1]] = 0
    return self.net[0].weight.grad


class one_var(nn.Module):
    def __init__(self,
                 model,
                 n_i,
                 dict_nets={},
                 dtype=torch.float32,
                 device="cpu",
                 ):
        super().__init__()
        N = model.N
        weight = torch.zeros((1), device=device, dtype=dtype)
        # nn.Parameter is a Tensor that's a module parameter.
        self.weight = nn.Parameter(weight)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight, mean=0.0, std=1/N)

    def forward(self, x):
        res = x.sum(-1) * self.weight
        return torch.sigmoid(res)


class CW_sign(nn.Module):
    def __init__(self,
                 model,
                 n_i,
                 dict_nets={},
                 dtype=torch.float32,
                 device="cpu",
                 ):
        super().__init__()
        N = model.N
        weight = torch.zeros((2), device=device, dtype=dtype)
        # nn.Parameter is a Tensor that's a module parameter.
        self.weight = nn.Parameter(weight)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight, mean=0.0, std=1/N)

    def forward(self, x):
        res = x.sum(-1) * self.weight[0] + \
            self.weight[1] * torch.sign(x.sum(-1))
        return torch.sigmoid(res)


class CW_net(nn.Module):
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


class SK_net_rs(nn.Module):
    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"set_exact": True},
                 dtype=torch.float32,
                 device="cpu",
                 learn_first_l=True
                 ):
        super().__init__()
        N = model.N
        N_i = N - n_i - 1
        self.n_i = n_i
        self.N_i = N_i
        self.learn_first_l = not dict_nets["set_exact"]

        if N_i > 0:
            # first layer
            # second layer
            weight_p = torch.zeros((1, N_i), device=device, dtype=dtype)
            weight_m = torch.zeros((1, N_i), device=device, dtype=dtype)
            bias_p = torch.zeros((1, N_i), device=device, dtype=dtype)
            bias_m = torch.zeros((1, N_i), device=device, dtype=dtype)
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

        self.first_l = nn.Linear(n_i, N_i+1, False,
                                 device=device, dtype=dtype)
        self.first_l.requires_grad_(learn_first_l)
        weight_0 = torch.zeros((1), device=device, dtype=dtype)
        bias_0 = torch.zeros((1), device=device, dtype=dtype)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

    def forward(self, x):
        m = self.first_l(x)
        m_n_i = m[:, 1:]
        m_i = m[:, 0]
        res_p = self.bias_p + self.weight_p * m_n_i
        res_p = self.weight_0p * F.logsigmoid(res_p)
        res_m = self.bias_m + self.weight_m * m_n_i
        res_m = self.weight_0m * F.logsigmoid(res_m)
        res_0 = self.bias_0 + self.weight_0 * m_i
        return torch.sigmoid(res_0 + res_p.sum(dim=1) + res_m.sum(dim=1))

    def set_params_exact(self, model, beta):
        n_i = self.n_i
        self.first_l.requires_grad_(self.learn_first_l)
        self.first_l.weight.data = model.J[n_i:, 0:n_i]


class SK_net_krsb(nn.Module):

    J = torch.zeros(1)
    learn_first_l = False

    @classmethod
    def set_params_exact(cls, model, beta):
        cls.J = model.J.clone()
        cls.J.requires_grad_(cls.learn_first_l)
        if cls.learn_first_l:
            cls.J = nn.Parameter(cls.J)

    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"k": 0},
                 dtype=torch.float32,
                 device="cpu",
                 learn_first_l=True,
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

        # if self.learn_first_l:
        self.Ji = self.J

        weight_0 = torch.zeros((1), device=device, dtype=dtype)
        bias_0 = torch.zeros((1), device=device, dtype=dtype)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

    def forward(self, x):
        m = self.first_l(x)
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

    def first_l(self, x):
        n_i = self.n_i
        return F.linear(x, self.Ji[n_i:, 0:n_i])


class SK_net_krsb_nofirst(nn.Module):

    J = torch.zeros(1)
    learn_first_l = False

    @classmethod
    def set_params_exact(cls, model, beta):
        cls.J = model.J.clone()
        cls.J.requires_grad_(cls.learn_first_l)
        if cls.learn_first_l:
            cls.J = nn.Parameter(cls.J)

    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"k": 0},
                 dtype=torch.float32,
                 device="cpu",
                 learn_first_l=True,
                 ):
        super().__init__()
        N = model.N
        N_i = N - n_i - 1
        self.n_i = n_i
        self.N_i = N_i
        self.k = dict_nets["k"] if "k" in dict_nets else 0
        if N_i > 0:
            # K layers
            weight_p = torch.zeros((self.k+1, N_i), device=device, dtype=dtype)
            weight_m = torch.zeros((self.k+1, N_i), device=device, dtype=dtype)
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

        weight_0 = torch.zeros((3), device=device, dtype=dtype)
        bias_0 = torch.zeros((1), device=device, dtype=dtype)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

    def forward(self, x):
        m = self.first_l(x)
        m_n_i = m[:, 1:]
        m_i = m[:, 0]
        res_p = F.logsigmoid(self.bias_p[0] + self.weight_p[0] * m_n_i)
        res_m = F.logsigmoid(self.bias_m[0] + self.weight_m[0] * m_n_i)
        for kk in range(1, self.k+1):
            res_p = F.logsigmoid(self.bias_p[kk] + self.weight_p[kk] * res_p)
            res_m = F.logsigmoid(self.bias_m[kk] + self.weight_m[kk] * res_m)

        res_p = self.weight_0[0] * res_p
        res_m = self.weight_0[1] * res_m
        res_0 = self.bias_0 + self.weight_0[2] * m_i

        return torch.sigmoid(res_0 + res_p.sum(dim=1) + res_m.sum(dim=1))

    def first_l(self, x):
        n_i = self.n_i
        return F.linear(x, self.J[n_i:, 0:n_i])


class SK_net_krsb_one(nn.Module):

    J = torch.zeros(1)
    learn_first_l = False

    @classmethod
    def set_params_exact(cls, model, beta):
        cls.J = model.J.clone()
        cls.J.requires_grad_(cls.learn_first_l)
        if cls.learn_first_l:
            cls.J = nn.Parameter(cls.J)

    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"k": 0},
                 dtype=torch.float32,
                 device="cpu",
                 learn_first_l=True,
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
            bias_p = torch.zeros((self.k+1, N_i), device=device, dtype=dtype)

            # nn.Parameter is a Tensor that's a module parameter.
            self.weight_p = nn.Parameter(weight_p)
            self.bias_p = nn.Parameter(bias_p)

            # initialize weights and biases
            torch.nn.init.normal_(self.weight_p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_p, mean=0.0, std=1/N)
        else:
            self.k = 0
            self.weight_p = torch.zeros((2, 1), device=device, dtype=dtype)
            self.bias_p = torch.zeros((1, 1), device=device, dtype=dtype)

        weight_0 = torch.zeros((1), device=device, dtype=dtype)
        bias_0 = torch.zeros((1), device=device, dtype=dtype)
        self.weight_0 = nn.Parameter(weight_0)
        self.bias_0 = nn.Parameter(bias_0)

        # initialize weights and biases
        torch.nn.init.normal_(self.weight_0, mean=0.0, std=1/N)
        torch.nn.init.normal_(self.bias_0, mean=0.0, std=1/N)

    def forward(self, x):
        m = self.first_l(x)
        m_n_i = m[:, 1:]
        m_i = m[:, 0]
        res_p = F.logsigmoid(self.bias_p[0] + self.weight_p[0] * m_n_i)
        for kk in range(1, self.k+1):
            res_p = F.logsigmoid(self.bias_p[kk] + self.weight_p[kk] * res_p)

        res_p = self.weight_p[-1] * res_p
        res_0 = self.bias_0 + self.weight_0 * m_i

        return torch.sigmoid(res_0 + res_p.sum(dim=1))

    def first_l(self, x):
        n_i = self.n_i
        return F.linear(x, self.J[n_i:, 0:n_i])


class SK_net_krsb_new(nn.Module):

    J = torch.zeros(1)
    learn_first_l = False

    @classmethod
    def set_params_exact(cls, model, beta):
        cls.J = model.J.clone()
        cls.J.requires_grad_(cls.learn_first_l)
        if cls.learn_first_l:
            cls.J = nn.Parameter(cls.J)

    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"k": 0},
                 dtype=torch.float32,
                 device="cpu",
                 learn_first_l=True,
                 ):
        super().__init__()
        N = model.N
        N_i = N - n_i - 1
        self.n_i = n_i
        self.N_i = N_i
        self.k = dict_nets["k"] if "k" in dict_nets else 0
        if N_i > 0:
            # K layers
            weight_p = torch.zeros((self.k+2, 1), device=device, dtype=dtype)
            weight_m = torch.zeros((self.k+2, 1), device=device, dtype=dtype)
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

    def forward(self, x):
        m = self.first_l(x)
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

    def first_l(self, x):
        n_i = self.n_i
        return F.linear(x, self.J[n_i:, 0:n_i])


class CW_net_sp(nn.Module):
    def __init__(self,
                 model,
                 n_i,
                 dict_nets={"num_extremes": 1},
                 dtype=torch.float32,
                 device="cpu",
                 ):
        super().__init__()
        N = model.N
        N_i = N - n_i - 1
        self.n_i = n_i
        self.N_i = N_i
        n_e = dict_nets["num_extremes"]
        self.n_e = n_e
        if N_i > 0:
            weight_p1 = torch.zeros((1, n_e), device=device, dtype=dtype)
            weight_m1 = torch.zeros((1, n_e), device=device, dtype=dtype)
            bias_p1 = torch.zeros((1, n_e), device=device, dtype=dtype)
            bias_m1 = torch.zeros((1, n_e), device=device, dtype=dtype)
            weight_p2 = torch.zeros((1, n_e), device=device, dtype=dtype)
            weight_m2 = torch.zeros((1, n_e), device=device, dtype=dtype)
            bias_p2 = torch.zeros((1, n_e), device=device, dtype=dtype)
            bias_m2 = torch.zeros((1, n_e), device=device, dtype=dtype)
            weight_0p = torch.zeros((1), device=device, dtype=dtype)
            weight_0m = torch.zeros((1), device=device, dtype=dtype)
            # nn.Parameter is a Tensor that's a module parameter.
            self.weight_p1 = nn.Parameter(weight_p1)
            self.bias_p1 = nn.Parameter(bias_p1)
            self.weight_m1 = nn.Parameter(weight_m1)
            self.bias_m1 = nn.Parameter(bias_m1)
            self.weight_p2 = nn.Parameter(weight_p2)
            self.bias_p2 = nn.Parameter(bias_p2)
            self.weight_m2 = nn.Parameter(weight_m2)
            self.bias_m2 = nn.Parameter(bias_m2)
            self.weight_0p = nn.Parameter(weight_0p)
            self.weight_0m = nn.Parameter(weight_0m)

            # initialize weights and biases
            torch.nn.init.normal_(self.weight_0p, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_0m, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_p1, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_p1, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_m1, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_m1, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_p2, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_p2, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.weight_m2, mean=0.0, std=1/N)
            torch.nn.init.normal_(self.bias_m2, mean=0.0, std=1/N)
        else:
            self.weight_p1 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.weight_m1 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_p1 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_m1 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.weight_p2 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.weight_m2 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_p2 = torch.zeros((1, 1), device=device, dtype=dtype)
            self.bias_m2 = torch.zeros((1, 1), device=device, dtype=dtype)
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
        res_p = self.bias_m2 + self.weight_p2 * torch.unsqueeze(m_i, dim=1)
        res_p = self.bias_p1 + self.weight_p1 * torch.cosh(res_p)
        res_p = self.weight_0p * torch.logsumexp(res_p, 1)
        res_m = self.bias_m2 + self.weight_m2 * torch.unsqueeze(m_i, dim=1)
        res_m = self.bias_m1 + self.weight_m1 * torch.cosh(res_m)
        res_m = self.weight_0m * torch.logsumexp(res_m, 1)
        res_0 = self.bias_0 + self.weight_0 * m_i
        return torch.sigmoid(res_0 + res_p + res_m)


class list_nets(ANN):
    def __init__(
        self,
        model,
        single_net,
        input_mask,
        dtype=torch.float32,
        device="cpu",
        eps=1e-10,
        dict_nets={"set_exact": False},
    ):
        net = []
        for n_i in range(model.N):
            net.append(single_net(model, n_i, device=device,
                       dtype=dtype, dict_nets=dict_nets))
        super(list_nets, self).__init__(
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

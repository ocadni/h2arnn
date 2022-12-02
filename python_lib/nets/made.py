import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from python_lib.nets.ann import ANN


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias, exclusive):
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n,
                                           bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        self.register_buffer('mask', torch.ones([self.n] * 2))
        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)
        else:
            self.mask = torch.tril(self.mask)
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return (super(MaskedLinear, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


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


class MADE(ANN):
    def __init__(
        self,
        model,
        bias=True,
        eps=1e-10,
        dtype=torch.float32,
        device="cpu",
        net_depth=1,
        net_width=1
    ):
        J_interaction = model.J_interaction.to(dtype=dtype, device=device)
        self.bias = bias
        self.n = model.N
        self.net_depth = net_depth
        self.net_width = net_width
        self.epsilon = eps
        layers = []
        layers.append(
            MaskedLinear(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.n,
                self.bias,
                exclusive=True))
        for count in range(self.net_depth - 2):
            layers.append(
                self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(self._build_simple_block(self.net_width, 1))
        layers.append(nn.Sigmoid())
        net = nn.Sequential(*layers)
        params = list(net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print(f'Total number of trainable parameters: {nparams}')
        input_mask = torch.tril(J_interaction, diagonal=-1)
        super(simple_layer, self).__init__(
            model, net, input_mask, dtype=dtype, device=device, eps=eps)
        self.net.params = params

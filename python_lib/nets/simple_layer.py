import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from python_lib.nets.ann import ANN


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


class simple_layer(ANN):
    def __init__(
        self,
        model,
        bias=True,
        eps=1e-10,
        dtype=torch.float32,
        device="cpu",
    ):
        J_interaction = model.J_interaction.to(dtype=dtype, device=device)
        bias = bias
        layers = []
        layer1 = myLayer(
            model.N,
            J_interaction,
            bias,
            device=device,
            dtype=dtype
        )
        layers.append(layer1)
        layers.append(nn.Sigmoid())
        net = nn.Sequential(*layers)
        params = list(net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print(f'Total number of trainable parameters: {nparams}')
        #input_mask = torch.tril(J_interaction, diagonal=-1)
        super(simple_layer, self).__init__(
            model, net, dtype=dtype, device=device, eps=eps)
        self.net.params = params

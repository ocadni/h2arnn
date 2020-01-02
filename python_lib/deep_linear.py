import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return nn.Linear(in_feat,out_feat, bias)
    else:
        return zero_linear(out_feat, bias)
            


class deep_linear(nn.Module):
    def __init__(self, features, bias, in_func = nn.ReLU()):
        super(deep_linear, self).__init__()
        layers = []
        for feat_i, feat in enumerate(features[:-1]):
            in_feat = feat
            out_feat = features[feat_i+1]
            layers.append(my_linear(in_feat,out_feat, bias))
            layers.append(in_func)
        layers[-1] = nn.Sigmoid()
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bp_nn
type_default = torch.float64

default_dtype_torch = torch.float64

class bp_nn_2(bp_nn.bp_nn):
    def __init__(self, n, model, bias,diagonal=-1, 
                 identity=False, z2=False, 
                 x_hat_clip=False, init_zero=False):
        super(bp_nn_2, self).__init__(n, model, bias, 
                                      diagonal=-1, identity=False, 
                                      z2=False, x_hat_clip=False, 
                                      init_zero=False)
        
        mask_layer1 = []

        for r_i, row in enumerate(self.J_interaction):
            mask_layer1.append([0] * n)
            for c_i, val in enumerate(row):
                if c_i < r_i and val != 0:
                    inter_e = [0] * n
                    inter_e[c_i] = 1
                    #print(r_i,c_i, val, inter_e)
                    mask_layer1.append(inter_e)
        mask_layer1 = np.array(mask_layer1)
        layer1 = bp_nn.myLinear(mask_layer1, True)
        
        layer2 = nn.PReLU()

        mask_layer3 = []
        count_node = 0
        for r_i, row in enumerate(self.J_interaction):
            inter_e = [0] * mask_layer1.shape[0]
            #print(inter_e, count_node)
            inter_e[count_node] = 1
            count_node += 1
            for c_i, val in enumerate(row):
                if c_i < r_i and val != 0:
                    inter_e[count_node] = 1
                    count_node += 1
            mask_layer3.append(inter_e)
        mask_layer3 = torch.Tensor(mask_layer3)
        layer3 = bp_nn.myLinear(mask_layer3, False)
        for param in layer3.parameters():
            param.requires_grad=False
        layer3.weight = torch.nn.Parameter(mask_layer3)   

        layer4 = nn.Sigmoid()

        layers = [layer1, layer2, layer3, layer4]
        net = nn.Sequential(*layers)

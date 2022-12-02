import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
type_default = torch.float64

class myLayer(nn.Linear):
    def __init__(self, n, J_interaction, bias, diagonal=-1, identity=False, init_zero = False):
        super(myLayer, self).__init__( n, n, bias)
        self.n = n

        self.register_buffer('mask', torch.tril(J_interaction, diagonal=diagonal))
        if identity:
            self.mask += torch.eye(self.n)
        
        self.weight.data *= self.mask
        #self.weight.data *= 0
        #self.bias.data *= 0
        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
        if init_zero:
            self.weight.data *= 0
            #print(self.weight.data)
            if bias:
                self.bias.data *= 0
            
    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    
default_dtype_torch = torch.float64

class plaq_nn(nn.Module):
    def __init__(self, n, model, bias, diagonal=-1, identity=False, z2=False, x_hat_clip=False, init_zero=False):
        super(plaq_nn  , self).__init__()
        self.n = n
        self.epsilon = 1e-10
        self.model = model
        self.J_interaction = torch.from_numpy(model.J_interaction).float()
        self.bias = bias
        self.z2 = z2
        self.x_hat_clip = x_hat_clip
        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer('x_hat_mask', torch.ones(self.n))
            self.x_hat_mask[0] = 0
            self.register_buffer('x_hat_bias', torch.zeros(self.n))
            self.x_hat_bias[0] = 0.5

        layers = []
        layer1 = myLayer(n, self.J_interaction, bias, diagonal=diagonal, identity=identity, init_zero = init_zero)
        layers.append(layer1)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_hat = self.net(x)
        
        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                          1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat
                                 

        return x_hat


    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, self.n])
        for i in range(self.n):
            x_hat = self.forward(sample)
            #print(x_hat)
            sample[:, i] = torch.bernoulli(
                    x_hat[:, i]) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = torch.randint(
                2, [batch_size, 1], dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat + self.epsilon) * mask +
                    torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        sample = sample.view(sample.shape[0], -1)
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)
        
        if self.z2:
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(
                    torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - np.log(2)
        
        return log_prob
    
    def compute_stat(self, beta, batch_size = 10000, print_=True):
        free_energy_mean = 0
        with torch.no_grad():
            sample, x_hat = self.sample(batch_size)
            log_prob = self.log_prob(sample).double()
            energy = self.model.energy(sample.double())
            loss = log_prob + beta * energy
            free_energy_mean = loss.mean() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            entropy_mean = -log_prob.mean() / self.n
            energy_mean = energy.mean() / self.n
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            self.F = free_energy_mean
            self.M = mag_mean
            self.E = energy_mean
            self.S = entropy_mean
            self.M_i = mag.numpy()
            self.F_std = free_energy_std
            if print_:
                print("\nfree_energy: {0:.3f},  std_fe: {1:.5f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f}".format(free_energy_mean.double(),
                                                    free_energy_std,
                                                    mag_mean,
                                                    entropy_mean,
                                                    energy_mean,
                                                        ), end="")

        return free_energy_mean
    
    def train(self, lr=1e-3, opt = "adam", beta = 1, max_step = 10000, 
              batch_size = 1000, std_fe_limit = 1e-4,
             batch_mean=10000,
             x_hat_clip = False):
        self.x_hat_clip = x_hat_clip
        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print('Total number of trainable parameters: {}'.format(nparams))
        named_params = list(self.net.named_parameters())
        #print("lr:{0}".format(lr))
        if opt == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr)
        elif opt == 'sgdm':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        elif opt == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.99)
        elif opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        elif opt == 'adam0.5':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))

        else:
            print("optimizer not found, setted Adam")
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9999))
        
        optimizer.zero_grad()
        for step in range(0, max_step + 1):
            optimizer.zero_grad()
            with torch.no_grad():
                sample, x_hat = self.sample(batch_size)
            assert not sample.requires_grad
            assert not x_hat.requires_grad
            
            if self.x_hat_clip:
                    # Clip value and preserve gradient
                with torch.no_grad():
                    delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                                  1 - self.x_hat_clip) - x_hat
                assert not delta_x_hat.requires_grad
                x_hat = x_hat + delta_x_hat

                # Force the first x_hat to be 0.5
            #if self.bias and not self.z2:
            #    x_hat = x_hat * self.x_hat_mask + self.x_hat_bias


            log_prob = self.log_prob(sample).to(type_default)
            with torch.no_grad():
                energy = self.model.energy(sample)
                #loss =  log_prob * (1. / beta) + energy
                loss =  log_prob + beta * energy
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()
            optimizer.step()

            free_energy_mean = loss.mean() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            #free_energy_mean = loss.mean() / self.n
            #free_energy_std = loss.std() / self.n
            entropy_mean = -log_prob.mean() / self.n
            energy_mean = energy.mean() / self.n
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            B1 = (self.net[0].bias.data[0] if "data" in self.net[0].bias else "none")
            B2 = (self.net[0].bias.data[1] if "data" in self.net[0].bias else "none")
            print("\r {beta:.2f} {step} fe: {0:.3f} +- {1:.5f} E: {E:.3f}, S: {S:.3f}, M: {2:.3}, W: {W1:.6}, {W2:.6}, Bias: {B1:.6f} - {B2:.6f}".format(
                free_energy_mean.double(),
                                                free_energy_std,
                                                mag_mean,
                                                step=step, beta=beta,
                                                W1 = self.net[0].weight.data[1][0],
                                                W2 = self.net[0].weight.data[2][0],
                                                B1 = B1,
                B2=B2,
                E = energy_mean,
                S = entropy_mean
            ), end="")
            if free_energy_std < std_fe_limit:
                break
                
        return {"free_energy_mean":free_energy_mean,
               "free_energy_std":free_energy_std,
               "entropy_mean":entropy_mean,
               "energy_mean":energy_mean,
               "mag":mag,
                "mag_mean":mag_mean
               }
    
    
class bp_nn_normed(bp_nn):
    def __init__(self, n, model, bias):
        super(bp_nn_normed, self).__init__(n, model, bias)
    
    def train(self, lr=1e-3, opt = "adam", beta = 1, max_step = 10000, 
              batch_size = 1000, std_fe_limit = 1e-4,
             batch_mean=10000,
             x_hat_clip = False):
        self.x_hat_clip = x_hat_clip
        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print('Total number of trainable parameters: {}'.format(nparams))
        named_params = list(self.net.named_parameters())
        #print("lr:{0}".format(lr))
        if opt == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr)
        elif opt == 'sgdm':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        elif opt == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.99)
        elif opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        elif opt == 'adam0.5':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))

        else:
            print("optimizer not found, setted Adam")
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9999))
        
        optimizer.zero_grad()
        for step in range(0, max_step + 1):
            optimizer.zero_grad()
            #with torch.no_grad():
            sample, p_sample = self.sample(batch_size)
            #assert not sample.requires_grad
            #assert not x_hat.requires_grad
            
            if self.x_hat_clip:
                    # Clip value and preserve gradient
                with torch.no_grad():
                    delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                                  1 - self.x_hat_clip) - x_hat
                assert not delta_x_hat.requires_grad
                x_hat = x_hat + delta_x_hat

                # Force the first x_hat to be 0.5
            #if self.bias and not self.z2:
            #    x_hat = x_hat * self.x_hat_mask + self.x_hat_bias


            log_prob = torch.log(p_sample).to(type_default)
            with torch.no_grad():
                energy = self.model.energy(sample)
                #loss =  log_prob * (1. / beta) + energy
                #p_sample, norm_p = self.prob_sample(sample, x_hat)
                #p_sample = p_sample.to(type_default)
                loss = log_prob + beta * energy
            assert not energy.requires_grad
            assert not loss.requires_grad
            loss_reinforce = (p_sample * ((loss - (p_sample * loss).sum()) * log_prob)).sum()
            loss_reinforce.backward()
            optimizer.step()

            free_energy_mean = (p_sample * loss).sum() / beta / self.n
            free_energy_std = loss.std() / beta / self.n
            #free_energy_mean = loss.mean() / self.n
            #free_energy_std = loss.std() / self.n
            entropy_mean = -log_prob.mean() / self.n
            energy_mean = energy.mean() / self.n
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            B1 = (self.net[0].bias.data[0] if "data" in self.net[0].bias else "none")
            B2 = (self.net[0].bias.data[1] if "data" in self.net[0].bias else "none")
            print("\r {beta:.2f} {step} fe: {0:.3f} +- {1:.5f} E: {E:.3f}, S: {S:.3f}, M: {2:.3}, W: {W1:.6}, {W2:.6}, Bias: {B1:.6f} - {B2:.6f}".format(
                free_energy_mean.double(),
                                                free_energy_std,
                                                mag_mean,
                                                step=step, beta=beta,
                                                W1 = self.net[0].weight.data[1][0],
                                                W2 = self.net[0].weight.data[2][0],
                                                B1 = B1,
                B2=B2,
                E = energy_mean,
                S = entropy_mean
            ), end="")
            if free_energy_std < std_fe_limit:
                break
                
        return {"free_energy_mean":free_energy_mean,
               "free_energy_std":free_energy_std,
               "entropy_mean":entropy_mean,
               "energy_mean":energy_mean,
               "mag":mag,
                "mag_mean":mag_mean
               }
    
    def prob_sample(self, sample, x_hat):
        mask = (sample + 1) / 2
        p_sample = (x_hat ** mask)*((1-x_hat)**(1-mask))
        p_sample = p_sample.prod(dim=1)
        norm = p_sample.sum()
        p_sample /= norm
        return p_sample, norm
    
    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, self.n])
        for i in range(self.n):
            x_hat = self.forward(sample)
            #print(x_hat)
            sample[:, i] = torch.bernoulli(
                    x_hat[:, i]) * 2 - 1
        with torch.no_grad():
            sample, count = sample.unique(dim=0, return_counts=True)
            p_sample = count.to(type_default) / batch_size
        print(count, p_sample)
        
        if self.z2:
            # Binary random int 0/1
            flip = torch.randint(
                2, [batch_size, 1], dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip

        return sample, p_sample

        
class bp_nn_rand(bp_nn):
    def __init__(self, n, model, bias):
        super(bp_nn_rand, self).__init__(n, model, bias)
        layers = []
        layer1 = myLayer_rand(n, self.J_interaction, bias)
        layer2 = myLayer_rand(n, self.J_interaction, bias)
        layers.append(layer1)
        layers.append(nn.Sigmoid())
        layers.append(layer2)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

class bp_nn_out_diag(bp_nn):
    def __init__(self, n, model, bias):
        super(bp_nn_out_diag, self).__init__(n, model, bias)
        layers = []
        layer1 = myLayer_out_diag(n, self.J_interaction, bias)
        layers.append(layer1)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    
class bp_nn2(bp_nn):
    def __init__(self, n, model, bias):
        super(bp_nn2, self).__init__(n, model, bias)
        num_edges = int(self.J_interaction.sum()/2)
        nodes_edges = []
        bias_layer1 = []
        for r_i, row in enumerate(self.J_interaction):
            for c_i, val in enumerate(row):
                if c_i < r_i and val != 0:
                    inter_e = [0] * n
                    inter_e[c_i] = 1
                    #print(r_i,c_i, val, inter_e)
                    nodes_edges.append(inter_e)
                    bias_layer1.append(1)
        nodes_edges = np.array(nodes_edges)
        layer1 = myLayer_rand(nodes_edges, bias)
        layer2 = myLinear(np.transpose(nodes_edges), bias)
        layers = [layer1, layer2, nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
    
import torch
import numpy as np
import math
from scipy.special import logsumexp, comb


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class model:
    '''This is the model class for the Ising model.'''

    def __init__(
        self,
        N,
        H,
        J,
        J_interaction,
        device="cpu",
        dtype=torch.float,
    ):
        '''Initialize the model. N is the number of spins. H is the field. J is the interaction. J_interaction is the interaction between spins.
        N: int, number of spins
        H: array, field
        J: matrix, coupling values
        J_interaction: matrix, interaction between spins [1 when is present 0 otherwise]
        device: str, device to use
        dtype: torch.dtype, data type to use
        '''
        self.N = N
        self.device = device
        self.dtype = dtype
        H = torch.Tensor(H)
        J = torch.Tensor(J)
        J_interaction = torch.Tensor(J_interaction)
        self.H = H.detach().clone().to(device=device, dtype=dtype)
        self.J = J.detach().clone().to(device=device, dtype=dtype)
        self.J_interaction = J_interaction.detach().clone().to(device=device, dtype=dtype)

        # self.Corr = torch.zeros_like(self.J, device=self.device, dtype=self.dtype)
        assert H.shape[0] == N
        assert J.shape[0] * J.shape[1] == N * N

    def exact_old(self, beta):
        assert self.N < 28
        J = self.J
        H = self.H
        N = self.N
        E_min = 0
        n_total = int(math.pow(2, self.N))
        Z = 0
        # print('Enumerating...')
        E_mean = 0
        M_mean = 0
        S_mean = 0
        M_i_mean = np.zeros(N)
        Corr = np.zeros(J.shape)
        for d in range(n_total):
            s = np.binary_repr(d, width=N)
            b_numpy = np.array(list(s)).astype(np.float32)
            b_numpy[b_numpy < 0.5] = -1
            b_ = torch.from_numpy(b_numpy).to(self.dtype)
            b = b_.view(N, 1)
            E = (b.t() @ J @ b - b.t() @ H).data[0][0]
            if E < E_min:
                E_min = E
            Z_temp = torch.exp(-beta * E)
            E_mean += E * Z_temp
            M_mean += b.sum() * Z_temp
            M_i_mean += b_numpy * Z_temp.numpy()
            Z += Z_temp
            S_mean += -Z_temp * torch.log(Z_temp)
            Corr += np.outer(b_numpy, b_numpy) * Z_temp.numpy()

            if d % 1000 == 0:
                print(
                    "\r{} / {}({:.2%}), E = {:.3}, Z = {:.3}, F = {:.3}".format(
                        d,
                        n_total,
                        d / n_total,
                        E.numpy(),
                        Z.numpy(),
                        -(1 / beta) * math.log(Z.numpy()),
                    ),
                    end="",
                )

        print("\r", end="")
        self.free_energy = -(1 / beta) * math.log((Z).numpy()) * (1.0 / self.N)
        self.E_mean = (E_mean / (Z * self.N)).numpy()
        self.S_mean = (S_mean / (Z * self.N) - beta * self.free_energy).numpy()
        self.M_i_mean = M_i_mean / Z.numpy()
        self.M_mean = abs(self.M_i_mean).mean()
        self.Corr = Corr / Z.numpy() - np.outer(self.M_i_mean, self.M_i_mean)
        self.Corr_neigh = self.J_interaction * self.Corr
        self.Z = Z
        # Corr -= torch.ger(M_i_mean,M_i_mean)
        print(
            "beta: {2:.1f}, Fe: {0:.3f}".format(
                self.free_energy, self.E_mean -
                (1.0 / beta) * self.S_mean, beta
            ),
            end=" ",
        )
        print(
            "Energy: {0:.3} M: {1:.3} S: {2:.3}".format(
                self.E_mean,
                self.M_mean,
                self.S_mean,
            ),
        )

        return self.free_energy

    def exact(self, beta, chunk=1000):
        assert self.N < 35
        J = self.J
        H = self.H
        N = self.N
        E_min = 0
        n_total = int(math.pow(2, N))
        all_conf_n = torch.arange(0, n_total)
        Z = 0
        # print('Enumerating...')
        E_mean = 0
        M_mean = 0
        M_abs_mean = 0
        S = 0
        M_i_mean = torch.zeros(N)
        Corr = np.zeros(J.shape)
        confs_split = torch.split(all_conf_n, chunk)
        log_Zs = torch.zeros(
            len(confs_split), dtype=self.dtype, device=self.device)
        Es_total = torch.zeros(
            n_total, dtype=self.dtype, device=self.device)

        for confs_n_i, confs_n in enumerate(confs_split):
            x = binary(confs_n, N).to(dtype=self.dtype, device=self.device)
            x[x < 0.5] = -1
            Es = self.energy(x)
            n_init = confs_n_i*chunk
            n_end = n_init+len(Es)
            Es_total[n_init:n_end] = Es
            log_Zs[confs_n_i] = torch.logsumexp(-beta * Es, dim=0)
            E_min_temp = torch.min(Es)
            if E_min_temp < E_min:
                E_min = E_min_temp
            Z_temp = torch.exp(-beta * Es)
            E_mean += (Es * Z_temp).sum(axis=0)
            M_mean += (x.sum(axis=1) * Z_temp).sum(axis=0)
            M_abs_mean += torch.abs(x.mean(axis=1) * Z_temp).sum(axis=0)
            M_i_mean += (x * Z_temp.reshape(Z_temp.shape[0], 1)).sum(axis=0)
            S += (-Z_temp * torch.log(Z_temp)).sum(axis=0)
            # Corr += np.outer(b_numpy, b_numpy) * Z_temp.numpy()

            if confs_n_i % 1 == 0:
                print(
                    "\r{} / {}({:.2%}), E = {:.3}, logZ = {:.3}, F = {:.3}".format(
                        confs_n_i * chunk,
                        n_total,
                        (confs_n_i * chunk) / n_total,
                        Es.mean(),
                        torch.logsumexp(
                            log_Zs[: confs_n_i + 1], dim=0),
                        -(1 / (beta*N))
                        * torch.logsumexp(log_Zs[: confs_n_i + 1], dim=0),
                    ),
                    end="  ",
                )

        print("\r", end="")
        log_Z = torch.logsumexp(-beta * Es_total, dim=0)
        Z = torch.exp(log_Z)

        self.free_energy = -(1 / beta) * log_Z * (1.0 / self.N)
        self.E_mean = (torch.sign(
            Es_total) * torch.exp(-beta*Es_total + torch.log(torch.abs(Es_total)) - log_Z)).sum() / N
        self.S = beta * (self.E_mean - self.free_energy)
        self.M_i_mean = M_i_mean / Z
        self.M_mean = self.M_i_mean.mean()
        self.M_abs_mean = M_abs_mean / Z
        # self.Corr = Corr / Z - torch.outer(self.M_i_mean, self.M_i_mean)
        # self.Corr_neigh = self.J_interaction * self.Corr
        self.Z = Z
        # Corr -= torch.ger(M_i_mean,M_i_mean)
        log_Z = log_Z.cpu().numpy()

        print(
            "beta: {2:.1f}, Fe: {0:.3f}".format(
                self.free_energy, self.E_mean - (1.0 / beta) * self.S, beta
            ),
            end=" ",
        )
        print(
            "Energy: {0:.3} M: {1:.3} S: {2:.3}".format(
                self.E_mean,
                self.M_mean,
                self.S,
            ),
        )
        return {
            "beta": beta,
            "free_energy_mean": self.free_energy.cpu().item(),
            "free_energy_std": 0,
            "entropy_mean": self.S.cpu().item(),
            "energy_mean": self.E_mean.cpu().item(),
            "mag": self.M_mean.cpu().item(),
            "mag_mean": self.M_abs_mean.cpu().numpy(),
        }

    def energy(self, samples):
        """
        Compute energy of samples, samples should be of size [m, n] where n is the number of spins, m is the number of samples.
        """
        samples = samples.view(samples.shape[0], -1).to(self.dtype)
        assert samples.shape[1] == self.N
        m = samples.shape[0]
        inter = -(
            (samples @ self.J).view(m, 1, self.N) @ samples.view(m, self.N, 1)
        ).squeeze()
        field = -(samples.view(m, 1, self.N) @
                  self.H.view(self.N, 1)).squeeze()
        return inter + field

    def prob_sample(self, beta, samples):
        if self.Z:
            return torch.exp(-beta * self.energy(samples)) / self.Z
        else:
            print("compute model.exact() first")
            return 0


class KW_exact_fast(model):

    def entropy_infN(self, m):
        if m == -1 or m == 1:
            return 0
        return -((1+m)/2)*np.log((1+m)/2)-((1-m)/2)*np.log((1-m)/2)

    def free_energy_exact_approximately(self, N, beta, J, h):
        elems = []
        for k in range(0, N+1):
            m = 1 - 2*k/N
            elems.append(N*(self.entropy_infN(m) +
                         (beta*J/2)*(m**2) + beta*h*m))
        return -logsumexp(elems)/(N*beta) + J/(2*N)

    def exact(self, beta):
        N = self.N
        J = self.J[0][1].item() * (2 * N)
        h = self.H[0].item()
        elems = []
        m_elems = []
        for k in range(0, N+1):
            m = 1 - 2*k/N
            val = np.log(comb(N, k)) + N*(beta*J/2)*(m**2) + N * beta*h*m
            elems.append(val)
            if m == 0:
                m_elems.append(0)
            else:
                m_elems.append(val + np.log(np.abs(m)))
        logZ = logsumexp(elems)
        fe = -logZ/(N*beta) + J/(2*N)
        m_abs = np.exp(logsumexp(m_elems) - logZ)
        return {
            "beta": beta,
            "free_energy": fe,
            "free_energy_std": 0,
            "mag_mean": m_abs,
        }

    def exact_infN(self, beta, err_=1e-6):
        N = self.N
        mm = 0.1
        h = self.H[0].item()
        J = self.J[0][1].item() * (2 * N)
        diff = err_+1
        while diff > err_:
            mm_ = np.tanh(beta*(J*mm+h))
            diff = mm_ - mm
            mm = mm_

        fe = -(self.entropy_infN(mm)/beta + (J/2)*(mm**2) + h*mm)

        return {
            "beta": beta,
            "free_energy": fe,
            "free_energy_std": 0,
            "mag_mean": mm,
        }

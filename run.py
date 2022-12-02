import numpy as np
import networkx as nx
import random
import pandas as pd
import torch
import pandas as pd
import time

import sys
import time
from pathlib import Path
import argparse
import warnings

from python_lib.models import spins_model
import python_lib.graph_gen as graph_gen
import python_lib.nets as nets
import python_lib.nets.simple_layer as simple_layer
import python_lib.run_lib as run_lib


def main():
    parser = argparse.ArgumentParser(
        description="Train Autoregressive neural networks")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/tests/",
        help="saving directory for data",
    )

    parser.add_argument(
        "--file_name",
        type=str,
        default="data",
        help="name of the file in which the marginals will be saved",
    )

    parser.add_argument(
        "--model", type=str, default="CW", help="Ising model"
    )
    parser.add_argument(
        "--net_spec", type=str, default="SL", help="Ising model"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device of pytorch"
    )
    parser.add_argument(
        "--suffix", type=str, default=None, help="suffix for file"
    )
    parser.add_argument(
        "--N", type=int, default=100, help="Number of spins", nargs="+"
    )
    parser.add_argument(
        "--J", type=float, default=1.0, help="j"
    )

    parser.add_argument(
        "--h", type=float, default=1.0, help="external field"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate"
    )

    parser.add_argument(
        "--std_fe_limit", type=float, default=1e-4, help="stopping threshold on the std of free energy"
    )
    parser.add_argument(
        "--max_step", type=int, default=500, help="Max number of steps for each value of beta",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size in the learning process",
    )

    parser.add_argument(
        "--batch_iter", type=int, default=20, help="Number of iteration in the computations of average quantities (tot num of samples batch_iter*batch_size)",
    )

    parser.add_argument(
        "--beta_range", type=float, default=[0.1, 2.0, 0.05], help="write start_beta end_beta incr_step. Ex. 0.1 2 0.01", nargs="+"
    )

    args = parser.parse_args()
    print("arguments:")
    print(args)

    device = args.device

    N = args.N
    JJ = args.J
    hh = args.h
    H = torch.ones(N)
    J_interaction = torch.ones(N, N)
    J = torch.ones(N, N)
    betas_r = tuple(args.beta_range)
    betas = np.arange(betas_r[0], betas_r[1]+betas_r[2], betas_r[2])
    net_spec = args.net_spec
    lr = args.lr
    max_step = args.max_step
    batch_size = args.batch_size
    std_fe_limit = args.std_fe_limit
    batch_iter = args.batch_iter
    suffix = args.suffix if args.suffix != None else net_spec
    stats = []

    if args.model == "CW":

        H = hh * H
        J_interaction = J_interaction - torch.eye(N, N)
        J_val = JJ/(2*N)
        J = J_val * J_interaction
        CW_model = spins_model.model(N, H, J, J_interaction, device=device)

        if net_spec == "exact":
            for beta_ in betas:
                stats.append(CW_model.exact(beta_))
            stats = pd.DataFrame(stats)
            stats = stats.add_suffix(suffix)

        elif net_spec == "exact_Ninf":
            for beta_ in betas:
                stats.append(CW_model.exact_infN(beta_))
            stats = pd.DataFrame(stats)
            stats = stats.add_suffix(suffix)

        else:
            net = {}
            if net_spec == "SL":
                net = simple_layer.simple_layer(
                    CW_model, device=device)
            else:
                print("ERROR: Nets not found!")
                return 0
            stats = run_lib.train_net(net,
                                      betas,
                                      lr=lr,
                                      max_step=max_step,
                                      batch_size=batch_size,
                                      std_fe_limit=std_fe_limit,
                                      suffix=args.suffix,
                                      batch_iter=batch_iter
                                      )
            stats["num_params"] = net.num_params(train=False)
            stats["num_train_params"] = net.num_params(train=True)

    stats["N"] = N
    stats["model"] = args.model
    stats["net_spec"] = net_spec
    stats["JJ"] = JJ
    stats["h"] = hh
    stats["lr"] = lr
    stats["max_step"] = max_step
    stats["batch_size"] = batch_size
    stats["std_fe_limit"] = std_fe_limit
    stats["batch_iter"] = batch_iter

    timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + \
        str(random.randint(1, 1000))
    file_name = f"_model{args.model}_net_spec{net_spec}_N{N}_J{JJ:.2}_h{hh:.2}"
    file_name = timestr + file_name + ".gzip"
    stats.to_pickle(args.save_dir + file_name)

    return True


main()

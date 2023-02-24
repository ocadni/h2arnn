import numpy as np
import networkx as nx
import random
import pandas as pd
import torch
import pandas as pd
import time
import os

import sys
import time
from pathlib import Path
import argparse
import warnings
from python_lib.models import spins_model
import python_lib.graph_gen as graph_gen
import python_lib.nets as nets
import python_lib.nets.simple_layer as simple_layer
import python_lib.nets.made as made
import python_lib.run_lib as run_lib
import python_lib.nets.h2_arnn as h2_arnn
import python_lib.nets.list_nets as list_nets
import python_lib.nets.cw_arnn as cw_arnn


def file_name(args, net=False):
    if args.save_net == "yes":
        if not os.path.exists(args.save_dir+"nets/"):
            os.makedirs(args.save_dir+"nets/")

    if net:
        return args.save_dir + "nets/" + f"N{args.N}_seed{args.seed}_model{args.model}_net_spec{args.net_spec}_J{args.J:.2}_h{args.h:.2}_lr{args.lr:.2}_max_step{args.max_step}_batch_size{args.batch_size}_std_fe_limit{args.std_fe_limit:.2}"
    else:
        return args.save_dir + f"N{args.N}_seed{args.seed}_model{args.model}_net_spec{args.net_spec}"


def parse_args():
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
        help="name of the file in which the results will be saved",
    )
    parser.add_argument(
        "--model", type=str, default="CW", help="Ising models: implemented so far [CW, SK]"
    )
    parser.add_argument(
        "--save_net", type=str, default="yes", help="Save net. String: yes or no"
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
        "--N", type=int, default=100, help="Number of spins"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for random init"
    )
    parser.add_argument(
        "--J", type=float, default=1.0, help="j"
    )
    parser.add_argument(
        "--h", type=float, default=0.0, help="external field"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--std_fe_limit", type=float, default=1e-4, help="stopping threshold on the std of free energy"
    )
    parser.add_argument(
        "--max_step", type=int, default=100, help="Max number of steps for each value of beta",
    )
    parser.add_argument(
        "--init_steps", type=int, default=5000, help="Number of learning steps at the beginnig of annealing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size in the learning process",
    )
    parser.add_argument(
        "--batch_iter", type=int, default=20, help="Number of iteration in the computations of average quantities (tot num of samples batch_iter*batch_size)",
    )
    parser.add_argument(
        "--stats_step", type=int, default=1, help="Number of iterations for saving stats. Default 1.",
    )
    parser.add_argument(
        "--num_threads", type=int, default=1, help="Setting the number of threads of pytorch. Default 1.",
    )
    parser.add_argument(
        "--beta_range", type=float, default=[0.01, 2.0, 200], help="write start_beta end_beta number of steps. Ex. 0.01 2 200", nargs="+"
    )

    args = parser.parse_args()
    print("arguments:")
    print(args)
    return args


def CW_case(args):

    device = args.device
    N = args.N
    JJ = args.J
    hh = args.h
    net_spec = args.net_spec
    lr = args.lr
    max_step = args.max_step
    batch_size = args.batch_size
    std_fe_limit = args.std_fe_limit
    batch_iter = args.batch_iter
    suffix = args.suffix if args.suffix != None else net_spec
    stats_step = args.stats_step
    H = torch.ones(N)
    J_interaction = torch.ones(N, N)
    J = torch.ones(N, N)

    betas_r = tuple(args.beta_range)
    betas = np.linspace(betas_r[0], betas_r[1], int(betas_r[2]))
    stats = []

    H = hh * H
    J_interaction = J_interaction - torch.eye(N, N)
    J_val = JJ/(2*N)
    J = J_val * J_interaction
    CW_model = spins_model.KW_exact_fast(
        N, H, J, J_interaction, device=device)

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

    elif net_spec == "CWARNN":
        input_mask = torch.tril(J_interaction, diagonal=-1)
        input_mask = input_mask.to(dtype=torch.bool)
        net = cw_arnn.CWARNN(
            CW_model, input_mask, device=device)
        stats = run_lib.train_net(net,
                                  betas,
                                  lr=lr,
                                  max_step=max_step,
                                  batch_size=batch_size,
                                  std_fe_limit=std_fe_limit,
                                  suffix=suffix,
                                  batch_iter=batch_iter,
                                  stats_step=stats_step,
                                  exact=True,
                                  init_steps=args.init_steps
                                  )
        stats["num_params"] = net.num_params(train=False)
        stats["num_train_params"] = net.num_params(train=True)
    else:
        net = {}
        if net_spec == "SL":
            net = simple_layer.simple_layer(
                CW_model, device=device)
        elif net_spec == "MADE":
            net = made.MADE(CW_model, bias=True, device=device)
        elif net_spec == "MADE_21":
            net = made.MADE(CW_model,
                            bias=True,
                            device=device,
                            net_depth=2,
                            net_width=1
                            )
        elif net_spec == "MADE_22":
            net = made.MADE(CW_model,
                            bias=True,
                            device=device,
                            net_depth=2,
                            net_width=2
                            )
        elif net_spec == "1Par":
            net = cw_arnn.oneP(CW_model,
                              device=device,
                              )
        elif net_spec == "CWARNN_inf":
            net = cw_arnn.CWARNN_inf(CW_model,
                                    device=device,
                                    )
        elif net_spec == "CWARNN_free":
            list_n = h2_arnn.CWARNN
            input_mask = torch.tril(J_interaction, diagonal=-1)
            input_mask = input_mask.to(dtype=torch.bool)
            net = h2_arnn.list_nets(
                CW_model, list_n, input_mask, device=device)
        else:
            print("ERROR: Nets not found!")
            return 0
        stats = run_lib.train_net(net,
                                  betas,
                                  lr=lr,
                                  max_step=max_step,
                                  batch_size=batch_size,
                                  std_fe_limit=std_fe_limit,
                                  suffix=suffix,
                                  batch_iter=batch_iter,
                                  stats_step=stats_step,
                                  save_net=args.save_net == "yes",
                                  namefile_net=file_name(args, net=True),
                                  init_steps=args.init_steps
                                  )
        stats["num_params"] = net.num_params(train=False)
        stats["num_train_params"] = net.num_params(train=True)

    return stats


def SK_case(args):

    device = args.device
    N = args.N
    JJ = args.J
    hh = args.h
    net_spec = args.net_spec
    lr = args.lr
    max_step = args.max_step
    batch_size = args.batch_size
    std_fe_limit = args.std_fe_limit
    batch_iter = args.batch_iter
    suffix = args.suffix if args.suffix != None else net_spec
    stats_step = args.stats_step

    betas_r = tuple(args.beta_range)
    betas = np.linspace(betas_r[0], betas_r[1], int(betas_r[2]))
    stats = []

    H = torch.ones(N)
    H = hh * H

    J_interaction = torch.ones(N, N) - torch.eye(N, N)
    J_prob = graph_gen.spin_glass(N, J=JJ, J_0=0)
    J = graph_gen.set_J(J_interaction, J_prob)
    SK_model = spins_model.model(N, H, J, J_interaction, device=device)

    dict_nets = {"set_exact": False}

    if net_spec == "exact":
        for beta_ in betas:
            stats.append(SK_model.exact(beta_))
        stats = pd.DataFrame(stats)
        stats = stats.add_suffix(suffix)
    else:
        if net_spec == "SK_rs_learn":
            dict_nets = {"set_exact": False}
            list_n = list_nets.SK_net_rs
            input_mask = torch.tril(J_interaction, diagonal=-1)
            input_mask = input_mask.to(dtype=torch.bool)
            net = list_nets.list_nets(
                SK_model, list_n, input_mask, device=device, dict_nets=dict_nets)

        elif net_spec == "SK_0rsb":
            list_n = list_nets.SK_net_krsb
            learn = False
            list_n.learn_first_l = learn
            list_n.set_params_exact(SK_model, 0.1)
            input_mask = torch.tril(J_interaction, diagonal=-1)
            input_mask = input_mask.to(dtype=torch.bool)
            dict_nets = {"k": 0, "set_exact": learn}
            net = list_nets.list_nets(
                SK_model, list_n, input_mask, device=device, dict_nets=dict_nets)

        elif net_spec == "SK_1rsb":
            list_n = list_nets.SK_net_krsb
            learn = False
            list_n.learn_first_l = learn
            list_n.set_params_exact(SK_model, 0.1)
            input_mask = torch.tril(J_interaction, diagonal=-1)
            input_mask = input_mask.to(dtype=torch.bool)
            dict_nets = {"k": 1, "set_exact": learn}
            net = list_nets.list_nets(
                SK_model, list_n, input_mask, device=device, dict_nets=dict_nets)

        elif net_spec == "SK_2rsb":
            list_n = list_nets.SK_net_krsb
            learn = False
            list_n.learn_first_l = learn
            list_n.set_params_exact(SK_model, 0.1)
            input_mask = torch.tril(J_interaction, diagonal=-1)
            input_mask = input_mask.to(dtype=torch.bool)
            dict_nets = {"k": 2, "set_exact": learn}
            net = list_nets.list_nets(
                SK_model, list_n, input_mask, device=device, dict_nets=dict_nets)

        elif net_spec == "SL":
            net = simple_layer.simple_layer(
                SK_model, device=device)
        elif net_spec == "MADE_21":
            net = made.MADE(SK_model,
                            bias=True,
                            device=device,
                            net_depth=2,
                            net_width=1
                            )
        elif net_spec == "MADE_22":
            net = made.MADE(SK_model,
                            bias=True,
                            device=device,
                            net_depth=2,
                            net_width=2
                            )
        else:
            print("ERROR: Nets not found!")
            return 0
        stats = run_lib.train_net(net,
                                  betas,
                                  lr=lr,
                                  max_step=max_step,
                                  batch_size=batch_size,
                                  std_fe_limit=std_fe_limit,
                                  suffix=suffix,
                                  batch_iter=batch_iter,
                                  stats_step=stats_step,
                                  save_net=args.save_net == "yes",
                                  namefile_net=file_name(args, net=True),
                                  init_steps=args.init_steps
                                  )
        stats["num_params"] = net.num_params(train=False)
        stats["num_train_params"] = net.num_params(train=True)

    return stats


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    stats = pd.DataFrame()

    if args.model == "CW":
        stats = CW_case(args)
    if args.model == "SK":
        stats = SK_case(args)

    for ff in vars(args):
        #print(ff, vars(args)[ff])
        stats[str(ff)] = str(vars(args)[ff])

    timestr = time.strftime("%Y%m%d_%H%M%S")
    file_name_str = file_name(args, net=False) + "_" + timestr + ".gzip"
    stats.to_pickle(file_name_str)

    return True


main()

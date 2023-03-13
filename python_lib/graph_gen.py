import networkx as nx
import numpy as np
from collections import defaultdict
import math
import torch
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import random


def fixed_value(val=1):
    return lambda: val


def spin_glass(N, J=1, J_0=0):
    sigma = J / np.sqrt(N)
    mu = J_0 / N
    return lambda: np.random.normal(loc=mu, scale=sigma)


def spin_glass_one():
    return lambda: 1 if random.random() < 0.5 else -1


def BA_interaction(n, m, rand=False):
    G = nx.barabasi_albert_graph(n, m)

    pos = graphviz_layout(G, prog="twopi", args="")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=500, with_labels=True)
    plt.axis("equal")
    plt.show()
    adiacency_dict = {}
    for n, nbrdict in G.adjacency():
        adiacency_dict[n] = nbrdict.keys()

    num_nodes = len(adiacency_dict)
    adiacency_matrix = np.zeros((num_nodes, num_nodes))
    for n in range(num_nodes):
        for n_n in range(num_nodes):
            is_there = n_n in adiacency_dict[n]
            if n_n in adiacency_dict[n]:
                adiacency_matrix[n][n_n] = 1
    assert adiacency_matrix.size == num_nodes * num_nodes
    return num_nodes, adiacency_matrix


def tree_interaction(d, h, rand=False):
    G = nx.balanced_tree(d - 1, h)
    if rand:
        a = sorted(G).copy()
        random.shuffle(a)
        mapping = {}
        for i, n in enumerate(sorted(G)):
            mapping[n] = a[i]
        G = nx.relabel_nodes(G, mapping)

    pos = graphviz_layout(G, prog="twopi", args="")
    # plt.figure(figsize=(8, 8))
    # nx.draw(G, pos, node_size=500,  with_labels=True)
    # plt.axis('equal')
    # plt.show()
    adiacency_dict = {}
    for n, nbrdict in G.adjacency():
        adiacency_dict[n] = nbrdict.keys()

    num_nodes = len(adiacency_dict)
    adiacency_matrix = np.zeros((num_nodes, num_nodes))
    for n in range(num_nodes):
        for n_n in range(num_nodes):
            is_there = n_n in adiacency_dict[n]
            if n_n in adiacency_dict[n]:
                adiacency_matrix[n][n_n] = 1
    assert adiacency_matrix.size == num_nodes * num_nodes
    return num_nodes, adiacency_matrix, G, pos


def plot_matrix_graph(J, ax, type_="any"):
    if torch.is_tensor(J):
        J = J.cpu().detach().numpy()

    G = nx.from_numpy_matrix(J)
    if type_ == "grid":
        pos = nx.spring_layout(G, iterations=1000)
    else:
        pos = graphviz_layout(G, prog="neato", args="")
    nx.draw(G, pos, ax=ax, node_size=100, with_labels=True)


def grid_2d_interaction(n, m, periodic=False):
    G = nx.grid_2d_graph(n, m, periodic=periodic)
    pos = graphviz_layout(G, prog="neato", args="")
    # plt.figure(figsize=(8, 8))
    # nx.draw(G, pos, node_size=500,  with_labels=True)
    # plt.axis('equal')
    # plt.show()
    adiacency_dict = {}
    for index, nbrdict in G.adjacency():
        pos_n = index[0] * n + index[1]
        neighs = [pp[0] * n + pp[1] for pp in nbrdict.keys()]
        adiacency_dict[pos_n] = neighs

    num_nodes = len(adiacency_dict)
    adiacency_matrix = np.zeros((num_nodes, num_nodes))
    for index in range(num_nodes):
        for n_n in range(num_nodes):
            is_there = n_n in adiacency_dict[index]
            if n_n in adiacency_dict[index]:
                adiacency_matrix[index][n_n] = 1
    assert adiacency_matrix.size == num_nodes * num_nodes
    plot_matrix_graph(adiacency_matrix, type_="grid")
    return num_nodes, adiacency_matrix


def set_J(J_inter, value_func):
    '''Set the lower-triangular-matrix values of J'''
    J = np.zeros(J_inter.shape)
    for i in range(len(J)):
        for j in range(i):
            if J_inter[i][j] != 0:
                J[i][j] = value_func()
    # J = (J + J.transpose())

    return J


def set_H(H, value_func):
    for i in range(len(H)):
        if H[i] != 0:
            H[i] = value_func()


def order_rand(N, J_interaction, H, num_swap=1):
    swaps = []
    nodes = list(range(N))
    for n in range(num_swap):
        ss_temp = random.sample(nodes, 2)
        swaps.append(ss_temp)
        # print(ss_temp)
        nodes.remove(ss_temp[0])
        nodes.remove(ss_temp[1])

    print(swaps)
    J_interaction_rand = J_interaction.copy()
    H_rand = H.copy()

    for w in swaps:
        rev = list(reversed(w))
        J_interaction_rand[w] = J_interaction_rand[rev]
        J_interaction_rand[:, w] = J_interaction_rand[:, rev]
        H_rand[w] = H_rand[rev]
    return J_interaction_rand, H_rand


def map_reorder_graph(J_interaction, root=0):
    map_nodes = {}
    root = 0
    counter = 0
    visited = {}
    to_visit = [root]
    while len(to_visit) != 0:
        node = to_visit.pop(0)
        neighs = J_interaction[node]
        map_nodes[node] = counter
        counter += 1
        # print(to_visit, map_nodes)
        for neigh, is_neigh in enumerate(neighs):
            if is_neigh:
                if neigh not in to_visit and neigh not in map_nodes:
                    to_visit.append(neigh)
    return map_nodes


def reorder_graph(J_interaction, root=0):
    map_node = map_reorder_graph(J_interaction, root=0)
    adiacent_list = defaultdict(list)
    for node, neighs in enumerate(J_interaction):
        for neigh, is_neigh in enumerate(neighs):
            if is_neigh:
                adiacent_list[node].append(neigh)
    adiacent_list
    new_adiacent_list = defaultdict(list)
    new_J_interaction = np.zeros(J_interaction.shape)
    for node, neighs in enumerate(J_interaction):
        for neigh, is_neigh in enumerate(neighs):
            if is_neigh:
                new_J_interaction[map_node[node]][map_node[neigh]] = 1

    return new_J_interaction

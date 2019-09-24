import networkx as nx
import numpy as np
from collections import defaultdict
import math
import torch
import matplotlib.pyplot as plt
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
import random

def fixed_value(val = 1):
    return lambda : val

def spin_glass(mu=0, sigma=1):
    
    return lambda : random.gauss(mu, sigma)

def spin_glass_one():
    return lambda : 1 if random.random() < 0.5 else -1

def BA_interaction(n, m, rand=False):
    G = nx.barabasi_albert_graph(n,m)

    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=500,  with_labels=True)
    plt.axis('equal')
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
    G = nx.balanced_tree(d-1, h)
    if rand:
        a = sorted(G).copy()
        random.shuffle(a)
        mapping={}
        for i, n in enumerate(sorted(G)):
            mapping[n] = a[i]
        G = nx.relabel_nodes(G, mapping)

    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=500,  with_labels=True)
    plt.axis('equal')
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

def plot_matrix_graph(J):
    G=nx.from_numpy_matrix(J)
    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=500,  with_labels=True)
    plt.axis('equal')
    plt.show()
    
def grid_2d_interaction(n, m, periodic=False):
    G = nx.grid_2d_graph(n,m, periodic=periodic)
    pos = graphviz_layout(G, prog='neato', args='')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=500,  with_labels=True)
    plt.axis('equal')
    plt.show()
    adiacency_dict = {}
    for index, nbrdict in G.adjacency():
        pos_n = index[0] * n + index[1]
        neighs = [pp[0]*n + pp[1] for pp in nbrdict.keys()]
        adiacency_dict[pos_n] = neighs

    num_nodes = len(adiacency_dict)
    adiacency_matrix = np.zeros((num_nodes, num_nodes))
    for index in range(num_nodes):
        for n_n in range(num_nodes):
            is_there = n_n in adiacency_dict[index]
            if n_n in adiacency_dict[index]:
                adiacency_matrix[index][n_n] = 1
    assert adiacency_matrix.size == num_nodes * num_nodes
    return num_nodes, adiacency_matrix


def set_J(J_inter, value_func):
    J = np.zeros(J_inter.shape)
    for i in range(len(J)):
        for j in range(i):
            if J_inter[i][j] != 0:
                J[i][j] = value_func()
    J = (J + J.transpose())            
    
    return J

def set_H(H, value_func):
    for i in range(len(H)):
        if H[i] != 0:
            H[i] = value_func()

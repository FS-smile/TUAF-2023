import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def node_iter(G):
    return G.nodes


def edge_iter(G):
    return G.edges


def edge_dict(G):
    edge_dict = G.edges
    return edge_dict


def node_dict(G):
    node_dict = G.nodes
    return node_dict

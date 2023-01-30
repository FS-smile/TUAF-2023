import networkx as nx
import numpy as np
import torch
import torch.utils.data
import util


class GraphSampler(torch.utils.data.Dataset):
    def __init__(self, G_list,  normalize=False, max_num_triples=0):
        self.adj_all = []
        self.feat_triple_all = []
        self.graph_label_all = []
        self.max_num_triples = max_num_triples
        self.feat_triple_dim = len(util.node_dict(G_list[0])[0]['feat_triple'])

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                I = np.eye(adj.shape[0])
                A_hat = adj+I
                D_hat = np.sum(A_hat, axis=0)
                d_hat = np.diag(np.power(D_hat, -0.5).flatten())
                norm_adj = d_hat.dot(A_hat).dot(d_hat)

            self.adj_all.append(norm_adj)
            self.graph_label_all.append(G.graph['label'])

            feat_label_list = np.zeros(
                (self.max_num_triples, self.feat_triple_dim), dtype=float)
            for i_f, u_f in enumerate(G.nodes()):
                feat_label_list[i_f, :] = util.node_dict(G)[u_f]['feat_triple']
            self.feat_triple_all.append(feat_label_list)

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_triples = adj.shape[0]
        if self.max_num_triples > num_triples:
            adj_padded = np.zeros((self.max_num_triples, self.max_num_triples))
            adj_padded[:num_triples, :num_triples] = adj
        else:
            adj_padded = adj
        return {'adj': adj_padded,  # adjacent matrix of triple-unit graph
                'feat_triple': self.feat_triple_all[idx],  # triple features
                'graph_label': self.graph_label_all[idx],  # graph label
                'num_triples': num_triples,  # the number of triples
                }

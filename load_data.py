import networkx as nx
import numpy as np
import os
import util


def read_graphfile(datadir, dataname, isTox=False):
    tox_node_label_num_dict = {'Tox21_ARE': 54, 'Tox21_AhR': 53,
                               'Tox21_HSE': 54, 'Tox21_MMP': 54, 'Tox21_p53': 54, 'Tox21_PPAR-gamma': 53}

    tox_edge_label_num_dict = {'Tox21_ARE': 4, 'Tox21_AhR': 4,
                               'Tox21_HSE': 4, 'Tox21_MMP': 4, 'Tox21_p53': 4, 'Tox21_PPAR-gamma': 4}
    if(isTox):
        data_name = dataname.split('_')
        file_name = data_name[0]+'_'+data_name[1]
    prefix = os.path.join(datadir, dataname, dataname)

    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    try:
        with open(filename_graph_indic) as f:
            i = 1
            for line in f:
                line = line.strip("\n")
                graph_indic[i] = int(line)
                i += 1
    except IOError:
        print('No graph indicator')

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                if("," in line):
                    line = line.strip(",")
                    node_labels += [int(line[0])]
                else:
                    node_labels += [int(line)]
        node_label_map_to_int = {
            item: i+1 for i, item in enumerate(np.unique(node_labels))}
        node_labels = np.array([node_label_map_to_int[l]
                               for l in node_labels])
        num_unique_node_labels = max(node_labels)

    except IOError:
        print('No node labels')

    filename_edge = prefix + '_edge_labels.txt'
    edge_labels = []
    one_hot_edge_labels = []
    try:
        with open(filename_edge) as f:
            for line in f:
                line = line.strip("\n")
                edge_labels += [int(line) - 1]

        edge_label_map_to_int = {
            item: i+1 for i, item in enumerate(np.unique(edge_labels))}
        edge_labels = np.array([edge_label_map_to_int[l]
                               for l in edge_labels])
        num_unique_edge_labels = max(edge_labels)
        for l in edge_labels:
            if(isTox):
                edge_label_one_hot = [0]*tox_edge_label_num_dict[file_name]
            else:
                edge_label_one_hot = [0]*num_unique_edge_labels
            edge_label_one_hot[l-1] = 1
            one_hot_edge_labels.append(edge_label_one_hot)
    except IOError:
        print('No edge labels')

    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    label_map_to_int = {val: i for i, val in enumerate(
        np.sort(label_vals))}
    print("Label Values, Label Mapping Dict", label_vals, label_map_to_int)
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels)+1)}
    index_graph = {i: [] for i in range(1, len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            if(len(edge_labels) > 0):
                adj_list[graph_indic[e0]].append(
                    (e0, e1, {'edge_label': one_hot_edge_labels[num_edges]}))
            else:
                one_hot_edge_label = [0] * num_unique_node_labels
                adj_list[graph_indic[e0]].append(
                    (e0, e1, {'edge_label': one_hot_edge_label}))

            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u-1 for u in set(index_graph[k])]

    graphs = []
    max_edge_num = 0
    for i in range(1, 1+len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        if len(G.edges) > max_edge_num:
            max_edge_num = len(G.edges)
        G.graph['label'] = graph_labels[i-1]
        for index, u in enumerate(util.node_iter(G)):
            if len(node_labels) > 0:
                if(isTox):
                    node_label_one_hot = [0] * \
                        tox_node_label_num_dict[file_name]
                else:
                    node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label-1] = 1
                util.node_dict(G)[u]['node_label'] = node_label_one_hot

        mapping = {}
        it = 0
        for n in util.node_iter(G):
            mapping[n] = it
            it += 1
        graphs.append(nx.relabel_nodes(G, mapping))

    max_nodes_num = max([G.number_of_nodes() for G in graphs])

    return graphs, max_nodes_num, max_edge_num

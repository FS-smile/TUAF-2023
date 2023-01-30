# -*- coding: utf-8 -*-
import torch
import csv
import time
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import random_split
from graph_sampler import GraphSampler
from Mymodel import outlier_model
from load_data import read_graphfile
from convert_triple_graph import convert_to_triple_graph
from loss import loss_function, init_center
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

def arg_parse():
    parser = argparse.ArgumentParser(description='TUAF Arguments.')
    parser.add_argument('--gpu', dest='gpu', type=int, default=0, help='gpu')
    parser.add_argument('--datadir', dest='datadir', default='./dataset', help='The directory where the dataset is located.')
    parser.add_argument('--dataset', dest='dataset', default='AIDS', help='Dataset name.')
    parser.add_argument("--outlier_label", dest='outlier_label', type=int,
                        default=0, help="Select which label is outlier.")  # Note: The value is the mapped label
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=50, type=int, help='Total epoch number.')
    parser.add_argument('--batch_size', dest='batch_size', default=1000, type=int, help='Batch size.')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension.')
    parser.add_argument('--output_dim', dest='output_dim', default=256, type=int, help='Output dimension.')
    parser.add_argument('--bn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used.')
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate.')
    parser.add_argument('--bias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--seed', dest='seed', type=int, default=999, help='Random seed.')
    parser.add_argument('--scheduler', dest='scheduler', action='store_const', const=False, default=True, help='Wether scheduler is used.')
    parser.add_argument('--step_size', dest='step_size', type=int, default=30, help='Step_size.')
    parser.add_argument('--gamma', dest='gamma', type=int, default=0.3, help='Gamma.')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--normalize', dest='normalize', action='store_const', const=True, 
                        default=True, help='Whether adjacent matrix normalization is used.')

    return parser.parse_args()

# Random seed setup
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Training process
def train(data_train_loader, data_valid_loader, model, dataname,args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if(args.scheduler):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    center = init_center(args, data_train_loader, model)
    
    min_auc_val = 0
    best_epoch = 0
    for epoch in tqdm(range(args.num_epochs)):
        min_epoch = 10
        total_loss_train = 0.0
        model.train()

        for batch_idx, data in enumerate(data_train_loader): 
            optimizer.zero_grad()  
            x_tr = Variable(data['feat_triple'].float(),
                                  requires_grad=False).cuda(args.gpu)  
            tuple_adj_tr = Variable(data['adj'].float(),
                                    requires_grad=False).cuda(args.gpu)
            re_tuple_adj,  h_tr = model(
                x_tr, tuple_adj_tr)
            loss_tr, dist_tr,  score_tr = loss_function(args, re_tuple_adj,  tuple_adj_tr,
                                                                     center,  h_tr)
            loss_tr.backward()
            optimizer.step()
            total_loss_train += loss_tr

        if(args.scheduler):
            scheduler.step()

        if epoch >= 0:  
            total_loss_valid = 0.0
            score_valid = []
            y = []
            emb_h_tr = []

            model.eval() 
            for batch_idx, data in enumerate(data_valid_loader):
                x_v = Variable(data['feat_triple'].float(),
                                     requires_grad=False).cuda(args.gpu) 
                tuple_adj_v = Variable(data['adj'].float(),
                                       requires_grad=False).cuda(args.gpu)
                re_adj_v,  h_v = model(
                    x_v, tuple_adj_v)
                loss_v, dist_v,  score_v = loss_function(args, re_adj_v,  tuple_adj_v,
                                                                      center,  h_v)

                valid_loss = np.array(loss_v.cpu().detach())
                total_loss_valid += valid_loss

                score_v_ = np.array(score_v.cpu().detach())
                score_valid.append(score_v_)

                if data['graph_label'] == args.outlier_label:
                    y.append(1)
                else:
                    y.append(0)
                emb_h_tr.append(h_v.cpu().detach().numpy())

            label_valid = np.array(score_valid)
            fpr_ab, tpr_ab, _ = roc_curve(y, label_valid)
            valid_roc_ab = auc(fpr_ab, tpr_ab)
            print("epoch:", epoch, "; total_loss_valid: %.10f" %
                  total_loss_valid, "; valid auc: %.10f" % valid_roc_ab)

        if epoch > min_epoch and valid_roc_ab >= min_auc_val:  
            min_auc_val = valid_roc_ab
            best_model = copy.deepcopy(model)
            print("valid_roc_ab", valid_roc_ab, "total_loss_valid: %.10f" %
                  total_loss_valid, "min_auc_val: %.10f" % min_auc_val)
            best_epoch = epoch
    best_model_path = "model/"+dataname+".pt"
    torch.save(best_model.state_dict(), best_model_path)
    print("Best Epoch:", best_epoch, "; Total Loss Valid: %.10f" % 
          total_loss_valid, "; Min Auc Val: %.10f" %  min_auc_val)
    return best_epoch, best_model_path, center


def test(data_test_loader, model,  best_model_path, center, dataname,args):

    model.load_state_dict(torch.load(best_model_path))
    loss_test = []
    emb_h_t = []
    total_loss_test = []
    y = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_test_loader):
            x_label = Variable(data['feat_triple'].float(),
                               requires_grad=False).cuda(args.gpu)  
            tuple_adj_t = Variable(data['adj'].float(),
                                 requires_grad=False).cuda(args.gpu)
            re_label_adj, h_label_ = model(
                x_label, tuple_adj_t)
            loss_t, label_dist,  score_tr = loss_function(args, re_label_adj,  tuple_adj_t,
                                                              center,  h_label_)
            test_loss = np.array(loss_t.cpu().detach())
            total_loss_test += test_loss

            loss_ = np.array(score_tr.cpu().detach())
            loss_test.append(loss_)
            if data['graph_label'] == args.outlier_label:
                y.append(1)
            else:
                y.append(0)
            emb_h_t.append(h_label_.cpu().detach().numpy())

    label_test = np.array(loss_test)

    fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
    test_roc_ab = auc(fpr_ab, tpr_ab)
    

    return test_roc_ab


if __name__ == '__main__':
        
    args = arg_parse()
    setup_seed(args.seed)
    dataname = args.dataset
    
    result_raw=[]
    filepath="results.csv"
    with open(filepath,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["Model","Dataset","AUC","OutlierClass","Seed","Batch_Size","Epoch_Num","Optimizer","LR","Step_Size","Gamma","Clip","Hidden_Dim","Output_Dim","Others","TimeStamp"])
    
    result_list=[]
    orign_graphs, max_node_num, max_edge_num = read_graphfile(
        args.datadir, dataname, isTox=False)
    datanum = len(orign_graphs)

    label_0 = [i for i, data in enumerate(
        orign_graphs) if data.graph['label'] == 0]
    label_1 = [i for i, data in enumerate(
        orign_graphs) if data.graph['label'] == 1]
    label_dist=[len(label_0),len(label_1)]

    # convert to triple-unit graphss
    print("Begin Conversion------------")
    graphs = []
    print("The total graph number: ", datanum)
    for index in range(0, len(orign_graphs)):
        if(index % 200 == 0):
            print(index)
        graphs.append(convert_to_triple_graph(
            orign_graphs[index]))
    print("End Conversion------------")

    print("graph_label(=0):", len(label_0), "graph_label(=1):",len(label_1))

    max_triples_num = max([G.number_of_nodes() for G in graphs])

    graphs_label = [graph.graph['label'] for graph in graphs]
    graphs_label_ = np.unique(graphs_label)

    feat_dim = len(graphs[0].nodes[0]['feat_triple'])
    print("The input feature dimension: ", feat_dim)

    # split the training and testing set
    num_training = int(len(graphs) * 0.9)
    num_test = len(graphs) - (num_training)
    train_graphs, test_graphs = random_split(
        graphs, [num_training,  num_test])

    train_graphs_label = [graph.graph['label'] for graph in train_graphs]
    test_graphs_label = [graph.graph['label'] for graph in test_graphs]

    dataset_sampler_test = GraphSampler(
        test_graphs, normalize=args.normalize, max_num_triples=max_triples_num)

    data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test,
                                                shuffle=False,
                                                batch_size=1)

    # 5-fold cross-validation
    kfd = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    result_auc = []
    for k, (train_index, valid_index) in enumerate(kfd.split(train_graphs, train_graphs_label)):
        print("======================The", k, "Fold==============================")

        graphs_train_ = [graphs[i] for i in train_index]
        graphs_valid = [graphs[i] for i in valid_index]

        graphs_train = []
        for graph in graphs_train_:
            if graph.graph['label'] != args.outlier_label:  #Using the normal graphs in the training process
                graphs_train.append(graph)

        graphs_train_label = [G.graph['label'] for G in graphs_train]
        graphs_valid_label = [G.graph['label'] for G in graphs_valid]
        print("training set label, valid set label: ",
            np.unique(graphs_train_label), np.unique(graphs_valid_label))

        num_train = len(graphs_train)
        num_valid = len(graphs_valid)
        print("The number of training set, the number of valid set: ", num_train, num_valid)

        dataset_sampler_train = GraphSampler(
            graphs_train, normalize=args.normalize, max_num_triples=max_triples_num)

        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train,
                                                        shuffle=True,
                                                        batch_size=args.batch_size)
        dataset_sampler_valid = GraphSampler(
            graphs_valid,  normalize=args.normalize, max_num_triples=max_triples_num)

        data_valid_loader = torch.utils.data.DataLoader(dataset_sampler_valid,
                                                        shuffle=False,
                                                        batch_size=1)

        my_model = outlier_model(feat_dim, args.hidden_dim, args.output_dim, args.dropout).cuda(args.gpu)
        # Training Process
        best_epoch, best_model_path, center = train(
            data_train_loader, data_valid_loader, my_model, dataname,args)
        # Testing Process
        result = test(data_test_loader, my_model,  best_model_path,
                    center, dataname,args)
        print('The currrent test dataset:',dataname)
        print('The test AUC of','the',k,'fold:','{}'.format(result))
        result_list.append(result)
        
    result_auc = np.array(result_list)
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)

    print('(TUAF) Dataset:',dataname,'\nfinal_result: {:.4f}±{:.4f}'.format(round(auc_avg, 4), round(auc_std, 4)))
    
    result_raw=['TUAF',dataname,str(format(auc_avg, '.4f'))+"±"+str(format(auc_std, '.4f')),args.outlier_label,args.seed,args.batch_size,args.num_epochs,'SGD',args.lr,args.step_size,args.gamma,args.clip,args.hidden_dim,args.output_dim,result_auc,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]
    with open(filepath,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(result_raw)
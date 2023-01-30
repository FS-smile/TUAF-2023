import torch
from torch.autograd import Variable


def loss_function(args, re_triple_adj,  tuple_adj,  center, h):
    dist, scores = anomaly_score(center, h)
    loss = torch.mean(dist) + torch.mean((re_triple_adj - tuple_adj) ** 2)
    return loss, dist, scores


def anomaly_score(data_center, outputs):
    dist = torch.sum((outputs - data_center) ** 2, dim=1)
    scores = torch.sqrt(dist)
    return dist, scores

# Initialize center c
def init_center(args, data, model, eps=0.001):
    outputs = []
    c = torch.zeros(args.output_dim).cuda(args.gpu)
    model.eval()
    with torch.no_grad():
        for index, g in enumerate(data):
            x = Variable(g['feat_triple'].float(),
                               requires_grad=False).cuda(args.gpu)
            tuple_adj = Variable(g['adj'].float(),
                                 requires_grad=False).cuda(args.gpu)
            _,  h = model(x, tuple_adj)
            outputs.append(torch.mean(h, dim=0))
        if len(outputs) == 1:
            outputs = torch.unsqueeze(outputs[0], 0)
        else:
            outputs = torch.stack(outputs, 0)

        # get the inputs of the batch
        n_samples = outputs.shape[0]
        c = torch.sum(outputs, dim=0)
    c /= n_samples

    # If c_i is too close to 0, set to +-eps.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

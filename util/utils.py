import copy

import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from dataset.aliexpress import AliExpressDataset
from dataset.movielens import MovieLensDataset
from model.dlrm import DLRMModel
from model.mmoe_v2 import MMoEModel


def get_dataset(name, path, groups_num):
    if 'AliExpress' in name:
        dataset = AliExpressDataset(path, groups_num)
        return dataset, dataset.user_groups
    elif 'MovieLens' in name:
        dataset = MovieLensDataset(path, groups_num)
        return dataset, dataset.user_groups
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    if name == "dlrm":
        print("Model: DLRM")
        return DLRMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(32, 16),
                         up_mlp_dims=(256, 128, 64), dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


def get_criterion(name, device):
    criterion = []
    for n in name.split(','):
        if n == 'bce':
            criterion.append(torch.nn.BCELoss().to(device))
        elif n == 'mse':
            criterion.append(torch.nn.MSELoss(reduction='mean').to(device))
        else:
            raise ValueError('unknown criterion name: ' + n)
    return criterion


def get_evaluation(name):
    evaluation = []
    for n in name.split(','):
        if n == 'auc':
            evaluation.append(roc_auc_score)
        elif n == 'rmse':
            evaluation.append(lambda y, y_hat: mean_squared_error(y, y_hat, squared=False))
        else:
            raise ValueError('unknown evaluation name: ' + n)
    return evaluation


def average_weights(w, embedding_name):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if key == embedding_name:
            continue
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_embeddings(w, global_weight, embedding_name):
    global_embedding_weight = global_weight[embedding_name]
    average_embedding_weight = copy.deepcopy(global_embedding_weight)
    word_num = global_embedding_weight.size()[0]
    embedding_dict = {}
    for word in range(word_num):
        embedding_dict[word] = []

    for i in range(len(w)):
        local_embedding_weight = w[i][embedding_name]
        for word in range(word_num):
            if local_embedding_weight[word].equal(global_embedding_weight[word]):
                continue
            embedding_dict[word].append(local_embedding_weight[word])

    for word in range(word_num):
        if len(embedding_dict[word]) == 0:
            average_embedding_weight[word] = global_embedding_weight[word]
        else:
            average_embedding_weight[word] = embedding_dict[word][0]
            for i in range(1, len(embedding_dict[word])):
                average_embedding_weight[word] += embedding_dict[word][i]
            average_embedding_weight[word] = torch.div(average_embedding_weight[word], len(embedding_dict[word]))

    return average_embedding_weight


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

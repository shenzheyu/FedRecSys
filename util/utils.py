import copy

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from dataset.aliexpress import AliExpressDataset
from dataset.movielens import MovieLensDataset
from model.dlrm import DLRMModel
from model.mmoe_v2 import MMoEModel
from model.wdl import WDLModel


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
        print('Model: MMoE')
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'dlrm':
        print('Model: DLRM')
        return DLRMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(32, 16),
                         up_mlp_dims=(256, 128, 64), dropout=0.2)
    elif name == 'wdl':
        print('Model: Wide&Deep')
        return WDLModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, deep_mlp_dims=(1024, 512, 256),
                        dropout=0.2)
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


def get_embedding_map(dataset, idxs, field_dims):
    embedding_map = {}
    global_offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
    embedding_idx = 0
    for idx in idxs:
        categorical_fields, _, _ = dataset[idx]
        for i in range(len(field_dims)):
            if categorical_fields[i] + global_offset[i] not in embedding_map.keys():
                embedding_map[categorical_fields[i] + global_offset[i]] = embedding_idx
                embedding_idx += 1
    return embedding_map


def average_weights(w, global_weight, embedding_name):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(global_weight)
    for key in w_avg.keys():
        if key == embedding_name:
            continue
        w_avg[key] = w[0][key]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_embeddings(local_weights, local_embedding_maps, global_weight, embedding_name,
                       update_condition=lambda x: True):
    global_embedding_weight = global_weight[embedding_name]
    average_embedding_weight = copy.deepcopy(global_embedding_weight)
    word_num = global_embedding_weight.size()[0]
    embedding_map = {}
    for word in range(word_num):
        embedding_map[word] = []

    for idx in range(len(local_weights)):
        for global_word, word in local_embedding_maps[idx].items():
            # not average embedding for user_id
            if update_condition(global_word):
                embedding_map[global_word].append(local_weights[idx][embedding_name][word])

    for word in range(word_num):
        if len(embedding_map[word]) == 0:
            average_embedding_weight[word] = global_embedding_weight[word]
        else:
            average_embedding_weight[word] = embedding_map[word][0]
            for i in range(1, len(embedding_map[word])):
                average_embedding_weight[word] += embedding_map[word][i]
            average_embedding_weight[word] = torch.div(average_embedding_weight[word], len(embedding_map[word]))

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

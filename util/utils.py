import copy

import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from dataset.aliexpress import AliExpressDataset
from dataset.movielens import MovieLensDataset
from model.dlrm import DLRMModel
from model.mmoe_v2 import MMoEModel
from model.wdl import WDLModel


def get_dataset(name, path, groups_num):
    if "AliExpress" in name:
        dataset = AliExpressDataset(path, groups_num)
        return dataset, dataset.user_groups
    elif "MovieLens" in name:
        dataset = MovieLensDataset(path, groups_num)
        return dataset, dataset.user_groups
    else:
        raise ValueError("unknown dataset name: " + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == "mmoe":
        print("Model: MMoE")
        return MMoEModel(
            categorical_field_dims,
            numerical_num,
            embed_dim=embed_dim,
            bottom_mlp_dims=(512, 256),
            tower_mlp_dims=(128, 64),
            task_num=task_num,
            expert_num=expert_num,
            dropout=0.2,
        )
    elif name == "dlrm":
        print("Model: DLRM")
        return DLRMModel(
            categorical_field_dims,
            numerical_num,
            embed_dim=embed_dim,
            bottom_mlp_dims=(32, 16),
            up_mlp_dims=(256, 128, 64),
            dropout=0.2,
        )
    elif name == "wdl":
        print("Model: Wide&Deep")
        return WDLModel(
            categorical_field_dims, numerical_num, embed_dim=embed_dim, deep_mlp_dims=(1024, 512, 256), dropout=0.2
        )
    else:
        raise ValueError("unknown model name: " + name)


def get_criterion(name, device):
    criterion = []
    for n in name.split(","):
        if n == "bce":
            criterion.append(torch.nn.BCELoss().to(device))
        elif n == "mse":
            criterion.append(torch.nn.MSELoss(reduction="mean").to(device))
        else:
            raise ValueError("unknown criterion name: " + n)
    return criterion


def get_evaluation(name):
    evaluation = []
    for n in name.split(","):
        if n == "auc":
            evaluation.append(roc_auc_score)
        elif n == "rmse":
            evaluation.append(lambda y, y_hat: mean_squared_error(y, y_hat, squared=False))
        else:
            raise ValueError("unknown evaluation name: " + n)
    return evaluation


def average_weights(w, local_train_num, global_weight, embedding_name):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(global_weight)
    for key in w_avg.keys():
        if key == embedding_name:
            continue
        w_avg[key] = w[0][key]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * local_train_num[i]
        w_avg[key] = torch.div(w_avg[key], sum(local_train_num))
    return w_avg


def average_embeddings(
        local_weights, local_embedding_maps, local_update_times, global_weight, embedding_name,
        update_condition=lambda x: True
):
    global_embedding_weight = global_weight[embedding_name]
    average_embedding_weight = copy.deepcopy(global_embedding_weight)
    word_num = global_embedding_weight.size()[0]
    global_embedding_map = {}
    global_update_times = {}
    for word in range(word_num):
        global_embedding_map[word] = []
        global_update_times[word] = 0

    for idx in range(len(local_weights)):
        # local_embedding_maps = (global_embedding_word, local_embedding_word)
        for global_embedding_word, local_embedding_word in local_embedding_maps[idx].items():
            # average controlled by update_condition
            if update_condition(global_embedding_word) and local_embedding_word in local_update_times[idx].keys():
                global_embedding_map[global_embedding_word].append(
                    local_weights[idx][embedding_name][local_embedding_word] * local_update_times[idx][local_embedding_word])
                global_update_times[global_embedding_word] += local_update_times[idx][local_embedding_word]

    for global_embedding_word in range(word_num):
        if len(global_embedding_map[global_embedding_word]) != 0:
            average_embedding_weight[global_embedding_word] = global_embedding_map[global_embedding_word][0]  # [128]
            for i in range(1, len(global_embedding_map[global_embedding_word])):
                average_embedding_weight[global_embedding_word] += global_embedding_map[global_embedding_word][i]
            # TODO: change it sample number-based weighted averaging (FedAvg); how about FedNova and FedOpt?
            # https://www.diva-portal.org/smash/get/diva2:1553472/FULLTEXT01.pdf
            average_embedding_weight[global_embedding_word] = torch.div(
                average_embedding_weight[global_embedding_word], global_update_times[global_embedding_word]
            )

    return average_embedding_weight


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

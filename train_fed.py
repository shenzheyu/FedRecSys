import copy
import random
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model.embedding_lookup import EmbeddingLookup
from model.mmoe import SparseMMOE
from util.data_prepare import data_prepare
from util.data_prepare import get_dataset
from util.movielens_dataset import MovielensDataset
from util.user_cluster_data import UserClusterData


class ClientThread(Thread):
    def __init__(self, client_idx, user_cluster_datas, model, embedding_lookups):
        super(ClientThread, self).__init__()
        self.sub_model = None
        self.client_idx = client_idx
        self.user_cluster_datas = user_cluster_datas
        self.model = model
        self.embedding_lookups = embedding_lookups

    def run(self) -> None:
        train_dataset, val_dataset, test_dataset = get_dataset(
            {self.client_idx: self.user_cluster_datas[self.client_idx]})
        self.sub_model, embedding_weights, embedding2idxes = client_prepare(train_dataset, val_dataset, test_dataset,
                                                                            self.model, self.embedding_lookups)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,
                                      collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                                    collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False,
                                     collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
        client_train(self.client_idx, self.sub_model, train_dataloader, val_dataloader, test_dataloader)

    def get_result(self):
        return self.sub_model


def client_prepare(train_dataset, val_dataset, test_dataset, global_model: SparseMMOE, embedding_lookups: dict):
    embedding_weights = {}
    embedding2idxes = {}
    for embedding_key in embedding_lookups.keys():
        embedding_weights[embedding_key], embedding2idxes[embedding_key] = \
            embedding_lookups[embedding_key].get_embeddings([train_dataset, val_dataset, test_dataset])

    local_model = copy.deepcopy(global_model)
    local_model.embedding2idxes = embedding2idxes
    for embedding_key in embedding_lookups.keys():
        local_model.embeddings[embedding_key] = nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(embedding_weights[embedding_key]), freeze=False)

    return local_model, embedding_weights, embedding2idxes


def merge_model(global_model: SparseMMOE, embedding_lookups: dict[str, EmbeddingLookup], local_models: [SparseMMOE]):
    # merge model
    global_weight = copy.deepcopy(local_models[0].state_dict())
    global_weight = {k: v for k, v in global_weight.items() if 'embedding' not in k}
    for key in global_weight.keys():
        for i in range(1, len(local_models)):
            global_weight[key] += local_models[i].state_dict()[key]
        global_weight[key] = torch.div(global_weight[key], len(local_models))
    global_model.load_state_dict(global_weight, strict=False)

    # merge embedding
    for embedding_key in embedding_lookups.keys():
        for word in embedding_lookups[embedding_key].embeddings.keys():
            vec = np.zeros_like(embedding_lookups[embedding_key].embeddings[word])
            update_count = 0
            for submodel in local_models:
                embedding_weights = submodel.embeddings[embedding_key].weight.data
                embedding2idxes = submodel.embedding2idxes[embedding_key]
                if word in embedding2idxes.keys():
                    vec += embedding_weights[embedding2idxes[word]].detach().numpy()
                    update_count += 1
            if update_count != 0:
                embedding_lookups[embedding_key].update_embedding(word, vec / update_count)


def client_train(index: int, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 test_dataloader: DataLoader):
    n_epochs = 10
    lr = 0.001
    loss_fn_1 = nn.MSELoss()
    loss_fn_2 = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = [list(), list()]
        for features, labels in train_dataloader:
            if len(features) == 1:
                print('batch size is 1')
                continue
            labels_hat = model(features)

            label_1 = torch.FloatTensor([label['rating'] for label in labels if 'rating' in label.keys()])
            label_2 = torch.FloatTensor([label['retrieve'] for label in labels if 'retrieve' in label.keys()])
            label_hat_1 = labels_hat[0][['rating' in label for label in labels], 0] * 5
            label_hat_2 = labels_hat[0][['retrieve' in label for label in labels], 0]

            loss_1 = loss_fn_1(label_hat_1, label_1)
            loss_2 = loss_fn_2(label_hat_2, label_2)
            loss = loss_1 + loss_2
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss[0].append(loss_1.item())
            epoch_loss[1].append(loss_2.item())

        losses.append(np.mean(epoch_loss))
        rmse_1, auc_2 = client_test(model, val_dataloader)
        train_rmse_1, train_auc_2 = client_test(model, train_dataloader)
        # print(f'[client {index}] task 1 train loss: {np.mean(epoch_loss[0]):.5f}, t'
        #       f'ask 2 train loss: {np.mean(epoch_loss[1]):.5f}, '
        #       f'train loss: {np.mean(epoch_loss[0]) + np.mean(epoch_loss[1]):.5f}, '
        #       f'val task 1 rmse: {rmse_1:.3f}, '
        #       f'val task 2 auc: {auc_2:.3f}, '
        #       f'train task 1 rmse: {train_rmse_1:.3f}, '
        #       f'train task 2 auc: {train_auc_2:.3f}')

    rmse_1, auc_2 = client_test(model, test_dataloader)
    # print(f'[client {index}] val task 1 rmse: {rmse_1:.3f}, val task 2 auc: {auc_2:.3f}')


def client_test(model: nn.Module, dataloader: DataLoader):
    task_1_true, task_2_true, task_1_score, task_2_score = [], [], [], []
    model.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            labels_hat = model(features)

            label_1 = torch.FloatTensor([label['rating'] for label in labels if 'rating' in label.keys()])
            label_2 = torch.FloatTensor([label['retrieve'] for label in labels if 'retrieve' in label.keys()])
            label_hat_1 = labels_hat[0][['rating' in label for label in labels], 0] * 5
            label_hat_2 = labels_hat[0][['retrieve' in label for label in labels], 0]

            task_1_true += list(label_1.float())
            task_2_true += list(label_2.float())
            task_1_score += list(label_hat_1.float())
            task_2_score += list(label_hat_2.float())

    rmse_1 = np.sqrt(mean_squared_error(task_1_true, task_1_score))
    auc_2 = roc_auc_score(task_2_true, task_2_score)
    return rmse_1, auc_2


def client_main(client_idx, user_cluster_datas, model, embedding_lookups):
    # 3. call client_prepare
    train_dataset, val_dataset, test_dataset = get_dataset({client_idx: user_cluster_datas[client_idx]})
    sub_model, embedding_weights, embedding2idxes = client_prepare(train_dataset, val_dataset, test_dataset, model,
                                                                   embedding_lookups)

    # 4. call client_train
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True,
                                  collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                                collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False,
                                 collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
    client_train(sub_model, train_dataloader, val_dataloader, test_dataloader)

    return sub_model


def server_train(device='cpu'):
    # 1. init model, dataset, embedding
    embedding_lookups = {}
    embedding_list = ['user_id', 'user_zipcode', 'movie_id']
    for embedding_key in embedding_list:
        embedding_lookups[embedding_key] = EmbeddingLookup(8, key=embedding_key)

    user_cluster_datas = data_prepare()

    train_dataset, val_dataset, test_dataset = get_dataset(user_cluster_datas)
    embedding_weights = {}
    embedding2idxes = {}
    for embedding_key in embedding_list:
        embedding_weights[embedding_key], embedding2idxes[embedding_key] = \
            embedding_lookups[embedding_key].get_embeddings([train_dataset, val_dataset, test_dataset])
    global_model = SparseMMOE(embedding_list=embedding_lookups.keys(), pretrained_embeddings=embedding_weights,
                              embedding2idxes=embedding2idxes)

    epoch = 1000
    for i in range(epoch):
        # 2. select k client
        client_num = 10
        client_idx_list = list(range(1, len(user_cluster_datas) + 1))
        random.shuffle(client_idx_list)

        sub_models = []
        client_threads = []
        for client_idx in client_idx_list[:client_num]:
            client_thread = ClientThread(client_idx, user_cluster_datas, global_model, embedding_lookups)
            client_thread.start()
            client_threads.append(client_thread)

        for client_thread in client_threads:
            client_thread.join()
            sub_model = client_thread.get_result()
            sub_models.append(sub_model)

        # 5. merge model and embedding
        merge_model(global_model, embedding_lookups, sub_models)
        train_rmse_1, train_auc_2 = server_test(global_model, embedding_lookups, train_dataset)
        rmse_1, auc_2 = server_test(global_model, embedding_lookups, val_dataset)
        print(f'[server] train task 1 rmse: {train_rmse_1:.3f}, train task 2 auc: {train_auc_2:.3f}, ')
        print(f'[server] val task 1 rmse: {rmse_1:.3f}, val task 2 auc: {auc_2:.3f}, ')

    rmse_1, auc_2 = server_test(global_model, embedding_lookups, test_dataset)
    print(f'[server] test task 1 rmse: {rmse_1:.3f}, testtask 2 auc: {auc_2:.3f}, ')


def server_test(model: SparseMMOE, embedding_lookups: dict[str, EmbeddingLookup], dataset: MovielensDataset):
    for embedding_key in embedding_lookups.keys():
        embedding_weights, embedding2idxes = embedding_lookups[embedding_key].get_embeddings([dataset])
        model.embeddings[embedding_key] = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(embedding_weights),
                                                                       freeze=False)
        model.embedding2idxes[embedding_key] = embedding2idxes
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False,
                            collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
    rmse_1, auc_2 = client_test(model, dataloader)
    return rmse_1, auc_2


if __name__ == '__main__':
    server_train()

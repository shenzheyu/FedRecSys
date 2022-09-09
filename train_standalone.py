import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from model.embedding_lookup import EmbeddingLookup
from model.mmoe import SparseMMOE
from util.data_prepare import data_prepare
from util.data_prepare import get_dataset


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
    n_epochs = 1000
    lr = 0.001
    weight_decay = 1e-5
    loss_fn_1 = nn.MSELoss()
    loss_fn_2 = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    print('start train')
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = [list(), list()]
        for features, labels in train_dataloader:
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
        rmse_1, auc_2 = test(model, val_dataloader)
        train_rmse_1, train_auc_2 = test(model, train_dataloader)
        print(f'epoch: {epoch}, task 1 train loss: {np.mean(epoch_loss[0]):.5f}, '
              f'task 2 train loss: {np.mean(epoch_loss[1]):.5f}, '
              f'val task 1 rmse: {rmse_1:.3f}, val task 2 auc: {auc_2:.3f}, '
              f'train task 1 rmse: {train_rmse_1:.3f}, train task 2 auc: {train_auc_2:.3f}')

    rmse_1, auc_2 = test(model, test_dataloader)
    print(f'test task 1 rmse: {rmse_1:.3f}, test task 2 auc: {auc_2:.3f}')


def test(model: nn.Module, dataloader: DataLoader):
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


if __name__ == '__main__':
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

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, drop_last=True, shuffle=True, collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))

    model = SparseMMOE(embedding_list=embedding_list, pretrained_embeddings=embedding_weights, embedding2idxes=embedding2idxes)

    train(model, train_dataloader, val_dataloader, test_dataloader)

    for embedding_key in embedding_list:
        embedding_lookups[embedding_key].update_embeddings(model.embeddings[embedding_key].weight.data.tolist(),
                                                           embedding2idxes[embedding_key])

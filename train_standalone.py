import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader

from model.embedding_lookup import EmbeddingLookup
from model.mmoe import SparseMMOE
from util.data_prepare import data_prepare


def test(model: nn.Module, dataloader: DataLoader):
    task_1_true, task_2_true, task_1_score, task_2_score = [], [], [], []
    model.eval()
    with torch.no_grad():
        for features, labels, tasks in dataloader:
            labels_hat = model(features)

            label_1 = labels[tasks[:, 0] == 1, 0] - 1
            label_2 = labels[tasks[:, 1] == 1, 1]
            label_hat_1 = torch.argmax(labels_hat[0][tasks[:, 0] == 1], dim=1)
            label_hat_2 = labels_hat[1][tasks[:, 1] == 1].view(-1)

            task_1_true += [label.tolist() for label in label_1]
            task_2_true += list(label_2.float())
            task_1_score += [label.tolist() for label in label_hat_1]
            task_2_score += list(label_hat_2)

    weighted_precision_1 = precision_score(task_1_true, task_1_score, average='weighted')
    auc_2 = roc_auc_score(task_2_true, task_2_score)
    return weighted_precision_1, auc_2


if __name__ == '__main__':
    embedding_lookup = EmbeddingLookup(4)
    train_dataset, val_dataset, test_dataset, embeddings, embedding_map = \
        data_prepare(batch_size=16, embedding_lookup=embedding_lookup)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    feature_num = len(train_dataset.features[0])
    model = SparseMMOE(feature_num=feature_num, pretrained_embedding=embeddings, embedding_map=embedding_map,
                       num_experts=6, experts_out=16, experts_hidden=32, towers_hidden=8, tasks=[5, 1])
    n_epochs = 1000
    lr = 1e-5
    loss_fn_1 = nn.CrossEntropyLoss(reduction='mean')
    loss_fn_2 = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    losses = []

    print('start train')
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = [list(), list()]
        for features, labels, tasks in train_dataloader:
            labels_hat = model(features)

            label_1 = F.one_hot(labels[tasks[:, 0] == 1, 0] - 1, num_classes=5)
            label_2 = labels[tasks[:, 1] == 1, 1].view(-1, 1)
            label_hat_1 = labels_hat[0][tasks[:, 0] == 1]
            label_hat_2 = labels_hat[1][tasks[:, 1] == 1]

            loss_1 = 0
            if label_1.shape[0] != 0:
                loss_1 = loss_fn_1(label_1.float(), label_hat_1)
            loss_2 = 0
            if label_2.shape[0] != 0:
                loss_2 = loss_fn_2(label_2.float(), label_hat_2)
            # loss = loss_1 + loss_2
            loss = loss_1
            if torch.isnan(loss):
                print('loss is nan')
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if type(epoch_loss[0]) != list:
                print('epoch loss 0 is not list')
            epoch_loss[0].append(loss_1.item())
            if type(epoch_loss[1]) != list:
                print('epoch loss 1 is not list')
            epoch_loss[1].append(loss_2.item())

        losses.append(np.mean(epoch_loss))
        weighted_precision_1, auc_2 = test(model, val_dataloader)
        a, b = test(model, train_dataloader)
        print(f'task 1 train loss: {np.mean(epoch_loss[0]):.5f}, task 2 train loss: {np.mean(epoch_loss[1]):.5f}, '
              f'train loss: {np.mean(epoch_loss[0]) + np.mean(epoch_loss[1]):.5f}, val task 1 weighted precision: {weighted_precision_1:.3f}, '
              f'val task 2 auc: {auc_2:.3f}')

    weighted_precision_1, auc_2 = test(model, test_dataloader)
    print(f'val task 1 weighted precision: {weighted_precision_1:.3f}, val task 2 auc: {auc_2:.3f}')

    # embedding_lookup.update_embeddings(embeddings, model.embedding.weight.data)

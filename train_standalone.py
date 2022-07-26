import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import explained_variance_score
from torch.utils.data import DataLoader

from model.embedding_lookup import EmbeddingLookup
from model.mmoe import SparseMMOE
from util.data_prepare import data_prepare


def test(model: nn.Module, dataloader: DataLoader):
    t1_pred, t2_pred, t1_target, t2_target = [], [], [], []
    model.eval()
    with torch.no_grad():
        for features, labels, tasks in dataloader:
            labels_hat = model(features)

            label_1 = labels[:, 0].view(-1, 1).float() / 5
            label_2 = labels[:, 1].view(-1, 1).float()
            label_hat_1 = labels_hat[0].mul(tasks[:, 0].view(-1, 1))
            label_hat_2 = labels_hat[1].mul(tasks[:, 1].view(-1, 1))

            for index in range(labels.shape[0]):
                if tasks[index, 0] == 1:
                    t1_pred.append(label_hat_1[index])
                    t1_target.append(label_1[index])
                if tasks[index, 1] == 1:
                    t2_pred.append(label_hat_2[index])
                    t2_target.append(label_2[index])

    auc_1 = explained_variance_score(t1_target, t1_pred)
    auc_2 = explained_variance_score(t2_target, t2_pred)
    return auc_1, auc_2


if __name__ == '__main__':
    embedding_lookup = EmbeddingLookup(4)
    train_dataset, val_dataset, test_dataset, embeddings, embedding_map = \
        data_prepare(batch_size=16, embedding_lookup=embedding_lookup)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    feature_num = len(train_dataset.features[0])
    model = SparseMMOE(feature_num=feature_num, pretrained_embedding=embeddings, embedding_map=embedding_map,
                       num_experts=6, experts_out=16, experts_hidden=32, towers_hidden=8, tasks=2)
    n_epochs = 80
    lr = 1e-4
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    losses = []

    print('start train')
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        for features, labels, tasks in train_dataloader:
            labels_hat = model(features)

            label_1 = labels[:, 0].view(-1, 1).float() / 5
            label_2 = labels[:, 1].view(-1, 1).float()
            label_hat_1 = labels_hat[0].mul(tasks[:, 0].view(-1, 1))
            label_hat_2 = labels_hat[1].mul(tasks[:, 1].view(-1, 1))

            loss_1 = loss_fn(label_1, label_hat_1)
            loss_2 = loss_fn(label_2, label_hat_2)
            loss = loss_1 + loss_2
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())

        losses.append(np.mean(epoch_loss))
        auc_1, auc_2 = test(model, val_dataloader)
        print(f'train loss: {np.mean(epoch_loss):.5f}, val task 1 auc: {auc_1:.3f}, val task 2 auc: {auc_2:.3f}')

    auc_1, auc_2 = test(model, test_dataloader)
    print(f'val task 1 auc: {auc_1:.3f}, val task 2 auc: {auc_2:.3f}')

    # embedding_lookup.update_embeddings(embeddings, model.embedding.weight.data)

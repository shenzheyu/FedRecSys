import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score

from util.utils import get_criterion
from util.utils import get_evaluation


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = torch.device(args.device)
        self.criterion = get_criterion(self.args.criterion_name, self.device)
        self.evaluation_func = get_evaluation(self.args.evaluation_name)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (60, 20, 20)
        idxs_train = idxs[:int(0.6*len(idxs))]
        idxs_val = idxs[int(0.6*len(idxs)): int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)

        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(self.trainloader):
                # skip the one-size batch to avoid error in batch_norm
                if categorical_fields.size(0) <= 1:
                    continue

                categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(
                    self.device), labels.to(self.device)

                y = model(categorical_fields, numerical_fields)
                loss_list = [self.criterion[i](y[i], labels[:, i].float()) for i in range(labels.size(1))]
                loss = 0
                for item in loss_list:
                    loss += item
                loss /= len(loss_list)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) != 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        labels_dict, predicts_dict, loss_dict = {}, {}, {}
        for i in range(self.args.task_num):
            labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()

        with torch.no_grad():
            for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(self.validloader):
                categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(
                    self.device), labels.to(self.device)

                # Inference
                y = model(categorical_fields, numerical_fields)
                for i in range(self.args.task_num):
                    if y[i].size(0) <= 1:
                        labels_dict[i].append(labels[:, i].tolist())
                        predicts_dict[i].append(y[i].tolist())
                        loss_dict[i].append(
                            self.criterion[i](y[i].view(-1), labels[:, i].float()).tolist())
                    else:
                        labels_dict[i].extend(labels[:, i].tolist())
                        predicts_dict[i].extend(y[i].tolist())
                        loss_dict[i].append(
                            self.criterion[i](y[i].view(-1), labels[:, i].float()).tolist())

        evaluation_results, loss_results = list(), list()
        for i in range(self.args.task_num):
            evaluation_results.append(self.evaluation_func[i](labels_dict[i], predicts_dict[i]))
            loss_results.append(np.array(loss_dict[i]).mean())
        return evaluation_results, loss_results


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()

    device = torch.device(args.device)
    criterion = get_criterion(args.criterion_name, device)
    evaluation_func = get_evaluation(args.evaluation_name)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(args.task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()

    with torch.no_grad():
        for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(testloader):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                device), labels.to(device)

            # Inference
            y = model(categorical_fields, numerical_fields)
            for i in range(args.task_num):
                if y[i].size(0) <= 1:
                    labels_dict[i].append(labels[:, i].tolist())
                    predicts_dict[i].append(y[i].tolist())
                    loss_dict[i].append(
                        criterion[i](y[i].view(-1), labels[:, i].float()).tolist())
                else:
                    labels_dict[i].extend(labels[:, i].tolist())
                    predicts_dict[i].extend(y[i].tolist())
                    loss_dict[i].append(
                        criterion[i](y[i].view(-1), labels[:, i].float()).tolist())

    evaluation_results, loss_results = list(), list()
    for i in range(args.task_num):
        evaluation_results.append(evaluation_func[i](labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return evaluation_results, loss_results
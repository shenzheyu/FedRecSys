import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, args, dataset, idxs, field_dims, global_model, logger):
        self.args = args
        self.logger = logger
        self.field_dims = field_dims
        self.embedding_map = self.get_embedding_map(dataset, idxs)
        self.trainloader, self.validloader, self.testloader, self.idxs_train, self.idxs_test = self.train_val_test(
            dataset, list(idxs))
        self.update_times = self.get_update_times(dataset)
        self.device = torch.device(args.device)
        self.criterion = get_criterion(self.args.criterion_name, self.device)
        self.evaluation_func = get_evaluation(self.args.evaluation_name)
        self.model = copy.deepcopy(global_model)
        self.model = self.model.to(self.device)
        self.model.embedding.update_offsets(None)
        # change the embedding tensor to personalized shape according to self.embedding_map
        self.fetch_embedding(global_model, True)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        random.shuffle(idxs)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)): int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        # map train, validation set by embedding map
        self.data2local(dataset, idxs_train)
        self.data2local(dataset, idxs_val)

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)

        return trainloader, validloader, testloader, idxs_train, idxs_test

    def get_embedding_map(self, dataset, idxs):
        embedding_map = {}
        global_offset = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        embedding_idx = 0
        for idx in idxs:
            categorical_fields, _, _ = dataset[idx]
            for i in range(len(self.field_dims)):
                if categorical_fields[i] + global_offset[i] not in embedding_map.keys():
                    embedding_map[categorical_fields[i] + global_offset[i]] = embedding_idx
                    embedding_idx += 1
        return embedding_map  # (global_embedding_index, local_embedding_idx)

    def get_update_times(self, dataset):
        update_times = {}  # {local_word -> update_times}
        for idx in self.idxs_train:
            categorical_fields, _, _ = dataset[idx]
            for i in range(len(self.field_dims)):
                if categorical_fields[i] in update_times.keys():
                    update_times[categorical_fields[i]] += 1
                else:
                    update_times[categorical_fields[i]] = 1
        return update_times

    def data2local(self, dataset, idxs):
        global_offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        for idx in idxs:
            map_function = lambda x: self.embedding_map[x]
            dataset.update_categorical_data(idx, np.vectorize(map_function)(dataset[idx][0] + global_offsets))

    def fetch_embedding(self, global_model, new_embedding=False, word_condition=lambda x: True):
        global_embedding = global_model.state_dict()[self.args.embedding_name]
        if new_embedding:
            embedding = torch.zeros((len(self.embedding_map), global_embedding.size(1)), dtype=torch.float32)
        else:
            embedding = copy.deepcopy(self.model.state_dict()[self.args.embedding_name])
        for global_idx, idx in self.embedding_map.items():
            if word_condition(global_idx):
                embedding[idx] = global_embedding[global_idx]
        self.model.embedding.update_embedding(copy.deepcopy(embedding))

    def fetch_weights(self, global_model):
        global_weight = global_model.state_dict()
        for name, param in global_weight.items():
            if name == self.args.embedding_name:
                continue
            self.model.state_dict()[name].copy_(copy.deepcopy(param))

    def update_weights(self):
        # Set mode to train model
        self.model.train()
        self.model = self.model.to(self.device)
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(self.trainloader):
                # skip the one-size batch to avoid error in batch_norm
                if categorical_fields.size(0) <= 1:
                    continue

                categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(
                    self.device), labels.to(self.device)

                y = self.model(categorical_fields, numerical_fields)
                loss_list = [self.criterion[i](y[i].view(-1), labels[:, i].float()) for i in range(labels.size(1))]
                loss = 0
                for item in loss_list:
                    loss += item
                loss /= len(loss_list)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) != 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self):
        """ Returns the inference accuracy and loss.
        """

        self.model.eval()
        self.model = self.model.to(self.device)
        labels_dict, predicts_dict, loss_dict = {}, {}, {}
        for i in range(self.args.task_num):
            labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()

        with torch.no_grad():
            for batch_idx, (categorical_fields, numerical_fields, labels) in enumerate(self.validloader):
                categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(
                    self.device), labels.to(self.device)

                # Inference
                y = self.model(categorical_fields, numerical_fields)
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
